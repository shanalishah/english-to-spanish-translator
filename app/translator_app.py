# app.py
# pip install -U streamlit gdown tensorflow keras numpy packaging

import os
import re
import string
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown
from keras.models import load_model
from keras.saving import register_keras_serializable
from transformer import Transformer
from packaging import version

# -----------------------------
# Basic env / version sanity
# -----------------------------
MIN_TF = "2.15.0"  # adjust if you truly require newer
assert version.parse(tf.__version__) >= version.parse(MIN_TF), (
    f"Requires TensorFlow >= {MIN_TF}, found {tf.__version__}"
)

# -----------------------------
# Google Drive file IDs
# -----------------------------
WEIGHTS_FILE_ID = "1r5_qQhb975vaO6XXV_SyI8ytzE3obV9u"
SOURCE_VEC_ID   = "10NfA0tF9zs2CHYSNAHmQ_nRU9LDwjv50"
TARGET_VEC_ID   = "1gXNAutl1HtPhMpNtmQ78JscLSkR2_Qid"

# -----------------------------
# Keras-serializable standardization (for loading vectorizers)
# -----------------------------
@register_keras_serializable()
def custom_standardization(input_string):
    strip_chars = string.punctuation + "¿"
    strip_chars = strip_chars.replace("[", "").replace("]", "")
    lower = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lower, f"[{re.escape(strip_chars)}]", "")

# -----------------------------
# Cache resources across reruns
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_resources():
    files = {
        "translation_transformer.weights.h5": WEIGHTS_FILE_ID,
        "source_vectorizer.keras": SOURCE_VEC_ID,
        "target_vectorizer.keras": TARGET_VEC_ID,
    }

    for fname, fid in files.items():
        if not os.path.exists(fname):
            url = f"https://drive.google.com/uc?id={fid}"
            gdown.download(url, fname, quiet=False)

    # Load vectorizers (saved with Keras)
    source_vectorization = load_model("source_vectorizer.keras")
    target_vectorization = load_model("target_vectorizer.keras")

    # Rebuild the Transformer and load weights
    vocab_size = 15000
    model = Transformer(
        n_layers=4,
        d_emb=128,
        n_heads=8,
        d_ff=512,
        dropout_rate=0.1,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
    )

    # Build model with real shapes before loading weights
    src_example = source_vectorization(["hello"])
    tgt_example = target_vectorization(["[start] hello [end]"])[:, :-1]
    _ = model((src_example, tgt_example))
    model.load_weights("translation_transformer.weights.h5")

    # Prepare decoding vocab
    spa_vocab = target_vectorization.get_vocabulary()
    spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))

    return source_vectorization, target_vectorization, model, spa_index_lookup

# -----------------------------
# Greedy decoding
# -----------------------------
def translate(input_sentence, source_vectorization, target_vectorization, model, spa_index_lookup, max_len=20):
    text = (input_sentence or "").strip()
    if not text:
        return ""

    tokenized_src = source_vectorization([text])
    decoded = "[start]"
    for _ in range(max_len):
        tokenized_tgt = target_vectorization([decoded])[:, :-1]
        preds = model((tokenized_src, tokenized_tgt))
        # SAFER: always pick the last timestep
        next_token_id = int(np.argmax(preds[0, -1, :]))
        next_token = spa_index_lookup.get(next_token_id, "")
        decoded += " " + next_token
        if next_token == "[end]":
            break

    return decoded.replace("[start] ", "").replace(" [end]", "").strip()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="English → Spanish Translator", layout="centered")
st.title("English → Spanish Translator 🌍")
st.caption("Transformer (TensorFlow + Keras). Enter English text and get a Spanish translation.")

user_input = st.text_input("Your English sentence:", placeholder="e.g., The weather is nice today.")

if user_input:
    with st.spinner("Translating…"):
        src_vec, tgt_vec, model, lookup = load_resources()
        out = translate(user_input, src_vec, tgt_vec, model, lookup)
    st.success(f"**Spanish:** {out}")
