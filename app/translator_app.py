# app/translator_app.py
# pip install streamlit tensorflow keras requests

import os
import io
import time
import string
import re
import hashlib
import requests
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.saving import register_keras_serializable
from keras.models import load_model

# Local module (your Transformer model definition)
from transformer import Transformer


# ============= Config =============

# GitHub Release asset URLs (public)
WEIGHTS_URL = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/translation_transformer.weights.h5"
SRC_VEC_URL = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/source_vectorizer.keras"
TGT_VEC_URL = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/target_vectorizer.keras"

# Optional integrity checks (comment out if you don’t want hash verification)
SHA256_EXPECTED = {
    "source_vectorizer.keras": "9260d7d760f115793408b0694afb36daa6646169cd840ee41352f9327d62b906",
    "target_vectorizer.keras": "47b0dc1848f2ca6963f5def3bfa705b0a39d4ee08aac6d0b4b755e61cd010d97",
    # weights hash you provided looked misspelled; skip or update if you have the correct one:
    # "translation_transformer.weights.h5": "9f0c1eea7407c3274c371850c3e72df87b3b51194f99d82e409779bcc2a25382",
}

ASSETS = [
    ("translation_transformer.weights.h5", WEIGHTS_URL),
    ("source_vectorizer.keras", SRC_VEC_URL),
    ("target_vectorizer.keras", TGT_VEC_URL),
]

# Model hyperparams (must match training)
VOCAB_SIZE = 15000
MAX_LEN = 20
N_LAYERS = 4
D_EMB = 128
N_HEADS = 8
D_FF = 512
DROPOUT = 0.1


# ============= Utils =============

def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _download_file(url: str, dest: str) -> None:
    """Download to disk if not present. Streams to avoid memory spikes."""
    if os.path.exists(dest):
        return
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    # optional verify
    exp = SHA256_EXPECTED.get(os.path.basename(dest))
    if exp:
        actual = _sha256_of_file(dest)
        if actual.lower() != exp.lower():
            raise RuntimeError(
                f"SHA256 mismatch for {dest}\nExpected: {exp}\nActual:   {actual}\n"
                "Re-upload the asset or update the expected hash."
            )

@register_keras_serializable()
def custom_standardization(input_string):
    # must match training-time preprocessing
    strip_chars = string.punctuation + "¿"
    strip_chars = strip_chars.replace("[", "").replace("]", "")
    x = tf.strings.lower(input_string)
    return tf.strings.regex_replace(x, f"[{re.escape(strip_chars)}]", "")


# ============= Caching loaders =============

@st.cache_resource(show_spinner=False)
def load_resources():
    # download artifacts
    for fname, url in ASSETS:
        _download_file(url, fname)

    # load vectorizers with custom object
    src_vec = load_model("source_vectorizer.keras",
                         custom_objects={"custom_standardization": custom_standardization})
    tgt_vec = load_model("target_vectorizer.keras",
                         custom_objects={"custom_standardization": custom_standardization})

    # rebuild model skeleton and load weights
    model = Transformer(
        n_layers=N_LAYERS, d_emb=D_EMB, n_heads=N_HEADS, d_ff=D_FF,
        dropout_rate=DROPOUT,
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE
    )

    # Build once by calling with real-shaped tensors
    # Use dummy pass with vectorizers so Keras creates variables
    _src = src_vec(["hello"])
    _tgt = tgt_vec(["[start] hello [end]"])[:, :-1]
    _ = model((_src, _tgt))
    model.load_weights("translation_transformer.weights.h5")

    # vocab lookup for decoding
    spa_vocab = tgt_vec.get_vocabulary()
    id_to_token = {i: tok for i, tok in enumerate(spa_vocab)}
    return src_vec, tgt_vec, model, id_to_token


# ============= Decoding =============

def translate(
    text: str,
    src_vec,
    tgt_vec,
    model,
    id_to_token,
    max_len: int = MAX_LEN,
) -> str:
    # Vectorize source
    tokenized_src = src_vec([text])

    decoded = "[start]"
    for _ in range(max_len):
        tokenized_tgt = tgt_vec([decoded])[:, :-1]  # teacher-forcing style
        preds = model((tokenized_src, tokenized_tgt))  # (1, tgt_len, vocab)
        next_id = int(np.argmax(preds[0, -1, :]))     # ALWAYS take last timestep
        next_tok = id_to_token.get(next_id, "")

        # skip junk tokens if they ever appear
        if not next_tok or next_tok in {"[start]", "[PAD]", "[pad]", "[UNK]", "[unk]"}:
            continue

        decoded += " " + next_tok
        if next_tok == "[end]":
            break

    # tidy output
    decoded = decoded.replace("[start] ", "").replace(" [end]", "").strip()
    return decoded


# ============= UI =============

st.set_page_config(page_title="English→Spanish Translator", layout="centered")
st.title("English → Spanish Translator (Transformer)")
st.caption("Enter an English sentence; the model generates a Spanish translation.")

user_text = st.text_input("Your English sentence", placeholder="e.g., The weather is nice today.", label_visibility="visible")

col1, col2 = st.columns([1, 3])
with col1:
    go = st.button("Translate", type="primary")
with col2:
    st.write("")

if go and user_text.strip():
    with st.spinner("Loading model & translating…"):
        t0 = time.time()
        src_vec, tgt_vec, model, id_to_token = load_resources()
        out = translate(user_text.strip(), src_vec, tgt_vec, model, id_to_token, max_len=MAX_LEN)
        ms = int((time.time() - t0) * 1000)
    if out:
        st.success(f"Spanish: {out}")
        st.caption(f"Done in ~{ms} ms")
    else:
        st.warning("The model produced an empty output. Try a different sentence or shorter text.")
elif go:
    st.info("Please enter a sentence to translate.")
