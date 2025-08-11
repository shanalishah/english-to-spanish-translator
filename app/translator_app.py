# translator_app.py
# Streamlit app: English ➜ Spanish Translator (Transformer, TensorFlow, Keras)
# Requirements: streamlit, tensorflow, keras, requests

import os
import re
import string
import hashlib
import requests
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.saving import register_keras_serializable

# Local transformer definition (same file you already have)
from transformer import Transformer

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="English → Spanish Translator", layout="centered")
st.title("English → Spanish Translator 🌍")
st.caption("Enter English text and get a Spanish translation. Uses a custom Transformer trained in TensorFlow/Keras.")

# -----------------------------
# GitHub Release assets (your links)
# -----------------------------
SOURCE_VEC_URL = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/source_vectorizer.keras"
TARGET_VEC_URL = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/target_vectorizer.keras"
WEIGHTS_URL    = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/translation_transformer.weights.h5"

# Optional integrity checks (the hashes you provided)
SHA256_EXPECTED = {
    "source_vectorizer.keras": "9260d7d760f115793408b0694afb36daa6646169cd840ee41352f9327d62b906",
    "target_vectorizer.keras": "47b0dc1848f2ca6963f5def3bfa705b0a39d4ee08aac6d0b4b755e61cd010d97",
    "translation_transformer.weights.h5": "9f0c1eea7407c3274c371850c3e72df87b3b51194f99d82e409779bcc2a25382",
}

# -----------------------------
# Training-time hyperparams (must match)
# -----------------------------
VOCAB_SIZE   = 15000
SEQ_LENGTH   = 20
N_LAYERS     = 4
D_EMB        = 128
N_HEADS      = 8
D_FF         = 512
DROPOUT_RATE = 0.1

# -----------------------------
# Keras custom objects (vectorizers depend on this)
# -----------------------------
@register_keras_serializable()
def custom_standardization(input_string):
    strip_chars = string.punctuation + "¿"
    strip_chars = strip_chars.replace("[", "").replace("]", "")
    return tf.strings.regex_replace(tf.strings.lower(input_string), f"[{re.escape(strip_chars)}]", "")

# -----------------------------
# Helpers
# -----------------------------
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _download(url: str, dest: str):
    if not os.path.exists(dest):
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(256 * 1024):
                if chunk:
                    f.write(chunk)
    expected = SHA256_EXPECTED.get(os.path.basename(dest), "")
    if expected:
        got = _sha256_file(dest)
        if got.lower() != expected.lower():
            raise ValueError(f"SHA256 mismatch for {dest}\nExpected: {expected}\nGot:      {got}")

def _postprocess(src_text: str, es_text: str) -> str:
    """Light cleanup: spacing, capitalization, Spanish question marks."""
    out = re.sub(r"\s+", " ", es_text).strip()
    if out:
        # Capitalize first letter
        if out[0].isalpha():
            out = out[0].upper() + out[1:]
    # Add ¿ ? if input was a question
    if src_text.strip().endswith("?"):
        out = out.rstrip(".! ")
        if not out.endswith("?"):
            out += "?"
        if not out.startswith("¿"):
            # Spanish style: open with inverted mark; typically lowercase after it
            out = "¿" + out[0].lower() + out[1:]
    return out

def _build_model():
    return Transformer(
        n_layers=N_LAYERS,
        d_emb=D_EMB,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout_rate=DROPOUT_RATE,
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
    )

# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_resources():
    # Download artifacts
    _download(SOURCE_VEC_URL, "source_vectorizer.keras")
    _download(TARGET_VEC_URL, "target_vectorizer.keras")
    _download(WEIGHTS_URL, "translation_transformer.weights.h5")

    # Load vectorizers (exactly like your working version)
    src_vec = load_model("source_vectorizer.keras")
    tgt_vec = load_model("target_vectorizer.keras")

    # Build + load model
    model = _build_model()

    # Trigger build with real shapes (same seed pattern you used)
    src_seed = src_vec(["hello"])
    tgt_seed = tgt_vec(["[start] hello [end]"])[:, :-1]
    _ = model((src_seed, tgt_seed))
    model.load_weights("translation_transformer.weights.h5")

    # Vocab & maps
    vocab = tgt_vec.get_vocabulary()
    id_to_tok = dict(enumerate(vocab))

    # We’ll just use the literal tokens used in training
    start_tok = "[start]"
    end_tok   = "[end]"

    return src_vec, tgt_vec, model, id_to_tok, start_tok, end_tok

# -----------------------------
# Greedy decode — exactly your working loop
# -----------------------------
def translate(text: str, src_vec, tgt_vec, model, id_to_tok, start_tok, end_tok, max_len=SEQ_LENGTH):
    if not text.strip():
        return ""

    tokenized_input_sentence = src_vec([text])
    decoded_sentence = start_tok  # seed EXACTLY like before

    for i in range(max_len):
        tokenized_target_sentence = tgt_vec([decoded_sentence])[:, :-1]
        predictions = model((tokenized_input_sentence, tokenized_target_sentence))
        # Select the i-th position (your original indexing)
        sampled_token_index = int(np.argmax(predictions[0, i, :]))
        sampled_token = id_to_tok.get(sampled_token_index, "")

        decoded_sentence += " " + sampled_token
        if sampled_token == end_tok:
            break

    # Clean: remove tokens and extra spaces
    out = decoded_sentence.replace(start_tok, "").replace(end_tok, "").strip()
    out = re.sub(r"\s+", " ", out)
    return out

# -----------------------------
# UI
# -----------------------------
with st.form("translate_form"):
    text = st.text_input("Your English sentence:", placeholder="e.g., Hello, how are you?")
    submitted = st.form_submit_button("Translate")

if submitted and text:
    try:
        with st.spinner("Loading model & translating..."):
            src_vec, tgt_vec, model, id_to_tok, start_tok, end_tok = load_resources()
            raw = translate(text, src_vec, tgt_vec, model, id_to_tok, start_tok, end_tok)
            result = _postprocess(text, raw)
        st.success(f"Spanish: {result}")
    except Exception as e:
        st.error(f"Translation failed: {e}")
