# translator_app.py
# Streamlit app: English → Spanish Translator (Transformer, TensorFlow, Keras)
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

# Local transformer definition
from transformer import Transformer

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="English → Spanish Translator", layout="centered")
st.title("English → Spanish Translator 🌍")
st.caption("Enter English text and get a Spanish translation (custom Transformer in TensorFlow/Keras).")

# Release asset URLs
SOURCE_VEC_URL = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/source_vectorizer.keras"
TARGET_VEC_URL = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/target_vectorizer.keras"
WEIGHTS_URL    = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/translation_transformer.weights.h5"

# SHA256 integrity (optional but recommended)
SHA256_EXPECTED = {
    "source_vectorizer.keras": "9260d7d760f115793408b0694afb36daa6646169cd840ee41352f9327d62b906",
    "target_vectorizer.keras": "47b0dc1848f2ca6963f5def3bfa705b0a39d4ee08aac6d0b4b755e61cd010d97",
    "translation_transformer.weights.h5": "9f0c1eea7407c3274c371850c3e72df87b3b51194f99d82e409779bcc2a25382",
}

# Must match training
VOCAB_SIZE   = 15000
SEQ_LENGTH   = 20
N_LAYERS     = 4
D_EMB        = 128
N_HEADS      = 8
D_FF         = 512
DROPOUT_RATE = 0.1

# -----------------------------
# Keras custom objects
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
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
    expected = SHA256_EXPECTED.get(os.path.basename(dest), "")
    if expected:
        got = _sha256_file(dest)
        if got.lower() != expected.lower():
            raise ValueError(f"SHA256 mismatch for {dest}\nExpected: {expected}\nGot:      {got}")

def _detect_special_tokens(vocab):
    for s, e in (("[start]", "[end]"), ("<start>", "<end>"), ("[CLS]", "[SEP]")):
        if s in vocab and e in vocab:
            return s, e
    return None, None

def _build_transformer():
    return Transformer(
        n_layers=N_LAYERS,
        d_emb=D_EMB,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout_rate=DROPOUT_RATE,
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
    )

def _postprocess_spanish(text: str, src: str) -> str:
    """Capitalize first letter; add ¿? or ¡! if English had ? or !; ensure final period if needed; tidy spaces."""
    t = re.sub(r"\s+", " ", text.strip())
    if not t:
        return t
    # Capitalize first letter (preserve leading punctuation like ¿¡)
    if t[0] in ("¿", "¡"):
        t = t[0] + t[1:2].upper() + t[2:]
    else:
        t = t[0].upper() + t[1:]

    src = src.strip()
    end = src[-1] if src else ""
    if end == "?":
        if not t.startswith("¿"):
            t = "¿" + t
        if not t.endswith("?"):
            t = t + "?"
    elif end == "!":
        if not t.startswith("¡"):
            t = "¡" + t
        if not t.endswith("!"):
            t = t + "!"
    else:
        if not t.endswith("."):
            t = t + "."
    return t

# -----------------------------
# Cached loader
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_resources():
    # Download artifacts
    _download(SOURCE_VEC_URL, "source_vectorizer.keras")
    _download(TARGET_VEC_URL, "target_vectorizer.keras")
    _download(WEIGHTS_URL, "translation_transformer.weights.h5")

    # Load vectorizers
    src_vec = load_model("source_vectorizer.keras", custom_objects={"custom_standardization": custom_standardization})
    tgt_vec = load_model("target_vectorizer.keras", custom_objects={"custom_standardization": custom_standardization})

    # Build + load model weights (shape with real vectorizer outputs)
    model = _build_transformer()
    src_tokens = src_vec(["hello"])
    tgt_tokens = tgt_vec(["[start] hello [end]"])[:, :-1]
    _ = model((src_tokens, tgt_tokens))
    model.load_weights("translation_transformer.weights.h5")

    # Vocab & tokens
    vocab = tgt_vec.get_vocabulary()
    id_to_tok = dict(enumerate(vocab))
    tok_to_id = {t: i for i, t in enumerate(vocab)}
    start_tok, end_tok = _detect_special_tokens(vocab)

    return src_vec, tgt_vec, model, id_to_tok, tok_to_id, start_tok, end_tok

# -----------------------------
# Greedy decode (token-by-token)
# -----------------------------
def translate(text: str, src_vec, tgt_vec, model, id_to_tok, tok_to_id, start_tok, end_tok, max_len=SEQ_LENGTH):
    if not text.strip():
        return ""

    src = src_vec([text])

    # Start seed
    tokens = [start_tok] if start_tok else []
    last_tok = None

    for step in range(max_len):
        seed = " ".join(tokens) if tokens else ""
        tgt = tgt_vec([seed])[:, :-1]  # teacher-forcing prep
        logits = model((src, tgt)).numpy()  # (1, tgt_len, vocab)

        # Use the last time-step from logits
        next_id = int(np.argmax(logits[0, -1, :]))
        next_tok = id_to_tok.get(next_id, "")

        # Stop conditions
        if end_tok and next_tok == end_tok:
            break
        if next_tok == "" or next_tok == start_tok:
            break
        if next_tok == last_tok:
            # tiny escape hatch to avoid loops
            break

        tokens.append(next_tok)
        last_tok = next_tok

    # Clean: remove special tokens + join
    out_tokens = [t for t in tokens if t and t not in {start_tok, end_tok, "[UNK]"}]
    out = " ".join(out_tokens).strip()
    out = re.sub(r"\s+", " ", out)
    return out

# -----------------------------
# UI
# -----------------------------
with st.form("translate_form"):
    text = st.text_input("Your English sentence:", placeholder="e.g., Hello, how are you?")
    submitted = st.form_submit_button("Translate")

if submitted and text:
    with st.spinner("Loading model & translating..."):
        src_vec, tgt_vec, model, id_to_tok, tok_to_id, start_tok, end_tok = load_resources()
        raw = translate(text, src_vec, tgt_vec, model, id_to_tok, tok_to_id, start_tok, end_tok)
        pretty = _postprocess_spanish(raw, text)
    st.success(f"Spanish: {pretty}")
