# app/translator_app.py
# Streamlit + Keras Transformer (English -> Spanish)
# Requires: streamlit, tensorflow, keras, numpy, requests

import os
import re
import string
import hashlib
import requests
import numpy as np
import streamlit as st
import tensorflow as tf
from io import BytesIO
from keras.models import load_model
from keras.saving import register_keras_serializable

# If transformer.py sits in repo root or app/, make sure import path is correct.
# In this setup, keep transformer.py next to this file in app/
from transformer import Transformer

# -----------------------------
# Config: GitHub Releases URLs
# -----------------------------
ARTIFACTS = {
    "source_vectorizer.keras": "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/source_vectorizer.keras",
    "target_vectorizer.keras": "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/target_vectorizer.keras",
    "translation_transformer.weights.h5": "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/translation_transformer.weights.h5",
}

# (Optional) If you want integrity checks, put SHA256 here (leave empty to skip)
SHA256 = {
    # "source_vectorizer.keras": "9260d7d760f115793408b0694afb36daa6646169cd840ee41352f9327d62b906",
    # "target_vectorizer.keras": "47b0dc1848f2ca6963f5def3bfa705b0a39d4ee08aac6d0b4b755e61cd010d97",
    # "translation_transformer.weights.h5": "<sha256-of-your-exact-weights-file>",
}

# -----------------------------
# Keras custom objects
# -----------------------------
@register_keras_serializable()
def custom_standardization(input_string):
    strip_chars = string.punctuation + "¿"
    strip_chars = strip_chars.replace("[", "").replace("]", "")
    s = tf.strings.lower(input_string)
    return tf.strings.regex_replace(s, f"[{re.escape(strip_chars)}]", "")

# -----------------------------
# Utilities
# -----------------------------
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _download(url: str, dest: str):
    # Idempotent download with small retries
    if os.path.exists(dest):
        return
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
            break
        except Exception:
            if attempt == 2:
                raise

    # Optional integrity check
    if dest in SHA256 and SHA256[dest]:
        actual = _sha256_file(dest)
        if actual.lower() != SHA256[dest].lower():
            st.warning(f"SHA256 mismatch for {dest}. Expected {SHA256[dest][:8]}…, got {actual[:8]}… (continuing).")

# -----------------------------
# Cached loader
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_resources():
    # Ensure artifacts exist locally
    for fname, url in ARTIFACTS.items():
        _download(url, fname)

    # Load vectorizers with custom standardization
    src_vec = load_model(
        "source_vectorizer.keras",
        custom_objects={"custom_standardization": custom_standardization},
    )
    tgt_vec = load_model(
        "target_vectorizer.keras",
        custom_objects={"custom_standardization": custom_standardization},
    )

    # Build model with the same hyperparams used in training
    vocab_size = 15000
    model = Transformer(
        n_layers=4, d_emb=128, n_heads=8, d_ff=512,
        dropout_rate=0.1,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size
    )

    # Trigger model build using real shapes from vectorizers
    example = "hello"
    src_tok = src_vec([example])
    tgt_tok = tgt_vec(["[start] hello [end]"])[:, :-1]
    _ = model((src_tok, tgt_tok))  # build graph

    # Load weights
    model.load_weights("translation_transformer.weights.h5")

    # Vocabulary lookup for decoder
    vocab = tgt_vec.get_vocabulary()
    id_to_tok = {i: t for i, t in enumerate(vocab)}
    tok_to_id = {t: i for i, t in enumerate(vocab)}

    return src_vec, tgt_vec, model, id_to_tok, tok_to_id

# -----------------------------
# Greedy decoding
# -----------------------------
def translate(sentence: str, src_vec, tgt_vec, model, id_to_tok, tok_to_id, max_len: int = 20) -> str:
    # Tokenize source
    src = src_vec([sentence])

    decoded = ["[start]"]
    end_id = tok_to_id.get("[end]")
    bad_tokens = {"[start]", "[PAD]", "[pad]", "[UNK]", "[unk]"}

    for _ in range(max_len):
        tgt_in = tgt_vec([" ".join(decoded)])[:, :-1]   # teacher-forced prefix
        logits = model((src, tgt_in))                   # (1, cur_len, vocab)
        last = logits[0, -1, :].numpy()                 # only final step scores

        # Don’t allow immediate stop
        if end_id is not None and len(decoded) == 1:
            last[end_id] = -1e9

        next_id = int(np.argmax(last))
        next_tok = id_to_tok.get(next_id, "")

        # Skip junk
        if (not next_tok) or (next_tok in bad_tokens):
            continue

        decoded.append(next_tok)
        if end_id is not None and next_id == end_id:
            break

    out = " ".join(decoded)
    out = out.replace("[start]", "").replace("[end]", "").strip()
    return out

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="English → Spanish Translator", layout="centered")
st.title("English → Spanish Translator")
st.caption("Transformer (TensorFlow/Keras). Enter English text and get a Spanish translation.")

user_text = st.text_input("Your English sentence", placeholder="e.g., The wetlands protect our communities from floods.")

if st.button("Translate", type="primary") or (user_text and "auto" not in st.session_state):
    if not user_text.strip():
        st.warning("Please enter a sentence.")
    else:
        with st.spinner("Loading model & translating…"):
            src_vec, tgt_vec, model, id_to_tok, tok_to_id = load_resources()
            result = translate(user_text.strip(), src_vec, tgt_vec, model, id_to_tok, tok_to_id)
        st.success(f"Spanish: {result}")

# Tiny footer
st.write("")
st.caption("Tip: If you update artifacts in a new GitHub release, click **Rerun → Clear cache** in Streamlit to refresh.")
