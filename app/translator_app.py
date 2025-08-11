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

# Local transformer definition
from transformer import Transformer

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="English → Spanish Translator", layout="centered")
st.title("English → Spanish Translator 🌍")
st.caption("Enter English text and get a Spanish translation. Uses a custom Transformer trained in TensorFlow/Keras.")

# -----------------------------
# Release assets (your GitHub URLs)
# -----------------------------
SOURCE_VEC_URL = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/source_vectorizer.keras"
TARGET_VEC_URL = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/target_vectorizer.keras"
WEIGHTS_URL    = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/translation_transformer.weights.h5"

# SHA256 checks (optional but recommended)
SHA256_EXPECTED = {
    "source_vectorizer.keras": "9260d7d760f115793408b0694afb36daa6646169cd840ee41352f9327d62b906",
    "target_vectorizer.keras": "47b0dc1848f2ca6963f5def3bfa705b0a39d4ee08aac6d0b4b755e61cd010d97",
    "translation_transformer.weights.h5": "9f0c1eea7407c3274c371850c3e72df87b3b51194f99d82e409779bcc2a25382",
}

# -----------------------------
# Model hyperparams (must match training)
# -----------------------------
VOCAB_SIZE   = 15000
SEQ_LENGTH   = 20
N_LAYERS     = 4
D_EMB        = 128
N_HEADS      = 8
D_FF         = 512
DROPOUT_RATE = 0.1

# -----------------------------
# Keras custom objects (for saved vectorizers)
# -----------------------------
@register_keras_serializable()
def custom_standardization(input_string):
    # Matches your training-time cleaning (punctuation stripped for the vectorizer)
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

def download_file(url: str, dest: str):
    """Download url to dest if not present. Verify optional SHA256 if provided."""
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

def detect_special_tokens(vocab):
    """Return (start_token, end_token) if present in vocabulary."""
    candidates = [("[start]", "[end]"), ("<start>", "<end>"), ("[CLS]", "[SEP]")]
    for s, e in candidates:
        if s in vocab and e in vocab:
            return s, e
    return None, None

def build_transformer():
    return Transformer(
        n_layers=N_LAYERS,
        d_emb=D_EMB,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout_rate=DROPOUT_RATE,
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
    )

def postprocess_text(src_text: str, es_text: str) -> str:
    """Light cleanup: add Spanish question marks if the source is a question, fix casing, spaces."""
    out = re.sub(r"\s+", " ", es_text).strip()

    # Basic capitalization (capitalize first letter if sentence-like)
    if out and out[0].isalpha():
        out = out[0].upper() + out[1:]

    # If English input ends with '?', ensure Spanish uses ¿ ... ?
    if src_text.strip().endswith("?"):
        out = out.rstrip(".! ")
        if not out.endswith("?"):
            out = f"{out}?"
        # Add opening inverted question mark if missing
        if not out.startswith("¿"):
            out = "¿" + out[0].lower() + out[1:]  # Spanish often uses lowercase after ¿

    return out

# -----------------------------
# Cached resource loader
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_resources():
    # 1) Ensure artifacts are present (download from GitHub Releases)
    download_file(SOURCE_VEC_URL, "source_vectorizer.keras")
    download_file(TARGET_VEC_URL, "target_vectorizer.keras")
    download_file(WEIGHTS_URL, "translation_transformer.weights.h5")

    # 2) Load vectorizers with custom standardization
    src_vec = load_model("source_vectorizer.keras", custom_objects={"custom_standardization": custom_standardization})
    tgt_vec = load_model("target_vectorizer.keras", custom_objects={"custom_standardization": custom_standardization})

    # 3) Build and load model
    model = build_transformer()

    # Trigger model build using vectorizer outputs
    seed_src = src_vec(["hello"])
    seed_tgt = tgt_vec(["[start] hello [end]"])[:, :-1]
    _ = model((seed_src, seed_tgt))
    model.load_weights("translation_transformer.weights.h5")

    # 4) Vocab maps & special tokens
    vocab = tgt_vec.get_vocabulary()
    id_to_tok = dict(enumerate(vocab))
    tok_to_id = {t: i for i, t in enumerate(vocab)}
    start_tok, end_tok = detect_special_tokens(vocab)

    return src_vec, tgt_vec, model, id_to_tok, tok_to_id, start_tok, end_tok

# -----------------------------
# Greedy decode with “non-special” top-k fallback
# -----------------------------
def translate(text: str, src_vec, tgt_vec, model, id_to_tok, tok_to_id, start_tok, end_tok, max_len=SEQ_LENGTH):
    if not text.strip():
        return ""

    # Fallbacks if tokens weren't detected (should not happen with your files)
    vocab = list(id_to_tok.values())
    if start_tok is None:
        start_tok = "[start]" if "[start]" in vocab else ""
    if end_tok is None:
        end_tok = "[end]" if "[end]" in vocab else None

    # Vectorize source
    src = src_vec([text])

    # Seed with start token if available
    tokens = [start_tok] if start_tok else []
    last_tok = None

    for step in range(max_len):
        seed = " ".join(tokens) if tokens else ""
        tgt = tgt_vec([seed])[:, :-1]

        # If tgt length is zero (no start token), make a minimal one to get valid shape
        if tgt.shape[1] == 0:
            tmp_seed = start_tok if start_tok else ""
            tgt = tgt_vec([tmp_seed])[:, :-1]

        logits = model((src, tgt)).numpy()  # (1, tgt_len, vocab)
        step_logits = logits[0, -1, :]      # last time-step

        # Choose best non-special candidate (skip "", [UNK], start, immediate end on first step, repeat)
        cand_ids = step_logits.argsort()[-10:][::-1]
        next_tok = None
        for cid in cand_ids:
            tok = id_to_tok.get(int(cid), "")
            if tok in {"", "[UNK]"}:
                continue
            if start_tok and tok == start_tok:
                continue
            if end_tok and tok == end_tok and step == 0:
                continue
            if tok == last_tok:
                continue
            next_tok = tok
            break

        if not next_tok:
            break
        if end_tok and next_tok == end_tok:
            break

        tokens.append(next_tok)
        last_tok = next_tok

    # Strip specials & tidy
    out_tokens = [t for t in tokens if t and t not in {start_tok, end_tok, "[UNK]"}]
    return " ".join(out_tokens).strip()

# -----------------------------
# UI
# -----------------------------
with st.form("translate_form"):
    text = st.text_input("Your English sentence:", placeholder="e.g., Hello, how are you?")
    submitted = st.form_submit_button("Translate")

if submitted and text:
    try:
        with st.spinner("Loading model & translating..."):
            src_vec, tgt_vec, model, id_to_tok, tok_to_id, start_tok, end_tok = load_resources()
            raw = translate(text, src_vec, tgt_vec, model, id_to_tok, tok_to_id, start_tok, end_tok)
            result = postprocess_text(text, raw)
        st.success(f"Spanish: {result}")
    except Exception as e:
        st.error(f"Translation failed: {e}")
