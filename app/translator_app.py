# translator_app.py
# Streamlit app: English ➜ Spanish Translator (Transformer, TensorFlow, Keras)
# Requirements: streamlit, tensorflow, keras, requests

import os
import string
import re
import hashlib
import requests
import numpy as np
import streamlit as st
import tensorflow as tf
from io import BytesIO
from keras.models import load_model
from keras.saving import register_keras_serializable

# Local transformer definition (your file in repo)
from transformer import Transformer

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="English → Spanish Translator", layout="centered")
st.title("English → Spanish Translator 🌍")
st.caption("Enter English text and get a Spanish translation. Uses a custom Transformer trained in TensorFlow/Keras.")

# GitHub Release asset URLs (the EXACT files that worked for you)
SOURCE_VEC_URL = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/source_vectorizer.keras"
TARGET_VEC_URL = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/target_vectorizer.keras"
WEIGHTS_URL    = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/translation_transformer.weights.h5"

# (Optional) If you want integrity checks, add SHA256 here (empty string = skip)
SHA256_EXPECTED = {
    "source_vectorizer.keras": "9260d7d760f115793408b0694afb36daa6646169cd840ee41352f9327d62b906",
    "target_vectorizer.keras": "47b0dc1848f2ca6963f5def3bfa705b0a39d4ee08aac6d0b4b755e61cd010d97",
    "translation_transformer.weights.h5": "9f0c1eea7407c3274c371850c3e72df87b3b51194f99d82e409779bcc2a25382",
}

# Training-time hyperparams (must match what produced the weights)
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

def download_file(url: str, dest: str):
    """Download url to dest if not present. Verify optional SHA256 if provided."""
    if not os.path.exists(dest):
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
    # integrity check (optional)
    expected = SHA256_EXPECTED.get(os.path.basename(dest), "")
    if expected:
        got = _sha256_file(dest)
        if got.lower() != expected.lower():
            raise ValueError(f"SHA256 mismatch for {dest}\nExpected: {expected}\nGot:      {got}")

def detect_special_tokens(vocab):
    """Return (start_token, end_token) by looking for known variants in vocab."""
    candidates = [("[start]", "[end]"), ("<start>", "<end>"), ("[CLS]", "[SEP]")]
    for s, e in candidates:
        if s in vocab and e in vocab:
            return s, e
    # As a fallback (not ideal), return None to indicate missing tokens
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

    # Trigger model build using real vectorizer outputs
    example_sentence = "hello"
    src_tokens = src_vec([example_sentence])
    # temp seed just to shape the decoder
    tmp_seed = "[start] hello [end]"
    tgt_tokens = tgt_vec([tmp_seed])[:, :-1]
    _ = model((src_tokens, tgt_tokens))  # build weights
    model.load_weights("translation_transformer.weights.h5")

    # 4) Prepare vocab maps
    vocab = tgt_vec.get_vocabulary()
    id_to_tok = dict(enumerate(vocab))
    tok_to_id = {t: i for i, t in enumerate(vocab)}

    # 5) Detect special tokens
    start_tok, end_tok = detect_special_tokens(vocab)

    return src_vec, tgt_vec, model, id_to_tok, tok_to_id, start_tok, end_tok

# -----------------------------
# Greedy decode
# -----------------------------
def translate(text: str, src_vec, tgt_vec, model, id_to_tok, tok_to_id, start_tok, end_tok, max_len=SEQ_LENGTH):
    if not text.strip():
        return ""

    # If special tokens missing, we can still try (but quality may degrade).
    if start_tok is None or end_tok is None:
        # Use a best-effort start token if available, else empty
        start_tok = start_tok or ""
        end_tok   = end_tok or ""

    # Vectorize source
    src = src_vec([text])

    # Begin with the start token (if present)
    decoded = start_tok.strip() if start_tok else ""
    if decoded:
        tokens = [decoded]
    else:
        tokens = []

    last_token = None
    for i in range(max_len):
        # Vectorize current target seed
        seed = " ".join(tokens) if tokens else ""
        tgt = tgt_vec([seed])[:, :-1]  # teacher-forcing prep

        # Predict next token distribution at position i
        logits = model((src, tgt)).numpy()  # shape: (1, tgt_len, vocab)
        step = min(i, logits.shape[1] - 1)
        next_id = int(np.argmax(logits[0, step, :]))
        next_tok = id_to_tok.get(next_id, "")

        # Stop conditions
        if end_tok and next_tok == end_tok:
            break
        if last_token == next_tok and next_tok in {"", start_tok}:  # prevent infinite repeats
            break

        tokens.append(next_tok)
        last_token = next_tok

    # Clean up: remove start token if present and any padding junk
    out = " ".join(t for t in tokens if t and t != start_tok)
    # A little tidy: collapse double spaces
    out = re.sub(r"\s+", " ", out).strip()
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
        result = translate(text, src_vec, tgt_vec, model, id_to_tok, tok_to_id, start_tok, end_tok)
    st.success(f"Spanish: {result}")

# -----------------------------
# Debug panel (safe to ignore)
# -----------------------------
with st.expander("🔧 Debug: vectorizers, tokens & quick sanity"):
    try:
        src_vec, tgt_vec, model, id_to_tok, tok_to_id, start_tok, end_tok = load_resources()
        vocab = tgt_vec.get_vocabulary()
        st.write("Target vocab size:", len(vocab))
        st.write("First 30 tokens:", vocab[:30])
        st.write("Detected start token:", start_tok)
        st.write("Detected end token:", end_tok)

        # Show how the target vectorizer tokenizes a seed
        seed = (start_tok or "") + (" " if start_tok else "") + "hello" + (" " + end_tok if end_tok else "")
        vec = tgt_vec([seed]).numpy()
        st.write(f"Vectorization of '{seed}' (first 12 ids):", list(vec[0][:12]))

        # Top-5 next-token sanity from seed-only
        seed_only = (start_tok or "")
        ex_tgt = tgt_vec([seed_only])[:, :-1]
        ex_src = src_vec(["hello"])
        logits = model((ex_src, ex_tgt))[0, -1, :].numpy()
        top5 = logits.argsort()[-5:][::-1]
        st.write("Top-5 next tokens from seed:", [(id_to_tok[i], float(logits[i])) for i in top5])
    except Exception as e:
        st.write("Debug error:", e)
