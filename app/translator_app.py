# translator_app.py
# Streamlit app: English ➜ Spanish Translator (Transformer, TensorFlow, Keras)

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

# Local transformer definition (file in repo)
from transformer import Transformer

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="English -> Spanish Translator", layout="centered")
st.title("English -> Spanish Translator 🌍")
st.caption("Enter English text and get a Spanish translation.")

# GitHub Release asset URLs (the EXACT files in your release)
SOURCE_VEC_URL = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/source_vectorizer.keras"
TARGET_VEC_URL = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/target_vectorizer.keras"
WEIGHTS_URL    = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/translation_transformer.weights.h5"

# Optional integrity checks (sha256)
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
    """
    Keep Spanish/English question/exclamation marks so the model can emit '¿' and '¡'
    and preserve '?' and '!'. Strip other punctuation (except [ and ] used by special tokens).
    """
    keep_chars = "?!¡¿"
    strip_chars = "".join(ch for ch in string.punctuation if ch not in keep_chars)
    strip_chars = strip_chars.replace("[", "").replace("]", "")
    lowered = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowered, f"[{re.escape(strip_chars)}]", "")

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
            for chunk in r.iter_content(chunk_size=256 * 1024):
                if chunk:
                    f.write(chunk)
    expected = SHA256_EXPECTED.get(os.path.basename(dest), "")
    if expected:
        got = _sha256_file(dest)
        if got.lower() != expected.lower():
            raise ValueError(f"SHA256 mismatch for {dest}\nExpected: {expected}\nGot:      {got}")

def detect_special_tokens(vocab):
    """Return (start_token, end_token) if present in vocab."""
    for s, e in (("[start]", "[end]"), ("<start>", "<end>"), ("[CLS]", "[SEP]")):
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

# -----------------------------
# Cached resource loader
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_resources():
    # Download artifacts
    download_file(SOURCE_VEC_URL, "source_vectorizer.keras")
    download_file(TARGET_VEC_URL, "target_vectorizer.keras")
    download_file(WEIGHTS_URL, "translation_transformer.weights.h5")

    # Load vectorizers with the custom standardization
    src_vec = load_model("source_vectorizer.keras",
                         custom_objects={"custom_standardization": custom_standardization})
    tgt_vec = load_model("target_vectorizer.keras",
                         custom_objects={"custom_standardization": custom_standardization})

    # Build and load model
    model = build_transformer()
    src_tokens = src_vec(["hello"])
    tgt_tokens = tgt_vec(["[start] hello [end]"])[:, :-1]
    _ = model((src_tokens, tgt_tokens))  # build
    model.load_weights("translation_transformer.weights.h5")

    vocab = tgt_vec.get_vocabulary()
    id_to_tok = dict(enumerate(vocab))
    tok_to_id = {t: i for i, t in enumerate(vocab)}
    start_tok, end_tok = detect_special_tokens(vocab)

    return src_vec, tgt_vec, model, id_to_tok, tok_to_id, start_tok, end_tok

# -----------------------------
# Greedy decode
# -----------------------------
def translate(text: str, src_vec, tgt_vec, model, id_to_tok, tok_to_id, start_tok, end_tok, max_len=SEQ_LENGTH):
    if not text.strip():
        return ""

    # Vectorize source (punctuation like ? ! ¿ ¡ is preserved by the vectorizer now)
    src = src_vec([text])

    # Seed with start token if available
    tokens = [start_tok] if start_tok else []
    last_token = None

    for i in range(max_len):
        seed = " ".join(tokens) if tokens else ""
        tgt = tgt_vec([seed])[:, :-1]

        logits = model((src, tgt)).numpy()  # (1, tgt_len, vocab)
        step = min(i, logits.shape[1] - 1)
        next_id = int(np.argmax(logits[0, step, :]))
        next_tok = id_to_tok.get(next_id, "")

        if end_tok and next_tok == end_tok:
            break
        if next_tok in {"", start_tok} or next_tok == last_token:
            # Avoid empty/start/loops; try to continue
            last_token = next_tok
            continue

        tokens.append(next_tok)
        last_token = next_tok

    # Join and clean (remove start token if present)
    out = " ".join(t for t in tokens if t and t != start_tok)
    out = re.sub(r"\s+", " ", out).strip()

    # Light post-processing: ensure Spanish inverted punctuation symmetry for short phrases
    if out and out.endswith("?") and not out.startswith("¿"):
        out = "¿" + out
    if out and out.endswith("!") and not out.startswith("¡"):
        out = "¡" + out

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
