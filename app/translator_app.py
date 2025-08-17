# translator_app.py
# Streamlit app: English → Spanish Translator (Transformer, TensorFlow, Keras)

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

# GitHub Release asset URLs
SOURCE_VEC_URL = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/source_vectorizer.keras"
TARGET_VEC_URL = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/target_vectorizer.keras"
WEIGHTS_URL    = "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/translation_transformer.weights.h5"

# Optional integrity checks
SHA256_EXPECTED = {
    "source_vectorizer.keras": "9260d7d760f115793408b0694afb36daa6646169cd840ee41352f9327d62b906",
    "target_vectorizer.keras": "47b0dc1848f2ca6963f5def3bfa705b0a39d4ee08aac6d0b4b755e61cd010d97",
    "translation_transformer.weights.h5": "9f0c1eea7407c3274c371850c3e72df87b3b51194f99d82e409779bcc2a25382",
}

# Training-time hyperparams (must match the weights)
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
    # This is what was used at training time
    strip_chars = string.punctuation + "¿"
    strip_chars = strip_chars.replace("[", "").replace("]", "")
    return tf.strings.regex_replace(
        tf.strings.lower(input_string),
        f"[{re.escape(strip_chars)}]",
        ""
    )

# -----------------------------
# Small helpers
# -----------------------------
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download_file(url: str, dest: str):
    """Download url to dest if not present. Verify SHA256 if provided."""
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
    """Return (start_token, end_token) by looking for known variants in vocab."""
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

# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_resources():
    # 1) Ensure artifacts are present
    download_file(SOURCE_VEC_URL, "source_vectorizer.keras")
    download_file(TARGET_VEC_URL, "target_vectorizer.keras")
    download_file(WEIGHTS_URL, "translation_transformer.weights.h5")

    # 2) Load vectorizers
    src_vec = load_model("source_vectorizer.keras", custom_objects={"custom_standardization": custom_standardization})
    tgt_vec = load_model("target_vectorizer.keras", custom_objects={"custom_standardization": custom_standardization})

    # 3) Build model & load weights
    model = build_transformer()

    # Trigger model build using real shapes
    src_tokens = src_vec(["hello"])
    tgt_tokens = tgt_vec(["[start] hello [end]"])[:, :-1]
    _ = model((src_tokens, tgt_tokens))
    model.load_weights("translation_transformer.weights.h5")

    # 4) Vocab maps & special tokens
    vocab = tgt_vec.get_vocabulary()
    id_to_tok = dict(enumerate(vocab))
    tok_to_id = {t: i for i, t in enumerate(vocab)}
    start_tok, end_tok = detect_special_tokens(vocab)

    return src_vec, tgt_vec, model, id_to_tok, tok_to_id, start_tok, end_tok

# -----------------------------
# Decoding + punctuation fix
# -----------------------------
def greedy_decode(src_text: str, src_vec, tgt_vec, model, id_to_tok, start_tok, end_tok, max_len=SEQ_LENGTH):
    if not src_text.strip():
        return ""
    # Vectorize source
    src = src_vec([src_text])

    # Start seed
    tokens = [start_tok] if start_tok else []

    last_token = None
    for i in range(max_len):
        seed = " ".join(tokens) if tokens else ""
        tgt = tgt_vec([seed])[:, :-1]

        logits = model((src, tgt)).numpy()  # (1, tgt_len, vocab)
        step = min(i, logits.shape[1] - 1)
        next_id = int(np.argmax(logits[0, step, :]))
        next_tok = id_to_tok.get(next_id, "")

        # Stop on end token
        if end_tok and next_tok == end_tok:
            break
        # Prevent repeats of empty/start
        if last_token == next_tok and next_tok in {"", start_tok}:
            break

        tokens.append(next_tok)
        last_token = next_tok

    # Join & clean
    out = " ".join(t for t in tokens if t and t != start_tok)
    out = re.sub(r"\s+", " ", out).strip()
    return out

def add_spanish_inverted_punctuation(src_en: str, es: str) -> str:
    """Heuristic post-processing to add ¿ / ¡ based on English punctuation."""
    text = es.strip()
    if not text:
        return text

    src = src_en.strip()
    # If source ends with ?, ensure Spanish ends with ? and starts with ¿
    if src.endswith("?"):
        text = text.rstrip(".! ")
        if not text.endswith("?"):
            text = text + "?"
        if not text.startswith("¿"):
            text = "¿" + text
        # Capitalize first letter after inverted mark
        if len(text) > 1:
            text = text[0] + text[1].upper() + text[2:]

    # If source ends with !, ensure Spanish ends with ! and starts with ¡
    if src.endswith("!"):
        text = text.rstrip(".? ")
        if not text.endswith("!"):
            text = text + "!"
        if not text.startswith("¡"):
            text = "¡" + text
        if len(text) > 1:
            text = text[0] + text[1].upper() + text[2:]

    return text

# -----------------------------
# UI
# -----------------------------
with st.form("translate_form"):
    text = st.text_input("Your English sentence:", placeholder='e.g., "Hello!", "How are you?"')
    submitted = st.form_submit_button("Translate")

if submitted and text:
    with st.spinner("Loading model & translating..."):
        src_vec, tgt_vec, model, id_to_tok, tok_to_id, start_tok, end_tok = load_resources()
        raw = greedy_decode(text, src_vec, tgt_vec, model, id_to_tok, start_tok, end_tok)
        result = add_spanish_inverted_punctuation(text, raw)
    st.success(f"Spanish: {result}")
