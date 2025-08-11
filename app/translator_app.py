import os
import hashlib
import requests
from pathlib import Path

import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model

from transformer import Transformer  # your architecture

# --- GitHub Release assets and SHA-256 hashes ---
ASSETS = {
    "source_vectorizer.keras": (
        "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/source_vectorizer.keras",
        "9260d7d760f115793408b0694afb36daa6646169cd840ee41352f9327d62b906",
    ),
    "target_vectorizer.keras": (
        "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/target_vectorizer.keras",
        "47b0dc1848f2ca6963f5def3bfa705b0a39d4ee08aac6d0b4b755e61cd010d97",
    ),
    "translation_transformer.weights.h5": (
        "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/translation_transformer.weights.h5",
        "9f0c1eea7407c3274c371850c3e72df87b3b51194f99d82e409779bcc2a25382",
    ),
}

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def download_with_verify(url, dest: Path, expected_hash: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_of(dest).lower() == expected_hash.lower():
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    if sha256_of(tmp).lower() != expected_hash.lower():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {dest.name}")
    tmp.replace(dest)

@st.cache_resource
def load_resources():
    for fname, (url, sha256) in ASSETS.items():
        download_with_verify(url, Path(fname), sha256)

    src_vec = load_model("source_vectorizer.keras")
    tgt_vec = load_model("target_vectorizer.keras")

    vocab_size = 15000
    model = Transformer(
        n_layers=4, d_emb=128, n_heads=8, d_ff=512,
        dropout_rate=0.1, src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size
    )

    example = "hello"
    src = src_vec([example])
    tgt = tgt_vec(["[start] hello [end]"])[:, :-1]
    model((src, tgt))
    model.load_weights("translation_transformer.weights.h5")

    spa_vocab = tgt_vec.get_vocabulary()
    lookup = {i: token for i, token in enumerate(spa_vocab)}

    return src_vec, tgt_vec, model, lookup

def translate(text, src_vec, tgt_vec, model, lookup, max_len=20):
    if not text:
        return ""
    tokenized = src_vec([text])
    decoded = "[start]"
    for _ in range(max_len):
        tokenized_tgt = tgt_vec([decoded])[:, :-1]
        preds = model((tokenized, tokenized_tgt))
        next_id = int(np.argmax(preds[0, -1, :]))
        token = lookup.get(next_id, "")
        decoded += " " + token
        if token == "[end]":
            break
    return decoded.replace("[start] ", "").replace(" [end]", "").strip()

# --- Streamlit Interface ---
st.set_page_config(page_title="English → Spanish Translator", layout="centered")
st.title("English → Spanish Translator")
st.caption("Powered by a Transformer model (TensorFlow/Keras)")

user_input = st.text_input("English sentence:", placeholder="Type here...")
if user_input:
    with st.spinner("Translating..."):
        src_vec, tgt_vec, model, lookup = load_resources()
        result = translate(user_input, src_vec, tgt_vec, model, lookup)
        st.success(f"**Spanish:** {result}")
