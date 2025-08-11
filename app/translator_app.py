# app/translator_app.py
# Streamlit + Keras Transformer (English → Spanish)
# pip install streamlit tensorflow keras numpy requests

import os
import re
import string
import hashlib
import math
import requests
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.saving import register_keras_serializable
from transformer import Transformer  # keep transformer.py next to this file

# -----------------------------
# Release asset URLs (your links)
# -----------------------------
ARTIFACTS = {
    "source_vectorizer.keras": "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/source_vectorizer.keras",
    "target_vectorizer.keras": "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/target_vectorizer.keras",
    "translation_transformer.weights.h5": "https://github.com/shanalishah/english-to-spanish-translator/releases/download/v1.0/translation_transformer.weights.h5",
}

# -----------------------------
# SHA-256 checks (from you)
# -----------------------------
SHA256 = {
    "source_vectorizer.keras": "9260d7d760f115793408b0694afb36daa6646169cd840ee41352f9327d62b906",
    "target_vectorizer.keras": "47b0dc1848f2ca6963f5def3bfa705b0a39d4ee08aac6d0b4b755e61cd010d97",
    "translation_transformer.weights.h5": "9f0c1eea7407c3274c371850c3e72df87b3b51194f99d82e409779bcc2a25382",
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
# Utils: download + verify
# -----------------------------
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _download(url: str, dest: str):
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
        except Exception as e:
            if attempt == 2:
                raise

    # Verify integrity
    expected = SHA256.get(dest)
    if expected:
        actual = _sha256_file(dest)
        if actual.lower() != expected.lower():
            raise RuntimeError(
                f"SHA256 mismatch for {dest}. Expected {expected[:12]}..., got {actual[:12]}..."
            )

# -----------------------------
# Cache: load resources
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_resources():
    # Download & verify
    for fname, url in ARTIFACTS.items():
        _download(url, fname)

    # Load vectorizers
    src_vec = load_model(
        "source_vectorizer.keras",
        custom_objects={"custom_standardization": custom_standardization},
    )
    tgt_vec = load_model(
        "target_vectorizer.keras",
        custom_objects={"custom_standardization": custom_standardization},
    )

    # Build model exactly as trained
    vocab_size = 15000
    model = Transformer(
        n_layers=4, d_emb=128, n_heads=8, d_ff=512,
        dropout_rate=0.1,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size
    )

    # Trigger build with real shapes
    src_tok = src_vec(["hello"])
    tgt_tok = tgt_vec(["[start] hello [end]"])[:, :-1]
    _ = model((src_tok, tgt_tok))

    # Load weights (verified above)
    model.load_weights("translation_transformer.weights.h5")

    # Vocab maps
    vocab = tgt_vec.get_vocabulary()
    id_to_tok = {i: t for i, t in enumerate(vocab)}
    tok_to_id = {t: i for i, t in enumerate(vocab)}

    return src_vec, tgt_vec, model, id_to_tok, tok_to_id

# -----------------------------
# Beam search decoder
# -----------------------------
def beam_search_translate(
    sentence: str,
    src_vec,
    tgt_vec,
    model,
    id_to_tok,
    tok_to_id,
    beam_width: int = 3,
    max_len: int = 20,
):
    src = src_vec([sentence])

    start = "[start]"
    end = "[end]"
    end_id = tok_to_id.get(end, None)
    bad = {"[start]", "[PAD]", "[pad]", "[UNK]", "[unk]"}

    # beams: list of (tokens_list, logprob)
    beams = [([start], 0.0)]

    for _ in range(max_len):
        new_beams = []
        for tokens, logp in beams:
            # If already ended, keep as-is
            if end_id is not None and tokens and tokens[-1] == end:
                new_beams.append((tokens, logp))
                continue

            tgt_in = tgt_vec([" ".join(tokens)])[:, :-1]  # teacher-forced prefix
            logits = model((src, tgt_in))[0, -1, :].numpy()

            # get top-k
            topk = int(min(beam_width * 3, logits.shape[-1]))
            idxs = np.argpartition(-logits, topk)[:topk]  # indices of topk candidates
            # sort topk by score desc
            idxs = idxs[np.argsort(-logits[idxs])]

            candidates = 0
            for idx in idxs:
                tok = id_to_tok.get(int(idx), "")
                if (not tok) or (tok in bad):
                    continue
                if end_id is not None and len(tokens) == 1 and idx == end_id:
                    # don't end immediately
                    continue

                new_tokens = tokens + [tok]
                new_logp = logp + float(logits[idx] - math.log(np.sum(np.exp(logits))))  # add log softmax
                new_beams.append((new_tokens, new_logp))
                candidates += 1
                if candidates >= beam_width:
                    break

        # prune
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]

        # early stop if all ended
        if all((end_id is None) or (b[0][-1] == end) for b in beams):
            break

    # best sequence
    best_tokens = beams[0][0] if beams else [start, end]
    out = " ".join(t for t in best_tokens if t not in {start, end}).strip()
    return out

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="English → Spanish Translator", layout="centered")
st.title("English → Spanish Translator 🇬🇧➡️🇪🇸")
st.caption("Transformer (TensorFlow/Keras). Enter English text to get a Spanish translation.")

user_text = st.text_input("Your English sentence", placeholder="e.g., Hello, how are you?")

if st.button("Translate", type="primary") and user_text.strip():
    with st.spinner("Loading model & translating…"):
        src_vec, tgt_vec, model, id_to_tok, tok_to_id = load_resources()

        # Quick sanity probe to catch mismatched artifacts early
        sanity = beam_search_translate("hello", src_vec, tgt_vec, model, id_to_tok, tok_to_id)
        if not sanity or sanity.lower() in {"", "por", "[end]"}:
            st.warning("Sanity check looks off (e.g., 'hello' → not close to 'hola'). "
                       "This usually means the downloaded artifacts don’t match the trained run. "
                       "Re-upload vectorizers + weights from the same training and update the release.")
        result = beam_search_translate(user_text.strip(), src_vec, tgt_vec, model, id_to_tok, tok_to_id)

    st.success(f"Spanish: {result}")
