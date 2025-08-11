import os
from pathlib import Path
import streamlit as st
import tensorflow as tf

from app.utils import ensure_file
from app.transformer import build_transformer_model
from app.inference import translate

st.set_page_config(page_title="English → Spanish Translator", layout="centered")

st.title("English → Spanish Translator")
st.caption("Transformer (TensorFlow) with a Streamlit interface.")

# ---- Read secrets (fall back to env in local dev) ----
WEIGHTS_ID = st.secrets.get("WEIGHTS_FILE_ID") or os.getenv("WEIGHTS_FILE_ID")
SRC_VEC_ID = st.secrets.get("SOURCE_VEC_ID")   or os.getenv("SOURCE_VEC_ID")
TGT_VEC_ID = st.secrets.get("TARGET_VEC_ID")   or os.getenv("TARGET_VEC_ID")

if not (WEIGHTS_ID and SRC_VEC_ID and TGT_VEC_ID):
    st.error("Missing Google Drive IDs. Please set WEIGHTS_FILE_ID, SOURCE_VEC_ID, TARGET_VEC_ID in secrets.")
    st.stop()

# ---- Local cache paths ----
ARTIFACTS = Path(".cache/models")
WEIGHTS_PATH = ARTIFACTS / "translation_transformer.weights.h5"
SRC_VEC_PATH = ARTIFACTS / "source_vectorizer.keras"
TGT_VEC_PATH = ARTIFACTS / "target_vectorizer.keras"

@st.cache_resource(show_spinner=True)
def load_everything():
    """Download artifacts (if needed) and load vectorizers + model."""
    # download files
    ensure_file(WEIGHTS_ID, WEIGHTS_PATH)
    ensure_file(SRC_VEC_ID, SRC_VEC_PATH)
    ensure_file(TGT_VEC_ID, TGT_VEC_PATH)

    # load vectorizers (saved as Keras objects)
    src_vectorizer = tf.keras.models.load_model(SRC_VEC_PATH)
    tgt_vectorizer = tf.keras.models.load_model(TGT_VEC_PATH)

    # build your model with the SAME architecture used for training
    model = build_transformer_model(src_vectorizer, tgt_vectorizer)

    demo_mode = False
    try:
        model.load_weights(WEIGHTS_PATH)
    except Exception as e:
        # shape mismatch or architecture difference → fall back to demo mode
        demo_mode = True
        st.warning(
            "Model weights could not be loaded (likely architecture mismatch). "
            "Running in demo mode. Paste your training architecture into app/transformer.py."
        )
    return model, src_vectorizer, tgt_vectorizer, demo_mode

model, src_vec, tgt_vec, demo_mode = load_everything()

# ---- UI ----
text = st.text_area("Enter English text", placeholder="e.g., The weather is nice today.", height=140)

col1, col2 = st.columns([1, 1])
with col1:
    max_len = st.slider("Max output tokens", min_value=20, max_value=120, value=60, step=5)
with col2:
    decoding = st.selectbox("Decoding", ["greedy"], index=0)  # you can add "beam" later

btn = st.button("Translate")

if btn:
    if not text.strip():
        st.info("Please enter some English text.")
        st.stop()
    with st.spinner("Translating..."):
        try:
            out = translate(
                text.strip(),
                model=model,
                src_vectorizer=src_vec,
                tgt_vectorizer=tgt_vec,
                strategy=decoding,
                max_target_len=max_len,
                demo_mode=demo_mode,
            )
            st.markdown("**Spanish:**")
            st.write(out)
        except Exception as e:
            st.error(f"Translation failed: {e}")
