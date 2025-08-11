from pathlib import Path
import gdown
import streamlit as st

@st.cache_resource(show_spinner=False)
def ensure_file(file_id: str, dest: Path) -> Path:
    """Download a file from Google Drive (by file ID) to `dest` if missing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(dest), quiet=False)
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"Failed to download {dest.name} from Google Drive")
    return dest
