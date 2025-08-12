# ui_mock.py – Streamlit front-end (UI-only preview; no processing)

import streamlit as st
import os
import shutil
import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, Tk

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="OCR Receipt Reader", layout="wide")
st.title("🧾 OCR Receipt Reader")

# --------------------------------------------------
# Constants & helpers
# --------------------------------------------------
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def get_downloads_folder() -> Path:
    return (Path(os.environ.get("USERPROFILE", "")) / "Downloads"
            if os.name == "nt" else Path.home() / "Downloads")

def pick_directory() -> str | None:
    try:
        root: Tk = tk.Tk()
        root.attributes("-topmost", True)
        root.withdraw()
        folder = filedialog.askdirectory()
        root.destroy()
        return folder or None
    except Exception:
        return None

# --------------------------------------------------
# Session-state init
# --------------------------------------------------
DEFAULTS = {
    "temp_dir": Path(tempfile.mkdtemp(prefix="receipt_uploads_")),
    "images": {},
    "folder_selected": None,
    "uploader_key": 0,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# --------------------------------------------------
# Upload controls
# --------------------------------------------------
st.subheader("📤 Upload receipts")

if st.button("📂  Select folder"):
    picked = pick_directory()
    if picked:
        st.session_state.images = {
            p.name: p for p in Path(picked).iterdir()
            if p.suffix.lower() in VALID_EXT
        }
        st.session_state.folder_selected = picked
        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
        st.session_state.temp_dir = Path(tempfile.mkdtemp(prefix="receipt_uploads_"))
        st.session_state.uploader_key += 1
        st.rerun()

col_up, col_btn = st.columns([4, 1])
with col_up:
    uploaded_files = st.file_uploader(
        "Drag & drop images here or click Browse",
        type=list(VALID_EXT),
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )

# Clear in-memory images if uploader is empty and no folder is selected
if uploaded_files is None and st.session_state.folder_selected is None:
    # delete temp files created from previous uploads
    for path in st.session_state.images.values():
        if path.parent == st.session_state.temp_dir:
            path.unlink(missing_ok=True)
    st.session_state.images.clear()

# Sync uploaded files with session state
if uploaded_files is not None:
    current = {uf.name for uf in uploaded_files}   # empty set if list == []
    # remove unchecked
    for name in list(st.session_state.images):
        if name not in current:
            path = st.session_state.images.pop(name)
            if path.parent == st.session_state.temp_dir:
                path.unlink(missing_ok=True)
    # add new
    for uf in uploaded_files:
        if uf.name in st.session_state.images:
            continue
        dest = st.session_state.temp_dir / uf.name
        base, ext = os.path.splitext(dest.name)
        ctr = 1
        while dest.exists():
            dest = dest.with_name(f"{base}_{ctr}{ext}")
            ctr += 1
        dest.write_bytes(uf.getbuffer())
        st.session_state.images[uf.name] = dest
    st.session_state.folder_selected = None

# --------------------------------------------------
# Process button (UI-only; no action)
# --------------------------------------------------
with col_btn:
    run_clicked = st.button(
        "⚙️  Process",
        disabled=not st.session_state.images,
        help="UI preview only. Processing is disabled in this mock."
    )

if run_clicked:
    st.info("This is a UI-only preview. Processing and downloads are disabled.")

# --------------------------------------------------
# Preview grid
# --------------------------------------------------
if st.session_state.images:
    st.write("### Selected images")
    cols_per_row = 4
    for idx, (name, p) in enumerate(st.session_state.images.items()):
        if idx % cols_per_row == 0:
            cols = st.columns(cols_per_row)
        with cols[idx % cols_per_row]:
            try:
                st.image(p.open("rb").read(), caption=name, width=180)
            except Exception:
                st.warning(f"Cannot preview {name}")
