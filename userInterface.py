# ui.py – Streamlit front-end that chains to enhancer and GCV in-process

import streamlit as st
import os
import shutil
import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, Tk

# extra deps for in-process pipeline
import cv2

# lazy imports so the UI still loads if files are missing
def _lazy_imports():
    from enhancer import enhance_file          # returns np.ndarray
    from gcv_engine import run_files as gcv_run_files  # runs OCR on file paths
    return enhance_file, gcv_run_files

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
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Run options")
run_ocr   = st.sidebar.checkbox("Run OCR after enhance", value=True)
strong    = st.sidebar.checkbox("Use strong enhance", value=False)
save_dbg  = st.sidebar.checkbox("Save enhanced copies for debugging", value=True)
dbg_dir_s = st.sidebar.text_input("Debug folder", value="images_enhanced_debug")
jsonl_nm  = st.sidebar.text_input("JSONL filename in Downloads", value="receipts.jsonl")
keep_temp = st.sidebar.checkbox("Keep temp enhanced files", value=False)

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
    for path in st.session_state.images.values():
        if path.parent == st.session_state.temp_dir:
            path.unlink(missing_ok=True)
    st.session_state.images.clear()

# Sync uploaded files with session state
if uploaded_files is not None:
    current = {uf.name for uf in uploaded_files}
    for name in list(st.session_state.images):
        if name not in current:
            path = st.session_state.images.pop(name)
            if path.parent == st.session_state.temp_dir:
                path.unlink(missing_ok=True)
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
# Process button
# --------------------------------------------------
with col_btn:
    run_clicked = st.button(
        "⚙️ SCAN FILES" if run_ocr else "⚙️  Enhance only",
        disabled=not st.session_state.images,
        help="Enhances in memory, saves debug copies, then runs GCV"
    )

# --------------------------------------------------
# Pipeline
# --------------------------------------------------
if run_clicked:
    logs = []
    try:
        enhance_file, gcv_run_files = _lazy_imports()
    except Exception as e:
        st.error(f"Could not import enhancer or gcv_engine. {e}")
        st.stop()

    src_paths = [p.resolve() for p in st.session_state.images.values()]
    dbg_dir   = Path(dbg_dir_s).resolve()
    if save_dbg:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="ui_enhanced_"))
    tmp_enh  = tmp_root / "images_enhanced"
    tmp_enh.mkdir(parents=True, exist_ok=True)

    progress = st.progress(0.0, text="Enhancing… 0 / 0")
    enhanced_paths: list[Path] = []

    # enhance files
    total = len(src_paths)
    for i, p in enumerate(src_paths, 1):
        try:
            img = enhance_file(p, strong=strong)
            out_name = p.with_suffix(".png").name
            tmp_out = tmp_enh / out_name
            cv2.imwrite(str(tmp_out), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            enhanced_paths.append(tmp_out)

            if save_dbg:
                dbg_out = dbg_dir / out_name
                cv2.imwrite(str(dbg_out), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            logs.append(f"[enhance] {i}/{total} -> {out_name}")
        except Exception as ex:
            logs.append(f"[error] enhance failed for {p.name} {ex}")

        progress.progress(i / total, text=f"Enhancing… {i} / {total}")

    progress.empty()

    # run OCR
    jsonl_path = None
    if run_ocr and enhanced_paths:
        jsonl_path = get_downloads_folder() / (jsonl_nm.strip() or "receipts.jsonl")
        try:
            rc = gcv_run_files([str(p) for p in enhanced_paths], out_path=str(jsonl_path))
            if rc == 0:
                logs.append(f"[gcv] wrote {jsonl_path}")
            else:
                logs.append(f"[gcv] exited with code {rc}")
        except Exception as ex:
            logs.append(f"[error] gcv_engine failed {ex}")

    # cleanup
    if not keep_temp:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass

    # show logs and results
    st.write("### Run logs")
    st.code("\n".join(logs) if logs else "(no logs)", language="bash")

    if save_dbg:
        st.success(f"Enhanced copies saved to {dbg_dir}")
    if run_ocr and jsonl_path and jsonl_path.exists():
        st.success(f"OCR JSONL saved to {jsonl_path}")
        st.toast("✅ OCR output written", icon="🗂️")
    else:
        st.toast("✅ Enhancement done", icon="✨")

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

# --------------------------------------------------
# Optional preview of enhanced images
# --------------------------------------------------
if save_dbg and Path(dbg_dir_s).exists():
    enhanced = sorted([p for p in Path(dbg_dir_s).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if enhanced:
        st.write("### Enhanced previews")
        cols_per_row = 4
        for idx, p in enumerate(enhanced[:24]):
            if idx % cols_per_row == 0:
                cols = st.columns(cols_per_row)
            with cols[idx % cols_per_row]:
                try:
                    st.image(p.open("rb").read(), caption=p.name, width=180)
                except Exception:
                    pass
