# ui.py – Streamlit front-end that chains to enhancer.py

import streamlit as st
import os
import sys
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

def run_pipeline(file_paths: list[Path],
                 save_debug: bool,
                 debug_dir: Path,
                 strong: bool,
                 run_ocr: bool,
                 jsonl_name: str,
                 parser_cmd: str,
                 keep_temp: bool) -> tuple[int, Path | None, Path, Path]:
    """
    Returns (exit_code, jsonl_path or None, debug_dir, log_path)
    """
    # Build a temp working folder and list file
    tmp_root = Path(tempfile.mkdtemp(prefix="ui_paths_"))
    lst = tmp_root / "paths.txt"
    lst.write_text("\n".join(str(p) for p in file_paths), encoding="utf-8")

    # Resolve enhancer script and output targets
    enhancer_py = str((Path(__file__).parent / "enhancer.py").resolve())
    downloads = get_downloads_folder()
    jsonl_path = (downloads / jsonl_name).resolve() if run_ocr and jsonl_name else None

    if save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Compose command for enhancer.py
    parts = [
        f'"{sys.executable}"',
        f'"{enhancer_py}"',
        f'--list "{lst}"',
    ]
    parts.append(f'--debug-dir "{debug_dir}"' if save_debug else "--no-debug")
    if strong:
        parts.append("--strong")
    if run_ocr:
        parts.append("--chain-gcv")
        if jsonl_path:
            parts.append(f'--gcv-out "{jsonl_path}"')
        if parser_cmd.strip():
            parts.append(f'--pipe-parser "{parser_cmd.strip()}"')
    if keep_temp:
        parts.append("--keep-temp")

    cmd = " ".join(parts)

    # Capture logs to a file so we can show them in the UI
    log_path = tmp_root / "run.log"
    cmd_with_redirect = f'{cmd} > "{log_path}" 2>&1'
    rc = os.system(cmd_with_redirect)

    return rc, jsonl_path, debug_dir, log_path

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
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Run options")
run_ocr = st.sidebar.checkbox("Run OCR after enhance", value=True)
strong = st.sidebar.checkbox("Use strong enhance", value=False)
save_debug = st.sidebar.checkbox("Save enhanced copies for debugging", value=True)
debug_dir_input = st.sidebar.text_input("Debug folder", value="images_enhanced_debug")
parser_cmd = st.sidebar.text_input("Pipe GCV output to command", value="", placeholder="python parser.py")
jsonl_name = st.sidebar.text_input("JSONL filename in Downloads", value="receipts.jsonl")
keep_temp = st.sidebar.checkbox("Keep temp enhanced files", value=False)

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
        "⚙️  Enhance → OCR" if run_ocr else "⚙️  Enhance only",
        disabled=not st.session_state.images,
        help="Runs enhancer.py then optionally gcv_engine.py"
    )

if run_clicked:
    paths = [p.resolve() for p in st.session_state.images.values()]
    debug_dir = Path(debug_dir_input).resolve()

    with st.spinner("Running enhancer"):
        rc, jsonl_path, dbg_dir, log_path = run_pipeline(
            file_paths=paths,
            save_debug=save_debug,
            debug_dir=debug_dir,
            strong=strong,
            run_ocr=run_ocr,
            jsonl_name=jsonl_name.strip() or "receipts.jsonl",
            parser_cmd=parser_cmd,
            keep_temp=keep_temp,
        )

    # Logs
    try:
        log_text = Path(log_path).read_text(encoding="utf-8")
    except Exception:
        log_text = "(no logs found)"

    st.write("### Run logs")
    st.code(log_text, language="bash")

    # Results
    if rc == 0:
        if save_debug:
            st.success(f"Enhanced copies saved to {dbg_dir}")
        if run_ocr and jsonl_path:
            st.success(f"OCR JSONL saved to {jsonl_path}")
            st.toast("✅ OCR output written", icon="🗂️")
        else:
            st.toast("✅ Enhancement done", icon="✨")
    else:
        st.error(f"Pipeline exited with code {rc}")

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
if save_debug and Path(debug_dir_input).exists():
    enhanced = sorted(
        [p for p in Path(debug_dir_input).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    if enhanced:
        st.write("### Enhanced previews")
        cols_per_row = 4
        for idx, p in enumerate(enhanced[:24]):  # show up to 24
            if idx % cols_per_row == 0:
                cols = st.columns(cols_per_row)
            with cols[idx % cols_per_row]:
                try:
                    st.image(p.open("rb").read(), caption=p.name, width=180)
                except Exception:
                    pass
