# ui.py – Streamlit front-end that chains to enhancer and GCV in-process
# Layout: upload (left) + extraction (right); results below
# Includes: instruction input, enhance controls, placeholder LLM step,
# results table, CSV download, demo-fill option, JSONL preview, and rebuild support

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
import inspect
import random
from datetime import date

import streamlit as st

# Optional folder picker for local runs
try:
    import tkinter as tk
    from tkinter import filedialog, Tk
except Exception:
    tk = None
    filedialog = None
    Tk = None

# extra deps for in-process pipeline
import cv2
import pandas as pd


# ------------------------------
# Lazy imports so the UI still loads if files are missing
# ------------------------------
def _lazy_imports():
    from enhancer import enhance_file          # returns np.ndarray
    from gcv_engine import run_files as gcv_run_files  # runs OCR on file paths
    return enhance_file, gcv_run_files


# ------------------------------
# Page config
# ------------------------------
st.set_page_config(page_title="OCR Receipt Reader", layout="wide")
st.title("🧾 OCR Receipt Reader")

# ------------------------------
# Style: orange section headers
# ------------------------------
st.markdown(
    """
    <style>
      .orange-subheader {
        font-weight: 700;
        font-size: 1.25rem;
        margin: 0 0 .5rem 0;
        color: #f59e0b; /* orange */
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------
# Constants and helpers
# ------------------------------
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_downloads_folder() -> Path:
    return (Path(os.environ.get("USERPROFILE", "")) / "Downloads"
            if os.name == "nt" else Path.home() / "Downloads")


def pick_directory() -> str | None:
    if Tk is None or filedialog is None:
        st.warning("Folder picker unavailable in this environment")
        return None
    try:
        root: Tk = Tk()
        root.attributes("-topmost", True)
        root.withdraw()
        folder = filedialog.askdirectory()
        root.destroy()
        return folder or None
    except Exception:
        return None


# --- Instruction helpers -----------------------------------------------------
def _to_snake(s: str) -> str:
    s = s.strip().replace("/", " ").replace("-", " ")
    s = " ".join(s.split())
    return s.lower().replace(" ", "_")


def parse_requested_fields(text: str) -> list[str]:
    """Parse user instruction into a list of column names.
    Accepts comma or newline separated tokens. Falls back to sensible defaults."""
    if not text:
        return ["file", "merchant_name", "total_amount"]
    raw = [tok for chunk in text.splitlines() for tok in chunk.split(",")]
    cleaned: list[str] = []
    seen: set[str] = set()
    for tok in raw:
        col = _to_snake(tok)
        if not col:
            continue
        if col in seen:
            continue
        seen.add(col)
        cleaned.append(col)
    if "file" not in seen:
        cleaned = ["file"] + cleaned
    return cleaned or ["file", "merchant_name", "total_amount"]


# --- Placeholder LLM step ----------------------------------------------------
def _demo_value_for(col: str, fname: str) -> str:
    base = col.lower()
    if base in {"file", "filename"}:
        return fname
    if any(k in base for k in ["merchant", "store", "seller", "vendor"]):
        return random.choice(["LaunderLux", "FreshFold", "AquaWash", "QuickClean"])
    if any(k in base for k in ["total", "amount", "price", "sum", "gross"]):
        return f"{random.randint(80, 1200)}.{random.randint(0,99):02d}"
    if "date" in base:
        return date.today().isoformat()
    if any(k in base for k in ["invoice", "receipt", "ref", "number", "no"]):
        return f"{random.randint(100000,999999)}-{random.randint(10,99)}"
    if any(k in base for k in ["address", "branch"]):
        return random.choice(["Main Branch", "Katipunan", "BGC", "QC"])
    return ""


def simulate_llm_processor(image_paths: list[Path], columns: list[str], demo_fill: bool) -> pd.DataFrame:
    """Temporary placeholder. Builds a DataFrame with requested columns.
    - 'file' from filename
    - demo values if demo_fill is True
    Replace later with llm_processor.process(ocr_text, instructions)."""
    rows = []
    for p in image_paths:
        row = {c: "" for c in columns}
        row["file"] = p.name
        if demo_fill:
            for c in columns:
                if c == "file":
                    continue
                row[c] = _demo_value_for(c, p.name)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ------------------------------
# Sidebar controls
# ------------------------------
st.sidebar.header("Run options")
run_ocr = st.sidebar.checkbox("Run OCR after enhance", value=True)

# Enhancement controls
mode = st.sidebar.selectbox(
    "Enhance mode",
    options=["auto", "gray", "binary"],
    index=0,
    help="auto picks mild vs binary; gray keeps grayscale; binary forces binarization",
)
strong = st.sidebar.checkbox("Use strong enhance", value=False, help="If your enhancer supports it")

save_dbg = st.sidebar.checkbox("Save enhanced copies for debugging", value=True)
dbg_dir_s = st.sidebar.text_input("Debug folder", value="images_enhanced_debug")
jsonl_nm = st.sidebar.text_input("JSONL filename in Downloads", value="receipts.jsonl")
keep_temp = st.sidebar.checkbox("Keep temp enhanced files", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Extraction instructions")
with st.sidebar.expander("Examples", expanded=False):
    st.markdown(
        "- merchant name, total amount\n"
        "- merchant_name, invoice_number, total_amount\n"
        "- store, date, price\n"
        "- Just the top merchant name and the final price\n"
    )

st.sidebar.markdown("---")
demo_fill = st.sidebar.checkbox("Demo-fill extracted values", value=False, help="Populates table with sample values for screenshots")


# ------------------------------
# Session-state init
# ------------------------------
DEFAULTS = {
    "temp_dir": Path(tempfile.mkdtemp(prefix="receipt_uploads_")),
    "images": {},
    "folder_selected": None,
    "uploader_key": 0,
    "instructions": "merchant name, total amount",
    "result_df": None,
    "result_csv_name": "extracted_receipts.csv",
    "last_enhanced_paths": [],  # as strings
    "ocr_jsonl_path": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================== MAIN LAYOUT ==================================
left_col, right_col = st.columns(2, gap="large")

# ------------------------------ LEFT: Upload ---------------------------------
with left_col:
    st.markdown('<div class="orange-subheader">📤 Upload receipts</div>', unsafe_allow_html=True)

    # Scan button sits here (replaces prior "Select folder" spot)
    run_clicked = st.button(
        "⚙️ SCAN FILES" if run_ocr else "⚙️  Enhance only",
        disabled=not st.session_state.images,
        help="Enhances in memory, saves debug copies, then runs GCV",
    )

    uploaded_files = st.file_uploader(
        "Drag and drop images here or click Browse",
        type=list(VALID_EXT),
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    # Clear in-memory images if uploader is empty and no folder is selected
    if uploaded_files is None and st.session_state.folder_selected is None:
        for path in st.session_state.images.values():
            if isinstance(path, Path) and path.parent == st.session_state.temp_dir:
                path.unlink(missing_ok=True)
        st.session_state.images.clear()

    # Sync uploaded files with session state
    if uploaded_files is not None:
        current = {uf.name for uf in uploaded_files}
        for name in list(st.session_state.images):
            if name not in current:
                path = st.session_state.images.pop(name)
                if isinstance(path, Path) and path.parent == st.session_state.temp_dir:
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

# ------------------------------ RIGHT: Extraction ----------------------------
with right_col:
    st.markdown('<div class="orange-subheader">✍️ What do you want to extract frm the images?</div>', unsafe_allow_html=True)
    st.session_state.instructions = st.text_area(
        "Extraction fields",
        value=st.session_state.instructions,
        height=90,
        placeholder="merchant name, total amount",
        label_visibility="collapsed",
    )
    requested_cols = parse_requested_fields(st.session_state.instructions)
    st.caption("Planned CSV columns: " + ", ".join(requested_cols))

# =============================== PIPELINE ====================================
if run_clicked:
    logs: list[str] = []
    try:
        enhance_file, gcv_run_files = _lazy_imports()
    except Exception as e:
        st.error(f"Could not import enhancer or gcv_engine. {e}")
        st.stop()

    src_paths = [p.resolve() for p in st.session_state.images.values()]
    dbg_dir = Path(dbg_dir_s).resolve()
    if save_dbg:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="ui_enhanced_"))
    tmp_enh = tmp_root / "images_enhanced"
    tmp_enh.mkdir(parents=True, exist_ok=True)

    progress = st.progress(0.0, text="Enhancing… 0 / 0")
    enhanced_paths: list[Path] = []

    # enhance files with flexible signature support
    total = len(src_paths)
    for i, p in enumerate(src_paths, 1):
        try:
            kwargs = {}
            try:
                sig = inspect.signature(enhance_file)
                if "mode" in sig.parameters:
                    kwargs["mode"] = mode
                if "strong" in sig.parameters:
                    kwargs["strong"] = strong
            except Exception:
                kwargs = {"strong": strong}

            img = enhance_file(p, **kwargs)
            out_name = p.with_suffix(".png").name
            tmp_out = tmp_enh / out_name
            cv2.imwrite(str(tmp_out), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            enhanced_paths.append(tmp_out)

            if save_dbg:
                dbg_out = dbg_dir / out_name
                cv2.imwrite(str(dbg_out), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            logs.append(f"[enhance] {i}/{total} -> {out_name} ({kwargs})")
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

    # save enhanced paths and jsonl path for reuse
    st.session_state.last_enhanced_paths = [str(p) for p in enhanced_paths]
    st.session_state.ocr_jsonl_path = str(jsonl_path) if jsonl_path else None

    # placeholder LLM step
    if enhanced_paths:
        requested_cols = parse_requested_fields(st.session_state.instructions)
        df = simulate_llm_processor(enhanced_paths, requested_cols, demo_fill)
        st.session_state.result_df = df
        st.session_state.result_csv_name = "extracted_receipts.csv"

    # cleanup
    if not keep_temp:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass

    # show logs and quick toasts
    st.write("### Run logs")
    st.code("\n".join(logs) if logs else "(no logs)", language="bash")

    if save_dbg:
        st.success(f"Enhanced copies saved to {dbg_dir}")
    if run_ocr and jsonl_path and Path(jsonl_path).exists():
        st.success(f"OCR JSONL saved to {jsonl_path}")
        st.toast("✅ OCR output written", icon="🗂️")
    else:
        st.toast("✅ Enhancement done", icon="✨")


# =============================== BELOW THE ROW ================================

# Selected images preview grid
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

# Optional preview of enhanced images
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

# OCR JSONL preview
if st.session_state.get("ocr_jsonl_path") and Path(st.session_state.ocr_jsonl_path).exists():
    st.write("### OCR output preview JSONL")
    try:
        preview_lines = []
        with open(st.session_state.ocr_jsonl_path, "r", encoding="utf-8", errors="ignore") as fh:
            for i, line in enumerate(fh):
                preview_lines.append(line.rstrip())
                if i >= 9:
                    break
        st.code("\n".join(preview_lines) if preview_lines else "(empty)", language="json")
    except Exception as ex:
        st.warning(f"Cannot preview JSONL. {ex}")

# Results section – table and CSV download
st.markdown("---")
st.markdown('<div class="orange-subheader">📊 Extracted data placeholder</div>', unsafe_allow_html=True)

# Rebuild button to change columns without rescanning
rebuild_cols = st.button("Rebuild table with current columns", disabled=not st.session_state.last_enhanced_paths)
if rebuild_cols and st.session_state.last_enhanced_paths:
    paths = [Path(p) for p in st.session_state.last_enhanced_paths]
    req_cols = parse_requested_fields(st.session_state.instructions)
    st.session_state.result_df = simulate_llm_processor(paths, req_cols, demo_fill)

if st.session_state.get("result_df") is not None and not st.session_state.result_df.empty:
    df = st.session_state.result_df
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_bytes = df_to_csv_bytes(df)
    st.download_button(
        label="📥 Download CSV",
        data=csv_bytes,
        file_name=st.session_state.result_csv_name,
        mime="text/csv",
    )

    with st.expander("Copy-friendly CSV text"):
        st.text_area("CSV", csv_bytes.decode("utf-8"), height=180)
else:
    st.caption("Run the pipeline to see extracted results here. This will later reflect real LLM output.")
