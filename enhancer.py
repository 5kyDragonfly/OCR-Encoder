# enhancer.py
# Enhance receipt images for OCR, save debug copies, and optionally chain to gcv_engine.py
#
# Requirements
#   pip install opencv-python pillow numpy python-dotenv
#
# Examples
#   python enhancer.py --files img1.jpg img2.png --debug-dir images_enhanced_debug --chain-gcv --gcv-out receipts.jsonl
#   python enhancer.py --list paths.txt --chain-gcv --pipe-parser "python parser.py"

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def read_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        # Pillow fallback then convert RGB to BGR
        img = np.array(Image.open(path).convert("RGB"))[:, :, ::-1]
    return img

def to_gray(bgr: np.ndarray) -> np.ndarray:
    if len(bgr.shape) == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def auto_scale(gray: np.ndarray, min_side: int = 1200, max_side: int = 2400) -> np.ndarray:
    h, w = gray.shape[:2]
    smin = min(h, w)
    smax = max(h, w)
    scale = 1.0
    if smin < min_side:
        scale = min_side / smin
    elif smax > max_side:
        scale = max_side / smax
    if abs(scale - 1.0) < 0.05:
        return gray
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)

def enhance_gray(gray: np.ndarray, strong: bool = False) -> np.ndarray:
    # 1 denoise
    h_val = 15 if not strong else 25
    den = cv2.fastNlMeansDenoising(gray, h=h_val, templateWindowSize=7, searchWindowSize=21)

    # 2 contrast via CLAHE
    clip = 3.0 if not strong else 4.0
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    eq = clahe.apply(den)

    # 3 unsharp mask for crisp edges
    blur = cv2.GaussianBlur(eq, (0, 0), sigmaX=1.2)
    sharp = cv2.addWeighted(eq, 1.5, blur, -0.5, 0)

    # 4 adaptive threshold
    block = 35 if not strong else 31
    C = 10 if not strong else 8
    th = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, C
    )

    # 5 small speck removal then slight close to join gaps
    kern_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kern_open, iterations=1)

    kern_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kern_close, iterations=1)

    # Ensure white background
    if closed.mean() < 127:
        closed = cv2.bitwise_not(closed)

    return closed

def enhance_file(path: Path, strong: bool = False) -> np.ndarray:
    bgr = read_image_bgr(path)
    gray = to_gray(bgr)
    gray = auto_scale(gray)
    return enhance_gray(gray, strong=strong)

def iter_inputs(files_and_dirs: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()
    for s in files_and_dirs:
        if not s:
            continue
        p = Path(s).expanduser()
        if not p.exists():
            logging.warning("Not found %s", s)
            continue
        if p.is_file():
            if p.suffix.lower() in VALID_EXT and str(p.resolve()) not in seen:
                out.append(p.resolve())
                seen.add(str(p.resolve()))
        elif p.is_dir():
            for g in p.rglob("*"):
                if g.is_file() and g.suffix.lower() in VALID_EXT and str(g.resolve()) not in seen:
                    out.append(g.resolve())
                    seen.add(str(g.resolve()))
    out.sort()
    return out

def save_png(img: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

def build_list_file(paths: List[Path], tmp_dir: Path) -> Path:
    lst = tmp_dir / "enhanced_paths.txt"
    lst.write_text("\n".join(str(p) for p in paths), encoding="utf-8")
    return lst

def chain_to_gcv(list_file: Path, gcv_out: str | None, parser_cmd: str | None) -> int:
    cmd = f'python gcv_engine.py --list "{list_file}"'
    if gcv_out:
        cmd += f' --out "{gcv_out}"'
    if parser_cmd:
        cmd += f' | {parser_cmd}'
    logging.info("Running %s", cmd)
    return os.system(cmd)

def run(
    inputs: List[str],
    debug_dir: Path | None,
    chain_gcv: bool,
    gcv_out: str | None,
    parser_cmd: str | None,
    keep_temp: bool,
    strong: bool,
) -> int:
    files = iter_inputs(inputs)
    if not files:
        logging.error("No valid images")
        return 2

    tmp_root = Path(tempfile.mkdtemp(prefix="enhanced_tmp_"))
    tmp_enh = tmp_root / "images_enhanced"
    tmp_enh.mkdir(parents=True, exist_ok=True)

    debug_paths: List[Path] = []
    enhanced_paths: List[Path] = []

    try:
        for i, p in enumerate(files, 1):
            try:
                img = enhance_file(p, strong=strong)
                # Save to temp working set as PNG
                out_name = p.with_suffix(".png").name
                tmp_path = tmp_enh / out_name
                save_png(img, tmp_path)
                enhanced_paths.append(tmp_path)

                # Optional debug copy
                if debug_dir:
                    dbg_path = debug_dir / out_name
                    save_png(img, dbg_path)
                    debug_paths.append(dbg_path)

                logging.info("Enhanced %d of %d -> %s", i, len(files), out_name)
            except Exception as e:
                logging.exception("Failed on %s %s", p.name, e)

        if not enhanced_paths:
            logging.error("No images were enhanced")
            return 2

        if chain_gcv:
            lst = build_list_file(enhanced_paths, tmp_root)
            rc = chain_to_gcv(lst, gcv_out=gcv_out, parser_cmd=parser_cmd)
            if rc != 0:
                logging.error("gcv_engine returned non zero %s", rc)
                return rc

        return 0
    finally:
        if not keep_temp:
            try:
                shutil.rmtree(tmp_root, ignore_errors=True)
            except Exception:
                pass

def main() -> None:
    ap = argparse.ArgumentParser(description="Enhance receipt images and optionally chain to GCV")
    ap.add_argument("--files", nargs="*", default=[], help="image files or directories")
    ap.add_argument("--list", help="text file with one path per line")
    ap.add_argument("--debug-dir", default="images_enhanced_debug", help="folder to save debug copies")
    ap.add_argument("--no-debug", action="store_true", help="do not save debug copies")
    ap.add_argument("--chain-gcv", action="store_true", help="run gcv_engine.py after enhancing")
    ap.add_argument("--gcv-out", help="path to write JSONL from gcv_engine. omit to stream to stdout or to parser")
    ap.add_argument("--pipe-parser", help="shell command to pipe GCV output to, example python parser.py")
    ap.add_argument("--keep-temp", action="store_true", help="keep temp enhanced files")
    ap.add_argument("--strong", action="store_true", help="use stronger denoise and threshold")
    args = ap.parse_args()

    inputs: List[str] = []
    if args.list:
        try:
            inputs.extend([ln.strip() for ln in Path(args.list).read_text(encoding="utf-8").splitlines() if ln.strip()])
        except Exception as e:
            logging.error("Cannot read list file %s", e)
            raise SystemExit(2)
    inputs.extend(args.files or [])

    debug_dir = None if args.no_debug else Path(args.debug_dir)

    rc = run(
        inputs=inputs,
        debug_dir=debug_dir,
        chain_gcv=args.chain_gcv,
        gcv_out=args.gcv_out,
        parser_cmd=args.pipe_parser,
        keep_temp=args.keep_temp,
        strong=args.strong,
    )
    raise SystemExit(rc)

if __name__ == "__main__":
    main()
