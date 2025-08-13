# gcv_engine.py
# Google Cloud Vision OCR runner for receipt images
# One JSON object per receipt on stdout or in a JSONL file
#
# Requirements
#   pip install google-cloud-vision python-dotenv
#   .env can provide either:
#     GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
#       or
#     GCV_CREDENTIALS_JSON='{"type":"service_account",...}'   # literal JSON
#
# Examples
#   python gcv_engine.py --files img1.jpg img2.png --out receipts.jsonl
#   python gcv_engine.py --list /tmp/files.txt | python parser.py

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

from dotenv import load_dotenv

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _load_credentials_from_env() -> Optional[str]:
    """
    Returns a temporary credentials path if we materialize JSON from env.
    Otherwise returns None.
    """
    load_dotenv()  # no-op if no .env

    # If user provided a path already we are done
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac and Path(gac).exists():
        return None

    # Allow literal JSON via env so you need no file on disk
    json_blob = os.environ.get("GCV_CREDENTIALS_JSON")
    if json_blob:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.write(json_blob.encode("utf-8"))
        tmp.flush()
        tmp.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
        return tmp.name

    # Nothing available
    return None

def _iter_files(files: Iterable[str]) -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []
    for f in files:
        if not f:
            continue
        p = Path(f).expanduser().resolve()
        if not p.exists():
            print(f"[warn] not found {f}", file=sys.stderr)
            continue
        if p.is_file() and p.suffix.lower() in VALID_EXT and str(p) not in seen:
            out.append(p)
            seen.add(str(p))
        elif p.is_dir():
            # Accept directories too in case caller mixes both
            for g in p.rglob("*"):
                if g.is_file() and g.suffix.lower() in VALID_EXT and str(g.resolve()) not in seen:
                    out.append(g.resolve())
                    seen.add(str(g.resolve()))
        else:
            print(f"[warn] unsupported path {f}", file=sys.stderr)
    out.sort()
    return out

def _sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def _ocr_gcv(image_bytes: bytes) -> Dict[str, Any]:
    try:
        from google.cloud import vision  # imported lazily
    except Exception as e:
        raise RuntimeError("google-cloud-vision not installed. pip install google-cloud-vision") from e

    client = vision.ImageAnnotatorClient()
    gimg = vision.Image(content=image_bytes)
    resp = client.document_text_detection(image=gimg)

    if getattr(resp, "error", None) and resp.error.message:
        raise RuntimeError(f"GCV error {resp.error.message}")

    ann = resp.full_text_annotation
    text = ann.text if ann and getattr(ann, "text", None) else ""

    locale = None
    try:
        if ann.pages and ann.pages[0].property.detected_languages:
            locale = ann.pages[0].property.detected_languages[0].language_code
    except Exception:
        locale = None

    return {"text": text, "locale": locale, "engine": "gcv_document_text_detection"}

def _jsonl_write(stream, obj: Dict[str, Any]) -> None:
    stream.write(json.dumps(obj, ensure_ascii=False) + "\n")
    stream.flush()

def run_files(file_paths: List[str], out_path: str | None = None, minimal: bool = False) -> int:
    tmp_cred = _load_credentials_from_env()
    files = _iter_files(file_paths)
    if not files:
        print("[error] no valid images", file=sys.stderr)
        return 2

    out_stream = open(out_path, "w", encoding="utf-8") if out_path else sys.stdout
    close_stream = out_stream is not sys.stdout
    try:
        for i, p in enumerate(files, 1):
            try:
                b = p.read_bytes()
            except Exception as e:
                print(f"[warn] cannot read {p.name} {e}", file=sys.stderr)
                continue

            try:
                ocr = _ocr_gcv(b)
            except Exception as e:
                print(f"[warn] ocr failed {p.name} {e}", file=sys.stderr)
                continue

            # normalize text: strip leading/trailing whitespace, keep internal newlines
            txt = (ocr["text"] or "").strip()

            if minimal:
                rec = {
                    "text": txt,
                    # include locale if present; omit if None to keep it minimal
                    **({"locale": ocr.get("locale")} if ocr.get("locale") else {})
                }
            else:
                # existing verbose record (kept for backward-compat)
                rec = {
                    "id": _sha1(b),
                    "filename": p.name,
                    "source_path": str(p),
                    "text": txt,
                    "locale": ocr.get("locale"),
                    "engine": ocr["engine"],
                }

            _jsonl_write(out_stream, rec)
            print(f"[info] {i}/{len(files)} {p.name}", file=sys.stderr)
    finally:
        if close_stream:
            out_stream.close()
        if tmp_cred:
            try:
                Path(tmp_cred).unlink(missing_ok=True)
            except Exception:
                pass
    return 0

def main() -> None:
    parser = argparse.ArgumentParser(description="OCR receipts with Google Cloud Vision to JSONL")
    parser.add_argument("--files", nargs="*", help="image files and or directories", default=[])
    parser.add_argument("--list", help="path to a text file with one image path per line")
    parser.add_argument("--out", help="output JSONL file. omit to write to stdout")
    parser.add_argument("--minimal", action="store_true",
                        help="emit minimal JSONL with only {'text', 'locale?'}")
    args = parser.parse_args()

    files: List[str] = []
    if args.list:
        try:
            lines = Path(args.list).read_text(encoding="utf-8").splitlines()
            files.extend([ln.strip() for ln in lines if ln.strip()])
        except Exception as e:
            print(f"[error] cannot read list file {e}", file=sys.stderr)
            sys.exit(2)
    files.extend(args.files or [])

    rc = run_files(files, out_path=args.out, minimal=args.minimal)
    sys.exit(rc)

if __name__ == "__main__":
    main()
