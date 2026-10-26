"""The `analyse-document` tool: classify a PDF as text, scan or mixed.

The frontend uses `kind` to preselect a conversion mode: OCR when a file has
images but no text layer.
"""

from pathlib import Path

import pymupdf

from .errors import TaskError
from .settings import MAX_PAGES

MIN_TEXT_CHARS = 20
MIN_IMAGE_COVERAGE = 0.05


def open_pdf(path: Path, check_limits: bool = True) -> pymupdf.Document:
    try:
        doc = pymupdf.open(path, filetype="pdf")
    except Exception as exc:
        raise TaskError("corrupt_pdf", str(exc)) from exc
    if doc.needs_pass:
        raise TaskError("password")
    if check_limits:
        if doc.page_count == 0:
            raise TaskError("no_pages_in_pdf")
        if doc.page_count > MAX_PAGES:
            raise TaskError("too_many_pages", f"{doc.page_count} > {MAX_PAGES}")
    return doc


def _image_coverage(page: pymupdf.Page) -> float:
    area = abs(page.rect)
    if not area:
        return 0.0
    covered = sum(abs(pymupdf.Rect(info["bbox"]) & page.rect) for info in page.get_image_info())
    return min(covered / area, 1.0)


def page_kind(page: pymupdf.Page) -> dict:
    text_chars = len(page.get_text("text").strip())
    coverage = _image_coverage(page)
    has_text = text_chars >= MIN_TEXT_CHARS
    has_image = coverage >= MIN_IMAGE_COVERAGE
    if has_text and has_image:
        kind = "mixed"
    elif has_text:
        kind = "text"
    elif has_image:
        kind = "scan"
    else:
        kind = "empty"
    return {"kind": kind, "text_chars": text_chars, "image_coverage": round(coverage, 3)}


def analyse(path: Path) -> dict:
    try:
        doc = pymupdf.open(path, filetype="pdf")
    except Exception as exc:
        raise TaskError("corrupt_pdf", str(exc)) from exc

    base = {
        "file_size": path.stat().st_size,
        "page_count": doc.page_count,
        "encrypted": bool(doc.is_encrypted),
        "password_protected": bool(doc.needs_pass),
    }
    if doc.needs_pass:
        return {**base, "contains_text": None, "contains_image": None, "kind": None, "pages": []}

    pages = [{"index": i, **page_kind(p)} for i, p in enumerate(doc) if i < 200]
    kinds = {p["kind"] for p in pages} - {"empty"}
    kind = kinds.pop() if len(kinds) == 1 else ("mixed" if kinds else "empty")
    return {
        **base,
        "contains_text": any(p["kind"] in ("text", "mixed") for p in pages),
        "contains_image": any(p["kind"] in ("scan", "mixed") for p in pages),
        "kind": kind,
        "pages": pages,
    }
