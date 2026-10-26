"""Helpers shared by the native-text and OCR pipelines."""

import re
from statistics import median

_URL = re.compile(
    r"(https?://|www\.)\S+|\b[\w-]+(\.[\w-]+)*\.(com|net|org|edu|gov|vn|info|io|cc|co)\b", re.I
)


_PAGE_NUMBER = re.compile(r"^\s*((trang|page)\s*(\d+(\s*/\s*\d+)?)?|\d+(\s*/\s*\d+)?)\s*$", re.I)


_SLASHES = str.maketrans({"\u2044": "/", "\u2215": "/", "\uff0f": "/"})


def canonical(text: str) -> str:
    """Fold look-alike characters OCR produces ("1⁄2" with a fraction slash)."""
    return text.translate(_SLASHES)


def looks_like_link(text: str) -> bool:
    return bool(_URL.search(text))


def looks_like_page_number(text: str) -> bool:
    return bool(_PAGE_NUMBER.match(canonical(text)))


# Looser, for OCR'd page labels: "Trang", "Trang 1/?", "Page 3", "2/5".
_PAGE_LABEL = re.compile(r"^\s*((trang|page)\b[\s\d/?.,|lI]*|\d{1,4}\s*(/\s*\d{1,4})?)\s*$", re.I)


def looks_like_page_label(text: str) -> bool:
    return bool(_PAGE_LABEL.match(canonical(text)))


def alignment(line_spans: list[tuple[float, float]], left: float, right: float) -> tuple[str, float]:
    """Classify a paragraph from its lines' (x0, x1) inside the text area [left, right].

    Returns (align, first_line_indent) with the indent in the input's unit.
    """
    width = max(right - left, 1)
    center = (left + right) / 2
    tol = 0.02 * width
    lefts = [a for a, _ in line_spans]
    rights = [b for _, b in line_spans]
    x0, x1 = min(lefts), max(rights)
    if all(abs((a + b) / 2 - center) < tol for a, b in line_spans) and x0 - left > 0.05 * width:
        return "center", 0.0
    if len(line_spans) == 1 and abs(x1 - right) < tol and x0 - left > 0.3 * width:
        return "right", 0.0
    align = "left"
    if len(rights) > 1 and right - x1 < 0.05 * width and all(abs(b - x1) < tol for b in rights[:-1]):
        align = "justify"
    body_left = median(lefts[1:]) if len(lefts) > 1 else lefts[0]
    return align, lefts[0] - body_left


def cell_align(spans: list[list[float]], x0: float, x1: float) -> str | None:
    """How a table cell's text lines sit between its rules: "center", "right",
    or None (left). A line filling the cell says nothing, so some line must
    leave a clear gap on the side(s) it is set away from."""
    if not spans:
        return None
    width = max(x1 - x0, 1e-6)
    tol = max(2.0, 0.08 * width)
    gaps = [(a - x0, x1 - b) for a, b in spans]
    if all(abs(left - right) <= tol for left, right in gaps) and any(min(g) > 0.15 * width for g in gaps):
        return "center"
    if all(right + tol < left for left, right in gaps) and max(right for _, right in gaps) <= 0.15 * width:
        if any(left > 0.3 * width for left, _ in gaps):
            return "right"
    return None
