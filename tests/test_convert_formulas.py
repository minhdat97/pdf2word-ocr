"""The slower models per conversion: a file with more scanned pages than MFR_MAX_PAGES is converted without
formula reading, one with more than VIETOCR_MAX_PAGES without VietOCR too (convert() turns them off for its own
thread only), and the result says so.

    python -m unittest discover -s tests
"""

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf  # noqa: E402

from app import convert, mathocr, vietocr  # noqa: E402
from app.native import image_page  # noqa: E402


def scan_pdf(pages: int) -> Path:
    """A PDF of `pages` image-only pages (page_kind "scan")."""
    pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 200, 280), False)
    pix.clear_with(200)
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page(width=200, height=280)
        page.insert_image(page.rect, pixmap=pix)
    path = Path(tempfile.mkstemp(suffix=".pdf")[1])
    doc.save(path)
    return path


class FormulaLimitTest(unittest.TestCase):
    def run_convert(self, pages: int, limit: int, vietocr_limit: int = 30) -> tuple[dict, list[tuple[bool, bool]]]:
        wanted = []

        def fake_ocr_page(page, languages):
            wanted.append((mathocr.available(), vietocr._wanted.get() and vietocr.available()))
            return image_page(page)

        path = scan_pdf(pages)
        try:
            with (
                mock.patch.object(convert, "MFR_MAX_PAGES", limit),
                mock.patch.object(convert, "VIETOCR_MAX_PAGES", vietocr_limit),
                mock.patch.object(vietocr, "VIETOCR_ENABLED", True),
                mock.patch.object(vietocr, "MODEL_FILES", ()),
                mock.patch.object(convert, "ocr_page", fake_ocr_page),
                mock.patch.object(mathocr, "MFR_ENABLED", True),
                mock.patch.object(mathocr, "MODEL_FILES", ()),  # the model counts as present
            ):
                out, info = convert.convert(path, "ocr", ["vie"], lambda i, n: None)
                out.unlink()
        finally:
            path.unlink()
        return info, wanted

    def test_long_scan_skips_formulas(self):
        info, wanted = self.run_convert(3, 2)
        self.assertTrue(info["formulas_skipped"])
        self.assertFalse(info["vietocr_skipped"])
        self.assertEqual(wanted, [(False, True)] * 3)
        self.assertTrue(mathocr._wanted.get())  # back on after the conversion

    def test_very_long_scan_skips_vietocr(self):
        info, wanted = self.run_convert(3, 1, vietocr_limit=2)
        self.assertTrue(info["formulas_skipped"] and info["vietocr_skipped"])
        self.assertEqual(wanted, [(False, False)] * 3)
        self.assertTrue(vietocr._wanted.get())
        self.assertIsNone(self.reader_while_skipped())

    def reader_while_skipped(self):
        with mock.patch.object(vietocr, "_reader", object()), vietocr.skipped():
            return vietocr.reader()  # a loaded model is not used either

    def test_short_scan_reads_formulas(self):
        info, wanted = self.run_convert(2, 2)
        self.assertFalse(info["formulas_skipped"])
        self.assertEqual(wanted, [(True, True)] * 2)

    def test_skip_is_per_thread(self):
        seen = {}
        inside, done = threading.Event(), threading.Event()

        def other():
            inside.wait()
            seen["other"] = mathocr._wanted.get()
            done.set()

        t = threading.Thread(target=other)
        t.start()
        with mathocr.skipped():
            inside.set()
            done.wait()
            seen["own"] = mathocr._wanted.get()
        t.join()
        self.assertEqual(seen, {"other": True, "own": False})


if __name__ == "__main__":
    unittest.main()
