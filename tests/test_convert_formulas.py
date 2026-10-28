"""The slower models per conversion: the formula model reads only in a careful conversion ("Đọc kỹ") of a file with
at most MFR_MAX_PAGES scanned pages (otherwise formulas are read without it), VietOCR only in one with at most
VIETOCR_MAX_PAGES (convert() turns them off for its own thread only), and the result says so. The task option
"careful" reaches convert().

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

from app import convert, mathocr, ocr, storage, tasks, vietocr  # noqa: E402
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
    def run_convert(
        self, pages: int, limit: int, vietocr_limit: int = 30, careful: bool = True
    ) -> tuple[dict, list[tuple[str, bool]]]:
        wanted = []

        def fake_ocr_page(page, languages):
            wanted.append((ocr.formula_reader(), vietocr._wanted.get() and vietocr.available()))
            layout = image_page(page)
            layout.formula_pictures = 2  # e.g. two fractions kept as pictures
            return layout

        path = scan_pdf(pages)
        try:
            with (
                mock.patch.object(convert, "MFR_MAX_PAGES", limit),
                mock.patch.object(convert, "VIETOCR_MAX_PAGES", vietocr_limit),
                mock.patch.object(vietocr, "VIETOCR_ENABLED", True),
                mock.patch.object(vietocr, "MODEL_FILES", ()),
                mock.patch.object(convert, "ocr_page", fake_ocr_page),
                mock.patch.object(mathocr, "installed", return_value=True),  # the model counts as present
                mock.patch.object(ocr, "FORMULAS", "scripts"),
            ):
                out, info = convert.convert(path, "ocr", ["vie"], lambda i, n: None, careful=careful)
                out.unlink()
        finally:
            path.unlink()
        return info, wanted

    def test_long_scan_skips_formulas(self):
        info, wanted = self.run_convert(3, 2)
        self.assertTrue(info["formulas_skipped"])
        self.assertFalse(info["vietocr_skipped"])
        self.assertEqual(info["formulas"], "scripts")
        self.assertEqual(wanted, [("scripts", True)] * 3)
        self.assertTrue(mathocr._wanted.get())  # back on after the conversion
        self.assertIsNone(ocr._formulas.get())

    def test_very_long_scan_skips_vietocr(self):
        info, wanted = self.run_convert(3, 1, vietocr_limit=2)
        self.assertTrue(info["formulas_skipped"] and info["vietocr_skipped"])
        self.assertEqual(wanted, [("scripts", False)] * 3)
        self.assertTrue(vietocr._wanted.get())
        self.assertIsNone(self.reader_while_skipped())

    def reader_while_skipped(self):
        with mock.patch.object(vietocr, "_reader", object()), vietocr.skipped():
            return vietocr.reader()  # a loaded model is not used either

    def test_short_scan_reads_formulas(self):
        info, wanted = self.run_convert(2, 2)
        self.assertFalse(info["formulas_skipped"])
        self.assertEqual((info["careful"], info["formulas"]), (True, "model"))
        self.assertEqual(wanted, [("model", True)] * 2)

    def test_default_conversion_reads_without_the_model(self):
        for pages in (2, 3):
            info, wanted = self.run_convert(pages, 2, careful=False)
            self.assertFalse(info["formulas_skipped"])  # the model was not wanted
            self.assertEqual((info["careful"], info["formulas"]), (False, "scripts"))
            self.assertEqual(wanted, [("scripts", True)] * pages)
            self.assertEqual(info["formula_pictures"], 2 * pages)  # the page offers "Đọc kỹ" for these

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


class CarefulOptionTest(unittest.TestCase):
    def payload(self, **options):
        return {"task_id": "0" * 32, "tool": "pdf-to-word", "input_tokens": ["x.pdf"], "options": options}

    def test_option_reaches_convert(self):
        with mock.patch.object(storage, "exists", return_value=True):
            self.assertFalse(tasks._validate(self.payload(ocr_type="ocr")).options["careful"])
            task = tasks._validate(self.payload(ocr_type="ocr", careful=True))
            self.assertTrue(task.options["careful"])
            with self.assertRaises(ValueError):
                tasks._validate(self.payload(ocr_type="ocr", careful="yes"))
        out = Path(tempfile.mkstemp(suffix=".docx")[1])
        with (
            mock.patch.object(tasks, "convert", return_value=(out, {})) as conv,
            mock.patch.object(storage, "store_output", return_value="y.docx"),
            mock.patch.object(storage, "path_for", return_value=Path("x.pdf")),
        ):
            tasks.TaskManager._run(None, task)
        out.unlink(missing_ok=True)
        self.assertEqual(task.status, "done")
        self.assertIs(conv.call_args.args[5], True)


if __name__ == "__main__":
    unittest.main()
