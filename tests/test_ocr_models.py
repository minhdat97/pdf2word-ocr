"""app/ocr.py with a model that fails while reading (damaged or mismatched files, e.g. a config.json of another
export next to old weights): the page is still converted from Tesseract's words, and the model is turned off.
Needs tesseract and tessdata/vie.traineddata.

    python -m unittest discover -s tests
"""

import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf  # noqa: E402

from app import mathocr, ocr, vietocr  # noqa: E402
from app.settings import TESSDATA_DIR  # noqa: E402


class BrokenReader:
    def read(self, images):
        raise RuntimeError("Got invalid dimensions for input: past_k for the following indices index: 0 Got: 4 Expected: 6")


@unittest.skipUnless(shutil.which("tesseract") and (TESSDATA_DIR / "vie.traineddata").is_file(), "tesseract or vie missing")
class BrokenModelTest(unittest.TestCase):
    def setUp(self):
        doc = pymupdf.open()
        page = doc.new_page(width=300, height=120)
        page.insert_text((20, 50), "Câu 1. Một vật có khối lượng 2 kg.", fontname="tiro", fontsize=16)
        self.page = page
        vietocr._failed = mathocr._reader_failed = False

    def tearDown(self):
        vietocr._failed, vietocr._reader = False, None
        mathocr._reader_failed, mathocr._reader = False, None

    def text(self, layout):
        return " ".join(el.text for el in layout.elements)

    def test_vietocr_failing_keeps_tesseract_words(self):
        with mock.patch.object(vietocr, "reader", side_effect=[BrokenReader(), None]), \
                mock.patch.object(mathocr, "available", return_value=False), self.assertLogs("pdf2word.vietocr", "ERROR"):
            layout = ocr.ocr_page(self.page, ["vie"])
        self.assertIn("Câu 1.", self.text(layout))
        self.assertTrue(vietocr._failed)

    def test_formula_model_failing_keeps_ocr_text(self):
        with mock.patch.object(vietocr, "reader", return_value=None), \
                mock.patch.object(ocr.mathregions, "find", side_effect=RuntimeError("past_k")), \
                mock.patch.object(mathocr, "available", return_value=True), self.assertLogs("pdf2word.mathocr", "ERROR"):
            layout = ocr.ocr_page(self.page, ["vie"])
        self.assertIn("2 kg", self.text(layout))
        self.assertTrue(mathocr._reader_failed)
        self.assertIsNone(mathocr.get_reader())


if __name__ == "__main__":
    unittest.main()
