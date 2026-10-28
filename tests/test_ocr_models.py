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
                mock.patch.object(mathocr, "installed", return_value=True), ocr.formulas_mode("model"), \
                self.assertLogs("pdf2word.mathocr", "ERROR"):
            layout = ocr.ocr_page(self.page, ["vie"])
            self.assertEqual(ocr.formula_reader(), "scripts")  # the model is off: formulas are read without it
        self.assertIn("2 kg", self.text(layout))
        self.assertTrue(mathocr._reader_failed)
        self.assertIsNone(mathocr.get_reader())

    def test_formula_pictures_are_counted(self):
        def keep_as_picture(lines, grids, gray, ink, rgb, dpi):  # a region read as "tall" and not read: a picture
            first = lines[0].words[0]
            pic = ocr.FormulaPicture(ocr._png(rgb[first.y0 : first.y1, first.x0 : first.x1]), 10.0, 10.0, 0.0)
            lines[0].words[0] = ocr.Word(first.x0, first.y0, first.x1, first.y1, 50.0, ocr.OBJ, first.line_no,
                                         first.em_px, math=pic)
            return 1

        with mock.patch.object(vietocr, "reader", return_value=None), \
                mock.patch.object(ocr, "read_formulas", side_effect=keep_as_picture):
            layout = ocr.ocr_page(self.page, ["vie"])
        self.assertEqual(layout.formula_pictures, 1)
        with mock.patch.object(vietocr, "reader", return_value=None), \
                mock.patch.object(ocr, "read_formulas", return_value=0):
            self.assertEqual(ocr.ocr_page(self.page, ["vie"]).formula_pictures, 0)

    def test_reading_without_the_model_failing_keeps_ocr_text(self):
        with mock.patch.object(vietocr, "reader", return_value=None), \
                mock.patch.object(ocr.mathregions, "find", side_effect=RuntimeError("bad region")), \
                mock.patch.object(mathocr, "installed", return_value=True), \
                mock.patch.object(ocr, "FORMULAS", "scripts"), self.assertLogs("pdf2word.ocr", "ERROR"):
            layout = ocr.ocr_page(self.page, ["vie"])
        self.assertIn("2 kg", self.text(layout))
        self.assertFalse(mathocr._reader_failed)  # a careful conversion can still use the model


if __name__ == "__main__":
    unittest.main()
