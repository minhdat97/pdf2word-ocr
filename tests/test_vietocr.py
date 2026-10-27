"""app.vietocr: the rule choosing between Tesseract and VietOCR, Pillow-exact resizing,
loading, and (when models/vietocr is there) one real reading.

    python -m unittest discover -s tests
"""

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from app import vietocr
from app.ocr import Line, Word
from app.settings import VIETOCR_DIR


class ChooseTest(unittest.TestCase):
    def test_same_letters_take_vietocr_marks_in_tesseract_case(self):
        self.assertEqual(vietocr.choose("Nguyên", 95, "nguyễn", 0.9), "Nguyễn")
        self.assertEqual(vietocr.choose("TRƯƠNG", 70, "trường", 0.4), "TRƯỜNG")
        self.assertEqual(vietocr.choose("dien", 96, "điện", 0.9), "dien")  # đ, ê are letters: other letters

    def test_sure_word_keeps_its_marks(self):
        self.assertEqual(vietocr.choose("khí", 93, "khi", 0.9), "khí")  # Tesseract >= 90 and VietOCR drops a mark
        self.assertEqual(vietocr.choose("khí", 85, "khi", 0.9), "khi")  # unsure: VietOCR's reading
        self.assertEqual(vietocr.choose("khi", 95, "khí", 0.9), "khí")  # adding a mark is allowed

    def test_other_letters_only_when_tesseract_unsure_and_vietocr_sure(self):
        self.assertEqual(vietocr.choose("nguyẻn", 80, "nghiên", 0.8), "nghiên")
        self.assertEqual(vietocr.choose("nguyẻn", 92, "nghiên", 0.8), "nguyẻn")
        self.assertEqual(vietocr.choose("nguyẻn", 80, "nghiên", 0.49), "nguyẻn")

    def test_question_marks_and_math_are_kept(self):
        self.assertEqual(vietocr.choose("R=", 60, "?", 0.9), "R=")
        self.assertEqual(vietocr.choose("(K)=t(°C)+273", 50, "?t(?C) % 273", 0.9), "(K)=t(°C)+273")
        self.assertEqual(vietocr.choose("x2+1", 50, "x^{2}+1", 0.9), "x^{2}+1")  # keeps "+": taken
        self.assertEqual(vietocr.choose("ai?", 50, "ai?", 0.9), "ai?")
        self.assertEqual(vietocr.choose("tại", 60, "", 0.0), "tại")
        self.assertEqual(vietocr.choose("A.83°C.", 88, "A.8,3 ?C.", 0.92), "A.8,3 °C.")  # VietOCR has no "°"
        self.assertEqual(vietocr.choose("c.-3°C.", 73, "C.3 ?C", 0.82), "c.-3°C.")  # would lose the minus

    def test_case_and_punctuation_of_a_sure_word_stay(self):
        # a word crop shows neither the line's capitals nor the end of its sentence
        self.assertEqual(vietocr.choose("lần.", 86, "Lần", 0.85), "lần.")
        self.assertEqual(vietocr.choose("B.3kPa.", 91, "B.3 kpa.", 0.88), "B.3 kPa.")
        self.assertEqual(vietocr.choose("NaNs.", 70, "Nan3", 0.70), "NaN3.")
        self.assertEqual(vietocr.choose("c.9km.", 50, "C.9 km.", 0.85), "C.9 km.")  # unsure: VietOCR's case
        # the final mark goes back only where VietOCR read the word up to it
        self.assertEqual(vietocr.choose("10.", 37, "10:3", 0.83), "10:3")
        self.assertEqual(vietocr.choose("x.10?", 13, "x.10-9", 0.88), "x.10-9")  # "?" for an exponent
        # a mark VietOCR leaves out elsewhere: Tesseract's word
        self.assertEqual(vietocr.choose("T,;", 63, "T,", 0.62), "T,;")
        self.assertEqual(vietocr.choose("[ĐNH]", 79, "ĐNH", 0.75), "[ĐNH]")
        # a mark read as another, or as a letter: VietOCR's word
        self.assertEqual(vietocr.choose("0.25", 68, "0,25", 0.93), "0,25")
        self.assertEqual(vietocr.choose("II.", 62, "III", 0.90), "III")


# expected values from Pillow 13: Image.fromarray(img).resize((w, h), Image.LANCZOS)
PILLOW_CASES = {
    (9, 20, 4, 3): [[120, 128, 126, 115], [133, 130, 121, 130], [130, 124, 123, 134]],  # columns first
    (20, 9, 4, 4): [[123, 138, 123, 120], [134, 134, 126, 118], [121, 136, 126, 120], [122, 130, 128, 120]],
    (5, 6, 11, 8): [  # enlarging
        [0, 0, 47, 83, 97, 121, 148, 172, 202, 233, 246],
        [101, 63, 20, 31, 87, 118, 135, 158, 190, 221, 234],
        [255, 176, 49, 2, 52, 86, 110, 133, 163, 194, 207],
        [219, 221, 195, 96, 11, 24, 89, 117, 140, 172, 185],
        [155, 204, 255, 204, 95, 52, 69, 90, 125, 155, 167],
        [152, 166, 195, 250, 247, 141, 40, 46, 106, 134, 145],
        [128, 134, 158, 235, 255, 137, 13, 16, 81, 108, 118],
        [113, 125, 152, 210, 211, 105, 1, 7, 67, 95, 106],
    ],
}


class ResizeTest(unittest.TestCase):
    def test_matches_pillow(self):
        for (h, w, nw, nh), expected in PILLOW_CASES.items():
            img = ((np.arange(h * w).reshape(h, w) * 37 + np.arange(w) * 11) % 256).astype(np.uint8)
            self.assertEqual(vietocr.resize(img, nw, nh).tolist(), expected, (h, w, nw, nh))
            rgb = np.repeat(img[:, :, None], 3, axis=2)
            self.assertTrue((vietocr.resize(rgb, nw, nh)[:, :, 1] == np.array(expected)).all())


class FakeReader:
    def __init__(self, reads):
        self.reads = reads
        self.shapes = []

    def read(self, images):
        self.shapes += [im.shape for im in images]
        return self.reads[: len(images)]


class RefineTest(unittest.TestCase):
    def test_words_get_their_reading(self):
        gray = np.full((200, 400), 255, np.uint8)
        ln = Line(block=1, par=1, baseline=60.0, em_px=40.0)
        ln.words = [Word(10, 40, 90, 70, 95.0, "Trương", 0, 40.0), Word(100, 30, 200, 80, 50.0, "xyz", 0, 40.0)]
        reader = FakeReader([("trường", 0.9), ("học", 0.8)])
        self.assertEqual(vietocr.refine_words([ln], gray, reader), 2)
        self.assertEqual([w.text for w in ln.words], ["Trường", "học"])
        # line rows 30..80 padded by a quarter of 50 px; word columns padded the same
        self.assertEqual(reader.shapes, [(74, 102), (74, 124)])


class LoaderTest(unittest.TestCase):
    def tearDown(self):
        vietocr._loaded, vietocr._reader = False, None

    def test_missing_models_mean_no_reader(self):
        vietocr._loaded, vietocr._reader = False, None
        with mock.patch.object(vietocr, "VIETOCR_DIR", Path("/nonexistent/models/vietocr")):
            self.assertIsNone(vietocr.reader())

    def test_disabled(self):
        vietocr._loaded, vietocr._reader = False, None
        with mock.patch.object(vietocr, "VIETOCR_ENABLED", False):
            self.assertIsNone(vietocr.reader())


@unittest.skipUnless(all((VIETOCR_DIR / f).is_file() for f in vietocr.MODEL_FILES), "models/vietocr not fetched")
class ModelTest(unittest.TestCase):
    def test_reads_a_rendered_word(self):
        import pymupdf

        doc = pymupdf.open()
        page = doc.new_page(width=200, height=60)
        page.insert_text((20, 40), "Toán 2026", fontname="tiro", fontsize=16)
        pix = page.get_pixmap(dpi=300, colorspace=pymupdf.csGRAY)
        gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)[:, : pix.width]
        rows, cols = np.nonzero(gray < 128)
        pad = 12
        crop = gray[rows.min() - pad : rows.max() + pad, cols.min() - pad : cols.max() + pad]
        reader = vietocr.VietOCR(VIETOCR_DIR, 1)
        ((text, prob),) = reader.read([crop])
        self.assertEqual(text, "Toán 2026")
        self.assertGreater(prob, 0.5)


if __name__ == "__main__":
    unittest.main()
