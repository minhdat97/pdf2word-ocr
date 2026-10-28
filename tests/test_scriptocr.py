"""Superscripts and subscripts read without the formula model (app/scriptocr.py): the checks on their own, the
reader choice in app/ocr.py, and, with tesseract and tessdata/eng, a printed page read end to end.
Run: python -m unittest discover -s tests"""

import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app import mathocr, mathregions, ocr, scriptocr  # noqa: E402
from app.mathocr import MathDecision  # noqa: E402
from app.settings import TESSDATA_DIR  # noqa: E402


class AlignTest(unittest.TestCase):
    def test_one_character_a_glyph(self):
        self.assertEqual(scriptocr._align("6,42", [1.0, 0.4, 1.0, 1.0]), ["6", ",", "4", "2"])

    def test_touching_characters_share_a_glyph(self):
        # "m/" printed touching is one glyph as wide as both
        self.assertEqual(scriptocr._align("m/s.", [2.3, 0.8, 0.3]), ["m/", "s", "."])

    def test_fewer_characters_than_glyphs_or_no_fit(self):
        self.assertIsNone(scriptocr._align("821.10", [1.0, 0.4, 1.0, 1.0, 0.3, 0.6, 1.0]))  # a comma not read
        self.assertIsNone(scriptocr._align("mm", [0.3]))  # two wide letters on a dot


class SpaceTest(unittest.TestCase):
    def test_no_space_inside_numbers_or_before_marks(self):
        self.assertFalse(scriptocr._space("6,42.", "10"))
        self.assertFalse(scriptocr._space("12", "3"))
        self.assertFalse(scriptocr._space("}", "."))
        self.assertTrue(scriptocr._space("}", "J."))
        self.assertTrue(scriptocr._space("4v_{O}", "="))


class LettersSeenTest(unittest.TestCase):
    def test_letters_outside_scripts_must_be_in_the_ocr_text(self):
        self.assertTrue(scriptocr._letters_seen(r"4,57.10^{-21}\ J.", "4,57.10-21 J."))
        self.assertFalse(scriptocr._letters_seen(r"4,57.10^{-21}\ I.", "4,57.10-21 J."))  # "J" misread
        self.assertTrue(scriptocr._letters_seen("m_{0}.", "mụ."))  # marks and the subscript's letter do not count
        self.assertTrue(scriptocr._letters_seen("v_{H}", "vh"))


class UnitShapeTest(unittest.TestCase):
    def test_words_after_a_script_are_units(self):
        for text in ("J.", "m/s.", "kPa", "mol.K", "g/cm"):
            self.assertTrue(scriptocr._unit(text), text)
        for text in ("I.", "gem", "Tr"):
            self.assertFalse(scriptocr._unit(text), text)

    def test_outline_of_a_character(self):
        xh = 20
        self.assertTrue(scriptocr._shape("m", [1], 34, 20, 0, xh))
        self.assertFalse(scriptocr._shape("m", [1], 34, 30, 0, xh))  # too high for an x-height letter
        self.assertFalse(scriptocr._shape("/", [1], 16, 20, 0, xh))  # a "c" read as "/"
        self.assertTrue(scriptocr._shape("/", [1], 12, 32, 4, xh))
        self.assertFalse(scriptocr._shape("g", [1], 20, 20, 0, xh))  # no descender
        self.assertFalse(scriptocr._shape("=", [1], 20, 4, -10, xh))  # one bar


class Glyphs:
    """A tiny page of filled rectangles: x-height 20 px, baseline at y=100, its components as mathregions._Page
    keeps them."""

    def __init__(self, boxes):
        ink = np.zeros((200, 400), np.uint8)
        for x0, y0, x1, y1 in boxes:
            ink[y0:y1, x0:x1] = 1
        _, self.lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        self.st = st.astype(np.int64)

    def box(self, c):
        x, y, w, h = (int(v) for v in self.st[c, :4])
        return x, y, x + w, y + h

    def union(self, comps):
        boxes = [self.box(c) for c in comps]
        return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


class MarksTest(unittest.TestCase):
    def test_punctuation_by_shape(self):
        g = Glyphs([(10, 95, 15, 100), (30, 94, 35, 106), (50, 82, 55, 87), (50, 95, 55, 100), (70, 82, 75, 87),
                    (70, 94, 75, 106), (90, 70, 95, 75)])
        # components are numbered top-down, left to right by OpenCV: look them up by their box
        at = {g.box(c)[:2]: c for c in range(1, len(g.st))}
        mark = lambda *corners: scriptocr._mark(g, [at[c] for c in corners], 100, 20)
        self.assertEqual(mark((10, 95)), ".")
        self.assertEqual(mark((30, 94)), ",")
        self.assertEqual(mark((50, 82), (50, 95)), ":")
        self.assertEqual(mark((70, 82), (70, 94)), ";")
        self.assertIsNone(mark((90, 70)))  # a raised dot is no punctuation

    def test_units_join_parts_one_above_the_other(self):
        g = Glyphs([(10, 80, 30, 84), (10, 90, 30, 94), (40, 70, 52, 100)])  # "=" then a tall glyph
        self.assertEqual([len(u) for u in scriptocr._units(g, [1, 2, 3])], [2, 1])


class CharTest(unittest.TestCase):
    def test_glyph_checks(self):
        # 0: a narrow digit 0 in a subscript; 1: a round letter O in a subscript; 2: a flat minus; 3: a digit on the
        # line; 4: a small letter o on the line; 5: a glyph going under the baseline (J); 6: a raised digit
        g = Glyphs([(10, 92, 18, 108), (30, 92, 46, 108), (60, 70, 72, 73), (80, 72, 92, 100), (100, 82, 112, 100),
                    (120, 72, 130, 104), (140, 60, 150, 76)])
        at = {g.box(c)[:2]: c for c in range(1, len(g.st))}
        char = lambda ch, corner, role, minus=None: scriptocr._char(g, ch, [at[corner]], role, 100, 20, minus)
        self.assertEqual(char("O", (10, 92), "sub"), "0")
        self.assertEqual(char("0", (30, 92), "sub"), "O")
        self.assertEqual(char("-", (60, 70), "super", minus=at[(60, 70)]), "-")
        self.assertIsNone(char("-", (80, 72), "super"))  # a "-" where the geometry found no minus
        self.assertIsNone(char("l", (80, 72), "sub"))  # 1, l or I: the glyph does not tell
        self.assertEqual(char("7", (80, 72), "base"), "7")
        self.assertEqual(char("0", (100, 82), "base"), "o")
        self.assertIsNone(char("5", (100, 82), "base"))  # a digit lower than a digit
        self.assertIsNone(char("I", (120, 72), "base"))  # "I" for a glyph going under the line
        self.assertEqual(char("J", (120, 72), "base"), "J")
        self.assertIsNone(char("2", (140, 60), "base"))  # a raised glyph read on the line


class ReaderChoiceTest(unittest.TestCase):
    def test_settings_and_skipped_model(self):
        with mock.patch.object(mathocr, "available", return_value=True):
            for setting in ("scripts", "off", "model"):
                with mock.patch.object(ocr, "FORMULAS", setting):
                    self.assertEqual(ocr.formula_reader(), setting)
                    with ocr.formulas_mode("model"):  # a careful conversion: the model whatever the setting
                        self.assertEqual(ocr.formula_reader(), "model")
                    with ocr.formulas_mode("scripts"):
                        self.assertEqual(ocr.formula_reader(), "scripts" if setting != "off" else "off")
            self.assertEqual(ocr.formula_reader(), "scripts")  # the default
        # no model (not installed, broken, or skipped for a long scan): the default way
        with mock.patch.object(mathocr, "available", return_value=False):
            for setting, want in (("scripts", "scripts"), ("off", "off"), ("model", "scripts")):
                with mock.patch.object(ocr, "FORMULAS", setting), ocr.formulas_mode("model"):
                    self.assertEqual(ocr.formula_reader(), want)
        with mock.patch.object(mathocr, "installed", return_value=True), mathocr.skipped():
            self.assertFalse(mathocr.available())

    def test_scripts_mode_reads_with_scriptocr_and_keeps_tall_regions_as_pictures(self):
        gray = np.full((300, 900), 255, np.uint8)
        gray[100:130, 100:300] = 0  # a "number"
        gray[60:180, 500:600] = 0  # a "fraction"
        ink = (gray < 128).astype(np.uint8)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        words = [ocr.Word(100, 100, 300, 130, 40.0, "642.107'J.", 0, 30), ocr.Word(500, 100, 600, 130, 20.0, "x", 0, 30)]
        lines = [ocr.Line(1, 1, 130, 30, words)]
        regions = [mathregions.MathRegion((100, 100, 300, 130), [words[0]], "option", "642.107'J.", 30, lines[0]),
                   mathregions.MathRegion((500, 60, 600, 180), [words[1]], "tall", "x", 30, lines[0])]
        reading = MathDecision("text", parts=[("6,42.10", None), ("-21", "super"), (" J.", None)], conf=0.9)
        with mock.patch.object(ocr, "FORMULAS", "scripts"), \
                mock.patch.object(mathregions, "find", return_value=regions), \
                mock.patch.object(scriptocr, "read_regions", return_value=[reading, None]) as read:
            placed = ocr.read_formulas(lines, [], gray, ink, rgb, 300)
        self.assertEqual(placed, 2)
        read.assert_called_once()
        self.assertEqual(lines[0].words[0].math, reading)
        self.assertIsInstance(lines[0].words[1].math, ocr.FormulaPicture)


@unittest.skipUnless(shutil.which("tesseract") and (TESSDATA_DIR / "eng.traineddata").is_file(), "tesseract or eng missing")
class PrintedPageTest(unittest.TestCase):
    """A page printed in Times (PyMuPDF), read as a scan would be: Tesseract's lines, the detector, this reader."""

    def page(self):
        import pymupdf

        doc = pymupdf.open()
        pg = doc.new_page(width=420, height=160)
        y = 60
        x = 40 + pymupdf.get_text_length("The energy is ", "tiro", 13)
        pg.insert_text((40, y), "The energy is ", fontname="tiro", fontsize=13)
        pg.insert_text((x, y), "6,42.10", fontname="tiro", fontsize=13)
        x += pymupdf.get_text_length("6,42.10", "tiro", 13)
        pg.insert_text((x + 0.5, y - 5), "-21", fontname="tiro", fontsize=8.5)  # Times has no U+2212
        x += pymupdf.get_text_length("-21", "tiro", 8.5) + 1
        pg.insert_text((x, y), " J for this gas", fontname="tiro", fontsize=13)
        y = 110
        pg.insert_text((40, y), "The moles of gas n", fontname="tiro", fontsize=13)
        x = 40 + pymupdf.get_text_length("The moles of gas n", "tiro", 13)
        pg.insert_text((x, y + 3), "1", fontname="tiro", fontsize=8.5)
        x += pymupdf.get_text_length("1", "tiro", 8.5)
        pg.insert_text((x, y), " at its temperature", fontname="tiro", fontsize=13)
        pix = pg.get_pixmap(dpi=300, colorspace=pymupdf.csGRAY, alpha=False)
        gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)[:, : pix.width].copy()
        return gray

    def test_reads_exponent_and_index(self):
        gray = self.page()
        _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        lines = ocr.tesseract_lines(gray, "eng", 300)
        regions = mathregions.find(lines, gray, ink, cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), 300)
        why = {}
        found = [d.as_latex() for d in scriptocr.read_regions(regions, gray, ink, lines, 300, why) if d]
        self.assertIn("6,42.10^{-21}", found, why)
        self.assertIn("n_{1}", found, why)


if __name__ == "__main__":
    unittest.main()
