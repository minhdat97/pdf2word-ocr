"""The "rules" formula detector (app/mathregions.py) on synthetic pages: scripts in running text, 2-D formulas
(fraction bar, bar over a letter, radical), answer labels Tesseract garbled, and how app/ocr.py writes such a
label. Glyphs are filled rectangles; 1em = 30 px, x-height 14 px, baseline at y=100.
Run: python -m unittest discover -s tests"""

import sys
import unittest
from itertools import pairwise
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app import mathocr, mathregions, ocr  # noqa: E402
from app.mathocr import MathDecision  # noqa: E402
from app.ocr import Line, Word  # noqa: E402

EM, XH, BASE = 30, 14, 100
BLUE = (0x33, 0x66, 0xFF)


class Canvas:
    def __init__(self, w=1600, h=400):
        self.rgb = np.full((h, w, 3), 255, np.uint8)

    def block(self, x0, y0, x1, y1, colour=(0, 0, 0)):
        self.rgb[y0:y1, x0:x1] = colour

    def letters(self, x0, n, base=BASE, high=XH, step=14):
        """n letters of x-height (or `high`) standing on the baseline; returns the right edge."""
        for k in range(n):
            self.block(x0 + k * step, base - high, x0 + k * step + 10, base)
        return x0 + (n - 1) * step + 10

    def arrays(self):
        gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        return gray, (gray < 128).astype(np.uint8), self.rgb


def word(x0, y0, x1, y1, text, conf=95.0, **kw) -> Word:
    return Word(x0, y0, x1, y1, conf, text, 0, EM, **kw)


def detect(canvas, lines):
    gray, ink, rgb = canvas.arrays()
    return mathregions.find(lines, gray, ink, rgb, 300, method="rules")


def prose_line(canvas, words_at, base=BASE):
    """Prose words ("khí", "là", ...) of x-height letters at the given x; returns their Words."""
    out = []
    for x, text in words_at:
        right = canvas.letters(x, len(text), base)
        out.append(word(x, base - XH, right, base, text))
    return out


class ScriptsTest(unittest.TestCase):
    def page(self):
        c = Canvas()
        words = prose_line(c, [(20, "khí"), (100, "là"), (700, "và"), (800, "của")])
        # "10^5": two digits at cap height, then a small 5 raised over the caps
        c.block(200, 79, 210, 100)
        c.block(214, 79, 224, 100)
        c.block(226, 74, 234, 86)
        words.append(word(200, 74, 234, 100, "105", conf=40))
        # "n_1": a letter, then a small 1 lowered under the baseline
        c.block(300, 86, 312, 100)
        c.block(314, 96, 319, 108)
        words.append(word(300, 86, 319, 108, "n,", conf=30))
        # "x,": a comma is lower than a subscript
        c.block(400, 86, 412, 100)
        c.block(414, 95, 418, 103)
        words.append(word(400, 86, 418, 103, "x,", conf=90))
        # "°C": a degree sign stops at the caps' height
        c.block(500, 79, 508, 88)
        c.block(511, 79, 524, 100)
        words.append(word(500, 79, 524, 100, "°C", conf=93))
        # "ế"-like: an accent just over a letter is no superscript
        c.block(600, 86, 612, 100)
        c.block(603, 75, 609, 82)
        words.append(word(600, 75, 612, 100, "ế", conf=60))
        return c, [Line(1, 1, BASE, EM, words)]

    def test_superscript_and_subscript_words(self):
        c, lines = self.page()
        regions = detect(c, lines)
        self.assertEqual([(r.kind, r.ocr_text) for r in regions], [("inline", "105"), ("inline", "n,")])
        self.assertEqual(regions[0].box, (200, 74, 234, 100))  # tight on the ink with its exponent
        self.assertEqual(regions[1].box, (300, 86, 319, 108))
        self.assertIs(regions[0].line, lines[0])
        self.assertEqual(regions[0].words, [lines[0].words[4]])

    def test_exponent_split_off_joins_its_number(self):
        c, lines = self.page()
        number = next(w for w in lines[0].words if w.text == "105")
        lines[0].words.remove(number)
        lines[0].words += [word(200, 79, 224, 100, "10", conf=93), word(226, 74, 234, 86, "5", conf=50)]
        regions = detect(c, lines)
        self.assertEqual(regions[0].box, (200, 74, 234, 100))
        self.assertEqual(regions[0].ocr_text, "10 5")

    def test_low_confidence_number_with_a_mark(self):
        c, lines = self.page()
        c.block(900, 79, 910, 100)
        c.block(912, 97, 915, 103)
        c.block(918, 79, 928, 100)
        lines[0].words.append(word(900, 79, 928, 103, "8,31", conf=42))
        lines[0].words.append(word(1000, 79, 1010, 100, "7", conf=42))  # no ink: nothing to read
        c.block(1100, 79, 1110, 100)
        lines[0].words.append(word(1100, 79, 1110, 100, "8", conf=40))  # a plain integer stays OCR's
        texts = [r.ocr_text for r in detect(c, lines)]
        self.assertIn("8,31", texts)
        self.assertNotIn("8", texts)

    def test_case_of_the_first_letter_from_its_height(self):
        c = Canvas()
        words = prose_line(c, [(20, "khí"), (100, "là"), (700, "và")])
        c.block(300, 86, 312, 100)  # "v" at x-height, read "V"
        c.block(314, 90, 326, 104)  # a small capital H sitting low
        words.append(word(300, 86, 326, 104, "Vh", conf=80))
        regions = detect(c, [Line(1, 1, BASE, EM, words)])
        self.assertEqual([r.ocr_text for r in regions], ["vh"])
        self.assertEqual([r.case_from_ink for r in regions], [True])


class TwoDimensionalTest(unittest.TestCase):
    def test_fraction_in_running_text_stops_at_prose(self):
        c = Canvas()
        words = prose_line(c, [(20, "tăng"), (100, "là"), (400, "lần")])
        c.block(250, 66, 270, 84)  # numerator
        c.block(240, 89, 290, 91)  # bar
        c.block(250, 96, 270, 116)  # denominator
        c.block(215, 86, 228, 100)  # "x" before it, joined
        words += [word(215, 86, 228, 100, "x", conf=50), word(240, 66, 290, 116, "E", conf=20)]
        regions = detect(c, [Line(1, 1, BASE, EM, words)])
        self.assertEqual([(r.kind, r.box) for r in regions], [("tall", (215, 66, 290, 116))])
        self.assertEqual([w.text for w in regions[0].words], ["x", "E"])

    def test_bar_over_a_letter(self):
        c = Canvas()
        words = prose_line(c, [(20, "giá"), (100, "trị"), (400, "và")])
        c.block(250, 72, 266, 75)  # the bar
        c.block(250, 79, 266, 100)  # "A"
        words.append(word(250, 79, 266, 100, "A", conf=89))
        regions = detect(c, [Line(1, 1, BASE, EM, words)])
        self.assertEqual([(r.kind, r.box, r.ocr_text) for r in regions], [("tall", (250, 72, 266, 100), "A")])

    def test_equals_and_minus_signs_are_no_bars(self):
        c = Canvas()
        words = prose_line(c, [(20, "khí"), (100, "là"), (400, "và")])
        c.block(250, 88, 270, 90)
        c.block(250, 94, 270, 96)  # "="
        c.block(300, 92, 320, 94)  # "-"
        words += [word(250, 88, 270, 96, "="), word(300, 92, 320, 94, "-")]
        self.assertEqual(detect(c, [Line(1, 1, BASE, EM, words)]), [])

    def test_radical(self):
        c = Canvas()
        words = prose_line(c, [(20, "tăng"), (400, "lần")])
        c.block(200, 79, 210, 100)  # "2"
        x0, top, bottom = 214, 64, 104
        pts = [(x0, 86), (x0 + 6, bottom), (x0 + 16, top), (x0 + 60, top)]
        for a, b in pairwise(pts):
            cv2.line(c.rgb, a, b, (0, 0, 0), 2)
        c.block(240, 79, 252, 100)  # "3" under it
        words.append(word(200, 64, 276, 105, "2/3", conf=30))
        _, ink, rgb = c.arrays()
        page = mathregions._Page([Line(1, 1, BASE, EM, words)], ink, rgb)
        self.assertIn("root", [k for k, _, _ in mathregions._seeds(page)])
        regions = detect(c, [Line(1, 1, BASE, EM, words)])
        self.assertEqual([(r.kind, r.ocr_text) for r in regions], [("tall", "2/3")])
        self.assertEqual(regions[0].box[0], 200)  # the "2" before the sign joins

class GarbledLabelsTest(unittest.TestCase):
    def page(self):
        """Two option rows at columns 20, 400, 800, 1200: the first read well, the second with "C." glued to
        its option ("c.9") and "D." read as "p.9"."""
        c = Canvas()
        lines = []
        for row, base in enumerate((100, 200)):
            words = []
            for k, x in enumerate((20, 400, 800, 1200)):
                c.block(x, base - 21, x + 16, base, BLUE)  # letter
                c.block(x + 19, base - 4, x + 23, base, BLUE)  # dot
                c.block(x + 40, base - 21, x + 50, base)  # "9"
                c.letters(x + 60, 2, base)  # "km"
                letter = "ABCD"[k]
                if row == 1 and k >= 2:
                    words.append(word(x, base - 21, x + 50, base, ("c" if k == 2 else "p") + ".9", conf=40))
                else:
                    words += [word(x, base - 21, x + 23, base, letter + ".", color="3366FF"),
                              word(x + 40, base - 21, x + 50, base, "9", conf=60)]
                words.append(word(x + 60, base - XH, x + 84, base, "km.", conf=60))
            lines.append(Line(1, row + 1, base, EM, words))
        return c, lines

    def test_label_found_on_the_glyphs(self):
        c, lines = self.page()
        regions = [r for r in detect(c, lines) if r.line is lines[1]]
        labelled = [r for r in regions if r.label is not None]
        self.assertEqual([r.label.text for r in labelled], ["C.", "D."])
        self.assertEqual(labelled[0].label.box, (800, 179, 823, 200))
        self.assertEqual(labelled[0].label.color, "3366FF")
        self.assertEqual(labelled[0].box, (840, 179, 884, 200))  # the option, not the label
        self.assertEqual(labelled[0].ocr_text, "9 km.")  # the label's text left out

    def test_label_in_black_found_by_its_weight(self):
        c, lines = self.page()
        c.rgb[(c.rgb == BLUE).all(axis=2)] = 0
        for ln in lines:
            for w in ln.words:
                w.color = None
        thin = c.rgb.copy()
        thin[180:200, 1203:1213] = 255  # "D" of row 2 drawn with thin strokes: not a label
        c.rgb = thin
        regions = [r for r in detect(c, lines) if r.line is lines[1] and r.label is not None]
        self.assertEqual([r.label.text for r in regions], ["C."])

    def test_label_is_written_when_read(self):
        c, lines = self.page()
        gray, ink, rgb = c.arrays()

        class Reader:
            def read(self, images):
                return [("9 \\mathrm { k m } .", 0.99) for _ in images]

        with mock.patch.object(mathocr, "available", return_value=True), \
                mock.patch.object(mathocr, "get_reader", return_value=Reader()), \
                mock.patch.object(mathregions, "MATH_DETECT", "rules"):
            ocr.read_formulas(lines, [], gray, ink, rgb, 300)
        row = sorted(lines[1].words, key=lambda w: w.x0)
        texts = [w.text for w in row]
        self.assertEqual(texts.count("C."), 1)
        self.assertNotIn("c.9", texts)
        label = next(w for w in row if w.text == "C.")
        self.assertEqual((label.x0, label.color), (800, "3366FF"))
        nxt = row[row.index(label) + 1]
        self.assertIsInstance(nxt.math, MathDecision)


if __name__ == "__main__":
    unittest.main()
