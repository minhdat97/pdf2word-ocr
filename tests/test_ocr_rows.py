"""app.ocr.join_rows: lines Tesseract split off a printed row are joined back; columns, headers and footers,
stacked fractions stay apart. Page 3000 px high, 1em = 45 px. Run: python -m unittest discover -s tests"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ocr import Line, Word, join_rows  # noqa: E402

EM, HEIGHT = 45, 3000


def line(block, base, *words, par=None, top=None):
    """A line of words (x0, x1, text) standing on `base`, glyphs 1em high (or from `top`)."""
    ln = Line(block, par or block, base, EM)
    y0 = base - EM if top is None else top
    ln.words = [Word(x0, y0, x1, base, 90.0, text, 0, EM) for x0, x1, text in words]
    return ln


def texts(lines):
    return [" ".join(w.text for w in ln.words) for ln in lines]


class JoinRowsTest(unittest.TestCase):
    def test_label_split_off_its_row(self):
        stem = line(1, 2300, (250, 1900, "Trộn hai loại khí"))
        row = line(1, 2420, (730, 760, "B."), (1210, 1240, "C."), (1690, 1720, "D."), par=2)
        label = line(4, 2420, (275, 315, "A."))
        after = line(5, 2600, (250, 1200, "Câu 20:"))
        lines = [stem, row, label, after]
        join_rows(lines, HEIGHT)
        self.assertEqual(texts(lines), ["Trộn hai loại khí", "A. B. C. D.", "Câu 20:"])
        self.assertIs(lines[1], row)

    def test_columns_stay_apart(self):
        left = [line(1, 400 + 60 * k, (100, 900, f"trái {k}")) for k in range(5)]
        right = [line(2, 400 + 60 * k, (1100, 1900, f"phải {k}")) for k in range(5)]
        lines = left + right
        join_rows(lines, HEIGHT)
        self.assertEqual(len(lines), 10)
        # a heading alone in its block beside a column's line: the column does not span it
        lines = left + [line(3, 520, (1100, 1500, "PHẦN II."))]
        join_rows(lines, HEIGHT)
        self.assertEqual(len(lines), 6)

    def test_footer_band_baseline_and_height(self):
        page = line(1, 1000, (100, 2000, "Câu 1: nội dung"))
        footer = [line(2, 2950, (100, 1500, "Trang")), line(3, 2950, (1600, 1700, "1/2"))]
        lines = [page] + footer
        join_rows(lines, HEIGHT)
        self.assertEqual(len(lines), 3)
        row = line(1, 1500, (230, 1940, "A. B. C. D."))
        numerator = line(2, 1490, (1955, 2040, "4"))  # baseline 10 px higher: part of a fraction
        lines = [row, numerator]
        join_rows(lines, HEIGHT)
        self.assertEqual(len(lines), 2)
        stack = line(3, 1500, (100, 150, "a b c d"), top=1300)  # labels of four lines read as one
        lines = [row, stack]
        join_rows(lines, HEIGHT)
        self.assertEqual(len(lines), 2)

    def test_overlapping_lines_stay(self):
        lines = [line(1, 1000, (100, 900, "một")), line(2, 1000, (850, 1200, "hai"))]
        join_rows(lines, HEIGHT)
        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
