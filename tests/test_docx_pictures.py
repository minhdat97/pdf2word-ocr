"""Floating pictures in the .docx: where Word puts them. Run: python -m unittest discover -s tests"""

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from lxml import etree  # noqa: E402

from app.docx_builder import build_docx  # noqa: E402
from app.model import PageLayout, Paragraph, Picture  # noqa: E402

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
EMU = 12700


def png(w=40, h=10):
    return cv2.imencode(".png", np.full((h, w, 3), 128, np.uint8))[1].tobytes()


def para(text, y0, y1=None, space_before=0.0):
    p = Paragraph(text, 36, y0, 559, y1 or y0 + 14, 12)
    p.space_before_pt = space_before
    return p


class BannerTest(unittest.TestCase):
    """A page opening with two banners across the column (a title box, a box of aims) above its first question:
    Word measures a picture anchored to a paragraph from the top of the paragraph's space before, and puts a line
    below a picture across the column by itself."""

    def test_banners_above_the_first_paragraph(self):
        page1 = PageLayout(595, 842, elements=[para("Cau 1.", 274), para("A. B. C. D.", 300, space_before=12)],
                           pictures=[Picture(png(), 36, 57, 560, 193), Picture(png(), 36, 177, 559, 267)])
        page2 = PageLayout(595, 842, elements=[para("Cau 2.", 57)])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.docx"
            build_docx([page1, page2], out)
            root = etree.fromstring(zipfile.ZipFile(out).read("word/document.xml"))
        first = next(p for p in root.iter(W + "p") if p.find(f".//{WP}anchor") is not None)
        before = int(first.find(f".//{W}spacing").get(W + "before"))
        self.assertEqual(before, 7 * 20)  # below the lower banner (267), not 274 - 57 on top of it
        offsets = sorted(int(a.find(f"{WP}positionV/{WP}posOffset").text) / EMU for a in first.iter(WP + "anchor"))
        self.assertEqual([round(v) for v in offsets], [0, 120])  # from the top margin (57): at 57 and 177 as in the PDF


if __name__ == "__main__":
    unittest.main()
