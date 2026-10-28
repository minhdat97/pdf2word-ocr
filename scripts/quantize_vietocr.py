"""Make an int8 copy of models/vietocr: the encoder's convolutions in int8, everything else as it was.

The encoder's CNN (VGG19) takes about 85 % of VietOCR's CPU time. Its convolutions are quantized
statically (QDQ: int8 weights per output channel, uint8 activations whose ranges are calibrated on
word crops of scanned pages, at the 99.999th percentile); the first two convolutions (on the image
itself) and the last 1x1 one stay float32, where int8 changed the most readings. The decoder and
config.json are copied unchanged. Measured on 30 scanned pages (onnxruntime on an ARM CPU, one
thread, KleidiAI off): VietOCR about 2.3 times faster, 2 % of the word readings different, about as
many words right as before. An int8 speed-up depends on the CPU's int8 instructions: measure it on
the server before relying on it.

    python scripts/quantize_vietocr.py models/vietocr models/vietocr-int8 scan1.pdf [scan2.pdf ...]

The PDFs are scans of Vietnamese text (a few hundred words or more, found with Tesseract as in a
conversion; at most MAX_CROPS are used). Needs onnx and onnxruntime (its quantization tools) besides
the app's requirements, and about 6 GB of memory. The copy used by the tool is published with the other
models (scripts/models.json, "vietocr-int8"); the service uses it with P2W_VIETOCR_INT8=1.
"""

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PERCENTILE = 99.999
BATCH = 16
MAX_CROPS = 600  # calibration keeps every activation in memory: ~6 GB for 500 words


def word_crops(pdfs: list[str]) -> list[np.ndarray]:
    """The word crops VietOCR would read on every page of the PDFs (app.ocr's page preparation)."""
    import cv2
    import pymupdf

    from app import ocr, vietocr

    crops = []
    for path in pdfs:
        with pymupdf.open(path) as doc:
            for page in doc:
                dpi = ocr.choose_dpi(page.rect)
                gray = cv2.cvtColor(ocr.render_rgb(page, dpi), cv2.COLOR_RGB2GRAY)
                turns = ocr.upright_turns(gray, dpi)
                if turns:
                    gray = np.ascontiguousarray(np.rot90(gray, turns))
                angle = ocr.skew_angle(gray)
                if abs(angle) >= ocr.MIN_DESKEW:
                    gray = ocr.rotate_image(gray, angle)
                _, rule_mask = ocr.find_tables(gray)
                gray[rule_mask > 0] = 255
                ocr.drop_light_ink(gray)
                lines = ocr.tesseract_lines(gray, "vie", dpi)
                crops += [c.copy() for c in vietocr.word_crops(lines, gray) if c.shape[0] >= 2 and c.shape[1] >= 2]
    return crops


def fold_fp16(model):
    """float16 weights followed by a Cast to float32 (export_vietocr.finish) -> float32 weights: the quantizer only
    sees a Conv's weights when they are an initializer. Same arithmetic."""
    from onnx import TensorProto, numpy_helper

    g = model.graph
    inits = {t.name: t for t in g.initializer}
    nodes, folded, new = [], set(), []
    for n in g.node:
        to = next((a.i for a in n.attribute if a.name == "to"), None)
        src = inits.get(n.input[0]) if n.op_type == "Cast" and n.input else None
        if src is not None and src.data_type == TensorProto.FLOAT16 and to == TensorProto.FLOAT:
            new.append(numpy_helper.from_array(numpy_helper.to_array(src).astype(np.float32), n.output[0]))
            folded.add(src.name)
        else:
            nodes.append(n)
    used = {i for n in nodes for i in n.input}
    keep = [t for t in g.initializer if t.name not in folded or t.name in used]
    del g.initializer[:]
    g.initializer.extend(keep + new)
    del g.node[:]
    g.node.extend(nodes)
    del g.value_info[:]
    return model


def float_convs(model) -> list[str]:
    """Convolutions kept float32: the first two of the graph and the 1x1 ones."""
    inits = {t.name: t for t in model.graph.initializer}
    convs = [n for n in model.graph.node if n.op_type == "Conv"]
    ones = [n for n in convs if n.input[1] in inits and list(inits[n.input[1]].dims[2:]) == [1, 1]]
    return list(dict.fromkeys(n.name for n in convs[:2] + ones))


def batches(crops: list[np.ndarray], src: Path) -> list[np.ndarray]:
    """Calibration inputs: the crops as VietOCR prepares them, BATCH of one width at a time."""
    from app.vietocr import VietOCR

    import json

    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    prep = VietOCR.__new__(VietOCR)
    prep.height, prep.min_width, prep.max_width = cfg["image_height"], cfg["image_min_width"], cfg["image_max_width"]
    by_width = defaultdict(list)
    for c in crops:
        x = prep.prepare(c)
        by_width[x.shape[-1]].append(x)
    return [np.stack(v[i : i + BATCH]) for v in by_width.values() for i in range(0, len(v), BATCH)]


def quantize(src: Path, dst: Path, inputs: list[np.ndarray]) -> None:
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization import calibrate

    # The histogram collector stacks each tensor of every batch; batches here have different widths: flatten them.
    collect = calibrate.HistogramCollector.collect

    def flat_collect(self, name_to_arr):
        return collect(self, {k: [np.concatenate([np.asarray(x).ravel() for x in v])] for k, v in name_to_arr.items()})

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(inputs)

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"image": x}

    dst.mkdir(parents=True, exist_ok=True)
    model = fold_fp16(onnx.load(src / "encoder.onnx"))
    tmp = dst / "encoder.float32.onnx"
    onnx.save(model, tmp)
    calibrate.HistogramCollector.collect = flat_collect
    try:
        quantize_static(
            str(tmp), str(dst / "encoder.onnx"), Reader(),
            quant_format=QuantFormat.QDQ, op_types_to_quantize=["Conv"], nodes_to_exclude=float_convs(model),
            per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.Percentile, extra_options={"CalibPercentile": PERCENTILE},
        )
    finally:
        calibrate.HistogramCollector.collect = collect
        tmp.unlink(missing_ok=True)
    for name in ("decoder.onnx", "config.json"):
        shutil.copyfile(src / name, dst / name)


def check(src: Path, dst: Path, crops: list[np.ndarray]) -> None:
    from app.vietocr import VietOCR

    sample = crops[:: max(1, len(crops) // 300)]
    a, b = VietOCR(src, 1).read(sample), VietOCR(dst, 1).read(sample)
    same = sum(x[0] == y[0] for x, y in zip(a, b))
    print(f"{same}/{len(sample)} sample words read the same as with {src}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path, help="models/vietocr")
    ap.add_argument("dst", type=Path, help="new folder, e.g. models/vietocr-int8")
    ap.add_argument("pdfs", nargs="+", help="scanned pages for the calibration")
    args = ap.parse_args()
    if args.dst.resolve() == args.src.resolve():
        sys.exit("dst must be another folder")
    crops = word_crops(args.pdfs)
    if len(crops) < 200:
        sys.exit(f"only {len(crops)} words in the PDFs: give more scanned pages")
    if len(crops) > MAX_CROPS:
        crops = [crops[i * len(crops) // MAX_CROPS] for i in range(MAX_CROPS)]
    print(f"{len(crops)} word crops")
    quantize(args.src, args.dst, batches(crops, args.src))
    check(args.src, args.dst, crops)


if __name__ == "__main__":
    main()
