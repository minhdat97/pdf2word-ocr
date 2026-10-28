"""Split the formula model's decoder_model.onnx in two, for app/mathocr.py.

pix2text-mfr-1.5's decoder_model.onnx has no cache: every decoding step projects
the encoder's output again into the cross-attention keys and values of its six
layers (12 matrix products over the 578 image tokens), about three quarters of
the decoder's time. This writes, next to it:

- cross_kv.onnx: encoder_hidden_states -> those 12 tensors (kv0 ... kv11), run
  once per crop;
- decoder_step.onnx: input_ids + kv0 ... kv11 -> logits, run at each step.

Same weights and the same operations as decoder_model.onnx, cut where the
encoder's part ends: the logits agree to float rounding (checked here with
onnxruntime on random inputs). FormulaReader uses the two files when both are in
the model folder. Not needed to run the tool; needs onnx and onnxruntime:

    python scripts/split_mfr_decoder.py models/mfr [OUT_DIR]
"""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import utils

OUTPUTS = ("cross_kv.onnx", "decoder_step.onnx")


def frontier(graph) -> list[str]:
    """Tensors computed from encoder_hidden_states alone that the steps (input_ids) use."""

    def closure(name):
        dep = {name}
        for node in graph.node:  # nodes are in topological order
            if any(i in dep for i in node.input):
                dep.update(node.output)
        return dep

    enc, ids = closure("encoder_hidden_states"), closure("input_ids")
    only_enc = set()
    for node in graph.node:
        ins = [i for i in node.input if i]
        if any(i in enc for i in ins) and not any(i in ids for i in ins):
            only_enc.update(node.output)
    cut = []
    for node in graph.node:
        ins = [i for i in node.input if i]
        if any(i in ids for i in ins):
            if "encoder_hidden_states" in ins:
                raise SystemExit(f"{node.name} reads encoder_hidden_states with input_ids: unexpected decoder")
            cut += [i for i in ins if i in only_enc and i not in cut]
    return cut


def rename(path: Path, names: dict[str, str]) -> None:
    model = onnx.load(str(path))
    for node in model.graph.node:
        node.input[:] = [names.get(i, i) for i in node.input]
        node.output[:] = [names.get(o, o) for o in node.output]
    for value in list(model.graph.input) + list(model.graph.output):
        value.name = names.get(value.name, value.name)
    del model.metadata_props[:]
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def check(src: Path, out: Path) -> float:
    import onnxruntime as ort

    whole, cross, step = (ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
                          for p in (src / "decoder_model.onnx", out / OUTPUTS[0], out / OUTPUTS[1]))
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((2, 578, 384)).astype(np.float32)
    ids = np.array([[1, 40, 7, 300], [1, 12, 99, 5]], np.int64)
    want = whole.run(None, {"input_ids": ids, "encoder_hidden_states": hidden})[0]
    kv = dict(zip([o.name for o in cross.get_outputs()], cross.run(None, {"encoder_hidden_states": hidden})))
    got = step.run(None, {"input_ids": ids, **kv})[0]
    return float(np.abs(want - got).max())


def main(argv: list[str]) -> int:
    if not argv or len(argv) > 2:
        print(__doc__)
        return 2
    src = Path(argv[0])
    out = Path(argv[1]) if len(argv) > 1 else src
    out.mkdir(parents=True, exist_ok=True)
    path = src / "decoder_model.onnx"
    cut = frontier(onnx.load(str(path)).graph)
    if len(cut) != 12:
        raise SystemExit(f"expected 12 cross-attention tensors, found {len(cut)}")
    utils.extract_model(str(path), str(out / OUTPUTS[0]), ["encoder_hidden_states"], cut)
    utils.extract_model(str(path), str(out / OUTPUTS[1]), ["input_ids"] + cut, ["logits"])
    names = {t: f"kv{k}" for k, t in enumerate(cut)}
    for name in OUTPUTS:
        rename(out / name, names)
    diff = check(src, out)
    print(f"{out / OUTPUTS[0]}, {out / OUTPUTS[1]}: largest logit difference {diff:.2e}")
    return 0 if diff < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
