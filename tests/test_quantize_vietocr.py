"""scripts/quantize_vietocr.py: float16 weights folded to float32, which convolutions stay float32.
Skipped without the onnx package (only needed to make the int8 model).

    python -m unittest discover -s tests
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

HAVE_ONNX = importlib.util.find_spec("onnx") is not None
ROOT = Path(__file__).resolve().parent.parent


def load_script():
    spec = importlib.util.spec_from_file_location("quantize_vietocr", ROOT / "scripts" / "quantize_vietocr.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["quantize_vietocr"] = mod
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(HAVE_ONNX, "onnx not installed")
class QuantizeScriptTest(unittest.TestCase):
    def model(self):
        """x -> conv a (3x3) -> conv b (3x3) -> conv c (3x3) -> conv d (1x1), weights stored as float16 + Cast."""
        from onnx import TensorProto, helper, numpy_helper

        nodes, inits = [], []
        prev = "x"
        for name, k in (("a", 3), ("b", 3), ("c", 3), ("d", 1)):
            w = np.random.default_rng(0).standard_normal((2, 2, k, k)).astype(np.float16)
            inits.append(numpy_helper.from_array(w, f"w_{name}__fp16"))
            nodes.append(helper.make_node("Cast", [f"w_{name}__fp16"], [f"w_{name}"], to=TensorProto.FLOAT))
            nodes.append(helper.make_node("Conv", [prev, f"w_{name}"], [f"y_{name}"], name=f"conv_{name}",
                                          pads=[k // 2] * 4))
            prev = f"y_{name}"
        graph = helper.make_graph(
            nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 8, 8])],
            [helper.make_tensor_value_info(prev, TensorProto.FLOAT, [1, 2, 8, 8])], inits,
        )
        return helper.make_model(graph)

    def test_fold_keeps_the_weights_as_float32(self):
        from onnx import TensorProto, numpy_helper

        q = load_script()
        m = q.fold_fp16(self.model())
        self.assertEqual([n.op_type for n in m.graph.node], ["Conv"] * 4)
        weights = {t.name: t for t in m.graph.initializer}
        self.assertEqual(sorted(weights), ["w_a", "w_b", "w_c", "w_d"])
        self.assertTrue(all(t.data_type == TensorProto.FLOAT for t in weights.values()))
        w = np.random.default_rng(0).standard_normal((2, 2, 3, 3)).astype(np.float16).astype(np.float32)
        self.assertTrue((numpy_helper.to_array(weights["w_a"]) == w).all())

    def test_first_two_and_1x1_convolutions_stay_float32(self):
        q = load_script()
        self.assertEqual(q.float_convs(q.fold_fp16(self.model())), ["conv_a", "conv_b", "conv_d"])


if __name__ == "__main__":
    unittest.main()
