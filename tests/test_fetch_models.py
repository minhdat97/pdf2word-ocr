"""scripts/fetch_models.py: copy from a folder, SHA-256 check, nothing written on a mismatch."""

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fetch_models.py"
spec = importlib.util.spec_from_file_location("fetch_models", SCRIPT)
fetch_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_models)


class FetchModelsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.src, self.dest = root / "src", root / "models"
        self.src.mkdir()
        data = b"weights" * 1000
        (self.src / "a.onnx").write_bytes(data)
        self.entry = {"name": "a.onnx", "url": "", "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        self.manifest = root / "models.json"

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, entry, *args):
        self.manifest.write_text(json.dumps({"m": {"files": [entry]}}))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = fetch_models.main(["--manifest", str(self.manifest), "--dest", str(self.dest), *args])
        return code, out.getvalue() + err.getvalue()

    def test_copy_check_and_keep(self):
        self.assertEqual(self.run_script(self.entry, "--check")[0], 1)
        code, out = self.run_script(self.entry, "--from", str(self.src))
        self.assertEqual(code, 0, out)
        self.assertEqual((self.dest / "m" / "a.onnx").read_bytes(), (self.src / "a.onnx").read_bytes())
        self.assertEqual(self.run_script(self.entry, "--check"), (0, "ok       m/a.onnx\n"))
        self.assertEqual(list((self.dest / "m").iterdir()), [self.dest / "m" / "a.onnx"])  # no temporary files

    def test_wrong_hash_writes_nothing(self):
        code, out = self.run_script({**self.entry, "sha256": "0" * 64}, "--from", str(self.src))
        self.assertEqual(code, 1)
        self.assertIn("does not match", out)
        self.assertEqual(list((self.dest / "m").iterdir()), [])

    def test_model_folders_and_model_option(self):
        sub = self.src / "m"
        sub.mkdir()
        other = b"other" * 100
        (sub / "a.onnx").write_bytes((self.src / "a.onnx").read_bytes())
        (self.src / "a.onnx").write_bytes(b"stale")  # DIR/<model>/ wins over DIR
        entry_n = {"name": "b.bin", "url": "", "sha256": hashlib.sha256(other).hexdigest(), "size": len(other)}
        self.manifest.write_text(json.dumps({"m": {"files": [self.entry]}, "n": {"files": [entry_n]}}))
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = fetch_models.main(["--manifest", str(self.manifest), "--dest", str(self.dest), "--from", str(self.src),
                                      "--model", "m"])
            bad = fetch_models.main(["--manifest", str(self.manifest), "--dest", str(self.dest), "--model", "x"])
        self.assertEqual(code, 0, out.getvalue())
        self.assertEqual(bad, 2)
        self.assertTrue((self.dest / "m" / "a.onnx").is_file())
        self.assertFalse((self.dest / "n").exists())

    def test_repo_manifest(self):
        manifest = json.loads(fetch_models.MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(set(manifest), {"vietocr", "mfr"})
        names = {e["name"] for e in manifest["mfr"]["files"]}
        self.assertEqual(names, {"encoder_model.onnx", "decoder_model.onnx", "tokenizer.json"})
        for model, spec in manifest.items():
            for e in spec["files"]:
                self.assertRegex(e["sha256"], r"^[0-9a-f]{64}$")
                self.assertGreater(e["size"], 0)
                if model == "mfr":  # pinned to one revision, never a moving branch
                    self.assertRegex(e["url"], r"^https://huggingface\.co/[\w.-]+/pix2text-mfr-1\.5/resolve/[0-9a-f]{40}/")

    def test_no_url_yet(self):
        code, out = self.run_script(self.entry)
        self.assertEqual(code, 1)
        self.assertIn("--from", out)


if __name__ == "__main__":
    unittest.main()
