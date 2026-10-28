"""Fetch the model files listed in scripts/models.json into models/ (not in git).

    python scripts/fetch_models.py                 # download from each file's URL
    python scripts/fetch_models.py --from DIR      # copy from a local folder instead
    python scripts/fetch_models.py --check         # only verify what is there
    python scripts/fetch_models.py --model mfr     # only this model (repeatable)

--from DIR takes each file from DIR/<model>/<name>, DIR/<name of the download> (the
release's file names, e.g. vietocr-encoder.onnx: a folder of downloaded release files
works) or DIR/<name>, the first one of the right size.

Each file is checked against its SHA-256 and size; files already present and
correct are kept, and a file another model already has (the same SHA-256: the
int8 VietOCR shares its decoder with the float16 one) is copied, not downloaded.
A model's new files are put in place only once all of them are there and correct
(fetched into models/.<model>.new/ first): a download that fails half way leaves
the model's old files as they were, never a mix of old and new files (new
config.json next to old weights) that loads but cannot read. Exit status 1 when
a file is missing or wrong (--check) or could not be fetched.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "scripts" / "models.json"
CHUNK = 1 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def is_ok(path: Path, entry: dict) -> bool:
    return path.is_file() and path.stat().st_size == entry["size"] and sha256(path) == entry["sha256"]


def download_name(entry: dict) -> str | None:
    """The file name of the entry's URL (the release asset), if it has one."""
    name = entry.get("url", "").rsplit("/", 1)[-1]
    return name or None


def local_file(entry: dict, model: str, folder: Path) -> Path:
    """The file to copy from --from DIR: DIR/<model>/<name>, DIR/<download name> or DIR/<name>, the first of the
    right size (else the first that exists: its check fails with a clear message)."""
    names = [folder / model / entry["name"]]
    if download_name(entry):
        names.append(folder / download_name(entry))
    names.append(folder / entry["name"])
    there = [p for p in names if p.is_file()]
    return next((p for p in there if p.stat().st_size == entry["size"]), there[0] if there else names[0])


def same_file(entry: dict, manifest: dict, dest: Path) -> Path | None:
    """A correct file with this entry's SHA-256 already in another model's folder."""
    for model, spec in manifest.items():
        for other in spec["files"]:
            path = dest / model / other["name"]
            if other["sha256"] == entry["sha256"] and is_ok(path, entry):
                return path
    return None


def downloadable(url: str) -> bool:
    # https only; plain http just for a local test server
    return url.startswith(("https://", "http://127.0.0.1:", "http://localhost:"))


def fetch(entry: dict, dest: Path, source: Path | None) -> None:
    """Copy or download one file to dest, verified, replacing dest only when correct.
    source: the file to copy instead of downloading."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            if source is not None:
                with open(source, "rb") as src:
                    shutil.copyfileobj(src, out, CHUNK)
            else:
                if not downloadable(entry.get("url", "")):
                    raise RuntimeError("no download URL in scripts/models.json: use --from DIR")
                with urllib.request.urlopen(entry["url"], timeout=60) as resp:
                    shutil.copyfileobj(resp, out, CHUNK)
        if not is_ok(Path(tmp), entry):
            raise RuntimeError("SHA-256 or size does not match scripts/models.json")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="source", type=Path, help="local folder holding the files")
    ap.add_argument("--dest", type=Path, default=ROOT / "models", help="models folder (default: models/)")
    ap.add_argument("--check", action="store_true", help="verify only, change nothing")
    ap.add_argument("--model", action="append", help="only this model of the manifest (repeatable)")
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    unknown = sorted(set(args.model or []) - set(manifest))
    if unknown:
        print(f"unknown model: {', '.join(unknown)} (manifest has {', '.join(manifest)})", file=sys.stderr)
        return 2
    failed = 0
    for model, spec in manifest.items():
        if args.model and model not in args.model:
            continue
        todo = []
        for entry in spec["files"]:
            dest = args.dest / model / entry["name"]
            label = f"{model}/{entry['name']}"
            if is_ok(dest, entry):
                print(f"ok       {label}")
            elif args.check:
                print(f"MISSING  {label}" if not dest.exists() else f"BAD      {label}")
                failed += 1
            else:
                todo.append(entry)
        if not todo:
            continue
        stage = args.dest / f".{model}.new"
        shutil.rmtree(stage, ignore_errors=True)
        bad = 0
        sources = {}
        for entry in todo:
            if args.source is not None:
                src = local_file(entry, model, args.source)
            else:
                src = same_file(entry, manifest, args.dest)
            sources[entry["name"]] = src
            try:
                fetch(entry, stage / entry["name"], src)
            except (OSError, RuntimeError, ValueError) as exc:  # report, go on with the other files
                print(f"FAILED   {model}/{entry['name']}: {exc}", file=sys.stderr)
                bad += 1
        if bad:
            shutil.rmtree(stage, ignore_errors=True)
            print(f"KEPT     {model}: none of its files changed (fix the failures above and run again)", file=sys.stderr)
            failed += bad
            continue
        (args.dest / model).mkdir(parents=True, exist_ok=True)
        for entry in todo:
            os.replace(stage / entry["name"], args.dest / model / entry["name"])
            src = sources[entry["name"]]
            how = f"from {src}" if args.source is not None else f"copy of {src.parent.name}/{src.name}" if src else "downloaded"
            print(f"fetched  {model}/{entry['name']} ({entry['size'] / 1e6:.1f} MB, {how})")
        shutil.rmtree(stage, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
