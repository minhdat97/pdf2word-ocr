"""Fetch the model files listed in scripts/models.json into models/ (not in git).

    python scripts/fetch_models.py                 # download from each file's URL
    python scripts/fetch_models.py --from DIR      # copy from a local folder instead
    python scripts/fetch_models.py --check         # only verify what is there
    python scripts/fetch_models.py --model mfr     # only this model (repeatable)

--from DIR takes a model's files from DIR/<model>/ when that folder has them, else from DIR.

Each file is checked against its SHA-256 and size; files already present and
correct are kept. A model's new files are put in place only once all of them
are there and correct (fetched into models/.<model>.new/ first): a download
that fails half way leaves the model's old files as they were, never a mix of
old and new files (new config.json next to old weights) that loads but cannot
read.
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


def fetch(entry: dict, dest: Path, source: Path | None) -> None:
    """Copy or download one file to dest, verified, replacing dest only when correct.
    source: the folder holding the file (copied instead of downloaded)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            if source is not None:
                with open(source / entry["name"], "rb") as src:
                    shutil.copyfileobj(src, out, CHUNK)
            else:
                if not entry.get("url", "").startswith("https://"):
                    raise RuntimeError("no download URL in scripts/models.json yet: use --from DIR")
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
        source = args.source
        if source is not None and (source / model / spec["files"][0]["name"]).is_file():
            source = source / model
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
        for entry in todo:
            try:
                fetch(entry, stage / entry["name"], source)
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
            print(f"fetched  {model}/{entry['name']} ({entry['size'] / 1e6:.1f} MB)")
        shutil.rmtree(stage, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
