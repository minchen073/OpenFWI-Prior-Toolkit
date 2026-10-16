#!/usr/bin/env python3
"""Download and verify the four OpenFWI examples and pretrained EDM models."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

import hashlib


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "assets/manifest.json").read_text())["files"]
    release = json.loads((root / "assets/release.json").read_text())["files"]
    destination = args.destination.resolve()
    pending = []
    for name, info in manifest.items():
        path = destination / name
        if path.is_symlink():
            raise ValueError(f"Refusing symlink: {path}")
        if path.exists():
            if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
                raise FileExistsError(
                    f"Existing asset has a different checksum: {path}"
                )
        else:
            pending.append(name)
    if not pending:
        print("All eight assets are present and verified.")
        return
    with tempfile.TemporaryDirectory(prefix="openfwi-prior-assets-") as temporary:
        temporary = Path(temporary)

        def download(name):
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {Path(name).name}", flush=True)
            with urllib.request.urlopen(release[name]["url"], timeout=60) as response:
                with path.open("wb") as output:
                    shutil.copyfileobj(response, output)
            info = manifest[name]
            if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
                raise ValueError(f"Asset checksum or size mismatch: {name}")
            return path

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(download, pending))
        for name in pending:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(temporary / name, path)
        print(f'Installed and verified all eight assets in {destination / "assets"}')


if __name__ == "__main__":
    main()
