"""Check the integrity of the included models and examples."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "assets/manifest.json").read_text())["files"]
    for name, info in manifest.items():
        path = root / name
        if path.stat().st_size != info["bytes"]:
            raise ValueError(f"Size mismatch: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != info["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
        print(f"Verified {name}")


if __name__ == "__main__":
    main()
