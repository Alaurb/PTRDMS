"""Verify the curated inputs and saved outputs without loading model weights."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    records = json.loads((ROOT / "metadata/dataset_manifest.json").read_text(encoding="utf-8"))
    for record in records:
        path = (ROOT / record["path"]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Invalid or missing dataset path: {record['path']}")
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Size mismatch: {record['path']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {record['path']}")
    print(f"Dataset integrity OK: {len(records)} files")

if __name__ == "__main__":
    main()
