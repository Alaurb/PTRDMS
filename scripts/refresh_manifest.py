"""Refresh selected SHA-256 records in the curated dataset manifest.

This is intentionally explicit: callers must name every generated artifact that
has changed, so a model or source panorama cannot silently be re-baselined.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "metadata" / "dataset_manifest.json"


def record_for(relative_path: str) -> dict[str, object]:
    path = (ROOT / relative_path).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Not a repository file: {relative_path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main(paths: list[str]) -> None:
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))
    refreshed = {record["path"]: record_for(record["path"]) for record in (record_for(item) for item in paths)}
    output = [refreshed.pop(record["path"], record) for record in records]
    output.extend(refreshed[key] for key in sorted(refreshed))
    MANIFEST.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Refreshed {len(paths)} manifest records; total {len(output)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", action="append", required=True, help="Repository-relative file to record")
    main(parser.parse_args().path)
