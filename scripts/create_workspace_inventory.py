#!/usr/bin/env python3
"""Create a lightweight top-level inventory without hashing or copying data."""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from pathlib import Path


EXCLUDED = {".git"}


def summarize(path: Path) -> tuple[int, int, float, int]:
    if path.is_file():
        stat = path.stat()
        return 1, stat.st_size, stat.st_mtime, 0
    file_count = total_bytes = errors = 0
    latest = 0.0
    for directory, dirnames, filenames in os.walk(path, onerror=lambda _: None):
        dirnames[:] = [name for name in dirnames if name not in EXCLUDED]
        for filename in filenames:
            candidate = Path(directory) / filename
            try:
                stat = candidate.stat()
            except OSError:
                errors += 1
                continue
            file_count += 1
            total_bytes += stat.st_size
            latest = max(latest, stat.st_mtime)
    return file_count, total_bytes, latest, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("metadata/asset_inventory.csv"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    if output.parent != root and root not in output.parents:
        raise SystemExit("--output must be within --root")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in sorted(root.iterdir(), key=lambda candidate: candidate.name.casefold()):
        if item.name in EXCLUDED or item == output:
            continue
        try:
            files, byte_count, latest, errors = summarize(item)
        except OSError:
            files, byte_count, latest, errors = 0, 0, 0.0, 1
        rows.append({
            "relative_path": item.name,
            "kind": "directory" if item.is_dir() else "file",
            "file_count": files,
            "bytes": byte_count,
            "latest_mtime_utc": datetime.fromtimestamp(latest, timezone.utc).isoformat() if latest else "",
            "scan_errors": errors,
        })
    fields = ["relative_path", "kind", "file_count", "bytes", "latest_mtime_utc", "scan_errors"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} top-level records to {output}")


if __name__ == "__main__":
    main()
