"""Reject leakage before creating classification crops."""
import hashlib
from pathlib import Path


def validate_splits(rows, image_root):
    assignments = {}
    root = Path(image_root).resolve()
    hashes = {}
    for row in rows:
        split = row["split"].strip()
        if split not in {"train", "val", "test"}:
            raise ValueError("Split must be train, val or test")
        for field in ("panorama_id", "group_id", "capture_date", "route"):
            if not row.get(field, "").strip():
                raise ValueError(f"Missing split provenance: {field}")
        source = (root / row["image"]).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"Invalid image: {source}")
        if source not in hashes:
            hashes[source] = hashlib.sha256(source.read_bytes()).hexdigest()
        keys = [(field, row[field].strip()) for field in ("panorama_id", "group_id", "capture_date", "route")]
        keys += [("image_hash", hashes[source])]
        for key in keys:
            previous = assignments.setdefault(key, split)
            if previous != split:
                raise ValueError(f"Split leakage: {key} occurs in {previous} and {split}")
    return {str(path.relative_to(root)): digest for path, digest in hashes.items()}
