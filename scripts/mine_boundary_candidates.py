"""Mine uncertain adjacent-stage crops for manual maturity-boundary review."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


CLASSES = ("immature-period", "green-maturity-period", "discoloration-period", "maturity-period")
PAIRS = {"imm_vs_gm": (0, 1), "gm_vs_dis": (1, 2)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--review-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path, action="append")
    parser.add_argument("--per-pair", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


def existing_targets(review_root: Path) -> set[Path]:
    return {path.resolve() for path in review_root.rglob("*") if path.is_symlink()}


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if tuple(checkpoint.get("classes", ())) != CLASSES:
        raise ValueError("Checkpoint classes do not match the paper four-stage schema")
    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    review_root = args.review_root.resolve()
    targets = existing_targets(review_root)
    candidates = []
    for root in args.candidate_root:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        candidates.extend(path for path in sorted(root.glob("*.png")) if path.resolve() not in targets)
    unique = list(dict.fromkeys(path.resolve() for path in candidates))
    if not unique:
        raise ValueError("No unreviewed candidate crops found")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(checkpoint["state_dict"]); model.to(device).eval()
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
                                    transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
    scored = {name: [] for name in PAIRS}
    with torch.no_grad():
        for path in unique:
            with Image.open(path) as image:
                probabilities = torch.softmax(model(transform(image.convert("RGB")).unsqueeze(0).to(device)), dim=1)[0].cpu().tolist()
            for name, (first, second) in PAIRS.items():
                pair_mass = probabilities[first] + probabilities[second]
                margin = abs(probabilities[first] - probabilities[second])
                # High probability that the fruit belongs to this adjacent pair,
                # coupled with a small pair margin, is most valuable for review.
                scored[name].append((pair_mass - margin, pair_mass, margin, path, probabilities))
    rows = []
    for name, values in scored.items():
        folder = review_root / "boundary_review" / name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        values.sort(key=lambda value: (-value[0], -value[1], value[2], str(value[3])))
        selected = values[:args.per_pair]
        if len(selected) < args.per_pair:
            raise ValueError(f"Only {len(selected)} candidates available for {name}")
        for number, (score, mass, margin, source, probabilities) in enumerate(selected, 1):
            target = folder / f"mined_{name}_{number:03d}__{source.name}"
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            rows.append({"pair": name, "rank": number, "source_crop": str(source), "review_link": str(target),
                         "pair_score": score, "pair_probability": mass, "pair_margin": margin,
                         "probabilities": json.dumps(dict(zip(CLASSES, probabilities)))} )
    # All selection and target-name validation occurs before changing review folders.
    for row in rows:
        Path(row["review_link"]).symlink_to(row["source_crop"])
    with args.manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"unreviewed_scored": len(unique), "selected": len(rows), "manifest": str(args.manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
