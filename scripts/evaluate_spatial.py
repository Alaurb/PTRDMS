"""Evaluate manually matched observations; not an object-detection recall metric."""
import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path


def evaluate(predictions, labels):
    def key(row):
        return row["frame"], row["side"], int(row["observation_index"])
    lookup = {key(row): row for row in predictions}
    if len(lookup) != len(predictions):
        raise ValueError("Duplicate prediction keys")
    seen, paired = set(), []
    for label in labels:
        k = key(label)
        if k in seen:
            raise ValueError("Duplicate ground-truth keys")
        seen.add(k)
        if not label.get("target_id") or not label.get("class_name"):
            raise ValueError("Ground truth requires manual target_id and class_name")
        if k not in lookup:
            raise ValueError(f"Label does not match an observation: {k}")
        xyz = [float(label[axis]) for axis in ("x", "y", "z")]
        pred_xyz = [float(lookup[k][axis]) for axis in ("x", "y", "z")]
        if not all(math.isfinite(v) for v in xyz + pred_xyz):
            raise ValueError("Non-finite coordinate")
        paired.append((lookup[k], label, [a-b for a,b in zip(pred_xyz, xyz)]))
    if not paired:
        raise ValueError("At least one independently annotated observation is required")
    tp = fp = fn = 0
    for (p1,g1,_), (p2,g2,_) in combinations(paired, 2):
        predicted = p1["track_id"] == p2["track_id"]
        actual = g1["target_id"] == g2["target_id"]
        tp += predicted and actual
        fp += predicted and not actual
        fn += actual and not predicted
    return dict(evaluation_scope="manually_matched_observations_only",
                observations=len(predictions), labeled_observations=len(paired),
                annotation_coverage=len(paired)/len(predictions),
                localization_rmse_m=math.sqrt(sum(sum(v*v for v in error) for _,_,error in paired)/len(paired)),
                height_mae_m=sum(abs(error[2]) for _,_,error in paired)/len(paired),
                classification_accuracy=sum(p["class_name"]==g["class_name"] for p,g,_ in paired)/len(paired),
                association_pair_precision=tp/(tp+fp) if tp+fp else None,
                association_pair_recall=tp/(tp+fn) if tp+fn else None,
                pair_tp=tp, pair_fp=fp, pair_fn=fn,
                limitations="Not detection precision/recall; manual correspondence and independent survey required. Null means undefined.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="association_links.json")
    parser.add_argument("--ground-truth", required=True, help="CSV: frame,side,observation_index,target_id,class_name,x,y,z")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    with Path(args.ground_truth).open(encoding="utf-8-sig", newline="") as stream:
        report = evaluate(predictions, list(csv.DictReader(stream)))
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
