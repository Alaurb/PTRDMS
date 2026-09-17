"""Evaluate final pipeline boxes on exhaustively annotated side views.

Truth JSON: {"views": [{"frame": "a.jpg", "side": "left", "objects":
[{"class_name": "harvest_ready", "bbox": [0,0,10,10]}]}]}.
Empty objects explicitly denotes a verified negative view. Predictions use
the pipeline detections.csv. Metrics apply only to listed views.
"""
import argparse
import csv
import json
import hashlib
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ripeness_demo.detectors import _box_iou, class_order_for


def evaluate(predictions, truth, taxonomy, iou=0.5):
    if not 0 < iou <= 1:
        raise ValueError("IoU must be in (0, 1]")
    classes = class_order_for([], taxonomy)
    views = {}
    for view in truth["views"]:
        key = (view["frame"], view["side"])
        if key in views or key[1] not in ("left", "right"):
            raise ValueError("Duplicate or invalid truth view")
        views[key] = view["objects"]
    if not views:
        raise ValueError("No exhaustively annotated views supplied")
    def validate(box, name):
        if name not in classes or len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Invalid class or bounding box")
    by_view = {key: [] for key in views}
    for row in predictions:
        key = (row["frame"], row["side"])
        if key not in views:
            continue
        box = [float(row[k]) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
        validate(box, row["class_name"])
        by_view[key].append((box, row["class_name"]))
    tp, fp, fn, confusion = Counter(), Counter(), Counter(), Counter()
    matched = missed = extra = wrong = 0
    per_view = []
    for key, objects in views.items():
        before = (matched, missed, extra, wrong)
        for obj in objects:
            validate(obj["bbox"], obj["class_name"])
        pred = by_view[key]
        pairs = sorted(((-_box_iou(box, obj["bbox"]), pi, gi)
                        for pi, (box, _) in enumerate(pred)
                        for gi, obj in enumerate(objects)
                        if _box_iou(box, obj["bbox"]) >= iou))
        used_p, used_g = set(), set()
        for _, pi, gi in pairs:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi); used_g.add(gi)
            predicted, actual = pred[pi][1], objects[gi]["class_name"]
            confusion[(actual, predicted)] += 1
            matched += 1
            if predicted == actual:
                tp[actual] += 1
            else:
                fp[predicted] += 1; fn[actual] += 1; wrong += 1
        for pi, (_, name) in enumerate(pred):
            if pi not in used_p:
                fp[name] += 1; extra += 1
        for gi, obj in enumerate(objects):
            if gi not in used_g:
                fn[obj["class_name"]] += 1; missed += 1
        per_view.append(dict(frame=key[0], side=key[1], truth_objects=len(objects), predictions=len(pred),
                             **dict(zip(("matched", "missed", "extra", "wrong_class"),
                                        (a-b for a,b in zip((matched,missed,extra,wrong),before))))))
    def ratio(a, b):
        return a / b if b else None
    return dict(scope="exhaustively_annotated_views_only", taxonomy=taxonomy,
                matching="class-agnostic descending-IoU greedy one-to-one; not COCO AP", iou=iou,
                views=len(views), matched=matched, missed=missed, extra=extra, wrong_class=wrong,
                total_prediction_records=len(predictions),
                evaluated_prediction_records=sum(len(p) for p in by_view.values()),
                excluded_prediction_records=sum((r["frame"],r["side"]) not in views for r in predictions),
                per_view=per_view,
                detection_precision=ratio(matched, matched+extra), detection_recall=ratio(matched, matched+missed),
                per_class={c: dict(tp=tp[c], fp=fp[c], fn=fn[c], precision=ratio(tp[c],tp[c]+fp[c]),
                                   recall=ratio(tp[c],tp[c]+fn[c]), f1=ratio(2*tp[c],2*tp[c]+fp[c]+fn[c])) for c in classes},
                confusion=[dict(actual=a, predicted=p, count=n) for (a,p),n in sorted(confusion.items())])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--taxonomy", required=True, choices=("paper_four_stage", "legacy", "green_gem"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--iou", type=float, default=.5)
    args = parser.parse_args()
    with Path(args.predictions).open(encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    report = evaluate(predictions, json.loads(Path(args.truth).read_text(encoding="utf-8")), args.taxonomy, args.iou)
    report["input_sha256"] = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                              for name, path in (("predictions", args.predictions), ("truth", args.truth))}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
