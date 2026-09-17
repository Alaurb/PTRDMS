# YOLOv8 detector experiments, review, and zero-shot comparison

## Paper-facing model roles

The manuscript-facing tomato detector is a supervised **YOLOv8** model. It is the only detector whose weights may be used in the manuscript's primary result table and deployment pipeline.

Grounding DINO is useful, but has a different role: it can be reported as a **frozen, zero-shot detector baseline**. It is not fine-tuned on the study data and it must not replace YOLOv8 in the manuscript's main method. SAM 2 receives prompts such as boxes or points; it may help generate masks or crops for annotation, but it is not an independent box detector and therefore is excluded from YOLOv8-versus-DINO detection tables.

## Fixed split and acceptance rule

Create the grouped split once, record its `manifest.csv` and `data.yaml` SHA-256 values, and do not alter the validation images while comparing models. In the current review run, the detector split contains 137 training images and 36 validation images. Any newly reviewed images are a new, versioned **training-only expansion** after group-leakage screening; they do not change the frozen validation set.

Every YOLOv8 run must preserve:

- the copied training `args.yaml`;
- SHA-256 values for the split manifest, dataset YAML, and `weights/best.pt`;
- Precision, Recall, box mAP50, and box mAP50-95 from the same selected epoch;
- hardware, Ultralytics version, seed, and runtime measurement.

The current accepted YOLOv8 baseline remains the reference. A YOLOv8 optimization can be marked **accepted** only when box mAP50-95 strictly improves and Precision, Recall, and mAP50 do not decrease on the same frozen validation split. Record failed trials as `rejected`; do not discard them.

Use the registry helper after each run:

```bash
python scripts/register_yolo_experiment.py \
  --run-dir outputs/yolov8_tomato_detector_v1/<run_name> \
  --registry-dir outputs/yolov8_tomato_detector_v1/experiment_registry \
  --split-manifest data/training/tomato_detector_v1/manifest.csv \
  --data-yaml data/training/tomato_detector_v1/data.yaml \
  --experiment-id <run_name> --status rejected --model YOLOv8n --task detect \
  --notes "What changed and why"
```

The tool refuses to register a run as `accepted` unless it meets the rule above.

## Browser annotation review

The standard-library review server edits only a standalone review package containing `images/`, `labels_to_review/`, and `audit_manifest.csv`. It never points at a frozen train/validation directory.

```bash
python scripts/serve_detection_review.py \
  --review-root data/review/tomato_detector_yolov8n_review_v1 \
  --host 127.0.0.1 --port 8766
```

Open the printed URL. Drag a box to move it, drag a corner to resize, and enable **Add-box mode** (or hold `Shift`) to draw a missing fruit even over an existing box. Save then approve an image only after false boxes are removed, missed fruit are added, and loose boxes are tightened. Approved files remain review data until a leakage check creates a named training-expansion version.

## Grounding DINO zero-shot baseline

Evaluate Grounding DINO as a detector only after all of these settings are fixed **without inspecting the frozen validation metrics**:

1. detector commit/checkpoint SHA-256 and inference environment;
2. one text prompt (for example `tomato.`) and box/text thresholds;
3. image pre-processing and non-maximum suppression settings;
4. mapping of DINO outputs to standard one-class YOLO boxes.

Run that frozen configuration on the exact 36 validation images and score it against the same human boxes using the same evaluator as YOLOv8. Report Precision, Recall, mAP50, mAP50-95, and per-image latency on the same GPU. State clearly in the table that DINO is *zero-shot* and YOLOv8 is *supervised on the 137 training images*; they answer different questions.

Do not use a prompt chosen by repeatedly maximizing the 36-image validation score. Choose it on a training-side development subset or retain the predeclared generic prompt `tomato.`. If DINO + SAM 2 masks are later evaluated, create manual instance masks and report mask AP separately; converting a SAM mask back into a box is not evidence that it improves box localization.

## Relation to four-stage maturity recognition

This detector protocol measures only whether a tomato is localized. Four-stage maturity recognition requires its own grouped crop split and per-class precision, recall, F1, and confusion matrix. Do not present one-class detector mAP as four-stage maturity accuracy.
