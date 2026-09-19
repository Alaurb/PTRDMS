# Four-stage maturity-classification evaluation

## Task and reviewed corpus

The maturity classifier assigns a reviewed tomato crop to one of four visual classes: immature period, green-maturity period, discoloration period, or maturity period. The corpus contains 815 usable crops from 210 source-image groups: 222 immature, 223 green-maturity, 228 discoloration, and 142 maturity. Forty-six crops labelled `unclassifiable` were excluded from training, validation, and testing.

All crops derived from the same source image were assigned to the same split. The reviewed crop images, annotations, five fixed splits, and exclusion list are published separately as the [Four-Stage Tomato Ripeness Crop Dataset for PTRDMS, Version v2](https://doi.org/10.5281/zenodo.22827683); this repository publishes the deployed weight and the evaluation record, not the crop corpus.

## Model-selection protocol

Five repeated, source-group-disjoint 70/15/15 holdouts were generated with seeds 20260915–20260919. Each repetition contained 572 training, 122 validation, and 121 test crops. The five repetitions are not five-fold cross-validation: their test sets may overlap, and no best single repetition was selected for reporting.

YOLOv8n-cls was initialized from `yolov8n-cls.pt`, trained at 224 pixels with the framework's standard classification augmentation, and selected using validation top-1 accuracy only. The held-out test split was used once for each repetition after selection. The detector and maturity classifier are evaluated separately; these crop-level scores are not end-to-end panorama-system scores.

## Repeated group-holdout results

| Model | Test accuracy, mean ± sample SD | Test macro-F1, mean ± sample SD |
|---|---:|---:|
| ImageNet-pretrained ResNet-18 | 0.588 ± 0.080 | 0.608 ± 0.076 |
| ImageNet-pretrained YOLOv8n-cls | **0.625 ± 0.022** | **0.640 ± 0.019** |

For YOLOv8n-cls, the mean class-wise F1 scores were 0.705 (immature), 0.546 (green maturity), 0.501 (discoloration), and 0.807 (maturity). Green-maturity and discoloration are the principal confusion pair. These results measure visual labels only; they do not establish internal maturity, firmness, soluble solids, or harvest readiness.

Two predeclared alternatives were retained as negative results: a colour-preserving 320-pixel recipe reached 0.626 ± 0.037 accuracy and 0.637 ± 0.039 macro-F1, and adding a separate 199-crop training-only expansion reached 0.623 ± 0.071 accuracy and 0.633 ± 0.068 macro-F1. Neither replaced the baseline because macro-F1 did not improve and variation increased.

## Deployed model

After model selection, the deployed `models/tomato_ripeness_classifier.pt` was trained on all 815 reviewed four-class crops for 11 fixed epochs (the median validation-selected epoch among the five selection runs). This full-corpus fit has no held-out evaluation split; its training-fit metrics are not reported as performance. Its class dictionary is:

```text
0 discoloration-period
1 green-maturity-period
2 immature-period
3 maturity-period
```

The weight SHA-256 is `934d2f956117e02e45bdb1e03922c85a51820c007c990497b2df4b68a295563c`.

## Limits

The corpus is modest, and repeated split variation remains material. The next confirmatory study should freeze an independent route/date-held-out test set before additional model selection. A separate diagnostic batch from a different acquisition date showed domain shift and had only one maturity-period crop; it is not used as a four-class external performance claim. The published model should therefore be used as an offline research component rather than evidence of universal field generalization.
