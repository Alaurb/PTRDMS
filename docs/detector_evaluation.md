# Tomato-detector evaluation

The tomato detector is a one-class YOLOv8n model evaluated independently from the four-stage crop classifier. The fixed validation set contains 36 projected views held out by complete panorama acquisition group; no left/right views from one panorama acquisition are placed in different splits.

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Accepted YOLOv8n baseline | 0.6310 | 0.7092 | 0.7006 | 0.3243 |

The selected epoch is the one with the highest validation mAP@0.5:0.95. Candidate changes are accepted only when mAP@0.5:0.95 improves and precision, recall, and mAP@0.5 do not decrease on this unchanged validation set. The detector split-manifest SHA-256 is `7c2eedfc1d499ff4af9dda9357ea48a06917825c623eb41e19129faebdcdd6d3`, and the deployed detector checkpoint SHA-256 is `168fc5eb556ebc5a8266b9d934af0c953da221d1de8ab41fb9bcb7f736bafabd`.

These are detector-only metrics. They do not measure end-to-end ripeness classification, unique-fruit counting, row exclusion, physical fruit height, navigation, or mapping accuracy.
