#!/usr/bin/env python3
"""Render an image-free RTK trigger path preview from exported evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _scale(value: float, lower: float, upper: float, start: int, end: int) -> float:
    return start + (value - lower) * (end - start) / (upper - lower)


def render(trace_csv: Path, metadata_json: Path, output_png: Path) -> None:
    with trace_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("At least two matched path rows are required")
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    east = [float(row["x_east_m"]) for row in rows]
    north = [float(row["y_north_m"]) for row in rows]
    deltas_ms = [float(row["match_delta_s"]) * 1000 for row in rows]
    x_padding = max((max(east) - min(east)) * .06, 1.0)
    y_padding = max((max(north) - min(north)) * .06, 1.0)
    x0, x1 = min(east) - x_padding, max(east) + x_padding
    y0, y1 = min(north) - y_padding, max(north) + y_padding

    width, height = 1400, 900
    left, top, right, bottom = 140, 120, 90, 150
    plot_left, plot_top = left, top
    plot_right, plot_bottom = width - right, height - bottom
    image = Image.new("RGB", (width, height), "#f8fbf9")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((left, 36), "RTK trigger path from test.bag", fill="#173d30", font=font)
    stats = metadata.get("match_delta_statistics_s", {})
    subtitle = (
        f"{len(rows):,} trigger/RTK matches · ROS bag record time · "
        f"max timing difference {float(stats.get('maximum', 0)) * 1000:.3f} ms"
    )
    draw.text((left, 62), subtitle, fill="#52675e", font=font)
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#b9c8c0", width=2)

    for fraction in range(0, 6):
        x = plot_left + fraction * (plot_right - plot_left) / 5
        y = plot_top + fraction * (plot_bottom - plot_top) / 5
        draw.line((x, plot_top, x, plot_bottom), fill="#e4ece7", width=1)
        draw.line((plot_left, y, plot_right, y), fill="#e4ece7", width=1)
        x_value = x0 + fraction * (x1 - x0) / 5
        y_value = y1 - fraction * (y1 - y0) / 5
        draw.text((x - 14, plot_bottom + 16), f"{x_value:.0f}", fill="#52675e", font=font)
        draw.text((plot_left - 48, y - 4), f"{y_value:.0f}", fill="#52675e", font=font)
    draw.text((plot_left + (plot_right - plot_left) // 2 - 35, height - 72), "East (m)", fill="#173d30", font=font)
    draw.text((26, plot_top + (plot_bottom - plot_top) // 2), "North (m)", fill="#173d30", font=font)

    points = [
        (
            _scale(x, x0, x1, plot_left, plot_right),
            _scale(y, y0, y1, plot_bottom, plot_top),
        )
        for x, y in zip(east, north)
    ]
    for index, (first, second) in enumerate(zip(points, points[1:])):
        ratio = index / max(len(points) - 2, 1)
        color = (int(29 + 195 * ratio), int(122 - 45 * ratio), int(150 - 100 * ratio))
        draw.line((first, second), fill=color, width=3)
    for index in range(0, len(points), 50):
        x, y = points[index]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#ffffff", outline="#173d30")
    for point, color, label in ((points[0], "#2f855a", "Start"), (points[-1], "#c2410c", "End")):
        x, y = point
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline="#ffffff", width=2)
        draw.text((x + 12, y - 5), label, fill="#173d30", font=font)

    legend_y = height - 38
    draw.line((left, legend_y, left + 110, legend_y), fill="#1d7a96", width=4)
    draw.line((left + 110, legend_y, left + 220, legend_y), fill="#d95f0e", width=4)
    draw.text((left + 232, legend_y - 5), "Path colour: earlier → later; hollow markers: every 50th trigger", fill="#52675e", font=font)
    draw.text((width - 350, height - 38), f"Median delta {sorted(deltas_ms)[len(deltas_ms)//2]:.3f} ms", fill="#52675e", font=font)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a path image from PTRDMS bag evidence")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.trace, args.metadata, args.output)


if __name__ == "__main__":
    main()
