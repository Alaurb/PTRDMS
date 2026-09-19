import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.render_path_evidence import render


class RenderPathEvidenceTests(unittest.TestCase):
    def test_renders_path_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.csv"
            with trace.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["x_east_m", "y_north_m", "match_delta_s"])
                writer.writeheader()
                writer.writerows([
                    {"x_east_m": 0, "y_north_m": 0, "match_delta_s": .01},
                    {"x_east_m": 1, "y_north_m": 2, "match_delta_s": .02},
                ])
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"match_delta_statistics_s": {"maximum": .02}}), encoding="utf-8")
            output = root / "preview.png"
            render(trace, metadata, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
