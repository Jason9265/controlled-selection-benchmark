import csv
import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "selection-benchmark"


class ReleaseTests(unittest.TestCase):
    def call(self, *args):
        return subprocess.run([sys.executable, str(CLI), *args], cwd=ROOT, check=True, text=True, capture_output=True)

    def test_audit_passes_with_released_artifacts(self):
        data = json.loads(self.call("validate-artifacts").stdout)
        self.assertTrue(data["pass"])
        self.assertEqual(data["counts"]["isprs_blind_scores"], 72)

    def test_audit_command_matches_validator(self):
        validator = json.loads(self.call("validate-artifacts").stdout)
        audit = json.loads(self.call("audit").stdout)
        self.assertEqual(audit, validator)

    def test_random_receipts_rebuild_from_released_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "control_receipts.json"
            self.call("freeze", "--workspace", temporary, "--output", str(output), "--released-manifest")
            rebuilt = json.loads(output.read_text())["receipts"]
            released = json.loads((ROOT / "artifacts/isprs/control_receipts.json").read_text())["receipts"]
            self.assertEqual(rebuilt, released)

    def test_release_contains_no_data_or_weights(self):
        manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
        prohibited = {".pt", ".pth", ".zip", ".tif", ".tiff"}
        self.assertFalse([row["path"] for row in manifest["files"] if Path(row["path"]).suffix.lower() in prohibited])

    def test_release_manifest_hashes_every_distributable_file(self):
        manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
        expected = {
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*")
            if path.is_file()
            and path.name != "RELEASE_MANIFEST.json"
            and ".git" not in path.parts
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        }
        recorded = {row["path"] for row in manifest["files"]}
        self.assertEqual(recorded, expected)
        for row in manifest["files"]:
            path = ROOT / row["path"]
            self.assertEqual(path.stat().st_size, row["bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])

    def test_cross_protocol_consistency_summary(self):
        data = json.loads((ROOT / "artifacts/evidence/cross_protocol_consistency_summary.json").read_text())
        score_path = ROOT / "artifacts/evidence/cross_protocol_score_rows.csv"
        with score_path.open(newline="", encoding="utf-8") as handle:
            scores = list(csv.DictReader(handle))
        self.assertEqual(len(scores), 1062)
        self.assertEqual(set(scores[0]), {
            "layer", "method", "region_or_direction", "architecture", "budget_units",
            "budget_fraction", "train_or_model_seed", "selection_seed", "miou_fixed",
        })
        self.assertEqual({layer: sum(row["layer"] == layer for row in scores) for layer in {
            "oem_v2", "loveda_v2", "v4f", "isprs"}},
            {"oem_v2": 648, "loveda_v2": 162, "v4f": 180, "isprs": 72})
        self.assertEqual(data["score_rows"]["sha256"], hashlib.sha256(score_path.read_bytes()).hexdigest())
        self.assertEqual(len(data["condition_cells"]), 228)
        controls = data["controls_by_layer"]
        score_lookup = {}
        for row in scores:
            key = (row["layer"], row["region_or_direction"], row["architecture"],
                   row["budget_fraction"], row["budget_units"],
                   row["train_or_model_seed"], row["selection_seed"])
            score_lookup.setdefault(key, {})[row["method"]] = float(row["miou_fixed"])
        rebuilt = []
        for cell in data["condition_cells"]:
            differences = []
            for seed in cell["seed_pairs"]:
                key = (cell["layer"], cell["region_or_direction"], cell["architecture"],
                       str(cell["budget_fraction"]), str(cell["budget_units"]),
                       str(seed["train_or_model_seed"]), str(seed["selection_seed"]))
                method_scores = score_lookup[key]
                differences.append(100 * (method_scores[cell["method"]] -
                                           max(method_scores[name] for name in controls[cell["layer"]])))
            self.assertEqual(len(differences), 3)
            for observed, recorded in zip(differences, cell["differences_vs_strongest_control_pp"]):
                self.assertAlmostEqual(observed, recorded, places=10)
            rebuilt.append((cell["layer"], sum(differences) / 3,
                            all(value > 0 for value in differences)))
        self.assertEqual(len(rebuilt), 228)
        expected = {
            "oem_v2": (9, 162, [72, 90, 0], 21, [-10.87, 7.74]),
            "loveda_v2": (6, 36, [4, 32, 0], 2, [-5.64, 3.78]),
            "v4f": (2, 24, [4, 20, 0], 0, [-6.06, 1.28]),
            "isprs": (1, 6, [1, 5, 0], 0, [-2.98, 0.08]),
        }
        fixed_expected = {
            "oem_v2": {"random": ([122, 40, 0], 59), "uniform_random": ([116, 46, 0], 46),
                       "original_uniform_random": ([99, 63, 0], 48)},
            "loveda_v2": {"random": ([23, 13, 0], 9), "uniform_random": ([20, 16, 0], 3),
                          "original_uniform_random": ([20, 16, 0], 5)},
            "v4f": {"uniform_patch": ([6, 18, 0], 3), "original_uniform": ([5, 19, 0], 3),
                    "original_quota": ([7, 17, 0], 1)},
            "isprs": {"patch_uniform": ([1, 5, 0], 0), "original_uniform": ([1, 5, 0], 0),
                      "original_quota": ([1, 5, 0], 1)},
        }
        for layer, (methods, cells, outcomes, consistent, value_range) in expected.items():
            summary = data["summary_by_layer"][layer]
            rows = [row for row in data["condition_cells"] if row["layer"] == layer]
            recomputed = [sum(row["mean_outcome"] == name for row in rows) for name in ("win", "loss", "tie")]
            self.assertEqual(summary["structured_methods"], methods)
            self.assertEqual(summary["condition_cells"], cells)
            self.assertEqual(summary["mean_wins_losses_ties"], outcomes)
            self.assertEqual(recomputed, outcomes)
            self.assertEqual(sum(row["three_seed_consistent_win"] for row in rows), consistent)
            observed_range = [round(min(row["mean_difference_pp"] for row in rows), 2),
                              round(max(row["mean_difference_pp"] for row in rows), 2)]
            self.assertEqual(observed_range, value_range)
            self.assertEqual(summary["universal_methods"], [])
            for control, (fixed_counts, fixed_consistent) in fixed_expected[layer].items():
                fixed = summary["fixed_control_sensitivity"][control]
                values = [row["differences_by_fixed_control_pp"][control] for row in rows]
                outcomes_from_cells = ["win" if sum(value) > 0 else "loss" if sum(value) < 0 else "tie" for value in values]
                self.assertEqual([outcomes_from_cells.count(name) for name in ("win", "loss", "tie")], fixed_counts)
                self.assertEqual(sum(all(item > 0 for item in value) for value in values), fixed_consistent)
                self.assertEqual(fixed["mean_wins_losses_ties"], fixed_counts)
                self.assertEqual(fixed["three_seed_consistent_wins"], fixed_consistent)
                self.assertEqual(fixed["universal_methods"], [])

    def test_score_rows_are_minimal_and_seed_complete(self):
        path = ROOT / "artifacts/evidence/cross_protocol_score_rows.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        prohibited = ("/home/", "\\", ".pt", ".pth", ".tif", ".zip")
        self.assertFalse([row for row in rows if any(token in value for value in row.values() for token in prohibited)])
        grouped = {}
        for row in rows:
            key = (row["layer"], row["method"], row["region_or_direction"], row["architecture"],
                   row["budget_fraction"], row["budget_units"])
            grouped.setdefault(key, set()).add((row["train_or_model_seed"], row["selection_seed"]))
        self.assertTrue(all(len(seed_pairs) == 3 for seed_pairs in grouped.values()))


if __name__ == "__main__":
    unittest.main()
