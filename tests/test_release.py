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
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        }
        recorded = {row["path"] for row in manifest["files"]}
        self.assertEqual(recorded, expected)
        for row in manifest["files"]:
            path = ROOT / row["path"]
            self.assertEqual(path.stat().st_size, row["bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])


if __name__ == "__main__":
    unittest.main()
