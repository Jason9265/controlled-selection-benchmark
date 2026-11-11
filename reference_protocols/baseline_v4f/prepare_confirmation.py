"""Materialize frozen v4f confirmation labels only after the development receipt."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / "baseline_v2"
sys.path.insert(0, str(V2))
import prepare_data as p


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    decision = read(ROOT / "reports/development_selection_receipt.json")
    frozen = read(ROOT / "reports/target_frozen.json")["regions"]
    dev = read(ROOT / "prepared/manifest.json")
    if decision["confirmation_label_access"] is not False or set(frozen) != {"mahe", "svaneti", "tonga"}:
        raise RuntimeError("invalid development barrier")
    out = ROOT / "confirmation"
    if (out / "manifest.json").exists():
        raise FileExistsError("confirmation manifest already exists")
    p.ROOT, p.BASE, p.PILOT = ROOT, V2, ROOT.parent / "baseline_v4c"
    p.DEADLINE = time.monotonic() + 14400
    p.STATE = {"archive_url": p.URL, "archive_size_bytes": p.SIZE, "network_bytes_this_run": 0,
               "previous_network_bytes": 0, "files": {}, "rate_limit_events": [], "started_at_utc": p.utc()}
    seen = {row["original_rgb_sha256"]: row["original_member"]
            for key in ("source_train", "source_val") for row in dev[key]}
    for row in dev["targets"]["mahe"]["pool"] + dev["targets"]["mahe"]["test"]:
        seen[row["original_rgb_sha256"]] = row["original_member"]
    archive = p.Archive()
    try:
        confirmation = {"schema": "oem-v4f-confirmation", "num_classes": 8, "classes": p.CLASSES,
                        "source_train": dev["source_train"], "source_val": dev["source_val"], "targets": {},
                        "target_freeze_sha256": sha(ROOT / "reports/target_frozen.json"),
                        "development_selection_sha256": sha(ROOT / "reports/development_selection_receipt.json")}
        audit = {}
        for name in ("svaneti", "tonga"):
            region = frozen[name]
            rows, images = p.prepare_region(archive, region, seen)
            for split in ("train", "val"):
                for row in rows[split]:
                    row["image"] = str((ROOT / row["image"]).resolve())
                    row["mask"] = str((ROOT / row["mask"]).resolve())
            confirmation["targets"][name] = {"pool": rows["train"], "test": rows["val"], "role": "confirmation"}
            n = len(rows["train"])
            budgets = [max(1, -(-n * pct // 100)) for pct in (1, 3, 5)]
            if len(set(budgets)) != 3:
                raise RuntimeError("collapsed budgets " + name)
            audit[name] = {"pool_patches": n, "test_patches": len(rows["val"]), "budgets": budgets,
                           "images": images, "target_label_statistics_computed": False}
        put(out / "manifest.json", confirmation)
        put(ROOT / "reports/confirmation_data_guard.json", {"status": "PASS", "manifest_sha256": sha(out / "manifest.json"),
            "development_selection_sha256": sha(ROOT / "reports/development_selection_receipt.json"),
            "targets": audit, "confirmation_access_began": True, "selection_after_access_prohibited": True})
    finally:
        archive.close()


if __name__ == "__main__":
    main()
