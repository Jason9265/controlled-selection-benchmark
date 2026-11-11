"""Create the immutable v4f architecture-specific development selection receipt."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config = json.loads((ROOT / "config.json").read_text())
    records = []
    for path in sorted((ROOT / "runs").rglob("complete.json")):
        row = json.loads(path.read_text())
        final = row["metrics"][-1]
        if final["step"] != 300:
            raise RuntimeError("unexpected final checkpoint: " + str(path))
        records.append({"arch": row["arch"], "seed": row["train_seed"],
                        "variant": row["variant"]["name"],
                        "development_miou": final["development_val"]["miou_fixed"],
                        "source_miou": final["source_val"]["miou_fixed"],
                        "path": str(path.relative_to(ROOT))})
    expected = len(config["architectures"]) * len(config["seed_pairs"]) * len(config["variants"])
    if len(records) != expected:
        raise RuntimeError(f"incomplete diagnostics: {len(records)} != {expected}")
    summary, selected = {}, {}
    for arch in config["architectures"]:
        variants = []
        for item in config["variants"]:
            rows = [r for r in records if r["arch"] == arch and r["variant"] == item["name"]]
            if len(rows) != len(config["seed_pairs"]):
                raise RuntimeError("missing paired seed")
            variants.append({"variant": item["name"], "mean_development_miou": sum(r["development_miou"] for r in rows) / len(rows),
                             "mean_source_miou": sum(r["source_miou"] for r in rows) / len(rows),
                             "per_seed": rows})
        variants.sort(key=lambda x: (-x["mean_development_miou"], x["variant"]))
        summary[arch] = variants
        selected[arch] = variants[0]["variant"]
    out = {"protocol": config["protocol"], "selection_rule": config["selection"]["development_selection"],
           "selected_variants": selected, "ranking": summary, "records": records,
           "config_sha256": sha(ROOT / "config.json"),
           "manifest_sha256": sha(ROOT / "prepared/manifest.json"),
           "confirmation_label_access": False}
    destination = ROOT / "reports/development_selection_receipt.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"selected_variants": selected, "confirmation_label_access": False}, indent=2))


if __name__ == "__main__":
    main()
