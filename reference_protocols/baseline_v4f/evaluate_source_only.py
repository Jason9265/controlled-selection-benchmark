"""Post-selection source-only confirmation evaluation for v4f."""
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / "baseline_v2"
OLD = ROOT.parent / "baseline"
sys.path[:0] = [str(V2), str(OLD)]
import engine as numeric
from runner import load_model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def main():
    config = json.loads((ROOT / "confirmation_config.json").read_text())
    manifest = json.loads((ROOT / config["manifest"]).read_text())
    base = json.loads((V2 / "configs/oem.json").read_text())
    for region, target in manifest["targets"].items():
        test = numeric.Pack(target["test"], ROOT, True, base["num_classes"])
        for arch in config["architectures"]:
            for train_seed, query_seed in config["seed_pairs"]:
                checkpoint = V2 / "runs/oem" / arch / "source" / str(train_seed) / "best.pt"
                model = load_model(checkpoint, arch, base["num_classes"], base)
                metrics = numeric.evaluate(model, test, base["num_classes"])
                put(ROOT / "confirmation_runs/source_only" / region / arch / f"m{train_seed}_q{query_seed}.json", {
                    "region": region, "arch": arch, "train_seed": train_seed, "query_seed": query_seed,
                    "metrics": metrics, "source_checkpoint_sha256": sha(checkpoint),
                    "confirmation_manifest_sha256": sha(ROOT / config["manifest"]),
                    "confirmation_test_labels_read_by_evaluation": True,
                })
                del model
                torch.cuda.empty_cache()
        del test
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
