"""Command-line tools for the IGARSS 2027 controlled-selection release.

The no-data commands audit the released evidence. Dataset-dependent commands
operate only on user-supplied official archives and never distribute them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

PACKAGE = Path(__file__).resolve().parents[1]
ARTIFACTS = PACKAGE / "artifacts"


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol():
    return read_json(ARTIFACTS / "isprs" / "protocol_frozen.json")


def receipt_errors(receipt_file: Path, expected: int) -> list[str]:
    data = read_json(receipt_file)
    receipts = data.get("receipts", [])
    cfg = protocol()
    errors: list[str] = []
    if data.get("expected_receipts") != expected or len(receipts) != expected:
        errors.append(f"expected {expected} receipts, found {len(receipts)}")
    seen = set()
    nested = defaultdict(list)
    for index, row in enumerate(receipts):
        key = tuple(row.get(k) for k in ("direction", "method", "model_seed", "selection_seed", "budget_crops"))
        if key in seen:
            errors.append(f"receipt {index}: duplicate key")
        seen.add(key)
        direction = row.get("direction")
        spec = cfg.get("directions", {}).get(direction)
        if not spec:
            errors.append(f"receipt {index}: unknown direction")
            continue
        ids, groups = row.get("ids", []), set(row.get("original_groups", []))
        if len(ids) != row.get("budget_crops") or len(set(ids)) != len(ids):
            errors.append(f"receipt {index}: non-unique or incorrect crop count")
        if groups & set(spec["target_holdout"]) or not groups <= set(spec["target_pool"]):
            errors.append(f"receipt {index}: pool/holdout violation")
        if row.get("target_holdout_labels_read_for_selection") is not False:
            errors.append(f"receipt {index}: holdout-label guard is not false")
        nested[(direction, row.get("method"), row.get("model_seed"), row.get("selection_seed"))].append(row)
    for key, rows in nested.items():
        previous: set[str] = set()
        for row in sorted(rows, key=lambda item: item["budget_crops"]):
            current = set(row["ids"])
            if not previous <= current:
                errors.append(f"{key}: budgets are not nested")
            previous = current
    return errors


def validate(_: argparse.Namespace) -> None:
    evidence = ARTIFACTS / "evidence"
    isprs = ARTIFACTS / "isprs"
    oem = read_json(evidence / "oem_result_audit.json")
    loveda = read_json(evidence / "loveda_result_audit.json")
    v4f = read_json(evidence / "confirmation_method_summary.json")
    final = read_json(isprs / "final_evidence_audit.json")
    ledger = read_json(evidence / "paper_evidence_ledger.json")
    checks = {
        "oem_saved_records": oem.get("passed") is True and sum(x["verified_runs"] for x in oem["architectures"]) == 648,
        "loveda_saved_records": loveda.get("passed") is True and sum(x["verified_runs"] for x in loveda["architectures"]) == 162,
        "v4f_confirmation": v4f.get("status") == "complete" and len(v4f.get("records", [])) == 180,
        "isprs_ledger": final.get("pass") is True and final.get("total_blind_holdout_scores") == 72,
        "control_receipts": not receipt_errors(isprs / "control_receipts.json", 54),
        "ripu_receipts": not receipt_errors(isprs / "ripu_receipts.json", 18),
        "claim_ledger": ledger.get("status") == "claim-scoped evidence complete" and all(ledger.get("checks", {}).values()),
    }
    payload = {
        "pass": all(checks.values()),
        "checks": checks,
        "counts": {"oem": 648, "loveda": 162, "v4f": 180, "isprs_blind_scores": 72},
        "receipt_errors": {
            "control": receipt_errors(isprs / "control_receipts.json", 54),
            "ripu": receipt_errors(isprs / "ripu_receipts.json", 18),
        },
        "scope": "Integrity audit only; it does not make a universal method-effectiveness claim.",
    }
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        raise SystemExit(1)


def summarize(args: argparse.Namespace) -> None:
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = ARTIFACTS / "figures" / "figure_1_isprs_budget_curves.json"
    figure = read_json(source)
    controls = read_json(ARTIFACTS / "isprs" / "control_summary.json")
    ripu = read_json(ARTIFACTS / "isprs" / "ripu_summary.json")
    paired = read_json(ARTIFACTS / "isprs" / "final_evidence_audit.json")["paired_comparisons"]
    rows = ["# Released ISPRS summary", "", "All values are target-holdout fixed-class mIoU; mean ± sample SD over three paired seeds.", "", "| Direction | Budget | RIPU − patch | RIPU − original-uniform | RIPU − original-quota |", "|---|---:|---:|---:|---:|"]
    table = defaultdict(dict)
    for item in paired:
        table[(item["direction"], item["budget_fraction"])][item["control_method"]] = item
    for (direction, budget), values in sorted(table.items()):
        fmt = lambda method: f"{values[method]['ripu_minus_control_mean'] * 100:+.2f} ± {values[method]['ripu_minus_control_sd'] * 100:.2f}"
        rows.append(f"| {direction} | {budget * 100:.0f}% | {fmt('patch_uniform')} | {fmt('original_uniform')} | {fmt('original_quota')} |")
    (out / "isprs_paired_differences.md").write_text("\n".join(rows) + "\n")
    payload = {"figure": figure, "control_summary": controls["summary"], "ripu_summary": ripu["summary"], "paired_comparisons": paired}
    (out / "isprs_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    for suffix in (".pdf", ".svg", ".png", ".json"):
        shutil.copy2(ARTIFACTS / "figures" / f"figure_1_isprs_budget_curves{suffix}", out / f"figure_1_isprs_budget_curves{suffix}")
    print(json.dumps({"output": str(out), "files": 6}, indent=2))


def archive_paths(data_root: Path, cfg: dict) -> dict[str, Path]:
    paths = {}
    for name in cfg["source_archives"]:
        candidates = list(data_root.rglob(name))
        if len(candidates) != 1:
            raise FileNotFoundError(f"expected exactly one {name} beneath {data_root}, found {len(candidates)}")
        paths[name] = candidates[0]
    return paths


def prepare(args: argparse.Namespace) -> None:
    """Materialize the frozen 512px crops from official archives supplied by the user."""
    import numpy as np
    from PIL import Image

    data_root, workspace = Path(args.data_root).resolve(), Path(args.workspace).resolve()
    cfg = protocol()
    archives = archive_paths(data_root, cfg)
    mismatches = {name: sha256(path) for name, path in archives.items() if sha256(path) != cfg["source_archives"][name]}
    if mismatches:
        raise RuntimeError(json.dumps({"unexpected_archive_sha256": mismatches}, indent=2))
    output = workspace / "prepared"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    (output / "images").mkdir(parents=True)
    (output / "labels").mkdir()
    colors = {(255, 255, 255): 0, (0, 0, 255): 1, (0, 255, 255): 2, (0, 255, 0): 3, (255, 255, 0): 4, (255, 0, 0): 5}

    def image_from_zip(zipped, name):
        import io
        with zipped.open(name) as handle:
            return Image.open(io.BytesIO(handle.read())).convert("RGB")

    def decode_label(image):
        array = np.asarray(image)
        label = np.full(array.shape[:2], 255, np.uint8)
        for color, klass in colors.items():
            label[(array == color).all(2)] = klass
        if np.any((label == 255) & np.any(array != 0, 2)):
            raise ValueError("unexpected non-boundary label color")
        return Image.fromarray(label)

    groups = set()
    for direction in cfg["directions"].values():
        for role in ("source_train", "source_val", "target_pool", "target_holdout"):
            groups.update(direction[role])
    rows = []

    def crop(domain, image_zip, label_zip, image_name, label_name, group):
        image, label = image_from_zip(image_zip, image_name), decode_label(image_from_zip(label_zip, label_name))
        width, height = image.size
        for row in range(0, height - 511, 512):
            for col in range(0, width - 511, 512):
                identifier = f"{domain}_{group}_r{row}_c{col}"
                image_path, label_path = output / "images" / f"{identifier}.png", output / "labels" / f"{identifier}.png"
                image.crop((col, row, col + 512, row + 512)).save(image_path)
                label.crop((col, row, col + 512, row + 512)).save(label_path)
                rows.append({"id": identifier, "domain": domain, "group": group, "role": "frozen", "image": str(image_path.relative_to(workspace)), "mask": str(label_path.relative_to(workspace)), "row": row, "col": col})

    potsdam = sorted(group for group in groups if group.startswith("top_potsdam"))
    vaihingen = sorted(groups - set(potsdam))
    with ZipFile(archives["3_Ortho_IRRG.zip"]) as images, ZipFile(archives["5_Labels_all_noBoundary.zip"]) as labels:
        for group in potsdam:
            crop("potsdam", images, labels, f"3_Ortho_IRRG/{group}_IRRG.tif", f"{group}_label_noBoundary.tif", group)
    with ZipFile(archives["ISPRS_semantic_labeling_Vaihingen.zip"]) as images, ZipFile(archives["ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip"]) as labels:
        for group in vaihingen:
            crop("vaihingen", images, labels, f"top/{group}.tif", f"{group}.tif", group)
    manifest = {"protocol_sha256": sha256(ARTIFACTS / "isprs" / "protocol_frozen.json"), "num_classes": 6, "crop": 512, "rows": rows, "directions": cfg["directions"], "target_holdout_label_access_for_selection": False, "target_holdout_labels_materialized_for_delayed_evaluation": True}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"prepared": str(output), "crops": len(rows)}, indent=2))


def random_order(rows, method, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    if method == "patch_uniform":
        return rng.permutation(len(rows)).tolist()
    by_group = defaultdict(list)
    for index, row in enumerate(rows):
        by_group[row["group"]].append(index)
    for values in by_group.values():
        rng.shuffle(values)
    if method == "original_uniform":
        ordered = []
        while by_group:
            group = rng.choice(sorted(by_group))
            ordered.append(by_group[group].pop())
            if not by_group[group]:
                del by_group[group]
        return ordered
    if method == "original_quota":
        groups = sorted(by_group)
        rng.shuffle(groups)
        ordered = []
        while any(by_group.values()):
            for group in groups:
                if by_group[group]:
                    ordered.append(by_group[group].pop())
        return ordered
    raise ValueError(method)


def freeze(args: argparse.Namespace) -> None:
    workspace, output = Path(args.workspace).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    cfg = protocol()
    manifest_path = workspace / "prepared" / "manifest.json"
    if not manifest_path.exists() and args.released_manifest:
        manifest_path = ARTIFACTS / "isprs" / "prepared_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("target_holdout_label_access_for_selection") is not False:
        raise RuntimeError("holdout selection guard failed")
    if args.method == "ripu_acquisition_only":
        shutil.copy2(ARTIFACTS / "isprs" / "ripu_receipts.json", output)
        print(json.dumps({"frozen": "released label-blind RIPU receipts", "receipts": 18, "note": "score regeneration requires frozen source checkpoints and the upstream DeepLab implementation"}))
        return
    receipts = []
    rows_by_group = defaultdict(list)
    for row in manifest["rows"]:
        rows_by_group[row["group"]].append(row)
    for direction, spec in cfg["directions"].items():
        pool = [row for group in spec["target_pool"] for row in rows_by_group[group]]
        holdout = {row["id"] for group in spec["target_holdout"] for row in rows_by_group[group]}
        budgets = [math.ceil(len(pool) * fraction) for fraction in cfg["budget_fractions"]]
        for model_seed, selection_seed in cfg["seed_pairs"]:
            for method in ("patch_uniform", "original_uniform", "original_quota"):
                order = random_order(pool, method, selection_seed)
                for fraction, budget in zip(cfg["budget_fractions"], budgets):
                    chosen = [pool[index] for index in order[:budget]]
                    ids = [row["id"] for row in chosen]
                    if len(ids) != len(set(ids)) or set(ids) & holdout:
                        raise RuntimeError("invalid frozen selection")
                    receipts.append({"direction": direction, "model_seed": model_seed, "selection_seed": selection_seed, "method": method, "budget_fraction": fraction, "budget_crops": budget, "ids": ids, "original_groups": [row["group"] for row in chosen], "target_holdout_labels_read_for_selection": False})
    released = {"status": "frozen-before-adaptation", "protocol_sha256": sha256(ARTIFACTS / "isprs" / "protocol_frozen.json"), "manifest_sha256": sha256(manifest_path), "expected_receipts": 54, "receipts": receipts, "target_holdout_label_access_for_selection": False, "target_pool_masks_read_for_selection": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(released, indent=2) + "\n")
    errors = receipt_errors(output, 54)
    if errors:
        raise RuntimeError(errors)
    print(json.dumps({"frozen": str(output), "receipts": 54}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="selection-benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-artifacts").set_defaults(func=validate)
    commands.add_parser(
        "audit",
        help="audit evidence counts, receipt integrity, and claim-scope guards",
    ).set_defaults(func=validate)
    summary = commands.add_parser("summarize")
    summary.add_argument("--output", required=True)
    summary.set_defaults(func=summarize)
    prep = commands.add_parser("prepare")
    prep.add_argument("--data-root", required=True)
    prep.add_argument("--workspace", required=True)
    prep.set_defaults(func=prepare)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--workspace", required=True)
    frozen.add_argument("--output", required=True)
    frozen.add_argument("--method", choices=("random", "ripu_acquisition_only"), default="random")
    frozen.add_argument("--released-manifest", action="store_true", help="recreate receipts from released metadata only; it cannot train models")
    frozen.set_defaults(func=freeze)
    source = commands.add_parser("train-source")
    source.add_argument("--workspace", required=True); source.add_argument("--artifacts", default=str(ARTIFACTS))
    source.add_argument("--deeplab-source", required=True); source.add_argument("--imagenet-resnet101", required=True)
    source.add_argument("--direction", choices=("potsdam_to_vaihingen", "vaihingen_to_potsdam"), required=True)
    source.add_argument("--seed", type=int, required=True); source.add_argument("--output", required=True)
    source.add_argument("--steps", type=int, default=800); source.add_argument("--eval-every", type=int, default=100)
    source.add_argument("--batch-size", type=int, default=4); source.add_argument("--device", default="cuda")
    source.set_defaults(func=lambda a: __import__("selection_benchmark.runner", fromlist=["train_source"]).train_source(a))
    run = commands.add_parser("run")
    run.add_argument("--workspace", required=True); run.add_argument("--artifacts", default=str(ARTIFACTS))
    run.add_argument("--receipt-file", required=True); run.add_argument("--index", type=int, required=True)
    run.add_argument("--source-checkpoint", required=True); run.add_argument("--deeplab-source", required=True)
    run.add_argument("--output", required=True); run.add_argument("--device", default="cuda")
    run.set_defaults(func=lambda a: __import__("selection_benchmark.runner", fromlist=["run_receipt"]).run_receipt(a))
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--workspace", required=True); evaluate.add_argument("--artifacts", default=str(ARTIFACTS))
    evaluate.add_argument("--run-root", required=True); evaluate.add_argument("--deeplab-source", required=True); evaluate.add_argument("--device", default="cuda")
    evaluate.set_defaults(func=lambda a: __import__("selection_benchmark.runner", fromlist=["evaluate_receipt"]).evaluate_receipt(a))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
