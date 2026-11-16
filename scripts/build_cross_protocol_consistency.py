#!/usr/bin/env python3
"""Build normalized score rows and cross-protocol descriptive summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


CONTROL_METHODS = {
    "oem_v2": ["random", "uniform_random", "original_uniform_random"],
    "loveda_v2": ["random", "uniform_random", "original_uniform_random"],
    "v4f": ["uniform_patch", "original_uniform", "original_quota"],
    "isprs": ["patch_uniform", "original_uniform", "original_quota"],
}
CSV_FIELDS = [
    "layer", "method", "region_or_direction", "architecture", "budget_units",
    "budget_fraction", "train_or_model_seed", "selection_seed", "miou_fixed",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def classify(values: list[float]) -> tuple[str, bool]:
    value = mean(values)
    return ("win" if value > 0 else "loss" if value < 0 else "tie"), all(item > 0 for item in values)


def matrix_score_rows(layer: str, root: Path) -> tuple[list[dict], list[dict]]:
    rows, sources = [], []
    for source in sorted(root.glob("*/all_results.json")):
        payload = json.loads(source.read_text(encoding="utf-8"))
        for row in payload["runs"]:
            rows.append({
                "layer": layer, "method": row["strategy"],
                "region_or_direction": row["region"], "architecture": row["arch"],
                "budget_units": row["budget"], "budget_fraction": row["planned_fraction"],
                "train_or_model_seed": row["train_seed"], "selection_seed": row["selection_seed"],
                "miou_fixed": row["miou_fixed"],
            })
        sources.append({"path_hint": f"{layer}/{source.parent.name}/all_results.json", "sha256": digest(source)})
    return rows, sources


def v4f_score_rows(path: Path) -> tuple[list[dict], list[dict]]:
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    budgets = defaultdict(set)
    for row in records:
        budgets[row["region"]].add(row["budget"])
    fractions = {region: dict(zip(sorted(values), (0.01, 0.03, 0.05))) for region, values in budgets.items()}
    rows = [{
        "layer": "v4f", "method": row["method"], "region_or_direction": row["region"],
        "architecture": row["arch"], "budget_units": row["budget"],
        "budget_fraction": fractions[row["region"]][row["budget"]],
        "train_or_model_seed": row["train_seed"], "selection_seed": row["query_seed"],
        "miou_fixed": row["miou_fixed"],
    } for row in records]
    return rows, [{"path_hint": "v4f/confirmation_method_summary.json", "sha256": digest(path)}]


def isprs_score_rows(control_path: Path, ripu_path: Path) -> tuple[list[dict], list[dict]]:
    rows = []
    for path in (control_path, ripu_path):
        for row in json.loads(path.read_text(encoding="utf-8"))["records"]:
            rows.append({
                "layer": "isprs", "method": row["method"],
                "region_or_direction": row["direction"], "architecture": "deeplabv3plus_resnet101",
                "budget_units": row["budget_crops"], "budget_fraction": row["budget_fraction"],
                "train_or_model_seed": row["model_seed"], "selection_seed": row["selection_seed"],
                "miou_fixed": row["miou_fixed"],
            })
    sources = [
        {"path_hint": "isprs/control_summary.json", "sha256": digest(control_path)},
        {"path_hint": "isprs/ripu_summary.json", "sha256": digest(ripu_path)},
    ]
    return rows, sources


def write_score_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: tuple(row[name] for name in CSV_FIELDS[:-1]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_score_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["budget_units"] = int(row["budget_units"])
        row["budget_fraction"] = float(row["budget_fraction"])
        row["train_or_model_seed"] = int(row["train_or_model_seed"])
        row["selection_seed"] = int(row["selection_seed"])
        row["miou_fixed"] = float(row["miou_fixed"])
    return rows


def condition_cells(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (row["layer"], row["region_or_direction"], row["architecture"], row["budget_fraction"],
               row["budget_units"], row["train_or_model_seed"], row["selection_seed"])
        by_key[key][row["method"]] = row["miou_fixed"]

    cells = []
    for layer, controls in CONTROL_METHODS.items():
        layer_rows = [row for row in rows if row["layer"] == layer]
        structured = sorted({row["method"] for row in layer_rows} - set(controls))
        cell_keys = sorted({
            (row["method"], row["region_or_direction"], row["architecture"], row["budget_fraction"], row["budget_units"])
            for row in layer_rows if row["method"] in structured
        })
        for method, region, architecture, fraction, units in cell_keys:
            seed_rows = sorted((key, scores) for key, scores in by_key.items()
                               if key[:5] == (layer, region, architecture, fraction, units))
            differences, seed_pairs = [], []
            fixed = {control: [] for control in controls}
            for key, scores in seed_rows:
                differences.append(100.0 * (scores[method] - max(scores[name] for name in controls)))
                for control in controls:
                    fixed[control].append(100.0 * (scores[method] - scores[control]))
                seed_pairs.append({"train_or_model_seed": key[5], "selection_seed": key[6]})
            outcome, consistent = classify(differences)
            cells.append({
                "layer": layer, "method": method, "region_or_direction": region,
                "architecture": architecture, "budget_units": units, "budget_fraction": fraction,
                "seed_pairs": seed_pairs, "differences_vs_strongest_control_pp": differences,
                "differences_by_fixed_control_pp": fixed, "mean_difference_pp": mean(differences),
                "mean_outcome": outcome, "three_seed_consistent_win": consistent,
            })
    return cells


def summarize(cells: list[dict]) -> dict:
    result = {}
    for layer, controls in CONTROL_METHODS.items():
        rows = [row for row in cells if row["layer"] == layer]
        method_summary = {}
        for method in sorted({row["method"] for row in rows}):
            method_rows = [row for row in rows if row["method"] == method]
            method_summary[method] = {
                "condition_cells": len(method_rows),
                "mean_wins_losses_ties": [sum(row["mean_outcome"] == name for row in method_rows) for name in ("win", "loss", "tie")],
                "three_seed_consistent_wins": sum(row["three_seed_consistent_win"] for row in method_rows),
            }
        fixed_summary = {}
        for control in controls:
            outcomes = [classify(row["differences_by_fixed_control_pp"][control]) for row in rows]
            fixed_summary[control] = {
                "mean_wins_losses_ties": [sum(outcome == name for outcome, _ in outcomes) for name in ("win", "loss", "tie")],
                "three_seed_consistent_wins": sum(consistent for _, consistent in outcomes),
                "universal_methods": [method for method in method_summary if all(
                    classify(row["differences_by_fixed_control_pp"][control])[1]
                    for row in rows if row["method"] == method)],
            }
        result[layer] = {
            "structured_methods": len(method_summary), "condition_cells": len(rows),
            "mean_wins_losses_ties": [sum(row["mean_outcome"] == name for row in rows) for name in ("win", "loss", "tie")],
            "three_seed_consistent_wins": sum(row["three_seed_consistent_win"] for row in rows),
            "mean_difference_range_pp": [min(row["mean_difference_pp"] for row in rows), max(row["mean_difference_pp"] for row in rows)],
            "method_summary": method_summary, "fixed_control_sensitivity": fixed_summary,
            "universal_methods": [method for method in method_summary if all(
                row["three_seed_consistent_win"] for row in rows if row["method"] == method)],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oem-root", type=Path, required=True)
    parser.add_argument("--loveda-root", type=Path, required=True)
    parser.add_argument("--v4f-summary", type=Path, required=True)
    parser.add_argument("--isprs-control-summary", type=Path, required=True)
    parser.add_argument("--isprs-ripu-summary", type=Path, required=True)
    parser.add_argument("--isprs-audit", type=Path, required=True)
    parser.add_argument("--score-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows, sources = [], []
    for layer, root in (("oem_v2", args.oem_root), ("loveda_v2", args.loveda_root)):
        layer_rows, layer_sources = matrix_score_rows(layer, root)
        rows.extend(layer_rows); sources.extend(layer_sources)
    layer_rows, layer_sources = v4f_score_rows(args.v4f_summary)
    rows.extend(layer_rows); sources.extend(layer_sources)
    layer_rows, layer_sources = isprs_score_rows(args.isprs_control_summary, args.isprs_ripu_summary)
    rows.extend(layer_rows); sources.extend(layer_sources)
    sources.append({"path_hint": "isprs/final_evidence_audit.json", "sha256": digest(args.isprs_audit)})

    if args.score_output:
        write_score_rows(args.score_output, rows)
        rows = read_score_rows(args.score_output)
    cells = condition_cells(rows)
    output = {
        "schema_version": 2,
        "analysis_scope": "Post-completion descriptive harmonization; not preregistered and not a hypothesis or equivalence test.",
        "estimand": "For each structured-method condition cell and paired seed, method mIoU minus the strongest of three controls in that seed.",
        "rules": {
            "condition_cell": "method x region/direction x architecture x budget; cells share data and models and are not independent samples",
            "mean_outcome": "win/loss/tie from the sign of the mean of three paired-seed differences; exact zero is a tie",
            "three_seed_consistent_win": "all three paired-seed differences are strictly positive",
            "universal_method": "the same named method is a three-seed-consistent win in every one of its tested condition cells",
            "strongest_control": "post-completion conservative attribution envelope, not a deployable baseline",
            "equivalence_margin": None, "cross_protocol_pooling": False,
        },
        "controls_by_layer": CONTROL_METHODS, "source_provenance": sources,
        "score_rows": None if not args.score_output else {
            "path": args.score_output.name, "rows": len(rows), "sha256": digest(args.score_output), "fields": CSV_FIELDS},
        "summary_by_layer": summarize(cells), "condition_cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "score_output": str(args.score_output) if args.score_output else None,
                      "score_rows": len(rows), "condition_cells": len(cells),
                      "summary_by_layer": output["summary_by_layer"]}, indent=2))


if __name__ == "__main__":
    main()
