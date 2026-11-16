#!/usr/bin/env python3
"""Build the paper's post-completion cross-protocol consistency summary.

The released JSON contains derived condition-cell comparisons only. The full OEM
and LoveDA run matrices are inputs to this script but are not redistributed.
"""

from __future__ import annotations

import argparse
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


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def classify(values: list[float]) -> tuple[str, bool]:
    value = mean(values)
    outcome = "win" if value > 0 else "loss" if value < 0 else "tie"
    return outcome, all(item > 0 for item in values)


def raw_matrix_cells(layer: str, root: Path) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    sources: list[dict] = []
    for source in sorted(root.glob("*/all_results.json")):
        payload = json.loads(source.read_text(encoding="utf-8"))
        records.extend(payload["runs"])
        sources.append({"path_hint": f"{layer}/{source.parent.name}/all_results.json", "sha256": digest(source)})

    controls = set(CONTROL_METHODS[layer])
    by_key: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in records:
        key = (row["region"], row["arch"], row["budget"], row["train_seed"], row["selection_seed"])
        by_key[key][row["strategy"]] = row["miou_fixed"]

    structured = sorted({row["strategy"] for row in records} - controls)
    cell_rows = []
    cell_keys = sorted({(row["strategy"], row["region"], row["arch"], row["budget"])
                        for row in records if row["strategy"] in structured})
    for method, region, arch, budget in cell_keys:
        seed_rows = sorted(
            (key, scores) for key, scores in by_key.items()
            if key[:3] == (region, arch, budget)
        )
        differences = []
        seeds = []
        for key, scores in seed_rows:
            strongest = max(scores[name] for name in controls)
            differences.append(100.0 * (scores[method] - strongest))
            seeds.append({"train_seed": key[3], "selection_seed": key[4]})
        outcome, consistent = classify(differences)
        cell_rows.append({
            "layer": layer,
            "method": method,
            "region_or_direction": region,
            "architecture": arch,
            "budget": budget,
            "seed_pairs": seeds,
            "differences_vs_strongest_control_pp": differences,
            "mean_difference_pp": mean(differences),
            "mean_outcome": outcome,
            "three_seed_consistent_win": consistent,
        })
    return cell_rows, sources


def v4f_cells(path: Path) -> tuple[list[dict], list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"]
    controls = set(CONTROL_METHODS["v4f"])
    by_key: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in records:
        key = (row["region"], row["arch"], row["budget"], row["train_seed"], row["query_seed"])
        by_key[key][row["method"]] = row["miou_fixed"]
    structured = sorted({row["method"] for row in records} - controls)
    cells = []
    for method, region, arch, budget in sorted({
        (row["method"], row["region"], row["arch"], row["budget"])
        for row in records if row["method"] in structured
    }):
        seed_rows = sorted((key, scores) for key, scores in by_key.items() if key[:3] == (region, arch, budget))
        differences = [100.0 * (scores[method] - max(scores[name] for name in controls)) for _, scores in seed_rows]
        outcome, consistent = classify(differences)
        cells.append({
            "layer": "v4f", "method": method, "region_or_direction": region,
            "architecture": arch, "budget": budget,
            "seed_pairs": [{"train_seed": key[3], "selection_seed": key[4]} for key, _ in seed_rows],
            "differences_vs_strongest_control_pp": differences,
            "mean_difference_pp": mean(differences), "mean_outcome": outcome,
            "three_seed_consistent_win": consistent,
        })
    return cells, [{"path_hint": "v4f/confirmation_method_summary.json", "sha256": digest(path)}]


def isprs_cells(path: Path) -> tuple[list[dict], list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in payload["paired_comparisons"]:
        grouped[(row["direction"], row["budget_fraction"])][row["control_method"]] = row
    cells = []
    for (direction, budget), controls in sorted(grouped.items()):
        differences = [100.0 * min(controls[name]["paired_differences"][index] for name in CONTROL_METHODS["isprs"])
                       for index in range(3)]
        outcome, consistent = classify(differences)
        cells.append({
            "layer": "isprs", "method": "ripu_acquisition_only",
            "region_or_direction": direction, "architecture": "deeplabv3plus_resnet101",
            "budget": budget, "seed_pairs": [{"pair_index": index + 1} for index in range(3)],
            "differences_vs_strongest_control_pp": differences,
            "mean_difference_pp": mean(differences), "mean_outcome": outcome,
            "three_seed_consistent_win": consistent,
        })
    return cells, [{"path_hint": "isprs/final_evidence_audit.json", "sha256": digest(path)}]


def summarize(cells: list[dict]) -> dict:
    result = {}
    for layer in ("oem_v2", "loveda_v2", "v4f", "isprs"):
        rows = [row for row in cells if row["layer"] == layer]
        counts = {name: sum(row["mean_outcome"] == name for row in rows) for name in ("win", "loss", "tie")}
        methods = sorted({row["method"] for row in rows})
        universal = [method for method in methods if all(
            row["three_seed_consistent_win"] for row in rows if row["method"] == method
        )]
        result[layer] = {
            "condition_cells": len(rows),
            "mean_wins_losses_ties": [counts["win"], counts["loss"], counts["tie"]],
            "three_seed_consistent_wins": sum(row["three_seed_consistent_win"] for row in rows),
            "mean_difference_range_pp": [min(row["mean_difference_pp"] for row in rows),
                                         max(row["mean_difference_pp"] for row in rows)],
            "universal_methods": universal,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oem-root", type=Path, required=True)
    parser.add_argument("--loveda-root", type=Path, required=True)
    parser.add_argument("--v4f-summary", type=Path, required=True)
    parser.add_argument("--isprs-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cells: list[dict] = []
    sources: list[dict] = []
    for layer, root in (("oem_v2", args.oem_root), ("loveda_v2", args.loveda_root)):
        layer_cells, layer_sources = raw_matrix_cells(layer, root)
        cells.extend(layer_cells); sources.extend(layer_sources)
    layer_cells, layer_sources = v4f_cells(args.v4f_summary)
    cells.extend(layer_cells); sources.extend(layer_sources)
    layer_cells, layer_sources = isprs_cells(args.isprs_audit)
    cells.extend(layer_cells); sources.extend(layer_sources)

    output = {
        "schema_version": 1,
        "analysis_scope": "Post-completion descriptive harmonization; not preregistered and not a hypothesis or equivalence test.",
        "estimand": "For each structured-method condition cell and paired seed, method mIoU minus the strongest of three controls in that seed.",
        "rules": {
            "mean_outcome": "win/loss/tie from the sign of the mean of three paired-seed differences; exact zero is a tie",
            "three_seed_consistent_win": "all three paired-seed differences are strictly positive",
            "equivalence_margin": None,
            "cross_protocol_pooling": False,
        },
        "controls_by_layer": CONTROL_METHODS,
        "source_provenance": sources,
        "summary_by_layer": summarize(cells),
        "condition_cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "condition_cells": len(cells),
                      "summary_by_layer": output["summary_by_layer"]}, indent=2))


if __name__ == "__main__":
    main()
