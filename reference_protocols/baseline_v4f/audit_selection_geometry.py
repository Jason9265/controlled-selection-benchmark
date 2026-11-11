"""Audit original coverage and projected-coordinate dispersion for frozen controls."""
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def main():
    manifest = read(ROOT / "confirmation/manifest.json")
    lookup = {region: target["pool"] for region, target in manifest["targets"].items()}
    methods = {"uniform_patch": ROOT / "confirmation_runs", "original_uniform": ROOT / "original_uniform_runs",
               "original_quota": ROOT / "original_quota_runs"}
    rows = []
    for method, folder in methods.items():
        for path in sorted(folder.rglob("adapt_complete.json")):
            item = read(path)
            region = item["region"]
            selected = [lookup[region][i] for i in item["indices"]]
            points = [x["center_xy"] for x in selected]
            pairs = [math.hypot(a[0]-b[0], a[1]-b[1]) for i, a in enumerate(points) for b in points[i+1:]]
            nearest = [min(math.hypot(a[0]-b[0], a[1]-b[1]) for j, b in enumerate(points) if j != i)
                       for i, a in enumerate(points)] if len(points) > 1 else []
            groups = [x["group"] for x in selected]
            rows.append({"method": method, "region": region, "arch": item["arch"], "train_seed": item["train_seed"],
                         "query_seed": item["query_seed"], "budget": item["budget"], "selected_patches": len(selected),
                         "unique_originals": len(set(groups)), "repeat_fraction": 1-len(set(groups))/len(groups),
                         "mean_pair_distance_m": sum(pairs)/len(pairs) if pairs else None,
                         "mean_nearest_distance_m": sum(nearest)/len(nearest) if nearest else None,
                         "epsg": sorted({x["epsg"] for x in selected})})
    # Selection is paired across architectures: each (region, method,
    # query_seed, budget) outcome is written once per architecture.  Geometry
    # is a property of the selected patches, so summarise independent query
    # selections rather than counting the same selection twice.
    independent_rows = []
    seen = set()
    for row in rows:
        selection_key = (row["method"], row["region"], row["query_seed"], row["budget"])
        if selection_key in seen:
            continue
        seen.add(selection_key)
        independent_rows.append(row)
    grouped = defaultdict(list)
    for row in independent_rows:
        grouped[(row["method"], row["region"], row["budget"])].append(row)
    summary = []
    for key, values in sorted(grouped.items()):
        summary.append({"method": key[0], "region": key[1], "budget": key[2], "n": len(values),
                        "unique_originals_mean": sum(x["unique_originals"] for x in values)/len(values),
                        "repeat_fraction_mean": sum(x["repeat_fraction"] for x in values)/len(values),
                        "mean_pair_distance_m": sum(x["mean_pair_distance_m"] for x in values)/len(values),
                        "mean_nearest_distance_m": sum(x["mean_nearest_distance_m"] for x in values)/len(values)})
    out = {"metric": "Euclidean distances between selected 256-patch centers in each region's projected CRS; never pooled across regions",
           "independence_unit": "one frozen selection per (method, region, query seed, budget); architecture duplicates removed from summary",
           "all_architecture_rows": rows, "independent_selection_rows": independent_rows, "summary": summary}
    destination = ROOT / "reports/selection_geometry_audit_v2.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(out, indent=2) + "\n")
    for row in summary:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
