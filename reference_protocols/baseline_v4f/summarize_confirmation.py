"""Create a descriptive, paired summary of frozen v4f confirmation results."""
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHODS = {
    "uniform_patch": "confirmation_runs",
    "original_uniform": "original_uniform_runs",
    "original_quota": "original_quota_runs",
    "entropy": "entropy_runs",
    "ripu_patch_adapted": "ripu_runs",
}


def read(path):
    return json.loads(Path(path).read_text())


def mean_sd(values):
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    return mean, math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def main():
    source = {}
    for path in sorted((ROOT / "confirmation_runs/source_only").rglob("*.json")):
        item = read(path)
        source[(item["region"], item["arch"], item["train_seed"], item["query_seed"])] = item["metrics"]["miou_fixed"]
    records = []
    for method, folder in METHODS.items():
        for path in sorted((ROOT / folder).rglob("evaluation.json")):
            ev = read(path)
            ad = read(path.with_name("adapt_complete.json"))
            key = (ad["region"], ad["arch"], ad["train_seed"], ad["query_seed"])
            records.append({"method": method, "region": ad["region"], "arch": ad["arch"],
                            "train_seed": ad["train_seed"], "query_seed": ad["query_seed"],
                            "budget": ad["budget"], "miou_fixed": ev["metrics"]["miou_fixed"],
                            "source_only_miou_fixed": source[key]})
    if len(records) != 180:
        raise RuntimeError(f"expected 180 method evaluations, found {len(records)}")
    random = {(x["region"], x["arch"], x["train_seed"], x["query_seed"], x["budget"]): x["miou_fixed"]
              for x in records if x["method"] == "uniform_patch"}
    grouped = defaultdict(list)
    for row in records:
        key = (row["method"], row["region"], row["arch"], row["budget"])
        row["delta_vs_source"] = row["miou_fixed"] - row["source_only_miou_fixed"]
        row["delta_vs_uniform_patch"] = row["miou_fixed"] - random[(row["region"], row["arch"], row["train_seed"], row["query_seed"], row["budget"])]
        grouped[key].append(row)
    summary = []
    for key, values in sorted(grouped.items()):
        miou, miou_sd = mean_sd([x["miou_fixed"] for x in values])
        source_delta, source_sd = mean_sd([x["delta_vs_source"] for x in values])
        random_delta, random_sd = mean_sd([x["delta_vs_uniform_patch"] for x in values])
        summary.append({"method": key[0], "region": key[1], "arch": key[2], "budget": key[3], "n_paired_seeds": len(values),
                        "miou_fixed_mean": miou, "miou_fixed_sample_sd": miou_sd,
                        "delta_vs_source_mean": source_delta, "delta_vs_source_sample_sd": source_sd,
                        "delta_vs_uniform_patch_mean": random_delta, "delta_vs_uniform_patch_sample_sd": random_sd,
                        "wins_vs_uniform_patch": sum(x["delta_vs_uniform_patch"] > 0 for x in values)})
    out = {"status": "complete", "metric": "fixed eight-class mIoU; empty-union IoU=0",
           "interpretation_scope": "descriptive paired seed summary; three seed pairs are not independent regions",
           "methods": METHODS, "records": records, "summary": summary}
    report = ROOT / "reports/confirmation_method_summary.json"
    if report.exists():
        raise FileExistsError(report)
    report.write_text(json.dumps(out, indent=2) + "\n")
    lines = ["# v4f confirmation method summary", "", "All rows use three paired model/selection seeds. Deltas are percentage points.", "",
             "| Method | Region | Architecture | Budget | mIoU mean ± SD | Δ source | Δ uniform patch | Wins |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        f = lambda mean, sd: f"{100*mean:.2f} ± {100*sd:.2f}"
        lines.append(f"| {row['method']} | {row['region']} | {row['arch']} | {row['budget']} | "
                     f"{f(row['miou_fixed_mean'], row['miou_fixed_sample_sd'])} | "
                     f"{f(row['delta_vs_source_mean'], row['delta_vs_source_sample_sd'])} | "
                     f"{f(row['delta_vs_uniform_patch_mean'], row['delta_vs_uniform_patch_sample_sd'])} | "
                     f"{row['wins_vs_uniform_patch']}/3 |")
    md = ROOT / "reports/confirmation_method_summary.md"
    if md.exists():
        raise FileExistsError(md)
    md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"records": len(records), "summary_rows": len(summary), "report": str(report)}))


if __name__ == "__main__":
    main()
