# Controlled Selection Benchmark

This is the auditable companion package for *Do Acquisition-Level Structured
Query Rules Consistently Help?*
It supports a conditional finding: across the recorded frozen protocols, no tested
structured selector showed a stable advantage over the specified patch-random and
original-image controls. It does **not** propose a new method or establish a
general method-effectiveness claim.

## Quick integrity check: no data or GPU required

```bash
python selection-benchmark validate-artifacts
python selection-benchmark audit
python selection-benchmark summarize --output outputs/rebuilt
```

The first command verifies 648 OEM saved-record adaptations, 162 LoveDA
saved-record adaptations, 180 v4f confirmation records, 72 ISPRS blind-holdout
scores, and the 54/18 ISPRS receipt sets. The second recreates the released
audit result. The third recreates the released pairwise table and copies the
data-linked figure assets into `outputs/rebuilt`.

`artifacts/evidence/cross_protocol_score_rows.csv` releases the 1062 minimal
fixed-class mIoU records used by the paper (648 OEM, 162 LoveDA, 180 v4f, and
72 ISPRS). Its fields identify the layer, named method, region/direction,
architecture, absolute and fractional budget, paired seeds, and score. It
contains no imagery, labels, weights, checkpoints, or private paths.

`artifacts/evidence/cross_protocol_consistency_summary.json` is rebuilt from
that CSV and contains 228 condition cells. It reports the strongest-control
envelope, each named method, and sensitivity to each of three fixed controls.
The layers remain separate; cells share data and models and are descriptive
conditions rather than independent samples. This is a post-completion
harmonization, not a preregistered hypothesis or equivalence test.

The builder accepts `--score-output` to write the normalized CSV before reading
it back to compute the summary. Source hashes bind the public records to the
saved inputs. Full OEM and LoveDA training directories are not redistributed,
but their paper-level descriptive results are independently recomputable from
the released score rows.

```bash
python scripts/build_cross_protocol_consistency.py \
  --oem-root OEM_RUN_ROOT --loveda-root LOVEDA_RUN_ROOT \
  --v4f-summary V4F_SUMMARY.json \
  --isprs-control-summary CONTROL_SUMMARY.json \
  --isprs-ripu-summary RIPU_SUMMARY.json \
  --isprs-audit FINAL_EVIDENCE_AUDIT.json \
  --score-output artifacts/evidence/cross_protocol_score_rows.csv \
  --output artifacts/evidence/cross_protocol_consistency_summary.json
```

The released ISPRS figure shows the three paired seed-level mIoU values as
filled markers at each direction and budget; the companion table retains the
mean and sample standard deviation. The protocol figure records the frozen,
label-blind selection boundary and one-time target-holdout scoring flow.

## Dataset-dependent ISPRS reproduction

After obtaining official archives yourself, use a separate local workspace:

```bash
python selection-benchmark prepare --data-root DATA_ROOT --workspace WORKSPACE
python selection-benchmark freeze --workspace WORKSPACE --output WORKSPACE/control_receipts.json
```

To verify random receipt reconstruction before obtaining data, pass an empty
workspace with `--released-manifest`; this uses only the released crop metadata
and cannot be used for training.


For training, clone the pinned MIT-licensed VainF DeepLab repository and obtain the
ImageNet ResNet-101 initialization weight independently. Train source models before
running an individual receipt:

```bash
python selection-benchmark train-source --workspace WORKSPACE --direction potsdam_to_vaihingen \
  --seed 2026 --steps 800 --deeplab-source VAINF_SOURCE \
  --imagenet-resnet101 RESNET101_WEIGHT --output WORKSPACE/source_p2v_2026
python selection-benchmark run --workspace WORKSPACE --receipt-file WORKSPACE/control_receipts.json \
  --index 0 --source-checkpoint WORKSPACE/source_p2v_2026/model_best_source_val.pt \
  --deeplab-source VAINF_SOURCE --output WORKSPACE/runs/receipt_0
```

Holdout scoring is deliberately separate and refuses to overwrite a result:

```bash
python selection-benchmark evaluate --workspace WORKSPACE --run-root WORKSPACE/runs/receipt_0 \
  --deeplab-source VAINF_SOURCE
```

Use `freeze --method ripu_acquisition_only` only to recover the released frozen
label-blind RIPU receipt set. The paper calls it an acquisition-only adaptation,
not a full RIPU reproduction.

## Layout

- `artifacts/`: public audit inputs, frozen receipts, summaries, and figures.
- `selection_benchmark/`: command implementation.
- `reference_protocols/`: source code and frozen settings for OEM/LoveDA v2 and
  v4f archival evidence; it contains no data, checkpoints, or result directories.
- `scripts/build_cross_protocol_consistency.py`: builds the normalized 1062-row
  score file and derives the 228-cell envelope and fixed-control summaries.

Read [DATA_ACCESS.md](DATA_ACCESS.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
before using the package. Version `v0.1.0` is archived at
[doi:10.5281/zenodo.24137437](https://doi.org/10.5281/zenodo.24137437); the
GitHub Release and Zenodo record distribute the same tagged tarball.
