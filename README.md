# Controlled Selection Benchmark

This is the auditable companion package for *When Do Structured Query Rules Help?*
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

Read [DATA_ACCESS.md](DATA_ACCESS.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
before using the package. Create a GitHub release from this directory, then archive
that exact release tarball in Zenodo and add the version-specific DOI to
`CITATION.cff`.
