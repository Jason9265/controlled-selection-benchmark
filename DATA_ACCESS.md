# Data access and release boundary

## What is released

The release includes frozen experiment receipts, result summaries, protocol JSON,
record-level audits, hashes, figure-generation inputs, and a normalized 1062-row
score file. The score rows are sufficient to independently recompute all 228
condition cells, strongest-control results, fixed-control sensitivities, and the
absence of a universal named method reported in the paper.

## What is not released

No original image, pixel label, data archive, trained model, ImageNet weight, or
third-party source tree is included. The public release must retain this boundary.

## Recreating the ISPRS preparation

1. Obtain the official archives under their own access conditions: `3_Ortho_IRRG.zip`,
   `5_Labels_all_noBoundary.zip`, `ISPRS_semantic_labeling_Vaihingen.zip`, and
   `ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip`.
2. Place exactly one copy of each archive anywhere beneath a local data directory.
3. Run `selection-benchmark prepare --data-root DATA_ROOT --workspace WORKSPACE`.
   The command checks the archive SHA-256 values frozen in the release before
   materializing non-overlapping 512px crops.
4. Use the resulting data only under the original ISPRS terms. The released results
   use the public labelled portion for an internal cross-region evaluation; they are
   not ISPRS leaderboard scores.

The OpenEarthMap and LoveDA materials include the minimal score records required
for paper-level descriptive re-analysis, but not full training directories,
intermediate tensors, checkpoints, logs, imagery, or labels. Reproducing the
training process still requires the original datasets and local protocol setup;
source hashes are recorded with the derived summary.
Their original data and licences remain authoritative.
LoveDA is academic/non-commercial under its dataset licence; OpenEarthMap imagery
and labels retain their original attribution and licence requirements.
