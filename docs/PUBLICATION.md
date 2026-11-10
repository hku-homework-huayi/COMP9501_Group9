# Publication and provenance notes

The measured experiment is the 2026-11-10 run on an RTX 4090 D. Its metrics, predictions, projections, permutations, splits and objective histories are retained as scientific evidence.

Repository preparation standardizes Python formatting and imports, adds optional development/report dependencies, groups reports under `reports/`, and removes machine-specific paths from report reproduction instructions. It does not change the numerical algorithms or fabricate/reselect experimental results.

`results/main/environment_*.json` and `results/main/final_source_manifest.json` describe files present at the original run or reporting stage. Their hashes are historical snapshots, not a claim that later formatted files have the same bytes. The publication commit identifies the maintained code version. Publication checks are saved separately from historical experiment logs.

The original 100-iteration ORL / GLRAM-NN-ER / d=3 record is preserved. The separately labeled supplemental check reaches identity after 134 accepted updates, reducing RMSE by an additional 0.0433%.

SSH authentication is configured outside the checkout; no private keys or tokens are included. Raw dataset archives, extracted images and the original paper are downloaded locally with checksum verification. Report figures contain the selected research examples already discussed in the report.

No official implementation was located in the publication/repository sources searched during the reproduction. This is an independent implementation; see `REPRODUCTION.md` for data and methodological differences.
