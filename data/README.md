# Data

Run `python scripts/download_data.py` from the repository root to download the checksum-pinned public PIE mirror, ORL 64×64 file and source paper. PIE extraction requires `bsdtar` with RAR support.

Only this document and `manifest.json` are tracked. Raw archives, images and MAT files stay local. Public PIE has **68 people × 24 images**, not the paper's **63 × 20** subset; original pose/illumination IDs are unavailable. ORL is additional validation. Original grayscale FERET is not included and no FERET measurements are claimed.

The extra 11,554-image PIE archive inspected during acquisition is not used by the experiment loader.
