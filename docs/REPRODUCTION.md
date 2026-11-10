# Reproduction protocol and numerical conventions

This document describes the measured 2026-11-10 experiment. Publication cleanup formats the code and improves packaging; saved measurements retain their original provenance.

## Method mapping

| Paper component | Implementation |
|---|---|
| Eq. (1), GLRAM / CSA | `src/ebsl/core.py:fit_bilinear`, alternating eigendecompositions |
| 2DPCA | Same objective with `U=I`, a single right projection `V` |
| Eq. (5), matching cost | Sum of squared source/reconstructed-target differences over **all training samples** |
| Eq. (8), LN | A clipped 5×5 neighborhood (98,596 allowed edges for 64×64), including identity |
| Eq. (10), NN | Identity plus 24 smallest Eq. (5) costs; rebuilt every iteration |
| Integral transport | SciPy sparse minimum-weight full bipartite matching; same optimal assignment objective as the paper's LP |
| Algorithm 1 | Warm-started projections, composed single shared permutation, objective and bijection checks |
| Eq. (14) | Null projection as `x - (x Qᵀ) Q`, where Q spans within-class residuals |
| Sec. 3.1 | ER on weighted centered projected class means, then 2DLDA on rearranged original training images |
| Sec. 3.2 | ER on projected between-class margin differences, k2=25 per class, then 2DMFA |
| Sec. 3.3 | Euclidean 1-nearest-neighbor in the learned feature space |

The permutation is **shared across every image**, learned only from the fit set, and frozen for test images. It never sorts each image independently. Reconstruction always applies the inverse permutation before comparison/visualization.

## Explicit implementation choices and protocol differences

- Images are 64×64 without resizing. Histogram equalization uses 256-bin CDF-min with uint8 rounding, then `x / 255 - 0.5`. Eye alignment follows the downloaded crops; landmarks are unavailable. The paper does not specify its histogram bin count or eye template.
- Bilinear compression directly fits the normalized matrices in Eq. (1), without another across-sample mean subtraction. Standard vector PCA is centered. The distinction is recorded because the paper refers to a zero-mean assumption in its description of 2DPCA but only explicitly states `/255 - 0.5` in the experiment section.
- Sparse exact matching replaces the original simplex implementation. CUDA performs dense matrix algebra and cost construction; CPU performs sparse matching and small generalized eigensystems. FP32 training, FP64 objective accumulation/eigensystems, TF32 disabled.
- GLRAM starts from truncated identity matrices; its previous factors warm-start each ER update. Stops at identity permutation, relative objective improvement ≤1e-7, numerical plateau, or a logged 100-iteration cap. Inner GLRAM uses 100 iterations / 1e-7 tolerance. A capped run is not labeled converged.
- 2DLDA/2DMFA use 10 alternating generalized eigensystem updates, unit Euclidean column normalization and a relative within-scatter ridge of 1e-6. These numerical conventions are exposed because the paper does not specify singular-matrix handling.
- For supervised ER, the proxy fit uses the same rank as the following classifier. This paper detail is insufficiently specified, so the choice is explicit.
- Classification uses **five saved stratified seeds (0–4)** and train-per-class counts 4, 5, 6. PIE therefore has 20, 19, 18 test images per class, and ORL has 6, 5, 4. These are not the paper's PIE G4/P16, G5/P15, G6/P14 settings.
- We report **both predeclared ranks 8×8 and 12×8**, with mean and sample standard deviation. No test labels select ranks, neighbors, or stopping points. The paper reports best test accuracy over a larger dimension/k2 search, so our scores cannot be compared as a direct reproduction of those numbers.
- The vector baselines use training-only 95%-energy PCA; 1DLDA/1DMFA retain `min(PCA dimension, number of classes − 1)` features. Their parameters are not selected using test accuracy.
- Compression is fitted on all images, as a dataset compression experiment. A separate train-only G6 seed42 holdout experiment assesses unseen-image reconstruction.
- Paper RMSE is `sqrt(sum_i ||X_i-Xhat_i||_F²/N)`. Per-pixel RMSE is also recorded and equals paper RMSE/64. CR is a theoretical scalar-storage ratio, not an actual image codec bitrate. The additional single shared permutation is charged using the paper's 2-bit header + 8/12-bit offset scheme. PCA CR follows the paper formula, which excludes the mean vector; artifacts include it.

## Data and provenance

`scripts/download_data.py` verifies fixed SHA-256 values before use. `data/manifest.json` records sources and limitations. The larger 11,554-image public PIE JPEG archive in `data/pie_public/CMU_PIE_Face/` was inspected during acquisition but **is not used in the reported experiments**, since it lacks class/illumination metadata. Source paper and raw archives remain under their original rights; local copies are for the requested research reproduction, with no redistribution or new license claim.

Prepared custom data can be loaded as `.npz` with `images` (N×H×W, uint8), `labels` (N,), optional `sample_ids` (unicode). Set the dataset name/path in a new config or `--dataset path/to/data.npz`. This accepts aligned original datasets once available; reproducing FERET's named ba/bc/bh split also requires a named-image split adapter, not the random split used by the public-data runner.

## Source

- [Journal paper supplied by the user](https://www.ee.columbia.edu/ln/dvmm/publications/09/xu_ebsl.pdf)
- [Public CMU PIE mirror](https://github.com/Gunther17/CMU_PIE-dataset)
- [Deng Cai's face data page](http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html)
- [NIST FERET availability and access information](https://www.nist.gov/itl/products-and-services/color-feret-database)
