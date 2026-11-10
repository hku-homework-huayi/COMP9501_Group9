# COMP9501 Group 9 - Element Rearrangement

**Enhancing Bilinear Subspace Learning by Element Rearrangement** (Xu et al., IEEE TPAMI, 2009) 的独立复现。实现 GLRAM、2DPCA、空间/特征邻域元素重排，以及 2DLDA、2DMFA 监督扩展。

An independent reproduction of **Enhancing Bilinear Subspace Learning by Element Rearrangement** (Xu et al., IEEE TPAMI, 2009), including GLRAM, 2DPCA, spatial/feature-neighborhood rearrangement, and the 2DLDA/2DMFA extensions.

> **范围 / Scope:** 这是公开数据上的方法复现，不是原论文表格的精确复现。公开 PIE 为 68 人 × 24 张，与原文 63 × 20 子集不同；原始灰度 FERET 未获得，ORL 为补充验证集。<br>
> This is a method reproduction on public data, not an exact replication of the published tables. Public PIE contains 68 people × 24 images rather than the original 63 × 20 subset. Original grayscale FERET was not obtained; ORL is supplemental validation.

## Reports and evidence / 报告与实验记录

- **[中英对照 PDF / Bilingual report](reports/DATA_9501_bilingual_reproduction_report.pdf)** - 12 pages, with paired explanations, tables, figures, limitations and references.
- **[中文 HTML 报告 / Chinese HTML report](reports/report_zh.html)** - download and open locally; figures are embedded.
- **[实验协议 / Reproduction protocol](docs/REPRODUCTION.md)** - equations, numerical conventions, data differences and selection rules.
- **[压缩结果 / Compression](results/main/compression.csv)**, **[分类结果 / Classification](results/main/classification.csv)**, **[独立重建 / Holdout reconstruction](results/main/holdout.csv)**.
- **[核验结果 / Verification](results/main/verification.json)** and **[统计汇总 / Summary](results/main/summary.json)**.

The recorded run used an **NVIDIA GeForce RTX 4090 D** and completed **462 main records**: 126 compression settings, 330 classification settings and 6 holdout reconstruction settings. Classification uses **five seeds**, with both predeclared feature shapes reported. Tests and independent numerical audits passed. A separate convergence check reaches identity after 134 accepted updates for the single main run capped at 100 iterations.

For four training images per person and fixed 8×8 features, mean test accuracy ± sample standard deviation is:

| Dataset | Method | Baseline | With LN rearrangement |
|---|---|---:|---:|
| Public PIE | 2DLDA | 84.37 ± 1.98% | 87.66 ± 0.84% |
| Public PIE | 2DMFA | 78.81 ± 2.81% | 86.50 ± 1.26% |
| ORL | 2DLDA | 85.17 ± 3.76% | 89.33 ± 2.79% |
| ORL | 2DMFA | 81.17 ± 3.72% | 87.25 ± 2.60% |

At rank d=5, NN rearrangement reduces GLRAM reconstruction RMSE by **7.31% on PIE** and **7.98% on ORL**. Not every classification setting improves; all negative results and vector baselines remain in the reports.

## Repository layout

```text
src/ebsl/              Bilinear models, shared permutations, supervised methods, data
scripts/               Download, training, auditing and report commands
configs/               Executed configuration and original-paper protocol reference
tests/                 Numerical and reproducibility tests
docs/                  Detailed methodology and publication notes
reports/               Bilingual PDF and Chinese HTML
results/main/          Metrics, histories, small models, predictions, splits and figures
results/convergence_check/  Supplemental convergence result
data/                  Provenance manifest; raw downloads are ignored
papers/                Citation and published Table 1; original PDF is downloaded locally
logs/                  Successful experiment and validation logs
```

Raw datasets, the third-party paper PDF, local environments, temporary files and SSH credentials are **not tracked**. Compact learned matrices, permutations, predictions and exact splits are tracked so the saved run can be audited after downloading the datasets.

## Setup

Python **3.10** is the measured runtime. CUDA is required for the supplied full-run configuration; numerical tests also have CPU coverage.

```bash
git clone https://github.com/hku-homework-huayi/COMP9501_Group9.git
cd COMP9501_Group9
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -e .
```

`requirements.txt` pins the CUDA 12.1 PyTorch wheel and the scientific packages used in the experiments. `bsdtar` (libarchive with RAR support) is required to extract the public PIE archive. On Ubuntu, the package is `libarchive-tools`.

```bash
.venv/bin/python scripts/download_data.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python scripts/verify_results.py
```

The downloader verifies fixed SHA-256 checksums. It fetches the original paper and public data from their cited sources rather than redistributing them in Git.

## Reproduce experiments

```bash
# Resume the recorded configuration and audit it.
bash scripts/reproduce.sh

# Train a fresh, separate run; keep the published measurements intact.
bash scripts/reproduce.sh results/fresh
```

Individual stages are also available:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
.venv/bin/python scripts/run_experiments.py --phase compression --output results/fresh
.venv/bin/python scripts/run_experiments.py --phase classification --output results/fresh
.venv/bin/python scripts/run_experiments.py --phase holdout --output results/fresh
.venv/bin/python scripts/verify_results.py --output results/fresh
.venv/bin/python scripts/build_report.py --output results/fresh
```

Completed records are resumable, and original completed runtime metadata is preserved. Changed configurations require a new output directory. The optional supplemental check is:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python scripts/check_convergence.py
```

## Reports and code checks

```bash
uv pip install --python .venv/bin/python -e '.[dev,reports]'
.venv/bin/ruff check src scripts tests
.venv/bin/ruff format --check src scripts tests
.venv/bin/python scripts/build_report.py --report reports/report_zh.html
.venv/bin/python scripts/build_bilingual_pdf.py
```

PDF generation on Linux uses DejaVu Sans and Droid Sans Fallback fonts (`fonts-dejavu-core` and `fonts-droid-fallback` on Ubuntu). Existing figures are sufficient to rebuild reports from a clone; training regenerates the ignored example-image arrays when needed.

## Provenance and interpretation

- **[Numerical conventions](docs/REPRODUCTION.md)** explain train-only permutations, null-space construction, regularization, RMSE/CR definitions and fixed parameter choices.
- **[Publication notes](docs/PUBLICATION.md)** distinguish the historical run from subsequent formatting and packaging work.
- Published values in **[paper_table1.json](papers/paper_table1.json)** are literature references, not local measurements.
- Data sources and their limits are recorded in **[data/manifest.json](data/manifest.json)**.
- Original paper/data rights remain with their respective holders; see **[NOTICE.md](NOTICE.md)**.

## Reference

Dong Xu, Shuicheng Yan, Stephen Lin, Thomas S. Huang, and Shih-Fu Chang. *Enhancing Bilinear Subspace Learning by Element Rearrangement*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 31(10):1913-1920, 2009. [DOI](https://doi.org/10.1109/TPAMI.2009.51) · [BibTeX](papers/citation.bib)
