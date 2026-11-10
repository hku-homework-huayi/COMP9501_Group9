#!/usr/bin/env python3
"""Audit saved evidence, independently recomputing predictions and reconstruction."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch

from ebsl.data import ROOT, load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/main")
    args = p.parse_args()
    out = ROOT / args.output
    config = json.loads((out / "config.json").read_text())
    paths = sorted((out / "records").glob("*.json"))
    rows = [json.loads(p.read_text()) for p in paths]
    expected = len(config["datasets"]) * (
        len(config["compression_ranks"]) * 7
        + len(config["seeds"])
        * len(config["train_per_class"])
        * (3 + 4 * len(config["classification_ranks"]))
        + 3
    )
    if len(rows) != expected:
        raise AssertionError(f"Incomplete experiments: {len(rows)}/{expected}")
    checks = {
        "expected_records": expected,
        "actual_records": len(rows),
        "permutation_models": 0,
        "monotone_histories": 0,
        "prediction_files": 0,
        "split_files": 0,
        "recomputed_classifications": 0,
        "recomputed_compression_models": 0,
        "max_rmse_discrepancy": 0.0,
        "capped_runs": [],
    }
    for path, r in zip(paths, rows):
        history = r.get("history", r.get("er_history", []))
        if history:
            objectives = np.array([h["sse"] for h in history])
            assert np.all(np.isfinite(objectives)) and np.all(np.diff(objectives) <= 1e-7)
            checks["monotone_histories"] += 1
        stop = r.get("stop_reason", r.get("er_stop_reason"))
        if stop == "max_iter":
            checks["capped_runs"].append(path.name)
        if r["phase"] == "classification":
            pred = np.load(out / "models" / (path.stem + "_predictions.npz"))
            assert int((pred["predictions"] == pred["targets"]).sum()) == r["correct"]
            assert abs((pred["predictions"] == pred["targets"]).mean() - r["accuracy"]) < 1e-12
            checks["prediction_files"] += 1
    for path in (out / "models").glob("*.npz"):
        d = np.load(path)
        if "perm" in d:
            assert np.array_equal(np.sort(d["perm"]), np.arange(len(d["perm"])))
            assert np.all(np.isfinite(d["u"])) and np.all(np.isfinite(d["v"]))
            checks["permutation_models"] += 1
    for path in (out / "splits").glob("*.npz"):
        d = np.load(path)
        a, b = d["train_indices"], d["test_indices"]
        assert not set(a) & set(b)
        assert len(np.unique(np.r_[a, b])) == len(a) + len(b)
        checks["split_files"] += 1
    # Reload all final models on CPU and recompute the main fixed-rank seed0
    # classification predictions independently using NumPy's explicit distances.
    torch.set_num_threads(4)
    for dataset in config["datasets"]:
        x, y, _ = load_dataset(dataset, config["histogram_equalization"])
        for path, r in zip(paths, rows):
            if r["dataset"] != dataset:
                continue
            if r["phase"] == "compression" and r["method"] != "PCA":
                d = np.load(out / "models" / (path.stem + ".npz"))
                u, v, perm = d["u"], d["v"], d["perm"]
                # Direct NumPy reconstruction in original coordinates; float64.
                z = x.reshape(len(x), -1)[:, perm].reshape(x.shape).astype(np.float64)
                rec = (
                    u.astype(np.float64)
                    @ (u.T.astype(np.float64) @ z @ v.astype(np.float64))
                    @ v.T.astype(np.float64)
                )
                rec = rec.reshape(len(x), -1)[:, np.argsort(perm)].reshape(x.shape)
                measured = float(np.sqrt(np.square(x - rec).sum() / len(x)))
                discrepancy = abs(measured - r["rmse"])
                checks["max_rmse_discrepancy"] = max(checks["max_rmse_discrepancy"], discrepancy)
                assert discrepancy < 1e-4, (path, measured, r["rmse"])
                checks["recomputed_compression_models"] += 1
            elif r["phase"] == "classification" and r["seed"] == 0 and r["rank"] == [8, 8]:
                d = np.load(out / "models" / (path.stem + ".npz"))
                split = np.load(out / "splits" / f"{dataset}_g{r['gallery']}_s0.npz")
                a, b = split["train_indices"], split["test_indices"]
                z = x.reshape(len(x), -1)[:, d["perm"]].reshape(x.shape).astype(np.float64)
                features = (d["u"].T.astype(np.float64) @ z @ d["v"].astype(np.float64)).reshape(
                    len(x), -1
                )
                train, test = features[a], features[b]
                distance = np.square(test[:, None, :] - train[None, :, :]).sum(-1)
                prediction = y[a][distance.argmin(1)]
                saved = np.load(out / "models" / (path.stem + "_predictions.npz"))
                assert np.array_equal(prediction, saved["predictions"]), path
                checks["recomputed_classifications"] += 1
    checks["status"] = "passed"
    (out / "verification.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
