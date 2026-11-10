#!/usr/bin/env python3
"""Reproducible CUDA experiment runner. Completed JSON records are resumable."""

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import scipy
import torch

from ebsl.core import BilinearModel, compression_ratio, fit_baseline, fit_er
from ebsl.data import ROOT, load_dataset, sha256, stratified_split, write_manifest
from ebsl.supervised import (
    er_proxy,
    fit_discriminant,
    nearest_neighbor,
    nullspace_projection,
    vector_features,
)


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def model_load(path, device, dtype):
    d = np.load(path)
    return BilinearModel(
        torch.tensor(d["u"], device=device, dtype=dtype),
        torch.tensor(d["v"], device=device, dtype=dtype),
        d["perm"],
        tuple(d["shape"]),
    )


def er_options(config):
    return {
        "max_iter": config["er_max_iter"],
        "tol": config["er_tol"],
        "inner_max_iter": config["inner_max_iter"],
        "inner_tol": config["inner_tol"],
    }


def compression(dataset, x, out, config):
    n, h, w = x.shape
    records = out / "records"
    models = out / "models"
    for d in config["compression_ranks"]:
        for mode in ["glram", "2dpca"]:
            for hood in [None, "ln", "nn"]:
                name = mode.upper() if mode == "glram" else "2DPCA"
                if hood:
                    name += "-" + hood.upper() + "-ER"
                tag = f"compression_{dataset}_{name}_d{d}"
                path = records / (tag + ".json")
                if path.exists():
                    continue
                t = time.perf_counter()
                if hood:
                    model = fit_er(x, (d, d), mode, hood, **er_options(config))
                else:
                    model = fit_baseline(
                        x, (d, d), mode, max_iter=config["inner_max_iter"], tol=config["inner_tol"]
                    )
                cr, bits = compression_ratio(n, (h, w), (d, d), mode, model.perm if hood else None)
                loss = model.history[-1]["sse"]
                row = {
                    "phase": "compression",
                    "dataset": dataset,
                    "method": name,
                    "d": d,
                    "n": n,
                    "shape": [h, w],
                    "rmse": (loss / n) ** 0.5,
                    "pixel_rmse": (loss / n / h / w) ** 0.5,
                    "compression_ratio": cr,
                    "index_bits": bits,
                    "iterations": len(model.history) - 1,
                    "stop_reason": model.stop_reason,
                    "seconds": time.perf_counter() - t,
                    "history": model.history,
                    "moved_fraction": float(np.mean(model.perm != np.arange(h * w))),
                    "eigen_energy": model.history[-1]["eigen_energy"],
                }
                model.save(models / (tag + ".npz"))
                write_json(path, row)
                print(json.dumps({k: v for k, v in row.items() if k != "history"}), flush=True)
    # Standard centered vector PCA for reference, using the exact training SVD.
    missing = [
        d
        for d in config["compression_ranks"]
        if not (records / f"compression_{dataset}_PCA_d{d}.json").exists()
    ]
    if missing:
        t = time.perf_counter()
        flat = x.flatten(1)
        mean = flat.mean(0)
        centered = flat - mean
        _, s, vh = torch.linalg.svd(centered, full_matrices=False)
        for d in missing:
            rec = (centered @ vh[:d].T) @ vh[:d] + mean
            loss = float((flat - rec).double().square().sum())
            row = {
                "phase": "compression",
                "dataset": dataset,
                "method": "PCA",
                "d": d,
                "n": n,
                "shape": [h, w],
                "rmse": (loss / n) ** 0.5,
                "pixel_rmse": (loss / n / h / w) ** 0.5,
                "compression_ratio": n * h * w / (d * (n + h * w)),
                "index_bits": 0,
                "iterations": 0,
                "stop_reason": "svd",
                "seconds": time.perf_counter() - t,
                "history": [],
                "eigen_energy": float(s[:d].square().sum() / s.square().sum()),
            }
            np.savez_compressed(
                models / f"compression_{dataset}_PCA_d{d}.npz",
                basis=vh[:d].cpu().numpy(),
                mean=mean.cpu().numpy(),
            )
            write_json(records / f"compression_{dataset}_PCA_d{d}.json", row)
            print(json.dumps(row), flush=True)
    # Save eight fixed, evenly spaced example indices across the dataset.
    if 5 in config["compression_ranks"]:
        ids = np.linspace(0, n - 1, 8, dtype=int)
        example = {"original": x[ids].cpu().numpy(), "indices": ids}
        for name in ["GLRAM", "GLRAM-LN-ER", "GLRAM-NN-ER", "2DPCA", "2DPCA-LN-ER", "2DPCA-NN-ER"]:
            m = model_load(models / f"compression_{dataset}_{name}_d5.npz", x.device, x.dtype)
            example[name] = m.reconstruct(x[ids]).cpu().numpy()
            if name == "GLRAM-LN-ER":
                example["rearranged"] = m.rearrange(x[ids]).cpu().numpy()
        np.savez_compressed(out / f"examples_{dataset}.npz", **example)


def classification(dataset, x, labels, ids, out, config):
    records = out / "records"
    models = out / "models"
    splits = out / "splits"
    for gallery in config["train_per_class"]:
        for seed in config["seeds"]:
            train_idx, test_idx = stratified_split(labels, gallery, seed)
            split_tag = f"{dataset}_g{gallery}_s{seed}"
            np.savez_compressed(
                splits / (split_tag + ".npz"),
                train_indices=train_idx,
                test_indices=test_idx,
                train_ids=np.array(ids)[train_idx],
                test_ids=np.array(ids)[test_idx],
            )
            train, test = x[train_idx], x[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            common = {
                "phase": "classification",
                "dataset": dataset,
                "gallery": gallery,
                "seed": seed,
                "n_train": len(train),
                "n_test": len(test),
                "classes": len(np.unique(labels)),
            }
            for kind in ["pca", "lda", "mfa"]:
                name = {"pca": "PCA95", "lda": "1DLDA", "mfa": "1DMFA"}[kind]
                tag = f"classification_{split_tag}_{name}"
                if (records / (tag + ".json")).exists():
                    continue
                t = time.perf_counter()
                a, b, dim = vector_features(train, test, y_train, kind, k2=config["mfa_k2"])
                pred = nearest_neighbor(a, y_train, b)
                row = {
                    **common,
                    "method": name,
                    "rank": [dim],
                    "accuracy": float(np.mean(pred == y_test)),
                    "correct": int(np.sum(pred == y_test)),
                    "seconds": time.perf_counter() - t,
                }
                np.savez_compressed(
                    models / (tag + "_predictions.npz"),
                    predictions=pred,
                    targets=y_test,
                    test_indices=test_idx,
                )
                write_json(records / (tag + ".json"), row)
                print(json.dumps(row), flush=True)
            required = []
            for ranks in config["classification_ranks"]:
                for kind in ["lda", "mfa"]:
                    tag = (
                        f"classification_{split_tag}_2D{kind.upper()}-LN-ER_r{ranks[0]}c{ranks[1]}"
                    )
                    if not (records / (tag + ".json")).exists():
                        required.append(tag)
            projected, diagnostic = nullspace_projection(train, y_train) if required else (None, {})
            proxies = (
                {
                    kind: er_proxy(train, y_train, kind, config["mfa_k2"], projected)
                    for kind in ["lda", "mfa"]
                }
                if required
                else {}
            )
            for ranks in config["classification_ranks"]:
                for kind in ["lda", "mfa"]:
                    for use_er in [False, True]:
                        name = "2D" + kind.upper() + ("-LN-ER" if use_er else "")
                        tag = f"classification_{split_tag}_{name}_r{ranks[0]}c{ranks[1]}"
                        path = records / (tag + ".json")
                        if path.exists():
                            continue
                        t = time.perf_counter()
                        er = None
                        if use_er:
                            er = fit_er(
                                proxies[kind], tuple(ranks), neighborhood="ln", **er_options(config)
                            )
                            er.save(models / (tag + "_er.npz"))
                        model = fit_discriminant(
                            train,
                            y_train,
                            tuple(ranks),
                            kind,
                            permutation=er.perm if er else None,
                            iterations=config["discriminant_iterations"],
                            ridge=config["ridge"],
                            k2=config["mfa_k2"],
                        )
                        pred = nearest_neighbor(
                            model.transform(train), y_train, model.transform(test)
                        )
                        row = {
                            **common,
                            "method": name,
                            "rank": ranks,
                            "accuracy": float(np.mean(pred == y_test)),
                            "correct": int(np.sum(pred == y_test)),
                            "seconds": time.perf_counter() - t,
                            "er_iterations": len(er.history) - 1 if er else 0,
                            "er_stop_reason": er.stop_reason if er else None,
                            "er_history": er.history if er else [],
                            "nullspace": diagnostic if er else None,
                        }
                        model.save(models / (tag + ".npz"))
                        np.savez_compressed(
                            models / (tag + "_predictions.npz"),
                            predictions=pred,
                            targets=y_test,
                            test_indices=test_idx,
                        )
                        write_json(path, row)
                        print(
                            json.dumps({k: v for k, v in row.items() if k != "er_history"}),
                            flush=True,
                        )


def holdout(dataset, x, labels, out, config):
    train_idx, test_idx = stratified_split(labels, 6, 42)
    train, test = x[train_idx], x[test_idx]
    for hood in [None, "ln", "nn"]:
        name = "GLRAM" + ("-" + hood.upper() + "-ER" if hood else "")
        tag = f"holdout_{dataset}_{name}_d5"
        path = out / "records" / (tag + ".json")
        if path.exists():
            continue
        t = time.perf_counter()
        m = (
            fit_er(train, (5, 5), neighborhood=hood, **er_options(config))
            if hood
            else fit_baseline(train, (5, 5))
        )
        loss = float((test - m.reconstruct(test)).double().square().sum())
        row = {
            "phase": "holdout",
            "dataset": dataset,
            "method": name,
            "d": 5,
            "train_per_class": 6,
            "seed": 42,
            "n_train": len(train),
            "n_test": len(test),
            "test_rmse": (loss / len(test)) ** 0.5,
            "train_rmse": m.history[-1]["rmse"],
            "seconds": time.perf_counter() - t,
            "stop_reason": m.stop_reason,
            "history": m.history,
        }
        m.save(out / "models" / (tag + ".npz"))
        np.savez_compressed(
            out / "splits" / f"holdout_{dataset}.npz",
            train_indices=train_idx,
            test_indices=test_idx,
        )
        write_json(path, row)
        print(json.dumps({k: v for k, v in row.items() if k != "history"}), flush=True)


def collect(out):
    all_rows = [json.loads(p.read_text()) for p in sorted((out / "records").glob("*.json"))]
    write_json(out / "all_results.json", all_rows)
    for phase in ["compression", "classification", "holdout"]:
        rows = [
            {
                k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                for k, v in r.items()
                if k not in ["history", "er_history"]
            }
            for r in all_rows
            if r["phase"] == phase
        ]
        if rows:
            fields = list(dict.fromkeys(k for r in rows for k in r))
            with (out / (phase + ".csv")).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/reproduction.json")
    p.add_argument(
        "--phase",
        choices=["all", "compression", "classification", "holdout", "collect"],
        default="all",
    )
    p.add_argument("--output", default="results/main")
    p.add_argument("--dataset", default=None)
    args = p.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    for d in ["records", "models", "splits"]:
        (out / d).mkdir(exist_ok=True)
    if args.phase == "collect":
        collect(out)
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    np.random.seed(0)
    dtype = {"float32": torch.float32, "float64": torch.float64}[config["dtype"]]
    device = config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    existing = out / "config.json"
    if existing.exists() and json.loads(existing.read_text()) != config:
        raise ValueError("Output directory belongs to a different configuration")
    write_json(existing, config)
    write_manifest()
    source_hashes = {
        str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src").rglob("*.py"))
    }
    metadata = {
        "started_local": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "device_total_memory": torch.cuda.get_device_properties(0).total_memory
        if device == "cuda"
        else None,
        "source_sha256": source_hashes,
        "runner_sha256": sha256(__file__),
        "config_sha256": sha256(ROOT / args.config),
        "deterministic_algorithms": True,
        "tf32": False,
        "command": sys.argv,
        "pid": os.getpid(),
    }
    metadata_path = out / f"environment_{args.phase}_{args.dataset or 'all'}.json"
    if metadata_path.exists() and "completed_local" in json.loads(metadata_path.read_text()):
        # Preserve the original completed run's timing/provenance on resume.
        (out / "invocations").mkdir(exist_ok=True)
        metadata_path = out / "invocations" / f"environment_{args.phase}_{os.getpid()}.json"
    write_json(metadata_path, metadata)
    t = time.perf_counter()
    for dataset in [args.dataset] if args.dataset else config["datasets"]:
        array, labels, ids = load_dataset(dataset, config["histogram_equalization"])
        x = torch.tensor(array, device=device, dtype=dtype)
        if args.phase in ["all", "compression"]:
            compression(dataset, x, out, config)
        if args.phase in ["all", "classification"]:
            classification(dataset, x, labels, ids, out, config)
        if args.phase in ["all", "holdout"]:
            holdout(dataset, x, labels, out, config)
        collect(out)
    metadata.update(
        completed_local=datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        total_seconds=time.perf_counter() - t,
        peak_torch_memory_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
    )
    write_json(metadata_path, metadata)
    print("COMPLETE", json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
