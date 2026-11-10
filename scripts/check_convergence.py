#!/usr/bin/env python3
"""Additional convergence check for the single main experiment that hit its cap.

This is a separately labeled supplemental experiment. Main results remain the
predeclared 100-iteration-budget results and are never overwritten.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch

from ebsl.core import fit_er
from ebsl.data import ROOT, load_dataset, sha256


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    x, _, _ = load_dataset("orl")
    x = torch.tensor(x, device="cuda")
    start = time.perf_counter()
    model = fit_er(
        x,
        (3, 3),
        neighborhood="nn",
        max_iter=500,
        callback=lambda h: print(json.dumps(h), flush=True) if h["iteration"] % 25 == 0 else None,
    )
    folder = ROOT / "results/convergence_check"
    folder.mkdir(exist_ok=True)
    model.save(folder / "orl_glram_nn_er_d3.npz")
    initial = json.loads(
        (ROOT / "results/main/records/compression_orl_GLRAM-NN-ER_d3.json").read_text()
    )
    row = {
        "purpose": "Additional convergence check, not a replacement for the main fixed-budget record",
        "dataset": "orl",
        "method": "GLRAM-NN-ER",
        "d": 3,
        "max_iter": 500,
        "actual_iterations": len(model.history) - 1,
        "stop_reason": model.stop_reason,
        "rmse": model.history[-1]["rmse"],
        "main_rmse": initial["rmse"],
        "additional_relative_reduction_percent": 100
        * (1 - model.history[-1]["rmse"] / initial["rmse"]),
        "seconds": time.perf_counter() - start,
        "history": model.history,
        "device": torch.cuda.get_device_name(),
        "source_sha256": sha256(ROOT / "src/ebsl/core.py"),
    }
    (folder / "result.json").write_text(json.dumps(row, indent=2))
    print("COMPLETE", json.dumps({k: v for k, v in row.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()
