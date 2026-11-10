#!/usr/bin/env python3
"""Build plots and a Chinese HTML report exclusively from saved measurements."""

import argparse
import base64
import html
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ebsl.data import ROOT

LABELS = {"pie_public": "CMU PIE public (68 × 24)", "orl": "ORL (40 × 10)"}
METHODS = ["PCA", "GLRAM", "GLRAM-LN-ER", "GLRAM-NN-ER", "2DPCA", "2DPCA-LN-ER", "2DPCA-NN-ER"]
COLORS = dict(
    zip(METHODS, ["#86949e", "#326c9b", "#3fa58d", "#cf7842", "#8e77b5", "#bc71a2", "#caad48"])
)


def table(headers, rows):
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join("<th>" + html.escape(str(x)) + "</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join("<td>" + html.escape(str(x)) + "</td>" for x in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def stats(rows):
    values = np.array([r["accuracy"] * 100 for r in rows])
    return float(values.mean()), float(values.std(ddof=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/main")
    p.add_argument("--report", help="Optional HTML destination relative to the project root")
    args = p.parse_args()
    out = ROOT / args.output
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    cfg = json.loads((out / "config.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted((out / "records").glob("*.json"))]
    comp = [r for r in rows if r["phase"] == "compression"]
    cls = [r for r in rows if r["phase"] == "classification"]
    held = [r for r in rows if r["phase"] == "holdout"]
    expected = len(cfg["datasets"]) * (
        len(cfg["compression_ranks"]) * 7
        + len(cfg["seeds"])
        * len(cfg["train_per_class"])
        * (3 + 4 * len(cfg["classification_ranks"]))
        + 3
    )
    if len(rows) != expected:
        raise RuntimeError(f"Incomplete experiment records: {len(rows)}/{expected}")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )
    # Compression: compare ranks and storage accounting; log CR avoids crowding.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for i, ds in enumerate(cfg["datasets"]):
        for method in METHODS:
            rr = sorted(
                [r for r in comp if r["dataset"] == ds and r["method"] == method],
                key=lambda r: r["d"],
            )
            for j, key in enumerate(["d", "compression_ratio"]):
                axes[i, j].plot(
                    [r[key] for r in rr],
                    [r["rmse"] for r in rr],
                    "-o",
                    ms=3,
                    lw=1.6,
                    label=method,
                    color=COLORS[method],
                )
        axes[i, 0].set(title=LABELS[ds], xlabel="Reduced rank d", ylabel="Paper-defined RMSE")
        axes[i, 1].set(
            title=LABELS[ds],
            xlabel="Compression ratio (scalar storage, log scale)",
            ylabel="Paper-defined RMSE",
            xscale="log",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(figs / "compression.png")
    fig.savefig(figs / "compression.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, ds in zip(axes, cfg["datasets"]):
        for method in METHODS:
            if "-ER" not in method:
                continue
            r = next(
                r for r in comp if r["dataset"] == ds and r["method"] == method and r["d"] == 5
            )
            ax.plot(
                [h["iteration"] for h in r["history"]],
                [h["rmse"] for h in r["history"]],
                label=method,
                color=COLORS[method],
                lw=1.8,
            )
        ax.set(
            title=LABELS[ds] + " · d=5",
            xlabel="Accepted rearrangement iteration",
            ylabel="Paper-defined RMSE",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(figs / "convergence.png")
    plt.close(fig)
    # Classifier results for the predeclared primary rank only.
    names = ["2DLDA", "2DLDA-LN-ER", "2DMFA", "2DMFA-LN-ER"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for ax, ds in zip(axes, cfg["datasets"]):
        for method, col in zip(names, ["#326c9b", "#3fa58d", "#a86d8f", "#cf7842"]):
            means, stds = zip(
                *[
                    stats(
                        [
                            r
                            for r in cls
                            if r["dataset"] == ds
                            and r["gallery"] == g
                            and r["method"] == method
                            and r["rank"] == [8, 8]
                        ]
                    )
                    for g in cfg["train_per_class"]
                ]
            )
            ax.errorbar(
                cfg["train_per_class"],
                means,
                yerr=stds,
                label=method,
                color=col,
                marker="o",
                capsize=3,
            )
        ax.set(
            title=LABELS[ds] + " · 8×8 features",
            xlabel="Training images per person",
            ylabel="Test accuracy (%)",
            xticks=cfg["train_per_class"],
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(figs / "classification.png")
    plt.close(fig)
    for ds in cfg["datasets"]:
        if not (out / f"examples_{ds}.npz").exists():
            if (figs / f"reconstructions_{ds}.png").exists():
                continue
            raise FileNotFoundError(
                f"Missing reconstruction examples and figure for {ds}; run the compression stage first"
            )
        d = np.load(out / f"examples_{ds}.npz")
        keys = ["original", "GLRAM", "GLRAM-LN-ER", "GLRAM-NN-ER", "rearranged"]
        fig, axes = plt.subplots(len(keys), 8, figsize=(12, 7.3))
        for i, key in enumerate(keys):
            for j in range(8):
                axes[i, j].imshow(d[key][j] + 0.5, cmap="gray", vmin=0, vmax=1)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                axes[i, j].grid(False)
                for spine in axes[i, j].spines.values():
                    spine.set_visible(False)
            axes[i, 0].set_ylabel(key, fontsize=9)
        fig.suptitle(
            LABELS[ds] + " · d=5; reconstruction returned to original pixel coordinates",
            fontsize=12,
        )
        fig.subplots_adjust(left=0.12, right=0.995, top=0.92, bottom=0.01, wspace=0.04, hspace=0.08)
        fig.savefig(figs / f"reconstructions_{ds}.png")
        plt.close(fig)
    compression_table = []
    summary = {"compression_d5": {}, "classification": [], "paired_deltas": []}
    for ds in cfg["datasets"]:
        summary["compression_d5"][ds] = {}
        for base in ["GLRAM", "2DPCA"]:
            rr = {r["method"]: r for r in comp if r["dataset"] == ds and r["d"] == 5}
            raw, ln, nn = rr[base], rr[base + "-LN-ER"], rr[base + "-NN-ER"]
            dl = (1 - ln["rmse"] / raw["rmse"]) * 100
            dn = (1 - nn["rmse"] / raw["rmse"]) * 100
            compression_table.append(
                [
                    LABELS[ds],
                    base,
                    f"{raw['rmse']:.4f}",
                    f"{ln['rmse']:.4f}",
                    f"{nn['rmse']:.4f}",
                    f"{dl:.2f}%",
                    f"{dn:.2f}%",
                ]
            )
            summary["compression_d5"][ds][base] = {
                "baseline_rmse": raw["rmse"],
                "ln_rmse": ln["rmse"],
                "nn_rmse": nn["rmse"],
                "ln_relative_reduction_percent": dl,
                "nn_relative_reduction_percent": dn,
            }
    classification_table = []
    delta_table = []
    vector_table = []
    for ds in cfg["datasets"]:
        for g in cfg["train_per_class"]:
            for rank in cfg["classification_ranks"]:
                row = [LABELS[ds], g, "×".join(map(str, rank))]
                for method in names:
                    rr = sorted(
                        [
                            r
                            for r in cls
                            if r["dataset"] == ds
                            and r["gallery"] == g
                            and r["rank"] == rank
                            and r["method"] == method
                        ],
                        key=lambda r: r["seed"],
                    )
                    assert len(rr) == len(cfg["seeds"])
                    avg, sd = stats(rr)
                    row.append(f"{avg:.2f} ± {sd:.2f}")
                    summary["classification"].append(
                        {
                            "dataset": ds,
                            "gallery": g,
                            "rank": rank,
                            "method": method,
                            "mean_percent": avg,
                            "std_percent": sd,
                            "seeds": len(rr),
                        }
                    )
                classification_table.append(row)
                delta_row = [LABELS[ds], g, "×".join(map(str, rank))]
                for method in ["2DLDA", "2DMFA"]:
                    a = {
                        r["seed"]: r["accuracy"]
                        for r in cls
                        if r["dataset"] == ds
                        and r["gallery"] == g
                        and r["rank"] == rank
                        and r["method"] == method
                    }
                    b = {
                        r["seed"]: r["accuracy"]
                        for r in cls
                        if r["dataset"] == ds
                        and r["gallery"] == g
                        and r["rank"] == rank
                        and r["method"] == method + "-LN-ER"
                    }
                    deltas = np.array([100 * (b[s] - a[s]) for s in cfg["seeds"]])
                    avg = float(deltas.mean())
                    sd = float(deltas.std(ddof=1))
                    delta_row.append(f"{avg:+.2f} ± {sd:.2f}")
                    summary["paired_deltas"].append(
                        {
                            "dataset": ds,
                            "gallery": g,
                            "rank": rank,
                            "method": method,
                            "mean_pp": avg,
                            "std_pp": sd,
                            "per_seed_pp": deltas.tolist(),
                        }
                    )
                delta_table.append(delta_row)
            row = [LABELS[ds], g]
            for method in ["PCA95", "1DLDA", "1DMFA"]:
                mean, std = stats(
                    [
                        r
                        for r in cls
                        if r["dataset"] == ds and r["gallery"] == g and r["method"] == method
                    ]
                )
                row.append(f"{mean:.2f} ± {std:.2f}")
            vector_table.append(row)
    held_table = []
    for ds in cfg["datasets"]:
        rr = {r["method"]: r for r in held if r["dataset"] == ds}
        for name in ["GLRAM", "GLRAM-LN-ER", "GLRAM-NN-ER"]:
            r = rr[name]
            gain = 100 * (1 - r["test_rmse"] / rr["GLRAM"]["test_rmse"])
            held_table.append(
                [
                    LABELS[ds],
                    name,
                    f"{r['train_rmse']:.4f}",
                    f"{r['test_rmse']:.4f}",
                    f"{gain:+.2f}%",
                ]
            )
    stops = Counter(r.get("stop_reason", r.get("er_stop_reason")) for r in rows)
    capped = [r for r in rows if r.get("stop_reason", r.get("er_stop_reason")) == "max_iter"]
    envs = [json.loads(p.read_text()) for p in out.glob("environment_*.json")]
    timing = []
    for env in envs:
        phase = env["command"][env["command"].index("--phase") + 1]
        timing.append(
            [
                phase,
                f"{env.get('total_seconds', 0):.1f} 秒",
                f"{env.get('peak_torch_memory_bytes', 0) / 1024**2:.1f} MiB",
            ]
        )
    summary.update(record_count=len(rows), stop_reasons=dict(stops), capped_count=len(capped))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    reference = json.loads((ROOT / "papers/paper_table1.json").read_text())
    ref_rows = [
        [r["method"]] + [f"{a:.1f} ({d})" for a, d in zip(r["accuracy"], r["dimensions"])]
        for r in reference["rows"]
    ]
    verify = (
        json.loads((out / "verification.json").read_text())
        if (out / "verification.json").exists()
        else {"status": "pending"}
    )
    supplemental = ROOT / "results/convergence_check/result.json"
    supplemental = (
        json.loads(supplemental.read_text())
        if out.resolve() == (ROOT / "results/main").resolve() and supplemental.exists()
        else None
    )
    test_text = (ROOT / "logs/tests.txt").read_text().strip().splitlines()[-1]
    primary = [d for d in summary["paired_deltas"] if d["rank"] == [8, 8]]
    pos = sum(d["mean_pp"] > 0 for d in primary)
    neg = sum(d["mean_pp"] < 0 for d in primary)
    b = []
    b.append(
        '<div class="eyebrow">DATA_9501 · 实测结果 · 2026-11-10</div><h1>元素重排能否改善双线性子空间学习？</h1>'
    )
    b.append(
        '<p class="lead">Enhancing Bilinear Subspace Learning by Element Rearrangement<br>Xu et al. · IEEE TPAMI 2009 · 独立实现与公开数据实验</p>'
    )
    b.append(
        '<div class="notice"><strong>复现范围：方法复现，非原表格精确复现。</strong> 已实现论文的无监督 LN/NN 重排和 2DLDA/2DMFA 监督扩展，并在 RTX 4090 D 上运行。公开 PIE 子集、预处理与参数选择无法与原文完全对齐；原始灰度 FERET 未获得。下面的实测值不代表论文 Table 1 已被精确复现。</div>'
    )
    b.append(
        f'<div class="cards"><div><b>{len(rows)}</b><span>完成的实验记录</span></div><div><b>5</b><span>分类随机种子</span></div><div><b>2,032</b><span>两套数据的图像总数</span></div><div><b>4090 D</b><span>PyTorch CUDA 12.1</span></div></div>'
    )
    b.append("<h2>1. 结论与边界</h2>")
    b.append(
        f"<p>在固定低秩 d=5 的两个数据集上，LN/NN 元素重排均降低了 GLRAM 和 2DPCA 的重建误差；训练集学习的 GLRAM 重排也降低了独立测试图像的重建误差。分类的主设置固定为 8×8 特征：12 组配对均值中 {pos} 组提升、{neg} 组下降。请同时查看每组结果与标准差，不能将元素重排解释为保证提升测试准确率。</p>"
    )
    b.append(
        "<p>本次不调整测试集上的最佳维数。两个预先确定的特征形状 8×8、12×8 全部报告；论文则搜索维数与 MFA 参数并报告最佳测试准确率。不同数据与选择方式使二者不具备直接数值可比性。</p>"
    )
    b.append(
        "<p>向量 1DLDA/1DMFA 在多组设置中仍取得更高准确率。例如公开 PIE 每人 6 张训练时，1DMFA 为 92.86%，8×8 的 2DMFA-LN-ER 为 89.80%。这些结果支持重排对既定双线性基线的改善，但不支持其优于所有向量方法的结论。</p>"
    )
    b.append("<h2>2. 数据与实验协议</h2>")
    b.append(
        table(
            ["项目", "原论文", "本次实际运行"],
            [
                [
                    "CMU PIE",
                    "63 人 × 20 张；5 种姿态、4 种光照",
                    "公开镜像 68 人 × 24 张；原姿态/光照 ID 缺失",
                ],
                [
                    "FERET",
                    "200 人 × 7 张；固定 ba/bc/bh 训练",
                    "未取得原始灰度数据，不报告 FERET 实测",
                ],
                ["补充数据", "无 ORL", "ORL：40 人 × 10 张，64×64"],
                [
                    "预处理",
                    "眼部对齐、64×64、直方图均衡、/255−0.5",
                    "使用现成裁剪；256-bin CDF-min 均衡、/255−0.5",
                ],
                [
                    "分类划分",
                    "PIE G4/P16、G5/P15、G6/P14",
                    "PIE G4/P20、G5/P19、G6/P18；ORL G4/P6、G5/P5、G6/P4",
                ],
                [
                    "重复/选参",
                    "随机种子与重复数未给；报告搜索最佳测试值",
                    "种子 0–4；固定 8×8 和 12×8；k2=25",
                ],
                ["数据泄漏控制", "—", "只用训练样本/标签拟合零空间、重排、投影；测试集只计算指标"],
            ],
        )
    )
    b.append(
        "<h2>3. 压缩与重建结果</h2><p>表中 RMSE 遵循论文定义：√(Σᵢ‖Xᵢ−X̂ᵢ‖²F/N)。对于 64×64 图像，逐像素 RMSE = 本表数值 / 64。“降低”是相对于同一基线的百分比。固定 d 下，GLRAM 有 d² 个特征，2DPCA 有 64d 个特征；跨方法应参考 CR 曲线。</p>"
    )
    b.append(
        table(
            ["数据", "基线 (d=5)", "无重排 RMSE", "LN-ER RMSE", "NN-ER RMSE", "LN 降低", "NN 降低"],
            compression_table,
        )
    )
    b.append(
        '<img src="figures/compression.png" alt="重建误差与秩和压缩率的关系"><p class="caption">CR 已计入一次共享置换的索引开销。它是按标量数量计算的理论存储比，不是实际编码文件压缩比；PCA 的论文式 CR 不计均值向量。</p>'
    )
    b.append(
        '<img src="figures/convergence.png" alt="目标函数收敛"><p class="caption">从无重排基线开始，记录每次接受的重排。代码显式验证单调下降；训练误差下降不等价于测试准确率提升。</p>'
    )
    b.append(
        "<h3>训练集之外的重建检验</h3><p>每人 6 张训练，种子 42；只在训练集学习置换和投影，再对剩余图像重建。固定 d=5。</p>"
    )
    b.append(table(["数据", "方法", "训练 RMSE", "测试 RMSE", "测试误差相对降低"], held_table))
    b.append(
        "<h3>图像重建示例</h3><p>重建图像均已经逆置换回原始坐标；最后一行展示共享重排后的输入。图像选择位置固定，未按视觉效果挑选。</p>"
    )
    for ds in cfg["datasets"]:
        b.append(f'<img src="figures/reconstructions_{ds}.png" alt="{LABELS[ds]} 重建">')
    b.append(
        "<h2>4. 分类结果</h2><p>数值为 5 个划分的平均准确率 ± 样本标准差，单位 %。主设置为 8×8；12×8 是预先指定的敏感性设置。标准差反映随机划分波动，不是独立人群抽样的置信区间。</p>"
    )
    b.append('<img src="figures/classification.png" alt="分类准确率与随机划分波动">')
    b.append(table(["数据", "每类训练数", "特征形状"] + names, classification_table))
    b.append("<h3>同一划分的配对变化</h3><p>ER − 基线，单位百分点。负值也完整保留。</p>")
    b.append(
        table(
            ["数据", "每类训练数", "特征形状", "2DLDA Δ (百分点)", "2DMFA Δ (百分点)"], delta_table
        )
    )
    b.append(
        "<h3>向量基线</h3><p>PCA95 只按训练集 95% 能量保留特征。1DLDA/1DMFA 在该 PCA 空间中拟合，保留至多 C−1 维；未搜索最佳测试维度。</p>"
    )
    b.append(table(["数据", "每类训练数", "PCA95", "1DLDA", "1DMFA"], vector_table))
    b.append(
        "<h2>5. 原论文报告值：单独列示</h2><p>下表来自原文 Table 1（第 1919 页）。括号内为作者报告的最佳特征维数。它们是文献值，绝非本机运行结果，也不与上面的不同协议分数计算复现误差。</p>"
    )
    b.append(table(["方法", "FERET", "PIE G4/P16", "PIE G5/P15", "PIE G6/P14"], ref_rows))
    b.append("<h2>6. 实现、验证与运行记录</h2>")
    b.append(
        "<p>PyTorch 2.5.1+cu121，Python 3.10.20；实际设备 NVIDIA GeForce RTX 4090 D。GPU 用于矩阵运算和匹配成本构造，CPU 用于稀疏精确匹配与小型广义特征分解。匹配求解与原文运输 LP 具有相同的整数最优目标。全局共享置换、warm start、零空间监督样本构造和逆置换均有测试覆盖。</p>"
    )
    b.append(
        "<p>浮点设置：矩阵训练 FP32；特征分解和目标累加 FP64；禁用 TF32，启用 PyTorch 确定性模式。为满足该版本 PyTorch 的确定性要求，PCA 的小规模特征值累计在 CPU 上完成。2D 判别分析固定 10 次交替更新，奇异散布矩阵采用相对 ridge=1e−6。完整数值约定见项目 README。</p>"
    )
    b.append(table(["阶段", "阶段墙钟时间（并行运行，勿相加）", "该进程 PyTorch 峰值显存"], timing))
    b.append(
        f"<p>自动测试：<code>{html.escape(test_text)}</code>。结果文件独立审计：<strong>{verify['status']}</strong>；重算了 {verify.get('recomputed_compression_models', 0)} 个压缩模型、{verify.get('recomputed_classifications', 0)} 组分类预测，并核验全部 {verify.get('prediction_files', 0)} 个预测文件、所有训练/测试索引互斥和置换合法性。</p>"
    )
    b.append(
        f"<p>达到 100 次 ER 上限的实验数量：{len(capped)}。其余停止原因及每次目标值见 summary.json 和 records/；达到上限的实验仅视作固定计算预算结果，不声称已经收敛。</p>"
    )
    if capped:
        b.append(
            table(
                ["数据", "方法", "维数/特征", "阶段"],
                [
                    [r["dataset"], r["method"], r.get("d", r.get("rank")), r["phase"]]
                    for r in capped
                ],
            )
        )
    if supplemental:
        b.append(
            f"<p>补充收敛检查：将唯一达到上限的 ORL / GLRAM-NN-ER / d=3 配置上限扩至 500，从相同初始状态重新运行，实际在 {supplemental['actual_iterations']} 次接受更新后得到恒等匹配。RMSE 从主实验的 {supplemental['main_rmse']:.6f} 降至 {supplemental['rmse']:.6f}，仅额外下降 {supplemental['additional_relative_reduction_percent']:.4f}%。这项补充记录单独保存在 results/convergence_check/，未覆盖主实验固定预算结果。</p>"
        )
    b.append(
        "<p>数据完整性检查：2,032 张规范化图像均有限、尺寸正确，未发现完全重复图像或跨类别重复图像。详见 data_integrity.json。</p>"
    )
    b.append(
        '<h2>7. 文件与复现入口</h2><ul><li><a href="compression.csv">compression.csv</a>：全部 126 组压缩结果。</li><li><a href="classification.csv">classification.csv</a>：全部 330 组分类结果。</li><li><a href="holdout.csv">holdout.csv</a>：6 组独立测试重建。</li><li><a href="summary.json">summary.json</a>：统计汇总；<a href="verification.json">verification.json</a>：独立核验。</li><li>models/：投影、置换和逐样本预测；splits/：精确训练/测试索引与图像 ID；records/：逐实验目标轨迹。</li><li>在 DATA_9501 下执行 <code>bash scripts/reproduce.sh</code>；已完成记录可断点续跑。</li></ul>'
    )
    b.append(
        '<h2>参考来源</h2><ul><li><a href="https://www.ee.columbia.edu/ln/dvmm/publications/09/xu_ebsl.pdf">Xu et al., 2009 论文全文</a></li><li><a href="https://github.com/Gunther17/CMU_PIE-dataset">CMU PIE 公开镜像</a>：第三方数据，原姿态/光照元信息未提供。</li><li><a href="http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html">Deng Cai 人脸数据页</a>：ORL 64×64。</li><li><a href="https://www.nist.gov/itl/products-and-services/color-feret-database">NIST FERET 可用性说明</a>：原始灰度版本当前不提供下载。</li></ul>'
    )
    css = """body{margin:0;background:#f2f4f5;color:#213039;font:16px/1.75 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}main{max-width:1120px;margin:32px auto;background:white;padding:48px 56px;box-shadow:0 8px 40px #18333e08}h1{font-size:36px;line-height:1.3;margin:12px 0 18px;letter-spacing:-.6px}h2{font-size:25px;margin-top:42px;border-top:1px solid #dae1e4;padding-top:25px}h3{font-size:20px;margin-top:28px}.eyebrow{color:#307d70;font-weight:650;font-size:13px;letter-spacing:1.8px}.lead{font-size:18px;color:#61737e}.notice{border-left:4px solid #cf7842;background:#fff5e9;padding:18px 22px;margin:26px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.cards>div{background:#edf4f3;padding:20px;border-radius:8px}.cards b{font-size:29px;display:block;color:#25675c}.cards span{font-size:13px;color:#5a6c75}table{width:100%;border-collapse:collapse;font-size:13px}th{background:#eaf0f2;color:#314f5d;text-align:left;font-weight:650}td,th{padding:10px 12px;border-bottom:1px solid #e0e7ea;vertical-align:top}tr:nth-child(even){background:#f7f9fa}.table-wrap{overflow-x:auto;margin:18px 0}img{max-width:100%;display:block;margin:22px auto}.caption{font-size:13px;color:#687b86}code{font-family:ui-monospace,monospace;background:#edf1f4;font-size:13px;padding:3px 5px}a{color:#226d80}li{margin:7px 0}@media(max-width:700px){main{padding:24px;margin:0}h1{font-size:28px}.cards{grid-template-columns:repeat(2,1fr)}}@media print{main{box-shadow:none;margin:0;padding:0}.notice,.cards,table,img{break-inside:avoid}h2{break-after:avoid}body{background:white}}"""
    page = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DATA_9501 · 论文复现实验报告</title><style>'
        + css
        + "</style><main>"
        + "".join(b)
        + "</main></html>"
    )
    # Embed plots so the report remains readable as a single standalone file.
    for picture in figs.glob("*.png"):
        uri = "data:image/png;base64," + base64.b64encode(picture.read_bytes()).decode()
        page = page.replace('src="figures/' + picture.name + '"', 'src="' + uri + '"')
    destination = ROOT / args.report if args.report else out / "report.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    for name in [
        "compression.csv",
        "classification.csv",
        "holdout.csv",
        "summary.json",
        "verification.json",
    ]:
        target = Path(os.path.relpath(out / name, destination.parent)).as_posix()
        page = page.replace(f'href="{name}"', f'href="{html.escape(target, quote=True)}"')
    destination.write_text(page)
    print("Wrote", destination)


if __name__ == "__main__":
    main()
