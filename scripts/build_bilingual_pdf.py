#!/usr/bin/env python3
"""Create the paired Chinese/English PDF from verified experiment records.

Install the reports extra: uv pip install --python .venv/bin/python -e '.[reports]'.
No experiments are rerun and no saved measurements are changed.
"""

import html
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/main"
OUTPUT = ROOT / "reports/DATA_9501_bilingual_reproduction_report.pdf"
FIGURES = RESULTS / "figures"
rows = json.loads((RESULTS / "all_results.json").read_text())
summary = json.loads((RESULTS / "summary.json").read_text())
verification = json.loads((RESULTS / "verification.json").read_text())
config = json.loads((RESULTS / "config.json").read_text())
reference = json.loads((ROOT / "papers/paper_table1.json").read_text())
supplement = json.loads((ROOT / "results/convergence_check/result.json").read_text())
manifest = json.loads((ROOT / "data/manifest.json").read_text())
assert len(rows) == 462 and verification["status"] == "passed"

for name, path in {
    "Text": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "Bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "Mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "Chinese": "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
}.items():
    pdfmetrics.registerFont(TTFont(name, path))

INK = colors.HexColor("#203640")
MUTED = colors.HexColor("#5c707b")
TEAL = colors.HexColor("#27796c")
PALE = colors.HexColor("#edf4f2")
LINE = colors.HexColor("#d6e1e4")
ORANGE = colors.HexColor("#a9632e")
PW, PH = A4
W = PW - 88
styles = {
    "cn": ParagraphStyle(
        "cn",
        fontName="Text",
        fontSize=9.7,
        leading=14.8,
        textColor=INK,
        wordWrap="CJK",
        spaceAfter=3,
    ),
    "en": ParagraphStyle(
        "en", fontName="Text", fontSize=8.7, leading=12.7, textColor=MUTED, spaceAfter=10
    ),
    "small": ParagraphStyle(
        "small",
        fontName="Text",
        fontSize=8,
        leading=11.5,
        textColor=MUTED,
        wordWrap="CJK",
        spaceAfter=6,
    ),
    "cell": ParagraphStyle(
        "cell", fontName="Text", fontSize=8.1, leading=11.1, textColor=INK, wordWrap="CJK"
    ),
    "headcell": ParagraphStyle(
        "headcell", fontName="Bold", fontSize=8.1, leading=11.4, textColor=INK, wordWrap="CJK"
    ),
    "section": ParagraphStyle(
        "section",
        fontName="Bold",
        fontSize=17,
        leading=23,
        textColor=INK,
        wordWrap="CJK",
        spaceAfter=3,
    ),
    "section_en": ParagraphStyle(
        "section_en", fontName="Text", fontSize=11.3, leading=16, textColor=TEAL, spaceAfter=14
    ),
    "sub": ParagraphStyle(
        "sub",
        fontName="Bold",
        fontSize=10.2,
        leading=15,
        textColor=TEAL,
        wordWrap="CJK",
        spaceBefore=8,
        spaceAfter=7,
    ),
    "title_cn": ParagraphStyle(
        "title_cn",
        fontName="Bold",
        fontSize=26,
        leading=36,
        textColor=INK,
        wordWrap="CJK",
        spaceAfter=10,
    ),
    "title_en": ParagraphStyle(
        "title_en", fontName="Bold", fontSize=16, leading=22, textColor=INK, spaceAfter=14
    ),
    "code": ParagraphStyle(
        "code",
        fontName="Mono",
        fontSize=8.3,
        leading=12,
        textColor=INK,
        wordWrap="CJK",
        spaceAfter=5,
    ),
}


def mixed(text):
    """Embed a true CJK font only for supported Chinese glyph runs."""
    text = html.escape(str(text)).replace("\n", "<br/>")
    return re.sub(
        r"([\u2e80-\u9fff\uf900-\ufaff\uff01-\uff60]+)", r'<font name="Chinese">\1</font>', text
    )


def p(text, style="cn"):
    return Paragraph(mixed(text), styles[style])


story = []


def bi(cn, en):
    story.append(KeepTogether([p(cn), p(en, "en")]))


def section(number, cn, en):
    if story:
        story.append(PageBreak())
    story.extend([p(f"{number:02d}  {cn}", "section"), p(en, "section_en")])


def sub(cn, en):
    story.append(p(cn + " / " + en, "sub"))


def tbl(headers, data, widths=None, font=8.1):
    headers = [Paragraph(mixed(h), styles["headcell"]) for h in headers]
    st = ParagraphStyle("scaled", parent=styles["cell"], fontSize=font, leading=font + 3)
    data = [[Paragraph(mixed(c), st) for c in row] for row in data]
    t = Table(
        [headers] + data,
        colWidths=widths or [W / len(headers)] * len(headers),
        repeatRows=1,
        hAlign="LEFT",
    )
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PALE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("TOPPADDING", (0, 1), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fa")]),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, LINE),
                ("LINEBELOW", (0, 1), (-1, -1), 0.35, LINE),
            ]
        )
    )
    story.extend([t, Spacer(1, 10)])


def figure(name, width=W):
    path = FIGURES / name
    with PILImage.open(path) as im:
        w, h = im.size
    story.extend([Image(str(path), width=width, height=width * h / w), Spacer(1, 5)])


def caption(cn, en):
    story.extend([p(cn, "small"), p(en, "small"), Spacer(1, 5)])


def classification(ds, g, method, rank):
    rr = [
        r
        for r in rows
        if r["phase"] == "classification"
        and r["dataset"] == ds
        and r["gallery"] == g
        and r["method"] == method
        and r["rank"] == rank
    ]
    assert len(rr) == 5
    a = np.array([r["accuracy"] * 100 for r in rr])
    return f"{a.mean():.2f} ± {a.std(ddof=1):.2f}"


def clstable(rank):
    data = []
    for ds, name in [("pie_public", "PIE"), ("orl", "ORL")]:
        for g in [4, 5, 6]:
            data.append(
                [name, str(g)]
                + [
                    classification(ds, g, m, rank)
                    for m in ["2DLDA", "2DLDA-LN-ER", "2DMFA", "2DMFA-LN-ER"]
                ]
            )
    tbl(
        [
            "数据\nData",
            "训练/人\nTrain /\nperson",
            "2DLDA",
            "2DLDA\n+ LN-ER",
            "2DMFA",
            "2DMFA\n+ LN-ER",
        ],
        data,
        [38, 65, 100, 102, 100, W - 405],
        font=8.2,
    )


class PageCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states = []

    def showPage(self):
        self.states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self.states)
        for state in self.states:
            self.__dict__.update(state)
            self.setStrokeColor(LINE)
            self.setLineWidth(0.5)
            self.line(44, 38, PW - 44, 38)
            self.setFillColor(MUTED)
            self.setFont("Text", 7.2)
            self.drawString(44, 25, "DATA_9501  |  Bilingual reproduction report  |  10 Nov 2026")
            self.drawRightString(PW - 44, 25, f"{self._pageNumber} / {total}")
            if self._pageNumber > 1:
                self.setFont("Text", 7)
                self.setFillColor(TEAL)
                self.drawString(44, PH - 28, "ELEMENT REARRANGEMENT  /  TPAMI 2009")
                self.drawRightString(PW - 44, PH - 28, "PUBLIC-DATA METHOD REPRODUCTION")
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)


# Page 1: self-contained summary.
story.append(p("DATA_9501  |  2026-11-10  |  中文 / ENGLISH", "small"))
story.append(Spacer(1, 12))
story.append(p("双线性子空间学习\n元素重排复现实验报告", "title_cn"))
story.append(
    p(
        "Bilinear Subspace Learning with Element Rearrangement\nBilingual Reproduction Report",
        "title_en",
    )
)
story.append(
    p(
        "Based on: Enhancing Bilinear Subspace Learning by Element Rearrangement\nDong Xu et al. · IEEE TPAMI 31(10), 1913-1920 (2009) · DOI: 10.1109/TPAMI.2009.51 [1]",
        "small",
    )
)
story.append(Spacer(1, 8))
tbl(
    ["复现定位 / Scope"],
    [
        [
            "公开数据上的方法复现，不是原论文表格的精确复现。PIE 子集与原文不同；未取得原始灰度 FERET。\n"
            "A method reproduction on public data, not an exact replication of the published tables. The PIE subset differs, and original grayscale FERET was not obtained."
        ]
    ],
    [W],
    font=9.1,
)
tbl(
    ["462", "5", "RTX 4090 D"],
    [
        [
            "主实验记录\nMain experiment records",
            "分类随机种子\nClassification seeds",
            "实际训练设备\nActual training device",
        ]
    ],
    [W / 3] * 3,
    font=9,
)
sub("主要发现", "Key findings")
bi(
    "固定 d=5 时，GLRAM 加入 NN 重排后，公开 PIE 和 ORL 的重建 RMSE 分别降低 7.31% 与 7.98%；独立测试图像的重建误差也下降。",
    "At d=5, NN rearrangement reduces GLRAM reconstruction RMSE by 7.31% on public PIE and 7.98% on ORL. Reconstruction errors also decrease on held-out images.",
)
bi(
    "固定 8×8 特征时，12 组分类配对均值中有 11 组提升、1 组下降。结果支持改善既定双线性基线，但不代表在所有设置或所有基线中最优。",
    "With fixed 8×8 features, mean accuracy improves in 11 of 12 paired settings and decreases in one. This supports gains over the corresponding bilinear baselines, not universal superiority.",
)
bi(
    "17 项测试通过。独立重算了 108 个压缩模型和 24 组分类预测，并核验全部 330 个预测文件；另完成一项延长迭代的收敛检查。",
    "All 17 tests pass. Independent checks recompute 108 compression models and 24 classification settings and audit all 330 prediction files. One additional convergence experiment is also completed.",
)
story.append(
    p(
        "阅读说明：每个中文段落后紧接对应英文。表格数值来自已保存的本机实验记录；原论文数值单独列示。\nReading guide: Each Chinese paragraph is followed by its English counterpart. Measured results come from saved local records; published values are shown separately.",
        "small",
    )
)

# Page 2: method.
section(2, "方法与实现", "Method and implementation")
bi(
    "论文通过学习一个供所有图像共用的像素置换，增强行列相关性，使双线性低秩投影更有效。置换只在拟合集上学习，并固定应用于测试图像。重建评估前会逆置换回原始坐标。",
    "The method learns one pixel permutation shared by all images to strengthen row/column correlations for bilinear low-rank projection. The permutation is learned only on the fit set, frozen at test time, and inverted before reconstruction evaluation. [1]",
)
story.append(p("Objective: min(U,V,R) sum_i ||R(X_i) - U U^T R(X_i) V V^T||_F^2", "code"))
story.append(
    p(
        "Features: Y_i = U^T R(X_i) V    |    Matching cost: c_pq = sum_i (X_i[p] - Xrec_i[q])^2",
        "code",
    )
)
tbl(
    ["组成 / Component", "本次实现 / Implementation"],
    [
        [
            "GLRAM / CSA",
            "交替特征分解更新 U、V；用上轮因子热启动。\nAlternating eigendecompositions for U and V, warm-started from the previous iteration.",
        ],
        ["2DPCA", "令 U=I，仅学习右投影 V。\nSet U=I and learn only the right projection V."],
        [
            "LN\nSpatial neighborhood",
            "裁剪后的 5×5 邻域，包含恒等边；64×64 时共 98,596 条边。\nClipped 5×5 neighborhoods including identity; 98,596 edges at 64×64.",
        ],
        [
            "NN\nFeature neighborhood",
            "恒等位置加 24 个最低重建匹配代价的目标位置；每次更新。\nIdentity plus 24 targets with the lowest reconstruction-matching cost, refreshed each iteration.",
        ],
        [
            "Integral assignment",
            "用稀疏最小权完美匹配求解与原文运输 LP 相同的整数最优目标。\nSparse minimum-weight perfect matching solves the same integral assignment objective as the transport LP.",
        ],
        [
            "2DLDA-LN-ER",
            "先投影至类内散布零空间，在加权、中心化类均值上学习重排，再拟合 2DLDA。\nLearn ER from weighted centered class means after within-class null projection, then fit 2DLDA.",
        ],
        [
            "2DMFA-LN-ER",
            "在投影后的跨类边界样本差上学习重排；每类 k2=25，再拟合 2DMFA。\nLearn ER from projected cross-class margin differences, with k2=25 per class, then fit 2DMFA.",
        ],
        [
            "Classifier",
            "低维空间中的欧氏距离 1 近邻。\nEuclidean 1-nearest-neighbor classification in the learned space.",
        ],
    ],
    [115, W - 115],
    font=8.3,
)
bi(
    "零空间投影采用 x - (xQᵀ)Q，其中 Q 的行张成类内残差的行空间，避免显式构造巨大的零空间基。所有监督统计只使用训练数据。",
    "Null projection uses x - (xQᵀ)Q, where the rows of Q span the row space of within-class residuals. This avoids materializing a large null basis. All supervised statistics use training data only.",
)

# Page 3: data and config.
section(3, "数据与实验协议", "Data and experimental protocol")
tbl(
    ["项目 / Item", "原论文 / Paper [1]", "本次 / This reproduction"],
    [
        [
            "CMU PIE",
            "63 人×20 张；5 种姿态、4 种光照。\n63 people ×20 images; 5 poses and 4 illuminations.",
            "公开镜像：68 人×24 张，1,632 张。\nPublic mirror: 68 people ×24 images; 1,632 total. [2]",
        ],
        [
            "FERET",
            "200 人×7 张；ba/bc/bh 训练。\n200 people ×7 images; ba/bc/bh for training.",
            "未取得原始灰度数据，不报告实测。\nOriginal grayscale data unavailable; no measured FERET results. [4]",
        ],
        [
            "ORL",
            "原文未使用。\nNot used.",
            "额外验证：40 人×10 张，400 张。\nAdditional validation: 40 people ×10 images. [3]",
        ],
        [
            "尺寸 / Size",
            "64×64，眼部对齐。\n64×64, eye aligned.",
            "64×64 现成裁剪，无额外缩放。\nProvided 64×64 crops; no resizing.",
        ],
        ["PIE 划分 / Split", "G4/P16, G5/P15, G6/P14", "G4/P20, G5/P19, G6/P18"],
        ["ORL 划分 / Split", "不适用 / Not applicable", "G4/P6, G5/P5, G6/P4"],
    ],
    [77, 205, W - 282],
    font=8.2,
)
bi(
    "G/P 表示每人训练/测试图像数。公开 PIE 缺少原始姿态、光照编号和眼部坐标，无法确认与论文子集及对齐模板一致。ORL 是补充验证集。",
    "G/P denotes training/test images per person. The public PIE mirror lacks original pose, illumination and eye-landmark metadata, so its subset and alignment cannot be matched to the paper. ORL is supplemental validation.",
)
tbl(
    ["设置 / Setting", "实际值 / Value"],
    [
        [
            "预处理 / Preprocessing",
            "每图 256-bin CDF-min 直方图均衡，uint8 四舍五入，再 /255 - 0.5。\nPer-image 256-bin CDF-min equalization, rounded uint8, then /255 - 0.5.",
        ],
        ["压缩秩 / Compression ranks", "d = 2, 3, 4, 5, 8, 12, 16, 24, 32"],
        [
            "分类维数与重复 / Classification",
            "固定 8×8、12×8；种子 0, 1, 2, 3, 4。\nFixed 8×8 and 12×8 features; seeds 0, 1, 2, 3, 4.",
        ],
        [
            "ER / GLRAM stopping",
            "最多 100 次；相对目标改善阈值 1e-7；另检查恒等匹配与数值平台。\nAt most 100 iterations; relative objective tolerance 1e-7; identity and numerical-plateau checks.",
        ],
        [
            "2DLDA / 2DMFA",
            "10 次交替更新；相对 ridge=1e-6；k2=25。\n10 alternating updates; relative ridge=1e-6; k2=25.",
        ],
    ],
    [135, W - 135],
    font=8.2,
)
bi(
    "分类维数为预先固定值，未用测试标签选参。论文报告搜索维数与 MFA 参数后的最佳测试准确率，因此两者不是相同的选参协议。",
    "Classification dimensions are fixed in advance; test labels do not select parameters. The paper reports the best test accuracy over dimension and MFA-parameter searches, so the selection protocols differ.",
)

# Page 4: compression.
section(4, "压缩与重建", "Compression and reconstruction")
bi(
    "固定 d=5 的同基线比较见下表。RMSE 越低越好；LN/NN 降幅相对于该行的无重排基线。",
    "The table compares each baseline at d=5. Lower RMSE is better; LN/NN reductions are relative to the unrearranged baseline in the same row.",
)
data = []
for ds, name in [("pie_public", "PIE"), ("orl", "ORL")]:
    for method in ["GLRAM", "2DPCA"]:
        r = summary["compression_d5"][ds][method]
        data.append(
            [
                name,
                method,
                f"{r['baseline_rmse']:.4f}",
                f"{r['ln_rmse']:.4f}",
                f"{r['nn_rmse']:.4f}",
                f"{r['ln_relative_reduction_percent']:.2f}%",
                f"{r['nn_relative_reduction_percent']:.2f}%",
            ]
        )
tbl(
    [
        "数据\nData",
        "方法\nMethod",
        "基线\nBase",
        "LN-ER",
        "NN-ER",
        "LN 降幅\nReduction",
        "NN 降幅\nReduction",
    ],
    data,
    [40, 65, 70, 70, 70, 96, W - 411],
)
figure("compression.png", width=470)
caption(
    "图 1. RMSE 随秩 d 和压缩比 CR 的变化。不同 d 独立优化，局部最优可能导致曲线局部不单调。",
    "Figure 1. RMSE versus rank d and compression ratio CR. Each rank is optimized separately; local optima can produce non-monotone segments.",
)
bi(
    "论文式 RMSE = √(Σᵢ‖Xᵢ-X̂ᵢ‖²F/N)，逐像素 RMSE 等于该值/64。固定 d 时，GLRAM 有 d² 个特征，2DPCA 有 64d 个特征；跨方法比较应参考 CR。",
    "Paper-defined RMSE is sqrt(sum_i ||X_i-Xhat_i||_F²/N); pixel RMSE is this value divided by 64. At fixed d, GLRAM has d² features and 2DPCA has 64d; cross-method comparisons should use CR.",
)
story.append(
    p(
        "CR 计入一次共享置换开销，是理论标量存储比，不是实际编解码器的码率。PCA 按论文 CR 公式不计均值向量。\nCR includes one shared-permutation overhead and measures theoretical scalar storage, not a codec bitrate. The paper-style PCA formula excludes its mean vector.",
        "small",
    )
)

# Page 5: convergence and unseen images.
section(5, "收敛与独立测试重建", "Convergence and held-out reconstruction")
figure("convergence.png")
caption(
    "图 2. d=5 的已接受更新轨迹。每次匹配和投影更新都检查目标值下降。",
    "Figure 2. Accepted-update trajectories at d=5. Objective decrease is checked after matching and projection updates.",
)
bi(
    "独立重建实验每人使用 6 张训练图像、种子 42；只在训练集学习置换及投影，剩余图像仅用于评估。",
    "The holdout experiment uses six training images per person and seed 42. Permutations and projections are fitted on training images only; all remaining images are used solely for evaluation.",
)
held = [r for r in rows if r["phase"] == "holdout"]
data = []
for ds, name in [("pie_public", "PIE"), ("orl", "ORL")]:
    rr = {r["method"]: r for r in held if r["dataset"] == ds}
    for method in ["GLRAM", "GLRAM-LN-ER", "GLRAM-NN-ER"]:
        r = rr[method]
        gain = 100 * (1 - r["test_rmse"] / rr["GLRAM"]["test_rmse"])
        data.append(
            [name, method, f"{r['train_rmse']:.4f}", f"{r['test_rmse']:.4f}", f"{gain:.2f}%"]
        )
tbl(
    [
        "数据\nData",
        "方法 / Method",
        "训练 RMSE\nTrain",
        "测试 RMSE\nTest",
        "测试误差降低\nTest reduction",
    ],
    data,
    [40, 145, 95, 95, W - 375],
)
bi(
    "两套数据上的 LN 和 NN 重排都改善了未见图像的重建。ORL 的独立测试中 LN 略优于 NN，说明特征邻域不保证处处更优。",
    "Both LN and NN improve reconstruction of unseen images on both datasets. LN slightly outperforms NN on the ORL holdout, so feature neighborhoods do not guarantee better generalization.",
)
sub("达到上限配置的补充检查", "Supplemental check of the capped run")
bi(
    f"主实验中只有 ORL / GLRAM-NN-ER / d=3 达到 100 次上限。将上限扩大至 500 后，从相同初态重新运行，在第 {supplement['actual_iterations']} 次接受更新后得到恒等匹配；RMSE 从 {supplement['main_rmse']:.6f} 降至 {supplement['rmse']:.6f}，额外降低 {supplement['additional_relative_reduction_percent']:.4f}%。主实验记录保持不变。",
    f"Only ORL / GLRAM-NN-ER / d=3 reaches the main 100-iteration cap. Repeating from the same initialization with a cap of 500 reaches identity after {supplement['actual_iterations']} accepted updates: RMSE falls from {supplement['main_rmse']:.6f} to {supplement['rmse']:.6f}, an additional {supplement['additional_relative_reduction_percent']:.4f}% reduction. Main records are unchanged.",
)

# Page 6: visual audit, original fixed sample choices.
section(6, "重建图像对照", "Reconstruction examples")
figure("reconstructions_pie_public.png", width=470)
caption(
    "图 3. 公开 PIE，d=5。各行：原图、GLRAM、LN-ER、NN-ER，以及重排后的输入。",
    "Figure 3. Public PIE at d=5. Rows: original, GLRAM, LN-ER, NN-ER, and rearranged input.",
)
figure("reconstructions_orl.png", width=470)
caption(
    "图 4. ORL，d=5，行顺序同上。所有重建已逆置换回原始坐标，展示样本按固定索引选取。",
    "Figure 4. ORL at d=5, with the same row order. Reconstructions are inverse-permuted to original coordinates; sample indices are fixed.",
)

# Page 7: full primary classification.
section(7, "分类主结果：8×8 特征", "Primary classification results: 8×8 features")
bi(
    "以下为 5 个划分的平均准确率 ± 样本标准差，单位 %。训练/测试人数相同，每个划分对所有方法复用。标准差描述划分波动，不是独立人群抽样的置信区间。",
    "Values are mean accuracy ± sample standard deviation across five splits, in percent. The same identities and each exact split are reused across methods. Standard deviation describes split variability, not a confidence interval for an independently sampled population.",
)
clstable([8, 8])
figure("classification.png")
caption(
    "图 5. 固定 8×8 特征的测试准确率；误差线为五次划分的样本标准差。",
    "Figure 5. Test accuracy at fixed 8×8 features; error bars show the sample standard deviation over five splits.",
)
sub("结果解读", "Interpretation")
bi(
    "每人 4 张训练时，PIE 的 2DLDA 从 84.37% 提升至 87.66%，2DMFA 从 78.81% 提升至 86.50%；ORL 对应从 85.17% 至 89.33%、81.17% 至 87.25%。",
    "With four training images per person, PIE improves from 84.37% to 87.66% for 2DLDA and from 78.81% to 86.50% for 2DMFA. ORL improves from 85.17% to 89.33% and from 81.17% to 87.25%, respectively.",
)
bi(
    "ORL 每人 6 张训练时，2DLDA 的均值从 93.875% 略降至 93.750%。这些结果不应表述为所有设置均获得提升。",
    "For ORL with six training images per person, mean 2DLDA accuracy decreases slightly from 93.875% to 93.750%. The results therefore should not be described as improvements in every setting.",
)

# Page 8: fixed alternate rank and vector references.
section(8, "维数敏感性与向量基线", "Dimension sensitivity and vector baselines")
bi(
    "12×8 是预先指定的第二个特征形状，所有结果完整报告，未按测试准确率在两种形状之间挑选最优值。",
    "12×8 is the second predeclared feature shape. Every result is reported; test accuracy does not choose the better of the two shapes.",
)
clstable([12, 8])
sub("向量方法的参考结果", "Vector-method reference results")
bi(
    "PCA95 按训练集 95% 能量保留特征。1DLDA/1DMFA 在该 PCA 空间中拟合，保留 min(PCA维数, C-1) 个特征；不搜索最佳测试维数。",
    "PCA95 retains 95% of training-set energy. 1DLDA/1DMFA are fitted in that PCA space and retain min(PCA dimension, C-1) features, without searching for the best test dimension.",
)
data = []
for ds, name in [("pie_public", "PIE"), ("orl", "ORL")]:
    for g in [4, 5, 6]:
        line = [name, g]
        for method in ["PCA95", "1DLDA", "1DMFA"]:
            a = np.array(
                [
                    r["accuracy"] * 100
                    for r in rows
                    if r["phase"] == "classification"
                    and r["dataset"] == ds
                    and r["gallery"] == g
                    and r["method"] == method
                ]
            )
            line.append(f"{a.mean():.2f} ± {a.std(ddof=1):.2f}")
        data.append(line)
tbl(
    ["数据\nData", "训练/人\nTrain/person", "PCA95", "1DLDA", "1DMFA"],
    data,
    [50, 90, 122, 122, W - 384],
    font=8.5,
)
bi(
    "向量方法在多组设置中仍更准确。例如 PIE 每人 6 张训练时，1DMFA 为 92.86%，高于 8×8 的 2DMFA-LN-ER（89.80%）和 12×8 的对应方法（90.33%）。不同特征容量与数据协议也影响跨方法比较。",
    "Vector methods remain more accurate in several settings. For PIE with six training images per person, 1DMFA reaches 92.86%, above 2DMFA-LN-ER at 8×8 (89.80%) and 12×8 (90.33%). Feature capacity and dataset protocol also affect cross-method comparisons.",
)

# Page 9: paired deltas, all settings.
section(9, "配对提升与稳健性", "Paired changes and robustness")
bi(
    "对每个随机种子，先计算同一划分上的 ER 准确率减去基线准确率，再汇总均值和样本标准差。单位为百分点；负值保留。",
    "For each seed, ER accuracy minus baseline accuracy is computed on the same split, then summarized by the mean and sample standard deviation. Units are percentage points; negative values are retained.",
)
data = []
for ds, name in [("pie_public", "PIE"), ("orl", "ORL")]:
    for g in [4, 5, 6]:
        for rank in [[8, 8], [12, 8]]:
            line = [name, g, "×".join(map(str, rank))]
            for method in ["2DLDA", "2DMFA"]:
                r = next(
                    r
                    for r in summary["paired_deltas"]
                    if r["dataset"] == ds
                    and r["gallery"] == g
                    and r["rank"] == rank
                    and r["method"] == method
                )
                line.append(f"{r['mean_pp']:+.2f} ± {r['std_pp']:.2f}")
            data.append(line)
tbl(
    [
        "数据\nData",
        "训练/人\nTrain/person",
        "特征\nFeatures",
        "2DLDA 变化\nChange (pp)",
        "2DMFA 变化\nChange (pp)",
    ],
    data,
    [45, 85, 70, 153, W - 353],
    font=8.7,
)
bi(
    "主设置中，2DMFA 的均值提升在两套数据的全部训练规模上都为正；2DLDA 的提升在训练样本增加后减弱。第二种特征形状还出现 PIE G6 的轻微下降。",
    "In the primary setting, mean 2DMFA gains are positive at every training size on both datasets, whereas 2DLDA gains weaken as training size increases. The second feature shape also exhibits a small decline for PIE G6.",
)
bi(
    "训练目标的单调下降只保证拟合目标改善，不保证测试识别率单调提高。本报告未对重叠随机划分做独立样本显著性推断，也未据此声称普遍最优。",
    "Monotone training-objective decrease guarantees only improvement of the fitting objective, not monotonically better test recognition. No independent-sample significance claim is made from these overlapping random splits, and no universal optimum is asserted.",
)

# Page 10: original numbers and limitations.
section(10, "文献对照与复现边界", "Published reference values and limitations")
bi(
    "下表转录自原文 Table 1，第 1919 页 [1]。数值是作者发表的百分比，括号内为其最佳特征维数；不是本机结果，不与本次不同协议的分数计算复现误差。",
    "The following values are transcribed from Table 1 on page 1919 of the paper [1]. They are published percentages, with the authors' best feature dimensions in parentheses. They are not local measurements, and no replication error is computed against our different protocol.",
)
data = [
    [r["method"]] + [f"{a:.1f} ({d})" for a, d in zip(r["accuracy"], r["dimensions"])]
    for r in reference["rows"]
]
tbl(
    ["方法 / Method", "FERET", "PIE\nG4/P16", "PIE\nG5/P15", "PIE\nG6/P14"],
    data,
    [134, 86, 95, 95, W - 410],
    font=8.2,
)
sub("影响直接对照的因素", "Factors preventing direct numerical replication")
bi(
    "数据：公开 PIE 是 68×24 子集，缺少姿态、光照与眼部模板信息；原文是 63×20。ORL 不在原文中。原始灰度 FERET 未获得，不作结果声明。",
    "Data: Public PIE is a 68×24 subset without original pose, illumination or eye-template metadata; the paper uses 63×20. ORL is absent from the paper. Original grayscale FERET was not obtained, and no measurements are claimed for it.",
)
bi(
    "选参：原文搜索 m′、n′ 从 2 到 40（步长 2），MFA 的 k2 从 25 到 600（步长 25），报告最佳测试结果。本次固定两种形状及 k2=25；原文未给随机种子与重复次数。",
    "Selection: The paper searches m′ and n′ from 2 to 40 in steps of 2, and MFA k2 from 25 to 600 in steps of 25, reporting the best test results. We fix two shapes and k2=25. The paper does not provide random seeds or repeat counts.",
)
bi(
    "数值细节：本次使用 256-bin CDF-min 均衡、相对 ridge=1e-6、与下游分类相同的监督 ER 代理秩。双线性压缩直接拟合 /255-0.5 后的矩阵，向量 PCA 另做均值中心化。原文对若干细节说明不足。",
    "Numerics: We use 256-bin CDF-min equalization, relative ridge=1e-6, and supervised ER proxy ranks matching the downstream classifier. Bilinear compression fits /255-0.5 normalized matrices directly; vector PCA is additionally centered. Several such details are underspecified in the paper.",
)
bi(
    "若要精确对齐，需要取得原始子集及样本标识、对齐模板与预处理说明、作者随机划分，以及其维数搜索和数值约定；FERET 还需实现具名图像的固定训练/测试划分。",
    "Exact alignment would require the original subset and sample IDs, alignment template and preprocessing details, author splits, and matching search/numerical conventions. FERET would also require a fixed split adapter using named images.",
)

# Page 11: validation and environment.
section(11, "运行环境与结果核验", "Runtime environment and verification")
bi(
    "所有训练已在 NVIDIA GeForce RTX 4090 D 上完成。GPU 执行矩阵运算和匹配代价构造；CPU 执行稀疏匹配和小型广义特征分解。训练结果直接读取已保存记录，本 PDF 未重新训练。",
    "Training is complete on an NVIDIA GeForce RTX 4090 D. The GPU performs matrix algebra and matching-cost construction; the CPU handles sparse matching and small generalized eigensystems. This PDF uses saved records and does not retrain models.",
)
tbl(
    ["环境项 / Item", "值 / Value"],
    [
        ["软件 / Software", "Python 3.10.20; PyTorch 2.5.1+cu121; CUDA runtime 12.1"],
        [
            "数值 / Numerics",
            "训练 FP32，特征分解和目标累加 FP64；TF32 关闭。\nFP32 training; FP64 eigensystems/objective accumulation; TF32 off.",
        ],
        [
            "确定性 / Determinism",
            "PyTorch 确定性模式；PCA 谱累计移至 CPU。\nPyTorch deterministic mode; PCA spectrum accumulation on CPU.",
        ],
    ],
    [130, W - 130],
    font=8.6,
)
data = []
for phase, cn in [("compression", "压缩"), ("classification", "分类"), ("holdout", "独立重建")]:
    env = json.loads((RESULTS / f"environment_{phase}_all.json").read_text())
    data.append(
        [
            cn + " / " + phase,
            f"{env['total_seconds']:.1f} s",
            f"{env['peak_torch_memory_bytes'] / 1024**2:.1f} MiB",
        ]
    )
tbl(
    ["阶段 / Phase", "墙钟时间 / Wall time", "PyTorch 峰值显存\nPeak allocated GPU memory"],
    data,
    [205, 125, W - 330],
)
caption(
    "部分阶段同时运行，墙钟时间不可相加；显存值是每进程的 PyTorch 已分配峰值，不是整机显存。",
    "Some phases overlap, so wall times must not be summed. Memory figures are per-process PyTorch peak allocations, not total device use.",
)
tbl(
    ["检查 / Check", "结果 / Result"],
    [
        ["自动测试 / Automated tests", "17 / 17 passed"],
        ["主记录 / Main records", "462 = 126 compression + 330 classification + 6 holdout"],
        ["预测文件 / Prediction files", "330 个全部核验 / All 330 audited"],
        [
            "独立重算 / Independent recomputation",
            "108 个压缩模型，24 组分类。\n108 compression models; 24 classification settings.",
        ],
        ["RMSE 最大差异 / Maximum RMSE discrepancy", f"{verification['max_rmse_discrepancy']:.3e}"],
        [
            "置换与轨迹 / Permutations and histories",
            "474 个置换模型合法；234 组目标轨迹单调。\n474 valid permutation models; 234 monotone objective histories.",
        ],
        [
            "划分与重复图像 / Splits and duplicates",
            "32 个划分文件互斥；2,032 张图像无完全重复。\n32 disjoint split files; no exact duplicates among 2,032 images.",
        ],
    ],
    [235, W - 235],
    font=8.4,
)
bi(
    "PCA 基线首次运行触发 CUDA cumsum 的确定性限制；改为 CPU FP64 谱累计并增加回归测试后，分类阶段已成功重跑。",
    "The initial PCA attempt hit a deterministic CUDA cumsum restriction. CPU FP64 spectrum accumulation and a regression test resolved it; the classification stage was rerun successfully.",
)

# Page 12: practical reproduction and full provenance references.
section(12, "复现入口与来源", "Reproduction entry points and sources")
bi(
    "克隆仓库后，在 COMP9501_Group9 根目录按 README 配置环境与下载数据。以下命令可复用已完成记录，或在新目录中启动独立完整运行。",
    "After cloning, set up the environment and download data from the COMP9501_Group9 root as described in README. The commands below resume saved records or launch a separate full run.",
)
story.append(p("cd COMP9501_Group9", "code"))
story.append(p("bash scripts/reproduce.sh", "code"))
story.append(p("bash scripts/reproduce.sh results/fresh", "code"))
tbl(
    ["文件 / File", "用途 / Purpose"],
    [
        ["reports/report_zh.html", "可离线打开的中文 HTML 报告。\nOffline Chinese HTML report."],
        [
            "results/main/{compression,classification,holdout}.csv",
            "全部主实验数值。\nAll main experiment measurements.",
        ],
        [
            "results/main/{models,splits,records}/",
            "模型、逐样本预测、精确划分及目标轨迹。\nModels, predictions, exact splits and objective histories.",
        ],
        [
            "results/main/{verification,data_integrity}.json",
            "独立重算与数据完整性结果。\nIndependent recomputation and data-integrity audits.",
        ],
        [
            "data/manifest.json; configs/reproduction.json",
            "来源、SHA-256 与固定实验参数。\nSources, SHA-256 hashes and fixed parameters.",
        ],
        [
            "results/convergence_check/result.json",
            "延长至恒等匹配的补充记录。\nSupplemental run to identity matching.",
        ],
        [
            "scripts/build_bilingual_pdf.py",
            "本中英对照 PDF 的可重复生成脚本。\nGenerator for this bilingual PDF.",
        ],
    ],
    [250, W - 250],
    font=7.9,
)
sub("参考来源", "References")
references = [
    (
        "[1] Xu, D., Yan, S., Lin, S., Huang, T. S., & Chang, S.-F. (2009). Enhancing Bilinear Subspace Learning by Element Rearrangement. IEEE TPAMI, 31(10), 1913-1920. DOI: 10.1109/TPAMI.2009.51.",
        "https://www.ee.columbia.edu/ln/dvmm/publications/09/xu_ebsl.pdf",
    ),
    (
        "[2] CMU PIE public mirror. Gunther17/CMU_PIE-dataset; cmupie_1632.rar. 第三方镜像，原始姿态/光照元信息缺失。 Third-party mirror without original pose/illumination metadata.",
        "https://github.com/Gunther17/CMU_PIE-dataset",
    ),
    (
        "[3] Deng Cai. Face datasets: ORL 64×64. 本次补充验证数据。 Supplemental validation data.",
        "http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html",
    ),
    (
        "[4] NIST. Color FERET Database, availability FAQ. 本次调查时原始灰度版本不提供下载。 Original grayscale version unavailable in the availability information consulted for this run.",
        "https://www.nist.gov/itl/products-and-services/color-feret-database",
    ),
]
for text, url in references:
    story.append(p(text, "small"))
    story.append(
        Paragraph(
            '<link href="' + html.escape(url) + '" color="#27796c">' + html.escape(url) + "</link>",
            styles["small"],
        )
    )
bi(
    "来源文件以固定 SHA-256 校验。未使用作者官方代码；本项目为独立实现。原始论文和数据的权利归其各自权利人，报告不赋予新的数据再分发许可。",
    "Source files are verified using fixed SHA-256 checksums. No official author implementation is used; this project is independent. Original paper and data rights remain with their respective holders; this report grants no new data-redistribution license.",
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    rightMargin=44,
    leftMargin=44,
    topMargin=50,
    bottomMargin=49,
    title="DATA_9501 - Bilingual Element Rearrangement Reproduction Report",
    author="DATA_9501 Reproduction Project",
    subject="Paired Chinese-English report of verified public-data reproduction experiments",
)
doc.build(story, canvasmaker=PageCanvas)
print(OUTPUT)
