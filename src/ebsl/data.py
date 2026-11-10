import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]


def equalize(image):
    """256-bin per-image histogram equalization (CDF-min convention).

    The paper does not specify MATLAB histeq's bin count. Our exact convention
    is explicit and deterministic. No statistics are fitted across samples.
    """
    a = np.asarray(image, dtype=np.uint8)
    hist = np.bincount(a.ravel(), minlength=256)
    cdf = hist.cumsum()
    lo = cdf[hist > 0][0]
    if cdf[-1] == lo:
        return a
    lut = np.rint((cdf - lo).clip(0) * 255 / (cdf[-1] - lo)).astype(np.uint8)
    return lut[a]


def load_dataset(name, equalization=True):
    if name == "pie_public":
        root = ROOT / "data/pie_public/cmupie_1632"
        files = sorted(root.glob("*/*.jpg"), key=lambda p: (int(p.parent.name), int(p.stem)))
        if len(files) != 1632:
            raise ValueError(f"Expected 1632 PIE mirror images, found {len(files)}")
        images = np.stack([np.asarray(Image.open(p).convert("L")) for p in files])
        labels = np.array([int(p.parent.name) for p in files]) - 1
        sample_ids = [str(p.relative_to(ROOT / "data")) for p in files]
    elif name == "orl":
        d = loadmat(ROOT / "data/ORL_64x64.mat")
        # MATLAB image(:) is column-major; transpose restores original images.
        images = d["fea"].reshape(-1, 64, 64).transpose(0, 2, 1).copy()
        labels = d["gnd"].ravel().astype(int) - 1
        sample_ids = [f"ORL_64x64.mat:row={i}" for i in range(len(images))]
    else:
        # External prepared data use the documented npz schema.
        d = np.load(name, allow_pickle=False)
        images, labels = d["images"], d["labels"]
        if images.dtype != np.uint8:
            raise ValueError("Custom images must be raw uint8, N x H x W")
        sample_ids = (
            d["sample_ids"].tolist() if "sample_ids" in d else [str(i) for i in range(len(images))]
        )
    if equalization:
        images = np.stack([equalize(a) for a in images])
    x = images.astype(np.float32) / 255.0 - 0.5
    return x, labels, sample_ids


def stratified_split(labels, train_per_class, seed):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for c in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == c))
        if len(indices) <= train_per_class:
            raise ValueError("Every class must have at least one test sample")
        train.extend(indices[:train_per_class])
        test.extend(indices[train_per_class:])
    return np.asarray(train), np.asarray(test)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest():
    manifest = {
        "paper": {
            "url": "https://www.ee.columbia.edu/ln/dvmm/publications/09/xu_ebsl.pdf",
            "sha256": sha256(ROOT / "papers/xu_ebsl.pdf"),
        },
        "pie_public": {
            "url": "https://github.com/Gunther17/CMU_PIE-dataset",
            "archive": "cmupie_1632.rar",
            "sha256": sha256(ROOT / "data/cmupie_1632.rar"),
            "images": 1632,
            "classes": 68,
            "per_class": 24,
            "resolution": [64, 64],
            "exact_paper_subset": False,
            "limitations": "Third-party public mirror; class labels from folders. Original camera/illumination identifiers and eye landmarks are absent. Not the paper's 63x20 subset; original alignment cannot be independently verified.",
        },
        "orl": {
            "url": "http://www.cad.zju.edu.cn/home/dengcai/Data/ORL/ORL_64x64.mat",
            "sha256": sha256(ROOT / "data/ORL_64x64.mat"),
            "images": 400,
            "classes": 40,
            "per_class": 10,
            "resolution": [64, 64],
            "exact_paper_subset": False,
            "limitations": "Additional external validation dataset, not used in the paper.",
        },
        "feret": {
            "status": "not_available",
            "original_grayscale_download": False,
            "source": "https://www.nist.gov/itl/products-and-services/color-feret-database",
            "limitations": "NIST does not currently offer original grayscale FERET for download. Color FERET requires a release agreement and account and is a different release.",
        },
        "preprocessing": {
            "histogram_equalization": "per-image, 256 bins, CDF-min, rounded uint8",
            "normalization": "float32 / 255 - 0.5",
            "resize": False,
            "dataset_mean_subtraction_for_bilinear": False,
        },
    }
    (ROOT / "data/manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
