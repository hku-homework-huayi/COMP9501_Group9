"""Sections 3.1--3.3: null-space ER proxies, 2DLDA, and 2DMFA."""

import numpy as np
import torch
from scipy.linalg import eigh

from .core import BilinearModel


def class_statistics(x, labels):
    classes = np.unique(labels)
    means = torch.stack([x[labels == c].mean(0) for c in classes])
    lookup = {int(c): i for i, c in enumerate(classes)}
    class_index = np.array([lookup[int(c)] for c in labels])
    counts = torch.tensor([(labels == c).sum() for c in classes], dtype=x.dtype, device=x.device)
    return means, class_index, counts


def margin_pairs(x, labels, k2=25):
    """For EACH class, choose k2 shortest original-space cross-class pairs."""
    flat = x.reshape(len(x), -1)
    dist = torch.cdist(flat, flat).square()
    pairs = []
    for c in np.unique(labels):
        a, b = np.flatnonzero(labels == c), np.flatnonzero(labels != c)
        sub = dist[
            torch.as_tensor(a, device=x.device)[:, None],
            torch.as_tensor(b, device=x.device)[None, :],
        ]
        kk = min(k2, sub.numel())
        idx = torch.topk(sub.flatten(), kk, largest=False, sorted=True).indices.cpu().numpy()
        pairs.extend(zip(a[idx // len(b)].tolist(), b[idx % len(b)].tolist()))
    return np.asarray(pairs, dtype=np.int64)


def nullspace_projection(x, labels):
    """Eq. (14), using I - Q Q^T instead of materializing the huge null basis.

    Q is the row space of within-class residuals. Work in float64 for rank
    detection; every operation uses TRAINING images and labels only.
    """
    means, ci, _ = class_statistics(x.double(), labels)
    residual = (x.double() - means[ci]).reshape(len(x), -1)
    _, s, vh = torch.linalg.svd(residual, full_matrices=False)
    threshold = float(s.max()) * max(residual.shape) * torch.finfo(torch.float64).eps
    rank = int((s > threshold).sum())
    q = vh[:rank]
    flat = x.double().reshape(len(x), -1)
    projected = flat - (flat @ q.T) @ q
    check = projected.reshape_as(x) - torch.stack(
        [projected[labels == c].mean(0) for c in np.unique(labels)]
    )[ci].reshape_as(x)
    relative = float(
        torch.linalg.vector_norm(check) / torch.linalg.vector_norm(flat).clamp_min(1e-30)
    )
    return projected.reshape_as(x).to(x.dtype), {
        "within_rank": rank,
        "null_within_relative": relative,
    }


def er_proxy(x, labels, kind, k2=25, projected=None):
    if projected is None:
        projected, _ = nullspace_projection(x, labels)
    if kind == "lda":
        means, _, counts = class_statistics(projected, labels)
        return (means - projected.mean(0)) * counts.sqrt()[:, None, None]
    if kind == "mfa":
        pairs = margin_pairs(x, labels, k2)
        return projected[pairs[:, 0]] - projected[pairs[:, 1]]
    raise ValueError(kind)


def discriminant_components(x, labels, kind, k2=25):
    means, ci, counts = class_statistics(x, labels)
    within = x - means[ci]
    if kind == "lda":
        between = (means - x.mean(0)) * counts.sqrt()[:, None, None]
    elif kind == "mfa":
        pairs = margin_pairs(x, labels, k2)
        between = x[pairs[:, 0]] - x[pairs[:, 1]]
        # Sum over all directed within-class pairs with k1=nc-1 equals
        # 2*nc * sum residual outer products. Counts may differ for custom data.
        within = within * (2 * counts[ci]).sqrt()[:, None, None]
    else:
        raise ValueError(kind)
    return within, between


def generalized_basis(sb, sw, rank, ridge=1e-6):
    a, b = sb.double().cpu().numpy(), sw.double().cpu().numpy()
    a, b = (a + a.T) * 0.5, (b + b.T) * 0.5
    reg = ridge * max(np.trace(b) / len(b), 1e-12)
    b = b + reg * np.eye(len(b))
    _, vec = eigh(a, b, subset_by_index=(len(b) - rank, len(b) - 1), driver="gvx")
    vec = vec[:, ::-1].copy()
    vec /= np.linalg.norm(vec, axis=0, keepdims=True).clip(1e-30)
    return torch.as_tensor(vec, device=sb.device, dtype=sb.dtype)


def fit_discriminant(
    x, labels, ranks, kind="lda", permutation=None, iterations=10, ridge=1e-6, k2=25
):
    n, h, w = x.shape
    r, c = ranks
    if permutation is None:
        permutation = np.arange(h * w)
    z = x.reshape(n, -1)[:, torch.as_tensor(permutation, device=x.device)].reshape_as(x)
    within, between = discriminant_components(z, labels, kind, k2)
    u = torch.eye(h, device=x.device, dtype=x.dtype)[:, :r]
    v = torch.eye(w, device=x.device, dtype=x.dtype)[:, :c]
    for _ in range(iterations):
        a, b = within @ v, between @ v
        sw = torch.einsum("nic,njc->ij", a, a)
        sb = torch.einsum("nic,njc->ij", b, b)
        u = generalized_basis(sb, sw, r, ridge)
        a, b = u.T @ within, u.T @ between
        sw = torch.einsum("nri,nrj->ij", a, a)
        sb = torch.einsum("nri,nrj->ij", b, b)
        v = generalized_basis(sb, sw, c, ridge)
    return BilinearModel(u, v, permutation, (h, w))


def nearest_neighbor(train_features, train_labels, test_features):
    a, b = train_features.flatten(1), test_features.flatten(1)
    indices = torch.cdist(b, a).argmin(1).cpu().numpy()
    return np.asarray(train_labels)[indices]


def vector_features(train, test, labels, kind="pca", energy=0.95, k2=25):
    """Vector baselines: 95%-energy PCA, followed by LDA/MFA when requested.

    Dimensions are fixed by TRAINING data; test accuracy never chooses rank.
    """
    a, b = train.flatten(1), test.flatten(1)
    mean = a.mean(0)
    centered = a - mean
    u, s, vh = torch.linalg.svd(centered, full_matrices=False)
    # CUDA cumsum is not deterministic in PyTorch 2.5; the tiny spectrum is
    # accumulated on CPU in float64, while the expensive SVD remains on CUDA.
    spectrum = s.detach().double().cpu().square()
    cumsum = spectrum.cumsum(0) / spectrum.sum()
    rank = int(torch.searchsorted(cumsum, torch.tensor(energy, dtype=cumsum.dtype))) + 1
    a, b = centered @ vh[:rank].T, (b - mean) @ vh[:rank].T
    if kind == "pca":
        return a, b, rank
    aa = a[:, :, None]
    within, between = discriminant_components(aa, labels, kind, k2)
    sw = within[:, :, 0].T @ within[:, :, 0]
    sb = between[:, :, 0].T @ between[:, :, 0]
    dim = min(rank, len(np.unique(labels)) - 1)
    proj = generalized_basis(sb, sw, dim)
    return a @ proj, b @ proj, dim
