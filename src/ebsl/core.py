"""Equations (1)--(10), shared permutations, and Algorithm 1.

Permutation convention: perm[q] is the original source pixel at destination q.
Sparse minimum weight perfect matching solves the same integral transportation
problem as the paper's simplex LP, without constructing a dense constraint matrix.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def top_eigenvectors(a, k):
    # Eigensystems are only 64 x 64. Double precision prevents unstable bases.
    values, vectors = torch.linalg.eigh((a.double() + a.double().T) * 0.5)
    return vectors[:, -k:].flip(1).to(a.dtype), values.flip(0).clamp_min(0)


def reconstruct(x, u, v):
    return u @ (u.T @ x @ v) @ v.T


def error(x, u, v):
    return float((x - reconstruct(x, u, v)).double().square().sum().item())


def fit_bilinear(x, ranks, mode="glram", initial=None, max_iter=100, tol=1e-7):
    """Uncentered Eq. (1). Input normalization is explicitly the caller's job.

    2DPCA uses U=I (paper section 2.1); its feature shape is m x d.
    GLRAM alternates optimal orthogonal left/right updates, warm-started for ER.
    """
    _, h, w = x.shape
    r, c = ranks
    if not (1 <= r <= h and 1 <= c <= w):
        raise ValueError("Invalid rank")
    if mode == "2dpca":
        u = torch.eye(h, dtype=x.dtype, device=x.device)
        v, eig_v = top_eigenvectors(torch.einsum("nhi,nhj->ij", x, x), c)
        return u, v, {"inner_iterations": 1, "eigen_energy": float(eig_v[:c].sum() / eig_v.sum())}
    if mode != "glram":
        raise ValueError(mode)
    if initial is None:
        u = torch.eye(h, dtype=x.dtype, device=x.device)[:, :r]
        v = torch.eye(w, dtype=x.dtype, device=x.device)[:, :c]
    else:
        u, v = (t.clone() for t in initial)
    previous = error(x, u, v)
    for it in range(max_iter):
        right = x @ v
        u, eig_u = top_eigenvectors(torch.einsum("nic,njc->ij", right, right), r)
        left = u.T @ x
        v, eig_v = top_eigenvectors(torch.einsum("nri,nrj->ij", left, left), c)
        current = error(x, u, v)
        if abs(previous - current) <= tol * max(previous, 1e-12):
            break
        previous = current
    # Energy definition in Sec. 4.1.2, from final conditional covariance spectra.
    right, left = x @ v, u.T @ x
    _, eig_u = top_eigenvectors(torch.einsum("nic,njc->ij", right, right), r)
    _, eig_v = top_eigenvectors(torch.einsum("nri,nrj->ij", left, left), c)
    energy = (eig_u[:r].sum() / eig_u.sum()) * (eig_v[:c].sum() / eig_v.sum())
    return u, v, {"inner_iterations": it + 1, "eigen_energy": float(energy)}


def spatial_edges(h, w, radius=2):
    grid = np.arange(h * w).reshape(h, w)
    rows, cols = [], []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            # Radius 2 sqrt(2), exactly the 5x5 neighborhood used in the paper.
            a, b = max(0, -dr), min(h, h - dr)
            c, d = max(0, -dc), min(w, w - dc)
            rows.append(grid[a:b, c:d].ravel())
            cols.append(grid[a + dr : b + dr, c + dc : d + dc].ravel())
    return np.concatenate(rows), np.concatenate(cols)


def assignment_edges(x, target, neighborhood, neighbors=25):
    """Eq. (5): sum over samples, source x[:,p] to reconstruction[:,q]."""
    n, h, w = x.shape
    a, b = x.reshape(n, -1), target.reshape(n, -1)
    size = h * w
    if neighborhood == "ln":
        rows, cols = spatial_edges(h, w)
        pieces = []
        # Limit temporary tensor size to roughly 32 MiB for float32.
        chunk = max(32, min(4096, 8_000_000 // n))
        for start in range(0, len(rows), chunk):
            p = torch.as_tensor(rows[start : start + chunk], device=x.device)
            q = torch.as_tensor(cols[start : start + chunk], device=x.device)
            pieces.append((a[:, p] - b[:, q]).square().sum(0).double().cpu().numpy())
        costs = np.concatenate(pieces)
    elif neighborhood == "nn":
        # Eq. (10): p itself plus k-1 targets with smallest Eq. (5) cost.
        k = min(neighbors, size)
        pieces, row_list, col_list = [], [], []
        b2 = b.square().sum(0)
        for start in range(0, size, 256):
            stop = min(start + 256, size)
            aa = a[:, start:stop]
            cost = (aa.square().sum(0)[:, None] + b2[None, :] - 2 * aa.T @ b).clamp_min(0)
            idx = torch.arange(stop - start, device=x.device)
            diag_cols = torch.arange(start, stop, device=x.device)
            # Reserve identity explicitly instead of counting it among the k-1 neighbors.
            cost[idx, diag_cols] = torch.inf
            _, cc = torch.topk(cost, k - 1, dim=1, largest=False, sorted=True)
            cc = torch.cat([diag_cols[:, None], cc], 1)
            rr = diag_cols[:, None].expand_as(cc)
            # Recompute shortlisted costs directly, retaining numerical fidelity.
            vals = (a[:, rr.reshape(-1)] - b[:, cc.reshape(-1)]).square().sum(0)
            pieces.append(vals.double().cpu().numpy())
            row_list.append(rr.reshape(-1).cpu().numpy())
            col_list.append(cc.reshape(-1).cpu().numpy())
        rows, cols, costs = (
            np.concatenate(row_list),
            np.concatenate(col_list),
            np.concatenate(pieces),
        )
    else:
        raise ValueError(neighborhood)
    return rows, cols, costs


def solve_assignment(size, rows, cols, costs):
    # Sparse solver treats zero as a missing edge. Adding the same positive
    # constant to EVERY allowed edge preserves all perfect matching objectives.
    scale = max(float(np.max(costs)), 1e-30)
    graph = csr_matrix((costs / scale + 1.0, (rows, cols)), shape=(size, size))
    src, dst = min_weight_full_bipartite_matching(graph)
    gather = np.empty(size, dtype=np.int64)
    gather[dst] = src
    if not np.array_equal(np.sort(gather), np.arange(size)):
        raise RuntimeError("Matching is not bijective")
    return gather


@dataclass
class BilinearModel:
    u: torch.Tensor
    v: torch.Tensor
    perm: np.ndarray
    shape: tuple
    history: list = field(default_factory=list)
    stop_reason: str = "baseline"

    def rearrange(self, x):
        idx = torch.as_tensor(self.perm, device=x.device)
        return x.reshape(len(x), -1)[:, idx].reshape(-1, *self.shape)

    def transform(self, x):
        return self.u.T @ self.rearrange(x) @ self.v

    def inverse_transform(self, z):
        out = self.u @ z @ self.v.T
        inverse = torch.as_tensor(np.argsort(self.perm), device=out.device)
        return out.reshape(len(out), -1)[:, inverse].reshape(-1, *self.shape)

    def reconstruct(self, x):
        return self.inverse_transform(self.transform(x))

    def save(self, path):
        np.savez_compressed(
            path,
            u=self.u.cpu().numpy(),
            v=self.v.cpu().numpy(),
            perm=self.perm,
            shape=np.asarray(self.shape),
        )


def fit_er(
    x,
    ranks,
    mode="glram",
    neighborhood="ln",
    max_iter=100,
    tol=1e-7,
    inner_max_iter=100,
    inner_tol=1e-7,
    callback=None,
):
    start = time.perf_counter()
    n, h, w = x.shape
    perm = np.arange(h * w)
    current = x.clone()
    u, v, info = fit_bilinear(current, ranks, mode, max_iter=inner_max_iter, tol=inner_tol)
    objective = error(current, u, v)
    hist = [
        {
            "iteration": 0,
            "sse": objective,
            "rmse": (objective / n) ** 0.5,
            "seconds": time.perf_counter() - start,
            "moved": 0,
            **info,
        }
    ]
    reason = "max_iter"
    for it in range(1, max_iter + 1):
        target = reconstruct(current, u, v)
        rows, cols, costs = assignment_edges(current, target, neighborhood)
        gather = solve_assignment(h * w, rows, cols, costs)
        moved = int(np.count_nonzero(gather != np.arange(h * w)))
        if not moved:
            reason = "identity"
            break
        idx = torch.as_tensor(gather, device=x.device)
        proposed = current.reshape(n, -1)[:, idx].reshape(n, h, w)
        surrogate = float((proposed - target).double().square().sum())
        if surrogate > objective + 2e-6 * max(objective, 1e-12):
            raise RuntimeError(f"Assignment increased surrogate: {surrogate} > {objective}")
        new_u, new_v, info = fit_bilinear(
            proposed, ranks, mode, initial=(u, v), max_iter=inner_max_iter, tol=inner_tol
        )
        new_objective = error(proposed, new_u, new_v)
        if new_objective > objective:
            if new_objective > objective + 2e-6 * max(objective, 1e-12):
                raise RuntimeError("Warm-started bilinear update increased objective")
            reason = "numerical_plateau"
            break
        improvement = (objective - new_objective) / max(objective, 1e-12)
        current, u, v, objective = proposed, new_u, new_v, new_objective
        perm = perm[gather]
        hist.append(
            {
                "iteration": it,
                "sse": objective,
                "rmse": (objective / n) ** 0.5,
                "surrogate_sse": surrogate,
                "moved": moved,
                "seconds": time.perf_counter() - start,
                **info,
            }
        )
        if callback:
            callback(hist[-1])
        if improvement <= tol:
            reason = "relative_tolerance"
            break
    return BilinearModel(u, v, perm, (h, w), hist, reason)


def fit_baseline(x, ranks, mode="glram", **kwargs):
    u, v, info = fit_bilinear(x, ranks, mode, **kwargs)
    sse = error(x, u, v)
    return BilinearModel(
        u,
        v,
        np.arange(x.shape[1] * x.shape[2]),
        tuple(x.shape[1:]),
        [{"iteration": 0, "sse": sse, "rmse": (sse / len(x)) ** 0.5, **info}],
    )


def compression_ratio(n, shape, ranks, mode, perm=None):
    h, w = shape
    r, c = ranks
    scalars = (n * h + w) * c if mode == "2dpca" else n * r * c + h * r + w * c
    extra_bits = 0
    if perm is not None:
        displacement = np.abs(perm - np.arange(h * w))
        full_bits = int(np.ceil(np.log2(h * w)))
        # 2-bit header; 8-bit magnitude for 0<offset<256; full index otherwise.
        extra_bits = int(
            np.sum(2 + np.where(displacement == 0, 0, np.where(displacement < 256, 8, full_bits)))
        )
    return n * h * w / (scalars + extra_bits / 32), extra_bits
