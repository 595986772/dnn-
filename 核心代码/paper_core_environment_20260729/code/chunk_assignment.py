"""Integer chunk assignment for window-level edge workload scheduling."""

import numpy as np
import torch


def normalize_simplex(a, eps=1e-12):
    a = np.asarray(a, dtype=np.float64)
    a = np.clip(a, 0.0, None)
    s = float(a.sum())
    if s <= eps:
        return np.ones_like(a, dtype=np.float64) / max(1, len(a))
    return a / s


def round_to_chunks_largest_remainder(a, M, eps=1e-12):
    """Map a continuous allocation ratio to integer micro-task counts."""
    if M is None or int(M) <= 0:
        a_norm = normalize_simplex(a, eps=eps)
        return None, a_norm, 0.0

    M = int(M)
    a_norm = normalize_simplex(a, eps=eps)
    raw = a_norm * M
    n = np.floor(raw).astype(np.int64)
    remain = int(M - n.sum())

    if remain > 0:
        frac = raw - n
        idx = np.argsort(-frac)[:remain]
        n[idx] += 1
    elif remain < 0:
        idx = np.argsort(raw - n)[:abs(remain)]
        for j in idx:
            if n[j] > 0:
                n[j] -= 1

    if n.sum() <= 0:
        n[int(np.argmax(a_norm))] = M

    diff = int(M - n.sum())
    while diff > 0:
        j = int(np.argmax(a_norm - n / max(1, M)))
        n[j] += 1
        diff -= 1
    while diff < 0:
        positive = np.where(n > 0)[0]
        if len(positive) == 0:
            break
        j = int(positive[np.argmax(n[positive] / max(1, M) - a_norm[positive])])
        n[j] -= 1
        diff += 1

    x = n.astype(np.float64) / float(M)
    rounding_l1 = float(np.abs(x - a_norm).sum())
    return n, x, rounding_l1


def stochastic_round_to_chunks_np(a, M, rng=None, eps=1e-12):
    """Stochastically round a simplex action to integer chunks while preserving sum."""
    if M is None or int(M) <= 0:
        return None
    rng = rng if rng is not None else np.random.default_rng()
    M = int(M)
    a_norm = normalize_simplex(a, eps=eps)
    raw = a_norm * M
    n = np.floor(raw).astype(np.int64)
    remain = int(M - n.sum())
    if remain > 0:
        frac = np.clip(raw - n, 0.0, None)
        if float(frac.sum()) <= eps:
            idx = np.argsort(-a_norm)[:remain]
        else:
            p = frac / float(frac.sum())
            idx = rng.choice(len(a_norm), size=remain, replace=False, p=p)
        n[idx] += 1
    diff = int(M - n.sum())
    while diff > 0:
        n[int(rng.integers(0, len(n)))] += 1
        diff -= 1
    while diff < 0:
        positive = np.where(n > 0)[0]
        if len(positive) == 0:
            break
        n[int(rng.choice(positive))] -= 1
        diff += 1
    return n


def assign_sized_chunks_to_targets(a, chunk_sizes, eps=1e-12):
    """Assign indivisible variable-size chunks to match target byte fractions.

    Chunks are processed largest first. Each chunk is placed on the selected
    server with the largest remaining byte deficit. The mapper never activates
    a server whose target fraction is zero and does not inspect delay, energy,
    queues, or channel state.
    """
    a_norm = normalize_simplex(a, eps=eps)
    sizes = np.asarray(chunk_sizes, dtype=np.float64)
    if sizes.ndim != 1 or len(sizes) == 0:
        raise ValueError("chunk_sizes must be a non-empty 1-D array")
    if not np.all(np.isfinite(sizes)) or np.any(sizes <= 0.0):
        raise ValueError("chunk_sizes must contain finite positive values")

    support = np.flatnonzero(a_norm > eps)
    if len(support) == 0:
        support = np.array([int(np.argmax(a_norm))], dtype=np.int64)
    total = float(sizes.sum())
    target = a_norm[support] * total
    target = target / max(float(target.sum()), eps) * total
    assigned = np.zeros(len(support), dtype=np.float64)
    counts_local = np.zeros(len(support), dtype=np.int64)
    assignment = np.empty(len(sizes), dtype=np.int64)

    order = np.argsort(-sizes, kind="stable")
    for chunk_idx in order:
        deficit = target - assigned
        local_idx = int(np.argmax(deficit))
        assigned[local_idx] += float(sizes[chunk_idx])
        counts_local[local_idx] += 1
        assignment[chunk_idx] = int(support[local_idx])

    counts = np.zeros(len(a_norm), dtype=np.int64)
    loads = np.zeros(len(a_norm), dtype=np.float64)
    counts[support] = counts_local
    loads[support] = assigned
    return counts, loads, assignment


def effective_chunk_stats(a_raw, a_exec, n_chunks=None):
    a_raw = normalize_simplex(a_raw)
    a_exec = normalize_simplex(a_exec)
    active = a_exec > 1e-12
    entropy = float(-(a_exec[active] * np.log(a_exec[active] + 1e-12)).sum()) if active.any() else 0.0
    out = {
        "effective_K": int(active.sum()),
        "effective_max_frac": float(a_exec.max()) if len(a_exec) else 0.0,
        "effective_entropy": entropy,
        "rounding_l1": float(np.abs(a_exec - a_raw).sum()),
    }
    if n_chunks is not None:
        n = np.asarray(n_chunks, dtype=np.int64)
        out["chunk_counts"] = n.tolist()
        out["min_positive_chunks"] = int(n[n > 0].min()) if np.any(n > 0) else 0
        out["max_chunks"] = int(n.max()) if len(n) else 0
    return out


def deterministic_round_to_chunks_tensor(a, M):
    """No-gradient largest-remainder rounding for a batch of torch simplex actions."""
    if a.ndim == 1:
        a_2d = a.unsqueeze(0)
        squeeze = True
    elif a.ndim == 2:
        a_2d = a
        squeeze = False
    else:
        raise ValueError("deterministic_round_to_chunks_tensor expects 1-D or 2-D tensor")

    a_np = a_2d.detach().cpu().numpy()
    m_np = np.asarray(M.detach().cpu().numpy() if torch.is_tensor(M) else M, dtype=np.int64)
    if m_np.ndim == 0:
        m_np = np.full((a_np.shape[0],), int(m_np), dtype=np.int64)
    out = []
    for row, m in zip(a_np, m_np):
        _, x, _ = round_to_chunks_largest_remainder(row, int(m))
        out.append(x)
    hard = torch.as_tensor(np.asarray(out, dtype=np.float32), dtype=a.dtype, device=a.device)
    return hard.squeeze(0) if squeeze else hard


def chunk_round_st_tensor(a, M):
    """Forward executes hard integer chunks; backward uses the continuous action."""
    hard = deterministic_round_to_chunks_tensor(a, M)
    return hard.detach() + a - a.detach()


def round_to_chunks_tensor_detached(a, M):
    """Hard deterministic tensor chunk rounding with no gradient path."""
    return deterministic_round_to_chunks_tensor(a, M).detach()


def round_to_chunks_tensor_st(a, M):
    """Straight-through deterministic tensor chunk rounding."""
    return chunk_round_st_tensor(a, M)
