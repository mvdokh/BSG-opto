"""Connectivity perturbation helpers (rewiring and synaptic weight scaling)."""
import numpy as np


def boost_opsin_to_I(W: np.ndarray, opsin_idx: np.ndarray, N_exc: int,
                     p: float, C0: float = 0.1, mode: str = 'rewire',
                     seed: int = 0) -> np.ndarray:
    """Raise opsin E -> I connection probability from C0 to p.

    mode='rewire': remove a random non-opsin E->I synapse for each new one
        so every I cell keeps its exact excitatory in-degree.
    mode='add': add new synapses without removing existing ones.
    """
    Wn = W.copy()
    rng = np.random.default_rng(seed)
    W_E = float(W[:, :N_exc].max())
    p_add = max(p - C0, 0.0) / (1.0 - C0)
    opset = set(np.asarray(opsin_idx).tolist())
    for i in range(N_exc, W.shape[0]):
        new = [j for j in opsin_idx if Wn[i, j] == 0 and rng.random() < p_add]
        if not new:
            continue
        Wn[i, new] = W_E
        if mode == 'rewire':
            cand = [j for j in np.flatnonzero(Wn[i, :N_exc]) if j not in opset]
            Wn[i, rng.choice(cand, min(len(new), len(cand)), replace=False)] = 0.0
    return Wn


def boost_opsin_to_all(W: np.ndarray, opsin_idx: np.ndarray, N_exc: int,
                       p: float, C0: float = 0.1, seed: int = 0) -> np.ndarray:
    """Raise opsin E -> ALL cells (E and I) connection probability from C0 to p.

    In-degree fixed: each new opsin->i synapse replaces a random existing
    non-opsin E->i synapse. No self-connections added.
    """
    Wn = W.copy()
    rng = np.random.default_rng(seed)
    W_E = float(W[:, :N_exc].max())
    p_add = max(p - C0, 0.0) / (1.0 - C0)
    opset = set(np.asarray(opsin_idx).tolist())
    for i in range(W.shape[0]):
        new = [j for j in opsin_idx if j != i and Wn[i, j] == 0 and rng.random() < p_add]
        if not new:
            continue
        Wn[i, new] = W_E
        cand = [j for j in np.flatnonzero(Wn[i, :N_exc]) if j not in opset]
        Wn[i, rng.choice(cand, min(len(new), len(cand)), replace=False)] = 0.0
    return Wn


def scale_opsin_to_I_weights(W: np.ndarray, opsin_idx: np.ndarray,
                             N_exc: int, k: float) -> np.ndarray:
    """Multiply all existing synapses from opsin cells onto I cells by factor k."""
    Wk = W.copy()
    Wk[N_exc:, np.asarray(opsin_idx, dtype=int)] *= k
    return Wk


def make_batched_opsin_to_I_extra(W: np.ndarray, ops_list, N_exc: int, k: float):
    """Build an `extra_input(R)` closure for `simulate_batch` when each trial has
    its own opsin->I synapses scaled by k (returns None if k == 1).
    """
    if k == 1:
        return None
    ops = np.stack(ops_list)                                               # (B, n_op)
    G = np.stack([(k - 1.0) * W[N_exc:, op] for op in ops]).astype(np.float32)  # (B, N_I, n_op)
    bidx = np.arange(len(ops_list))[:, None]

    def extra(R):
        add = np.matmul(G, R[ops, bidx][:, :, None])[:, :, 0]              # (B, N_I)
        out = np.zeros_like(R)
        out[N_exc:] = add.T
        return out

    return extra
