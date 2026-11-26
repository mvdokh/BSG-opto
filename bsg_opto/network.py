"""Network construction, dominant eigenmode, and neuron phase sorting."""
from dataclasses import replace
import numpy as np
from spinalsim import generate_W_dale_by_radius
from .config import NetworkConfig


class BalancedNetwork:
    """One balanced E/I network realization with eigenmode phase sorting."""

    def __init__(self, cfg: NetworkConfig = NetworkConfig(), W: np.ndarray | None = None,
                 _phase_ref: 'BalancedNetwork | None' = None):
        self.cfg = cfg
        self.N = cfg.N
        self.N_exc = cfg.N_exc
        self.N_inh = cfg.N_inh
        self.W = generate_W_dale_by_radius(**cfg.to_bsg_kwargs()) if W is None else W
        self.W_E = float(self.W[:, :self.N_exc].max())
        if _phase_ref is not None:
            self.eigvals = _phase_ref.eigvals
            self.osc_mode = _phase_ref.osc_mode
            self.phase = _phase_ref.phase
            self.phase_sort = _phase_ref.phase_sort
        else:
            ev, evec = np.linalg.eig(self.W)
            k = int(ev.real.argmax())
            self.eigvals = ev
            self.osc_mode = np.conj(evec[:, k])
            self.phase = np.angle(self.osc_mode)
            self.phase_sort = np.argsort(self.phase)

    @property
    def leading_eigval(self) -> complex:
        return complex(self.eigvals[int(self.eigvals.real.argmax())])

    def ei_phase_sort(self) -> np.ndarray:
        """Sort E cells (0..N_exc-1) and I cells (N_exc..N-1) separately by phase."""
        return np.concatenate([
            np.argsort(self.phase[:self.N_exc]),
            self.N_exc + np.argsort(self.phase[self.N_exc:]),
        ])

    def row_lookup(self, order: np.ndarray | None = None) -> np.ndarray:
        """Inverse permutation: row_lookup[neuron_id] = display row index."""
        order = self.phase_sort if order is None else order
        inv = np.empty(self.N, dtype=int)
        inv[order] = np.arange(self.N)
        return inv

    def with_weights(self, W_new: np.ndarray, keep_phase: bool = True) -> 'BalancedNetwork':
        """Return a network with modified W, optionally keeping original phase ordering."""
        return BalancedNetwork(self.cfg, W=W_new, _phase_ref=self if keep_phase else None)

    def with_seed(self, seed: int) -> 'BalancedNetwork':
        return BalancedNetwork(replace(self.cfg, seed=seed))
