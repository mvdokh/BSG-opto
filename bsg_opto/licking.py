"""6-8 Hz Mammalian Licking Balanced Sequence Generator (Licking BSG) model.

Adapts the Lindén et al. (2022) balanced Dale-law motor network from slow spinal
scratching (~0.5-0.8 Hz) to fast mammalian brainstem rhythmic licking (6-8 Hz,
IRt/hypoglossal XII premotor rhythm generator) via two synergistic mechanisms:
  1. Faster mammalian membrane/synaptic timescale: tau_V = 18 ms (vs. 50 ms).
  2. Phase-shifted rotational E/I connectivity (N = 300, C = 0.15, W_bulk = 0.58,
     rot_alpha = 0.68, rot_psi_deg = 40 deg) obeying strict Dale's law and exact
     per-neuron E/I in-degree balance (W @ 1 = 0), which isolates a fast complex-
     conjugate rotational eigenmode (lambda_1 ≈ 1.00 ± 0.85i) cleanly outside the
     random bulk.

Fully compatible with `dt = 1 ms` `spinalsim` / `simulate_batch` and all `bsg_opto`
analysis, optogenetic stimulation, and XII nerve readout helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .config import NeuronConfig


@dataclass(frozen=True)
class LickingNetworkConfig:
    """Parameters for the 6-8 Hz balanced Dale-law brainstem licking network."""
    N: int = 300
    C: float = 0.15
    frac_inh: float = 0.5
    g: float = 1.0
    W_bulk: float = 0.58
    rot_alpha: float = 0.68
    rot_psi_deg: float = 40.0
    seed: int = 1

    @property
    def N_inh(self) -> int:
        return int(self.N * self.frac_inh)

    @property
    def N_exc(self) -> int:
        return self.N - self.N_inh


def licking_neuron_config(**overrides) -> NeuronConfig:
    """Return a standard `NeuronConfig` calibrated for 6-8 Hz brainstem licking (tau_V = 18 ms)."""
    base = NeuronConfig(tau_V=18.0, noise_ampl=4.0, threshold=20.0, gain=1.2, fmax=50.0, V_init=0.0)
    return replace(base, **overrides) if overrides else base


def generate_W_licking(
    N: int = 300,
    C: float = 0.15,
    frac_inh: float = 0.5,
    g: float = 1.0,
    W_bulk: float = 0.58,
    rot_alpha: float = 0.68,
    rot_psi_deg: float = 40.0,
    seed: int = 1,
) -> np.ndarray:
    """Generate a sparse, strictly Dale-law balanced connectivity matrix W for 6-8 Hz licking.

    Every neuron receives exactly n_e = round(C * N_exc) excitatory synapses (+w_e) and
    n_i = round(C * N_inh) inhibitory synapses (-w_i), maintaining exact row-sum E/I balance
    (W @ 1 = 0 for g = 1, frac_inh = 0.5). Presynaptic E neurons preferentially project
    forward along the protraction-retraction phase circle (+rot_psi_deg), while I neurons
    preferentially inhibit the anti-phase (+rot_psi_deg + 180 deg), adding constructively
    to create an isolated fast complex-conjugate eigenmode.
    """
    rng = np.random.default_rng(seed)
    N_inh = int(N * frac_inh)
    N_exc = N - N_inh
    n_e = int(round(C * N_exc))
    n_i = int(round(C * N_inh))

    th_E = np.linspace(-np.pi, np.pi, N_exc, endpoint=False)
    th_I = np.linspace(-np.pi, np.pi, N_inh, endpoint=False)
    th = np.concatenate([th_E, th_I])
    psi = np.deg2rad(rot_psi_deg)

    W = np.zeros((N, N), dtype=np.float64)
    w0 = W_bulk / np.sqrt(C * (1.0 - C) * (1.0 + g ** 2) / 2.0)
    w_e = w0 / np.sqrt(float(N))
    w_i = -g * w0 / np.sqrt(float(N))

    for i in range(N):
        p_e = np.clip(1.0 + rot_alpha * np.cos(th[i] - th_E - psi), 0.02, None)
        p_e /= p_e.sum()
        W[i, rng.choice(N_exc, size=n_e, replace=False, p=p_e)] = w_e

        p_i = np.clip(1.0 - rot_alpha * np.cos(th[i] - th_I - psi), 0.02, None)
        p_i /= p_i.sum()
        W[i, N_exc + rng.choice(N_inh, size=n_i, replace=False, p=p_i)] = w_i

    return W


class LickingNetwork:
    """Drop-in BalancedNetwork-compatible 6-8 Hz licking network realization."""

    def __init__(self, cfg: LickingNetworkConfig = LickingNetworkConfig(),
                 W: np.ndarray | None = None):
        self.cfg = cfg
        self.N = cfg.N
        self.N_exc = cfg.N_exc
        self.N_inh = cfg.N_inh
        self.W = generate_W_licking(
            N=cfg.N, C=cfg.C, frac_inh=cfg.frac_inh, g=cfg.g,
            W_bulk=cfg.W_bulk, rot_alpha=cfg.rot_alpha,
            rot_psi_deg=cfg.rot_psi_deg, seed=cfg.seed,
        ) if W is None else W
        self.W_E = float(self.W[:, :self.N_exc].max())
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
        return np.concatenate([
            np.argsort(self.phase[:self.N_exc]),
            self.N_exc + np.argsort(self.phase[self.N_exc:]),
        ])

    def with_seed(self, seed: int) -> 'LickingNetwork':
        return LickingNetwork(replace(self.cfg, seed=seed))
