"""Linear stability analysis, rhythmicity metrics, PCA, and nerve readouts."""
import numpy as np
from spinalsim import calc_PCA, simple_readout
from .config import NeuronConfig


def transfer_slope(V, neuron: NeuronConfig = NeuronConfig()) -> np.ndarray:
    """Derivative df/dV of tanh_f_V (vectorized)."""
    V = np.asarray(V, dtype=float)
    x = V - neuron.threshold
    return np.where(
        x <= 0,
        neuron.gain / np.cosh(neuron.gain * x / neuron.threshold) ** 2,
        neuron.gain / np.cosh(neuron.gain * x / neuron.fmax) ** 2,
    )


def invert_transfer(R, neuron: NeuronConfig = NeuronConfig()) -> np.ndarray:
    """Invert tanh_f_V to recover membrane potential V from firing rate R."""
    th, gn, fm = neuron.threshold, neuron.gain, neuron.fmax
    y = np.clip(np.asarray(R, dtype=float) - th, -th + 1e-6, fm - 1e-6)
    with np.errstate(invalid='ignore'):
        x = np.where(y <= 0, th / gn * np.arctanh(y / th), fm / gn * np.arctanh(y / fm))
    return x + th


def jacobian_eigs(W: np.ndarray, Vmean: np.ndarray,
                  neuron: NeuronConfig = NeuronConfig()) -> np.ndarray:
    """Eigenvalues of the linearized recurrent dynamics diag(f'(Vmean)) @ W."""
    return np.linalg.eigvals(transfer_slope(Vmean, neuron)[:, None] * W)


def leading_eig(W: np.ndarray, Vmean: np.ndarray,
                neuron: NeuronConfig = NeuronConfig()) -> complex:
    lam = jacobian_eigs(W, Vmean, neuron)
    return complex(lam[int(lam.real.argmax())])


def leading_real_eig(W: np.ndarray, Vmean: np.ndarray,
                     neuron: NeuronConfig = NeuronConfig()) -> float:
    return float(jacobian_eigs(W, Vmean, neuron).real.max())


def lam_from_rates(W: np.ndarray, R: np.ndarray, win: tuple[int, int],
                   neuron: NeuronConfig = NeuronConfig()) -> complex:
    Vmean = invert_transfer(R[:, win[0]:win[1]], neuron).mean(1)
    return leading_eig(W, Vmean, neuron)


def unlit_mean(values: np.ndarray, stim_idx=None) -> float:
    """Mean of a length-N per-neuron array excluding stimulated indices."""
    if stim_idx is None or len(stim_idx) == 0:
        return float(np.mean(values))
    mask = np.ones(len(values), dtype=bool)
    mask[np.asarray(stim_idx, dtype=int)] = False
    return float(values[mask].mean())


def compute_metrics(R: np.ndarray, stim_idx, win: tuple[int, int]) -> dict:
    """Rhythmicity (mean temporal SD), firing rates, and peak PSD frequency on unlit cells."""
    N = R.shape[0]
    stim_idx = np.asarray(stim_idx, dtype=int)
    non = np.setdiff1d(np.arange(N), stim_idx)
    seg = R[:, win[0]:win[1]]
    segc = seg - seg.mean(1, keepdims=True)
    F = np.fft.rfft(segc[non], axis=1)
    fr = np.fft.rfftfreq(seg.shape[1], 1e-3)
    psd = np.mean(np.abs(F) ** 2, 0)
    band = (fr > 0.2) & (fr < 10.0)
    return dict(
        rate_stim=float(seg[stim_idx].mean()) if stim_idx.size else float('nan'),
        rate_nonstim=float(seg[non].mean()),
        osc_sd=float(seg[non].std(1).mean()),
        peak_freq=float(fr[band][psd[band].argmax()]),
    )


def compute_pca(R: np.ndarray, win: tuple[int, int]):
    """Run BergLab/BSG PCA on R[:, win[0]:win[1]]."""
    return calc_PCA(R[:, win[0]:win[1]])


def nerve_readout(net, R: np.ndarray, mode: str = 'limb', seed: int = 0) -> dict:
    """Paper's two-opposite-phase-pool nerve readout (±pi/2, spread pi/8).

    mode='limb': labels ('flex', 'ext'), ylabel 'Nerve'
    mode='xii' : labels ('protract', 'retract'), ylabel 'XII nerve'
                 (hypoglossal protractor/retractor interpretation)
    """
    np.random.seed(seed)
    upper = simple_readout(R, net.phase, net.N_exc,
                           np.pi / 2, -np.pi / 2, np.pi / 8, np.pi / 8)
    lower = simple_readout(R, net.phase, net.N_exc,
                           -np.pi / 2, np.pi / 2, np.pi / 8, np.pi / 8)
    if mode == 'xii':
        labels, ylabel = ('protract', 'retract'), 'XII nerve'
    else:
        labels, ylabel = ('flex', 'ext'), 'Nerve'
    return dict(upper=upper['nerve'], lower=lower['nerve'],
                upper_dict=upper, lower_dict=lower,
                labels=labels, ylabel=ylabel)
