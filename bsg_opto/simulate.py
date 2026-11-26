"""Single-trial (spinalsim) and batched BLAS network simulation backends."""
import numpy as np
from spinalsim import simulate_network, tanh_f_V, stim_varying_individual, gain_const_common
from .config import ChR2Config, NeuronConfig, ProtocolConfig
from .opsin import chr2_fixed_waveform, chr2_current, chr2_params, step_waveform


def simulate_single(W: np.ndarray, I_ext: np.ndarray,
                    neuron: NeuronConfig = NeuronConfig(),
                    seed: int = 5) -> np.ndarray:
    """Run one simulation via BergLab/BSG `spinalsim.simulate_network`."""
    return simulate_network(
        W,
        tau_V=neuron.tau_V,
        t_steps=I_ext.shape[1],
        noise_ampl=neuron.noise_ampl,
        seed=seed,
        stim_func=stim_varying_individual,
        gain_func=gain_const_common,
        I_e=I_ext,
        V_init=neuron.V_init,
        f_V_func=tanh_f_V,
        threshold=neuron.threshold,
        gain=neuron.gain,
        fmax=neuron.fmax,
    )


def run_pulse(W: np.ndarray, stim_idx, amp: float,
              proto: ProtocolConfig = ProtocolConfig(),
              neuron: NeuronConfig = NeuronConfig(),
              chr2: ChR2Config = ChR2Config(),
              seed: int = 5, opsin: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Run a single light or current pulse from rest."""
    N = W.shape[0]
    I = np.zeros((N, proto.T), dtype=float)
    idx = np.asarray(stim_idx, dtype=int)
    if idx.size:
        wave = chr2_fixed_waveform(proto.T, proto.stim_on, proto.stim_off, chr2) if opsin \
            else step_waveform(proto.T, proto.stim_on, proto.stim_off)
        I[idx, :] = amp * wave
    return simulate_single(W, I, neuron, seed=seed), I


def run_tonic_opsin(W: np.ndarray, opsin_idx, D: float,
                    proto: ProtocolConfig = ProtocolConfig(),
                    neuron: NeuronConfig = NeuronConfig(),
                    chr2: ChR2Config = ChR2Config(),
                    phi=None, i_max: float | None = None,
                    seed: int = 5, light: bool = True,
                    baseline_offsets: np.ndarray | None = None) -> np.ndarray:
    """Run a network with tonic drive D (+ optional per-cell offsets) and ChR2 light."""
    N = W.shape[0]
    if baseline_offsets is None:
        I = np.full((N, proto.T), float(D), dtype=float)
    else:
        I = np.repeat((float(D) + np.asarray(baseline_offsets, dtype=float))[:, None],
                      proto.T, axis=1)
    idx = np.asarray(opsin_idx, dtype=int)
    if light and idx.size:
        amp = chr2.i_max if i_max is None else float(i_max)
        if phi is None:
            I[idx, :] += amp * chr2_fixed_waveform(proto.T, proto.stim_on, proto.stim_off, chr2)
        else:
            I[idx, :] += chr2_current(proto.T, proto.stim_on, proto.stim_off, phi, amp, chr2)
    return simulate_single(W, I, neuron, seed=seed)


# ---------------------------------------------------------------- batched BLAS backend
def f_rate(V, th: float, gain: float, fmax: float):
    x = V - th
    return th + np.where(x <= 0, th * np.tanh(gain * x / th), fmax * np.tanh(gain * x / fmax))


def simulate_batch(W: np.ndarray, T: int, ext_fn, n_trials: int,
                   neuron: NeuronConfig = NeuronConfig(),
                   seed: int = 0, windows: dict | None = None,
                   record: dict | None = None, rec_every: int = 10,
                   snap_cols=(0,), dtype=np.float32, extra_input=None) -> dict:
    """Batched BLAS re-implementation of `spinalsim.simulate_network` for large N
    or large trial sweeps. Verified identical to spinalsim at zero noise.
    """
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    Wd = W.astype(dtype)
    snap_cols = list(snap_cols)
    windows = windows or {}
    V = np.full((N, n_trials), neuron.V_init, dtype=dtype)
    R = np.full((N, n_trials), neuron.V_init, dtype=dtype)
    acc = {k: [np.zeros((N, n_trials)), np.zeros((N, n_trials)), np.zeros((N, n_trials))]
           for k in windows}
    rec = {k: np.zeros((T, len(fn(R, V)))) for k, fn in (record or {}).items()}
    snaps = np.zeros((len(snap_cols), N, T // rec_every), dtype=dtype)
    a = dtype(1.0 / neuron.tau_V)
    for t in range(1, T):
        I = Wd @ R
        if extra_input is not None:
            I += extra_input(R)
        if neuron.noise_ampl > 0:
            I += neuron.noise_ampl * rng.standard_normal((N, n_trials), dtype=dtype)
        I += ext_fn(t)
        V += a * (I - V)
        R = f_rate(V, neuron.threshold, neuron.gain, neuron.fmax).astype(dtype)
        for k, (t0, t1) in windows.items():
            if t0 <= t < t1:
                acc[k][0] += R
                acc[k][1] += R.astype(np.float64) ** 2
                acc[k][2] += V
        for k, fn in (record or {}).items():
            rec[k][t] = fn(R, V)
        if t % rec_every == 0 and snap_cols:
            snaps[:, :, t // rec_every] = R[:, snap_cols].T
    out = {}
    for k, (t0, t1) in windows.items():
        n = t1 - t0
        m = acc[k][0] / n
        out[k] = dict(
            mean=m,
            sd=np.sqrt(np.maximum(acc[k][1] / n - m ** 2, 0.0)),
            Vmean=acc[k][2] / n,
        )
    out['rec'] = rec
    out['snaps'] = snaps
    return out


def make_batched_ext(N: int, cols, proto: ProtocolConfig = ProtocolConfig(),
                     chr2: ChR2Config = ChR2Config(), light: bool = True):
    """Build an `ext_fn(t)` closure for `simulate_batch`.

    Each element of `cols` is a dict or tuple:
      - dict with keys: 'base' (scalar or length-N array), 'op' (indices or None),
        'phi' (array or None), 'i_max' (float, optional -> defaults to chr2.i_max)
    """
    base_cols = []
    rows, cidx, amp, ss, td = [], [], [], [], []
    for j, c in enumerate(cols):
        b = c['base']
        base_cols.append(np.full(N, float(b), dtype=np.float32) if np.ndim(b) == 0
                         else np.asarray(b, dtype=np.float32))
        op = c.get('op')
        if light and op is not None and len(op):
            phi = np.asarray(c['phi'], dtype=float)
            imax = float(c.get('i_max', chr2.i_max))
            s_, t_ = chr2_params(phi, chr2)
            rows.append(np.asarray(op, dtype=int))
            cidx.append(np.full(len(op), j, dtype=int))
            amp.append(imax * phi)
            ss.append(s_)
            td.append(t_)
    base_mat = np.stack(base_cols, axis=1)
    has_light = len(rows) > 0
    if has_light:
        rows = np.concatenate(rows)
        cidx = np.concatenate(cidx)
        amp = np.concatenate(amp)
        ss = np.concatenate(ss)
        td = np.concatenate(td)
        off_val = amp * (ss + (1.0 - ss) * np.exp(-(proto.stim_off - 1 - proto.stim_on) / td))

    def ext(t: int):
        E = base_mat.copy()
        if has_light and t >= proto.stim_on:
            if t < proto.stim_off:
                if chr2.desensitize:
                    v = amp * (ss + (1.0 - ss) * np.exp(-(t - proto.stim_on) / td))
                else:
                    v = amp
            else:
                v = off_val * np.exp(-(t - proto.stim_off + 1) / chr2.tau_off) if chr2.desensitize \
                    else np.zeros_like(amp)
            E[rows, cidx] += v.astype(np.float32)
        return E

    return ext
