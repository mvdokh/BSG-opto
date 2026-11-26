"""Channelrhodopsin photocurrent waveforms and opsin cell selection helpers."""
import numpy as np
from .config import ChR2Config, NeuronConfig, ProtocolConfig


def step_waveform(T_len: int, on: int, off: int) -> np.ndarray:
    """Unit step [on, off)."""
    w = np.zeros(T_len, dtype=float)
    w[on:off] = 1.0
    return w


def chr2_fixed_waveform(T_len: int, on: int, off: int,
                        cfg: ChR2Config = ChR2Config()) -> np.ndarray:
    """Normalized (peak = 1) ChR2 waveform with fixed desensitization."""
    if not cfg.desensitize:
        return step_waveform(T_len, on, off)
    w = np.zeros(T_len, dtype=float)
    t = np.arange(off - on)
    w[on:off] = cfg.ss_frac + (1.0 - cfg.ss_frac) * np.exp(-t / cfg.tau_des)
    if off < T_len:
        w[off:] = w[off - 1] * np.exp(-np.arange(T_len - off) / cfg.tau_off)
    return w


def chr2_params(phi, cfg: ChR2Config = ChR2Config()) -> tuple[np.ndarray, np.ndarray]:
    """Intensity-dependent ChR2 steady/peak ratio and desensitization tau (ms).

    Brighter relative light phi -> deeper and faster desensitization; matches
    fixed (ss_frac, tau_des) exactly at phi = 1.
    """
    phi = np.asarray(phi, dtype=float)
    h = phi * (1.0 + cfg.k_light) / (phi + cfg.k_light)
    ss = cfg.ss_dim + (cfg.ss_frac - cfg.ss_dim) * h
    tau_d = cfg.tau_des_dim + (cfg.tau_des - cfg.tau_des_dim) * h
    return ss, tau_d


def chr2_current(T_len: int, on: int, off: int, phi,
                 i_max: float | None = None,
                 cfg: ChR2Config = ChR2Config()) -> np.ndarray:
    """Photocurrent array (n_cells x T_len) for cells receiving relative light phi."""
    i_max = cfg.i_max if i_max is None else float(i_max)
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    if not cfg.desensitize:
        I = np.zeros((phi.size, T_len), dtype=float)
        I[:, on:off] = (i_max * phi)[:, None]
        return I
    ss, tau_d = chr2_params(phi, cfg)
    I = np.zeros((phi.size, T_len), dtype=float)
    t = np.arange(off - on)[None, :]
    I[:, on:off] = (i_max * phi)[:, None] * (
        ss[:, None] + (1.0 - ss[:, None]) * np.exp(-t / tau_d[:, None])
    )
    if off < T_len:
        I[:, off:] = I[:, off - 1:off] * np.exp(-np.arange(T_len - off)[None, :] / cfg.tau_off)
    return I


# ---------------------------------------------------------------- cell selection
def pick_random_exc(N_exc: int, n_stim: int, seed: int) -> np.ndarray:
    """Randomly pick n_stim excitatory neuron indices without replacement."""
    return np.sort(np.random.default_rng(seed).choice(N_exc, size=n_stim, replace=False))


def draw_variable_light(N_exc: int, n_stim: int, draw_idx: int,
                        seed_offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic draw of opsin E cells and relative light levels phi ~ U(0, 1)."""
    op = pick_random_exc(N_exc, n_stim, seed=1000 + seed_offset + draw_idx)
    phi = np.random.default_rng(7000 + seed_offset + draw_idx).uniform(0.0, 1.0, n_stim)
    return op, phi


def draw_heterogeneous_baselines(N: int, N_exc: int, n_stim: int,
                                 draw_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw opsin cells, random light levels phi, and standard-normal baseline offsets z."""
    op, phi = draw_variable_light(N_exc, n_stim, draw_idx)
    z = np.random.default_rng(9000 + draw_idx).standard_normal(N)
    return op, phi, z


def rank_reference_drive(net, proto: ProtocolConfig, neuron: NeuronConfig,
                         n_stim: int, noise_seeds=(5, 6, 7)):
    """Rank E cells by rhythmic modulation (SD) and mean rate during reference drive."""
    from .simulate import run_pulse
    a, b = proto.analysis_win
    sd = np.zeros(net.N_exc)
    mu = np.zeros(net.N_exc)
    all_idx = np.arange(net.N)
    for s in noise_seeds:
        R, _ = run_pulse(net.W, all_idx, proto.i_ref, proto, neuron, ChR2Config(),
                         seed=s, opsin=False)
        sd += R[:net.N_exc, a:b].std(1)
        mu += R[:net.N_exc, a:b].mean(1)
    sd /= len(noise_seeds)
    mu /= len(noise_seeds)
    hi_idx = np.sort(np.argsort(sd)[::-1][:n_stim])
    act_idx = np.sort(np.argsort(mu)[::-1][:n_stim])
    return sd, mu, hi_idx, act_idx


def find_model_best_exc(net, n_stim: int, amp: float, chr2: ChR2Config,
                        neuron: NeuronConfig, Tq: int = 1500) -> np.ndarray:
    """Greedy search for n_stim E cells maximizing noise-free linear Re(lambda_max)."""
    from .simulate import simulate_single
    from .analysis import invert_transfer, leading_real_eig
    from dataclasses import replace

    quiet = replace(neuron, noise_ampl=0.0)
    wave = chr2_fixed_waveform(Tq, 200, Tq, chr2)

    def score(idx):
        I = np.zeros((net.N, Tq))
        I[idx, :] = amp * wave
        R = simulate_single(net.W, I, quiet, seed=5)
        return leading_real_eig(net.W, invert_transfer(R[:, 1000:], neuron).mean(1), neuron)

    chosen: list[int] = []
    for _ in range(n_stim):
        best = max((c for c in range(net.N_exc) if c not in chosen),
                   key=lambda c: score(chosen + [c]))
        chosen.append(best)
    return np.sort(chosen)
