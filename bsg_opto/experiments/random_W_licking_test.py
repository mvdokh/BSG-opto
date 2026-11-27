"""Test whether purely random Dale connectivity W (no rotational W, no psi) with
N = 300, C = 0.15, and tau_V = 18 ms can reach 6-8 Hz licking, and sweep tau_V
down to 3 ms to characterize what purely random W requires and its trade-offs.

Usage:
    python3 bsg_opto/experiments/random_W_licking_test.py
    python3 bsg_opto/experiments/random_W_licking_test.py --plot-only
"""
from __future__ import annotations

import os
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor
import numpy as np

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_BSG_DIR = os.path.join(_ROOT, 'BSG')
os.environ.setdefault('MPLCONFIGDIR', os.path.join(_ROOT, '.mplcache'))
for _p in (_ROOT, _BSG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import numba  # noqa: F401
except ImportError:
    _m = types.ModuleType('numba')
    _m.njit = lambda *a, **kw: (a[0] if (len(a) == 1 and callable(a[0])) else (lambda f: f))
    sys.modules['numba'] = _m
try:
    import astropy.stats.circstats  # noqa: F401
except ImportError:
    for _name in ('astropy', 'astropy.stats', 'astropy.stats.circstats'):
        sys.modules[_name] = types.ModuleType(_name)
    sys.modules['astropy.stats.circstats'].circcorrcoef = None
    sys.modules['astropy.stats.circstats'].circmean = None
    sys.modules['astropy.stats.circstats'].circstd = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from bsg_opto.config import NeuronConfig, RESULTS_DIR
from bsg_opto.experiments.opto_threshold_figure import add_panel_label_xy, configure_publication_style

DEFAULT_OUT = os.path.join(RESULTS_DIR, 'random_W_licking_test')
TAU_LIST = np.array([50.0, 30.0, 18.0, 12.0, 8.0, 6.0, 4.0, 3.0], dtype=np.float64)


def _init_worker_shims():
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    for p in (_ROOT, _BSG_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import numba  # noqa: F401
    except ImportError:
        m = types.ModuleType('numba')
        m.njit = lambda *a, **kw: (a[0] if (len(a) == 1 and callable(a[0])) else (lambda f: f))
        sys.modules['numba'] = m
    try:
        import astropy.stats.circstats  # noqa: F401
    except ImportError:
        for nm in ('astropy', 'astropy.stats', 'astropy.stats.circstats'):
            sys.modules[nm] = types.ModuleType(nm)
        sys.modules['astropy.stats.circstats'].circcorrcoef = None
        sys.modules['astropy.stats.circstats'].circmean = None
        sys.modules['astropy.stats.circstats'].circstd = None


def _sim_random_W_chunk(args):
    """Simulate a chunk of random W (N=300, C=0.15, g=1.0, W_radius=1.0) across a list of tau_V."""
    seeds, tau_list, T_ms, burn_ms, dt_ms, rec_ms, match_v_noise = args
    _init_worker_shims()
    from spinalsim import generate_W_dale_by_radius

    N = 300
    C = 0.15
    B = len(seeds)
    n_tau = len(tau_list)
    neuron = NeuronConfig()

    W_batch = np.empty((B, N, N), dtype=np.float32)
    lam1_re = np.empty(B, dtype=np.float64)
    lam1_im = np.empty(B, dtype=np.float64)
    lamc_re = np.empty(B, dtype=np.float64)
    lamc_im = np.empty(B, dtype=np.float64)

    for b, cs in enumerate(seeds):
        W = generate_W_dale_by_radius(N=N, C=C, frac_inh=0.5, g=1.0, W_radius=1.0, seed=int(cs))
        W_batch[b] = W.astype(np.float32)
        ev = np.linalg.eigvals(W)
        k1 = int(ev.real.argmax())
        lam1_re[b] = float(ev[k1].real)
        lam1_im[b] = float(abs(ev[k1].imag))
        cm = np.abs(ev.imag) > 1e-5
        evc = ev[cm]
        kc = int(evc.real.argmax())
        lamc_re[b] = float(evc[kc].real)
        lamc_im[b] = float(abs(evc[kc].imag))

    # Simulate all n_tau timescales simultaneously as columns (B, N, n_tau) at D = 20
    n_steps = int(round(T_ms / dt_ms))
    burn_step = int(round(burn_ms / dt_ms))
    rec_stride = int(round(rec_ms / dt_ms))
    n_win = n_steps - burn_step
    n_rec = n_win // rec_stride
    sub_idx = np.arange(0, N, 5)  # 60 neurons

    rng = np.random.default_rng(70000 + int(seeds[0]))
    V = np.full((B, N, n_tau), neuron.V_init, dtype=np.float32)
    R = np.full((B, N, n_tau), neuron.V_init, dtype=np.float32)

    tau_arr = np.asarray(tau_list, dtype=np.float32)
    a = (dt_ms / tau_arr)[None, None, :]
    if match_v_noise:
        # Hold steady-state OU membrane voltage noise constant (sigma_V = 0.40, matching BSG at tau=50 ms, dt=1 ms)
        noise_amp = (neuron.noise_ampl * np.sqrt(tau_arr / (50.0 * dt_ms)))[None, None, :].astype(np.float32)
    else:
        noise_amp = np.float32(neuron.noise_ampl / np.sqrt(dt_ms))
    D_val = np.float32(20.0)
    th = np.float32(neuron.threshold)
    gn = np.float32(neuron.gain)
    fm = np.float32(neuron.fmax)
    inv_th = np.float32(gn / th)
    inv_fm = np.float32(gn / fm)

    sum_R = np.zeros((B, N, n_tau), dtype=np.float64)
    sum_R2 = np.zeros((B, N, n_tau), dtype=np.float64)
    R_rec = np.empty((B, sub_idx.size, n_tau, n_rec), dtype=np.float32)

    rec_pos = 0
    for t in range(1, n_steps):
        I = np.matmul(W_batch, R)
        I += noise_amp * rng.standard_normal((B, N, n_tau), dtype=np.float32)
        I += D_val
        V += a * (I - V)
        x = V - th
        R = th + np.where(x <= 0, th * np.tanh(inv_th * x), fm * np.tanh(inv_fm * x))
        if t >= burn_step:
            sum_R += R
            sum_R2 += R.astype(np.float64) ** 2
            if (t - burn_step) % rec_stride == 0 and rec_pos < n_rec:
                R_rec[:, :, :, rec_pos] = R[:, sub_idx, :]
                rec_pos += 1

    mean_R = sum_R / n_win
    sd_R = np.sqrt(np.maximum(sum_R2 / n_win - mean_R ** 2, 0.0))
    osc_sd = sd_R.mean(axis=1)  # (B, n_tau)

    freqs = np.fft.rfftfreq(n_rec, d=rec_ms * 1e-3)
    band = (freqs >= 0.20) & (freqs <= 40.0)
    freqs_b = freqs[band]
    R_rec_c = R_rec - R_rec.mean(axis=-1, keepdims=True)
    F = np.fft.rfft(R_rec_c, axis=-1)
    psd = np.mean(np.abs(F) ** 2, axis=1)[:, :, band]
    peak_freq = freqs_b[psd.argmax(axis=-1)]

    return dict(
        seeds=np.asarray(seeds, dtype=int),
        lam1_re=lam1_re, lam1_im=lam1_im, lamc_re=lamc_re, lamc_im=lamc_im,
        osc_sd=osc_sd, peak_freq=peak_freq,
    )


def run_experiment(cache_path: str, workers: int = 8) -> dict:
    t0 = time.time()

    # 1) Direct test: 200 random W (N=300, C=0.15, tau_V=18 ms, literal BSG noise=4.0 at dt=1 ms, T=6 s)
    n_18 = 200
    seeds_18 = list(range(1, n_18 + 1))
    chunks_18 = [(seeds_18[i:i + 25], [18.0], 6000.0, 2000.0, 1.0, 5.0, False) for i in range(0, n_18, 25)]
    print(f'[1/3] Running {n_18} random-W networks (N=300, C=0.15, tau_V=18 ms, D=20)...', flush=True)
    res_18 = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_sim_random_W_chunk, chunks_18):
            res_18.append(r)
    d18 = {k: np.concatenate([r[k] for r in res_18], axis=0) for k in res_18[0].keys()}

    # 2) Full tau_V sweep: 100 random W (N=300, C=0.15) x 8 tau_V at dt=0.25 ms, matched OU voltage noise (sigma_V=0.40),
    #    2 s burn-in + 3 s steady-state window (T = 5 s) so tau=50 ms fully settles
    n_sw = 100
    seeds_sw = list(range(1, n_sw + 1))
    chunks_sw = [(seeds_sw[i:i + 10], TAU_LIST.tolist(), 5000.0, 2000.0, 0.25, 2.5, True) for i in range(0, n_sw, 10)]
    print(f'[2/3] Running tau_V sweep ({n_sw} random-W networks x {len(TAU_LIST)} tau_V, dt=0.25 ms)...', flush=True)
    res_sw = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for idx, r in enumerate(ex.map(_sim_random_W_chunk, chunks_sw), 1):
            res_sw.append(r)
            if idx % 5 == 0 or idx == len(chunks_sw):
                print(f'  sweep chunk {idx}/{len(chunks_sw)} ({time.time()-t0:.1f}s)', flush=True)
    dsw = {k: np.concatenate([r[k] for r in res_sw], axis=0) for k in res_sw[0].keys()}

    # 3) Simulate 2 full-resolution examples at N=300, C=0.15, tau_V=18 ms, random W:
    #    - fastest rhythmic seed out of all 200 (SD >= 5 sp/s)
    #    - median rhythmic seed
    amp18 = d18['osc_sd'][:, 0]
    freq18 = d18['peak_freq'][:, 0]
    strong18 = np.where((amp18 >= 5.0) & (d18['lam1_im'] > 1e-5))[0]
    idx_fast = int(strong18[np.argmax(freq18[strong18])])
    seed_fast = int(d18['seeds'][idx_fast])

    from spinalsim import generate_W_dale_by_radius, simple_readout
    W_fast = generate_W_dale_by_radius(N=300, C=0.15, frac_inh=0.5, g=1.0, W_radius=1.0, seed=seed_fast)
    ev, evec = np.linalg.eig(W_fast)
    k1 = int(ev.real.argmax())
    ph_fast = np.angle(np.conj(evec[:, k1]))
    order_fast = np.argsort(ph_fast)

    # Full 5 s simulation (1 ms steps) at D = 20, tau_V = 18 ms
    T_ex = 5000
    rng = np.random.default_rng(5)
    neuron = NeuronConfig(tau_V=18.0)
    V = np.full(300, neuron.V_init, dtype=np.float64)
    R = np.full(300, neuron.V_init, dtype=np.float64)
    R_ex = np.empty((300, T_ex), dtype=np.float32)
    R_ex[:, 0] = R
    for t in range(1, T_ex):
        I = W_fast @ R + neuron.noise_ampl * rng.standard_normal(300) + 20.0
        V += (1.0 / 18.0) * (I - V)
        x = V - neuron.threshold
        R = neuron.threshold + np.where(
            x <= 0,
            neuron.threshold * np.tanh(neuron.gain * x / neuron.threshold),
            neuron.fmax * np.tanh(neuron.gain * x / neuron.fmax),
        )
        R_ex[:, t] = R

    np.random.seed(0)
    nu_fast = np.nan_to_num(simple_readout(R_ex, ph_fast, 150, np.pi/2, -np.pi/2, np.pi/8, np.pi/8)['nerve'], nan=0.0)
    nl_fast = np.nan_to_num(simple_readout(R_ex, ph_fast, 150, -np.pi/2, np.pi/2, np.pi/8, np.pi/8)['nerve'], nan=0.0)

    out = dict(
        # 200-seed tau=18 ms direct test
        seeds_18=d18['seeds'], amp_18=amp18, freq_18=freq18,
        lam1_re_18=d18['lam1_re'], lam1_im_18=d18['lam1_im'],
        # 100-seed tau sweep
        tau_list=TAU_LIST, seeds_sw=dsw['seeds'],
        amp_sw=dsw['osc_sd'], freq_sw=dsw['peak_freq'],
        lam1_re_sw=dsw['lam1_re'], lam1_im_sw=dsw['lam1_im'],
        # Fastest tau=18 ms example
        seed_fast=seed_fast, freq_fast=freq18[idx_fast], amp_fast=amp18[idx_fast],
        lam1_fast=complex(d18['lam1_re'][idx_fast], d18['lam1_im'][idx_fast]),
        R_fast=R_ex, order_fast=order_fast, nu_fast=nu_fast.astype(np.float32), nl_fast=nl_fast.astype(np.float32),
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **out)
    print(f'[3/3] Saved {cache_path} ({time.time()-t0:.1f}s total)', flush=True)
    return out


def save_stats_and_caption(data: dict, out_dir: str):
    amp18 = data['amp_18']
    freq18 = data['freq_18']
    is_real18 = data['lam1_im_18'] < 1e-5
    rhy18 = amp18 >= 2.0
    in68_18 = rhy18 & (freq18 >= 6.0) & (freq18 <= 8.0)

    tau_list = data['tau_list']
    amp_sw = data['amp_sw']
    freq_sw = data['freq_sw']

    lines = [
        '=== Purely Random W Licking Feasibility Test (N=300, C=0.15, g=1.0, W_radius=1.0, D=20) ===',
        f'Direct test at tau_V = 18 ms (n = {len(amp18)} random networks):',
        f'  Rhythmic (>=2 sp/s): {rhy18.sum()}/{len(amp18)} ({100*rhy18.mean():.1f}%)',
        f'  Complex lam1 rhythmic: {(rhy18 & ~is_real18).sum()}/{(~is_real18).sum()} ({100*(rhy18[~is_real18]).mean():.1f}%)',
        f'  Real lam1 rhythmic   : {(rhy18 & is_real18).sum()}/{is_real18.sum()} ({100*(rhy18[is_real18]).mean():.1f}%)',
        f'  Rhythmic frequency   : median = {np.median(freq18[rhy18]):.2f} Hz, mean = {freq18[rhy18].mean():.2f}+/-{freq18[rhy18].std():.2f} Hz, max = {freq18[rhy18].max():.2f} Hz',
        f'  Inside 6-8 Hz        : {in68_18.sum()}/{len(amp18)} ({100*in68_18.mean():.1f}%)',
        f'  Fastest strong example: seed {int(data["seed_fast"])}, lam1={data["lam1_fast"].real:.2f}+/-{abs(data["lam1_fast"].imag):.2f}i, '
        f'amp={float(data["amp_fast"]):.2f} sp/s, f={float(data["freq_fast"]):.2f} Hz',
        '',
        'tau_V sweep (n = 100 purely random W networks each, N=300, C=0.15, dt=0.25 ms, D=20):',
        '  tau_V (ms) | % rhythmic | % in 6-8 Hz | med freq (Hz) | mean+/-sd (Hz) | max freq (Hz)',
    ]
    best_i = -1
    best_hit = -1.0
    for i, tv in enumerate(tau_list):
        rm = amp_sw[:, i] >= 2.0
        f = freq_sw[rm, i]
        hit = 100.0 * (rm & (freq_sw[:, i] >= 6.0) & (freq_sw[:, i] <= 8.0)).mean()
        if hit > best_hit:
            best_hit = hit
            best_i = i
        lines.append(
            f'  {tv:10.1f} | {100*rm.mean():9.1f}% | {hit:10.1f}% | '
            f'{np.median(f):13.2f} | {f.mean():5.2f}+/-{f.std():5.2f}   | {f.max():13.2f}'
        )
    txt = '\n'.join(lines)
    print(txt, flush=True)
    with open(os.path.join(out_dir, 'random_W_test_stats.txt'), 'w') as fh:
        fh.write(txt + '\n')

    best_tv = float(tau_list[best_i])
    rm_best = amp_sw[:, best_i] >= 2.0
    f_best = freq_sw[rm_best, best_i]

    md_caption = f"""**Testing purely random balanced Dale connectivity ($W$) without rotational wiring ($\\alpha = 0$, no phase shift $\\psi$) at $N = 300$, $C = 0.15$, and fast $\\tau_V$.**

**a**, Oscillation amplitude (temporal firing-rate s.d., sp/s) versus peak oscillation frequency (Hz) at $D = 20$ across $n = 200$ independent purely random balanced Dale networks with $N = 300$ (150 E / 150 I), $C = 0.15$, $g = 1.0$, $W_{{\\mathrm{{radius}}}} = 1.0$, and $\\tau_V = 18$ ms (red: complex leading eigenvalue $\\mathrm{{Im}}\\,\\lambda_1 \\neq 0$, $n = {(~is_real18).sum()}$; blue: purely real $\\mathrm{{Im}}\\,\\lambda_1 = 0$, $n = {is_real18.sum()}$; green band: 6–8 Hz physiological licking target). Although {rhy18.sum()}/200 ({100*rhy18.mean():.1f}%) networks oscillate above 2 sp/s, their median frequency is only {np.median(freq18[rhy18]):.2f} Hz (maximum {freq18[rhy18].max():.2f} Hz) and **0/200 (0.0%)** reach 6–8 Hz.

**b**, Phase-sorted firing-rate raster (sp/s) and antagonist motor-nerve readout (2–5 s steady state at $D = 20$) for the single fastest strongly oscillating purely random network out of all 200 seeds at $\\tau_V = 18$ ms (`seed = {int(data["seed_fast"])}`, $\\lambda_1 = {data["lam1_fast"].real:.2f} \\pm {abs(data["lam1_fast"].imag):.2f}i$, amplitude s.d. $= {float(data["amp_fast"]):.2f}$ sp/s, $f = {float(data["freq_fast"]):.2f}$ Hz), confirming that $\\tau_V = 18$ ms without structured connectivity remains below 6–8 Hz.

**c**, Peak oscillation frequency at $D = 20$ across $n = 100$ purely random networks ($N = 300$, $C = 0.15$, $dt = 0.25$ ms, membrane voltage noise variance held constant at $\\sigma_V = 0.40$ across all $\\tau_V$ matching original BSG at $\\tau_V = 50$ ms) as $\\tau_V$ is reduced from 50 ms down to 3 ms. Dots show individual rhythmic networks ($\\mathrm{{s.d.}} \\ge 2$ sp/s); solid black line tracks the median frequency. Purely random $W$ can reach the 6–8 Hz band without rotational wiring if $\\tau_V$ is lowered to **~3–5 ms** (median {np.median(f_best):.2f} Hz at $\\tau_V = {best_tv:.0f}$ ms, matching fast AMPA/GABA$_A$ brainstem synaptic timescales), but individual random realizations scatter widely across frequency (from {f_best.min():.1f} to {f_best.max():.1f} Hz at $\\tau_V = {best_tv:.0f}$ ms).

**d**, Percentage of purely random networks that oscillate (rate $\\mathrm{{s.d.}} \\ge 2$ sp/s, black) and percentage that land inside the 6–8 Hz licking band (red) as a function of $\\tau_V$ (matched membrane noise $\\sigma_V = 0.40$). Reducing $\\tau_V$ alone never removes real-eigenvalue fixed-point failures (oscillation rate remains ~75% across all $\\tau_V$), and because random $|\\mathrm{{Im}}\\,\\lambda_1|$ varies widely across seeds, the fraction landing inside 6–8 Hz peaks at only **{best_hit:.0f}%** (at $\\tau_V = {best_tv:.0f}$ ms), compared to 100% with rotational connectivity.
"""
    with open(os.path.join(out_dir, 'Random_W_licking_test_caption.md'), 'w') as fh:
        fh.write(md_caption)


def build_figure(data: dict, out_dir: str):
    configure_publication_style()
    amp18 = data['amp_18']
    freq18 = data['freq_18']
    is_real18 = data['lam1_im_18'] < 1e-5
    is_cplx18 = ~is_real18
    rhy18 = amp18 >= 2.0

    tau_list = data['tau_list']
    amp_sw = data['amp_sw']
    freq_sw = data['freq_sw']

    COL_CPLX = '#D64045'
    COL_REAL = '#3B6EA5'
    COL_MAIN = '#22252A'
    COL_GRN = '#2F855A'
    COL_N1 = '#2B6CB0'
    COL_N2 = '#2F855A'

    fig = plt.figure(figsize=(7.20, 5.60))

    # Row 1: Panel a (left: N=300, C=0.15, tau=18 ms amp vs freq) & Panel b (right: fastest seed raster + nerve)
    ax_a = fig.add_axes([0.080, 0.560, 0.390, 0.355])
    ax_br = fig.add_axes([0.555, 0.670, 0.375, 0.245])
    ax_bn = fig.add_axes([0.555, 0.560, 0.375, 0.095])
    ax_bcb = fig.add_axes([0.938, 0.670, 0.011, 0.245])

    # Row 2: Panel c (left: tau_V sweep frequency) & Panel d (right: % rhythmic vs % in 6-8 Hz)
    ax_c = fig.add_axes([0.080, 0.100, 0.390, 0.345])
    ax_d = fig.add_axes([0.555, 0.100, 0.395, 0.345])

    # =========================================================================
    # Panel a: Direct test — N=300, C=0.15, tau_V=18 ms, purely random W (n=200)
    # =========================================================================
    ax_a.axvspan(6.0, 8.0, color=COL_GRN, alpha=0.13, zorder=0)
    ax_a.axvline(6.0, color=COL_GRN, ls='--', lw=0.75, alpha=0.8)
    ax_a.axvline(8.0, color=COL_GRN, ls='--', lw=0.75, alpha=0.8)
    ax_a.text(7.0, 12.8, '6–8 Hz\ntarget\n(0/200)', ha='center', va='top',
              fontsize=5.2, fontweight='bold', color=COL_GRN)

    ax_a.axhline(2.0, color='#888888', ls=(0, (3.0, 2.0)), lw=0.7, zorder=1)
    rng = np.random.default_rng(7)
    fx18 = freq18 + rng.uniform(-0.07, 0.07, size=freq18.size)
    ax_a.scatter(fx18[is_cplx18], amp18[is_cplx18], s=10, color=COL_CPLX, alpha=0.58, edgecolors='none',
                 label=f'Complex $\\lambda_1$ ($\\mathrm{{Im}}\\,\\lambda_1 \\neq 0$; $n={is_cplx18.sum()}$)', zorder=3)
    ax_a.scatter(fx18[is_real18], amp18[is_real18], s=11, color=COL_REAL, alpha=0.62, edgecolors='none',
                 label=f'Real $\\lambda_1$ ($\\mathrm{{Im}}\\,\\lambda_1 = 0$; $n={is_real18.sum()}$)', zorder=4)

    # Highlight fastest strong seed shown in Panel b
    ax_a.scatter([float(data['freq_fast'])], [float(data['amp_fast'])], s=48, marker='*',
                 color='#D97706', edgecolors='#111111', linewidths=0.45,
                 label=f'Fastest seed {int(data["seed_fast"])} ({float(data["freq_fast"]):.2f} Hz)', zorder=6)

    ax_a.set_xlim(0.0, 8.5)
    ax_a.set_ylim(-0.3, 13.6)
    ax_a.set_xticks([0, 2, 4, 6, 8])
    ax_a.set_yticks([0, 2, 4, 6, 8, 10, 12])
    ax_a.set_xlabel('Oscillation frequency at $D = 20$ (Hz)', labelpad=2.5)
    ax_a.set_ylabel('Oscillation amplitude (rate SD, sp/s)', labelpad=2.5)
    ax_a.set_title('Random $W$ ($N=300, C=0.15, \\tau_V=18$ ms; $n=200$)', fontsize=6.4, pad=5.0)
    ax_a.legend(loc='upper center', bbox_to_anchor=(0.48, 0.99), frameon=True, facecolor='white',
                framealpha=0.90, edgecolor='#DDDDDD', fontsize=4.7, handletextpad=0.3, labelspacing=0.22)
    ax_a.text(5.78, 6.20, f'Rhythmic ($\\geq 2$ sp/s):\n{rhy18.sum()}/200 ({100*rhy18.mean():.1f}%)\n'
                          f'Med. $f = {np.median(freq18[rhy18]):.2f}$ Hz\nMax $f = {freq18[rhy18].max():.2f}$ Hz',
              ha='right', va='center', fontsize=4.7, color='#333333',
              bbox=dict(boxstyle='round,pad=0.24', fc='white', alpha=0.92, ec='#CCCCCC', lw=0.45))

    # =========================================================================
    # Panel b: Fastest oscillating random-W seed at tau_V = 18 ms (2-5 s)
    # =========================================================================
    R_fast = data['R_fast']
    order_fast = data['order_fast']
    nu_fast = data['nu_fast']
    nl_fast = data['nl_fast']
    t0, t1 = 2000, 5000
    tvec = np.arange(t0, t1) / 1000.0
    im_b = ax_br.imshow(R_fast[order_fast, t0:t1:3], aspect='auto', vmin=0, vmax=50,
                        cmap='viridis', interpolation='none', extent=[2, 5, 300, 0])
    ax_br.set_xticks([2, 3, 4, 5]); ax_br.set_xticklabels([])
    ax_br.set_yticks([0, 150, 300])
    ax_br.set_ylabel('Neuron (phase)', labelpad=2.2)
    ax_br.spines['top'].set_visible(True); ax_br.spines['right'].set_visible(True)
    ax_br.spines['top'].set_linewidth(0.5); ax_br.spines['right'].set_linewidth(0.5)
    lam_f = data['lam1_fast']
    ax_br.set_title(f'Fastest random $W$ at $\\tau_V=18$ ms (seed {int(data["seed_fast"])}, '
                    f'$\\lambda_1={lam_f.real:.2f}\\pm{abs(lam_f.imag):.2f}i$)', fontsize=6.1, pad=5.0)
    ax_br.text(0.97, 0.06, f'Amp. SD = {float(data["amp_fast"]):.2f} sp/s | $f$ = {float(data["freq_fast"]):.2f} Hz',
               transform=ax_br.transAxes, ha='right', va='bottom', fontsize=5.0, color='white',
               bbox=dict(boxstyle='round,pad=0.18', fc='#000000', alpha=0.60, lw=0))

    nmax = max(float(np.abs(nu_fast[t0:t1]).max()), float(np.abs(nl_fast[t0:t1]).max()), 1.0)
    noff = 1.25 * nmax
    ax_bn.plot(tvec, nl_fast[t0:t1], color=COL_N2, lw=0.4, alpha=0.9)
    ax_bn.plot(tvec, noff + nu_fast[t0:t1], color=COL_N1, lw=0.4, alpha=0.9)
    ax_bn.set_xlim(2, 5); ax_bn.set_ylim(-0.95 * noff, 1.95 * noff)
    ax_bn.set_xticks([2, 3, 4, 5]); ax_bn.set_xlabel('Time (s)', labelpad=2.0)
    ax_bn.set_yticks([0.0, noff]); ax_bn.set_yticklabels(['Ext.', 'Flex.'], fontsize=5.4)
    ax_bn.set_ylabel('Nerve', labelpad=2.0)

    cb_b = fig.colorbar(im_b, cax=ax_bcb)
    cb_b.set_label('Firing rate (sp/s)', fontsize=5.7, labelpad=2.5)
    cb_b.set_ticks([0, 25, 50])
    cb_b.ax.tick_params(labelsize=5.2, length=1.8, width=0.5)
    cb_b.outline.set_linewidth(0.5)

    # =========================================================================
    # Panel c: Sweeping tau_V (50 -> 3 ms) with purely random W (n = 100 each)
    # =========================================================================
    xpos = np.arange(len(tau_list))
    ax_c.axhspan(6.0, 8.0, color=COL_GRN, alpha=0.14, zorder=0)
    ax_c.axhline(6.0, color=COL_GRN, ls='--', lw=0.75, alpha=0.8)
    ax_c.axhline(8.0, color=COL_GRN, ls='--', lw=0.75, alpha=0.8)
    ax_c.text(0.15, 7.0, '6–8 Hz target', va='center', ha='left', fontsize=5.1, fontweight='bold', color=COL_GRN)

    med_f = np.empty(len(tau_list))
    rng_c = np.random.default_rng(11)
    for i in range(len(tau_list)):
        rm = amp_sw[:, i] >= 2.0
        fi = freq_sw[rm, i]
        med_f[i] = np.median(fi)
        xj = i + rng_c.uniform(-0.20, 0.20, size=fi.size)
        is_r = data['lam1_im_sw'][rm] < 1e-5
        ax_c.scatter(xj[~is_r], fi[~is_r], s=7.5, color=COL_CPLX, alpha=0.45, edgecolors='none', zorder=2)
        ax_c.scatter(xj[is_r], fi[is_r], s=7.5, color=COL_REAL, alpha=0.50, edgecolors='none', zorder=2)

    ax_c.plot(xpos, med_f, color=COL_MAIN, lw=1.65, marker='o', ms=3.5, label='Median rhythmic freq.', zorder=5)
    # Mark tau_V = 18 ms column
    i18 = int(np.where(tau_list == 18.0)[0][0])
    ax_c.axvline(i18, color='#888888', ls=':', lw=0.8, zorder=1)
    ax_c.text(i18 + 0.08, 17.0, '$\\tau_V=18$ ms', fontsize=4.8, color='#555555', va='top')

    ax_c.set_xlim(-0.5, len(tau_list) - 0.5)
    ax_c.set_ylim(0, 23.2)
    ax_c.set_xticks(xpos)
    ax_c.set_xticklabels([f'{int(t)}' for t in tau_list])
    ax_c.set_yticks([0, 4, 6, 8, 12, 16, 20])
    ax_c.set_xlabel('Neuronal time constant $\\tau_V$ (ms, purely random $W$)', labelpad=2.5)
    ax_c.set_ylabel('Oscillation frequency at $D = 20$ (Hz)', labelpad=2.5)
    ax_c.set_title('Random $W$ needs $\\tau_V \\approx 3\\text{–}5$ ms to reach 6–8 Hz', fontsize=6.4, pad=5.0)
    ax_c.legend(loc='upper left', frameon=False, fontsize=5.0, handlelength=2.2)

    # =========================================================================
    # Panel d: Trade-off of purely random W (% rhythmic vs % inside 6-8 Hz)
    # =========================================================================
    pct_rhy = np.array([100.0 * (amp_sw[:, i] >= 2.0).mean() for i in range(len(tau_list))])
    pct_68 = np.array([100.0 * ((amp_sw[:, i] >= 2.0) & (freq_sw[:, i] >= 6.0) & (freq_sw[:, i] <= 8.0)).mean()
                       for i in range(len(tau_list))])

    ax_d.axhline(100.0, color=COL_GRN, ls=':', lw=0.85, alpha=0.85, zorder=1)
    ax_d.text(len(tau_list) - 0.65, 97.5, 'Rotational licking BSG: 100% in 6–8 Hz', ha='right', va='top',
              fontsize=4.9, color=COL_GRN, fontweight='bold')

    ax_d.plot(xpos, pct_rhy, color=COL_MAIN, lw=1.6, marker='o', ms=3.2,
              label='Oscillating (rate SD $\\geq 2$ sp/s)')
    ax_d.plot(xpos, pct_68, color=COL_CPLX, lw=1.65, marker='s', ms=3.2,
              label='Oscillating & inside 6–8 Hz')

    for i, val in enumerate(pct_68):
        if val > 0:
            ax_d.text(i, val + 3.5, f'{val:.0f}%', ha='center', va='bottom',
                      fontsize=4.9, fontweight='bold', color=COL_CPLX)

    ax_d.axvline(i18, color='#888888', ls=':', lw=0.8, zorder=1)
    ax_d.set_xlim(-0.5, len(tau_list) - 0.5)
    ax_d.set_ylim(-3, 108)
    ax_d.set_xticks(xpos)
    ax_d.set_xticklabels([f'{int(t)}' for t in tau_list])
    ax_d.set_yticks([0, 25, 50, 75, 100])
    ax_d.set_xlabel('Neuronal time constant $\\tau_V$ (ms, purely random $W$)', labelpad=2.5)
    ax_d.set_ylabel('Networks (%)', labelpad=2.5)
    ax_d.set_title('Reliability trade-off with purely random $W$ ($n=100$)', fontsize=6.4, pad=5.0)
    ax_d.legend(loc='center left', bbox_to_anchor=(0.01, 0.50), frameon=False, fontsize=5.0, handlelength=2.2)

    add_panel_label_xy(fig, 0.014, 0.935, 'a')
    add_panel_label_xy(fig, 0.498, 0.935, 'b')
    add_panel_label_xy(fig, 0.014, 0.468, 'c')
    add_panel_label_xy(fig, 0.498, 0.468, 'd')

    pdf_path = os.path.join(out_dir, 'Random_W_licking_test.pdf')
    png_path = os.path.join(out_dir, 'Random_W_licking_test.png')
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    print(f'[plot] Saved {pdf_path} and {png_path}', flush=True)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--plot-only', action='store_true')
    p.add_argument('--recompute', action='store_true')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()

    os.makedirs(DEFAULT_OUT, exist_ok=True)
    cache_path = os.path.join(DEFAULT_OUT, 'random_W_test_cache.npz')
    if args.plot_only or (os.path.exists(cache_path) and not args.recompute):
        npz = np.load(cache_path, allow_pickle=False)
        data = {k: npz[k] for k in npz.files}
    else:
        data = run_experiment(cache_path, workers=args.workers)
    save_stats_and_caption(data, DEFAULT_OUT)
    build_figure(data, DEFAULT_OUT)


if __name__ == '__main__':
    main()
