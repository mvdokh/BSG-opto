"""Benchmark and characterize 1,000 random realizations of the original BSG model
(Lindén et al., 2022) across tonic drive D = 1..20.

Usage:
    python3 -m bsg_opto.experiments.bsg_benchmark_1000
    python3 -m bsg_opto.experiments.bsg_benchmark_1000 --plot-only
    .venv/bin/python run.py bsg-benchmark-1000
"""
from __future__ import annotations

import os
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# Limit per-process BLAS threads before importing numpy heavily in workers
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
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
from matplotlib.patches import Circle
from scipy.stats import spearmanr

from bsg_opto.config import NetworkConfig, NeuronConfig, RESULTS_DIR
from bsg_opto.network import BalancedNetwork
from bsg_opto.analysis import leading_real_eig, nerve_readout, transfer_slope
from bsg_opto.experiments.opto_threshold_figure import add_panel_label_xy, configure_publication_style

DESCRIPTION = 'Benchmark 1,000 random BSG network realizations across tonic drive D = 1..20'
DEFAULT_OUT = os.path.join(RESULTS_DIR, 'bsg_benchmark_1000')


def add_args(parser):
    parser.add_argument('--n-models', type=int, default=1000, help='Number of random BSG models (default: 1000)')
    parser.add_argument('--workers', type=int, default=8, help='Parallel worker processes (default: 8)')
    parser.add_argument('--out-dir', type=str, default=DEFAULT_OUT, help='Output directory')
    parser.add_argument('--plot-only', action='store_true', help='Re-render figure from cached data')
    parser.add_argument('--recompute', action='store_true', help='Force recomputation')


# =============================================================================
# 1. Parallel Batched 1,000-Model Simulation Worker
# =============================================================================
def _simulate_chunk(args):
    """Simulate a chunk of connectivity seeds across all D = 1..20 simultaneously."""
    seeds, T, burn_in, rec_every = args
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

    # Ensure shims exist in spawned subprocesses on macOS
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

    from spinalsim import generate_W_dale_by_radius

    cfg_base = NetworkConfig()
    neuron = NeuronConfig()
    N = cfg_base.N
    B = len(seeds)
    drives = np.arange(1, 21, dtype=np.float32)
    nD = drives.size

    # Build W batch & static eigenstructure for every model in chunk
    W_batch = np.empty((B, N, N), dtype=np.float32)
    lam1_re = np.empty(B, dtype=np.float64)
    lam1_im = np.empty(B, dtype=np.float64)
    lamc_re = np.empty(B, dtype=np.float64)
    lamc_im = np.empty(B, dtype=np.float64)

    for b, cs in enumerate(seeds):
        W = generate_W_dale_by_radius(
            N=N, C=cfg_base.C, frac_inh=cfg_base.frac_inh,
            g=cfg_base.g, W_radius=cfg_base.W_radius, seed=int(cs),
        )
        W_batch[b] = W.astype(np.float32)
        ev = np.linalg.eigvals(W)
        k1 = int(ev.real.argmax())
        lam1_re[b] = float(ev[k1].real)
        lam1_im[b] = float(abs(ev[k1].imag))
        cmask = np.abs(ev.imag) > 1e-5
        ev_c = ev[cmask]
        kc = int(ev_c.real.argmax())
        lamc_re[b] = float(ev_c[kc].real)
        lamc_im[b] = float(abs(ev_c[kc].imag))

    # Batched Euler simulation: (B, N, nD)
    rng = np.random.default_rng(50000 + int(seeds[0]))
    V = np.full((B, N, nD), neuron.V_init, dtype=np.float32)
    R = np.full((B, N, nD), neuron.V_init, dtype=np.float32)
    D_bcast = drives[None, None, :]

    a = np.float32(1.0 / neuron.tau_V)
    noise_amp = np.float32(neuron.noise_ampl)
    th = np.float32(neuron.threshold)
    gn = np.float32(neuron.gain)
    fm = np.float32(neuron.fmax)
    inv_th = np.float32(gn / th)
    inv_fm = np.float32(gn / fm)

    sum_R = np.zeros((B, N, nD), dtype=np.float64)
    sum_R2 = np.zeros((B, N, nD), dtype=np.float64)
    sum_V = np.zeros((B, N, nD), dtype=np.float64)
    n_win = T - burn_in
    n_rec = n_win // rec_every
    # Record 40 strided neurons every rec_every ms for crisp FFT frequency & spectral purity
    sub_idx = np.arange(0, N, 5)
    R_rec = np.empty((B, sub_idx.size, nD, n_rec), dtype=np.float32)

    rec_pos = 0
    for t in range(1, T):
        I = np.matmul(W_batch, R)
        I += noise_amp * rng.standard_normal((B, N, nD), dtype=np.float32)
        I += D_bcast
        V += a * (I - V)
        x = V - th
        R = th + np.where(x <= 0, th * np.tanh(inv_th * x), fm * np.tanh(inv_fm * x))
        if t >= burn_in:
            sum_R += R
            sum_R2 += R.astype(np.float64) ** 2
            sum_V += V
            if (t - burn_in) % rec_every == 0 and rec_pos < n_rec:
                R_rec[:, :, :, rec_pos] = R[:, sub_idx, :]
                rec_pos += 1

    mean_R = sum_R / n_win                          # (B, N, nD)
    var_R = np.maximum(sum_R2 / n_win - mean_R ** 2, 0.0)
    sd_R = np.sqrt(var_R)
    mean_V = sum_V / n_win

    osc_sd = sd_R.mean(axis=1)                      # (B, nD): mean temporal SD across all N neurons
    mean_rate = mean_R.mean(axis=1)                 # (B, nD): mean firing rate across neurons
    spatial_sd = mean_R.std(axis=1)                 # (B, nD): spatial heterogeneity of time-averaged rates

    # Spectral analysis over the 6 s analysis window (dt = rec_every ms)
    freqs = np.fft.rfftfreq(n_rec, d=rec_every * 1e-3)
    band = (freqs >= 0.15) & (freqs <= 15.0)
    freqs_b = freqs[band]
    R_rec_c = R_rec - R_rec.mean(axis=-1, keepdims=True)
    F = np.fft.rfft(R_rec_c, axis=-1)               # (B, 40, nD, n_freq)
    psd = np.mean(np.abs(F) ** 2, axis=1)           # (B, nD, n_freq)
    psd_b = psd[:, :, band]
    peak_i = psd_b.argmax(axis=-1)                  # (B, nD)
    peak_freq = freqs_b[peak_i]
    # Narrowband spectral concentration: power within +/- 0.35 Hz of peak divided by total band power
    tot_pow = np.maximum(psd_b.sum(axis=-1), 1e-12)
    spec_conc = np.empty((B, nD), dtype=np.float64)
    for b in range(B):
        for d in range(nD):
            fp = peak_freq[b, d]
            nb = np.abs(freqs_b - fp) <= 0.35
            spec_conc[b, d] = psd_b[b, d, nb].sum() / tot_pow[b, d]

    # Exact linearized leading Re(lambda_max) at D = 20 for every model in chunk
    lam_d20 = np.empty(B, dtype=np.float64)
    for b in range(B):
        lam_d20[b] = leading_real_eig(W_batch[b].astype(np.float64), mean_V[b, :, -1], neuron)

    return dict(
        seeds=np.asarray(seeds, dtype=int),
        lam1_re=lam1_re, lam1_im=lam1_im, lamc_re=lamc_re, lamc_im=lamc_im,
        osc_sd=osc_sd, mean_rate=mean_rate, spatial_sd=spatial_sd,
        peak_freq=peak_freq, spec_conc=spec_conc, lam_d20=lam_d20,
    )


def run_benchmark(n_models: int, workers: int, cache_path: str) -> dict:
    t0 = time.time()
    T = 8000       # 8 s total: 2 s transient burn-in + 6 s steady-state analysis window
    burn_in = 2000
    rec_every = 10  # 10 ms sampling (100 Hz Nyquist, 0.167 Hz FFT resolution)

    all_seeds = list(range(1, n_models + 1))
    chunk_size = 25
    chunks = [(all_seeds[i:i + chunk_size], T, burn_in, rec_every)
              for i in range(0, n_models, chunk_size)]

    print(f'[bench] Running {n_models} random BSG models x 20 drives (D=1..20) = '
          f'{n_models * 20:,} simulations ({T/1000:.0f} s each) on {workers} workers...', flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for idx, res in enumerate(ex.map(_simulate_chunk, chunks), start=1):
            results.append(res)
            if idx % 8 == 0 or idx == len(chunks):
                print(f'  chunk {idx:2d}/{len(chunks)} ({idx * chunk_size:4d}/{n_models} models, '
                      f'{time.time()-t0:.1f}s)', flush=True)

    data = {k: np.concatenate([r[k] for r in results], axis=0) for k in results[0].keys()}
    data['drives'] = np.arange(1, 21)

    # Simulate 3 full-resolution (1 ms, 8 s) archetype examples at D = 20 for panel b:
    #   1) Classic strong oscillator (seed = 1, Lindén et al. Fig. 2)
    #   2) High-frequency complex oscillator (high |Im lam_1| with strong oscillation)
    #   3) Static / non-oscillatory fixed-point realization (purely real leading eigenvalue Im(lam_1) = 0)
    is_real_lead = data['lam1_im'] < 1e-5
    osc_d20 = data['osc_sd'][:, -1]
    freq_d20 = data['peak_freq'][:, -1]

    seed_classic = 1
    # Fast clean oscillator among strong oscillators (SD >= 5 Hz, highest peak_freq)
    cand_fast = np.where((osc_d20 >= 5.0) & (data['spec_conc'][:, -1] >= 0.45) & (~is_real_lead))[0]
    seed_fast = int(data['seeds'][cand_fast[np.argmax(freq_d20[cand_fast])]]) if cand_fast.size else 2
    # Clean static fixed-point failure among purely real leading eigenvalue models (lowest SD at D=20)
    cand_static = np.where(is_real_lead & (osc_d20 < 1.5))[0]
    if cand_static.size:
        # Pick representative median-low static model with high spatial_sd
        ord_st = cand_static[np.argsort(data['spatial_sd'][cand_static, -1])[::-1]]
        seed_static = int(data['seeds'][ord_st[min(5, ord_st.size - 1)]])
    else:
        seed_static = int(data['seeds'][np.argmin(osc_d20)])

    ex_seeds = np.array([seed_classic, seed_fast, seed_static], dtype=int)
    neuron = NeuronConfig()
    ex_R = np.empty((3, 200, T), dtype=np.float32)
    ex_order = np.empty((3, 200), dtype=int)
    ex_nerve_u = np.empty((3, T), dtype=np.float32)
    ex_nerve_l = np.empty((3, T), dtype=np.float32)
    ex_sd = np.empty(3)
    ex_freq = np.empty(3)
    ex_lam1 = np.empty(3, dtype=np.complex128)

    from bsg_opto.simulate import simulate_batch, make_batched_ext
    from bsg_opto.config import ChR2Config, ProtocolConfig
    proto_ex = ProtocolConfig(T=T, stim_on=1000, stim_off=T)
    for k, cs in enumerate(ex_seeds):
        net_k = BalancedNetwork(NetworkConfig(seed=int(cs)))
        cols = [dict(base=20.0, op=None, phi=None)]
        out_k = simulate_batch(
            net_k.W, T, make_batched_ext(net_k.N, cols, proto_ex, ChR2Config(), light=False),
            1, neuron=neuron, seed=5, windows={'on': (burn_in, T)},
            snap_cols=(0,), rec_every=1, dtype=np.float64,
        )
        Rk = out_k['snaps'][0]
        ex_R[k] = Rk.astype(np.float32)
        ex_order[k] = net_k.phase_sort
        nr = nerve_readout(net_k, Rk, mode='limb', seed=0)
        ex_nerve_u[k] = nr['upper'].astype(np.float32)
        ex_nerve_l[k] = nr['lower'].astype(np.float32)
        idx_m = int(np.where(data['seeds'] == cs)[0][0])
        ex_sd[k] = float(out_k['on']['sd'][:, 0].mean())
        ex_freq[k] = float(data['peak_freq'][idx_m, -1])
        ex_lam1[k] = complex(data['lam1_re'][idx_m], data['lam1_im'][idx_m])

    data.update(
        ex_seeds=ex_seeds, ex_R=ex_R, ex_order=ex_order,
        ex_nerve_u=ex_nerve_u, ex_nerve_l=ex_nerve_l,
        ex_sd=ex_sd, ex_freq=ex_freq, ex_lam1=ex_lam1,
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **data)
    print(f'[bench] Saved {cache_path} ({time.time()-t0:.1f}s total)', flush=True)
    return data


# =============================================================================
# 2. Publication Benchmark Figure (Single 6-Panel Figure a-f)
# =============================================================================
def build_figure(data: dict, out_dir: str):
    configure_publication_style()
    drives = data['drives']                  # 1..20
    osc = data['osc_sd']                     # (1000, 20)
    freq = data['peak_freq']                 # (1000, 20)
    sconc = data['spec_conc']                # (1000, 20)
    sp_sd = data['spatial_sd']               # (1000, 20)
    lam1_re = data['lam1_re']
    lam1_im = data['lam1_im']
    n_models = osc.shape[0]

    is_real_lead = lam1_im < 1e-5
    is_cplx_lead = ~is_real_lead
    osc_d20 = osc[:, -1]
    is_rhythmic_20 = osc_d20 >= 2.0
    is_strong_20 = osc_d20 >= 5.0

    # Palette
    COL_MAIN = '#22252A'
    COL_CPLX = '#D64045'
    COL_REAL = '#3B6EA5'
    COL_TEAL = '#1D8A99'
    COL_N1 = '#2B6CB0'
    COL_N2 = '#2F855A'

    fig = plt.figure(figsize=(7.20, 7.65))

    # Row 1 (top, y in [0.695, 0.945]):
    #   a: Rhythmicity trajectories & percentiles across D = 1..20 (n = 1,000 models)
    #   b: Fraction of models rhythmic vs D + onset threshold D_th histogram
    #   c: 3 archetype examples at D = 20 (classic oscillator, fast oscillator, real-eigenvalue fixed point)
    ax_a = fig.add_axes([0.075, 0.708, 0.375, 0.238])
    ax_b = fig.add_axes([0.535, 0.708, 0.430, 0.238])

    # Row 2 (middle, y in [0.398, 0.618]): 3 archetype rasters + nerves at D = 20 (panel c)
    c_x = [0.075, 0.378, 0.680]
    c_w = 0.256
    ax_cr, ax_cn = [], []
    for x0 in c_x:
        ax_cr.append(fig.add_axes([x0, 0.478, c_w, 0.138]))
        ax_cn.append(fig.add_axes([x0, 0.402, c_w, 0.066]))
    ax_ccb = fig.add_axes([0.946, 0.478, 0.011, 0.138])

    # Row 3 (bottom, y in [0.072, 0.305]):
    #   d: Structural predictor — Leading eigenvalue (Re lam_1, |Im lam_1|) map of all 1,000 models
    #   e: Oscillatory SD vs static spatial SD at D = 20 (oscillators vs static fixed points!)
    #   f: Distribution of oscillation frequency at D = 20 vs analytical Hopf prediction
    ax_d = fig.add_axes([0.075, 0.072, 0.218, 0.225])
    ax_dcb = fig.add_axes([0.299, 0.072, 0.010, 0.225])
    ax_e = fig.add_axes([0.425, 0.072, 0.218, 0.225])
    ax_f = fig.add_axes([0.734, 0.072, 0.226, 0.225])

    # =========================================================================
    # Panel a: Oscillation amplitude (temporal rate SD) across D = 1..20 (n = 1,000)
    # =========================================================================
    COL_REF = '#D97706'
    p50 = np.median(osc, axis=0)
    ax_a.axhline(2.0, color='#888888', ls=(0, (2.5, 2.0)), lw=0.65, alpha=0.85, zorder=1)
    ax_a.text(1.35, 2.22, 'Rhythmic threshold (2 sp/s)', fontsize=4.6, color='#666666', va='bottom', ha='left')

    ax_a.plot(drives, p50, color=COL_MAIN, lw=1.65, marker='o', ms=2.2, ls='-',
              label=f'All models (median, $n={n_models:,}$)', zorder=5)
    ax_a.plot(drives, np.median(osc[is_cplx_lead], axis=0), color=COL_CPLX, lw=1.4, ls='-',
              label=f'Complex $\\lambda_1$ ($\\mathrm{{Im}}\\,\\lambda_1 \\neq 0$, rotational; $n={is_cplx_lead.sum()}$)', zorder=4)
    ax_a.plot(drives, np.median(osc[is_real_lead], axis=0), color=COL_REAL, lw=1.4, ls='-',
              label=f'Real $\\lambda_1$ ($\\mathrm{{Im}}\\,\\lambda_1 = 0$, non-rotational; $n={is_real_lead.sum()}$)', zorder=4)
    ax_a.plot(drives, osc[0], color=COL_REF, lw=1.35, ls=(0, (3.0, 1.5)),
              label='Reference seed 1', zorder=6)

    ax_a.set_xlim(1, 20.3); ax_a.set_ylim(-0.3, 10.6)
    ax_a.set_xticks([1, 5, 10, 15, 20]); ax_a.set_yticks([0, 2, 4, 6, 8, 10])
    ax_a.set_xlabel('Global tonic drive $D$', labelpad=2.2)
    ax_a.set_ylabel('Oscillation amplitude (rate SD, sp/s)', labelpad=2.2)
    ax_a.set_title(f'Oscillation amplitude across {n_models:,} random BSG networks ($D = 1\\text{{–}}20$)', fontsize=6.4, pad=5)
    ax_a.legend(loc='upper left', frameon=False, fontsize=4.65, handlelength=2.3, labelspacing=0.25)

    # =========================================================================
    # Panel b: % rhythmic networks vs D + onset drive D_th distribution
    # =========================================================================
    pct_ge2 = 100.0 * (osc >= 2.0).mean(axis=0)
    pct_ge5 = 100.0 * (osc >= 5.0).mean(axis=0)
    pct_ge2_cplx = 100.0 * (osc[is_cplx_lead] >= 2.0).mean(axis=0)
    pct_ge2_real = 100.0 * (osc[is_real_lead] >= 2.0).mean(axis=0)

    ax_b.plot(drives, pct_ge2, color=COL_MAIN, lw=1.6, marker='o', ms=2.3, ls='-',
              label='All models (rate SD $\\geq 2$ sp/s)')
    ax_b.plot(drives, pct_ge5, color='#6E7785', lw=1.15, ls=(0, (3.0, 1.5)), marker='s', ms=1.9,
              label='Strong rhythm (rate SD $\\geq 5$ sp/s)')
    ax_b.plot(drives, pct_ge2_cplx, color=COL_CPLX, lw=1.35, ls='-',
              label='Complex $\\lambda_1$ ($\\mathrm{{Im}}\\,\\lambda_1 \\neq 0$, rotational)')
    ax_b.plot(drives, pct_ge2_real, color=COL_REAL, lw=1.35, ls='-',
              label='Real $\\lambda_1$ ($\\mathrm{{Im}}\\,\\lambda_1 = 0$, non-rotational)')

    ax_b.set_xlim(1, 20.3); ax_b.set_ylim(-2, 105)
    ax_b.set_xticks([1, 5, 10, 15, 20]); ax_b.set_yticks([0, 25, 50, 75, 100])
    ax_b.set_xlabel('Global tonic drive $D$', labelpad=2.2)
    ax_b.set_ylabel('Rhythmic networks (%)', labelpad=2.2)
    ax_b.set_title('Fraction of random networks oscillating vs. tonic drive', fontsize=6.4, pad=5)
    ax_b.legend(loc='upper left', frameon=False, fontsize=4.75, handlelength=2.3, labelspacing=0.22)

    # Annotate exact D = 20 fractions right on all 4 curves (cleanly separated)
    ax_b.text(19.9, pct_ge2_cplx[-1] + 2.0, f'{pct_ge2_cplx[-1]:.1f}%', ha='right', va='bottom',
              fontsize=5.2, fontweight='bold', color=COL_CPLX)
    ax_b.text(19.9, pct_ge2[-1] + 2.2, f'{pct_ge2[-1]:.1f}%', ha='right', va='bottom',
              fontsize=5.2, fontweight='bold', color=COL_MAIN)
    ax_b.text(19.9, pct_ge5[-1] + 2.2, f'{pct_ge5[-1]:.1f}%', ha='right', va='bottom',
              fontsize=5.2, fontweight='bold', color='#5A6270')
    ax_b.text(19.9, pct_ge2_real[-1] - 3.0, f'{pct_ge2_real[-1]:.1f}%', ha='right', va='top',
              fontsize=5.2, fontweight='bold', color=COL_REAL)

    # Clean inset showing onset drive D_th histogram cleanly lifted above y = 0 curves
    d_th_list = []
    for m in range(n_models):
        hits = np.where(osc[m] >= 2.0)[0]
        if hits.size:
            d_th_list.append(drives[hits[0]])
    d_th_arr = np.asarray(d_th_list)
    ax_ins = fig.add_axes([0.592, 0.774, 0.140, 0.088])
    ax_ins.hist(d_th_arr, bins=np.arange(8.5, 21.5, 1.0), color=COL_TEAL, alpha=0.82, edgecolor='white', lw=0.4)
    ax_ins.axvline(np.median(d_th_arr), color='#111111', ls='--', lw=0.8)
    ax_ins.set_xlim(9, 20.5)
    ax_ins.set_xticks([10, 15, 20])
    ax_ins.set_yticks([])
    ax_ins.tick_params(labelsize=4.9, length=1.8, width=0.5)
    ax_ins.set_xlabel('Onset $D_{\\mathrm{th}}$', fontsize=5.0, labelpad=1.0)
    ax_ins.set_title(f'Med. $D_{{\\mathrm{{th}}}} = {np.median(d_th_arr):.0f}$', fontsize=5.0, pad=2.0)

    # =========================================================================
    # Panel c: 3 Archetype realizations at D = 20 (raster + nerve, 2-8 s window)
    # =========================================================================
    ex_R = data['ex_R']
    ex_order = data['ex_order']
    ex_seeds = data['ex_seeds']
    ex_sd = data['ex_sd']
    ex_freq = data['ex_freq']
    ex_lam1 = data['ex_lam1']
    ex_nu, ex_nl = data['ex_nerve_u'], data['ex_nerve_l']

    # Show 2.0-8.0 s (6 s steady state) on shared nerve amplitude scale
    # Guard against NaN nerve traces when a real-eigenvalue network has phases in {0, pi}
    t0_s, t1_s = 2000, 8000
    tvec_c = np.arange(t0_s, t1_s) / 1000.0
    ex_nu = np.nan_to_num(ex_nu, nan=0.0)
    ex_nl = np.nan_to_num(ex_nl, nan=0.0)
    # For static real-mode seed (k=2), if nerve readout was empty due to real eigenvector phases {0, pi},
    # compute readout using its leading complex eigenmode phase so its tonic/flat motor output is accurately shown
    if np.abs(ex_nu[2, t0_s:t1_s]).max() == 0.0 and np.abs(ex_nl[2, t0_s:t1_s]).max() == 0.0:
        from spinalsim import simple_readout
        net_st = BalancedNetwork(NetworkConfig(seed=int(ex_seeds[2])))
        cmask = np.abs(net_st.eigvals.imag) > 1e-5
        ev_all, evec_all = np.linalg.eig(net_st.W)
        c_idx = np.where(np.abs(ev_all.imag) > 1e-5)[0]
        kc = c_idx[ev_all[c_idx].real.argmax()]
        ph_c = np.angle(np.conj(evec_all[:, kc]))
        np.random.seed(0)
        ex_nu[2] = np.nan_to_num(simple_readout(ex_R[2], ph_c, net_st.N_exc, np.pi/2, -np.pi/2, np.pi/8, np.pi/8)['nerve'], nan=0.0)
        ex_nl[2] = np.nan_to_num(simple_readout(ex_R[2], ph_c, net_st.N_exc, -np.pi/2, np.pi/2, np.pi/8, np.pi/8)['nerve'], nan=0.0)
    nerve_max = max(float(np.abs(ex_nu[:, t0_s:t1_s]).max()), float(np.abs(ex_nl[:, t0_s:t1_s]).max()), 1.0)
    nerve_off = 1.25 * nerve_max

    c_titles = [
        f'Classic oscillator (seed {ex_seeds[0]}, $\\lambda_1 = {ex_lam1[0].real:.2f} \\pm {abs(ex_lam1[0].imag):.2f}i$)',
        f'Faster complex mode (seed {ex_seeds[1]}, $\\lambda_1 = {ex_lam1[1].real:.2f} \\pm {abs(ex_lam1[1].imag):.2f}i$)',
        f'Real-mode fixed point (seed {ex_seeds[2]}, $\\lambda_1 = {ex_lam1[2].real:.2f}$)',
    ]
    # Colored group dots matching Panels a & b:
    #   red = Complex lam_1, grey = Strong rhythm (>=5 sp/s), amber = Reference seed 1, blue = Real lam_1
    COL_STRONG = '#6E7785'
    c_dot_groups = [
        [COL_CPLX, COL_STRONG, COL_REF],
        [COL_CPLX, COL_STRONG],
        [COL_REAL],
    ]
    im_c = None
    for k in range(3):
        axr, axn = ax_cr[k], ax_cn[k]
        im_c = axr.imshow(ex_R[k][ex_order[k], t0_s:t1_s:5], aspect='auto', vmin=0, vmax=50,
                          cmap='viridis', interpolation='none', extent=[2, 8, 200, 0])
        axr.set_xticks([2, 4, 6, 8]); axr.set_xticklabels([])
        axr.set_yticks([0, 100, 200])
        axr.spines['top'].set_visible(True); axr.spines['right'].set_visible(True)
        axr.spines['top'].set_linewidth(0.5); axr.spines['right'].set_linewidth(0.5)

        # Title + inline colored group dots centered cleanly above each raster
        children = [TextArea(c_titles[k], textprops=dict(fontsize=5.5, color='#000000'))]
        dot_cols = c_dot_groups[k]
        da_w = 5.4 * len(dot_cols)
        da = DrawingArea(da_w, 6.0, 0.0, 0.0)
        for j, dcol in enumerate(dot_cols):
            da.add_artist(Circle((2.4 + 5.2 * j, 3.0), radius=1.95, facecolor=dcol,
                                 edgecolor='#222222', linewidth=0.35))
        children.append(da)
        pack = HPacker(children=children, align='center', pad=0.0, sep=2.6)
        ab = AnchoredOffsetbox(loc='lower center', child=pack, frameon=False,
                               bbox_to_anchor=(0.5, 1.025), bbox_transform=axr.transAxes,
                               pad=0.0, borderpad=0.0)
        axr.add_artist(ab)
        if k == 0:
            axr.set_ylabel('Neuron (phase)', labelpad=2.0)
        else:
            axr.set_yticklabels([])

        badge = (f'Amp. SD = {ex_sd[k]:.2f} sp/s | $f$ = {ex_freq[k]:.2f} Hz'
                 if k < 2 else f'Amp. SD = {ex_sd[k]:.2f} sp/s (static)')
        axr.text(0.97, 0.06, badge, transform=axr.transAxes, ha='right', va='bottom', fontsize=4.8,
                 color='white', bbox=dict(boxstyle='round,pad=0.16', fc='#000000', alpha=0.58, lw=0))

        axn.plot(tvec_c, ex_nl[k, t0_s:t1_s], color=COL_N2, lw=0.35, alpha=0.88)
        axn.plot(tvec_c, nerve_off + ex_nu[k, t0_s:t1_s], color=COL_N1, lw=0.35, alpha=0.88)
        axn.set_xlim(2, 8); axn.set_ylim(-0.95 * nerve_off, 1.95 * nerve_off)
        axn.set_xticks([2, 4, 6, 8]); axn.set_xlabel('Time (s)', labelpad=1.8)
        axn.set_yticks([0.0, nerve_off])
        if k == 0:
            axn.set_yticklabels(['Ext.', 'Flex.'], fontsize=5.5)
            axn.set_ylabel('Nerve', labelpad=2.0)
        else:
            axn.set_yticklabels([])

    cb_c = fig.colorbar(im_c, cax=ax_ccb)
    cb_c.set_label('Firing rate (sp/s)', fontsize=5.9, labelpad=2.5)
    cb_c.set_ticks([0, 25, 50])
    cb_c.ax.tick_params(labelsize=5.4, length=1.8, width=0.5)
    cb_c.outline.set_linewidth(0.5)

    # =========================================================================
    # Panel d: Leading eigenvalue map (Re lam_1 vs |Im lam_1|) colored by D=20 SD
    # =========================================================================
    norm_d = mcolors.Normalize(vmin=0.0, vmax=10.5)
    sc_d = ax_d.scatter(lam1_re, lam1_im, c=osc_d20, cmap='magma', norm=norm_d,
                        s=5.5, alpha=0.82, edgecolors='#222222', linewidths=0.12, zorder=3)
    ax_d.axhline(0.0, color='#888888', lw=0.5, ls=':', zorder=1)
    ax_d.set_xlim(0.84, 1.24); ax_d.set_ylim(-0.03, 0.62)
    ax_d.set_xticks([0.9, 1.0, 1.1, 1.2]); ax_d.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax_d.set_xlabel('Leading Re $\\lambda_1(W)$', labelpad=2.2)
    ax_d.set_ylabel('Leading $|$Im $\\lambda_1(W)|$', labelpad=2.2)
    ax_d.set_title('Eigenvalue structure of 1,000 networks', fontsize=6.4, pad=4.5)
    ax_d.annotate(f'Real $\\lambda_1$ ($n={is_real_lead.sum()}$)\n(mostly non-rhythmic)',
                  xy=(1.07, 0.01), xytext=(1.09, 0.46),
                  fontsize=4.9, color='#222222', ha='center',
                  arrowprops=dict(arrowstyle='->', lw=0.65, color='#333333'))

    cb_d = fig.colorbar(sc_d, cax=ax_dcb)
    cb_d.set_label('Rate SD at $D=20$ (sp/s)', fontsize=5.7, labelpad=2.5)
    cb_d.set_ticks([0, 2, 5, 8, 10])
    cb_d.ax.tick_params(labelsize=5.2, length=1.8, width=0.5)
    cb_d.outline.set_linewidth(0.5)

    # =========================================================================
    # Panel e: Dynamic rhythmicity (temporal SD) vs static heterogeneity (spatial SD) at D=20
    # =========================================================================
    ax_e.axvline(2.0, color='#888888', ls='--', lw=0.65, zorder=1)
    ax_e.scatter(osc_d20[is_cplx_lead], sp_sd[is_cplx_lead, -1], s=5.5, color=COL_CPLX, alpha=0.55,
                 edgecolors='none', label='Complex $\\lambda_1$ ($\\mathrm{Im}\\,\\lambda_1 \\neq 0$)', zorder=3)
    ax_e.scatter(osc_d20[is_real_lead], sp_sd[is_real_lead, -1], s=7.0, color=COL_REAL, alpha=0.72,
                 edgecolors='none', label='Real $\\lambda_1$ ($\\mathrm{Im}\\,\\lambda_1 = 0$)', zorder=4)
    ax_e.set_xlim(-0.4, 12.0); ax_e.set_ylim(-0.3, 22.8)
    ax_e.set_xticks([0, 3, 6, 9, 12]); ax_e.set_yticks([0, 5, 10, 15, 20])
    ax_e.set_xlabel('Temporal rate SD (sp/s)\n(raster x-axis)', labelpad=2.0)
    ax_e.set_ylabel('Spatial rate SD (sp/s)\n(raster y-axis)', labelpad=2.0)
    ax_e.set_title('Rhythmic vs. static states ($D = 20$)', fontsize=6.4, pad=4.5)
    ax_e.text(-0.15, 22.1, 'Static\nfixed pts', fontsize=4.5, color=COL_REAL, fontweight='bold', va='top', ha='left')
    ax_e.text(11.6, 16.0, 'Rhythmic\noscillators', fontsize=4.6, color=COL_CPLX, fontweight='bold',
              va='top', ha='right')
    ax_e.legend(loc='upper right', frameon=False,
                fontsize=4.7, markerscale=1.25, handletextpad=0.25, labelspacing=0.22)

    # =========================================================================
    # Panel f: Frequency distribution at D = 20 vs analytical Hopf prediction
    # =========================================================================
    # Filter to rhythmic models (rate SD >= 2 sp/s)
    rhy_mask = is_rhythmic_20
    f_rhy = freq[rhy_mask, -1]
    # Analytical Hopf frequency f_Hopf = (gain * |Im lam_c|) / (2*pi*tau_V)
    f_hopf = (1.2 * data['lamc_im'][rhy_mask]) / (2.0 * np.pi * 0.050)
    rho_f, _ = spearmanr(f_hopf, f_rhy)

    ax_f.scatter(f_hopf, f_rhy, s=5.5, color=COL_TEAL, alpha=0.55, edgecolors='none', zorder=3)
    ax_f.plot([0, 2.2], [0, 2.2], color='#444444', ls='--', lw=0.75, label='$f_{\\mathrm{sim}} = f_{\\mathrm{Hopf}}$', zorder=2)
    ax_f.set_xlim(0, 2.2); ax_f.set_ylim(0, 2.2)
    ax_f.set_xticks([0, 0.5, 1.0, 1.5, 2.0]); ax_f.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax_f.set_xlabel('Hopf pred. $f_{\\mathrm{Hopf}} = f$\u2032$|\\mathrm{Im}\\,\\lambda_1| / 2\\pi\\tau_V$ (Hz)', labelpad=2.2)
    ax_f.set_ylabel('Oscillation frequency (Hz)', labelpad=2.2)
    ax_f.set_title('Slow oscillation frequency ($\\tau_V = 50$ ms)', fontsize=6.4, pad=4.5)
    ax_f.text(0.04, 0.94,
              f'Rhythmic ($n={rhy_mask.sum()}$):\n'
              f'med. = {np.median(f_rhy):.2f} Hz\n'
              f'mean = {f_rhy.mean():.2f} $\\pm$ {f_rhy.std():.2f} Hz\n'
              f'max = {f_rhy.max():.2f} Hz\n'
              f'$\\rho = {rho_f:.2f}$',
              transform=ax_f.transAxes, ha='left', va='top', fontsize=4.8, color='#1A1A1A',
              bbox=dict(boxstyle='round,pad=0.22', fc='white', alpha=0.88, ec='#CCCCCC', lw=0.45))

    # Panel letters a-f
    add_panel_label_xy(fig, 0.015, 0.958, 'a')
    add_panel_label_xy(fig, 0.482, 0.958, 'b')
    add_panel_label_xy(fig, 0.015, 0.630, 'c')
    add_panel_label_xy(fig, 0.015, 0.308, 'd')
    add_panel_label_xy(fig, 0.376, 0.308, 'e')
    add_panel_label_xy(fig, 0.680, 0.308, 'f')

    pdf_path = os.path.join(out_dir, 'BSG_1000_model_benchmark.pdf')
    png_path = os.path.join(out_dir, 'BSG_1000_model_benchmark.png')
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    print(f'[plot] Saved {pdf_path} and {png_path}', flush=True)


def save_stats_and_caption(data: dict, out_dir: str):
    drives = data['drives']
    osc = data['osc_sd']
    freq = data['peak_freq']
    sp_sd = data['spatial_sd']
    lam1_re, lam1_im = data['lam1_re'], data['lam1_im']
    n_models = osc.shape[0]
    is_real = lam1_im < 1e-5
    is_cplx = ~is_real
    osc20 = osc[:, -1]
    rhy20 = osc20 >= 2.0
    str20 = osc20 >= 5.0
    f_rhy = freq[rhy20, -1]
    f_hopf = (1.2 * data['lamc_im'][rhy20]) / (2.0 * np.pi * 0.050)
    rho_f, _ = spearmanr(f_hopf, f_rhy)

    d_th = np.array([drives[np.where(osc[m] >= 2.0)[0][0]] for m in range(n_models) if np.any(osc[m] >= 2.0)])
    p25_20, p50_20, p75_20 = np.percentile(osc20, [25, 50, 75])
    p25_c, p50_c, p75_c = np.percentile(osc20[is_cplx], [25, 50, 75])
    p25_r, p50_r, p75_r = np.percentile(osc20[is_real], [25, 50, 75])

    lines = [
        f'=== Original BSG 1,000-Model Benchmark Summary (N=200, C=0.1, g=1, W_radius=1, tau_V=50 ms, D=1..20) ===',
        f'Total models: {n_models} | Complex leading lambda_1: {is_cplx.sum()} ({100*is_cplx.mean():.1f}%) | '
        f'Purely real leading lambda_1: {is_real.sum()} ({100*is_real.mean():.1f}%)',
        f'At D = 20:',
        f'  All models      : median rate SD = {p50_20:.2f} sp/s (mean {osc20.mean():.2f}+/-{osc20.std():.2f} sp/s), '
        f'>=2 sp/s: {rhy20.sum()}/{n_models} ({100*rhy20.mean():.1f}%), >=5 sp/s: {str20.sum()}/{n_models} ({100*str20.mean():.1f}%)',
        f'  Complex lam_1   : median rate SD = {p50_c:.2f} sp/s, '
        f'>=2 sp/s: {(osc20[is_cplx]>=2).sum()}/{is_cplx.sum()} ({100*(osc20[is_cplx]>=2).mean():.1f}%)',
        f'  Real lam_1      : median rate SD = {p50_r:.2f} sp/s, '
        f'>=2 sp/s: {(osc20[is_real]>=2).sum()}/{is_real.sum()} ({100*(osc20[is_real]>=2).mean():.1f}%), '
        f'median spatial rate SD = {np.median(sp_sd[is_real, -1]):.2f} sp/s',
        f'  Onset drive D_th (first D >= 2 sp/s, n={d_th.size}): median = {np.median(d_th):.1f}, '
        f'range = {d_th.min()}-{d_th.max()}',
        f'  Oscillation frequency at D = 20 (rhythmic models, n={rhy20.sum()}): '
        f'median = {np.median(f_rhy):.2f} Hz, mean = {f_rhy.mean():.2f}+/-{f_rhy.std():.2f} Hz, '
        f'max = {f_rhy.max():.2f} Hz; Spearman vs Hopf pred rho = {rho_f:.3f}',
        '',
        'Archetypes (D = 20):',
    ]
    for k, nm in enumerate(['Classic', 'Faster complex', 'Static real-mode']):
        lines.append(f'  {nm:16s} (seed {data["ex_seeds"][k]:4d}): lam1={data["ex_lam1"][k].real:.3f}+/-{abs(data["ex_lam1"][k].imag):.3f}i, '
                     f'rate_SD={data["ex_sd"][k]:.2f} sp/s, peak_f={data["ex_freq"][k]:.2f} Hz')
    lines += ['', 'Drive D: median rate SD (sp/s) | % >= 2 sp/s | % >= 5 sp/s | med osc freq (Hz, rhythmic)']
    for i, D in enumerate(drives):
        q50 = np.median(osc[:, i])
        rm = osc[:, i] >= 2.0
        mf = f'{np.median(freq[rm, i]):.2f}' if rm.any() else ' -- '
        lines.append(f'  D={D:2d}: {q50:6.2f} | {100*rm.mean():5.1f}% | '
                     f'{100*(osc[:,i]>=5.0).mean():5.1f}% | {mf}')

    txt = '\n'.join(lines)
    print(txt, flush=True)
    with open(os.path.join(out_dir, 'benchmark_stats.txt'), 'w') as fh:
        fh.write(txt + '\n')

    ex_seeds = data['ex_seeds']
    ex_lam1 = data['ex_lam1']
    ex_sd = data['ex_sd']
    ex_freq = data['ex_freq']

    # Write official markdown caption (.md as requested)
    md_caption = f"""**Fig. 1 | Characterization of oscillation amplitude, fixed-point failures, and oscillation frequency across 1,000 random realizations of the original BSG model.**

**a**, Population oscillation amplitude—quantified as the mean temporal standard deviation (s.d.) of neuronal firing rate in spikes per second (sp/s) across all $N = 200$ neurons over a 6 s steady-state window after 2 s burn-in—as a function of uniform global tonic drive $D = 1\\text{{–}}20$ across $n = 1{{,}}000$ independent random balanced Dale networks ($N = 200$, 100 E / 100 I, $C = 0.1$, $g = 1.0$, spectral radius $= 1.0$, $\\tau_V = 50$ ms, $V_{{\\mathrm{{th}}}} = 20$, gain $= 1.2$, $f_{{\\max}} = 50$ sp/s above threshold, noise $= 4.0$). Note that temporal rate s.d. (sp/s) measures how strongly neuronal firing rates swing up and down over time (oscillation amplitude), whereas oscillation frequency (cycles per second, Hz) is shown separately in **c** and **f**. Solid black curve with markers shows the overall median across all 1,000 networks (reaching {p50_20:.2f} sp/s at $D = 20$); solid red and solid blue curves show the medians of networks with complex leading eigenvalues ($\\mathrm{{Im}}\\,\\lambda_1 \\neq 0$, rotational dominant mode; $n = {is_cplx.sum()}$, {100*is_cplx.mean():.1f}%) versus purely real leading eigenvalues ($\\mathrm{{Im}}\\,\\lambda_1 = 0$, non-rotational dominant mode; $n = {is_real.sum()}$, {100*is_real.mean():.1f}%); dashed amber curve marks the original reference network (`seed = 1`); horizontal dashed grey line marks the 2 sp/s rhythmicity threshold.

**b**, Percentage of random networks exhibiting rhythmic firing (temporal rate $\\mathrm{{s.d.}} \\ge 2$ sp/s, solid black; strong rhythm $\\mathrm{{s.d.}} \\ge 5$ sp/s, dashed grey) across $D = 1\\text{{–}}20$. At $D = 20$, {100*rhy20.mean():.1f}% ({rhy20.sum()}/1,000) of all networks oscillate above 2 sp/s and {100*str20.mean():.1f}% ({str20.sum()}/1,000) exceed 5 sp/s: {100*(osc20[is_cplx]>=2).mean():.1f}% of complex-$\\lambda_1$ ($\\mathrm{{Im}}\\,\\lambda_1 \\neq 0$, rotational) networks oscillate (red) versus only {100*(osc20[is_real]>=2).mean():.1f}% of real-$\\lambda_1$ ($\\mathrm{{Im}}\\,\\lambda_1 = 0$, non-rotational) networks (blue). Inset: distribution of onset tonic drive $D_{{\\mathrm{{th}}}}$ (first drive with temporal rate $\\mathrm{{s.d.}} \\ge 2$ sp/s, $n = {d_th.size}$ networks; median $D_{{\\mathrm{{th}}}} = {np.median(d_th):.0f}$).

**c**, Representative phase-sorted population firing-rate rasters (top, in sp/s) and antagonist flexor/extensor motor-nerve readouts (bottom; shared scale, 2–8 s steady state) at $D = 20$ illustrating three dynamical archetypes, with coloured dots following each title indicating membership in the groups from **a** and **b** (red: complex $\\lambda_1$; grey: strong rhythm $\\mathrm{{s.d.}} \\ge 5$ sp/s; amber: reference `seed = 1`; blue: real $\\lambda_1$): the classic reference oscillator (`seed = {ex_seeds[0]}`, $\\lambda_1 = {ex_lam1[0].real:.2f} \\pm {abs(ex_lam1[0].imag):.2f}i$, amplitude $\\mathrm{{s.d.}} = {ex_sd[0]:.2f}$ sp/s, oscillation frequency $f = {ex_freq[0]:.2f}$ Hz), a faster complex-mode oscillator (`seed = {ex_seeds[1]}`, $\\lambda_1 = {ex_lam1[1].real:.2f} \\pm {abs(ex_lam1[1].imag):.2f}i$, amplitude $\\mathrm{{s.d.}} = {ex_sd[1]:.2f}$ sp/s, $f = {ex_freq[1]:.2f}$ Hz), and a purely real leading-eigenvalue realization trapped in a static non-oscillatory fixed point (`seed = {ex_seeds[2]}`, $\\lambda_1 = {ex_lam1[2].real:.2f}$, amplitude $\\mathrm{{s.d.}} = {ex_sd[2]:.2f}$ sp/s).

**d**, Leading connectivity eigenvalue $(\\mathrm{{Re}}\\,\\lambda_1, |\\mathrm{{Im}}\\,\\lambda_1|)$ of each of the 1,000 random matrices $W$, coloured by temporal rate s.d. (sp/s) at $D = 20$. Networks whose rightmost eigenvalue lands on the real axis ($\\mathrm{{Im}}\\,\\lambda_1 = 0$) predominantly undergo a non-oscillatory pitchfork-like instability rather than a Hopf oscillation.

**e**, Temporal firing-rate s.d. (oscillation amplitude, sp/s) versus spatial s.d. of time-averaged neuronal rates (sp/s) at $D = 20$, separating true rhythmic oscillators (high temporal s.d.) from static heterogeneous fixed-point states (low temporal s.d. $< 2$ sp/s but elevated spatial rate s.d., median {np.median(sp_sd[is_real, -1]):.2f} sp/s among real-$\\lambda_1$ networks).

**f**, Simulated peak oscillation frequency (cycles per second, Hz) at $D = 20$ across all rhythmic BSG networks ($n = {rhy20.sum()}$) plotted against the linearized Hopf frequency prediction $f_{{\\mathrm{{Hopf}}}} = f'(20)|\\mathrm{{Im}}\\,\\lambda_1| / (2\\pi\\tau_V)$ (Spearman $\\rho = {rho_f:.2f}$; dashed identity line). Because $\\tau_V = 50$ ms and random Dale matrices yield modest $|\\mathrm{{Im}}\\,\\lambda_1|$, the original BSG model oscillates strictly in a slow regime (median {np.median(f_rhy):.2f} Hz, mean {f_rhy.mean():.2f} $\\pm$ {f_rhy.std():.2f} Hz, maximum {f_rhy.max():.2f} Hz across all 1,000 models), never approaching fast mammalian licking frequencies (6–8 Hz).
"""
    with open(os.path.join(out_dir, 'BSG_1000_model_benchmark_caption.md'), 'w') as fh:
        fh.write(md_caption)


def run(n_models: int = 1000, workers: int = 8, out_dir: str = DEFAULT_OUT,
        plot_only: bool = False, recompute: bool = False, **_):
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, 'benchmark_1000_cache.npz')
    if plot_only or (os.path.exists(cache_path) and not recompute):
        print(f'[load] Loading cached 1,000-model benchmark from {cache_path}', flush=True)
        npz = np.load(cache_path, allow_pickle=False)
        data = {k: npz[k] for k in npz.files}
    else:
        data = run_benchmark(n_models, workers, cache_path)
    save_stats_and_caption(data, out_dir)
    build_figure(data, out_dir)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=DESCRIPTION)
    add_args(p)
    args = p.parse_args()
    run(**vars(args))
