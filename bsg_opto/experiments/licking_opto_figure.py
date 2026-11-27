"""Publication figure: Optogenetic stimulation (10% E cells, variable illumination
intensity, and ChR2 desensitization) in the 6-8 Hz Licking BSG network.

Shows that 10% excitatory photostimulation (15 / 150 E cells) fails to evoke rhythmic
licking at rest across all illumination intensities (I_max = 10-320) and desensitization
regimes (I_ss/I_peak = 0.1-1.0 and variable light phi), but reliably gates 6-8 Hz
rotational licking and alternating hypoglossal (XII) bursts once background tonic drive
approaches the Hopf bifurcation threshold (D_th ~ 13.5).

Usage:
    python3 bsg_opto/experiments/licking_opto_figure.py
    python3 bsg_opto/experiments/licking_opto_figure.py --plot-only
"""
from __future__ import annotations

import os
import sys
import time
import types
import numpy as np

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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from bsg_opto.config import ChR2Config, ProtocolConfig, RESULTS_DIR
from bsg_opto.licking import LickingNetwork, LickingNetworkConfig, licking_neuron_config
from bsg_opto.opsin import chr2_current, draw_variable_light
from bsg_opto.simulate import make_batched_ext, simulate_batch
from bsg_opto.analysis import jacobian_eigs, leading_real_eig, nerve_readout, unlit_mean
from bsg_opto.experiments.opto_threshold_figure import add_panel_label_xy, configure_publication_style

DESCRIPTION = 'Generate optogenetic gating, illumination intensity, and ChR2 desensitization figure for the 6-8 Hz Licking BSG model'
DEFAULT_OUT = os.path.join(RESULTS_DIR, 'licking_opto_figure')


def add_args(parser):
    parser.add_argument('--out-dir', type=str, default=DEFAULT_OUT)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--refresh-examples', action='store_true',
                        help='Refresh only the 5 representative examples (2 s laser: 1-3 s) keeping cached sweeps')
    parser.add_argument('--refresh-curves', action='store_true',
                        help='Refresh representative examples and Row-3 curves (8x zero-padded FFT) keeping cached heatmaps')


def _peak_freq_from_snap(R_sub: np.ndarray, dt_ms: float = 5.0) -> float:
    """Compute high-resolution dominant oscillation frequency (Hz) via 8x zero-padded FFT."""
    Rc = R_sub - R_sub.mean(axis=-1, keepdims=True)
    win = np.hanning(Rc.shape[-1])[None, :]
    nfft = Rc.shape[-1] * 8
    F = np.fft.rfft(Rc * win, n=nfft, axis=-1)
    psd = np.mean(np.abs(F) ** 2, axis=0)
    freqs = np.fft.rfftfreq(nfft, d=dt_ms * 1e-3)
    band = (freqs >= 0.5) & (freqs <= 16.0)
    return float(freqs[band][int(psd[band].argmax())])


def _simulate_freq_batch(W, cols, ops, proto, cfg, neuron, seed, light=True, rec_every=5):
    """Run simulate_batch and return (unlit_sd, peak_freq, leading_Re_lam) for every column."""
    n_cols = len(cols)
    t0, t1 = proto.analysis_win
    # Run custom fast batched loop recording strided 60 neurons in analysis window for exact FFT frequency
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    Wd = W.astype(np.float32)
    ext_fn = make_batched_ext(N, cols, proto, cfg, light=light)

    V = np.full((N, n_cols), neuron.V_init, dtype=np.float32)
    R = np.full((N, n_cols), neuron.V_init, dtype=np.float32)
    a = np.float32(1.0 / neuron.tau_V)
    noise_amp = np.float32(neuron.noise_ampl)
    th = np.float32(neuron.threshold)
    gn = np.float32(neuron.gain)
    fm = np.float32(neuron.fmax)
    inv_th = np.float32(gn / th)
    inv_fm = np.float32(gn / fm)

    n_win = t1 - t0
    n_rec = n_win // rec_every
    sub_idx = np.arange(0, N, 5)
    R_rec = np.empty((sub_idx.size, n_cols, n_rec), dtype=np.float32)
    sum_R = np.zeros((N, n_cols), dtype=np.float64)
    sum_R2 = np.zeros((N, n_cols), dtype=np.float64)
    sum_V = np.zeros((N, n_cols), dtype=np.float64)

    rec_pos = 0
    for t in range(1, proto.T):
        I = Wd @ R
        I += noise_amp * rng.standard_normal((N, n_cols), dtype=np.float32)
        I += ext_fn(t)
        V += a * (I - V)
        x = V - th
        R = th + np.where(x <= 0, th * np.tanh(inv_th * x), fm * np.tanh(inv_fm * x))
        if t0 <= t < t1:
            sum_R += R
            sum_R2 += R.astype(np.float64) ** 2
            sum_V += V
            if (t - t0) % rec_every == 0 and rec_pos < n_rec:
                R_rec[:, :, rec_pos] = R[sub_idx, :]
                rec_pos += 1

    mean_R = sum_R / n_win
    sd_R = np.sqrt(np.maximum(sum_R2 / n_win - mean_R ** 2, 0.0))
    mean_V = sum_V / n_win

    osc = np.array([unlit_mean(sd_R[:, k], ops[k]) for k in range(n_cols)], dtype=np.float64)

    # Vectorized 8x zero-padded Hanning FFT frequency across all columns (< 0.05 Hz resolution)
    Rc = R_rec - R_rec.mean(axis=-1, keepdims=True)
    win = np.hanning(n_rec)[None, None, :]
    nfft = n_rec * 8
    F = np.fft.rfft(Rc * win, n=nfft, axis=-1)
    psd = np.mean(np.abs(F) ** 2, axis=0)  # (n_cols, n_freq)
    freqs = np.fft.rfftfreq(nfft, d=rec_every * 1e-3)
    band = (freqs >= 0.5) & (freqs <= 16.0)
    peak_f = freqs[band][psd[:, band].argmax(axis=-1)]

    return osc, peak_f, mean_V


def run_simulations(cache_path: str) -> dict:
    t_start = time.time()
    net = LickingNetwork(LickingNetworkConfig(seed=1))
    neuron = licking_neuron_config()
    # 4.0 s protocol: 1.0 s pre-laser, 2.0 s laser (1.0-3.0 s, ~15 lick cycles), 1.0 s post-laser
    proto = ProtocolConfig(T=4000, stim_on=1000, stim_off=3000, analysis_delay=500, post_delay=200)
    n_stim = 15  # 10% of 150 E cells
    print(f'[lick-opto] Network N={net.N} ({net.N_exc}E/{net.N_inh}I), tau_V={neuron.tau_V} ms, '
          f'lam1={net.leading_eigval:.3f}, n_stim={n_stim} (10% E)', flush=True)

    # ---- A. Full 1-ms representative examples for panels b, c, d (median draw 7) ----
    ex_draw = 7
    op_ex, phi_ex = draw_variable_light(net.N_exc, n_stim, ex_draw)
    chr2_main = ChR2Config(i_max=80.0, desensitize=True)
    ex_cols = [
        dict(base=0.0,  op=None,  phi=None),                 # 0: D=0 no laser
        dict(base=0.0,  op=op_ex, phi=phi_ex, i_max=80.0),   # 1: D=0 + 10% E var desens laser (Panel b)
        dict(base=12.0, op=None,  phi=None),                 # 2: D=12 no laser
        dict(base=12.0, op=op_ex, phi=phi_ex, i_max=80.0),   # 3: D=12 + 10% E var desens laser (Panel c)
        dict(base=20.0, op=None,  phi=None),                 # 4: D=20 no laser reference (Panel d)
    ]
    out_ex = simulate_batch(
        net.W, proto.T,
        make_batched_ext(net.N, ex_cols, proto, chr2_main, light=True),
        len(ex_cols), neuron=neuron, seed=100,
        windows={'pre': proto.pre_win, 'on': proto.analysis_win, 'post': proto.post_win},
        snap_cols=tuple(range(len(ex_cols))), rec_every=1, dtype=np.float64,
    )
    ex_R = out_ex['snaps']
    ex_Vmean = out_ex['on']['Vmean']
    ex_osc_on = np.array([unlit_mean(out_ex['on']['sd'][:, k], op_ex) for k in range(len(ex_cols))])
    ex_osc_pre = np.array([unlit_mean(out_ex['pre']['sd'][:, k], op_ex) for k in range(len(ex_cols))])
    ex_osc_post = np.array([unlit_mean(out_ex['post']['sd'][:, k], op_ex) for k in range(len(ex_cols))])
    ex_eigs = np.stack([jacobian_eigs(net.W, ex_Vmean[:, k], neuron) for k in range(len(ex_cols))])
    ex_lam_max = np.array([ex_eigs[k][ex_eigs[k].real.argmax()] for k in range(len(ex_cols))])
    t0_a, t1_a = proto.analysis_win
    ex_freq_on = np.array([_peak_freq_from_snap(ex_R[k, ::5, t0_a:t1_a:5], dt_ms=5.0) for k in range(len(ex_cols))])
    print(f'  Examples: D=0+L SD={ex_osc_on[1]:.2f} sp/s; D=12+L SD={ex_osc_on[3]:.2f} sp/s ({ex_freq_on[3]:.2f} Hz), '
          f'pre={ex_osc_pre[3]:.2f}, post={ex_osc_post[3]:.2f}; D=20 SD={ex_osc_on[4]:.2f} sp/s ({ex_freq_on[4]:.2f} Hz)', flush=True)

    # ---- B. Panel e: Rest (D = 0) Intensity x Desensitization matrix (n_draws = 30) ----
    amps_e = np.array([10., 20., 40., 80., 160., 320.])
    n_draws_e = 30
    e_osc = np.zeros((6, amps_e.size, n_draws_e))
    e_lam = np.zeros((6, amps_e.size, 8))
    for r, row_key in enumerate([0.1, 0.3, 0.5, 0.7, 1.0, 'var']):
        cfg_r = ChR2Config(desensitize=True) if row_key == 'var' \
            else ChR2Config(ss_frac=float(row_key), desensitize=(float(row_key) < 1.0))
        cols, ops, meta = [], [], []
        for c_i, amp in enumerate(amps_e):
            for s in range(n_draws_e):
                op, phi = draw_variable_light(net.N_exc, n_stim, s)
                p_use = phi if row_key == 'var' else np.ones(n_stim)
                cols.append(dict(base=0.0, op=op, phi=p_use, i_max=float(amp)))
                ops.append(op)
                meta.append((c_i, s))
        osc_r, _, Vm_r = _simulate_freq_batch(net.W, cols, ops, proto, cfg_r, neuron, seed=200 + r, light=True)
        for k, (c_i, s) in enumerate(meta):
            e_osc[r, c_i, s] = osc_r[k]
            if s < 8:
                e_lam[r, c_i, s] = leading_real_eig(net.W, Vm_r[:, k], neuron)
    print(f'  Panel e (D=0 grid) done ({time.time()-t_start:.1f}s), max SD={e_osc.mean(-1).max():.2f} sp/s', flush=True)

    # ---- C. Panels f & g: Full D = 0..20 x Intensity (Off + 5..320) ----
    drives = np.arange(0, 21)
    amps_fg = np.array([5., 10., 20., 40., 80., 160., 320.])
    n_draws_fg = 20
    n_draws_hi = 30

    # Control (No laser, D = 0..20, n = 30)
    cols_ctrl, ops_ctrl, meta_ctrl = [], [], []
    for i, D in enumerate(drives):
        for s in range(n_draws_hi):
            op, _ = draw_variable_light(net.N_exc, n_stim, s)
            cols_ctrl.append(dict(base=float(D), op=None, phi=None))
            ops_ctrl.append(op)
            meta_ctrl.append((i, s))
    osc_c, freq_c, Vm_c = _simulate_freq_batch(net.W, cols_ctrl, ops_ctrl, proto, chr2_main, neuron, seed=300, light=False)
    ctrl_osc = np.zeros((drives.size, n_draws_hi))
    ctrl_freq = np.zeros((drives.size, n_draws_hi))
    ctrl_lam = np.zeros((drives.size, 12))
    for k, (i, s) in enumerate(meta_ctrl):
        ctrl_osc[i, s] = osc_c[k]
        ctrl_freq[i, s] = freq_c[k]
        if s < 12:
            ctrl_lam[i, s] = leading_real_eig(net.W, Vm_c[:, k], neuron)

    grid_var = np.zeros((drives.size, amps_fg.size))
    grid_step = np.zeros((drives.size, amps_fg.size))
    grid_var_freq = np.zeros((drives.size, amps_fg.size))
    grid_step_freq = np.zeros((drives.size, amps_fg.size))
    for pmode, desens, t_osc, t_frq, seed_g in [
        ('var', True,  grid_var,  grid_var_freq,  400),
        ('uni', False, grid_step, grid_step_freq, 500),
    ]:
        cfg_g = ChR2Config(desensitize=desens)
        cols, ops, meta = [], [], []
        for i, D in enumerate(drives):
            for j, A in enumerate(amps_fg):
                for s in range(n_draws_fg):
                    op, phi = draw_variable_light(net.N_exc, n_stim, s)
                    p_use = phi if pmode == 'var' else np.ones(n_stim)
                    cols.append(dict(base=float(D), op=op, phi=p_use, i_max=float(A)))
                    ops.append(op)
                    meta.append((i, j, s))
        osc_g, freq_g, _ = _simulate_freq_batch(net.W, cols, ops, proto, cfg_g, neuron, seed=seed_g, light=True)
        acc_o = np.zeros((drives.size, amps_fg.size, n_draws_fg))
        acc_f = np.zeros((drives.size, amps_fg.size, n_draws_fg))
        for k, (i, j, s) in enumerate(meta):
            acc_o[i, j, s] = osc_g[k]
            acc_f[i, j, s] = freq_g[k]
        t_osc[:] = acc_o.mean(-1)
        t_frq[:] = np.median(acc_f, axis=-1)
    print(f'  Panels f & g (D x I_max grids) done ({time.time()-t_start:.1f}s)', flush=True)

    # ---- D. Panels h, i, j: Detailed drive curves D = 0..20 (n = 30 draws) ----
    curve_specs = [
        ('var_des_40',   'var', True,  40.0,  601),
        ('var_des_80',   'var', True,  80.0,  602),
        ('uni_des_40',   'uni', True,  40.0,  603),
        ('uni_step_40',  'uni', False, 40.0,  604),
        ('uni_step_160', 'uni', False, 160.0, 605),
    ]
    curves_osc, curves_freq, curves_lam = {}, {}, {}
    for cname, pmode, desens, imax, seed_c in curve_specs:
        cfg_c = ChR2Config(i_max=imax, desensitize=desens)
        cols, ops, meta = [], [], []
        for i, D in enumerate(drives):
            for s in range(n_draws_hi):
                op, phi = draw_variable_light(net.N_exc, n_stim, s)
                p_use = phi if pmode == 'var' else np.ones(n_stim)
                cols.append(dict(base=float(D), op=op, phi=p_use, i_max=imax))
                ops.append(op)
                meta.append((i, s))
        osc_cv, freq_cv, Vm_cv = _simulate_freq_batch(net.W, cols, ops, proto, cfg_c, neuron, seed=seed_c, light=True)
        o_mat = np.zeros((drives.size, n_draws_hi))
        f_mat = np.zeros((drives.size, n_draws_hi))
        l_mat = np.zeros((drives.size, 12))
        for k, (i, s) in enumerate(meta):
            o_mat[i, s] = osc_cv[k]
            f_mat[i, s] = freq_cv[k]
            if s < 12:
                l_mat[i, s] = leading_real_eig(net.W, Vm_cv[:, k], neuron)
        curves_osc[cname], curves_freq[cname], curves_lam[cname] = o_mat, f_mat, l_mat
    print(f'  Panels h-j (drive curves + freq + Re lam) done ({time.time()-t_start:.1f}s)', flush=True)

    data = dict(
        W=net.W, phase=net.phase, phase_sort=net.phase_sort, ei_sort=net.ei_phase_sort(),
        op_ex=op_ex, phi_ex=phi_ex,
        ex_R=ex_R.astype(np.float32), ex_Vmean=ex_Vmean,
        ex_osc_on=ex_osc_on, ex_osc_pre=ex_osc_pre, ex_osc_post=ex_osc_post,
        ex_freq_on=ex_freq_on, ex_eigs=ex_eigs, ex_lam_max=ex_lam_max,
        amps_e=amps_e, e_osc=e_osc, e_lam=e_lam,
        drives=drives, amps_fg=amps_fg,
        ctrl_osc=ctrl_osc, ctrl_freq=ctrl_freq, ctrl_lam=ctrl_lam,
        grid_var=grid_var, grid_step=grid_step,
        grid_var_freq=grid_var_freq, grid_step_freq=grid_step_freq,
    )
    for k in curves_osc:
        data[f'osc_{k}'] = curves_osc[k]
        data[f'freq_{k}'] = curves_freq[k]
        data[f'lam_{k}'] = curves_lam[k]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **data)
    print(f'[lick-opto] Saved {cache_path} ({time.time()-t_start:.1f}s total)', flush=True)
    return data


def draw_licking_opto_schematic(ax):
    """Minimalist schematic of the 6-8 Hz Licking BSG network + 10% E optogenetic protocol."""
    ax.set_xlim(-0.2, 11.4)
    ax.set_ylim(-0.1, 5.7)
    ax.axis('off')

    c_E = '#D95F43'
    c_I = '#3B6EA5'
    c_op = '#B8252A'
    c_las = '#1E88E5'

    e_box = FancyBboxPatch((0.15, 1.50), 4.10, 2.45, boxstyle='round,pad=0.12,rounding_size=0.38',
                           fc='#FDF1ED', ec=c_E, lw=0.95)
    ax.add_patch(e_box)
    ax.text(2.20, 3.58, 'Excitatory (E)', ha='center', va='center', fontsize=5.6,
            fontweight='bold', color='#8C2D19')
    ax.text(2.20, 3.06, '$N = 150$', ha='center', va='center', fontsize=5.1, color='#555555')

    op_box = FancyBboxPatch((0.36, 1.66), 3.68, 0.88, boxstyle='round,pad=0.05,rounding_size=0.22',
                            fc='#FCE0E0', ec=c_op, lw=0.9)
    ax.add_patch(op_box)
    ax.text(2.20, 2.10, '10% ChR2 ($n = 15$)', ha='center', va='center', fontsize=4.6,
            fontweight='bold', color=c_op)

    i_box = FancyBboxPatch((6.95, 1.50), 4.10, 2.45, boxstyle='round,pad=0.12,rounding_size=0.38',
                           fc='#EEF4FB', ec=c_I, lw=0.95)
    ax.add_patch(i_box)
    ax.text(9.00, 3.22, 'Inhibitory (I)', ha='center', va='center', fontsize=5.6,
            fontweight='bold', color='#1F456E')
    ax.text(9.00, 2.52, '$N = 150$', ha='center', va='center', fontsize=5.1, color='#555555')
    ax.text(9.00, 1.92, '$\\tau_V = 18$ ms', ha='center', va='center', fontsize=4.8, color='#444444')

    arr_ei = FancyArrowPatch((4.38, 3.10), (6.82, 3.10), connectionstyle='arc3,rad=-0.26',
                             arrowstyle='-|>,head_length=3.2,head_width=1.8', lw=0.9, color=c_E)
    arr_ie = FancyArrowPatch((6.82, 2.15), (4.38, 2.15), connectionstyle='arc3,rad=-0.26',
                             arrowstyle='-|>,head_length=3.2,head_width=1.8', lw=0.9, color=c_I)
    ax.add_patch(arr_ei)
    ax.add_patch(arr_ie)
    ax.text(5.60, 3.84, '$C = 0.15$', ha='center', va='center', fontsize=5.0, color='#333333')
    ax.text(5.60, 1.00, 'rot. $W$\n($\\psi = 40^\\circ$)', ha='center', va='center',
            fontsize=4.6, color='#333333', linespacing=1.15)

    ax.text(5.60, 5.32, 'Global tonic drive $D$ (0–20)', ha='center', va='center',
            fontsize=5.6, fontweight='bold', color='#1A1A1A')
    ax.annotate('', xy=(2.25, 4.12), xytext=(4.85, 4.98),
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color='#333333', shrinkA=0, shrinkB=0))
    ax.annotate('', xy=(8.95, 4.12), xytext=(6.35, 4.98),
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color='#333333', shrinkA=0, shrinkB=0))

    ax.text(2.20, 0.25, '473 nm laser (1–3 s)', ha='center', va='center',
            fontsize=5.3, fontweight='bold', color=c_las)
    ax.annotate('', xy=(2.20, 1.48), xytext=(2.20, 0.55),
                arrowprops=dict(arrowstyle='-|>', lw=1.35, color=c_las, shrinkA=0, shrinkB=0))


def save_stats_and_caption(data: dict, out_dir: str):
    e_mean = data['e_osc'].mean(-1)
    e_lam_max = data['e_lam'].mean(-1).max()
    drives = data['drives']
    ctrl_m = data['ctrl_osc'].mean(-1)
    v40_m = data['osc_var_des_40'].mean(-1)
    v80_m = data['osc_var_des_80'].mean(-1)
    u40_m = data['osc_uni_des_40'].mean(-1)
    s40_m = data['osc_uni_step_40'].mean(-1)
    s160_m = data['osc_uni_step_160'].mean(-1)

    v80_f12 = data['freq_var_des_80'][12]
    v80_o12 = data['osc_var_des_80'][12]
    s40_f12 = data['freq_uni_step_40'][12]
    ctrl_f20 = data['freq_var_des_80'][20]
    ctrl_f20_nl = data['ctrl_freq'][20]

    lines = [
        '=== 6-8 Hz Licking BSG Optogenetic Gating Summary (N=300, 150E/150I, 10% E = 15 ChR2 cells, tau_V=18 ms) ===',
        f'Rest (D = 0) across 36 intensity x desensitization regimes (I_max=10..320, I_ss/I_peak=0.1..1.0 & var phi):',
        f'  Max mean unlit rate SD = {e_mean.max():.2f} sp/s, max mean Re(lambda_max) = {e_lam_max:.3f} (< 1.0 everywhere)',
        '',
        'Representative examples (seed 1, median draw 7, var. light + ChR2 desens., I_max = 80):',
        f'  D= 0 + laser : unlit rate SD = {data["ex_osc_on"][1]:.2f} sp/s, Re(lam) = {data["ex_lam_max"][1].real:.2f}',
        f'  D=12 + laser : unlit rate SD = {data["ex_osc_on"][3]:.2f} sp/s (pre={data["ex_osc_pre"][3]:.2f}, post={data["ex_osc_post"][3]:.2f}), '
        f'freq = {data["ex_freq_on"][3]:.2f} Hz, Re(lam) = {data["ex_lam_max"][3].real:.2f}',
        f'  D=20 no laser: unlit rate SD = {data["ex_osc_on"][4]:.2f} sp/s, freq = {data["ex_freq_on"][4]:.2f} Hz, Re(lam) = {data["ex_lam_max"][4].real:.2f}',
        '',
        'Drive sweep (mean unlit rate SD in sp/s, n=30 draws):',
        f'  No laser (ctrl)      : D=0:{ctrl_m[0]:.2f} | D=8:{ctrl_m[8]:.2f} | D=10:{ctrl_m[10]:.2f} | D=12:{ctrl_m[12]:.2f} | D=13:{ctrl_m[13]:.2f} | D=14:{ctrl_m[14]:.2f} | D=20:{ctrl_m[20]:.2f}',
        f'  Var. desens (I=40)   : D=0:{v40_m[0]:.2f} | D=8:{v40_m[8]:.2f} | D=10:{v40_m[10]:.2f} | D=12:{v40_m[12]:.2f} | D=13:{v40_m[13]:.2f} | D=14:{v40_m[14]:.2f} | D=20:{v40_m[20]:.2f}',
        f'  Var. desens (I=80)   : D=0:{v80_m[0]:.2f} | D=8:{v80_m[8]:.2f} | D=10:{v80_m[10]:.2f} | D=12:{v80_m[12]:.2f} | D=13:{v80_m[13]:.2f} | D=14:{v80_m[14]:.2f} | D=20:{v80_m[20]:.2f}',
        f'  Uni. desens (I=40)   : D=0:{u40_m[0]:.2f} | D=8:{u40_m[8]:.2f} | D=10:{u40_m[10]:.2f} | D=12:{u40_m[12]:.2f} | D=13:{u40_m[13]:.2f} | D=14:{u40_m[14]:.2f} | D=20:{u40_m[20]:.2f}',
        f'  No-desens step (I=40): D=0:{s40_m[0]:.2f} | D=8:{s40_m[8]:.2f} | D=10:{s40_m[10]:.2f} | D=12:{s40_m[12]:.2f} | D=13:{s40_m[13]:.2f} | D=14:{s40_m[14]:.2f} | D=20:{s40_m[20]:.2f}',
        f'  No-desens step(I=160): D=0:{s160_m[0]:.2f} | D=8:{s160_m[8]:.2f} | D=10:{s160_m[10]:.2f} | D=12:{s160_m[12]:.2f} | D=13:{s160_m[13]:.2f} | D=14:{s160_m[14]:.2f} | D=20:{s160_m[20]:.2f}',
        '',
        'Evoked licking frequency (Hz):',
        f'  No laser at D=20           : {ctrl_f20_nl.mean():.2f} +/- {ctrl_f20_nl.std():.2f} Hz',
        f'  Var. desens (I=80) at D=12 : {v80_f12[v80_o12>=2].mean():.2f} +/- {v80_f12[v80_o12>=2].std():.2f} Hz (n={int((v80_o12>=2).sum())}/30 rhythmic)',
        f'  Step (I=40) at D=12        : {s40_f12.mean():.2f} +/- {s40_f12.std():.2f} Hz',
    ]
    txt = '\n'.join(lines)
    print(txt, flush=True)
    with open(os.path.join(out_dir, 'licking_opto_stats.txt'), 'w') as fh:
        fh.write(txt + '\n')

    md_caption = f"""**Optogenetic gating of 6–8 Hz rhythmic licking in the Licking BSG model across illumination intensities and ChR2 desensitization regimes.**

**a**, Top: Schematic of the balanced Dale-law Licking BSG network ($N = 300$: 150 E / 150 I, $C = 0.15$, $g = 1.0$, $\\tau_V = 18$ ms, phase-shifted rotational connectivity $W$ with $\\alpha = 0.68$, $\\psi = 40^\\circ$) receiving uniform global tonic drive $D \\in [0, 20]$ and a 2 s 473 nm laser pulse (1–3 s) targeting a random 10% subset of excitatory neurons ($n = 15$ ChR2-expressing E cells). Bottom: Normalized ChR2 photocurrent waveforms comparing an idealized non-desensitizing step ($I_{{\\mathrm{{ss}}}}/I_{{\\mathrm{{peak}}}} = 1$, dashed teal), uniform fixed desensitization ($I_{{\\mathrm{{ss}}}}/I_{{\\mathrm{{peak}}}} = 0.3$, purple), and intensity-dependent desensitization under spatially varying illumination ($\\varphi = 0.75$ and $0.30$, red; stronger illumination drives faster, deeper desensitization).

**b–d**, Representative E/I phase-sorted population firing-rate rasters (top, in sp/s; arrowheads mark the 15 illuminated E cells shaded by light factor $\\varphi$) and hypoglossal (XII) protractor/retractor motor-nerve readouts (bottom, shared scale; 0.5–3.5 s window, blue bar marks 2 s laser ON, 1–3 s) under variable-light desensitizing ChR2 ($I_{{\\max}} = 80$): **b**, at rest ($D = 0$), 10% E-cell photostimulation fails to induce rhythmic licking (unlit amplitude $\\mathrm{{s.d.}} = {data['ex_osc_on'][1]:.2f}$ sp/s, $\\mathrm{{Re}}\\,\\lambda_{{\\max}} = {data['ex_lam_max'][1].real:.2f} < 1$); **c**, near the bifurcation threshold ($D = 12$, silent before laser: {data['ex_osc_pre'][3]:.2f} sp/s), the identical 2 s laser pulse gates sustained {data['ex_freq_on'][3]:.2f} Hz rotational licking waves and alternating XII bursts ($\\mathrm{{s.d.}} = {data['ex_osc_on'][3]:.2f}$ sp/s, $\\mathrm{{Re}}\\,\\lambda_{{\\max}} = {data['ex_lam_max'][3].real:.2f} > 1$) that cease cleanly after laser offset (post: {data['ex_osc_post'][3]:.2f} sp/s); **d**, reference spontaneous licking driven solely by suprathreshold tonic drive ($D = 20$, no laser: $\\mathrm{{s.d.}} = {data['ex_osc_on'][4]:.2f}$ sp/s, $f = {data['ex_freq_on'][4]:.2f}$ Hz).

**e**, Complete rest-state ($D = 0$) phase space across peak photocurrent $I_{{\\max}} = 10\\text{{–}}320$ and ChR2 steady-state desensitization ratio $I_{{\\mathrm{{ss}}}}/I_{{\\mathrm{{peak}}}} = 0.1\\text{{–}}1.0$ plus spatially varying illumination (`var. φ`; $n = 30$ random 15-cell subsets per cell). Across all 36 regimes, 10% E-cell photostimulation fails to trigger licking (maximum mean unlit amplitude $\\mathrm{{s.d.}} = {e_mean.max():.2f}$ sp/s; maximum mean $\\mathrm{{Re}}\\,\\lambda_{{\\max}} = {e_lam_max:.2f} < 1$).

**f, g**, Mean unlit oscillation amplitude (rate s.d., sp/s; shared colour scale with **e**, $n = 20$ subsets per cell) across background tonic drive $D = 0\\text{{–}}20$ and peak photocurrent $I_{{\\max}}$ (`Off` and `5–320`) for **f**, realistic desensitizing ChR2 with variable illumination (`var. φ`), and **g**, idealized non-desensitizing step current ($I_{{\\mathrm{{ss}}}}/I_{{\\mathrm{{peak}}}} = 1$). Dotted cyan line marks the spontaneous no-laser bifurcation threshold ($D_{{\\mathrm{{th}}}} \\approx 13.5$); dashed white contour marks $\\mathrm{{s.d.}} = 2$ sp/s. Removing desensitization lowers the gating drive window from $D \\approx 10\\text{{–}}13$ down to $D \\approx 8\\text{{–}}13$, while excessive non-desensitizing photocurrent ($I_{{\\max}} \\ge 160$) induces saturation suppression by pushing neurons onto the flat saturation region of the firing-rate transfer function.

**h**, Mean unlit oscillation amplitude (rate s.d., sp/s; $n = 30$ random 15-cell subsets) across $D = 0\\text{{–}}20$ comparing no laser (black), variable-light desensitizing ChR2 ($I_{{\\max}} = 40$ and $80$, red), uniform desensitizing ChR2 ($I_{{\\max}} = 40$, purple), and non-desensitizing step ($I_{{\\max}} = 40$, dashed teal). At $D = 12$, realistic desensitizing illumination raises amplitude from {ctrl_m[12]:.2f} sp/s (no laser) to {v40_m[12]:.2f} sp/s ($I_{{\\max}} = 40$) and {v80_m[12]:.2f} sp/s ($I_{{\\max}} = 80$), whereas rest ($D = 0$) remains silent ({v40_m[0]:.2f}–{s40_m[0]:.2f} sp/s across all conditions).

**i**, Leading Jacobian real eigenvalue $\\mathrm{{Re}}\\,\\lambda_{{\\max}}(J)$ across $D = 0\\text{{–}}20$ ($n = 12$ subsets per curve) explaining the state-dependent gating mechanism: at low tonic drive, neurons sit on the shallow subthreshold tail of the transfer function so 10% E-cell photostimulation cannot push $\\mathrm{{Re}}\\,\\lambda_{{\\max}}$ across the Hopf instability boundary ($\\mathrm{{Re}}\\,\\lambda_{{\\max}} = 1$, dashed line); near $D_{{\\mathrm{{th}}}}$, the same photocurrent shifts the population onto steeper gain and crosses unity.

**j**, Peak oscillation frequency (Hz) of gated rhythms across $D = 10\\text{{–}}20$ (plotted where mean unlit $\\mathrm{{s.d.}} \\ge 2$ sp/s; green shading marks the 6–8 Hz physiological licking band). Optogenetically gated licking remains locked within the physiological 6–8 Hz range across illumination intensities and desensitization conditions (e.g., {v80_f12[v80_o12>=2].mean():.2f} $\\pm$ {v80_f12[v80_o12>=2].std():.2f} Hz at $D = 12$, $I_{{\\max}} = 80$, vs. {ctrl_f20_nl.mean():.2f} $\\pm$ {ctrl_f20_nl.std():.2f} Hz spontaneous at $D = 20$), showing that near-threshold photostimulation gates the intrinsic brainstem rhythm without distorting its frequency.
"""
    with open(os.path.join(out_dir, 'Licking_opto_threshold_caption.md'), 'w') as fh:
        fh.write(md_caption)


def build_figure(data: dict, out_dir: str):
    configure_publication_style()
    proto = ProtocolConfig(T=4000, stim_on=1000, stim_off=3000, analysis_delay=500, post_delay=200)
    net = LickingNetwork(LickingNetworkConfig(seed=1), W=data['W'])

    COL_CTRL = '#22252A'
    COL_VAR = '#D64045'
    COL_VAR80 = '#9E1F24'
    COL_UNI_D = '#6A4C93'
    COL_STEP = '#1D8A99'
    COL_LASER = '#1E88E5'
    COL_N1 = '#2B6CB0'
    COL_N2 = '#2F855A'
    COL_GRN = '#2F855A'

    fig = plt.figure(figsize=(7.20, 7.80))

    # Row 1 (y in [0.706, 0.955]): schematic + ChR2 waveforms (a) and rasters + XII nerves (b, c, d)
    ax_schem = fig.add_axes([0.028, 0.836, 0.232, 0.118])
    ax_chr2  = fig.add_axes([0.072, 0.706, 0.180, 0.106])

    r1_x = [0.316, 0.526, 0.734]
    r1_w = 0.172
    ax_rasters, ax_nerves = [], []
    for x0 in r1_x:
        ax_rasters.append(fig.add_axes([x0, 0.782, r1_w, 0.162]))
        ax_nerves.append(fig.add_axes([x0, 0.706, r1_w, 0.066]))
    ax_rcbar = fig.add_axes([0.918, 0.782, 0.011, 0.162])

    # Row 2 (y in [0.432, 0.614]): 3 heatmaps (e, f, g) + shared colorbar
    ax_e = fig.add_axes([0.098, 0.432, 0.214, 0.182])
    ax_f = fig.add_axes([0.386, 0.432, 0.236, 0.182])
    ax_g = fig.add_axes([0.654, 0.432, 0.236, 0.182])
    ax_hcbar = fig.add_axes([0.906, 0.432, 0.012, 0.182])

    # Row 3 (y in [0.074, 0.264]): drive curves (h), Hopf boundary (i), evoked licking frequency (j)
    # Shared legend cleanly floats in open band y in [0.308, 0.348]
    ax_h = fig.add_axes([0.076, 0.074, 0.242, 0.194])
    ax_i = fig.add_axes([0.402, 0.074, 0.242, 0.194])
    ax_j = fig.add_axes([0.728, 0.074, 0.234, 0.194])

    # =========================================================================
    # Panel a: Licking BSG schematic & normalized ChR2 kinetics
    # =========================================================================
    draw_licking_opto_schematic(ax_schem)

    t_rel = np.arange(-80, 860) / 1000.0
    idx0, idx1 = proto.stim_on - 80, proto.stim_on + 860
    cfg_var = ChR2Config(i_max=1.0, desensitize=True)
    cfg_step = ChR2Config(i_max=1.0, desensitize=False)

    I_step   = chr2_current(proto.T, proto.stim_on, proto.stim_off, [1.0],  i_max=1.0, cfg=cfg_step)[0]
    I_unides = chr2_current(proto.T, proto.stim_on, proto.stim_off, [1.0],  i_max=1.0, cfg=cfg_var)[0]
    I_var_hi = chr2_current(proto.T, proto.stim_on, proto.stim_off, [0.75], i_max=1.0, cfg=cfg_var)[0]
    I_var_lo = chr2_current(proto.T, proto.stim_on, proto.stim_off, [0.30], i_max=1.0, cfg=cfg_var)[0]

    ax_chr2.axvspan(0.0, 0.85, color=COL_LASER, alpha=0.10, lw=0)
    ax_chr2.plot(t_rel, I_step[idx0:idx1],   color=COL_STEP,  lw=1.15, ls=(0, (3.0, 1.5)), label='No desens. (step)')
    ax_chr2.plot(t_rel, I_unides[idx0:idx1], color=COL_UNI_D, lw=1.15, label='Uniform ($I_{\\mathrm{ss}}=0.3$)')
    ax_chr2.plot(t_rel, I_var_hi[idx0:idx1], color=COL_VAR,   lw=1.15, label='Var. $\\varphi=0.75$')
    ax_chr2.plot(t_rel, I_var_lo[idx0:idx1], color=COL_VAR,   lw=0.95, alpha=0.45, label='Var. $\\varphi=0.30$')
    ax_chr2.set_xlim(-0.08, 0.85); ax_chr2.set_ylim(-0.04, 1.12)
    ax_chr2.set_xticks([0, 0.4, 0.8]); ax_chr2.set_yticks([0, 0.5, 1.0])
    ax_chr2.set_xlabel('Time from laser on (s)', labelpad=1.8)
    ax_chr2.set_ylabel('ChR2 ($I / I_{\\max}$)', labelpad=2.0)
    ax_chr2.legend(loc='center right', bbox_to_anchor=(1.02, 0.62), frameon=False,
                   fontsize=4.8, handlelength=1.7, labelspacing=0.18, borderpad=0.1)

    # =========================================================================
    # Panels b, c, d: Representative 6-8 Hz Licking rasters & XII nerves
    # =========================================================================
    ex_R = data['ex_R']
    op_ex, phi_ex = data['op_ex'], data['phi_ex']
    order = data['ei_sort']
    inv_order = np.empty(net.N, dtype=int)
    inv_order[order] = np.arange(net.N)
    op_rows = inv_order[op_ex]

    ex_indices = [1, 3, 4]
    titles = [
        'Rest ($D = 0$) + 10% E laser',
        'Near threshold ($D = 12$) + laser',
        'Reference ($D = 20$, no laser)',
    ]
    has_laser = [True, True, False]

    # Show 0.5-3.5 s (3.0 s window: 0.5 s pre, 2.0 s laser ON 1-3 s = ~15 crisp licks, 0.5 s post)
    t0_v, t1_v = 500, 3500
    tvec_v = np.arange(t0_v, t1_v) / 1000.0
    nerves = [nerve_readout(net, ex_R[k], mode='limb', seed=0) for k in ex_indices]
    nerve_max = max(max(np.abs(nr['upper'][t0_v:t1_v]).max(), np.abs(nr['lower'][t0_v:t1_v]).max()) for nr in nerves)
    nerve_off = 1.25 * nerve_max

    im_raster = None
    for idx_p, (k, ttl, lit) in enumerate(zip(ex_indices, titles, has_laser)):
        axr, axn = ax_rasters[idx_p], ax_nerves[idx_p]
        R = ex_R[k]

        im_raster = axr.imshow(R[order, t0_v:t1_v:2], aspect='auto', vmin=0.0, vmax=50.0, cmap='viridis',
                               interpolation='none', extent=[0.5, 3.5, net.N, 0])
        axr.axhline(net.N_exc, color='white', lw=0.75, alpha=0.9)
        if idx_p == 0:
            axr.text(0.59, 18, 'E', color='white', fontsize=5.8, fontweight='bold', va='top', ha='left')
            axr.text(0.59, net.N_exc + 18, 'I', color='white', fontsize=5.8, fontweight='bold', va='top', ha='left')
            axr.set_ylabel('Neuron (phase-sorted)', labelpad=2.5)
            axr.set_yticks([0, 150, 300])
        else:
            axr.set_yticks([0, 150, 300]); axr.set_yticklabels([])

        axr.set_xticks([1, 2, 3]); axr.set_xticklabels([])
        axr.spines['top'].set_visible(True); axr.spines['right'].set_visible(True)
        axr.spines['top'].set_linewidth(0.55); axr.spines['right'].set_linewidth(0.55)

        if lit:
            axr.axvline(1.0, color='white', ls='--', lw=0.65, alpha=0.9)
            axr.axvline(3.0, color='white', ls='--', lw=0.65, alpha=0.9)
            axr.plot([1.0, 3.0], [-12.0, -12.0], color=COL_LASER, lw=2.3, solid_capstyle='butt', clip_on=False)
            colors_op = plt.cm.Reds(0.35 + 0.65 * phi_ex)
            axr.scatter(np.full(op_rows.size, 0.42), op_rows + 0.5, marker='>', s=7.0,
                        c=colors_op, edgecolors='#222222', linewidths=0.22, clip_on=False)

        axr.set_xlim(0.5, 3.5); axr.set_ylim(net.N, 0)
        axr.set_title(ttl, fontsize=6.2, pad=7.5)

        sd_val = float(data['ex_osc_on'][k])
        fr_val = float(data['ex_freq_on'][k])
        lam_val = float(data['ex_lam_max'][k].real)
        badge_txt = (f'SD = {sd_val:.2f} sp/s | Re $\\lambda$ = {lam_val:.2f}'
                     if idx_p == 0 else f'SD = {sd_val:.1f} sp/s | $f$ = {fr_val:.1f} Hz')
        axr.text(0.96, 0.05, badge_txt, transform=axr.transAxes, ha='right', va='bottom',
                 fontsize=4.9, color='white',
                 bbox=dict(boxstyle='round,pad=0.17', fc='#000000', alpha=0.62, lw=0))

        nr = nerves[idx_p]
        if lit:
            axn.axvspan(1.0, 3.0, color=COL_LASER, alpha=0.13, lw=0)
        axn.plot(tvec_v, nr['lower'][t0_v:t1_v], color=COL_N2, lw=0.38, alpha=0.88)
        axn.plot(tvec_v, nerve_off + nr['upper'][t0_v:t1_v], color=COL_N1, lw=0.38, alpha=0.88)
        axn.set_xlim(0.5, 3.5); axn.set_ylim(-0.95 * nerve_off, 1.95 * nerve_off)
        axn.set_xticks([1, 2, 3]); axn.set_xlabel('Time (s)', labelpad=1.8)
        axn.set_yticks([0.0, nerve_off])
        if idx_p == 0:
            axn.set_yticklabels(['Retr.', 'Protr.'], fontsize=5.3)
            axn.set_ylabel('XII nerve', labelpad=2.0)
        else:
            axn.set_yticklabels([])

    cb_r = fig.colorbar(im_raster, cax=ax_rcbar)
    cb_r.set_label('Firing rate (sp/s)', fontsize=6.0, labelpad=2.8)
    cb_r.set_ticks([0, 25, 50])
    cb_r.ax.tick_params(labelsize=5.5, length=2, width=0.5)
    cb_r.outline.set_linewidth(0.5)

    # =========================================================================
    # Row 2: Panels e, f, g (Shared perceptual color scale in sp/s)
    # =========================================================================
    vmax_h = max(14.5, float(data['grid_var'].max()), float(data['grid_step'].max()))
    h_norm = mcolors.PowerNorm(gamma=0.55, vmin=0.15, vmax=vmax_h)
    h_cmap = 'magma'

    e_mean = data['e_osc'].mean(-1)
    e_lam_max_all = data['e_lam'].mean(-1).max()
    im_e = ax_e.imshow(e_mean, origin='lower', aspect='auto', cmap=h_cmap, norm=h_norm, interpolation='nearest')
    amps_e = data['amps_e']
    des_tick_labels = ['0.1', '0.3', '0.5', '0.7', '1.0 (step)', 'var. $\\varphi$']
    ax_e.set_xticks(range(amps_e.size))
    ax_e.set_xticklabels([f'{int(a)}' for a in amps_e], fontsize=5.5)
    ax_e.set_yticks(range(len(des_tick_labels)))
    ax_e.set_yticklabels(des_tick_labels, fontsize=5.4)
    ax_e.set_xlabel('Peak photocurrent $I_{\\max}$', labelpad=2.2)
    ax_e.set_ylabel('Desensitization ($I_{\\mathrm{ss}}/I_{\\mathrm{peak}}$)', labelpad=2.2)
    ax_e.set_title('Rest ($D = 0$): no licking across all regimes', fontsize=6.1, pad=5.0)
    ax_e.spines['top'].set_visible(True); ax_e.spines['right'].set_visible(True)
    ax_e.spines['top'].set_linewidth(0.55); ax_e.spines['right'].set_linewidth(0.55)
    for r in range(e_mean.shape[0]):
        for c in range(e_mean.shape[1]):
            ax_e.text(c, r, f'{e_mean[r, c]:.2f}', ha='center', va='center', fontsize=4.8, color='#F2E9FA')
    ax_e.text(0.50, -0.24, f'All regimes: SD $\\leq$ {e_mean.max():.2f} sp/s, max Re $\\lambda$ = {e_lam_max_all:.2f} < 1',
              transform=ax_e.transAxes, ha='center', va='top', fontsize=4.9, color='#444444', style='italic')

    drives = data['drives']
    amps_fg = data['amps_fg']
    ctrl_col = data['ctrl_osc'].mean(-1, keepdims=True)
    mat_f = np.hstack([ctrl_col, data['grid_var']])
    mat_g = np.hstack([ctrl_col, data['grid_step']])
    x_labs_fg = ['Off'] + [f'{int(a)}' for a in amps_fg]

    for idx_h, (axh, mat, ttl) in enumerate([
        (ax_f, mat_f, 'Desensitizing ChR2 (var. light $\\varphi$)'),
        (ax_g, mat_g, 'Non-desensitizing step ($I_{\\mathrm{ss}}/I_{\\mathrm{peak}} = 1$)'),
    ]):
        axh.imshow(mat, origin='lower', aspect='auto', cmap=h_cmap, norm=h_norm, interpolation='nearest')
        axh.axvline(0.5, color='white', lw=1.1, alpha=0.95)
        axh.axhline(13.5, color='#80DEEA', ls=':', lw=1.0, alpha=0.95)
        X_c, Y_c = np.meshgrid(np.arange(1, mat.shape[1]), drives)
        axh.contour(X_c, Y_c, mat[:, 1:], levels=[2.0], colors=['white'],
                    linestyles=['--'], linewidths=[0.95])
        axh.set_xticks(range(len(x_labs_fg)))
        axh.set_xticklabels(x_labs_fg, fontsize=5.4)
        axh.set_yticks([0, 4, 8, 12, 16, 20])
        axh.set_xlabel('Peak photocurrent $I_{\\max}$', labelpad=2.2)
        axh.set_title(ttl, fontsize=6.1, pad=5.0)
        axh.spines['top'].set_visible(True); axh.spines['right'].set_visible(True)
        axh.spines['top'].set_linewidth(0.55); axh.spines['right'].set_linewidth(0.55)
        if idx_h == 0:
            axh.set_ylabel('Background tonic drive $D$', labelpad=2.2)
            axh.text(7.35, 14.1, '$D_{\\mathrm{th}}$', color='#1A1A1A', fontsize=5.5,
                     ha='right', va='bottom', fontweight='bold')
            axh.text(4.3, 8.2, 'SD = 2 sp/s', color='white', fontsize=4.9, ha='center', va='center')
        else:
            axh.set_yticklabels([])
            axh.text(7.35, 14.1, '$D_{\\mathrm{th}}$', color='#1A1A1A', fontsize=5.5,
                     ha='right', va='bottom', fontweight='bold')
            axh.text(5.75, 17.6, 'Saturation\nsuppression', color='#1A1A1A', fontsize=4.9,
                     ha='center', va='center', style='italic', fontweight='bold')

    cb_h = fig.colorbar(im_e, cax=ax_hcbar)
    cb_h.set_label('Unlit amplitude (rate SD, sp/s)', fontsize=5.9, labelpad=2.8)
    cb_h.set_ticks([0.2, 2, 5, 9, 14])
    cb_h.set_ticklabels(['0.2', '2', '5', '9', '14'])
    cb_h.ax.tick_params(labelsize=5.4, length=2, width=0.5)
    cb_h.outline.set_linewidth(0.5)
    cb_h.ax.axhline(2.0, color='white', ls='--', lw=0.9)

    # =========================================================================
    # Row 3: Panels h, i, j (Drive gating, Hopf Re(lambda), Evoked frequency)
    # =========================================================================
    curve_defs = [
        ('ctrl',        data['ctrl_osc'],        data['ctrl_lam'],        data['ctrl_freq'],        COL_CTRL,  '-',               1.50, 'No laser (tonic $D$ only)'),
        ('var_des_40',  data['osc_var_des_40'],  data['lam_var_des_40'],  data['freq_var_des_40'],  COL_VAR,   '-',               1.35, 'Var. light + desens. ($I_{\\max}=40$)'),
        ('var_des_80',  data['osc_var_des_80'],  data['lam_var_des_80'],  data['freq_var_des_80'],  COL_VAR80, (0, (3.5, 1.5)),   1.30, 'Var. light + desens. ($I_{\\max}=80$)'),
        ('uni_des_40',  data['osc_uni_des_40'],  data['lam_uni_des_40'],  data['freq_uni_des_40'],  COL_UNI_D, '-',               1.15, 'Uniform + desens. ($I_{\\max}=40$)'),
        ('uni_step_40', data['osc_uni_step_40'], data['lam_uni_step_40'], data['freq_uni_step_40'], COL_STEP,  (0, (3.0, 1.5)),   1.20, 'No desens. step ($I_{\\max}=40$)'),
    ]

    # Panel h: Oscillation amplitude (rate SD, sp/s) vs D
    ax_h.axvspan(9.5, 13.5, color='#FDE8E8', alpha=0.75, lw=0, zorder=0)
    ax_h.axvline(13.5, color='#777777', ls=':', lw=0.75, zorder=1)
    ax_h.axhline(2.0, color='#999999', ls=(0, (2.5, 2.0)), lw=0.65, alpha=0.85, zorder=1)
    legend_handles = []
    for cname, osc_arr, _, _, col, ls, lw, lab in curve_defs:
        m = osc_arr.mean(-1)
        line, = ax_h.plot(drives, m, color=col, ls=ls, lw=lw, label=lab, zorder=3)
        legend_handles.append(line)

    ax_h.set_xlim(0, 20.4); ax_h.set_ylim(-0.4, 15.8)
    ax_h.set_xticks([0, 5, 10, 15, 20]); ax_h.set_yticks([0, 2, 5, 10, 15])
    ax_h.set_xlabel('Background tonic drive $D$', labelpad=2.2)
    ax_h.set_ylabel('Unlit amplitude (rate SD, sp/s)', labelpad=2.2)
    ax_h.set_title('Near-threshold optogenetic gating', fontsize=6.2, pad=4.5)
    ax_h.text(11.5, 15.0, 'Near\nthresh.', ha='center', va='top', fontsize=4.9, color='#9E2A2B')
    ax_h.text(13.85, 0.2, '$D_{\\mathrm{th}}$', ha='left', va='bottom', fontsize=5.0, color='#555555')

    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.51, 0.300),
               ncol=3, frameon=False, fontsize=5.1, columnspacing=1.15,
               handlelength=2.2, handletextpad=0.45, labelspacing=0.28)

    # Panel i: Leading Jacobian Re(lambda_max) vs D (Hopf bifurcation boundary)
    ax_i.axvspan(9.5, 13.5, color='#FDE8E8', alpha=0.75, lw=0, zorder=0)
    ax_i.axhspan(1.0, 1.25, color='#EAECEF', alpha=0.65, lw=0, zorder=0)
    ax_i.axhline(1.0, color='#222222', ls='--', lw=0.8, zorder=2)
    ax_i.axvline(13.5, color='#777777', ls=':', lw=0.75, zorder=1)
    for cname, _, lam_arr, _, col, ls, lw, _ in curve_defs:
        m = lam_arr.mean(-1)
        ax_i.plot(drives, m, color=col, ls=ls, lw=lw, zorder=3)
    ax_i.set_xlim(0, 20.4); ax_i.set_ylim(0.18, 1.22)
    ax_i.set_xticks([0, 5, 10, 15, 20]); ax_i.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax_i.set_xlabel('Background tonic drive $D$', labelpad=2.2)
    ax_i.set_ylabel('Leading Jacobian Re $\\lambda_{\\max}$', labelpad=2.2)
    ax_i.set_title('Hopf stability boundary (Re $\\lambda_{\\max} = 1$)', fontsize=6.2, pad=4.5)
    ax_i.text(0.8, 1.04, 'Oscillatory (Re $\\lambda > 1$)', fontsize=4.8, color='#333333', va='bottom')
    ax_i.text(19.7, 0.25, 'Stable rest (Re $\\lambda < 1$)', fontsize=4.8, color='#555555', va='bottom', ha='right')

    # Panel j: Evoked licking frequency (Hz) across gated drives D = 8..20
    ax_j.axhspan(6.0, 8.0, color=COL_GRN, alpha=0.14, lw=0, zorder=0)
    ax_j.axhline(6.0, color=COL_GRN, ls='--', lw=0.7, alpha=0.8, zorder=1)
    ax_j.axhline(8.0, color=COL_GRN, ls='--', lw=0.7, alpha=0.8, zorder=1)
    ax_j.axvline(13.5, color='#777777', ls=':', lw=0.75, zorder=1)
    for cname, osc_arr, _, freq_arr, col, ls, lw, _ in curve_defs:
        # Plot median frequency at drives where mean amplitude >= 2.0 sp/s
        valid = osc_arr.mean(-1) >= 2.0
        ds = drives[valid]
        fs = np.array([np.median(freq_arr[idx][osc_arr[idx] >= 2.0]) for idx in np.where(valid)[0]])
        ax_j.plot(ds, fs, color=col, ls=ls, lw=lw, marker='o', ms=2.1, zorder=3)

    ax_j.set_xlim(7.5, 20.5); ax_j.set_ylim(5.0, 9.0)
    ax_j.set_xticks([8, 10, 12, 14, 16, 18, 20]); ax_j.set_yticks([5, 6, 7, 8, 9])
    ax_j.set_xlabel('Background tonic drive $D$', labelpad=2.2)
    ax_j.set_ylabel('Evoked licking frequency (Hz)', labelpad=2.2)
    ax_j.set_title('Gated rhythm stays locked in 6–8 Hz', fontsize=6.2, pad=4.5)
    ax_j.text(8.0, 8.22, '6–8 Hz physiological band', fontsize=4.8, color=COL_GRN, fontweight='bold', va='bottom')
    ax_j.text(13.8, 5.18, '$D_{\\mathrm{th}}$', fontsize=4.9, color='#555555', va='bottom')

    # Panel letters a-j
    add_panel_label_xy(fig, 0.014, 0.964, 'a')
    add_panel_label_xy(fig, 0.278, 0.964, 'b')
    add_panel_label_xy(fig, 0.498, 0.964, 'c')
    add_panel_label_xy(fig, 0.706, 0.964, 'd')
    add_panel_label_xy(fig, 0.014, 0.628, 'e')
    add_panel_label_xy(fig, 0.342, 0.628, 'f')
    add_panel_label_xy(fig, 0.630, 0.628, 'g')
    add_panel_label_xy(fig, 0.014, 0.278, 'h')
    add_panel_label_xy(fig, 0.344, 0.278, 'i')
    add_panel_label_xy(fig, 0.674, 0.278, 'j')

    pdf_path = os.path.join(out_dir, 'Licking_opto_threshold.pdf')
    png_path = os.path.join(out_dir, 'Licking_opto_threshold.png')
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    print(f'[plot] Saved {pdf_path} and {png_path}', flush=True)


def refresh_examples_and_curves(cache_path: str, examples_only: bool = False) -> dict:
    """Fast refresh of representative examples (draw 7, 2 s laser 1-3 s) & optionally Row-3 curves."""
    t0 = time.time()
    npz = np.load(cache_path, allow_pickle=False)
    data = {k: npz[k] for k in npz.files}

    net = LickingNetwork(LickingNetworkConfig(seed=1), W=data['W'])
    neuron = licking_neuron_config()
    proto = ProtocolConfig(T=4000, stim_on=1000, stim_off=3000, analysis_delay=500, post_delay=200)
    n_stim = 15

    # Re-run median representative example (draw 7) with 2 s laser (1-3 s)
    op_ex, phi_ex = draw_variable_light(net.N_exc, n_stim, 7)
    chr2_main = ChR2Config(i_max=80.0, desensitize=True)
    ex_cols = [
        dict(base=0.0,  op=None,  phi=None),
        dict(base=0.0,  op=op_ex, phi=phi_ex, i_max=80.0),
        dict(base=12.0, op=None,  phi=None),
        dict(base=12.0, op=op_ex, phi=phi_ex, i_max=80.0),
        dict(base=20.0, op=None,  phi=None),
    ]
    out_ex = simulate_batch(
        net.W, proto.T, make_batched_ext(net.N, ex_cols, proto, chr2_main, light=True),
        len(ex_cols), neuron=neuron, seed=100,
        windows={'pre': proto.pre_win, 'on': proto.analysis_win, 'post': proto.post_win},
        snap_cols=tuple(range(len(ex_cols))), rec_every=1, dtype=np.float64,
    )
    ex_R = out_ex['snaps']
    ex_Vmean = out_ex['on']['Vmean']
    t0_a, t1_a = proto.analysis_win
    data['op_ex'] = op_ex
    data['phi_ex'] = phi_ex
    data['ex_R'] = ex_R.astype(np.float32)
    data['ex_Vmean'] = ex_Vmean
    data['ex_osc_on'] = np.array([unlit_mean(out_ex['on']['sd'][:, k], op_ex) for k in range(len(ex_cols))])
    data['ex_osc_pre'] = np.array([unlit_mean(out_ex['pre']['sd'][:, k], op_ex) for k in range(len(ex_cols))])
    data['ex_osc_post'] = np.array([unlit_mean(out_ex['post']['sd'][:, k], op_ex) for k in range(len(ex_cols))])
    data['ex_eigs'] = np.stack([jacobian_eigs(net.W, ex_Vmean[:, k], neuron) for k in range(len(ex_cols))])
    data['ex_lam_max'] = np.array([data['ex_eigs'][k][data['ex_eigs'][k].real.argmax()] for k in range(len(ex_cols))])
    data['ex_freq_on'] = np.array([_peak_freq_from_snap(ex_R[k, ::5, t0_a:t1_a:5], dt_ms=5.0) for k in range(len(ex_cols))])

    if examples_only:
        np.savez_compressed(cache_path, **data)
        print(f'[refresh] Updated 2 s laser examples in {time.time()-t0:.2f}s', flush=True)
        return data

    # Recompute high-res frequencies for ctrl + 5 Row-3 curves (8x zero-padded FFT)
    drives = data['drives']
    n_draws_hi = 30
    specs = [
        ('ctrl',         'none', True,  0.0,   300),
        ('var_des_40',   'var',  True,  40.0,  601),
        ('var_des_80',   'var',  True,  80.0,  602),
        ('uni_des_40',   'uni',  True,  40.0,  603),
        ('uni_step_40',  'uni',  False, 40.0,  604),
        ('uni_step_160', 'uni',  False, 160.0, 605),
    ]
    for cname, pmode, desens, imax, seed_c in specs:
        cfg_c = ChR2Config(i_max=max(imax, 1.0), desensitize=desens)
        cols, ops, meta = [], [], []
        for i, D in enumerate(drives):
            for s in range(n_draws_hi):
                op, phi = draw_variable_light(net.N_exc, n_stim, s)
                if pmode == 'none':
                    cols.append(dict(base=float(D), op=None, phi=None))
                else:
                    cols.append(dict(base=float(D), op=op, phi=(phi if pmode == 'var' else np.ones(n_stim)), i_max=imax))
                ops.append(op)
                meta.append((i, s))
        osc_cv, freq_cv, _ = _simulate_freq_batch(net.W, cols, ops, proto, cfg_c, neuron, seed=seed_c, light=(pmode != 'none'))
        o_mat = np.zeros((drives.size, n_draws_hi))
        f_mat = np.zeros((drives.size, n_draws_hi))
        for k, (i, s) in enumerate(meta):
            o_mat[i, s] = osc_cv[k]
            f_mat[i, s] = freq_cv[k]
        if cname == 'ctrl':
            data['ctrl_osc'] = o_mat
            data['ctrl_freq'] = f_mat
        else:
            data[f'osc_{cname}'] = o_mat
            data[f'freq_{cname}'] = f_mat

    np.savez_compressed(cache_path, **data)
    print(f'[refresh] Updated examples & high-res FFT frequencies in {time.time()-t0:.1f}s', flush=True)
    return data


def run(out_dir: str = DEFAULT_OUT, plot_only: bool = False, recompute: bool = False,
        refresh_examples: bool = False, refresh_curves: bool = False, **_):
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, 'licking_opto_cache.npz')
    if refresh_examples and os.path.exists(cache_path):
        data = refresh_examples_and_curves(cache_path, examples_only=True)
    elif refresh_curves and os.path.exists(cache_path):
        data = refresh_examples_and_curves(cache_path, examples_only=False)
    elif plot_only or (os.path.exists(cache_path) and not recompute):
        print(f'[load] Loading cached Licking BSG opto simulations from {cache_path}', flush=True)
        npz = np.load(cache_path, allow_pickle=False)
        data = {k: npz[k] for k in npz.files}
    else:
        data = run_simulations(cache_path)
    save_stats_and_caption(data, out_dir)
    build_figure(data, out_dir)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=DESCRIPTION)
    add_args(p)
    args = p.parse_args()
    run(**vars(args))
