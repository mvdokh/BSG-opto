"""Publication figure: 10% E-cell optogenetic stimulation fails to evoke rhythmic
firing across illumination intensities and desensitization regimes unless background
tonic drive is already near threshold, plus 500-shuffle subset variability analysis.

Usage:
    python3 -m bsg_opto.experiments.opto_threshold_figure
    python3 -m bsg_opto.experiments.opto_threshold_figure --plot-only
    .venv/bin/python run.py opto-threshold-figure
"""
from __future__ import annotations

import os
import sys
import time
import types
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Ensure repo root and BSG/ are importable when invoked directly as a script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_BSG_DIR = os.path.join(_ROOT, 'BSG')
os.environ.setdefault('MPLCONFIGDIR', os.path.join(_ROOT, '.mplcache'))
for _p in (_ROOT, _BSG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Graceful shim if numba / astropy are absent in the active interpreter
# (BSG/spinalsim.py stays completely untouched; simulations use pure-NumPy BLAS.)
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

from bsg_opto.config import ChR2Config, NetworkConfig, NeuronConfig, ProtocolConfig, RESULTS_DIR
from bsg_opto.network import BalancedNetwork
from bsg_opto.opsin import chr2_current, chr2_params, draw_variable_light, pick_random_exc
from bsg_opto.simulate import make_batched_ext, simulate_batch
from bsg_opto.analysis import jacobian_eigs, leading_real_eig, nerve_readout, transfer_slope, unlit_mean

DESCRIPTION = 'Generate publication multi-panel figure and caption for 10% optogenetic gating and shuffle variability'

DEFAULT_OUT = os.path.join(RESULTS_DIR, 'opto_threshold_figure')


def add_args(parser):
    parser.add_argument('--out-dir', type=str, default=DEFAULT_OUT, help='Output directory')
    parser.add_argument('--plot-only', action='store_true', help='Re-render figure from cached simulation data')
    parser.add_argument('--recompute', action='store_true', help='Force recomputation even if caches exist')


# =============================================================================
# 1. Core Parameter-Sweep Simulations & Caching
# =============================================================================
def run_main_simulations(cache_path: str) -> dict:
    """Run parameter sweeps and representative examples, caching to sim_cache.npz."""
    t_start = time.time()
    net = BalancedNetwork(NetworkConfig())  # N=200 (100E/100I), C=0.1, g=1, radius=1, seed=1
    neuron = NeuronConfig()
    proto = ProtocolConfig()
    n_stim = 10  # 10% of excitatory cells (10 / 100 E cells)

    print(f'[sim] Network N={net.N} ({net.N_exc}E/{net.N_inh}I), C={net.cfg.C}, '
          f'leading eig(W)={net.leading_eigval:.4f}', flush=True)

    # ---- A. Representative examples (full 1-ms resolution for panels b, c, d, j)
    ex_draw = 19
    op_ex, phi_ex = draw_variable_light(net.N_exc, n_stim, ex_draw)
    chr2_main = ChR2Config(i_max=80.0, desensitize=True)
    ex_cols = [
        dict(base=0.0, op=None, phi=None),
        dict(base=0.0, op=op_ex, phi=phi_ex, i_max=80.0),
        dict(base=12.0, op=None, phi=None),
        dict(base=12.0, op=op_ex, phi=phi_ex, i_max=80.0),
        dict(base=12.0, op=op_ex, phi=phi_ex, i_max=40.0),
        dict(base=20.0, op=None, phi=None),
        dict(base=0.0, op=op_ex, phi=phi_ex, i_max=40.0),
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

    # ---- B. Panel e: Rest (D = 0) Intensity x Desensitization matrix (n_draws = 30)
    amps_e = np.array([10., 20., 40., 80., 160., 320.])
    n_draws_e = 30
    e_osc = np.zeros((6, amps_e.size, n_draws_e))
    e_lam = np.zeros((6, amps_e.size, 10))
    for r, row_key in enumerate([0.1, 0.3, 0.5, 0.7, 1.0, 'var']):
        cfg_r = ChR2Config(desensitize=True) if row_key == 'var' \
            else ChR2Config(ss_frac=float(row_key), desensitize=(float(row_key) < 1.0))
        cols, meta = [], []
        for c_i, amp in enumerate(amps_e):
            for s in range(n_draws_e):
                op, phi = draw_variable_light(net.N_exc, n_stim, s)
                p_use = phi if row_key == 'var' else np.ones(n_stim)
                cols.append(dict(base=0.0, op=op, phi=p_use, i_max=float(amp)))
                meta.append((c_i, s, op))
        out = simulate_batch(
            net.W, proto.T, make_batched_ext(net.N, cols, proto, cfg_r, light=True),
            len(cols), neuron=neuron, seed=200 + r,
            windows={'on': proto.analysis_win}, snap_cols=(), dtype=np.float32,
        )
        for idx_col, (c_i, s, op) in enumerate(meta):
            e_osc[r, c_i, s] = unlit_mean(out['on']['sd'][:, idx_col], op)
            if s < 10:
                e_lam[r, c_i, s] = leading_real_eig(net.W, out['on']['Vmean'][:, idx_col], neuron)

    # ---- C. Panels f & g: Full D = 0..20 x Intensity (Off + 5..320)
    drives = np.arange(0, 21)
    amps_fg = np.array([5., 10., 20., 40., 80., 160., 320.])
    n_draws_fg = 20
    n_draws_hi = 30

    cols_ctrl, meta_ctrl = [], []
    for i, D in enumerate(drives):
        for s in range(n_draws_hi):
            op, _ = draw_variable_light(net.N_exc, n_stim, s)
            cols_ctrl.append(dict(base=float(D), op=None, phi=None))
            meta_ctrl.append((i, s, op))
    out_ctrl = simulate_batch(
        net.W, proto.T, make_batched_ext(net.N, cols_ctrl, proto, chr2_main, light=False),
        len(cols_ctrl), neuron=neuron, seed=300,
        windows={'on': proto.analysis_win}, snap_cols=(), dtype=np.float32,
    )
    ctrl_osc = np.zeros((drives.size, n_draws_hi))
    ctrl_lam = np.zeros((drives.size, 15))
    for idx_col, (i, s, op) in enumerate(meta_ctrl):
        ctrl_osc[i, s] = unlit_mean(out_ctrl['on']['sd'][:, idx_col], op)
        if s < 15:
            ctrl_lam[i, s] = leading_real_eig(net.W, out_ctrl['on']['Vmean'][:, idx_col], neuron)

    grid_var = np.zeros((drives.size, amps_fg.size))
    grid_step = np.zeros((drives.size, amps_fg.size))
    grid_var_lam = np.zeros((drives.size, amps_fg.size))
    grid_step_lam = np.zeros((drives.size, amps_fg.size))
    for gname, pmode, desens, target_osc, target_lam, seed_g in [
        ('var_des', 'var', True, grid_var, grid_var_lam, 400),
        ('uni_step', 'uni', False, grid_step, grid_step_lam, 500),
    ]:
        cfg_g = ChR2Config(desensitize=desens)
        cols, meta = [], []
        for i, D in enumerate(drives):
            for j, A in enumerate(amps_fg):
                for s in range(n_draws_fg):
                    op, phi = draw_variable_light(net.N_exc, n_stim, s)
                    p_use = phi if pmode == 'var' else np.ones(n_stim)
                    cols.append(dict(base=float(D), op=op, phi=p_use, i_max=float(A)))
                    meta.append((i, j, s, op))
        out = simulate_batch(
            net.W, proto.T, make_batched_ext(net.N, cols, proto, cfg_g, light=True),
            len(cols), neuron=neuron, seed=seed_g,
            windows={'on': proto.analysis_win}, snap_cols=(), dtype=np.float32,
        )
        acc_osc = np.zeros((drives.size, amps_fg.size, n_draws_fg))
        acc_lam = np.zeros((drives.size, amps_fg.size, 5))
        for idx_col, (i, j, s, op) in enumerate(meta):
            acc_osc[i, j, s] = unlit_mean(out['on']['sd'][:, idx_col], op)
            if s < 5:
                acc_lam[i, j, s] = leading_real_eig(net.W, out['on']['Vmean'][:, idx_col], neuron)
        target_osc[:] = acc_osc.mean(-1)
        target_lam[:] = acc_lam.mean(-1)

    # ---- D. Panels h & i: Detailed drive curves D = 0..20 (n_draws = 30, n_lam = 15)
    curve_specs = [
        ('var_des_40', 'var', True, 40.0, 601),
        ('uni_des_40', 'uni', True, 40.0, 602),
        ('uni_step_40', 'uni', False, 40.0, 603),
        ('var_des_80', 'var', True, 80.0, 604),
        ('uni_des_80', 'uni', True, 80.0, 605),
        ('uni_step_80', 'uni', False, 80.0, 606),
    ]
    curves_osc, curves_lam, curves_post = {}, {}, {}
    for cname, pmode, desens, imax, seed_c in curve_specs:
        cfg_c = ChR2Config(i_max=imax, desensitize=desens)
        cols, meta = [], []
        for i, D in enumerate(drives):
            for s in range(n_draws_hi):
                op, phi = draw_variable_light(net.N_exc, n_stim, s)
                p_use = phi if pmode == 'var' else np.ones(n_stim)
                cols.append(dict(base=float(D), op=op, phi=p_use, i_max=imax))
                meta.append((i, s, op))
        out = simulate_batch(
            net.W, proto.T, make_batched_ext(net.N, cols, proto, cfg_c, light=True),
            len(cols), neuron=neuron, seed=seed_c,
            windows={'on': proto.analysis_win, 'post': proto.post_win},
            snap_cols=(), dtype=np.float32,
        )
        osc = np.zeros((drives.size, n_draws_hi))
        post = np.zeros((drives.size, n_draws_hi))
        lam = np.zeros((drives.size, 15))
        for idx_col, (i, s, op) in enumerate(meta):
            osc[i, s] = unlit_mean(out['on']['sd'][:, idx_col], op)
            post[i, s] = unlit_mean(out['post']['sd'][:, idx_col], op)
            if s < 15:
                lam[i, s] = leading_real_eig(net.W, out['on']['Vmean'][:, idx_col], neuron)
        curves_osc[cname], curves_post[cname], curves_lam[cname] = osc, post, lam

    # ---- E. Mixed 10% E+I control & Multi-network verification (10 seeds)
    mixed_osc = {}
    for ss_val in (0.3, 1.0):
        cfg_m = ChR2Config(i_max=80.0, ss_frac=ss_val, desensitize=(ss_val < 1.0))
        cols, ops = [], []
        for s in range(n_draws_hi):
            op = np.sort(np.random.default_rng(8000 + s).choice(net.N, 20, replace=False))
            cols.append(dict(base=0.0, op=op, phi=np.ones(20), i_max=80.0))
            ops.append(op)
        out = simulate_batch(
            net.W, proto.T, make_batched_ext(net.N, cols, proto, cfg_m, light=True),
            len(cols), neuron=neuron, seed=700,
            windows={'on': proto.analysis_win}, snap_cols=(), dtype=np.float32,
        )
        mixed_osc[ss_val] = np.array([unlit_mean(out['on']['sd'][:, s], ops[s]) for s in range(n_draws_hi)])

    net_seeds = np.arange(1, 11)
    multi_net = np.zeros((net_seeds.size, 5))
    for ni, cs in enumerate(net_seeds):
        n2 = BalancedNetwork(NetworkConfig(seed=int(cs)))
        cols, meta = [], []
        for ci, (D, light) in enumerate([(0., False), (0., True), (12., False), (12., True), (20., False)]):
            for s in range(15):
                op, phi = draw_variable_light(n2.N_exc, n_stim, s)
                cols.append(dict(base=D, op=op if light else None, phi=phi if light else None, i_max=40.0))
                meta.append((ci, s, op))
        out = simulate_batch(
            n2.W, proto.T, make_batched_ext(n2.N, cols, proto, ChR2Config(i_max=40.0), light=True),
            len(cols), neuron=neuron, seed=900 + int(cs),
            windows={'on': proto.analysis_win}, snap_cols=(), dtype=np.float32,
        )
        acc = [[] for _ in range(5)]
        for idx_col, (ci, s, op) in enumerate(meta):
            acc[ci].append(unlit_mean(out['on']['sd'][:, idx_col], op))
        multi_net[ni] = [np.mean(a) for a in acc]

    data = dict(
        W=net.W, phase=net.phase, phase_sort=net.phase_sort, ei_sort=net.ei_phase_sort(),
        op_ex=op_ex, phi_ex=phi_ex,
        ex_R=ex_R.astype(np.float32), ex_Vmean=ex_Vmean, ex_osc_on=ex_osc_on,
        ex_osc_pre=ex_osc_pre, ex_osc_post=ex_osc_post, ex_eigs=ex_eigs, ex_lam_max=ex_lam_max,
        amps_e=amps_e, e_osc=e_osc, e_lam=e_lam,
        drives=drives, amps_fg=amps_fg,
        ctrl_osc=ctrl_osc, ctrl_lam=ctrl_lam,
        grid_var=grid_var, grid_step=grid_step, grid_var_lam=grid_var_lam, grid_step_lam=grid_step_lam,
        mixed_osc_03=mixed_osc[0.3], mixed_osc_10=mixed_osc[1.0],
        net_seeds=net_seeds, multi_net=multi_net,
    )
    for k, v in curves_osc.items():
        data[f'osc_{k}'] = v
        data[f'post_{k}'] = curves_post[k]
        data[f'lam_{k}'] = curves_lam[k]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **data)
    print(f'[sim] Saved main cache to {cache_path} ({time.time()-t_start:.1f}s)', flush=True)
    return data


# =============================================================================
# 2. 500-Shuffle Subset Variability & Mechanistic Predictor Analysis
# =============================================================================
def run_shuffle_simulations(cache_path: str) -> dict:
    """Shuffle which 10 E cells are illuminated 500 times at D=0 and D=12,
    decompose subset vs noise variance, compute closed-form linear-response
    predictors from W alone, and test causal top-10 vs bottom-10 subsets.
    """
    t0 = time.time()
    net = BalancedNetwork(NetworkConfig())
    neuron, proto = NeuronConfig(), ProtocolConfig()
    W, N, N_exc = net.W, net.N, net.N_exc
    n_stim = 10
    N_shuf = 500

    # Deterministic pool of 500 independent random 10-cell subsets + light profiles
    ops_mat = np.stack([pick_random_exc(N_exc, n_stim, 10000 + s) for s in range(N_shuf)])
    phis_mat = np.stack([np.random.default_rng(20000 + s).uniform(0.0, 1.0, n_stim) for s in range(N_shuf)])

    def _run_pool(D: float, imax: float, pmode: str, seed: int):
        cfg = ChR2Config(i_max=imax, desensitize=True)
        cols = [
            dict(base=D, op=ops_mat[s], phi=(phis_mat[s] if pmode == 'var' else np.ones(n_stim)), i_max=imax)
            for s in range(N_shuf)
        ]
        out = simulate_batch(
            W, proto.T, make_batched_ext(N, cols, proto, cfg, light=True),
            N_shuf, neuron=neuron, seed=seed,
            windows={'on': proto.analysis_win}, snap_cols=(), dtype=np.float32,
        )
        osc = np.array([unlit_mean(out['on']['sd'][:, s], ops_mat[s]) for s in range(N_shuf)])
        lam = np.array([leading_real_eig(W, out['on']['Vmean'][:, s], neuron) for s in range(N_shuf)])
        return osc, lam

    osc_d0_var40, lam_d0_var40 = _run_pool(0.0, 40.0, 'var', seed=101)
    osc_d0_uni80, lam_d0_uni80 = _run_pool(0.0, 80.0, 'uni', seed=102)
    osc_d12_uni40, lam_d12_uni40 = _run_pool(12.0, 40.0, 'uni', seed=103)
    osc_d12_var40, lam_d12_var40 = _run_pool(12.0, 40.0, 'var', seed=104)
    print(f'[shuffle] 500-shuffle pools finished ({time.time()-t0:.1f}s)', flush=True)

    # Variance decomposition: 20 fixed subsets x 25 independent noise seeds (D=12, uni_des I=40)
    n_sub, n_rep = 20, 25
    cols_vd, meta_vd = [], []
    for i in range(n_sub):
        for r in range(n_rep):
            cols_vd.append(dict(base=12.0, op=ops_mat[i], phi=np.ones(n_stim), i_max=40.0))
            meta_vd.append((i, r))
    out_vd = simulate_batch(
        W, proto.T, make_batched_ext(N, cols_vd, proto, ChR2Config(i_max=40.0), light=True),
        len(cols_vd), neuron=neuron, seed=999,
        windows={'on': proto.analysis_win}, snap_cols=(), dtype=np.float32,
    )
    vd_mat = np.zeros((n_sub, n_rep))
    for c, (i, r) in enumerate(meta_vd):
        vd_mat[i, r] = unlit_mean(out_vd['on']['sd'][:, c], ops_mat[i])

    # Analytical zero-simulation predictors computed strictly from connectivity W
    s0 = float(transfer_slope(12.0, neuron))
    L = np.linalg.inv(np.eye(N) - s0 * W)  # linear-response resolvent operator
    g_eff = (L.sum(0) - np.diag(L)) / (N - 1)  # per-neuron multi-synaptic population impact

    A_bin = (W > 0).astype(float)
    k_diff = A_bin[:N_exc, :N_exc].sum(0) - A_bin[N_exc:, :N_exc].sum(0)
    ev, vr = np.linalg.eig(W)
    v_mag = np.abs(vr[:N_exc, int(ev.real.argmax())])
    phase_E = net.phase[:N_exc]

    cfg40 = ChR2Config(i_max=40.0, desensitize=True)
    # Compute analytical predictors for BOTH variable-light (main) and uniform-light shuffles
    def _compute_predictors(pmode: str):
        I_ss_mat = np.zeros((N, N_shuf))
        pred_1hop = np.zeros(N_shuf)
        pred_dV = np.zeros(N_shuf)
        pred_lam = np.zeros(N_shuf)
        pred_totI = np.zeros(N_shuf)
        pred_vmag = np.zeros(N_shuf)
        pred_phaseR = np.zeros(N_shuf)
        for s in range(N_shuf):
            op = ops_mat[s]
            p = phis_mat[s] if pmode == 'var' else np.ones(n_stim)
            ss, _ = chr2_params(p, cfg40)
            i_ss = 40.0 * p * ss
            I_ss_mat[op, s] = i_ss
            pred_totI[s] = i_ss.sum()
            pred_1hop[s] = np.dot(i_ss, k_diff[op])
            pred_dV[s] = np.dot(i_ss, g_eff[op])
            pred_vmag[s] = np.dot(i_ss, v_mag[op])
            pred_phaseR[s] = np.abs(np.mean(np.exp(1j * phase_E[op])))
        dV_lin = L @ I_ss_mat
        for s in range(N_shuf):
            pred_lam[s] = leading_real_eig(W, 12.0 + dV_lin[:, s], neuron)
        return dict(
            pred_lam=pred_lam, pred_dV=pred_dV, pred_1hop=pred_1hop,
            pred_totI=pred_totI, pred_vmag=pred_vmag, pred_phaseR=pred_phaseR,
        )

    pv_var = _compute_predictors('var')
    pv_uni = _compute_predictors('uni')

    # Causal validation: top-10 vs bottom-10 E cells ranked solely by analytical g_eff
    top10 = np.sort(np.argsort(g_eff[:N_exc])[::-1][:n_stim])
    bot10 = np.sort(np.argsort(g_eff[:N_exc])[:n_stim])
    causal_cols = [
        dict(base=0.0, op=top10, phi=np.ones(n_stim), i_max=40.0),
        dict(base=0.0, op=bot10, phi=np.ones(n_stim), i_max=40.0),
        dict(base=12.0, op=top10, phi=np.ones(n_stim), i_max=40.0),
        dict(base=12.0, op=bot10, phi=np.ones(n_stim), i_max=40.0),
    ]
    causal_ops = [top10, bot10, top10, bot10]
    out_c = simulate_batch(
        W, proto.T, make_batched_ext(N, causal_cols, proto, cfg40, light=True),
        len(causal_cols), neuron=neuron, seed=42,
        windows={'on': proto.analysis_win}, snap_cols=(), dtype=np.float64,
    )
    causal_osc = np.array([unlit_mean(out_c['on']['sd'][:, k], causal_ops[k]) for k in range(4)])
    causal_lam = np.array([leading_real_eig(W, out_c['on']['Vmean'][:, k], neuron) for k in range(4)])
    # Analytical predicted Re(lam) for top10 and bot10 at D=12
    ss_u, _ = chr2_params(np.ones(n_stim), cfg40)
    causal_pred_lam = np.zeros(2)
    causal_pred_dV = np.zeros(2)
    for idx_c, op in enumerate((top10, bot10)):
        ivec = np.zeros(N); ivec[op] = 40.0 * ss_u
        causal_pred_lam[idx_c] = leading_real_eig(W, 12.0 + L @ ivec, neuron)
        causal_pred_dV[idx_c] = float(np.dot(40.0 * ss_u, g_eff[op]))

    shuf = dict(
        ops_mat=ops_mat, phis_mat=phis_mat, g_eff_E=g_eff[:N_exc],
        osc_d0_var40=osc_d0_var40, lam_d0_var40=lam_d0_var40,
        osc_d0_uni80=osc_d0_uni80, lam_d0_uni80=lam_d0_uni80,
        osc_d12_uni40=osc_d12_uni40, lam_d12_uni40=lam_d12_uni40,
        osc_d12_var40=osc_d12_var40, lam_d12_var40=lam_d12_var40,
        vd_mat=vd_mat,
        causal_osc=causal_osc, causal_lam=causal_lam,
        causal_pred_lam=causal_pred_lam, causal_pred_dV=causal_pred_dV,
    )
    for k, v in pv_var.items():
        shuf[f'var_{k}'] = v
    for k, v in pv_uni.items():
        shuf[f'uni_{k}'] = v

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **shuf)
    print(f'[shuffle] Saved shuffle cache to {cache_path} ({time.time()-t0:.1f}s)', flush=True)
    return shuf


# =============================================================================
# 3. Publication Plotting Style & Helpers
# =============================================================================
def configure_publication_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7.0,
        'axes.titlesize': 6.8,
        'axes.labelsize': 6.6,
        'xtick.labelsize': 5.9,
        'ytick.labelsize': 5.9,
        'legend.fontsize': 5.6,
        'axes.linewidth': 0.65,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.4,
        'ytick.major.size': 2.4,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def add_panel_label_xy(fig, x: float, y: float, letter: str):
    """Place bold lowercase panel letter at exact figure coordinates."""
    fig.text(x, y, letter, fontsize=9.2, fontweight='bold',
             va='bottom', ha='left', color='#000000')


def draw_network_schematic(ax):
    """Minimalist schematic of balanced E/I network + 10% E optogenetic protocol."""
    ax.set_xlim(-0.2, 11.4)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    c_E = '#D95F43'
    c_I = '#3B6EA5'
    c_op = '#B8252A'
    c_las = '#1E88E5'

    e_box = FancyBboxPatch((0.20, 1.35), 4.36, 2.48, boxstyle='round,pad=0.14,rounding_size=0.40',
                           fc='#FDF1ED', ec=c_E, lw=0.95)
    ax.add_patch(e_box)
    ax.text(2.38, 3.46, 'Excitatory (E)', ha='center', va='center', fontsize=5.8,
            fontweight='bold', color='#8C2D19')
    ax.text(2.38, 2.94, '$N = 100$', ha='center', va='center', fontsize=5.3, color='#555555')

    op_box = FancyBboxPatch((0.42, 1.54), 3.92, 0.90, boxstyle='round,pad=0.06,rounding_size=0.25',
                            fc='#FCE0E0', ec=c_op, lw=0.9)
    ax.add_patch(op_box)
    ax.text(2.38, 1.99, '10% ChR2 (n = 10)', ha='center', va='center', fontsize=4.7,
            fontweight='bold', color=c_op)

    i_box = FancyBboxPatch((6.64, 1.35), 4.36, 2.48, boxstyle='round,pad=0.14,rounding_size=0.40',
                           fc='#EEF4FB', ec=c_I, lw=0.95)
    ax.add_patch(i_box)
    ax.text(8.82, 3.00, 'Inhibitory (I)', ha='center', va='center', fontsize=5.8,
            fontweight='bold', color='#1F456E')
    ax.text(8.82, 2.24, '$N = 100$', ha='center', va='center', fontsize=5.3, color='#555555')

    arr_ei = FancyArrowPatch((4.66, 3.00), (6.54, 3.00), connectionstyle='arc3,rad=-0.26',
                             arrowstyle='-|>,head_length=3.3,head_width=1.9', lw=0.9, color=c_E)
    arr_ie = FancyArrowPatch((6.54, 2.05), (4.66, 2.05), connectionstyle='arc3,rad=-0.26',
                             arrowstyle='-|>,head_length=3.3,head_width=1.9', lw=0.9, color=c_I)
    ax.add_patch(arr_ei)
    ax.add_patch(arr_ie)
    ax.text(5.60, 3.68, '$C = 0.1$', ha='center', va='center', fontsize=5.3, color='#333333')
    ax.text(5.60, 1.35, '$g = 1$', ha='center', va='center', fontsize=5.3, color='#333333')

    ax.text(5.60, 5.18, 'Global tonic drive $D$ (0–20)', ha='center', va='center',
            fontsize=5.8, fontweight='bold', color='#1A1A1A')
    ax.annotate('', xy=(2.45, 4.00), xytext=(4.85, 4.86),
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color='#333333', shrinkA=0, shrinkB=0))
    ax.annotate('', xy=(8.75, 4.00), xytext=(6.35, 4.86),
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color='#333333', shrinkA=0, shrinkB=0))

    ax.text(2.38, 0.28, '473 nm laser (1–6 s)', ha='center', va='center',
            fontsize=5.7, fontweight='bold', color=c_las)
    ax.annotate('', xy=(2.38, 1.38), xytext=(2.38, 0.56),
                arrowprops=dict(arrowstyle='-|>', lw=1.35, color=c_las, shrinkA=0, shrinkB=0))


# =============================================================================
# 4. Render Complete 4-Row Publication Figure (Panels a–m)
# =============================================================================
def build_figure(data: dict, shuf: dict, out_dir: str):
    configure_publication_style()
    proto = ProtocolConfig()
    net = BalancedNetwork(NetworkConfig(), W=data['W'])

    # Restrained colorblind-safe publication palette
    COL_CTRL = '#22252A'     # deep charcoal (no laser)
    COL_VAR = '#D64045'      # vermilion red (variable light + int.-dep. desens.)
    COL_UNI_D = '#6A4C93'    # deep indigo-violet (uniform + fixed desens. ss=0.3)
    COL_STEP = '#1D8A99'     # deep teal (uniform, non-desensitizing step ss=1.0)
    COL_LASER = '#1E88E5'    # optogenetic blue
    COL_N1 = '#2B6CB0'       # nerve 1 (flexor)
    COL_N2 = '#2F855A'       # nerve 2 (extensor)

    # Full two-column journal page: 183 mm x 242 mm (7.20 in x 9.52 in)
    fig = plt.figure(figsize=(7.20, 9.52))

    # =========================================================================
    # Precise manual grid across 4 cleanly spaced rows
    # =========================================================================
    # Row 1 (y in [0.752, 0.962]): schematic + ChR2 waveforms (a) and rasters + nerves (b, c, d)
    ax_schem = fig.add_axes([0.032, 0.864, 0.228, 0.098])
    ax_chr2  = fig.add_axes([0.072, 0.754, 0.178, 0.092])

    r1_x = [0.320, 0.525, 0.728]
    r1_w = 0.175
    ax_rasters, ax_nerves = [], []
    for x0 in r1_x:
        ax_rasters.append(fig.add_axes([x0, 0.818, r1_w, 0.136]))
        ax_nerves.append(fig.add_axes([x0, 0.754, r1_w, 0.055]))
    ax_rcbar = fig.add_axes([0.916, 0.818, 0.011, 0.136])

    # Row 2 (y in [0.536, 0.692]): 3 phase-space heatmaps (e, f, g) + shared colorbar
    ax_e = fig.add_axes([0.102, 0.536, 0.212, 0.154])
    ax_f = fig.add_axes([0.388, 0.536, 0.236, 0.154])
    ax_g = fig.add_axes([0.656, 0.536, 0.236, 0.154])
    ax_hcbar = fig.add_axes([0.908, 0.536, 0.012, 0.154])

    # Row 3 (y in [0.278, 0.432]): drive curves (h), Hopf boundary (i), Jacobian spectrum (j)
    # Shared 2x2 legend floats cleanly in the open band y in [0.448, 0.484]
    ax_h = fig.add_axes([0.076, 0.278, 0.246, 0.154])
    ax_i = fig.add_axes([0.402, 0.278, 0.246, 0.154])
    ax_j = fig.add_axes([0.728, 0.278, 0.222, 0.154])

    # Row 4 (y in [0.050, 0.202]): 500-shuffle distributions (k), linear-response scatter (l), predictor bar chart (m)
    ax_k = fig.add_axes([0.076, 0.050, 0.226, 0.152])
    ax_l = fig.add_axes([0.370, 0.050, 0.214, 0.152])
    ax_lcbar = fig.add_axes([0.592, 0.050, 0.010, 0.152])
    ax_m = fig.add_axes([0.812, 0.050, 0.158, 0.152])

    # =========================================================================
    # Panel a: Network schematic & normalized ChR2 photocurrent kinetics
    # =========================================================================
    draw_network_schematic(ax_schem)

    t_rel = np.arange(-80, 860) / 1000.0
    idx0, idx1 = proto.stim_on - 80, proto.stim_on + 860
    cfg_var = ChR2Config(i_max=1.0, desensitize=True)
    cfg_step = ChR2Config(i_max=1.0, desensitize=False)

    I_step   = chr2_current(proto.T, proto.stim_on, proto.stim_off, [1.0],  i_max=1.0, cfg=cfg_step)[0]
    I_unides = chr2_current(proto.T, proto.stim_on, proto.stim_off, [1.0],  i_max=1.0, cfg=cfg_var)[0]
    I_var_hi = chr2_current(proto.T, proto.stim_on, proto.stim_off, [0.75], i_max=1.0, cfg=cfg_var)[0]
    I_var_lo = chr2_current(proto.T, proto.stim_on, proto.stim_off, [0.30], i_max=1.0, cfg=cfg_var)[0]

    ax_chr2.axvspan(0.0, 0.85, color=COL_LASER, alpha=0.10, lw=0)
    ax_chr2.plot(t_rel, I_step[idx0:idx1],   color=COL_STEP,  lw=1.15, ls='--', label='No desens. (step)')
    ax_chr2.plot(t_rel, I_unides[idx0:idx1], color=COL_UNI_D, lw=1.15, label='Uniform ($I_{\\mathrm{ss}}=0.3$)')
    ax_chr2.plot(t_rel, I_var_hi[idx0:idx1], color=COL_VAR,   lw=1.15, label='Var. $\\varphi=0.75$')
    ax_chr2.plot(t_rel, I_var_lo[idx0:idx1], color=COL_VAR,   lw=0.95, alpha=0.48, label='Var. $\\varphi=0.30$')
    ax_chr2.set_xlim(-0.08, 0.85)
    ax_chr2.set_ylim(-0.04, 1.12)
    ax_chr2.set_xticks([0, 0.4, 0.8])
    ax_chr2.set_yticks([0, 0.5, 1.0])
    ax_chr2.set_xlabel('Time from laser on (s)', labelpad=1.8)
    ax_chr2.set_ylabel('ChR2 ($I / I_{\\max}$)', labelpad=2)
    ax_chr2.legend(loc='center right', bbox_to_anchor=(1.01, 0.62), frameon=False,
                   fontsize=5.0, handlelength=1.35, labelspacing=0.20, borderpad=0.1)

    # =========================================================================
    # Panels b, c, d: Representative rasters & antagonist motor nerve readouts
    # =========================================================================
    ex_R = data['ex_R']
    op_ex, phi_ex = data['op_ex'], data['phi_ex']
    order = data['ei_sort']
    inv_order = np.empty(net.N, dtype=int)
    inv_order[order] = np.arange(net.N)
    op_rows = inv_order[op_ex]

    ex_indices = [1, 3, 5]
    titles = [
        'Rest ($D = 0$) + 10% E laser',
        'Near threshold ($D = 12$) + laser',
        'Reference ($D = 20$, no laser)',
    ]
    has_laser = [True, True, False]

    nerves = [nerve_readout(net, ex_R[k], mode='limb', seed=0) for k in ex_indices]
    nerve_max = max(max(np.abs(nr['upper']).max(), np.abs(nr['lower']).max()) for nr in nerves)
    nerve_off = 1.25 * nerve_max

    t_end = proto.T / 1000.0
    im_raster = None
    for idx_p, (k, ttl, lit) in enumerate(zip(ex_indices, titles, has_laser)):
        axr, axn = ax_rasters[idx_p], ax_nerves[idx_p]
        R = ex_R[k]

        im_raster = axr.imshow(R[order], aspect='auto', vmin=0.0, vmax=50.0, cmap='viridis',
                               interpolation='none', extent=[0, t_end, net.N, 0])
        axr.axhline(net.N_exc, color='white', lw=0.75, alpha=0.9)
        if idx_p == 0:
            axr.text(0.22, 14, 'E', color='white', fontsize=5.8, fontweight='bold', va='top', ha='left')
            axr.text(0.22, net.N_exc + 14, 'I', color='white', fontsize=5.8, fontweight='bold', va='top', ha='left')
            axr.set_ylabel('Neuron (phase-sorted)', labelpad=2.5)
            axr.set_yticks([0, 100, 200])
        else:
            axr.set_yticks([0, 100, 200])
            axr.set_yticklabels([])

        axr.set_xticks([0, 3, 6, 9])
        axr.set_xticklabels([])
        axr.spines['top'].set_visible(True); axr.spines['right'].set_visible(True)
        axr.spines['top'].set_linewidth(0.55); axr.spines['right'].set_linewidth(0.55)

        if lit:
            axr.axvline(proto.stim_on / 1000.0, color='white', ls='--', lw=0.65, alpha=0.9)
            axr.axvline(proto.stim_off / 1000.0, color='white', ls='--', lw=0.65, alpha=0.9)
            axr.plot([proto.stim_on / 1000.0, proto.stim_off / 1000.0], [-8.0, -8.0],
                     color=COL_LASER, lw=2.3, solid_capstyle='butt', clip_on=False)
            colors_op = plt.cm.Reds(0.35 + 0.65 * phi_ex)
            axr.scatter(np.full(op_rows.size, -0.26), op_rows + 0.5, marker='>', s=8.0,
                        c=colors_op, edgecolors='#222222', linewidths=0.25, clip_on=False)

        axr.set_xlim(0, t_end)
        axr.set_ylim(net.N, 0)
        axr.set_title(ttl, fontsize=6.3, pad=7.0)

        sd_val = float(data['ex_osc_on'][k])
        lam_val = float(data['ex_lam_max'][k].real)
        badge_txt = f'SD = {sd_val:.2f} Hz | Re $\\lambda$ = {lam_val:.2f}'
        axr.text(0.96, 0.05, badge_txt, transform=axr.transAxes, ha='right', va='bottom',
                 fontsize=5.0, color='white',
                 bbox=dict(boxstyle='round,pad=0.17', fc='#000000', alpha=0.56, lw=0))

        nr = nerves[idx_p]
        if lit:
            axn.axvspan(proto.stim_on / 1000.0, proto.stim_off / 1000.0, color=COL_LASER, alpha=0.13, lw=0)
        axn.plot(proto.tvec, nr['lower'], color=COL_N2, lw=0.35, alpha=0.88)
        axn.plot(proto.tvec, nerve_off + nr['upper'], color=COL_N1, lw=0.35, alpha=0.88)
        axn.set_xlim(0, t_end)
        axn.set_ylim(-0.95 * nerve_off, 1.95 * nerve_off)
        axn.set_xticks([0, 3, 6, 9])
        axn.set_xlabel('Time (s)', labelpad=1.8)
        axn.set_yticks([0.0, nerve_off])
        if idx_p == 0:
            axn.set_yticklabels(['Ext.', 'Flex.'], fontsize=5.6)
            axn.set_ylabel('Nerve', labelpad=2.5)
        else:
            axn.set_yticklabels([])

    cb_r = fig.colorbar(im_raster, cax=ax_rcbar)
    cb_r.set_label('Firing rate (Hz)', fontsize=6.2, labelpad=2.8)
    cb_r.set_ticks([0, 25, 50])
    cb_r.ax.tick_params(labelsize=5.6, length=2, width=0.5)
    cb_r.outline.set_linewidth(0.5)

    # =========================================================================
    # Row 2: Panels e, f, g (Shared perceptual color scale)
    # =========================================================================
    h_norm = mcolors.PowerNorm(gamma=0.55, vmin=0.12, vmax=9.2)
    h_cmap = 'magma'

    e_mean = data['e_osc'].mean(-1)
    e_lam_max_all = data['e_lam'].mean(-1).max()
    im_e = ax_e.imshow(e_mean, origin='lower', aspect='auto', cmap=h_cmap, norm=h_norm, interpolation='nearest')
    amps_e = data['amps_e']
    des_tick_labels = ['0.1', '0.3', '0.5', '0.7', '1.0 (step)', 'var. $\\varphi$']
    ax_e.set_xticks(range(amps_e.size))
    ax_e.set_xticklabels([f'{int(a)}' for a in amps_e], fontsize=5.6)
    ax_e.set_yticks(range(len(des_tick_labels)))
    ax_e.set_yticklabels(des_tick_labels, fontsize=5.5)
    ax_e.set_xlabel('Peak photocurrent $I_{\\max}$', labelpad=2.2)
    ax_e.set_ylabel('Desensitization ($I_{\\mathrm{ss}}/I_{\\mathrm{peak}}$)', labelpad=2.2)
    ax_e.set_title('Rest ($D = 0$): no rhythm across all regimes', fontsize=6.3, pad=5.0)
    ax_e.spines['top'].set_visible(True); ax_e.spines['right'].set_visible(True)
    ax_e.spines['top'].set_linewidth(0.55); ax_e.spines['right'].set_linewidth(0.55)
    for r in range(e_mean.shape[0]):
        for c in range(e_mean.shape[1]):
            ax_e.text(c, r, f'{e_mean[r, c]:.2f}', ha='center', va='center',
                      fontsize=4.9, color='#F2E9FA')
    ax_e.text(0.50, -0.235, f'All regimes: SD $\\leq$ {e_mean.max():.2f} Hz, max Re $\\lambda$ = {e_lam_max_all:.2f} < 1',
              transform=ax_e.transAxes, ha='center', va='top', fontsize=5.0, color='#444444', style='italic')

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
        axh.set_xticklabels(x_labs_fg, fontsize=5.5)
        axh.set_yticks([0, 4, 8, 12, 16, 20])
        axh.set_xlabel('Peak photocurrent $I_{\\max}$', labelpad=2.2)
        axh.set_title(ttl, fontsize=6.3, pad=5.0)
        axh.spines['top'].set_visible(True); axh.spines['right'].set_visible(True)
        axh.spines['top'].set_linewidth(0.55); axh.spines['right'].set_linewidth(0.55)
        if idx_h == 0:
            axh.set_ylabel('Background tonic drive $D$', labelpad=2.2)
            axh.text(7.35, 14.0, '$D_{\\mathrm{th}}$', color='#80DEEA', fontsize=5.6,
                     ha='right', va='bottom', fontweight='bold')
            axh.text(4.1, 9.5, 'SD = 2 Hz', color='white', fontsize=5.0, ha='center', va='center')
        else:
            axh.set_yticklabels([])
            axh.text(7.35, 14.0, '$D_{\\mathrm{th}}$', color='#80DEEA', fontsize=5.6,
                     ha='right', va='bottom', fontweight='bold')
            axh.text(5.35, 17.5, 'Saturation\nsuppression', color='#E6DFF0', fontsize=5.1,
                     ha='center', va='center', style='italic')

    cb_h = fig.colorbar(im_e, cax=ax_hcbar)
    cb_h.set_label('Unlit rhythmicity (SD, Hz)', fontsize=6.2, labelpad=2.8)
    cb_h.set_ticks([0.2, 1, 2, 4, 6, 8.7])
    cb_h.set_ticklabels(['0.2', '1', '2', '4', '6', '8.7'])
    cb_h.ax.tick_params(labelsize=5.5, length=2, width=0.5)
    cb_h.outline.set_linewidth(0.5)
    cb_h.ax.axhline(2.0, color='white', ls='--', lw=0.9)

    # =========================================================================
    # Row 3: Panels h, i, j (Drive curves, Hopf boundary, Complex spectrum)
    # =========================================================================
    curve_defs = [
        ('ctrl',        data['ctrl_osc'],       data['ctrl_lam'],       COL_CTRL,  '-',  1.40, 'No laser (tonic $D$ only)'),
        ('var_des_40',  data['osc_var_des_40'], data['lam_var_des_40'], COL_VAR,   '-',  1.30, 'Var. light + desens. ($I_{\\max}=40$)'),
        ('uni_des_40',  data['osc_uni_des_40'], data['lam_uni_des_40'], COL_UNI_D, '-',  1.10, 'Uniform + desens. ($I_{\\max}=40$)'),
        ('uni_step_40', data['osc_uni_step_40'],data['lam_uni_step_40'],COL_STEP,  '--', 1.10, 'No desens. step ($I_{\\max}=40$)'),
    ]

    ax_h.axvspan(10.0, 13.5, color='#FDE8E8', alpha=0.80, lw=0, zorder=0)
    ax_h.axvline(13.5, color='#777777', ls=':', lw=0.75, zorder=1)
    ax_h.axhline(2.0, color='#999999', ls='--', lw=0.6, alpha=0.8, zorder=1)
    legend_handles = []
    for cname, osc_arr, _, col, ls, lw, lab in curve_defs:
        m, sd = osc_arr.mean(-1), osc_arr.std(-1)
        line, = ax_h.plot(drives, m, color=col, ls=ls, lw=lw, label=lab, zorder=3)
        ax_h.fill_between(drives, np.maximum(0.0, m - sd), m + sd, color=col, alpha=0.14, lw=0, zorder=2)
        legend_handles.append(line)

    ax_h.set_xlim(0, 20.4); ax_h.set_ylim(-0.2, 9.8)
    ax_h.set_xticks([0, 5, 10, 15, 20]); ax_h.set_yticks([0, 2, 4, 6, 8])
    ax_h.set_xlabel('Background tonic drive $D$', labelpad=2.2)
    ax_h.set_ylabel('Unlit rhythmicity (SD, Hz)', labelpad=2.2)
    ax_h.set_title('Near-threshold gating of rhythmicity', fontsize=6.3, pad=4.5)
    ax_h.text(11.75, 9.25, 'Near\nthresh.', ha='center', va='top', fontsize=5.1, color='#9E2A2B')
    ax_h.text(13.85, -0.05, '$D_{\\mathrm{th}}$', ha='left', va='bottom', fontsize=5.1, color='#555555')
    ax_h.annotate('$D=20$\nref.', xy=(19.95, 8.69), xytext=(16.1, 8.70),
                  ha='center', va='center', fontsize=5.2, color=COL_CTRL,
                  arrowprops=dict(arrowstyle='->', lw=0.65, color=COL_CTRL, shrinkB=2.5))

    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.365, 0.449),
               ncol=2, frameon=False, fontsize=5.4, columnspacing=1.35,
               handlelength=1.7, handletextpad=0.45, labelspacing=0.22)

    ax_i.axvspan(10.0, 13.5, color='#FDE8E8', alpha=0.80, lw=0, zorder=0)
    ax_i.axhspan(1.0, 1.18, color='#EAECEF', alpha=0.65, lw=0, zorder=0)
    ax_i.axhline(1.0, color='#222222', ls='--', lw=0.8, zorder=2)
    ax_i.axvline(13.5, color='#777777', ls=':', lw=0.75, zorder=1)
    for cname, _, lam_arr, col, ls, lw, lab in curve_defs:
        m, sd = lam_arr.mean(-1), lam_arr.std(-1)
        ax_i.plot(drives, m, color=col, ls=ls, lw=lw, zorder=3)
        ax_i.fill_between(drives, m - sd, m + sd, color=col, alpha=0.14, lw=0, zorder=2)

    ax_i.set_xlim(0, 20.4); ax_i.set_ylim(0.32, 1.16)
    ax_i.set_xticks([0, 5, 10, 15, 20]); ax_i.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax_i.set_xlabel('Background tonic drive $D$', labelpad=2.2)
    ax_i.set_ylabel('Leading Re $\\lambda_{\\max}$', labelpad=2.2)
    ax_i.set_title('Hopf stability boundary (Re $\\lambda = 1$)', fontsize=6.3, pad=4.5)
    ax_i.text(0.45, 1.022, 'Rhythmic (Re $\\lambda > 1$)', fontsize=5.0, color='#222222', va='bottom')
    ax_i.text(0.45, 0.974, 'Non-rhythmic (Re $\\lambda < 1$)', fontsize=5.0, color='#666666', va='top')
    ax_i.text(13.85, 0.355, '$D_{\\mathrm{th}}$', ha='left', va='bottom', fontsize=5.1, color='#555555')

    ex_eigs = data['ex_eigs']
    ax_j.axvspan(1.0, 1.45, color='#EAECEF', alpha=0.65, lw=0, zorder=0)
    ax_j.axhline(0, color='#CCCCCC', lw=0.5, zorder=1)
    ax_j.axvline(0, color='#CCCCCC', lw=0.5, zorder=1)
    ax_j.axvline(1.0, color='#222222', ls='--', lw=0.8, zorder=2)
    spec_defs = [
        (0, '#B0B5BD', 2.8, 0.70, 'o', '$D=0$ rest'),
        (1, COL_LASER, 4.0, 0.78, 'o', '$D=0$ + laser'),
        (3, COL_VAR,   5.0, 0.88, 'o', '$D=12$ + laser'),
        (5, COL_CTRL,  3.6, 0.68, 'o', '$D=20$ ref.'),
    ]
    for k_idx, col, sz, alp, mk, lab in spec_defs:
        ev = ex_eigs[k_idx]
        ax_j.scatter(ev.real, ev.imag, s=sz, c=col, alpha=alp, marker=mk, edgecolors='none',
                     label=lab, zorder=3 + (k_idx == 3))
        if k_idx in (1, 3, 5):
            lead_v = ev[int(ev.real.argmax())]
            ax_j.scatter([lead_v.real, lead_v.real], [lead_v.imag, -lead_v.imag],
                         s=16, facecolors='none', edgecolors=col, linewidths=0.9, zorder=6)

    ax_j.set_xlim(-1.25, 1.32); ax_j.set_ylim(-1.25, 1.25)
    ax_j.set_xticks([-1, 0, 1]); ax_j.set_yticks([-1, 0, 1])
    ax_j.set_aspect('equal', adjustable='box')
    ax_j.set_xlabel('Re $\\lambda$', labelpad=2.2)
    ax_j.set_ylabel('Im $\\lambda$', labelpad=2.2)
    ax_j.set_title("Jacobian spectrum $f$\u2032($V$)$\\,W$", fontsize=6.3, pad=4.5)
    ax_j.text(0.95, 1.15, 'Re $\\lambda = 1$', ha='right', va='top', fontsize=4.9, color='#222222')
    ax_j.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.88, edgecolor='none',
                fontsize=4.9, markerscale=1.35, handletextpad=0.22, labelspacing=0.20, borderpad=0.18)

    # =========================================================================
    # Row 4: Panels k, l, m — 500-Shuffle Variability & Mechanistic Explanation
    # =========================================================================
    # ---- Panel k: Distribution across 500 random shuffles of the 10 E cells
    rng_jit = np.random.default_rng(7)
    shuf_groups = [
        (shuf['osc_d0_var40'],  COL_LASER, '$D=0$\nvar. 40'),
        (shuf['osc_d0_uni80'],  '#5C85B8', '$D=0$\nuni. 80'),
        (shuf['osc_d12_var40'], COL_VAR,   '$D=12$\nvar. 40'),
        (shuf['osc_d12_uni40'], COL_UNI_D, '$D=12$\nuni. 40'),
    ]
    ax_k.axhline(2.0, color='#999999', ls='--', lw=0.6, alpha=0.85, zorder=1)
    ax_k.axhline(data['ctrl_osc'][-1].mean(), color=COL_CTRL, ls=':', lw=0.75, alpha=0.75, zorder=1)

    for xi, (vals, col, _) in enumerate(shuf_groups):
        jit = rng_jit.uniform(-0.22, 0.22, vals.size)
        ax_k.scatter(xi + jit, vals, s=2.4, color=col, alpha=0.38, edgecolors='none', zorder=2)
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        ax_k.plot([xi - 0.26, xi + 0.26], [q50, q50], color='#111111', lw=1.3, zorder=5)
        ax_k.plot([xi, xi], [q25, q75], color='#111111', lw=1.8, alpha=0.85, zorder=4)
        pct_hi = 100.0 * np.mean(vals >= 2.0)
        lbl_pct = f'{pct_hi:.1f}%' if (0 < pct_hi < 1) else f'{pct_hi:.0f}%'
        ax_k.text(xi, 9.85, lbl_pct, ha='center', va='bottom', fontsize=5.0, fontweight='bold', color=col)

    ax_k.set_xlim(-0.55, 3.55)
    ax_k.set_ylim(-0.3, 11.1)
    ax_k.set_xticks(range(len(shuf_groups)))
    ax_k.set_xticklabels([g[2] for g in shuf_groups], fontsize=5.3)
    ax_k.set_yticks([0, 2, 4, 6, 8, 10])
    ax_k.set_ylabel('Unlit rhythmicity (SD, Hz)', labelpad=2.2)
    ax_k.set_title('500 random 10-cell shuffles (% $\\geq$ 2 Hz)', fontsize=6.1, pad=4.5)

    # Clean variance-decomposition callout inside empty mid-left space of panel k
    vd = shuf['vd_mat']
    var_pct_sub = 100.0 * vd.mean(1).var() / vd.var()
    ax_k.text(0.03, 0.72,
              f'Variance at $D=12$:\n{var_pct_sub:.1f}% cell subset\n{100.0-var_pct_sub:.1f}% trial noise',
              transform=ax_k.transAxes, ha='left', va='top', fontsize=4.8, color='#222222',
              bbox=dict(boxstyle='round,pad=0.22', fc='#F5F6F8', ec='#CCCCCC', lw=0.5))
    ax_k.text(3.48, 8.92, '$D=20$', ha='right', va='bottom', fontsize=4.7, color=COL_CTRL)

    # ---- Panel l: Analytical zero-simulation linear-response prediction explains the 500 shuffles!
    x_pred = shuf['var_pred_lam']
    y_osc = shuf['osc_d12_var40']
    c_dV = shuf['var_pred_dV']
    rho_l, _ = spearmanr(x_pred, y_osc)
    r_l, _ = pearsonr(x_pred, y_osc)

    vmax_dv = float(np.percentile(np.abs(c_dV), 98))
    norm_dv = mcolors.TwoSlopeNorm(vmin=-vmax_dv, vcenter=0.0, vmax=vmax_dv)
    ax_l.axvline(1.0, color='#222222', ls='--', lw=0.75, zorder=1)
    ax_l.axhline(2.0, color='#999999', ls='--', lw=0.6, alpha=0.85, zorder=1)
    sc = ax_l.scatter(x_pred, y_osc, c=c_dV, cmap='coolwarm', norm=norm_dv,
                      s=6.0, alpha=0.82, edgecolors='#222222', linewidths=0.18, zorder=3)

    # Mark causal top-10 and bottom-10 g_eff subsets (uniform I=40) cleanly with diamonds
    ax_l.scatter([shuf['causal_pred_lam'][0]], [shuf['causal_osc'][2]], marker='D', s=22,
                 fc='#B8252A', ec='black', lw=0.6, zorder=6, label='Top-10 $g_{\\mathrm{eff}}$')
    ax_l.scatter([shuf['causal_pred_lam'][1]], [shuf['causal_osc'][3]], marker='D', s=22,
                 fc='#2166AC', ec='black', lw=0.6, zorder=6, label='Bot-10 $g_{\\mathrm{eff}}$')

    x_lo = min(0.74, float(shuf['causal_pred_lam'][1]) - 0.03)
    ax_l.set_xlim(x_lo, 1.13)
    ax_l.set_ylim(-0.2, 9.0)
    ax_l.set_xticks([0.8, 0.9, 1.0, 1.1])
    ax_l.set_yticks([0, 2, 4, 6, 8])
    ax_l.set_xlabel('Predicted Re $\\lambda_{\\max}^{\\mathrm{pred}}$ (from $W$ alone)', labelpad=2.2)
    ax_l.set_ylabel('Simulated rhythmicity (SD, Hz)', labelpad=2.2)
    ax_l.set_title('Linear-response prediction ($D = 12$, $n = 500$)', fontsize=6.1, pad=4.5)
    ax_l.text(0.04, 0.94, f'Spearman $\\rho = {rho_l:.2f}$\nPearson $r = {r_l:.2f}$',
              transform=ax_l.transAxes, ha='left', va='top', fontsize=5.0, fontweight='bold',
              color='#1A1A1A', bbox=dict(boxstyle='round,pad=0.20', fc='white', alpha=0.85, ec='none'))
    ax_l.legend(loc='upper left', bbox_to_anchor=(0.01, 0.69), frameon=False,
                fontsize=4.7, handletextpad=0.22, labelspacing=0.20)

    cb_l = fig.colorbar(sc, cax=ax_lcbar)
    cb_l.set_label('Net $\\langle\\Delta V\\rangle$', fontsize=5.5, labelpad=2.0)
    cb_l.ax.tick_params(labelsize=5.0, length=1.8, width=0.5)
    cb_l.outline.set_linewidth(0.5)

    # ---- Panel m: Ranked predictor comparison (why multi-synaptic amplification matters)
    pred_items = [
        ('Linear-resp. Re $\\lambda^{\\mathrm{pred}}$', shuf['var_pred_lam'], COL_VAR),
        ('Multi-syn. net $\\langle\\Delta V\\rangle$',   shuf['var_pred_dV'],  '#E07A5F'),
        ('1-hop E$-$I degree',                          shuf['var_pred_1hop'], '#7C8798'),
        ('Total current $\\Sigma I_{\\mathrm{ss}}$',    shuf['var_pred_totI'], '#AEB6C2'),
        ('Eigenmode $|v_i|$',                           shuf['var_pred_vmag'], '#AEB6C2'),
        ('Phase coherence $R$',                         shuf['var_pred_phaseR'], '#AEB6C2'),
    ]
    labels_m = [it[0] for it in pred_items][::-1]
    r2_vals = [pearsonr(it[1], y_osc)[0] ** 2 for it in pred_items][::-1]
    rho_vals = [abs(spearmanr(it[1], y_osc)[0]) for it in pred_items][::-1]
    cols_m = [it[2] for it in pred_items][::-1]

    y_pos = np.arange(len(labels_m))
    ax_m.barh(y_pos, r2_vals, height=0.62, color=cols_m, edgecolor='#222222', linewidth=0.35, zorder=3)
    for yi, (r2v, rhov) in enumerate(zip(r2_vals, rho_vals)):
        ax_m.text(r2v + 0.02, yi, f'{r2v:.2f} ($\\rho$={rhov:.2f})',
                  va='center', ha='left', fontsize=4.65, color='#222222')

    ax_m.set_yticks(y_pos)
    ax_m.set_yticklabels(labels_m, fontsize=5.1)
    ax_m.set_xlim(0, 1.04)
    ax_m.set_xticks([0, 0.3, 0.6])
    ax_m.set_xlabel('Variance explained ($R^2$)', labelpad=2.2)
    ax_m.set_title('What drives subset variability?', fontsize=6.1, pad=4.5)

    # =========================================================================
    # Bold lowercase panel letters a–m, aligned cleanly across all 4 rows
    # =========================================================================
    add_panel_label_xy(fig, 0.015, 0.968, 'a')
    add_panel_label_xy(fig, 0.278, 0.968, 'b')
    add_panel_label_xy(fig, 0.506, 0.968, 'c')
    add_panel_label_xy(fig, 0.712, 0.968, 'd')

    add_panel_label_xy(fig, 0.015, 0.702, 'e')
    add_panel_label_xy(fig, 0.342, 0.702, 'f')
    add_panel_label_xy(fig, 0.640, 0.702, 'g')

    add_panel_label_xy(fig, 0.015, 0.440, 'h')
    add_panel_label_xy(fig, 0.354, 0.440, 'i')
    add_panel_label_xy(fig, 0.684, 0.440, 'j')

    add_panel_label_xy(fig, 0.015, 0.212, 'k')
    add_panel_label_xy(fig, 0.324, 0.212, 'l')
    add_panel_label_xy(fig, 0.678, 0.212, 'm')

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, 'Figure1_opto_threshold.pdf')
    png_path = os.path.join(out_dir, 'Figure1_opto_threshold.png')
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    print(f'[plot] Saved {pdf_path} and {png_path}', flush=True)


# =============================================================================
# 5. Summary Report
# =============================================================================
def print_summary_stats(data: dict, shuf: dict, out_dir: str):
    drives = data['drives']
    e_mean = data['e_osc'].mean(-1)
    e_lam_mean = data['e_lam'].mean(-1)
    ctrl_m = data['ctrl_osc'].mean(-1)
    ctrl_sd = data['ctrl_osc'].std(-1)
    ctrl_lm = data['ctrl_lam'].mean(-1)

    vd = shuf['vd_mat']
    var_tot = vd.var()
    var_bet = vd.mean(1).var()
    var_wit = vd.var(1).mean()

    y_var = shuf['osc_d12_var40']
    y_uni = shuf['osc_d12_uni40']

    lines = [
        '=== BSG-opto Threshold & Shuffle-Variability Quantitative Summary ===',
        'Examples (draw s=19):',
        f'  D=0  no laser : on={data["ex_osc_on"][0]:.2f} Hz, Re(lam)={data["ex_lam_max"][0].real:.3f}',
        f'  D=0  I_max=80 : pre={data["ex_osc_pre"][1]:.2f}, on={data["ex_osc_on"][1]:.2f}, post={data["ex_osc_post"][1]:.2f} Hz, Re(lam)={data["ex_lam_max"][1].real:.3f}',
        f'  D=12 no laser : on={data["ex_osc_on"][2]:.2f} Hz, Re(lam)={data["ex_lam_max"][2].real:.3f}',
        f'  D=12 I_max=80 : pre={data["ex_osc_pre"][3]:.2f}, on={data["ex_osc_on"][3]:.2f}, post={data["ex_osc_post"][3]:.2f} Hz, Re(lam)={data["ex_lam_max"][3].real:.3f}',
        f'  D=12 I_max=40 : pre={data["ex_osc_pre"][4]:.2f}, on={data["ex_osc_on"][4]:.2f}, post={data["ex_osc_post"][4]:.2f} Hz, Re(lam)={data["ex_lam_max"][4].real:.3f}',
        f'  D=20 reference: on={data["ex_osc_on"][5]:.2f} Hz, Re(lam)={data["ex_lam_max"][5].real:.3f}',
        '',
        'Panel e (Rest D=0, 6 intensities x 6 desensitization regimes, n=30 draws):',
        f'  SD range across all 36 tiles: {e_mean.min():.2f} to {e_mean.max():.2f} Hz',
        f'  Re(lam) range across all 36 tiles: {e_lam_mean.min():.3f} to {e_lam_mean.max():.3f}',
        f'  10% mixed E+I (20/200 cells) at D=0, I=80: ss=0.3 -> {data["mixed_osc_03"].mean():.2f}+/-{data["mixed_osc_03"].std():.2f} Hz; '
        f'ss=1.0 -> {data["mixed_osc_10"].mean():.2f}+/-{data["mixed_osc_10"].std():.2f} Hz',
        '',
        'Panels h & i key points (n=30 draws, I_max=40):',
    ]
    for D in (0, 8, 10, 11, 12, 13, 14, 20):
        i = int(np.where(drives == D)[0][0])
        v40 = data['osc_var_des_40'][i]; l_v40 = data['lam_var_des_40'][i]
        u40 = data['osc_uni_des_40'][i]; s40 = data['osc_uni_step_40'][i]
        c0 = data['ctrl_osc'][i]
        succ40 = 100.0 * np.mean((v40 >= 2.0) & (v40 >= 2.0 * c0))
        lines.append(
            f'  D={D:2d}: ctrl={ctrl_m[i]:.2f}+/-{ctrl_sd[i]:.2f} (lam={ctrl_lm[i]:.2f}) | '
            f'var40={v40.mean():.2f}+/-{v40.std():.2f} (lam={l_v40.mean():.2f}, succ={succ40:.0f}%) | '
            f'uni40={u40.mean():.2f}+/-{u40.std():.2f} | step40={s40.mean():.2f}+/-{s40.std():.2f}'
        )
    mn = data['multi_net']
    lines += [
        '',
        'Multi-network check (10 independent connectivity seeds, 15 draws each, I_max=40 var_des):',
        f'  D=0  no laser: {mn[:,0].mean():.2f} +/- {mn[:,0].std():.2f} Hz',
        f'  D=0  10% E   : {mn[:,1].mean():.2f} +/- {mn[:,1].std():.2f} Hz (max network mean {mn[:,1].max():.2f} Hz)',
        f'  D=12 no laser: {mn[:,2].mean():.2f} +/- {mn[:,2].std():.2f} Hz',
        f'  D=12 10% E   : {mn[:,3].mean():.2f} +/- {mn[:,3].std():.2f} Hz (max network mean {mn[:,3].max():.2f} Hz)',
        f'  D=20 no laser: {mn[:,4].mean():.2f} +/- {mn[:,4].std():.2f} Hz',
        '',
        'Panels k-m (500 random shuffles of 10 E cells):',
        f'  D=0  var. I=40 : {shuf["osc_d0_var40"].mean():.2f}+/-{shuf["osc_d0_var40"].std():.2f} Hz '
        f'(range {shuf["osc_d0_var40"].min():.2f}-{shuf["osc_d0_var40"].max():.2f}, >=2Hz: {(shuf["osc_d0_var40"]>=2).sum()}/500, '
        f'max Re lam={shuf["lam_d0_var40"].max():.3f})',
        f'  D=0  uni. I=80 : {shuf["osc_d0_uni80"].mean():.2f}+/-{shuf["osc_d0_uni80"].std():.2f} Hz '
        f'(med {np.median(shuf["osc_d0_uni80"]):.2f}, >=2Hz: {(shuf["osc_d0_uni80"]>=2).sum()}/500)',
        f'  D=12 var. I=40 : {y_var.mean():.2f}+/-{y_var.std():.2f} Hz '
        f'(range {y_var.min():.2f}-{y_var.max():.2f}, med {np.median(y_var):.2f}, >=2Hz: {(y_var>=2).sum()}/500={100*(y_var>=2).mean():.1f}%)',
        f'  D=12 uni. I=40 : {y_uni.mean():.2f}+/-{y_uni.std():.2f} Hz '
        f'(range {y_uni.min():.2f}-{y_uni.max():.2f}, med {np.median(y_uni):.2f}, >=2Hz: {(y_uni>=2).sum()}/500={100*(y_uni>=2).mean():.1f}%)',
        f'  Variance decomposition (20 subsets x 25 noise seeds, D=12 uni I=40): '
        f'between-subset={100*var_bet/var_tot:.1f}%, within-subset noise={100*var_wit/var_tot:.1f}% '
        f'(mean within-subset SD={vd.std(1).mean():.2f} Hz, subset means {vd.mean(1).min():.2f}-{vd.mean(1).max():.2f} Hz)',
        f'  Per-E-cell multi-synaptic impact g_eff: mean={shuf["g_eff_E"].mean():.4f}, std={shuf["g_eff_E"].std():.4f}, '
        f'range=[{shuf["g_eff_E"].min():.4f}, {shuf["g_eff_E"].max():.4f}]',
        '  Predictors of D=12 var. I=40 (n=500):',
    ]
    for nm, key in [
        ('Linear-resp Re(lam)_pred', 'var_pred_lam'),
        ('Multi-synaptic net <dV>', 'var_pred_dV'),
        ('1-hop E-I out-degree', 'var_pred_1hop'),
        ('Total photocurrent sum(I_ss)', 'var_pred_totI'),
        ('Eigenmode amplitude |v_i|', 'var_pred_vmag'),
        ('Oscillatory phase coherence R', 'var_pred_phaseR'),
    ]:
        rp, _ = pearsonr(shuf[key], y_var)
        rs, _ = spearmanr(shuf[key], y_var)
        lines.append(f'    {nm:30s}: r={rp:+.3f}, R2={rp**2:.3f}, rho={rs:+.3f}')
    rp_u, _ = pearsonr(shuf['uni_pred_lam'], y_uni)
    rs_u, _ = spearmanr(shuf['uni_pred_lam'], y_uni)
    lines += [
        f'  Uniform D=12 I=40 Linear-resp Re(lam)_pred: r={rp_u:+.3f}, R2={rp_u**2:.3f}, rho={rs_u:+.3f}',
        f'  Causal top-10 vs bottom-10 g_eff (I=40): '
        f'D=0 top={shuf["causal_osc"][0]:.2f} Hz (lam={shuf["causal_lam"][0]:.3f}), bot={shuf["causal_osc"][1]:.2f} Hz; '
        f'D=12 top={shuf["causal_osc"][2]:.2f} Hz (lam={shuf["causal_lam"][2]:.3f}), bot={shuf["causal_osc"][3]:.2f} Hz (lam={shuf["causal_lam"][3]:.3f})',
    ]
    txt = '\n'.join(lines)
    print(txt, flush=True)
    with open(os.path.join(out_dir, 'simulation_stats.txt'), 'w') as fh:
        fh.write(txt + '\n')


def run(out_dir: str = DEFAULT_OUT, plot_only: bool = False, recompute: bool = False, **_):
    os.makedirs(out_dir, exist_ok=True)
    main_cache = os.path.join(out_dir, 'sim_cache.npz')
    shuf_cache = os.path.join(out_dir, 'shuffle_cache.npz')

    if plot_only or (os.path.exists(main_cache) and not recompute):
        print(f'[load] Loading main cache from {main_cache}', flush=True)
        npz = np.load(main_cache, allow_pickle=False)
        data = {k: npz[k] for k in npz.files}
    else:
        data = run_main_simulations(main_cache)

    if plot_only or (os.path.exists(shuf_cache) and not recompute):
        print(f'[load] Loading shuffle cache from {shuf_cache}', flush=True)
        npz = np.load(shuf_cache, allow_pickle=False)
        shuf = {k: npz[k] for k in npz.files}
    else:
        shuf = run_shuffle_simulations(shuf_cache)

    print_summary_stats(data, shuf, out_dir)
    build_figure(data, shuf, out_dir)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=DESCRIPTION)
    add_args(p)
    args = p.parse_args()
    run(**vars(args))
