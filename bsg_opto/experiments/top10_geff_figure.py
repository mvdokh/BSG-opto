"""Standalone figure inspecting the top-10 excitatory neurons ranked by
analytical multi-synaptic population impact g_eff.

Usage:
    python3 -m bsg_opto.experiments.top10_geff_figure
    .venv/bin/python run.py top10-geff
"""
from __future__ import annotations

import os
import sys
import time
import types
import numpy as np
from scipy.stats import pearsonr

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

from bsg_opto.config import ChR2Config, NetworkConfig, NeuronConfig, ProtocolConfig, RESULTS_DIR
from bsg_opto.network import BalancedNetwork
from bsg_opto.opsin import pick_random_exc
from bsg_opto.simulate import make_batched_ext, simulate_batch
from bsg_opto.analysis import leading_real_eig, nerve_readout, transfer_slope, unlit_mean
from bsg_opto.experiments.opto_threshold_figure import add_panel_label_xy, configure_publication_style

DESCRIPTION = 'Generate standalone figure characterizing the top-10 g_eff excitatory neurons'
DEFAULT_OUT = os.path.join(RESULTS_DIR, 'opto_threshold_figure')


def add_args(parser):
    parser.add_argument('--out-dir', type=str, default=DEFAULT_OUT, help='Output directory')
    parser.add_argument('--i-max', type=float, default=40.0, help='Peak photocurrent (default: 40)')


def run(out_dir: str = DEFAULT_OUT, i_max: float = 40.0, **_):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    net = BalancedNetwork(NetworkConfig())
    neuron = NeuronConfig()
    proto = ProtocolConfig()
    chr2 = ChR2Config(i_max=i_max, desensitize=True)
    W, N, N_exc = net.W, net.N, net.N_exc
    n_stim = 10

    # 1. Analytical multi-synaptic population impact g_eff at D = 12
    s0 = float(transfer_slope(12.0, neuron))
    L = np.linalg.inv(np.eye(N) - s0 * W)
    g_eff_E = (L.sum(0) - np.diag(L))[:N_exc] / (N - 1)
    rank_desc = np.argsort(g_eff_E)[::-1]
    top10 = rank_desc[:n_stim]       # ordered highest -> lowest g_eff
    mid80 = rank_desc[n_stim:-n_stim]
    bot10 = rank_desc[-n_stim:]

    A_bin = (W > 0).astype(float)
    k_to_E = A_bin[:N_exc, :N_exc].sum(0)
    k_to_I = A_bin[N_exc:, :N_exc].sum(0)
    k_diff = k_to_E - k_to_I
    phase_E = net.phase[:N_exc]

    # Cumulative hop-by-hop contribution sum_{m=1..H} (s0 W)^m averaged over unlit neurons
    max_hops = 12
    cum_hops = np.zeros((max_hops, N_exc))
    P = np.eye(N)
    S = np.zeros((N, N))
    for h in range(max_hops):
        P = P @ (s0 * W)
        S = S + P
        cum_hops[h] = (S.sum(0) - np.diag(S))[:N_exc] / (N - 1)

    print('=== Top-10 excitatory neurons by multi-synaptic impact g_eff ===')
    print('Rank | E ID |   g_eff  | k_E | k_I | k_E-k_I | phase (rad)')
    for rk, nid in enumerate(top10, start=1):
        print(f'  {rk:2d} | {nid:4d} | {g_eff_E[nid]:+.5f} | {int(k_to_E[nid]):3d} | '
              f'{int(k_to_I[nid]):3d} | {int(k_diff[nid]):+7d} | {phase_E[nid]:+.2f}')
    print(f'Top-10 mean g_eff = {g_eff_E[top10].mean():+.4f}, '
          f'Mid-80 = {g_eff_E[mid80].mean():+.4f}, Bot-10 = {g_eff_E[bot10].mean():+.4f}')

    # 2. Full-resolution rasters for Top-10 g_eff at D = 0, D = 6, D = 12 (and Bot-10 at D = 12)
    ex_specs = [
        ('D=0, Top-10 $g_{\\mathrm{eff}}$', 0.0, top10),
        ('D=6, Top-10 $g_{\\mathrm{eff}}$', 6.0, top10),
        ('D=12, Top-10 $g_{\\mathrm{eff}}$', 12.0, top10),
        ('D=12, Bot-10 $g_{\\mathrm{eff}}$', 12.0, bot10),
    ]
    ex_cols = [dict(base=D, op=op, phi=np.ones(n_stim), i_max=i_max) for _, D, op in ex_specs]
    out_ex = simulate_batch(
        W, proto.T, make_batched_ext(N, ex_cols, proto, chr2, light=True),
        len(ex_cols), neuron=neuron, seed=42,
        windows={'pre': proto.pre_win, 'on': proto.analysis_win, 'post': proto.post_win},
        snap_cols=tuple(range(len(ex_cols))), rec_every=1, dtype=np.float64,
    )
    ex_R = out_ex['snaps']
    ex_sd = np.array([unlit_mean(out_ex['on']['sd'][:, k], ex_specs[k][2]) for k in range(len(ex_specs))])
    ex_lam = np.array([leading_real_eig(W, out_ex['on']['Vmean'][:, k], neuron) for k in range(len(ex_specs))])
    for k, (lbl, _, _) in enumerate(ex_specs):
        print(f'  Example {lbl:28s}: SD = {ex_sd[k]:.2f} sp/s, Re(lam) = {ex_lam[k]:.3f}', flush=True)

    # 3. Full tonic drive sweep D = 0..20: Top-10 g_eff, Bot-10 g_eff, Random 10% E (n=30), No laser
    drives = np.arange(0, 21)
    n_rep_fixed = 10   # 10 noise seeds for top10, bot10, ctrl
    n_rand = 30        # 30 random 10-cell draws
    cols_sw, meta_sw = [], []
    for i, D in enumerate(drives):
        for r in range(n_rep_fixed):
            cols_sw.append(dict(base=float(D), op=None, phi=None))
            meta_sw.append(('ctrl', i, r, top10))
            cols_sw.append(dict(base=float(D), op=top10, phi=np.ones(n_stim), i_max=i_max))
            meta_sw.append(('top10', i, r, top10))
            cols_sw.append(dict(base=float(D), op=bot10, phi=np.ones(n_stim), i_max=i_max))
            meta_sw.append(('bot10', i, r, bot10))
        for s in range(n_rand):
            op_r = pick_random_exc(N_exc, n_stim, 30000 + s)
            cols_sw.append(dict(base=float(D), op=op_r, phi=np.ones(n_stim), i_max=i_max))
            meta_sw.append(('rand', i, s, op_r))

    out_sw = simulate_batch(
        W, proto.T, make_batched_ext(N, cols_sw, proto, chr2, light=True),
        len(cols_sw), neuron=neuron, seed=77,
        windows={'on': proto.analysis_win}, snap_cols=(), dtype=np.float32,
    )
    sw_osc = {
        'ctrl': np.zeros((drives.size, n_rep_fixed)),
        'top10': np.zeros((drives.size, n_rep_fixed)),
        'bot10': np.zeros((drives.size, n_rep_fixed)),
        'rand': np.zeros((drives.size, n_rand)),
    }
    sw_lam = {k: np.zeros((drives.size, 5)) for k in sw_osc}
    for c, (kind, i, rep, op) in enumerate(meta_sw):
        sw_osc[kind][i, rep] = unlit_mean(out_sw['on']['sd'][:, c], op)
        if rep < 5:
            sw_lam[kind][i, rep] = leading_real_eig(W, out_sw['on']['Vmean'][:, c], neuron)
    print(f'[sim] Drive sweep finished ({time.time()-t0:.1f}s)', flush=True)

    # =========================================================================
    # Plot 6-panel standalone publication figure (183 mm x 148 mm)
    # =========================================================================
    configure_publication_style()
    COL_TOP = '#B8252A'
    COL_BOT = '#2166AC'
    COL_MID = '#B6BDC8'
    COL_RAND = '#6A4C93'
    COL_CTRL = '#22252A'
    COL_LASER = '#1E88E5'
    COL_N1 = '#2B6CB0'
    COL_N2 = '#2F855A'

    fig = plt.figure(figsize=(7.20, 5.85))

    # Row 1: a (ranked g_eff bars), b (multi-hop cascade), c (1-hop vs g_eff + phase inset)
    ax_a = fig.add_axes([0.078, 0.595, 0.258, 0.325])
    ax_b = fig.add_axes([0.412, 0.595, 0.238, 0.325])
    ax_c = fig.add_axes([0.728, 0.595, 0.238, 0.325])

    # Row 2: d (D=0 Top-10 raster+nerve), e (D=12 Top-10 raster+nerve), f (D=0..20 sweep)
    ax_dr = fig.add_axes([0.078, 0.178, 0.212, 0.278])
    ax_dn = fig.add_axes([0.078, 0.080, 0.212, 0.085])
    ax_er = fig.add_axes([0.332, 0.178, 0.212, 0.278])
    ax_en = fig.add_axes([0.332, 0.080, 0.212, 0.085])
    ax_cb = fig.add_axes([0.556, 0.178, 0.011, 0.278])
    ax_f  = fig.add_axes([0.668, 0.095, 0.298, 0.360])

    # ---- Panel a: Ranked g_eff across all 100 E cells + exact Top-10 ID table
    g_sorted = g_eff_E[rank_desc]
    bar_cols = [COL_TOP] * n_stim + [COL_MID] * (N_exc - 2 * n_stim) + [COL_BOT] * n_stim
    ax_a.axhline(0.0, color='#333333', lw=0.7, zorder=1)
    ax_a.bar(np.arange(1, N_exc + 1), g_sorted, width=0.88, color=bar_cols, edgecolor='none', zorder=3)
    ax_a.set_xlim(0, N_exc + 1)
    ax_a.set_ylim(-0.085, 0.145)
    ax_a.set_xticks([1, 10, 50, 90, 100])
    ax_a.set_yticks([-0.05, 0.0, 0.05, 0.10])
    ax_a.set_xlabel('Excitatory neuron rank (1–100)', labelpad=2.2)
    ax_a.set_ylabel('Multi-synaptic impact $g_{\\mathrm{eff}}$', labelpad=2.2)
    ax_a.set_title('Ranked $g_{\\mathrm{eff}}$ across all 100 E cells', fontsize=6.6, pad=5)

    # Highlight Top-10 bracket and clean ID list box
    ax_a.annotate('Top-10 $g_{\\mathrm{eff}}$', xy=(5.5, 0.128), xytext=(24, 0.128),
                  fontsize=5.3, fontweight='bold', color=COL_TOP, va='center',
                  arrowprops=dict(arrowstyle='->', lw=0.75, color=COL_TOP))
    ax_a.annotate('Bot-10 $g_{\\mathrm{eff}}$', xy=(95.5, -0.068), xytext=(62, -0.072),
                  fontsize=5.3, fontweight='bold', color=COL_BOT, va='center',
                  arrowprops=dict(arrowstyle='->', lw=0.75, color=COL_BOT))
    id_line1 = ', '.join(str(int(x)) for x in top10[:5])
    id_line2 = ', '.join(str(int(x)) for x in top10[5:])
    ax_a.text(0.97, 0.78, f'Top-10 E neuron IDs:\n{id_line1},\n{id_line2}',
              transform=ax_a.transAxes, ha='right', va='top', fontsize=4.8, color='#222222',
              bbox=dict(boxstyle='round,pad=0.25', fc='#FDF3F3', ec='#E4B4B4', lw=0.55))

    # ---- Panel b: Multi-hop recurrent cascade (hops 1..12)
    hops = np.arange(1, max_hops + 1)
    ax_b.axhline(0.0, color='#444444', ls='--', lw=0.65, zorder=1)
    for grp, col, lab in [
        (top10, COL_TOP, 'Top-10 $g_{\\mathrm{eff}}$'),
        (mid80, '#6E7785', 'Middle 80 E'),
        (bot10, COL_BOT, 'Bot-10 $g_{\\mathrm{eff}}$'),
    ]:
        m = cum_hops[:, grp].mean(1)
        sd = cum_hops[:, grp].std(1)
        ax_b.plot(hops, m, color=col, lw=1.3, marker='o', ms=2.4, label=lab, zorder=3)
        ax_b.fill_between(hops, m - sd, m + sd, color=col, alpha=0.16, lw=0, zorder=2)

    ax_b.set_xlim(0.7, max_hops + 0.3)
    ax_b.set_ylim(-0.085, 0.145)
    ax_b.set_xticks([1, 3, 6, 9, 12])
    ax_b.set_yticks([-0.05, 0.0, 0.05, 0.10])
    ax_b.set_xlabel('Recurrent path depth (synaptic hops)', labelpad=2.2)
    ax_b.set_ylabel('Cumulative unlit $\\langle\\Delta V\\rangle$ / unit $I$', labelpad=2.2)
    ax_b.set_title('Multi-hop E/I cascade divergence', fontsize=6.6, pad=5)
    ax_b.legend(loc='upper left', frameon=False, fontsize=5.2, handlelength=1.4, labelspacing=0.22)

    # ---- Panel c: 1-hop E-I out-degree vs g_eff + polar phase inset
    r_1h, _ = pearsonr(k_diff, g_eff_E)
    ax_c.axhline(0.0, color='#CCCCCC', lw=0.6, zorder=1)
    ax_c.axvline(0.0, color='#CCCCCC', lw=0.6, zorder=1)
    ax_c.scatter(k_diff[mid80], g_eff_E[mid80], s=11, c=COL_MID, alpha=0.75, edgecolors='none', zorder=2)
    ax_c.scatter(k_diff[bot10], g_eff_E[bot10], s=18, c=COL_BOT, edgecolors='#111111', lw=0.35, zorder=4)
    ax_c.scatter(k_diff[top10], g_eff_E[top10], s=20, c=COL_TOP, edgecolors='#111111', lw=0.35, zorder=5)
    # Label key top-4 neuron IDs cleanly with non-overlapping offsets (including #94 with k_E - k_I = -2!)
    id_offsets = {
        int(top10[0]): (4.5, 1.0),    # #3
        int(top10[1]): (4.5, 3.5),    # #22
        int(top10[2]): (4.5, -5.5),   # #30
        int(top10[3]): (-6.0, -9.5),  # #94 (negative 1-hop yet rank #4!)
    }
    for nid, (ox, oy) in id_offsets.items():
        ax_c.annotate(f'#{nid}', xy=(k_diff[nid], g_eff_E[nid]),
                      xytext=(ox, oy), textcoords='offset points',
                      fontsize=4.7, color=COL_TOP, fontweight='bold')
    ax_c.set_xlim(-13.5, 13.5)
    ax_c.set_ylim(-0.085, 0.145)
    ax_c.set_xticks([-10, -5, 0, 5, 10])
    ax_c.set_yticks([-0.05, 0.0, 0.05, 0.10])
    ax_c.set_xlabel('1-hop E$-$I out-degree ($k_E - k_I$)', labelpad=2.2)
    ax_c.set_ylabel('Multi-synaptic impact $g_{\\mathrm{eff}}$', labelpad=2.2)
    ax_c.set_title('Local wiring vs. multi-synaptic impact', fontsize=6.6, pad=5)
    ax_c.text(0.04, 0.94, f'$r = {r_1h:.2f}$ ($R^2 = {r_1h**2:.2f}$)',
              transform=ax_c.transAxes, ha='left', va='top', fontsize=5.2, color='#222222')

    # Clean miniature polar inset in bottom-right showing Top-10 phases are spread across the cycle
    ax_pol = fig.add_axes([0.858, 0.618, 0.092, 0.112], projection='polar')
    ax_pol.scatter(phase_E[mid80], np.ones(mid80.size), s=4, c=COL_MID, alpha=0.55)
    ax_pol.scatter(phase_E[top10], np.ones(top10.size), s=14, c=COL_TOP, edgecolors='k', lw=0.25, zorder=4)
    ax_pol.set_ylim(0, 1.32)
    ax_pol.set_yticks([])
    ax_pol.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax_pol.set_xticklabels(['', '', '', ''])
    ax_pol.spines['polar'].set_linewidth(0.45)
    ax_pol.grid(linewidth=0.35, alpha=0.5)
    ax_pol.set_title('Phase', fontsize=4.8, pad=2)

    # ---- Panels d & e: Rasters + antagonist nerves for Top-10 g_eff at D=0 and D=12
    order = net.ei_phase_sort()
    inv_order = np.empty(N, dtype=int)
    inv_order[order] = np.arange(N)
    top_rows = inv_order[top10]
    t_end = proto.T / 1000.0

    nerves = [nerve_readout(net, ex_R[k], mode='limb', seed=0) for k in (0, 2)]
    nerve_off = 1.25 * max(max(np.abs(nr['upper']).max(), np.abs(nr['lower']).max()) for nr in nerves)

    im_r = None
    for idx_p, (k_ex, axr, axn, ttl) in enumerate([
        (0, ax_dr, ax_dn, 'Rest ($D = 0$) + Top-10 $g_{\\mathrm{eff}}$ ($I_{\\max}=40$)'),
        (2, ax_er, ax_en, 'Near threshold ($D = 12$) + Top-10 $g_{\\mathrm{eff}}$'),
    ]):
        R = ex_R[k_ex]
        im_r = axr.imshow(R[order], aspect='auto', vmin=0, vmax=50, cmap='viridis',
                          interpolation='none', extent=[0, t_end, N, 0])
        axr.axhline(N_exc, color='white', lw=0.75, alpha=0.9)
        axr.axvline(proto.stim_on / 1000.0, color='white', ls='--', lw=0.65, alpha=0.9)
        axr.axvline(proto.stim_off / 1000.0, color='white', ls='--', lw=0.65, alpha=0.9)
        axr.plot([proto.stim_on / 1000.0, proto.stim_off / 1000.0], [-7.5, -7.5],
                 color=COL_LASER, lw=2.3, solid_capstyle='butt', clip_on=False)
        axr.scatter(np.full(top_rows.size, -0.26), top_rows + 0.5, marker='>', s=9.5,
                    c=COL_TOP, edgecolors='#111111', linewidths=0.25, clip_on=False)
        axr.set_xlim(0, t_end); axr.set_ylim(N, 0)
        axr.set_xticks([0, 3, 6, 9]); axr.set_xticklabels([])
        axr.set_yticks([0, 100, 200])
        axr.spines['top'].set_visible(True); axr.spines['right'].set_visible(True)
        axr.spines['top'].set_linewidth(0.55); axr.spines['right'].set_linewidth(0.55)
        axr.set_title(ttl, fontsize=6.1, pad=7.0)
        if idx_p == 0:
            axr.text(0.22, 14, 'E', color='white', fontsize=5.8, fontweight='bold', va='top')
            axr.text(0.22, N_exc + 14, 'I', color='white', fontsize=5.8, fontweight='bold', va='top')
            axr.set_ylabel('Neuron (phase-sorted)', labelpad=2.5)
        else:
            axr.set_yticklabels([])

        axr.text(0.96, 0.05, f'SD = {ex_sd[k_ex]:.2f} sp/s | Re $\\lambda$ = {ex_lam[k_ex]:.2f}',
                 transform=axr.transAxes, ha='right', va='bottom', fontsize=5.0, color='white',
                 bbox=dict(boxstyle='round,pad=0.17', fc='#000000', alpha=0.58, lw=0))

        nr = nerves[idx_p]
        axn.axvspan(proto.stim_on / 1000.0, proto.stim_off / 1000.0, color=COL_LASER, alpha=0.13, lw=0)
        axn.plot(proto.tvec, nr['lower'], color=COL_N2, lw=0.35, alpha=0.88)
        axn.plot(proto.tvec, nerve_off + nr['upper'], color=COL_N1, lw=0.35, alpha=0.88)
        axn.set_xlim(0, t_end); axn.set_ylim(-0.95 * nerve_off, 1.95 * nerve_off)
        axn.set_xticks([0, 3, 6, 9]); axn.set_xlabel('Time (s)', labelpad=2.0)
        axn.set_yticks([0.0, nerve_off])
        if idx_p == 0:
            axn.set_yticklabels(['Ext.', 'Flex.'], fontsize=5.6)
            axn.set_ylabel('Nerve', labelpad=2.5)
        else:
            axn.set_yticklabels([])

    cb = fig.colorbar(im_r, cax=ax_cb)
    cb.set_label('Firing rate (sp/s)', fontsize=6.0, labelpad=2.5)
    cb.set_ticks([0, 25, 50])
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)

    # ---- Panel f: Full drive sweep D = 0..20 comparing Top-10, Random 10%, Bot-10, and No laser
    ax_f.axhline(2.0, color='#999999', ls='--', lw=0.6, alpha=0.85, zorder=1)
    ax_f.axvline(13.5, color='#777777', ls=':', lw=0.75, zorder=1)
    for key, col, ls, lw, lab in [
        ('top10', COL_TOP,  '-',  1.45, 'Top-10 $g_{\\mathrm{eff}}$ ($I_{\\max}=40$)'),
        ('rand',  COL_RAND, '-',  1.15, 'Random 10% E ($n=30$)'),
        ('ctrl',  COL_CTRL, '--', 1.25, 'No laser (tonic $D$ only)'),
        ('bot10', COL_BOT,  '-',  1.40, 'Bot-10 $g_{\\mathrm{eff}}$ ($I_{\\max}=40$)'),
    ]:
        m = sw_osc[key].mean(1)
        sd = sw_osc[key].std(1)
        ax_f.plot(drives, m, color=col, ls=ls, lw=lw, label=lab, zorder=3)
        ax_f.fill_between(drives, np.maximum(0.0, m - sd), m + sd, color=col, alpha=0.14, lw=0, zorder=2)

    ax_f.set_xlim(0, 20.4)
    ax_f.set_ylim(-0.2, 9.6)
    ax_f.set_xticks([0, 5, 10, 15, 20])
    ax_f.set_yticks([0, 2, 4, 6, 8])
    ax_f.set_xlabel('Background tonic drive $D$', labelpad=2.2)
    ax_f.set_ylabel('Unlit amplitude (rate SD, sp/s)', labelpad=2.2)
    ax_f.set_title('Top-10 vs. Bot-10 $g_{\\mathrm{eff}}$ across drive $D$', fontsize=6.5, pad=5)
    ax_f.text(13.85, 0.72, '$D_{\\mathrm{th}}$', fontsize=5.3, color='#555555')
    ax_f.legend(loc='upper left', frameon=False, fontsize=5.2, handlelength=1.5, labelspacing=0.25)

    # Panel letters a-f
    add_panel_label_xy(fig, 0.015, 0.942, 'a')
    add_panel_label_xy(fig, 0.354, 0.942, 'b')
    add_panel_label_xy(fig, 0.672, 0.942, 'c')
    add_panel_label_xy(fig, 0.015, 0.480, 'd')
    add_panel_label_xy(fig, 0.312, 0.480, 'e')
    add_panel_label_xy(fig, 0.612, 0.480, 'f')

    pdf_path = os.path.join(out_dir, 'Figure2_top10_geff.pdf')
    png_path = os.path.join(out_dir, 'Figure2_top10_geff.png')
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    print(f'[plot] Saved {pdf_path} and {png_path}', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=DESCRIPTION)
    add_args(p)
    run(**vars(p.parse_args()))
