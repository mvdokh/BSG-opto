"""Full tonic-drive sweep figure (D = 0..20, 0-10 s) matching Figure 1 panels b-d:
Panel a: No laser (global tonic drive D = 0..20 only).
Panel b: 10% E-cell ChR2 laser (1-6 s, variable light, I_max = 80) across D = 0..20.

Usage:
    python3 -m bsg_opto.experiments.drive_sweep_rasters
    .venv/bin/python run.py drive-sweep-rasters
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

from bsg_opto.config import ChR2Config, NetworkConfig, NeuronConfig, ProtocolConfig, RESULTS_DIR
from bsg_opto.network import BalancedNetwork
from bsg_opto.opsin import draw_variable_light
from bsg_opto.simulate import make_batched_ext, simulate_batch
from bsg_opto.analysis import leading_real_eig, nerve_readout, unlit_mean
from bsg_opto.experiments.opto_threshold_figure import add_panel_label_xy, configure_publication_style

DESCRIPTION = 'Generate full D = 0..20 (0-10 s) raster + nerve panels with and without 10% E laser'
DEFAULT_OUT = os.path.join(RESULTS_DIR, 'opto_threshold_figure')


def add_args(parser):
    parser.add_argument('--out-dir', type=str, default=DEFAULT_OUT, help='Output directory')
    parser.add_argument('--i-max', type=float, default=80.0, help='Peak photocurrent (default: 80)')
    parser.add_argument('--draw', type=int, default=19, help='Opsin draw index matching Fig. 1 b-d (default: 19)')


def run(out_dir: str = DEFAULT_OUT, i_max: float = 80.0, draw: int = 19, **_):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    net = BalancedNetwork(NetworkConfig())
    neuron = NeuronConfig()
    # 0-10 s simulation (T = 10000 ms), laser ON 1-6 s, analysis window 2-6 s
    proto = ProtocolConfig(T=10000, stim_on=1000, stim_off=6000)
    chr2 = ChR2Config(i_max=i_max, desensitize=True)
    W, N, N_exc = net.W, net.N, net.N_exc

    n_stim = 10  # 10% of excitatory cells
    op_ex, phi_ex = draw_variable_light(N_exc, n_stim, draw)
    drives = np.arange(0, 21)

    # Batch all 42 trials (21 no-laser + 21 with 10% E laser) at full 1-ms resolution
    cols = []
    for D in drives:
        cols.append(dict(base=float(D), op=None, phi=None))
    for D in drives:
        cols.append(dict(base=float(D), op=op_ex, phi=phi_ex, i_max=i_max))

    out = simulate_batch(
        W, proto.T, make_batched_ext(N, cols, proto, chr2, light=True),
        len(cols), neuron=neuron, seed=100,
        windows={'on': proto.analysis_win},
        snap_cols=tuple(range(len(cols))), rec_every=1, dtype=np.float64,
    )
    snaps = out['snaps']  # (42, N, 10000)
    R_ctrl = snaps[:21]
    R_opto = snaps[21:]

    sd_ctrl = np.array([unlit_mean(out['on']['sd'][:, i], op_ex) for i in range(21)])
    sd_opto = np.array([unlit_mean(out['on']['sd'][:, 21 + i], op_ex) for i in range(21)])
    lam_ctrl = np.array([leading_real_eig(W, out['on']['Vmean'][:, i], neuron) for i in range(21)])
    lam_opto = np.array([leading_real_eig(W, out['on']['Vmean'][:, 21 + i], neuron) for i in range(21)])

    print(f'[sim] 42 trials x 10 s completed in {time.time()-t0:.1f}s', flush=True)
    for i, D in enumerate(drives):
        print(f'  D={D:2d} | no-laser: SD={sd_ctrl[i]:4.2f} sp/s, Re(lam)={lam_ctrl[i]:.2f} | '
              f'+laser: SD={sd_opto[i]:4.2f} sp/s, Re(lam)={lam_opto[i]:.2f}')

    # Precompute antagonist nerve readouts on a single shared vertical scale across all 42 panels
    nerves_ctrl = [nerve_readout(net, R_ctrl[i], mode='limb', seed=0) for i in range(21)]
    nerves_opto = [nerve_readout(net, R_opto[i], mode='limb', seed=0) for i in range(21)]
    all_nerves = nerves_ctrl + nerves_opto
    nerve_max = max(max(np.abs(nr['upper']).max(), np.abs(nr['lower']).max()) for nr in all_nerves)
    nerve_off = 1.25 * nerve_max

    # =========================================================================
    # Plot crisp 2-panel (a: No laser, b: + 10% E laser) x (3 rows x 7 cols) figure
    # =========================================================================
    configure_publication_style()
    COL_LASER = '#1E88E5'
    COL_N1 = '#2B6CB0'
    COL_N2 = '#2F855A'

    order = net.ei_phase_sort()
    inv_order = np.empty(N, dtype=int)
    inv_order[order] = np.arange(N)
    op_rows = inv_order[op_ex]
    colors_op = plt.cm.Reds(0.35 + 0.65 * phi_ex)
    t_end = proto.T / 1000.0  # 10.0 s

    fig = plt.figure(figsize=(13.6, 12.2))

    # Shared horizontal layout across 7 columns (leaving clean right margin for colorbar)
    n_rows_blk, n_cols = 3, 7
    x_left, x_right = 0.048, 0.940
    col_gap = 0.0155
    col_w = (x_right - x_left - (n_cols - 1) * col_gap) / n_cols

    # Vertical layout: Panel a (top half: y in [0.532, 0.945]), Panel b (bottom half: y in [0.042, 0.455])
    # Each block has 3 paired (raster + nerve) rows cleanly spaced
    def _block_row_boxes(y_bot: float, y_top: float):
        row_gap = 0.032
        row_h = (y_top - y_bot - (n_rows_blk - 1) * row_gap) / n_rows_blk
        r_frac = 0.67
        inner_gap = 0.0045
        n_h = row_h * (1.0 - r_frac) - inner_gap
        r_h = row_h * r_frac
        boxes = []
        for r in range(n_rows_blk):
            ry_top = y_top - r * (row_h + row_gap)
            ry_rast = ry_top - r_h
            ry_nerv = ry_rast - inner_gap - n_h
            boxes.append((ry_rast, r_h, ry_nerv, n_h))
        return boxes

    blocks = [
        ('a', 'No laser (global tonic drive $D = 0\\text{–}20$ only, 0–10 s)',
         False, R_ctrl, nerves_ctrl, sd_ctrl, lam_ctrl, 0.536, 0.938, 0.962),
        ('b', f'10% E-cell ChR2 laser ($1\\text{{–}}6$ s, variable light, $I_{{\\max}} = {i_max:g}$) across tonic drive $D = 0\\text{{–}}20$',
         True, R_opto, nerves_opto, sd_opto, lam_opto, 0.044, 0.446, 0.472),
    ]

    im_ref = None
    for letter, header, lit, R_arr, N_arr, sd_arr, lam_arr, y_bot, y_top, y_hdr in blocks:
        # Bold panel letter & clear section header
        fig.text(0.014, y_hdr, letter, fontsize=12.0, fontweight='bold', va='bottom', ha='left', color='#000000')
        fig.text(0.048, y_hdr + 0.002, header, fontsize=9.2, fontweight='bold', va='bottom', ha='left', color='#111111')

        row_boxes = _block_row_boxes(y_bot, y_top)
        for idx, D in enumerate(drives):
            r = idx // n_cols
            c = idx % n_cols
            x0 = x_left + c * (col_w + col_gap)
            ry_rast, r_h, ry_nerv, n_h = row_boxes[r]

            axr = fig.add_axes([x0, ry_rast, col_w, r_h])
            axn = fig.add_axes([x0, ry_nerv, col_w, n_h])

            # Downsample display raster slightly in time (every 5 ms -> 2000 px wide) for lighter crisp PDF
            im_ref = axr.imshow(R_arr[idx][order, ::5], aspect='auto', vmin=0.0, vmax=50.0, cmap='viridis',
                                interpolation='none', extent=[0, t_end, N, 0])
            axr.axhline(N_exc, color='white', lw=0.65, alpha=0.9)
            axr.set_xlim(0, t_end)
            axr.set_ylim(N, 0)
            axr.set_xticks([0, 2, 4, 6, 8, 10])
            axr.set_xticklabels([])
            axr.set_yticks([0, 100, 200])
            axr.tick_params(labelsize=5.8, length=2.0, width=0.55)
            axr.spines['top'].set_visible(True); axr.spines['right'].set_visible(True)
            axr.spines['top'].set_linewidth(0.5); axr.spines['right'].set_linewidth(0.5)

            if c == 0:
                axr.set_ylabel('Neuron', fontsize=6.4, labelpad=2.0)
                if r == 0:
                    axr.text(0.28, 16, 'E', color='white', fontsize=5.8, fontweight='bold', va='top')
                    axr.text(0.28, N_exc + 16, 'I', color='white', fontsize=5.8, fontweight='bold', va='top')
            else:
                axr.set_yticklabels([])

            if lit:
                axr.axvline(proto.stim_on / 1000.0, color='white', ls='--', lw=0.6, alpha=0.9)
                axr.axvline(proto.stim_off / 1000.0, color='white', ls='--', lw=0.6, alpha=0.9)
                axr.plot([proto.stim_on / 1000.0, proto.stim_off / 1000.0], [-8.5, -8.5],
                         color=COL_LASER, lw=2.1, solid_capstyle='butt', clip_on=False)
                axr.scatter(np.full(op_rows.size, -0.25), op_rows + 0.5, marker='>', s=6.5,
                            c=colors_op, edgecolors='#222222', linewidths=0.22, clip_on=False)
                axr.set_xlim(0, t_end)
                axr.set_ylim(N, 0)

            axr.set_title(f'$D = {D}$', fontsize=7.0, fontweight='bold', pad=6.0 if lit else 3.0)

            axr.text(0.96, 0.05, f'SD = {sd_arr[idx]:.2f} | Re $\\lambda$ = {lam_arr[idx]:.2f}',
                     transform=axr.transAxes, ha='right', va='bottom', fontsize=4.9, color='white',
                     bbox=dict(boxstyle='round,pad=0.16', fc='#000000', alpha=0.58, lw=0))

            # Antagonist motor-nerve readout below raster
            nr = N_arr[idx]
            if lit:
                axn.axvspan(proto.stim_on / 1000.0, proto.stim_off / 1000.0, color=COL_LASER, alpha=0.14, lw=0)
            axn.plot(proto.tvec, nr['lower'], color=COL_N2, lw=0.32, alpha=0.88)
            axn.plot(proto.tvec, nerve_off + nr['upper'], color=COL_N1, lw=0.32, alpha=0.88)
            axn.set_xlim(0, t_end)
            axn.set_ylim(-0.95 * nerve_off, 1.95 * nerve_off)
            axn.set_xticks([0, 2, 4, 6, 8, 10])
            axn.set_yticks([0.0, nerve_off])
            axn.tick_params(labelsize=5.8, length=2.0, width=0.55)
            if r == n_rows_blk - 1:
                axn.set_xlabel('Time (s)', fontsize=6.4, labelpad=1.8)
            else:
                axn.set_xticklabels([])
            if c == 0:
                axn.set_yticklabels(['Ext.', 'Flex.'], fontsize=5.5)
                axn.set_ylabel('Nerve', fontsize=6.2, labelpad=2.0)
            else:
                axn.set_yticklabels([])

        # Dedicated vertical colorbar on the right edge of each block (aligned to block height)
        ax_cb = fig.add_axes([0.950, y_bot + 0.06, 0.010, (y_top - y_bot) - 0.12])
        cb = fig.colorbar(im_ref, cax=ax_cb)
        cb.set_label('Firing rate (sp/s)', fontsize=6.8, labelpad=3.0)
        cb.set_ticks([0, 25, 50])
        cb.ax.tick_params(labelsize=6.0, length=2.2, width=0.55)
        cb.outline.set_linewidth(0.5)

    # Subtle horizontal separator line cleanly dividing Panel a and Panel b
    sep_line = plt.Line2D([0.014, 0.980], [0.502, 0.502], transform=fig.transFigure,
                          color='#CCCCCC', lw=0.8, ls='-')
    fig.add_artist(sep_line)

    pdf_path = os.path.join(out_dir, 'Figure3_drive_sweep_0_20.pdf')
    png_path = os.path.join(out_dir, 'Figure3_drive_sweep_0_20.png')
    fig.savefig(pdf_path, dpi=350)
    fig.savefig(png_path, dpi=350)
    plt.close(fig)
    print(f'[plot] Saved {pdf_path} and {png_path} ({time.time()-t0:.1f}s total)', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=DESCRIPTION)
    add_args(p)
    run(**vars(p.parse_args()))
