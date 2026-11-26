"""Experiment 6: Stronger synaptic weights from opsin E cells onto inhibitory cells."""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from ..config import ChR2Config, NetworkConfig, NeuronConfig, ProtocolConfig, RESULTS_DIR
from ..network import BalancedNetwork
from ..opsin import draw_variable_light
from ..connectivity import make_batched_opsin_to_I_extra, scale_opsin_to_I_weights
from ..simulate import make_batched_ext, simulate_batch
from ..analysis import leading_real_eig, unlit_mean
from ..plotting import plot_raster, save_fig, shade_light

DESCRIPTION = 'Scale opsin E -> I synaptic weights by k = 1, 2, 4, 8, 16 across tonic drive D = 1..15'


def add_args(parser):
    parser.add_argument('--N', type=int, default=200, help='Total neurons (default: 200)')
    parser.add_argument('--pct', type=float, default=10.0, help='Percent of E cells with opsin (default: 10)')
    parser.add_argument('--d-min', type=int, default=1)
    parser.add_argument('--d-max', type=int, default=15)
    parser.add_argument('--i-max', type=float, default=80.0)
    parser.add_argument('--n-draws', type=int, default=20)
    parser.add_argument('--out-dir', type=str, default=os.path.join(RESULTS_DIR, '07_opsin_to_i_strength'))


def run(N: int = 200, pct: float = 10.0, d_min: int = 1, d_max: int = 15,
        i_max: float = 80.0, n_draws: int = 20,
        out_dir: str = os.path.join(RESULTS_DIR, '07_opsin_to_i_strength'), **_):
    out_sub = os.path.join(out_dir, f'results_N{N}')
    os.makedirs(out_sub, exist_ok=True)
    net = BalancedNetwork(NetworkConfig(N=N))
    neuron, proto, chr2 = NeuronConfig(), ProtocolConfig(), ChR2Config(i_max=i_max)
    n_op = int(round(pct / 100.0 * net.N_exc))
    drives = np.arange(d_min, d_max + 1)
    ks = np.array([1, 2, 4, 8, 16])
    n_lam = 3
    ex_D = [d for d in (1, 5, 10, 13, 15) if d_min <= d <= d_max]
    ex_K = [1, 4, 16]

    keys = ('osc', 'post', 'lam', 'rate_lit', 'rate_unlitE', 'rate_I')
    S = {(int(k), light): {q: np.full((drives.size, n_draws), np.nan) for q in keys}
         for k in ks for light in (True, False)}
    EX = {}
    op0, _ = draw_variable_light(net.N_exc, n_op, 0)
    unlit0 = np.setdiff1d(np.arange(net.N_exc), op0)
    for k in ks:
        for light in (True, False):
            t0 = time.time()
            cols, meta, ops_list = [], [], []
            for i, D in enumerate(drives):
                for s in range(n_draws):
                    op, phi = draw_variable_light(net.N_exc, n_op, s)
                    cols.append(dict(base=float(D), op=op, phi=phi, i_max=i_max))
                    ops_list.append(op)
                    meta.append((i, s, op))
            snap_cols, record = [], None
            if light and int(k) in ex_K:
                snap_cols = [j for j, (i, s, _) in enumerate(meta) if s == 0 and drives[i] in ex_D]
                record = {'m': lambda R, V, sc=snap_cols: np.concatenate([
                    R[op0][:, sc].mean(0), R[unlit0][:, sc].mean(0), R[net.N_exc:][:, sc].mean(0)
                ])}
            out = simulate_batch(
                net.W, proto.T, make_batched_ext(net.N, cols, proto, chr2, light=light), len(cols),
                neuron=neuron, seed=int(100 * k + light),
                windows={'on': proto.analysis_win, 'post': proto.post_win},
                snap_cols=snap_cols, record=record,
                extra_input=make_batched_opsin_to_I_extra(net.W, ops_list, net.N_exc, float(k)),
            )
            r = S[(int(k), light)]
            for j, (i, s, op) in enumerate(meta):
                unlitE = np.setdiff1d(np.arange(net.N_exc), op)
                r['osc'][i, s] = unlit_mean(out['on']['sd'][:, j], op)
                r['post'][i, s] = unlit_mean(out['post']['sd'][:, j], op)
                r['rate_lit'][i, s] = out['on']['mean'][op, j].mean()
                r['rate_unlitE'][i, s] = out['on']['mean'][unlitE, j].mean()
                r['rate_I'][i, s] = out['on']['mean'][net.N_exc:, j].mean()
                if s < n_lam:
                    Wk = scale_opsin_to_I_weights(net.W, op, net.N_exc, float(k))
                    r['lam'][i, s] = leading_real_eig(Wk, out['on']['Vmean'][:, j], neuron)
            if snap_cols:
                EX[int(k)] = (out['snaps'], out['rec']['m'])
            print(f'N={N} k={k} light={light}: {time.time() - t0:.0f}s', flush=True)

    ref_cols = [dict(base=20.0, op=draw_variable_light(net.N_exc, n_op, s)[0], phi=None) for s in range(5)]
    out_ref = simulate_batch(net.W, proto.T, make_batched_ext(net.N, ref_cols, proto, chr2, light=False), 5,
                             neuron=neuron, seed=7, windows={'on': proto.analysis_win}, snap_cols=[])
    osc_ref = np.mean([unlit_mean(out_ref['on']['sd'][:, j], ref_cols[j]['op']) for j in range(5)])

    # Figure 1: examples
    fig = plt.figure(figsize=(4 * len(ex_D), 4.6 * len(ex_K)))
    gs = fig.add_gridspec(2 * len(ex_K), len(ex_D), height_ratios=[1, 0.55] * len(ex_K), hspace=0.55, wspace=0.25)
    nE = len(ex_D)
    for a_, k in enumerate(ex_K):
        snaps, m = EX[k]
        for c, D in enumerate(ex_D):
            ax = fig.add_subplot(gs[2 * a_, c])
            plot_raster(ax, snaps[c], net.phase_sort, proto, vmax=50.0, interpolation='antialiased')
            ax.set_title(f'opsin→I ×{k},  D = {D}', fontsize=11, fontweight='bold')
            if c == 0:
                ax.set_ylabel('Neuron (phase-sorted)')
            ax = fig.add_subplot(gs[2 * a_ + 1, c]); shade_light(ax, proto)
            ax.plot(proto.tvec, m[:, c], c='#ff4f4f', lw=1, label='lit E')
            ax.plot(proto.tvec, m[:, nE + c], c='#e09b22', lw=1, label='unlit E')
            ax.plot(proto.tvec, m[:, 2 * nE + c], c='#3a6fd8', lw=1, label='I')
            ax.set_xlim(0, proto.T / 1000.0); ax.set_ylim(0, 60); ax.set_xlabel('Time (s)')
            if c == 0:
                ax.set_ylabel('Mean rate (Hz)')
            if a_ == 0 and c == 0:
                ax.legend(fontsize=7, frameon=False, ncol=3, loc='upper right')
    fig.suptitle(f'N = {N}: ChR2 in random {pct:g} % of E ({n_op}), random light (max I = {i_max:g}), '
                 f'opsin→I synapses ×k, tonic drive D, light 1-6 s', fontsize=12, y=0.995)
    save_fig(fig, os.path.join(out_sub, f'opsinI_examples_N{N}'), dpi=120)

    # Figure 2: summary
    cm = plt.cm.plasma(np.linspace(0, 0.85, ks.size))
    fig, axs = plt.subplots(1, 5, figsize=(25, 4.4))
    ax = axs[0]
    for c, k in zip(cm, ks):
        v = S[(int(k), True)]['osc']
        ax.errorbar(drives, v.mean(1), v.std(1), c=c, marker='o', ms=4, capsize=2, label=f'×{k}, light')
        ax.plot(drives, S[(int(k), False)]['osc'].mean(1), c=c, ls=':', lw=1.2)
    ax.plot([], [], c='k', ls=':', label='dotted: same k, no light')
    ax.axhline(osc_ref, c='k', ls='--', lw=0.8, label=f'reference D = 20 ({osc_ref:.1f})')
    ax.set_yscale('log'); ax.set_xlabel('Tonic drive D'); ax.set_ylabel('Rhythmicity of unlit cells (SD, Hz)')
    ax.set_title(f'N = {N}, {n_draws} draws per point', fontsize=10); ax.legend(fontsize=7, frameon=False, ncol=2)

    for ax, q, ttl in [(axs[1], 'rate_unlitE', 'Unlit E rate during light'),
                       (axs[2], 'rate_I', 'I rate during light'),
                       (axs[3], 'rate_lit', 'Lit E rate during light')]:
        for c, k in zip(cm, ks):
            v = S[(int(k), True)][q]
            ax.errorbar(drives, v.mean(1), v.std(1), c=c, marker='o', ms=3, capsize=2, label=f'×{k}')
            if q != 'rate_lit':
                ax.plot(drives, S[(int(k), False)][q].mean(1), c=c, ls=':', lw=1)
        ax.set_xlabel('Tonic drive D'); ax.set_ylabel('Hz'); ax.set_title(ttl + ' (dotted = no light)', fontsize=10)
        ax.legend(fontsize=7, frameon=False)

    ax = axs[4]
    for c, k in zip(cm, ks):
        v = S[(int(k), True)]['lam']
        ax.errorbar(drives, np.nanmean(v, 1), np.nanstd(v, 1), c=c, marker='o', ms=3, capsize=2, label=f'×{k}')
        ax.plot(drives, np.nanmean(S[(int(k), False)]['lam'], 1), c=c, ls=':', lw=1)
    ax.axhline(1, c='k', ls='--', lw=0.8)
    ax.set_xlabel('Tonic drive D'); ax.set_ylabel('Re λ_max'); ax.set_title(f'Rhythm threshold (Re λ = 1), {n_lam} draws', fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_sub, f'opsinI_summary_N{N}'))

    q_ = lambda v: f'{np.nanmean(v):.2f}±{np.nanstd(v):.2f}'
    lines = [f'N = {N} ({net.N_exc} E : {net.N_inh} I), C = {net.cfg.C}; opsin {n_op} E cells; random light max {i_max}; '
             f'baseline E weight {net.W_E:.4f}; reference D=20 osc {osc_ref:.2f}', '']
    for k in ks:
        lines.append(f'--- opsin->I synapses x{k} (weight {k * net.W_E:.3f}) ---')
        lines.append('D: osc light (max) | osc no light | post | rate lit/unlitE/I light | unlitE/I no light | Re lam light/none')
        L, C0 = S[(int(k), True)], S[(int(k), False)]
        for i, D in enumerate(drives):
            lines.append(
                f'{D:2d}: {q_(L["osc"][i])} ({np.nanmax(L["osc"][i]):.2f}) | {q_(C0["osc"][i])} | {q_(L["post"][i])} | '
                f'{L["rate_lit"][i].mean():.1f}/{L["rate_unlitE"][i].mean():.1f}/{L["rate_I"][i].mean():.1f} | '
                f'{C0["rate_unlitE"][i].mean():.1f}/{C0["rate_I"][i].mean():.1f} | '
                f'{np.nanmean(L["lam"][i]):.2f}/{np.nanmean(C0["lam"][i]):.2f}'
            )
    txt = '\n'.join(lines); print(txt)
    with open(os.path.join(out_sub, f'opsinI_summary_N{N}.txt'), 'w') as fh:
        fh.write(txt + '\n')
