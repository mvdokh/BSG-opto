"""Experiment 3: Random E opsin cells with higher connection probability onto I cells."""
import os
import numpy as np
import matplotlib.pyplot as plt

from ..config import ChR2Config, NetworkConfig, NeuronConfig, ProtocolConfig, RESULTS_DIR
from ..network import BalancedNetwork
from ..opsin import pick_random_exc
from ..connectivity import boost_opsin_to_I
from ..simulate import run_pulse
from ..analysis import compute_metrics
from ..plotting import plot_heatmap_grid, plot_light_trace, plot_raster, save_fig, shade_light

DESCRIPTION = 'Increase opsin E -> I connection probability (rewire vs add) from rest'


def add_args(parser):
    parser.add_argument('--pct', type=float, default=5.0, help='Percent of E cells with opsin (default: 5)')
    parser.add_argument('--i-max', type=float, default=80.0, help='Peak photocurrent amplitude (default: 80)')
    parser.add_argument('--n-draws', type=int, default=20, help='Random opsin draws per p_EI (default: 20)')
    parser.add_argument('--out-dir', type=str, default=os.path.join(RESULTS_DIR, '03_ei_connectivity'))


def run(pct: float = 5.0, i_max: float = 80.0, n_draws: int = 20,
        out_dir: str = os.path.join(RESULTS_DIR, '03_ei_connectivity'), **_):
    os.makedirs(out_dir, exist_ok=True)
    net0 = BalancedNetwork(NetworkConfig())
    neuron, proto, chr2 = NeuronConfig(), ProtocolConfig(), ChR2Config(i_max=i_max)
    C0 = net0.cfg.C
    a, b = proto.analysis_win
    n_stim = int(round(pct / 100.0 * net0.N_exc))
    pct_str, tag = f'{pct:g} %', f'{pct:g}pct'

    P = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0])
    keys = ('osc', 'rate_unlitE', 'rate_I', 'osc_nolight', 'osc_ref')
    res = {mode: {k: np.zeros((P.size, n_draws)) for k in keys} for mode in ('rewire', 'add')}
    for mode in res:
        for i, p in enumerate(P):
            for s in range(n_draws):
                op = pick_random_exc(net0.N_exc, n_stim, 1000 + s)
                Wm = boost_opsin_to_I(net0.W, op, net0.N_exc, float(p), C0, mode=mode, seed=s)
                R, _ = run_pulse(Wm, op, i_max, proto, neuron, chr2, seed=100 + s)
                m = compute_metrics(R, op, (a, b))
                unlitE = np.setdiff1d(np.arange(net0.N_exc), op)
                r = res[mode]
                r['osc'][i, s] = m['osc_sd']
                r['rate_unlitE'][i, s] = R[unlitE, a:b].mean()
                r['rate_I'][i, s] = R[net0.N_exc:, a:b].mean()
                if mode == 'rewire':
                    R0, _ = run_pulse(Wm, [], 0.0, proto, neuron, chr2, seed=100 + s)
                    r['osc_nolight'][i, s] = compute_metrics(R0, op, (a, b))['osc_sd']
                    if s < 5:
                        Rg, _ = run_pulse(Wm, np.arange(net0.N), proto.i_ref, proto, neuron, chr2,
                                          seed=100 + s, opsin=False)
                        r['osc_ref'][i, s] = compute_metrics(Rg, op, (a, b))['osc_sd']
    res['rewire']['osc_ref'] = res['rewire']['osc_ref'][:, :5]

    amps = np.array([10, 20, 40, 80, 160, 320])
    grid = np.zeros((P.size, amps.size))
    for i, p in enumerate(P):
        for k, A in enumerate(amps):
            v = []
            for s in range(10):
                op = pick_random_exc(net0.N_exc, n_stim, 1000 + s)
                Wm = boost_opsin_to_I(net0.W, op, net0.N_exc, float(p), C0, mode='rewire', seed=s)
                R, _ = run_pulse(Wm, op, float(A), proto, neuron, chr2, seed=100 + s)
                v.append(compute_metrics(R, op, (a, b))['osc_sd'])
            grid[i, k] = np.mean(v)

    R_ref, _ = run_pulse(net0.W, np.arange(net0.N), proto.i_ref, proto, neuron, chr2, seed=5, opsin=False)
    osc_ref0 = compute_metrics(R_ref, [], (a, b))['osc_sd']
    R_rest, _ = run_pulse(net0.W, [], 0.0, proto, neuron, chr2, seed=5)
    osc_rest0 = compute_metrics(R_rest, [], (a, b))['osc_sd']

    # ---- Figure
    ex_op = pick_random_exc(net0.N_exc, n_stim, 1000)
    C_OP = '#ff4f4f'
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(4, 3, height_ratios=[0.8, 1, 0.7, 1], hspace=0.55, wspace=0.35)
    for col, p in enumerate([0.1, 0.5, 1.0]):
        Wm = boost_opsin_to_I(net0.W, ex_op, net0.N_exc, p, C0, mode='rewire', seed=0)
        R, _ = run_pulse(Wm, ex_op, i_max, proto, neuron, chr2, seed=100)
        lab = f'baseline (p = {C0:g})' if p == C0 else f'p_EI = {p:g}'

        ax = fig.add_subplot(gs[0, col])
        blk = (Wm[net0.N_exc:, :net0.N_exc] > 0).astype(float)
        img = np.ones((net0.N_inh, net0.N_exc, 3)); img[blk > 0] = (0.35, 0.35, 0.35)
        for j in ex_op:
            img[blk[:, j] > 0, j] = (1, 0.2, 0.2)
        ax.imshow(img, aspect='auto', interpolation='none')
        ax.set_xlabel('presynaptic E cell'); ax.set_ylabel('postsynaptic I cell')
        k_op = blk[:, ex_op].sum(0).mean()
        k_other = blk[:, np.setdiff1d(np.arange(net0.N_exc), ex_op)].sum(0).mean()
        ax.set_title(f'E→I connections, {lab}\nopsin cells (red) contact {k_op:.0f} I cells, others {k_other:.0f}', fontsize=9)

        ax = fig.add_subplot(gs[1, col])
        plot_raster(ax, R, net0.phase_sort, proto, opsin_idx=ex_op, vmax=60.0, marker_color=C_OP, marker_x=-0.1)
        ax.set_title(f'{pct_str} E ChR2 (peak I = {i_max:g}, 5 s), {lab}', fontsize=10)
        ax.set_ylabel('Neuron (phase-sorted)')

        ax = fig.add_subplot(gs[2, col])
        shade_light(ax, proto); plot_light_trace(ax, proto, chr2)
        unlit = np.setdiff1d(np.arange(net0.N_exc), ex_op)
        ax.plot(proto.tvec, R[ex_op].mean(0), c=C_OP, label='lit E')
        ax.plot(proto.tvec, R[unlit].mean(0), c='#e09b22', label='unlit E')
        ax.plot(proto.tvec, R[net0.N_exc:].mean(0), c='#3a6fd8', label='I')
        ax.set_xlim(0, proto.T / 1000.0); ax.set_ylim(0, 75); ax.set_xlabel('Time (s)')
        ax.set_ylabel('Mean rate (Hz)'); ax.legend(fontsize=7, frameon=False, ncol=3, loc='upper right')

    ax = fig.add_subplot(gs[3, 0])
    for mode, c, ls in [('rewire', C_OP, '-'), ('add', '#8a2be2', '--')]:
        o = res[mode]['osc']
        ax.errorbar(P, o.mean(1), o.std(1), c=c, ls=ls, marker='o', capsize=2, label=f'light on ({mode})')
    ax.plot(P, res['rewire']['osc_nolight'].mean(1), c='.6', ls='--', marker='.', label='no light')
    o = res['rewire']['osc_ref']
    ax.errorbar(P, o.mean(1), o.std(1), c='k', ls=':', marker='s', ms=3, capsize=2, label='global drive (same network)')
    ax.set_xlabel('p_EI: opsin E → I connection probability'); ax.set_ylabel('Unlit amplitude (rate SD, sp/s)')
    ax.set_title(f'{n_draws} random {pct_str} opsin draws per point', fontsize=9)
    ax.set_yscale('log'); ax.legend(fontsize=7, frameon=False)

    ax = fig.add_subplot(gs[3, 1])
    for k, c, lab in [('rate_unlitE', '#e09b22', 'unlit E'), ('rate_I', '#3a6fd8', 'I')]:
        for mode, ls in [('rewire', '-'), ('add', '--')]:
            v = res[mode][k]
            ax.errorbar(P, v.mean(1), v.std(1), c=c, ls=ls, marker='o', ms=4, capsize=2, label=f'{lab} ({mode})')
    ax.axhline(R_rest[:, proto.stim_on:proto.stim_off].mean(), c='.6', ls='--', lw=0.8, label='rest')
    ax.set_xlabel('p_EI'); ax.set_ylabel('Mean rate during light (Hz)')
    ax.set_title('Where the extra drive goes', fontsize=9); ax.legend(fontsize=7, frameon=False, ncol=2)

    plot_heatmap_grid(fig.add_subplot(gs[3, 2]), grid, [str(x) for x in amps], [f'{p:g}' for p in P],
                      'Peak photocurrent I', 'p_EI',
                      f'p_EI × light intensity (rewire)\nreference rhythm = {osc_ref0:.1f}, no light = {osc_rest0:.2f}')
    save_fig(fig, os.path.join(out_dir, f'opto_EI_connectivity_{tag}'))

    q = lambda v: f'{v.mean():.2f}±{v.std():.2f}'
    lines = [f'{pct_str} opsin E cells ({n_stim}), I_OPTO = {i_max}, light {proto.stim_on}-{proto.stim_off} ms',
             f'original network: reference osc SD = {osc_ref0:.2f}, no light = {osc_rest0:.2f}', '']
    for mode in res:
        lines.append(f'[{mode}] p_EI: osc light-on | unlit E rate | I rate' +
                     (' | osc no-light | osc global drive' if mode == 'rewire' else ''))
        for i, p in enumerate(P):
            r = res[mode]
            ln = f'  {p:4.2f}: {q(r["osc"][i])} (max {r["osc"][i].max():.2f}) | {q(r["rate_unlitE"][i])} | {q(r["rate_I"][i])}'
            if mode == 'rewire':
                ln += f' | {q(r["osc_nolight"][i])} | {q(r["osc_ref"][i])}'
            lines.append(ln)
    lines += ['', 'p_EI x amplitude (rewire), rows p, cols ' + str(amps.tolist())]
    lines += [f'  {p:4.2f}: ' + ' '.join(f'{g:.2f}' for g in grid[i]) for i, p in enumerate(P)]
    txt = '\n'.join(lines); print(txt)
    with open(os.path.join(out_dir, f'opto_EI_connectivity_{tag}.txt'), 'w') as fh:
        fh.write(txt + '\n')
