"""Experiment 8: Opsin E cells with elevated output connectivity to ALL cells (E & I),
across one or more max light intensities (each saved to its own I<val>/ subfolder).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from ..config import ChR2Config, NetworkConfig, NeuronConfig, ProtocolConfig, RESULTS_DIR
from ..network import BalancedNetwork
from ..opsin import draw_variable_light
from ..connectivity import boost_opsin_to_all
from ..simulate import run_pulse, run_tonic_opsin
from ..analysis import compute_metrics, lam_from_rates, nerve_readout
from ..plotting import plot_heatmap_grid, plot_light_model, plot_nerve, plot_raster, save_fig, shade_light

DESCRIPTION = 'Altered connectivity (opsin -> all cells hub rewiring, fixed in-degree) across tonic drive D = 0..20 and max intensities'


def add_args(parser):
    parser.add_argument('--pct', type=float, default=10.0, help='Percent of E cells with opsin (default: 10)')
    parser.add_argument('--p-op', type=float, default=0.5, help='Main opsin->all connection prob (default: 0.5)')
    parser.add_argument('--i-max', type=float, nargs='+', default=[80.0],
                        help='One or more max peak photocurrents, e.g. --i-max 10 20 40 80 160 320 (default: 80)')
    parser.add_argument('--d-min', type=int, default=0)
    parser.add_argument('--d-max', type=int, default=20)
    parser.add_argument('--n-draws', type=int, default=20)
    parser.add_argument('--out-dir', type=str, default=os.path.join(RESULTS_DIR, '09_tonic_altconn'))


def _run_one_intensity(net: BalancedNetwork, pct: float, p_op: float, i_max: float,
                       drives: np.ndarray, n_draws: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    neuron, proto, chr2 = NeuronConfig(), ProtocolConfig(), ChR2Config(i_max=i_max)
    d_ctx = np.arange(0, 21)
    a, b = proto.analysis_win
    C0 = net.cfg.C
    n_stim = int(round(pct / 100.0 * net.N_exc))
    pct_str = f'{pct:g} %'
    p_sweep = np.array([0.1, 0.25, 0.5, 0.75, 1.0])
    wcache = {}

    def W_alt(s: int, p: float = p_op) -> np.ndarray:
        key = (s, p)
        if key not in wcache:
            op, _ = draw_variable_light(net.N_exc, n_stim, s)
            wcache[key] = boost_opsin_to_all(net.W, op, net.N_exc, p, C0, seed=5000 + s)
        return wcache[key]

    conds = ('var', 'uni', 'ctrl')
    keys = ('osc', 'post', 'lam', 'rate_lit', 'rate_unlitE', 'rate_I')
    S = {c: {k: np.zeros((drives.size, n_draws)) for k in keys} for c in conds}
    for i, D in enumerate(drives):
        for s in range(n_draws):
            op, phi = draw_variable_light(net.N_exc, n_stim, s)
            Wd = W_alt(s)
            unlitE = np.setdiff1d(np.arange(net.N_exc), op)
            for c, R in (
                ('var', run_tonic_opsin(Wd, op, float(D), proto, neuron, chr2, phi=phi, i_max=i_max, seed=100 + s)),
                ('uni', run_tonic_opsin(Wd, op, float(D), proto, neuron, chr2, phi=np.ones(op.size), i_max=i_max, seed=100 + s)),
                ('ctrl', run_tonic_opsin(Wd, op, float(D), proto, neuron, chr2, phi=phi, i_max=i_max, seed=100 + s, light=False)),
            ):
                m = compute_metrics(R, op, (a, b))
                S[c]['osc'][i, s] = m['osc_sd']
                S[c]['post'][i, s] = compute_metrics(R, op, proto.post_win)['osc_sd']
                S[c]['lam'][i, s] = lam_from_rates(Wd, R, (a, b), neuron).real
                S[c]['rate_lit'][i, s] = R[op, a:b].mean()
                S[c]['rate_unlitE'][i, s] = R[unlitE, a:b].mean()
                S[c]['rate_I'][i, s] = R[net.N_exc:, a:b].mean()

    ctx = np.array([np.mean([
        compute_metrics(run_tonic_opsin(net.W, [], float(D), proto, neuron, chr2, seed=500 + s, light=False), [], (a, b))['osc_sd']
        for s in range(5)
    ]) for D in d_ctx])

    grid_imax = np.array([10., 20., 40., 80., 160., 320.])
    grid = np.zeros((drives.size, grid_imax.size))
    for i, D in enumerate(drives):
        for k, imx in enumerate(grid_imax):
            if np.isclose(imx, i_max):
                grid[i, k] = S['var']['osc'][i, :min(10, n_draws)].mean()
            else:
                v = []
                for s in range(10):
                    op, phi = draw_variable_light(net.N_exc, n_stim, s)
                    R = run_tonic_opsin(W_alt(s), op, float(D), proto, neuron, chr2, phi=phi, i_max=float(imx), seed=100 + s)
                    v.append(compute_metrics(R, op, (a, b))['osc_sd'])
                grid[i, k] = np.mean(v)

    n_sw = min(10, n_draws)
    SW = {k: np.zeros((p_sweep.size, drives.size, n_sw)) for k in ('on', 'off')}
    for pi_, p in enumerate(p_sweep):
        if np.isclose(p, p_op):
            SW['on'][pi_] = S['var']['osc'][:, :n_sw]
            SW['off'][pi_] = S['ctrl']['osc'][:, :n_sw]
        else:
            for i, D in enumerate(drives):
                for s in range(n_sw):
                    op, phi = draw_variable_light(net.N_exc, n_stim, s)
                    Wd = W_alt(s, float(p))
                    SW['on'][pi_, i, s] = compute_metrics(
                        run_tonic_opsin(Wd, op, float(D), proto, neuron, chr2, phi=phi, i_max=i_max, seed=100 + s),
                        op, (a, b))['osc_sd']
                    SW['off'][pi_, i, s] = compute_metrics(
                        run_tonic_opsin(Wd, op, float(D), proto, neuron, chr2, phi=phi, i_max=i_max, seed=100 + s, light=False),
                        op, (a, b))['osc_sd']

    R_ref, _ = run_pulse(net.W, np.arange(net.N), proto.i_ref, proto, neuron, chr2, seed=5, opsin=False)
    osc_ref = compute_metrics(R_ref, [], (a, b))['osc_sd']

    # Figure 1: light model
    plot_light_model(out_dir, proto, chr2, net.N_exc, n_stim, i_max=i_max)

    # Figure 2: examples
    op, phi = draw_variable_light(net.N_exc, n_stim, 0)
    unlit = np.setdiff1d(np.arange(net.N_exc), op)
    order = net.ei_phase_sort()
    ex_R, ex_nerve = [], []
    for D in drives:
        R = run_tonic_opsin(W_alt(0), op, float(D), proto, neuron, chr2, phi=phi, i_max=i_max, seed=100)
        nr = nerve_readout(net, R, mode='xii')
        ex_R.append(R.astype(np.float32)); ex_nerve.append((nr['upper'], nr['lower']))
    nerve_off = 1.2 * max(max(np.abs(p).max(), np.abs(r).max()) for p, r in ex_nerve)
    rate_ylim = 70 if i_max > 80 else 50

    n_blk = int(np.ceil(drives.size / 5))
    fig = plt.figure(figsize=(20, 8 * n_blk))
    gs = fig.add_gridspec(4 * n_blk, 5, height_ratios=[1, 0.45, 0.45, 0.3] * n_blk, hspace=0.12, wspace=0.25)
    for j, D in enumerate(drives):
        r0, c = 4 * (j // 5), j % 5
        R = ex_R[j]; prot, retr = ex_nerve[j]
        ax = fig.add_subplot(gs[r0, c])
        plot_raster(ax, R, order, proto, opsin_idx=op, phi=phi, split_ei=net.N_exc, vmax=50.0)
        ax.set_title(f'tonic drive D = {D}', fontsize=11, fontweight='bold')
        ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel('Neuron\n(E, I each sorted by phase)')

        ax = fig.add_subplot(gs[r0 + 1, c]); shade_light(ax, proto)
        ax.plot(proto.tvec, R[op].mean(0), c='#ff4f4f', lw=1, label='lit E (mean)')
        ax.plot(proto.tvec, R[unlit].mean(0), c='#e09b22', lw=1, label='unlit E')
        ax.plot(proto.tvec, R[net.N_exc:].mean(0), c='#3a6fd8', lw=1, label='I')
        ax.set_xlim(0, proto.T / 1000.0); ax.set_ylim(0, rate_ylim); ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel('Mean rate (Hz)')
        if j == 0:
            ax.legend(fontsize=6, frameon=False, ncol=3, loc='upper right')

        ax = fig.add_subplot(gs[r0 + 2, c])
        plot_nerve(ax, proto.tvec, prot, retr, nerve_off, ('protract', 'retract'), proto)
        ax.set_xlabel('Time (s)')
        if c == 0:
            ax.set_ylabel('XII nerve')
    fig.suptitle(f'Opsin cells project to all cells with p = {p_op:g} (baseline {C0:g}), in-degree fixed. '
                 f'Tonic drive D + ChR2 in random {pct_str} of E, random light (max I = {i_max:g}), light 1-6 s\n'
                 f'Raster: E (top) and I (bottom), each sorted by phase preference.  XII nerve: protractor / retractor',
                 fontsize=11, y=0.905)
    save_fig(fig, os.path.join(out_dir, 'tonic_examples_altconn'), dpi=130)

    # Figure 3: summary
    C_col = {'var': '#ff4f4f', 'uni': '#8a2be2', 'ctrl': '.5'}
    L_lab = {'var': 'random light (main)', 'uni': 'uniform max light', 'ctrl': 'no light'}
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.3))
    ax = axs[0]
    ax.plot(d_ctx, ctx, c='.75', lw=3, alpha=0.6, label='original network, tonic only')
    for c in conds:
        v = S[c]['osc']
        ax.errorbar(drives + {'var': 0, 'uni': 0.15, 'ctrl': -0.15}[c], v.mean(1), v.std(1),
                    c=C_col[c], marker='o', ms=4, capsize=2, label=L_lab[c])
    ax.plot(drives, S['var']['post'].mean(1), c=C_col['var'], ls=':', label='after light off (random light)')
    ax.axvline(20, c='k', ls=':', lw=0.8)
    ax.set_yscale('log'); ax.set_xlabel('Tonic drive D (all cells)'); ax.set_ylabel('Rhythmicity of unlit cells (SD, Hz)')
    ax.set_title(f'opsin→all p = {p_op:g}, max I = {i_max:g}, {n_draws} draws (orig. ref D = 20: {osc_ref:.1f})', fontsize=9.5)
    ax.legend(fontsize=7, frameon=False, loc='upper left')

    ax = axs[1]
    for c in conds:
        v = S[c]['lam']
        ax.errorbar(drives, v.mean(1), v.std(1), c=C_col[c], marker='o', ms=4, capsize=2, label=L_lab[c])
    ax.axhline(1, c='k', ls='--', lw=0.8)
    ax.set_xlabel('Tonic drive D'); ax.set_ylabel('Re λ_max (linearised at mean state)')
    ax.set_title('Distance from rhythm threshold (Re λ = 1)', fontsize=10); ax.legend(fontsize=7, frameon=False)

    ax = axs[2]
    for k, col, lab in [('rate_lit', '#ff4f4f', 'lit E'), ('rate_unlitE', '#e09b22', 'unlit E'), ('rate_I', '#3a6fd8', 'I')]:
        ax.errorbar(drives, S['var'][k].mean(1), S['var'][k].std(1), c=col, marker='o', ms=4, capsize=2, label=f'{lab} (random light)')
        ax.plot(drives, S['uni'][k].mean(1), c=col, ls='--', lw=1, label=f'{lab} (uniform max)')
    ax.plot(drives, S['ctrl']['rate_unlitE'].mean(1), c='.5', marker='.', label='E, no light')
    ax.set_xlabel('Tonic drive D'); ax.set_ylabel('Mean rate during light (Hz)')
    ax.set_title('Rates', fontsize=10); ax.legend(fontsize=6, frameon=False, ncol=2)

    matches = np.where(np.isclose(grid_imax, i_max))[0]
    plot_heatmap_grid(axs[3], grid, [f'{x:g}' for x in grid_imax], [str(d) for d in drives],
                      'Max peak photocurrent (light varies 0-max per cell)', 'Tonic drive D',
                      'Tonic × max light (random light)',
                      highlight_col=int(matches[0]) if matches.size else None)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'tonic_summary_altconn'))

    # Figure 4: p_out sweep
    cmp = plt.cm.plasma(np.linspace(0, 0.85, p_sweep.size))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
    for c_, (pi_, p) in zip(cmp, enumerate(p_sweep)):
        axs[0].plot(drives, SW['on'][pi_].mean(-1), c=c_, marker='o', ms=3, label=f'p = {p:g}')
        axs[0].plot(drives, SW['off'][pi_].mean(-1), c=c_, ls=':', lw=1)
        e = SW['on'][pi_] - SW['off'][pi_]
        axs[1].errorbar(drives + 0.06 * pi_, e.mean(-1), e.std(-1), c=c_, marker='o', ms=3, capsize=2, label=f'p = {p:g}')
    axs[0].plot([], [], c='k', ls=':', label='dotted: no light')
    axs[0].set_yscale('log'); axs[0].set_xlabel('Tonic drive D'); axs[0].set_ylabel('Rhythmicity of unlit cells (SD, Hz)')
    axs[0].set_title(f'Opsin→all connection prob (max I = {i_max:g}, baseline {C0:g}), random light, {n_sw} draws', fontsize=9.5)
    axs[0].legend(fontsize=7, frameon=False)
    axs[1].axhline(0, c='k', lw=0.6)
    axs[1].set_xlabel('Tonic drive D'); axs[1].set_ylabel('Light effect (light − no light, Hz)')
    axs[1].set_title(f'How much the light adds (max I = {i_max:g}), by opsin output connectivity', fontsize=9.5)
    axs[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'pop_sweep_altconn'))

    q = lambda v: f'{v.mean():.2f}±{v.std():.2f}'
    lines = [
        f'opsin->all p = {p_op} (baseline {C0}), in-degree fixed; {pct_str} opsin E ({n_stim} cells); '
        f'light per cell phi ~ U(0,1), peak = {i_max}*phi; reference D=20 osc {osc_ref:.2f}', '',
        'D: osc random light (max) | uniform max light (max) | no light | post (random) | Re lam random/uniform/none | rate lit/unlitE/I (random)',
    ]
    for i, D in enumerate(drives):
        v, u, n_ = S['var'], S['uni'], S['ctrl']
        lines.append(
            f'{D:2d}: {q(v["osc"][i])} ({v["osc"][i].max():.2f}) | {q(u["osc"][i])} ({u["osc"][i].max():.2f}) | '
            f'{q(n_["osc"][i])} | {q(v["post"][i])} | {v["lam"][i].mean():.2f}/{u["lam"][i].mean():.2f}/'
            f'{n_["lam"][i].mean():.2f} | {v["rate_lit"][i].mean():.1f}/{v["rate_unlitE"][i].mean():.1f}/{v["rate_I"][i].mean():.1f}'
        )
    lines += ['', 'context tonic only D=0..20: ' + ' '.join(f'{x:.2f}' for x in ctx),
              '', 'tonic x max light (random light), cols ' + str(grid_imax.tolist())]
    lines += [f'  {D:2d}: ' + ' '.join(f'{g:.2f}' for g in grid[i]) for i, D in enumerate(drives)]
    lines += ['', f'p_out sweep ({n_sw} draws, max I = {i_max:g}): D: ' + ' | '.join(f'p={p:g} on/off' for p in p_sweep)]
    lines += [f'  {D:2d}: ' + ' | '.join(f'{SW["on"][k, i].mean():.2f}/{SW["off"][k, i].mean():.2f}' for k in range(p_sweep.size))
              for i, D in enumerate(drives)]
    txt = '\n'.join(lines); print(txt)
    with open(os.path.join(out_dir, 'tonic_summary_altconn.txt'), 'w') as fh:
        fh.write(txt + '\n')


def run(pct: float = 10.0, p_op: float = 0.5, i_max=(80.0,), d_min: int = 0, d_max: int = 20,
        n_draws: int = 20, out_dir: str = os.path.join(RESULTS_DIR, '09_tonic_altconn'), **_):
    net = BalancedNetwork(NetworkConfig())
    drives = np.arange(d_min, d_max + 1)
    intensities = [float(i_max)] if np.isscalar(i_max) else [float(x) for x in i_max]
    for imx in intensities:
        sub = os.path.join(out_dir, f'I{imx:g}')
        print(f'=== tonic-altconn: max I = {imx:g} -> {sub} ===', flush=True)
        _run_one_intensity(net, pct, p_op, imx, drives, n_draws, sub)
