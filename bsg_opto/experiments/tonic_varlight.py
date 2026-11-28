"""Experiment 5: Tonic drive + 10% variable-light ChR2 (N=200 single-trial or N=2000 batched)."""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from ..config import ChR2Config, NetworkConfig, NeuronConfig, ProtocolConfig, RESULTS_DIR
from ..network import BalancedNetwork
from ..opsin import draw_variable_light
from ..simulate import make_batched_ext, run_pulse, run_tonic_opsin, simulate_batch
from ..analysis import compute_metrics, lam_from_rates, leading_real_eig, nerve_readout, unlit_mean
from ..plotting import plot_heatmap_grid, plot_light_model, plot_nerve, plot_raster, save_fig, shade_light

DESCRIPTION = 'Tonic drive + variable light per opsin cell with intensity-dependent desensitization (N=200 or N=2000)'


def add_args(parser):
    parser.add_argument('--N', type=int, default=200, help='Total neurons, e.g. 200 or 2000 (default: 200)')
    parser.add_argument('--pct', type=float, default=10.0, help='Percent of E cells with opsin (default: 10)')
    parser.add_argument('--d-min', type=int, default=0, help='Min tonic drive D (default: 0 for N=200, 1 for N=2000)')
    parser.add_argument('--d-max', type=int, default=20, help='Max tonic drive D (default: 20)')
    parser.add_argument('--i-max', type=float, default=80.0, help='Max peak photocurrent (default: 80)')
    parser.add_argument('--n-draws', type=int, default=20, help='Random opsin draws per D (default: 20)')
    parser.add_argument('--conn-seed', type=int, default=1, help='Network connectivity random seed (default: 1)')
    parser.add_argument('--seed', type=int, default=0, help='Seed offset for opsin selection, light, and noise (default: 0)')
    parser.add_argument('--out-dir', type=str, default=None)


def _run_small(net: BalancedNetwork, pct: float, drives: np.ndarray, i_max: float,
               n_draws: int, out_dir: str, seed_offset: int = 0):
    neuron, proto, chr2 = NeuronConfig(), ProtocolConfig(), ChR2Config(i_max=i_max)
    d_ctx = np.arange(0, 21)
    a, b = proto.analysis_win
    n_stim = int(round(pct / 100.0 * net.N_exc))
    pct_str, tag = f'{pct:g} %', f'{pct:g}pct'

    conds = ('var', 'uni', 'ctrl')
    keys = ('osc', 'post', 'lam', 'rate_lit', 'rate_unlitE', 'rate_I')
    S = {c: {k: np.zeros((drives.size, n_draws)) for k in keys} for c in conds}
    for i, D in enumerate(drives):
        for s in range(n_draws):
            op, phi = draw_variable_light(net.N_exc, n_stim, s, seed_offset=seed_offset)
            unlitE = np.setdiff1d(np.arange(net.N_exc), op)
            nseed = 100 + seed_offset + s
            for c, R in (
                ('var', run_tonic_opsin(net.W, op, float(D), proto, neuron, chr2, phi=phi, i_max=i_max, seed=nseed)),
                ('uni', run_tonic_opsin(net.W, op, float(D), proto, neuron, chr2, phi=np.ones(op.size), i_max=i_max, seed=nseed)),
                ('ctrl', run_tonic_opsin(net.W, op, float(D), proto, neuron, chr2, phi=phi, i_max=i_max, seed=nseed, light=False)),
            ):
                m = compute_metrics(R, op, (a, b))
                S[c]['osc'][i, s] = m['osc_sd']
                S[c]['post'][i, s] = compute_metrics(R, op, proto.post_win)['osc_sd']
                S[c]['lam'][i, s] = lam_from_rates(net.W, R, (a, b), neuron).real
                S[c]['rate_lit'][i, s] = R[op, a:b].mean()
                S[c]['rate_unlitE'][i, s] = R[unlitE, a:b].mean()
                S[c]['rate_I'][i, s] = R[net.N_exc:, a:b].mean()

    ctx = np.array([np.mean([
        compute_metrics(run_tonic_opsin(net.W, [], float(D), proto, neuron, chr2, seed=500 + seed_offset + s, light=False), [], (a, b))['osc_sd']
        for s in range(5)
    ]) for D in d_ctx])

    scales = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    grid = np.zeros((drives.size, scales.size))
    for i, D in enumerate(drives):
        for k, sc in enumerate(scales):
            v = []
            for s in range(10):
                op, phi = draw_variable_light(net.N_exc, n_stim, s, seed_offset=seed_offset)
                R = run_tonic_opsin(net.W, op, float(D), proto, neuron, chr2, phi=phi, i_max=i_max * sc, seed=100 + seed_offset + s)
                v.append(compute_metrics(R, op, (a, b))['osc_sd'])
            grid[i, k] = np.mean(v)

    R_ref, _ = run_pulse(net.W, np.arange(net.N), proto.i_ref, proto, neuron, chr2, seed=5 + seed_offset, opsin=False)
    osc_ref = compute_metrics(R_ref, [], (a, b))['osc_sd']

    # Figure 1: light model
    plot_light_model(out_dir, proto, chr2, net.N_exc, n_stim, i_max=i_max, seed_offset=seed_offset)

    # Figure 2: examples (split E/I raster + mean rates + XII nerve)
    op, phi = draw_variable_light(net.N_exc, n_stim, 0, seed_offset=seed_offset)
    unlit = np.setdiff1d(np.arange(net.N_exc), op)
    order = net.ei_phase_sort()
    ex_R, ex_nerve = [], []
    for D in drives:
        R = run_tonic_opsin(net.W, op, float(D), proto, neuron, chr2, phi=phi, i_max=i_max, seed=100 + seed_offset)
        nr = nerve_readout(net, R, mode='xii', seed=seed_offset)
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
    fig.suptitle(f'Tonic drive D + ChR2 in random {pct_str} of E cells, random light per cell (max I = {i_max:g}; '
                 f'marker shade = light level), intensity-dependent desensitization, light 1-6 s '
                 f'(conn_seed={net.cfg.seed}, seed={seed_offset})\n'
                 f'Raster: E (top) and I (bottom), each sorted by phase preference.  Hypoglossal (XII) output: '
                 f'protractor / retractor = paper\'s two-pool readout (±π/2, π/8 spread), same scale for all D',
                 fontsize=11, y=0.905)
    save_fig(fig, os.path.join(out_dir, f'tonic_examples_{tag}_varlight'), dpi=130)

    # Figure 3: summary
    C_col = {'var': '#ff4f4f', 'uni': '#8a2be2', 'ctrl': '.5'}
    L_lab = {'var': 'random light (main)', 'uni': 'uniform max light', 'ctrl': 'no light'}
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.3))
    ax = axs[0]
    ax.plot(d_ctx, ctx, c='.75', lw=3, alpha=0.6, label='tonic only, D = 0-20')
    for c in conds:
        v = S[c]['osc']
        ax.errorbar(drives + {'var': 0, 'uni': 0.15, 'ctrl': -0.15}[c], v.mean(1), v.std(1),
                    c=C_col[c], marker='o', ms=4, capsize=2, label=L_lab[c])
    ax.plot(drives, S['var']['post'].mean(1), c=C_col['var'], ls=':', label='after light off (random light)')
    ax.axvline(20, c='k', ls=':', lw=0.8)
    ax.set_yscale('log'); ax.set_xlabel('Tonic drive D (all cells)'); ax.set_ylabel('Unlit amplitude (rate SD, sp/s)')
    ax.set_title(f'{n_draws} draws per point (D = 20 ref: {osc_ref:.1f}; conn_seed={net.cfg.seed}, seed={seed_offset})', fontsize=9.5)
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

    plot_heatmap_grid(axs[3], grid, [f'{i_max * s:g}' for s in scales], [str(d) for d in drives],
                      'Max peak photocurrent (light varies 0-max per cell)', 'Tonic drive D',
                      'Tonic × max light (random light)')
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'tonic_summary_{tag}_varlight'))

    q = lambda v: f'{v.mean():.2f}±{v.std():.2f}'
    lines = [
        f'conn_seed={net.cfg.seed}, seed_offset={seed_offset}; {pct_str} opsin E ({n_stim} cells); '
        f'light per cell phi ~ U(0,1), peak = {i_max}*phi; '
        f'desens: ss {chr2.ss_dim}->{chr2.ss_frac}, tau {chr2.tau_des_dim}->{chr2.tau_des} ms (K={chr2.k_light}); '
        f'reference D=20 osc {osc_ref:.2f}', '',
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
              '', 'tonic x max light (random light), cols ' + str((i_max * scales).tolist())]
    lines += [f'  {D:2d}: ' + ' '.join(f'{g:.2f}' for g in grid[i]) for i, D in enumerate(drives)]
    txt = '\n'.join(lines); print(txt)
    with open(os.path.join(out_dir, f'tonic_summary_{tag}_varlight.txt'), 'w') as fh:
        fh.write(txt + '\n')


def _run_large(net: BalancedNetwork, pct: float, drives: np.ndarray, i_max: float,
               n_draws: int, out_dir: str, seed_offset: int = 0):
    neuron, proto, chr2 = NeuronConfig(), ProtocolConfig(), ChR2Config(i_max=i_max)
    d_ctx = np.arange(0, 21)
    n_stim = int(round(pct / 100.0 * net.N_exc))
    n_lam = 5
    conds = {'var': lambda op, phi: (op, phi), 'uni': lambda op, phi: (op, np.ones_like(phi)),
             'ctrl': lambda op, phi: (None, phi)}
    keys = ('osc', 'post', 'lam', 'rate_lit', 'rate_unlitE', 'rate_I')
    S = {c: {k_: np.full((drives.size, n_draws), np.nan) for k_ in keys} for c in conds}
    ex_draw = draw_variable_light(net.N_exc, n_stim, 0, seed_offset=seed_offset)
    snaps_var, traces_var = None, None
    for c, fn in conds.items():
        t0 = time.time()
        cols, meta = [], []
        for i, D in enumerate(drives):
            for s in range(n_draws):
                op, phi = draw_variable_light(net.N_exc, n_stim, s, seed_offset=seed_offset)
                op_c, phi_c = fn(op, phi)
                cols.append(dict(base=float(D), op=op_c, phi=phi_c, i_max=i_max))
                meta.append((i, s, op))
        snap_cols, record = (), None
        if c == 'var':
            snap_cols = [j for j, (i, s, _) in enumerate(meta) if s == 0]
            op0 = ex_draw[0]; unlit0 = np.setdiff1d(np.arange(net.N_exc), op0)
            record = {'means': lambda R, V: np.concatenate([
                R[op0][:, snap_cols].mean(0), R[unlit0][:, snap_cols].mean(0), R[net.N_exc:][:, snap_cols].mean(0)
            ])}
        out = simulate_batch(net.W, proto.T, make_batched_ext(net.N, cols, proto, chr2), len(cols),
                             neuron=neuron, seed={'var': 11, 'uni': 12, 'ctrl': 13}[c] + seed_offset,
                             windows={'on': proto.analysis_win, 'post': proto.post_win},
                             snap_cols=snap_cols, record=record)
        for j, (i, s, op) in enumerate(meta):
            unlitE = np.setdiff1d(np.arange(net.N_exc), op)
            S[c]['osc'][i, s] = unlit_mean(out['on']['sd'][:, j], op)
            S[c]['post'][i, s] = unlit_mean(out['post']['sd'][:, j], op)
            S[c]['rate_lit'][i, s] = out['on']['mean'][op, j].mean()
            S[c]['rate_unlitE'][i, s] = out['on']['mean'][unlitE, j].mean()
            S[c]['rate_I'][i, s] = out['on']['mean'][net.N_exc:, j].mean()
            if s < n_lam:
                S[c]['lam'][i, s] = leading_real_eig(net.W, out['on']['Vmean'][:, j], neuron)
        if c == 'var':
            snaps_var, traces_var = out['snaps'], out['rec']['means']
        print(f'condition {c}: {time.time() - t0:.0f}s', flush=True)

    cols = [dict(base=float(D), op=None, phi=None) for D in d_ctx for _ in range(5)]
    out = simulate_batch(net.W, proto.T, make_batched_ext(net.N, cols, proto, chr2, light=False), len(cols),
                         neuron=neuron, seed=21 + seed_offset, windows={'on': proto.analysis_win}, snap_cols=())
    ctx = np.array([[unlit_mean(out['on']['sd'][:, j]) for j in range(i * 5, i * 5 + 5)] for i in range(d_ctx.size)])
    ctx_lam = np.array([leading_real_eig(net.W, out['on']['Vmean'][:, i * 5], neuron) for i in range(d_ctx.size)])

    scales = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    n_grid = 10
    cols, meta = [], []
    for i, D in enumerate(drives):
        for kk, sc in enumerate(scales):
            for s in range(n_grid):
                op, phi = draw_variable_light(net.N_exc, n_stim, s, seed_offset=seed_offset)
                cols.append(dict(base=float(D), op=op, phi=phi, i_max=i_max * sc))
                meta.append((i, kk, op))
    out = simulate_batch(net.W, proto.T, make_batched_ext(net.N, cols, proto, chr2), len(cols),
                         neuron=neuron, seed=31 + seed_offset, windows={'on': proto.analysis_win}, snap_cols=())
    grid = np.zeros((drives.size, scales.size)); cnt = np.zeros_like(grid)
    for j, (i, kk, op) in enumerate(meta):
        grid[i, kk] += unlit_mean(out['on']['sd'][:, j], op); cnt[i, kk] += 1
    grid /= cnt
    osc_ref = ctx[-1].mean()

    n_blk = int(np.ceil(drives.size / 5))
    fig = plt.figure(figsize=(20, 6 * n_blk))
    gs = fig.add_gridspec(2 * n_blk, 5, height_ratios=[1, 0.6] * n_blk, hspace=0.5, wspace=0.25)
    nD = drives.size
    for j, D in enumerate(drives):
        r0, c = 2 * (j // 5), j % 5
        ax = fig.add_subplot(gs[r0, c])
        plot_raster(ax, snaps_var[j], net.phase_sort, proto, vmax=50.0, interpolation='antialiased')
        ax.set_title(f'tonic drive D = {D}', fontsize=11, fontweight='bold')
        if c == 0:
            ax.set_ylabel('Neuron (phase-sorted)')
        ax = fig.add_subplot(gs[r0 + 1, c]); shade_light(ax, proto)
        ax.plot(proto.tvec, traces_var[:, j], c='#ff4f4f', lw=1, label='lit E (mean)')
        ax.plot(proto.tvec, traces_var[:, nD + j], c='#e09b22', lw=1, label='unlit E')
        ax.plot(proto.tvec, traces_var[:, 2 * nD + j], c='#3a6fd8', lw=1, label='I')
        ax.set_xlim(0, proto.T / 1000.0); ax.set_ylim(0, 50); ax.set_xlabel('Time (s)')
        if c == 0:
            ax.set_ylabel('Mean rate (Hz)')
        if j == 0:
            ax.legend(fontsize=6, frameon=False, ncol=3, loc='upper right')
    fig.suptitle(f'N = {net.N} ({net.N_exc} E : {net.N_inh} I, C = {net.cfg.C}). Tonic drive D + ChR2 in random '
                 f'{pct:g} % of E ({n_stim}), random light per cell (max I = {i_max:g}), light 1-6 s', fontsize=12, y=0.93)
    save_fig(fig, os.path.join(out_dir, f'tonic_examples_N{net.N}'), dpi=120)

    C_col = {'var': '#ff4f4f', 'uni': '#8a2be2', 'ctrl': '.5'}
    L_lab = {'var': 'random light (main)', 'uni': 'uniform max light', 'ctrl': 'no light'}
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.3))
    ax = axs[0]
    ax.plot(d_ctx, ctx.mean(1), c='.75', lw=3, alpha=0.6, label='tonic only, D = 0-20')
    for c in conds:
        v = S[c]['osc']
        ax.errorbar(drives + {'var': 0, 'uni': 0.15, 'ctrl': -0.15}[c], v.mean(1), v.std(1),
                    c=C_col[c], marker='o', ms=4, capsize=2, label=L_lab[c])
    ax.plot(drives, S['var']['post'].mean(1), c=C_col['var'], ls=':', label='after light off (random light)')
    ax.axvline(20, c='k', ls=':', lw=0.8)
    ax.set_yscale('log'); ax.set_xlabel('Tonic drive D (all cells)'); ax.set_ylabel('Unlit amplitude (rate SD, sp/s)')
    ax.set_title(f'N = {net.N}, {n_draws} draws per point (D = 20: {osc_ref:.1f})', fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc='upper left')

    ax = axs[1]
    ax.plot(d_ctx, ctx_lam, c='.75', lw=3, alpha=0.6, label='tonic only')
    for c in conds:
        v = S[c]['lam']
        ax.errorbar(drives, np.nanmean(v, 1), np.nanstd(v, 1), c=C_col[c], marker='o', ms=4, capsize=2, label=L_lab[c])
    ax.axhline(1, c='k', ls='--', lw=0.8)
    ax.set_xlabel('Tonic drive D'); ax.set_ylabel('Re λ_max (linearised at mean state)')
    ax.set_title(f'Distance from rhythm threshold ({n_lam} draws)', fontsize=10); ax.legend(fontsize=7, frameon=False)

    ax = axs[2]
    for k_, col, lab in [('rate_lit', '#ff4f4f', 'lit E'), ('rate_unlitE', '#e09b22', 'unlit E'), ('rate_I', '#3a6fd8', 'I')]:
        ax.errorbar(drives, S['var'][k_].mean(1), S['var'][k_].std(1), c=col, marker='o', ms=4, capsize=2, label=f'{lab} (random light)')
        ax.plot(drives, S['uni'][k_].mean(1), c=col, ls='--', lw=1, label=f'{lab} (uniform max)')
    ax.plot(drives, S['ctrl']['rate_unlitE'].mean(1), c='.5', marker='.', label='E, no light')
    ax.set_xlabel('Tonic drive D'); ax.set_ylabel('Mean rate during light (Hz)')
    ax.set_title('Rates', fontsize=10); ax.legend(fontsize=6, frameon=False, ncol=2)

    plot_heatmap_grid(axs[3], grid, [f'{i_max * s_:g}' for s_ in scales], [str(d) for d in drives],
                      'Max peak photocurrent (light varies 0-max per cell)', 'Tonic drive D',
                      f'Tonic × max light (random light, {n_grid} draws)')
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'tonic_summary_N{net.N}'))

    q = lambda v: f'{np.nanmean(v):.2f}±{np.nanstd(v):.2f}'
    lines = [
        f'N = {net.N} ({net.N_exc} E : {net.N_inh} I), C = {net.cfg.C}; opsin {n_stim} E cells ({pct:g} %); '
        f'phi ~ U(0,1), peak = {i_max}*phi; leading eig of W = {net.leading_eigval:.3f}', '',
        'D: osc random light (max) | uniform max (max) | no light | post (random) | Re lam random/uniform/none | rate lit/unlitE/I (random)',
    ]
    for i, D in enumerate(drives):
        v, u, n_ = S['var'], S['uni'], S['ctrl']
        lines.append(
            f'{D:2d}: {q(v["osc"][i])} ({v["osc"][i].max():.2f}) | {q(u["osc"][i])} ({u["osc"][i].max():.2f}) | '
            f'{q(n_["osc"][i])} | {q(v["post"][i])} | {np.nanmean(v["lam"][i]):.2f}/{np.nanmean(u["lam"][i]):.2f}/'
            f'{np.nanmean(n_["lam"][i]):.2f} | {v["rate_lit"][i].mean():.1f}/{v["rate_unlitE"][i].mean():.1f}/{v["rate_I"][i].mean():.1f}'
        )
    lines += ['', 'context tonic only D=0..20 (osc SD | Re lam):']
    lines += [f'  {D:2d}: {ctx[i].mean():.2f} | {ctx_lam[i]:.3f}' for i, D in enumerate(d_ctx)]
    lines += ['', 'tonic x max light (random light), cols ' + str((i_max * scales).tolist())]
    lines += [f'  {D:2d}: ' + ' '.join(f'{g:.2f}' for g in grid[i]) for i, D in enumerate(drives)]
    txt = '\n'.join(lines); print(txt)
    with open(os.path.join(out_dir, f'tonic_summary_N{net.N}.txt'), 'w') as fh:
        fh.write(txt + '\n')


def run(N: int = 200, pct: float = 10.0, d_min: int = 0, d_max: int = 20,
        i_max: float = 80.0, n_draws: int = 20, conn_seed: int = 1, seed: int = 0,
        out_dir: str | None = None, **_):
    if out_dir is None:
        sub = '05_tonic_varlight_10pct' if N == 200 else f'06_tonic_varlight_N{N}'
        out_dir = os.path.join(RESULTS_DIR, sub)
    os.makedirs(out_dir, exist_ok=True)
    net = BalancedNetwork(NetworkConfig(N=N, seed=conn_seed))
    drives = np.arange(d_min, d_max + 1)
    if N <= 200:
        _run_small(net, pct, drives, i_max, n_draws, out_dir, seed_offset=seed)
    else:
        _run_large(net, pct, drives, i_max, n_draws, out_dir, seed_offset=seed)
