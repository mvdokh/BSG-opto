"""Experiment 4: Uniform tonic drive D + fixed-intensity ChR2 in 5% of E cells."""
import os
import numpy as np
import matplotlib.pyplot as plt

from ..config import ChR2Config, NetworkConfig, NeuronConfig, ProtocolConfig, RESULTS_DIR
from ..network import BalancedNetwork
from ..opsin import pick_random_exc
from ..simulate import run_pulse, run_tonic_opsin
from ..analysis import compute_metrics, lam_from_rates
from ..plotting import plot_heatmap_grid, plot_light_trace, plot_raster, save_fig, shade_light

DESCRIPTION = 'Uniform tonic drive (D = 1..15) + fixed-intensity desensitizing ChR2 in 5% of E cells'


def add_args(parser):
    parser.add_argument('--pct', type=float, default=5.0, help='Percent of E cells with opsin (default: 5)')
    parser.add_argument('--d-min', type=int, default=1, help='Min tonic drive D (default: 1)')
    parser.add_argument('--d-max', type=int, default=15, help='Max tonic drive D (default: 15)')
    parser.add_argument('--i-max', type=float, default=80.0, help='Peak photocurrent (default: 80)')
    parser.add_argument('--n-draws', type=int, default=20, help='Random opsin draws per D (default: 20)')
    parser.add_argument('--out-dir', type=str, default=os.path.join(RESULTS_DIR, '04_tonic_drive_5pct'))


def run(pct: float = 5.0, d_min: int = 1, d_max: int = 15, i_max: float = 80.0,
        n_draws: int = 20, out_dir: str = os.path.join(RESULTS_DIR, '04_tonic_drive_5pct'), **_):
    os.makedirs(out_dir, exist_ok=True)
    net = BalancedNetwork(NetworkConfig())
    neuron, proto, chr2 = NeuronConfig(), ProtocolConfig(), ChR2Config(i_max=i_max)
    drives = np.arange(d_min, d_max + 1)
    d_ctx = np.arange(0, 21)
    a, b = proto.analysis_win
    n_stim = int(round(pct / 100.0 * net.N_exc))
    pct_str, tag = f'{pct:g} %', f'{pct:g}pct'

    keys = ('osc_on', 'osc_ctrl', 'osc_post', 'lam_on', 'lam_ctrl', 'f_on',
            'rate_lit', 'rate_unlitE', 'rate_I', 'rate_ctrl')
    S = {k: np.zeros((drives.size, n_draws)) for k in keys}
    for i, D in enumerate(drives):
        for s in range(n_draws):
            op = pick_random_exc(net.N_exc, n_stim, 1000 + s)
            unlitE = np.setdiff1d(np.arange(net.N_exc), op)
            R = run_tonic_opsin(net.W, op, float(D), proto, neuron, chr2, seed=100 + s, light=True)
            Rc = run_tonic_opsin(net.W, op, float(D), proto, neuron, chr2, seed=100 + s, light=False)
            m, mc = compute_metrics(R, op, (a, b)), compute_metrics(Rc, op, (a, b))
            S['osc_on'][i, s] = m['osc_sd']; S['osc_ctrl'][i, s] = mc['osc_sd']
            S['osc_post'][i, s] = compute_metrics(R, op, proto.post_win)['osc_sd']
            S['lam_on'][i, s] = lam_from_rates(net.W, R, (a, b), neuron).real
            S['lam_ctrl'][i, s] = lam_from_rates(net.W, Rc, (a, b), neuron).real
            S['f_on'][i, s] = m['peak_freq']
            S['rate_lit'][i, s] = R[op, a:b].mean()
            S['rate_unlitE'][i, s] = R[unlitE, a:b].mean()
            S['rate_I'][i, s] = R[net.N_exc:, a:b].mean()
            S['rate_ctrl'][i, s] = Rc[:, a:b].mean()

    ctx = np.zeros((d_ctx.size, 5))
    ctx_lam = np.zeros(d_ctx.size)
    for i, D in enumerate(d_ctx):
        for s in range(5):
            Rc = run_tonic_opsin(net.W, [], float(D), proto, neuron, chr2, seed=500 + s, light=False)
            ctx[i, s] = compute_metrics(Rc, [], (a, b))['osc_sd']
            if s == 0:
                ctx_lam[i] = lam_from_rates(net.W, Rc, (a, b), neuron).real

    amps = np.array([20, 40, 80, 160, 320])
    grid = np.zeros((drives.size, amps.size))
    for i, D in enumerate(drives):
        for k, A in enumerate(amps):
            grid[i, k] = np.mean([
                compute_metrics(
                    run_tonic_opsin(net.W, pick_random_exc(net.N_exc, n_stim, 1000 + s), float(D),
                                    proto, neuron, chr2, i_max=float(A), seed=100 + s),
                    pick_random_exc(net.N_exc, n_stim, 1000 + s), (a, b)
                )['osc_sd']
                for s in range(10)
            ])

    R_ref, _ = run_pulse(net.W, np.arange(net.N), proto.i_ref, proto, neuron, chr2, seed=5, opsin=False)
    osc_ref = compute_metrics(R_ref, [], (a, b))['osc_sd']

    # ---- Figure 1: examples
    ex = pick_random_exc(net.N_exc, n_stim, 1000)
    unlit_ex = np.setdiff1d(np.arange(net.N_exc), ex)
    n_blk = int(np.ceil(drives.size / 5))
    fig = plt.figure(figsize=(20, 6 * n_blk))
    gs = fig.add_gridspec(2 * n_blk, 5, height_ratios=[1, 0.6] * n_blk, hspace=0.5, wspace=0.25)
    for j, D in enumerate(drives):
        r0, c = 2 * (j // 5), j % 5
        R = run_tonic_opsin(net.W, ex, float(D), proto, neuron, chr2, seed=100)
        ax = fig.add_subplot(gs[r0, c])
        plot_raster(ax, R, net.phase_sort, proto, opsin_idx=ex, vmax=50.0)
        ax.set_title(f'tonic drive D = {D}', fontsize=11, fontweight='bold')
        if c == 0:
            ax.set_ylabel('Neuron (phase-sorted)')
        ax = fig.add_subplot(gs[r0 + 1, c])
        shade_light(ax, proto); plot_light_trace(ax, proto, chr2, 50.0)
        ax.plot(proto.tvec, R[ex].mean(0), c='#ff4f4f', lw=1, label='lit E')
        ax.plot(proto.tvec, R[unlit_ex].mean(0), c='#e09b22', lw=1, label='unlit E')
        ax.plot(proto.tvec, R[net.N_exc:].mean(0), c='#3a6fd8', lw=1, label='I')
        ax.set_xlim(0, proto.T / 1000.0); ax.set_ylim(0, 50); ax.set_xlabel('Time (s)')
        if c == 0:
            ax.set_ylabel('Mean rate (Hz)')
        if j == 0:
            ax.legend(fontsize=6, frameon=False, ncol=2, loc='upper right')
    fig.suptitle(f'Tonic drive D to all cells + ChR2 in random {pct_str} of E cells '
                 f'(peak I = {i_max:g}, desensitizing, light 1-6 s). Paper\'s rhythm drive: D = 20.',
                 fontsize=12, y=0.995)
    save_fig(fig, os.path.join(out_dir, f'tonic_examples_{tag}'), dpi=130)

    # ---- Figure 2: summary
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.3))
    ax = axs[0]
    ax.plot(d_ctx, ctx.mean(1), c='.5', marker='.', label='tonic drive only (no light)')
    for k, c, lab in [('osc_on', '#ff4f4f', f'light on {pct_str} E'), ('osc_post', '#8a2be2', 'after light off')]:
        ax.errorbar(drives, S[k].mean(1), S[k].std(1), c=c, marker='o', ms=4, capsize=2, label=lab)
    ax.axvspan(drives[0] - 0.5, drives[-1] + 0.5, color='#4fa3ff', alpha=0.1, lw=0)
    ax.axvline(20, c='k', ls=':', lw=0.8)
    ax.set_yscale('log'); ax.set_xlabel('Tonic drive D (all cells)')
    ax.set_ylabel('Rhythmicity of unlit cells (SD, Hz)')
    ax.set_title(f'Rhythm vs tonic drive ({n_draws} opsin draws)', fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc='upper left')

    ax = axs[1]
    ax.plot(d_ctx, ctx_lam, c='.5', marker='.', label='tonic only')
    ax.errorbar(drives, S['lam_on'].mean(1), S['lam_on'].std(1), c='#ff4f4f', marker='o', ms=4, capsize=2, label='tonic + light')
    ax.axhline(1, c='k', ls='--', lw=0.8)
    ax.set_xlabel('Tonic drive D'); ax.set_ylabel('Re λ_max (linearised at mean state)')
    ax.set_title('Distance from rhythm threshold (Re λ = 1)', fontsize=10); ax.legend(fontsize=7, frameon=False)

    ax = axs[2]
    for k, c, lab in [('rate_lit', '#ff4f4f', 'lit E'), ('rate_unlitE', '#e09b22', 'unlit E'),
                      ('rate_I', '#3a6fd8', 'I'), ('rate_ctrl', '.5', 'all cells, no light')]:
        ax.errorbar(drives, S[k].mean(1), S[k].std(1), c=c, marker='o', ms=4, capsize=2, label=lab)
    ax.set_xlabel('Tonic drive D'); ax.set_ylabel('Mean rate during light (Hz)')
    ax.set_title('Rates', fontsize=10); ax.legend(fontsize=7, frameon=False)

    plot_heatmap_grid(axs[3], grid, [str(x) for x in amps], [str(d) for d in drives],
                      'Peak photocurrent I', 'Tonic drive D',
                      f'Tonic × light intensity\n(reference rhythm at D = 20: {osc_ref:.1f})')
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'tonic_summary_{tag}'))

    q = lambda v: f'{v.mean():.2f}±{v.std():.2f}'
    lines = [
        f'{pct_str} opsin E ({n_stim} cells), ChR2 peak I = {i_max}, desens tau={chr2.tau_des} ms, '
        f'ss={chr2.ss_frac}, light {proto.stim_on}-{proto.stim_off} ms; paper drive = {proto.i_ref} (osc {osc_ref:.2f})', '',
        'D: osc light-on | osc no-light | osc post | Re lam on | Re lam ctrl | rate lit / unlitE / I | ctrl rate',
    ]
    for i, D in enumerate(drives):
        lines.append(
            f'{D:2d}: {q(S["osc_on"][i])} (max {S["osc_on"][i].max():.2f}) | {q(S["osc_ctrl"][i])} | '
            f'{q(S["osc_post"][i])} | {q(S["lam_on"][i])} | {q(S["lam_ctrl"][i])} | '
            f'{S["rate_lit"][i].mean():.1f} / {S["rate_unlitE"][i].mean():.1f} / {S["rate_I"][i].mean():.1f} | '
            f'{S["rate_ctrl"][i].mean():.1f}'
        )
    lines += ['', 'context, tonic only D=0..20: osc SD | Re lam (sim mean state)']
    lines += [f'  {D:2d}: {ctx[i].mean():.2f} | {ctx_lam[i]:.3f}' for i, D in enumerate(d_ctx)]
    lines += ['', 'tonic x peak intensity, cols ' + str(amps.tolist())]
    lines += [f'  {D:2d}: ' + ' '.join(f'{g:.2f}' for g in grid[i]) for i, D in enumerate(drives)]
    txt = '\n'.join(lines); print(txt)
    with open(os.path.join(out_dir, f'tonic_summary_{tag}.txt'), 'w') as fh:
        fh.write(txt + '\n')
