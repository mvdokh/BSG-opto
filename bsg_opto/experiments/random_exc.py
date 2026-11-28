"""Experiment 1: Photostimulating a random fraction of excitatory cells from rest."""
import os
from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt

from ..config import ChR2Config, NetworkConfig, NeuronConfig, ProtocolConfig, RESULTS_DIR
from ..network import BalancedNetwork
from ..opsin import pick_random_exc
from ..simulate import run_pulse
from ..analysis import compute_metrics, compute_pca, invert_transfer, jacobian_eigs, lam_from_rates, leading_eig, nerve_readout
from ..plotting import plot_light_trace, plot_nerve, plot_raster, save_fig, shade_light

DESCRIPTION = 'Random E-cell photostimulation from rest vs paper Fig. 2 global drive (+ robustness sweeps)'


def add_args(parser):
    parser.add_argument('--pct', type=float, default=15.0, help='Percent of E cells with opsin (default: 15)')
    parser.add_argument('--i-max', type=float, default=80.0, help='Peak photocurrent amplitude (default: 80)')
    parser.add_argument('--no-desens', action='store_true', help='Disable ChR2 desensitization (plain current step)')
    parser.add_argument('--n-draws', type=int, default=50, help='Random opsin draws for robustness (default: 50)')
    parser.add_argument('--out-dir', type=str, default=os.path.join(RESULTS_DIR, '01_random_exc'))


def run(pct: float = 15.0, i_max: float = 80.0, no_desens: bool = False,
        n_draws: int = 50, out_dir: str = os.path.join(RESULTS_DIR, '01_random_exc'), **_):
    os.makedirs(out_dir, exist_ok=True)
    net_cfg = NetworkConfig()
    neuron = NeuronConfig()
    proto = ProtocolConfig()
    chr2 = ChR2Config(i_max=i_max, desensitize=not no_desens)
    net = BalancedNetwork(net_cfg)

    n_stim = int(round(pct / 100.0 * net.N_exc))
    pct_str = f'{pct:g} %'
    tag = f'{pct:g}pct'
    stim_idx = pick_random_exc(net.N_exc, n_stim, seed=0)
    non_stim_E = np.setdiff1d(np.arange(net.N_exc), stim_idx)
    inh = np.arange(net.N_exc, net.N)

    R_opto, _ = run_pulse(net.W, stim_idx, i_max, proto, neuron, chr2, seed=5, opsin=True)
    R_glob, _ = run_pulse(net.W, np.arange(net.N), proto.i_ref, proto, neuron, chr2, seed=5, opsin=False)
    R_rest, _ = run_pulse(net.W, [], 0.0, proto, neuron, chr2, seed=5, opsin=True)

    wA, wB = proto.analysis_win
    m_rest, m_opto, m_glob = (compute_metrics(R, stim_idx, (wA, wB)) for R in (R_rest, R_opto, R_glob))
    m_post = compute_metrics(R_opto, stim_idx, proto.post_win)
    lam_rest, lam_opto, lam_glob = (lam_from_rates(net.W, R, (wA, wB), neuron) for R in (R_rest, R_opto, R_glob))
    lam_theta = leading_eig(net.W, np.full(net.N, neuron.threshold), neuron)

    # ---- Figure 1: main comparison
    rng_r = np.random.default_rng(1)
    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(4, 3, width_ratios=[2.4, 2.4, 1.1], hspace=0.75, wspace=0.3)
    ref_n = nerve_readout(net, R_glob, mode='limb')
    nerve_off = 1.2 * np.abs(ref_n['upper']).max()

    cols = [
        (R_opto, f'Random {n_stim}/{net.N_exc} E ({pct_str}), ChR2 peak I = {i_max:g}, 5 s'),
        (R_glob, f'Reference: all {net.N} cells, I = {proto.i_ref:g}, 5 s (paper Fig. 2)'),
    ]
    for col, (R, title) in enumerate(cols):
        ax = fig.add_subplot(gs[0, col])
        plot_raster(ax, R, net.phase_sort, proto,
                    opsin_idx=stim_idx if col == 0 else None, vmax=60.0, marker_x=-0.08)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Neuron (phase-sorted)')

        ax = fig.add_subplot(gs[1, col])
        shade_light(ax, proto)
        if col == 0:
            plot_light_trace(ax, proto, chr2)
            ax.plot(proto.tvec, R[stim_idx].mean(0), c='#ff4f4f', label='lit E')
            ax.plot(proto.tvec, R[non_stim_E].mean(0), c='#e09b22', label='unlit E')
        else:
            ax.plot(proto.tvec, R[:net.N_exc].mean(0), c='#e09b22', label='E')
        ax.plot(proto.tvec, R[inh].mean(0), c='#3a6fd8', label='I')
        ax.set_ylabel('Mean rate (Hz)')
        ax.legend(fontsize=7, frameon=False, loc='upper right')
        ax.set_xlim(0, proto.T / 1000.0); ax.set_ylim(0, 75)

        ax = fig.add_subplot(gs[2, col])
        shade_light(ax, proto)
        pool = non_stim_E if col == 0 else np.arange(net.N)
        ax.plot(proto.tvec, R[rng_r.choice(pool, 8, replace=False)].T, lw=0.7)
        ax.set_ylabel('Rate, 8 unlit cells' if col == 0 else 'Rate, 8 cells')
        ax.set_xlim(0, proto.T / 1000.0); ax.set_ylim(0, 75)

        nr = nerve_readout(net, R, mode='limb')
        ax = fig.add_subplot(gs[3, col])
        plot_nerve(ax, proto.tvec, nr['upper'], nr['lower'], nerve_off, nr['labels'], proto, lw=0.5)
        ax.set_xlabel('Time (s)'); ax.set_ylabel(nr['ylabel'])

    PCs, _, _ = compute_pca(R_glob, (wA, wB))
    mu = R_glob[:, wA:wB].mean(1, keepdims=True)
    ax = fig.add_subplot(gs[0:2, 2])
    for R, c, lab in [(R_glob, '.6', 'global drive'), (R_opto, '#ff4f4f', f'{pct_str} E opto')]:
        P = PCs[:, :2].T.real @ (R - mu)
        ax.plot(P[0, proto.stim_on:proto.stim_off], P[1, proto.stim_on:proto.stim_off], c=c, lw=0.7, label=lab)
    ax.set_title('Light-on trajectory\n(PC1/PC2 of the reference rhythm)', fontsize=9)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=7, frameon=False)
    ax.set_aspect('equal', 'datalim')

    ax = fig.add_subplot(gs[2:4, 2])
    for R, c, lab in [(R_rest, '.75', 'rest'), (R_opto, '#ff4f4f', f'{pct_str} E opto'), (R_glob, 'k', 'global drive')]:
        lam = jacobian_eigs(net.W, invert_transfer(R[:, wA:wB], neuron).mean(1), neuron)
        ax.scatter(lam.real, lam.imag, s=3, c=c, label=lab)
    ax.axvline(1, c='k', ls='--', lw=0.8)
    ax.set_xlabel('Re λ'); ax.set_ylabel('Im λ')
    ax.set_title("Spectrum of f'(V)·W during light\n(rhythm if Re λ > 1)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc='lower left'); ax.set_aspect('equal', 'datalim')
    save_fig(fig, os.path.join(out_dir, f'opto_main_{tag}'))

    # ---- Robustness sweeps
    draws = []
    for s in range(n_draws):
        idx = pick_random_exc(net.N_exc, n_stim, 1000 + s)
        R, _ = run_pulse(net.W, idx, i_max, proto, neuron, chr2, seed=100 + s)
        m = compute_metrics(R, idx, (wA, wB))
        m['post'] = compute_metrics(R, idx, proto.post_win)['osc_sd']
        m['lam'] = lam_from_rates(net.W, R, (wA, wB), neuron).real
        draws.append(m)
    oa = np.array([d['osc_sd'] for d in draws])
    po = np.array([d['post'] for d in draws])
    lr = np.array([d['lam'] for d in draws])

    amps = np.array([5, 10, 20, 40, 80, 160, 320])
    amp_res = np.zeros((amps.size, 10, 3))
    for i, a in enumerate(amps):
        for s in range(10):
            idx = pick_random_exc(net.N_exc, n_stim, 2000 + s)
            R, _ = run_pulse(net.W, idx, float(a), proto, neuron, chr2, seed=200 + s)
            m = compute_metrics(R, idx, (wA, wB))
            amp_res[i, s] = m['osc_sd'], m['rate_stim'], lam_from_rates(net.W, R, (wA, wB), neuron).real

    fracs = np.array([0.05, 0.15, 0.3, 0.5, 0.75, 1.0])
    frac_res = np.zeros((fracs.size, 10, 2))
    for i, f in enumerate(fracs):
        for s in range(10):
            idx = pick_random_exc(net.N_exc, int(round(f * net.N_exc)), 3000 + s)
            R, _ = run_pulse(net.W, idx, i_max, proto, neuron, chr2, seed=300 + s)
            frac_res[i, s] = compute_metrics(R, idx, (wA, wB))['osc_sd'], lam_from_rates(net.W, R, (wA, wB), neuron).real

    fig, axs = plt.subplots(1, 4, figsize=(16, 3.5))
    ax = axs[0]
    ax.boxplot([np.full(n_draws, m_rest['osc_sd']), oa, po, [m_glob['osc_sd']]],
               tick_labels=['no light', f'light on\n{pct_str} E', 'after light', 'global\ndrive'])
    ax.set_ylabel('Unlit amplitude\n(temporal rate SD, sp/s)')
    ax.set_title(f'{n_draws} random {pct_str} E draws', fontsize=9)

    ax = axs[1]
    ax.hist(lr, 15, color='#ff4f4f', alpha=0.8)
    ax.axvline(1, c='k', ls='--'); ax.axvline(lam_glob.real, c='.5', ls=':', label='global drive')
    ax.set_xlabel('Re λ_max during light'); ax.set_ylabel('# draws')
    ax.set_title('Stability (rhythm needs Re λ > 1)', fontsize=9); ax.legend(fontsize=7, frameon=False)

    for ax, x, res, xl, ttl in [
        (axs[2], amps, amp_res, 'Peak photocurrent I', f'{pct_str} E: light intensity'),
        (axs[3], 100 * fracs, frac_res, f'% of E cells lit (peak I = {i_max:g})', 'How many E cells?'),
    ]:
        ax.errorbar(x, res[:, :, 0].mean(1), res[:, :, 0].std(1), marker='o', c='#ff4f4f')
        ax.axhline(m_glob['osc_sd'], c='.5', ls=':', label='global drive')
        ax.axhline(m_rest['osc_sd'], c='.75', ls='--', label='no light')
        ax.set_xlabel(xl); ax.set_ylabel('Rhythmicity (SD)'); ax.set_title(ttl, fontsize=9); ax.set_ylim(0, 10)
        ax2 = ax.twinx(); ax2.plot(x, res[:, :, -1].mean(1), 'k.--', lw=0.8)
        ax2.axhline(1, c='k', lw=0.5); ax2.set_ylabel('Re λ_max (dashed)'); ax2.set_ylim(0.3, 1.25)
        ax2.text(x[0], 1.01, 'Re λ = 1 (rhythm threshold)', fontsize=6, va='bottom')
        ax.legend(fontsize=7, frameon=False, loc='center left')
    axs[2].set_xscale('log'); axs[2].axvline(i_max, c='#4fa3ff', lw=4, alpha=0.3)
    axs[2].text(i_max, axs[2].get_ylim()[1] * 0.95, f' peak {i_max:g}', fontsize=7, va='top')
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'opto_robustness_{tag}'))

    fmt = lambda m: ', '.join(f'{k}={v:.2f}' for k, v in m.items())
    lines = [
        f'{pct_str} E cells ({n_stim}); I_OPTO = {i_max}, desens={chr2.desensitize}, light {proto.stim_on}-{proto.stim_off} ms',
        f'Lit E cells (seed 0): {stim_idx.tolist()}',
        f'rest     : {fmt(m_rest)}  Re lam={lam_rest.real:.3f}',
        f'opto     : {fmt(m_opto)}  Re lam={lam_opto.real:.3f}',
        f'post     : {fmt(m_post)}',
        f'global   : {fmt(m_glob)}  lam={lam_glob:.3f} (V=theta approx: {lam_theta:.3f})',
        f'{n_draws} draws: osc SD {oa.mean():.2f}+-{oa.std():.2f}, post {po.mean():.2f}, '
        f'Re lam {lr.mean():.3f}+-{lr.std():.3f} (max {lr.max():.3f}), n(Re lam>1)={(lr > 1).sum()}',
        f'rate lit E {np.mean([d["rate_stim"] for d in draws]):.1f} Hz, '
        f'unlit {np.mean([d["rate_nonstim"] for d in draws]):.1f} Hz',
        'amp sweep (amp: oscSD, rate_lit, Re lam): ' + '; '.join(
            f'{a}: {amp_res[i,:,0].mean():.2f}, {amp_res[i,:,1].mean():.1f}, {amp_res[i,:,2].mean():.3f}'
            for i, a in enumerate(amps)),
        'frac sweep (frac: oscSD, Re lam): ' + '; '.join(
            f'{f:.2f}: {frac_res[i,:,0].mean():.2f}, {frac_res[i,:,1].mean():.3f}' for i, f in enumerate(fracs)),
    ]
    txt = '\n'.join(lines); print(txt)
    with open(os.path.join(out_dir, f'opto_summary_{tag}.txt'), 'w') as fh:
        fh.write(txt + '\n')
