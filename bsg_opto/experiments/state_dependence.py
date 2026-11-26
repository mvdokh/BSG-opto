"""Experiment 7: State dependence (quiescent vs aroused, uniform vs heterogeneous baselines)."""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from ..config import ChR2Config, NetworkConfig, NeuronConfig, ProtocolConfig, RESULTS_DIR
from ..network import BalancedNetwork
from ..opsin import draw_heterogeneous_baselines
from ..connectivity import make_batched_opsin_to_I_extra, scale_opsin_to_I_weights
from ..simulate import make_batched_ext, simulate_batch
from ..analysis import leading_real_eig, unlit_mean
from ..plotting import plot_raster, save_fig, shade_light

DESCRIPTION = 'Quiescent vs aroused baseline state (mu) and per-cell baseline spread (sigma), plus intermediate->aroused sweep figure'


def add_args(parser):
    parser.add_argument('--pct', type=float, default=10.0)
    parser.add_argument('--i-max', type=float, default=80.0)
    parser.add_argument('--n-draws', type=int, default=20)
    parser.add_argument('--sweep-only', action='store_true', help='Only generate the intermediate->aroused sweep figure')
    parser.add_argument('--no-sweep', action='store_true', help='Skip the intermediate->aroused sweep figure')
    parser.add_argument('--out-dir', type=str, default=os.path.join(RESULTS_DIR, '08_state_dependence'))


def run_sweep_figure(net: BalancedNetwork, pct: float = 10.0, i_max: float = 80.0,
                     n_draws: int = 20, out_dir: str = os.path.join(RESULTS_DIR, '08_state_dependence')):
    """Generate the 6-row intermediate -> aroused baseline sweep figure."""
    os.makedirs(out_dir, exist_ok=True)
    neuron, proto, chr2 = NeuronConfig(), ProtocolConfig(), ChR2Config(i_max=i_max)
    n_op = int(round(pct / 100.0 * net.N_exc))
    mus, sigmas, success_sd = [6, 8, 10, 11, 12, 13], [0, 4], 2.0

    cols, meta = [], []
    for i, mu in enumerate(mus):
        for j, sig in enumerate(sigmas):
            for s in range(n_draws):
                op, phi, z = draw_heterogeneous_baselines(net.N, net.N_exc, n_op, s)
                cols.append(dict(base=float(mu) + float(sig) * z, op=op, phi=phi, i_max=i_max))
                meta.append((i, j, s))
    snap_cols = [c for c, (i, j, s) in enumerate(meta) if s == 0]
    op0 = draw_heterogeneous_baselines(net.N, net.N_exc, n_op, 0)[0]
    un0 = np.setdiff1d(np.arange(net.N_exc), op0)
    rec = {'m': lambda R, V: np.concatenate([
        R[op0][:, snap_cols].mean(0), R[un0][:, snap_cols].mean(0), R[net.N_exc:][:, snap_cols].mean(0)
    ])}
    osc = np.zeros((len(mus), len(sigmas), n_draws, 2))
    EX = {}
    for li, light in enumerate((False, True)):
        out = simulate_batch(net.W, proto.T, make_batched_ext(net.N, cols, proto, chr2, light=light), len(cols),
                             neuron=neuron, seed=50, windows={'on': proto.analysis_win},
                             snap_cols=snap_cols, record=rec)
        for c, (i, j, s) in enumerate(meta):
            osc[i, j, s, li] = unlit_mean(out['on']['sd'][:, c], cols[c]['op'])
        EX[light] = (out['snaps'], out['rec']['m'])
    effect = osc[..., 1] - osc[..., 0]
    success = (osc[..., 1] >= success_sd) & (osc[..., 1] >= 2.0 * osc[..., 0])

    label = lambda mu: 'intermediate' if mu <= 8 else ('near threshold' if mu >= 13 else 'aroused')
    nr, nc, nS = len(mus), 2 * len(sigmas), len(snap_cols)
    fig = plt.figure(figsize=(4.2 * nc, 4.0 * nr))
    gs = fig.add_gridspec(3 * nr, nc, height_ratios=[1, 0.45, 0.38] * nr, hspace=0.15, wspace=0.25)
    for r, mu in enumerate(mus):
        for cc, (j, light) in enumerate([(jj, lt) for jj in range(len(sigmas)) for lt in (False, True)]):
            sig = sigmas[j]
            p = [k for k, c in enumerate(snap_cols) if meta[c][0] == r and meta[c][1] == j][0]
            snaps, m = EX[light]
            ax = fig.add_subplot(gs[3 * r, cc])
            plot_raster(ax, snaps[p], net.phase_sort, proto, vmax=50.0, light_lines=light)
            ex_osc = osc[r, j, 0, int(light)]
            if light:
                ttl = (f'σ = {sig}, LIGHT   example {ex_osc:.2f} Hz\n'
                       f'{n_draws} draws: effect {effect[r, j].mean():+.2f}, success {100 * success[r, j].mean():.0f}%')
            else:
                ttl = f'σ = {sig}, no light   example {ex_osc:.2f} Hz\n{n_draws} draws: mean {osc[r, j, :, 0].mean():.2f} Hz'
            ax.set_title(ttl, fontsize=8.5, color='#c0392b' if light else 'k', fontweight='bold' if r == 0 else 'normal')
            if cc == 0:
                ax.set_ylabel(f'{label(mu)}\n(μ = {mu})\n\nneuron', fontsize=10)
            ax.set_xticklabels([])

            ax = fig.add_subplot(gs[3 * r + 1, cc])
            if light:
                shade_light(ax, proto)
            ax.plot(proto.tvec, m[:, p], c='#ff4f4f', lw=1, label='opsin E')
            ax.plot(proto.tvec, m[:, nS + p], c='#e09b22', lw=1, label='other E')
            ax.plot(proto.tvec, m[:, 2 * nS + p], c='#3a6fd8', lw=1, label='I')
            ax.set_xlim(0, proto.T / 1000.0); ax.set_ylim(0, 50)
            if r == nr - 1:
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xticklabels([])
            if cc == 0:
                ax.set_ylabel('Hz')
            if r == 0 and cc == 0:
                ax.legend(fontsize=7, frameon=False, ncol=3, loc='upper right')
    fig.suptitle(f'Baseline (tonic) drive sweep, intermediate → aroused: same photostimulation ({pct:g} % E, random light, 1–6 s)\n'
                 f'μ = mean baseline drive (rhythm starts on its own at ≈13–14), σ = spread across cells', fontsize=11, y=1.0)
    save_fig(fig, os.path.join(out_dir, 'state_sweep_intermediate_aroused'), dpi=110)


def run(pct: float = 10.0, i_max: float = 80.0, n_draws: int = 20,
        sweep_only: bool = False, no_sweep: bool = False,
        out_dir: str = os.path.join(RESULTS_DIR, '08_state_dependence'), **_):
    os.makedirs(out_dir, exist_ok=True)
    net = BalancedNetwork(NetworkConfig())
    if sweep_only:
        run_sweep_figure(net, pct, i_max, n_draws, out_dir)
        return

    neuron, proto, chr2 = NeuronConfig(), ProtocolConfig(), ChR2Config(i_max=i_max)
    n_op = int(round(pct / 100.0 * net.N_exc))
    mus = np.array([0, 2, 4, 6, 8, 10, 11, 12, 13])
    sigmas = np.array([0, 2, 4, 6])
    ks = [1, 4]
    success_sd = 2.0
    ex_mu, ex_sig = [0, 6, 11], [0, 4]

    shape = (len(ks), mus.size, sigmas.size, n_draws)
    R_ = {q: np.zeros(shape + (2,)) for q in ('osc', 'post', 'rateE_pre', 'rate_unlitE', 'rate_I', 'rate_lit', 'lam')}
    EX = {}
    for a_, k in enumerate(ks):
        cols, meta, ops_list = [], [], []
        for i, mu in enumerate(mus):
            for j, sig in enumerate(sigmas):
                for s in range(n_draws):
                    op, phi, z = draw_heterogeneous_baselines(net.N, net.N_exc, n_op, s)
                    cols.append(dict(base=float(mu) + float(sig) * z, op=op, phi=phi, i_max=i_max))
                    ops_list.append(op)
                    meta.append((i, j, s))
        for li, light in enumerate((False, True)):
            t0 = time.time()
            snap_cols, record = [], None
            if k == 1:
                snap_cols = [c for c, (i, j, s) in enumerate(meta) if s == 0 and mus[i] in ex_mu and sigmas[j] in ex_sig]
                op0 = draw_heterogeneous_baselines(net.N, net.N_exc, n_op, 0)[0]
                un0 = np.setdiff1d(np.arange(net.N_exc), op0)
                record = {'m': lambda R, V, sc=snap_cols: np.concatenate([
                    R[op0][:, sc].mean(0), R[un0][:, sc].mean(0), R[net.N_exc:][:, sc].mean(0)
                ])}
            out = simulate_batch(
                net.W, proto.T, make_batched_ext(net.N, cols, proto, chr2, light=light), len(cols),
                neuron=neuron, seed=50 + a_,
                windows={'on': proto.analysis_win, 'post': proto.post_win, 'pre': proto.pre_win},
                snap_cols=snap_cols, record=record,
                extra_input=make_batched_opsin_to_I_extra(net.W, ops_list, net.N_exc, float(k)),
            )
            for c, (i, j, s) in enumerate(meta):
                op = cols[c]['op']; un = np.setdiff1d(np.arange(net.N_exc), op)
                idx = (a_, i, j, s, li)
                R_['osc'][idx] = unlit_mean(out['on']['sd'][:, c], op)
                R_['post'][idx] = unlit_mean(out['post']['sd'][:, c], op)
                R_['rateE_pre'][idx] = out['pre']['mean'][:net.N_exc, c].mean()
                R_['rate_unlitE'][idx] = out['on']['mean'][un, c].mean()
                R_['rate_I'][idx] = out['on']['mean'][net.N_exc:, c].mean()
                R_['rate_lit'][idx] = out['on']['mean'][op, c].mean()
                if s < 3:
                    Wk = scale_opsin_to_I_weights(net.W, op, net.N_exc, float(k))
                    R_['lam'][idx] = leading_real_eig(Wk, out['on']['Vmean'][:, c], neuron)
                else:
                    R_['lam'][idx] = np.nan
            if snap_cols:
                EX[light] = (snap_cols, meta, out['snaps'], out['rec']['m'])
            print(f'k={k} light={light}: {len(cols)} trials, {time.time() - t0:.0f}s', flush=True)

    osc_off, osc_on = R_['osc'][..., 0], R_['osc'][..., 1]
    effect = osc_on - osc_off
    success = (osc_on >= success_sd) & (osc_on >= 2.0 * osc_off)
    spont = osc_off >= success_sd

    ref_cols = [dict(base=20.0, op=draw_heterogeneous_baselines(net.N, net.N_exc, n_op, s)[0], phi=None) for s in range(5)]
    out_ref = simulate_batch(net.W, proto.T, make_batched_ext(net.N, ref_cols, proto, chr2, light=False), 5,
                             neuron=neuron, seed=3, windows={'on': proto.analysis_win}, snap_cols=[])
    osc_ref = np.mean([unlit_mean(out_ref['on']['sd'][:, c], ref_cols[c]['op']) for c in range(5)])

    # Figure 1: examples (clean layout with spacer rows)
    nr, nc = len(ex_mu), 2 * len(ex_sig)
    fig = plt.figure(figsize=(4.2 * nc, 5.6 * nr))
    gs = fig.add_gridspec(3 * nr, nc, height_ratios=[1, 0.45, 0.35] * nr, hspace=0.15, wspace=0.25)
    state_name = {0: 'quiescent', 6: 'intermediate', 11: 'aroused'}
    for r, mu in enumerate(ex_mu):
        for cc, (sig, light) in enumerate([(sg, lt) for sg in ex_sig for lt in (False, True)]):
            sc, meta_ex, snaps, m = EX[light]
            pos = [p for p, c in enumerate(sc) if mus[meta_ex[c][0]] == mu and sigmas[meta_ex[c][1]] == sig][0]
            ax = fig.add_subplot(gs[3 * r, cc])
            plot_raster(ax, snaps[pos], net.phase_sort, proto, vmax=50.0, light_lines=light)
            if r == 0:
                ax.set_title(f'σ = {sig}, {"LIGHT" if light else "no light"}', fontsize=11,
                             fontweight='bold', color='#c0392b' if light else 'k')
            if cc == 0:
                ax.set_ylabel(f'{state_name[mu]}\n(μ = {mu})\n\nneuron', fontsize=10)
            ax.set_xticklabels([])

            ax = fig.add_subplot(gs[3 * r + 1, cc])
            if light:
                shade_light(ax, proto)
            n_ = len(sc)
            ax.plot(proto.tvec, m[:, pos], c='#ff4f4f', lw=1, label='opsin E')
            ax.plot(proto.tvec, m[:, n_ + pos], c='#e09b22', lw=1, label='other E')
            ax.plot(proto.tvec, m[:, 2 * n_ + pos], c='#3a6fd8', lw=1, label='I')
            ax.set_xlim(0, proto.T / 1000.0); ax.set_ylim(0, 50)
            if r == nr - 1:
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xticklabels([])
            if cc == 0:
                ax.set_ylabel('Hz')
            if r == 0 and cc == 0:
                ax.legend(fontsize=7, frameon=False, ncol=3, loc='upper right')
    fig.suptitle(f'Same photostimulation ({pct:g} % E, random light, 1–6 s) in different network states\n'
                 f'μ = mean baseline drive, σ = spread of baselines across cells', fontsize=12, y=1.0)
    save_fig(fig, os.path.join(out_dir, 'state_examples'), dpi=120)

    # Figure 2: summary
    cm = plt.cm.viridis(np.linspace(0, 0.85, sigmas.size))
    fig, axs = plt.subplots(2, 3, figsize=(18, 9.5))
    ax = axs[0, 0]
    for c, (j, sig) in zip(cm, enumerate(sigmas)):
        ax.plot(mus, osc_on[0, :, j].mean(-1), c=c, marker='o', ms=4, label=f'σ = {sig}, light')
        ax.plot(mus, osc_off[0, :, j].mean(-1), c=c, ls=':', lw=1.2)
    ax.plot([], [], c='k', ls=':', label='dotted: no light')
    ax.axhline(osc_ref, c='k', ls='--', lw=0.8, label=f'reference rhythm ({osc_ref:.1f})')
    ax.set_yscale('log'); ax.set_xlabel('Baseline drive μ  (quiescent → aroused)')
    ax.set_ylabel('Rhythmicity of unlit cells (SD, Hz)'); ax.set_title('Same light, different states', fontsize=10)
    ax.legend(fontsize=6, frameon=False, ncol=2, loc='upper left', bbox_to_anchor=(0.0, 0.92))

    ax = axs[0, 1]
    for c, (j, sig) in zip(cm, enumerate(sigmas)):
        e = effect[0, :, j]
        ax.errorbar(mus + 0.08 * j, e.mean(-1), e.std(-1), c=c, marker='o', ms=4, capsize=2, label=f'σ = {sig}')
    ax.axhline(0, c='k', lw=0.6)
    ax.set_xlabel('Baseline drive μ'); ax.set_ylabel('Light effect (light − no light, Hz)')
    ax.set_title('How much the light adds', fontsize=10); ax.legend(fontsize=7, frameon=False)

    ax = axs[0, 2]
    for c, (j, sig) in zip(cm, enumerate(sigmas)):
        ax.plot(mus, 100 * success[0, :, j].mean(-1), c=c, marker='o', ms=4, label=f'σ = {sig}: light evokes rhythm')
        ax.plot(mus, 100 * spont[0, :, j].mean(-1), c=c, ls=':', lw=1.2)
    ax.plot([], [], c='k', ls=':', label='dotted: rhythm already there w/o light')
    ax.set_xlabel('Baseline drive μ'); ax.set_ylabel('% of trials'); ax.set_ylim(-3, 103)
    ax.set_title(f'Success rate (rhythmicity ≥ {success_sd:g} and ≥ 2× no light)', fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    ax = axs[1, 0]
    for c, (j, sig) in zip(cm, enumerate(sigmas)):
        ax.plot(R_['rateE_pre'][0, :, j, :, 0].mean(-1), effect[0, :, j].mean(-1), c=c, marker='o', ms=4, label=f'σ = {sig}')
    ax.axhline(0, c='k', lw=0.6)
    ax.set_xlabel('Baseline E firing rate before light (Hz)  — measurable in vivo')
    ax.set_ylabel('Light effect (Hz)'); ax.set_title('Predicting the light effect from baseline activity', fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    for ax, a_ in [(axs[1, 1], 0), (axs[1, 2], 1)]:
        M = effect[a_].mean(-1)
        v = max(abs(effect.mean(-1)).max(), 0.5)
        im = ax.imshow(M.T, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v)
        ax.set_xticks(range(mus.size)); ax.set_xticklabels(mus); ax.set_yticks(range(sigmas.size)); ax.set_yticklabels(sigmas)
        for i in range(mus.size):
            for j in range(sigmas.size):
                ax.text(i, j, f'{M[i, j]:.1f}\n{100 * success[a_, i, j].mean():.0f}%', ha='center', va='center', fontsize=6.5)
        plt.colorbar(im, ax=ax, label='Light effect (Hz)')
        ax.set_xlabel('Baseline drive μ'); ax.set_ylabel('Baseline spread σ')
        ax.set_title(f'Light effect (and % success), opsin→I synapses ×{ks[a_]}', fontsize=10)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'state_summary'))

    lines = [f'N = {net.N}, opsin {n_op} E cells, random light max {i_max}; reference rhythm (D = 20) = {osc_ref:.2f}; '
             f'success = osc >= {success_sd} and >= 2x no-light', '']
    for a_, k in enumerate(ks):
        lines.append(f'=== opsin->I synapses x{k} ===')
        lines.append('mu sigma: E rate pre | osc no light | osc light (max) | effect | success % | spont % | '
                     'unlit E / I rate light | Re lam off/on')
        for i, mu in enumerate(mus):
            for j, sig in enumerate(sigmas):
                lines.append(
                    f'{mu:2d} {sig}: {R_["rateE_pre"][a_, i, j, :, 0].mean():5.1f} | {osc_off[a_, i, j].mean():.2f} | '
                    f'{osc_on[a_, i, j].mean():.2f} ({osc_on[a_, i, j].max():.2f}) | {effect[a_, i, j].mean():+.2f} | '
                    f'{100 * success[a_, i, j].mean():3.0f} | {100 * spont[a_, i, j].mean():3.0f} | '
                    f'{R_["rate_unlitE"][a_, i, j, :, 1].mean():.1f} / {R_["rate_I"][a_, i, j, :, 1].mean():.1f} | '
                    f'{np.nanmean(R_["lam"][a_, i, j, :, 0]):.2f}/{np.nanmean(R_["lam"][a_, i, j, :, 1]):.2f}'
                )
    txt = '\n'.join(lines); print(txt)
    with open(os.path.join(out_dir, 'state_summary.txt'), 'w') as fh:
        fh.write(txt + '\n')

    if not no_sweep:
        run_sweep_figure(net, pct, i_max, n_draws, out_dir)
