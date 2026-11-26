"""Experiment 2: Random vs high-drive / most-active / model-best E-cell stimulation."""
import os
import numpy as np
import matplotlib.pyplot as plt

from ..config import ChR2Config, NetworkConfig, NeuronConfig, ProtocolConfig, RESULTS_DIR
from ..network import BalancedNetwork
from ..opsin import find_model_best_exc, pick_random_exc, rank_reference_drive
from ..simulate import run_pulse
from ..analysis import compute_metrics, lam_from_rates, nerve_readout
from ..plotting import plot_light_trace, plot_nerve, plot_raster, save_fig, shade_light

DESCRIPTION = 'Compare random E cells vs top reference-rhythm contributors ("high-drive"), most-active, and greedy model-best'


def add_args(parser):
    parser.add_argument('--pct', type=float, default=15.0, help='Percent of E cells with opsin (default: 15)')
    parser.add_argument('--i-max', type=float, default=80.0, help='Peak photocurrent amplitude (default: 80)')
    parser.add_argument('--no-desens', action='store_true', help='Disable ChR2 desensitization')
    parser.add_argument('--out-dir', type=str, default=os.path.join(RESULTS_DIR, '02_random_vs_highdrive'))


def run(pct: float = 15.0, i_max: float = 80.0, no_desens: bool = False,
        out_dir: str = os.path.join(RESULTS_DIR, '02_random_vs_highdrive'), **_):
    os.makedirs(out_dir, exist_ok=True)
    net = BalancedNetwork(NetworkConfig())
    neuron = NeuronConfig()
    proto = ProtocolConfig()
    chr2 = ChR2Config(i_max=i_max, desensitize=not no_desens)
    win = proto.analysis_win

    n_stim = int(round(pct / 100.0 * net.N_exc))
    pct_str, tag = f'{pct:g} %', f'{pct:g}pct'

    R_ref, _ = run_pulse(net.W, np.arange(net.N), proto.i_ref, proto, neuron, chr2, seed=5, opsin=False)
    contrib_sd, _, hi_idx, act_idx = rank_reference_drive(net, proto, neuron, n_stim)
    rand_idx = pick_random_exc(net.N_exc, n_stim, seed=0)

    cache = os.path.join(out_dir, f'model_best_I{int(i_max)}_{tag}.npy')
    if os.path.exists(cache) and chr2.desensitize:
        best_idx = np.load(cache)
    else:
        best_idx = find_model_best_exc(net, n_stim, i_max, chr2, neuron)
        if chr2.desensitize:
            np.save(cache, best_idx)

    R_rand, _ = run_pulse(net.W, rand_idx, i_max, proto, neuron, chr2, seed=5)
    R_hi, _ = run_pulse(net.W, hi_idx, i_max, proto, neuron, chr2, seed=5)
    R_rest, _ = run_pulse(net.W, [], 0.0, proto, neuron, chr2, seed=5)
    m_ref = compute_metrics(R_ref, hi_idx, win)
    m_rest = compute_metrics(R_rest, hi_idx, win)

    n_rand, n_noise = 50, 20
    rand_sd, rand_lam = [], []
    for s in range(n_rand):
        idx = pick_random_exc(net.N_exc, n_stim, 1000 + s)
        R, _ = run_pulse(net.W, idx, i_max, proto, neuron, chr2, seed=100 + s)
        rand_sd.append(compute_metrics(R, idx, win)['osc_sd'])
        rand_lam.append(lam_from_rates(net.W, R, win, neuron).real)
    rand_sd, rand_lam = np.array(rand_sd), np.array(rand_lam)

    def fixed_stats(idx):
        sd, lam = [], []
        for s in range(n_noise):
            R, _ = run_pulse(net.W, idx, i_max, proto, neuron, chr2, seed=100 + s)
            sd.append(compute_metrics(R, idx, win)['osc_sd'])
            lam.append(lam_from_rates(net.W, R, win, neuron).real)
        return np.array(sd), np.array(lam)

    hi_sd, hi_lam = fixed_stats(hi_idx)
    act_sd, act_lam = fixed_stats(act_idx)
    best_sd, best_lam = fixed_stats(best_idx)

    amps = np.array([10, 20, 40, 80, 160, 320])
    sw = {k: np.zeros((amps.size, 10)) for k in ('rand', 'hi')}
    for i, A in enumerate(amps):
        for s in range(10):
            idx = pick_random_exc(net.N_exc, n_stim, 2000 + s)
            R, _ = run_pulse(net.W, idx, float(A), proto, neuron, chr2, seed=200 + s)
            sw['rand'][i, s] = compute_metrics(R, idx, win)['osc_sd']
            R, _ = run_pulse(net.W, hi_idx, float(A), proto, neuron, chr2, seed=200 + s)
            sw['hi'][i, s] = compute_metrics(R, hi_idx, win)['osc_sd']

    conn_seeds = np.arange(2, 12)
    net_res = np.zeros((conn_seeds.size, 3))
    for j, cs in enumerate(conn_seeds):
        n2 = net.with_seed(int(cs))
        _, _, hi2, _ = rank_reference_drive(n2, proto, neuron, n_stim, noise_seeds=(5,))
        R, _ = run_pulse(n2.W, hi2, i_max, proto, neuron, chr2, seed=5)
        net_res[j, 1] = compute_metrics(R, hi2, win)['osc_sd']
        r = []
        for s in range(10):
            idx = pick_random_exc(n2.N_exc, n_stim, 4000 + s)
            R, _ = run_pulse(n2.W, idx, i_max, proto, neuron, chr2, seed=400 + s)
            r.append(compute_metrics(R, idx, win)['osc_sd'])
        net_res[j, 0] = np.mean(r)
        R, _ = run_pulse(n2.W, np.arange(n2.N), proto.i_ref, proto, neuron, chr2, seed=5, opsin=False)
        net_res[j, 2] = compute_metrics(R, hi2, win)['osc_sd']

    # ---- Figure
    C_R, C_H, C_A, C_B = '#ff4f4f', '#8a2be2', '#e09b22', '#2a9d8f'
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 0.8, 0.7, 1.1], hspace=0.65, wspace=0.45)
    non_hi = np.setdiff1d(np.arange(net.N_exc), hi_idx)
    non_rand = np.setdiff1d(np.arange(net.N_exc), rand_idx)

    ax = fig.add_subplot(gs[0, 0:2])
    plot_raster(ax, R_ref, net.phase_sort, proto, vmax=60.0)
    for idx, c, x in [(hi_idx, C_H, -0.1), (rand_idx, C_R, -0.35)]:
        rows = np.where(np.isin(net.phase_sort, idx))[0]
        ax.scatter(np.full(rows.size, x), rows + 0.5, marker='>', s=12, c=c, clip_on=False)
    ax.set_title('Reference: 5 s global drive (I = 20 to all cells)\n▶ high-drive E (purple) / random E (red)', fontsize=10)
    ax.set_ylabel('Neuron (phase-sorted)'); ax.set_xlabel('Time (s)')

    ax = fig.add_subplot(gs[0, 2:4])
    order = np.argsort(contrib_sd)[::-1]
    cols = np.array(['.8'] * net.N_exc, dtype=object)
    cols[np.isin(order, rand_idx)] = C_R; cols[np.isin(order, hi_idx)] = C_H
    ax.bar(np.arange(net.N_exc), contrib_sd[order], color=list(cols), width=1)
    ax.set_xlabel('E cells, ranked by contribution to the reference rhythm')
    ax.set_ylabel('Rhythmic modulation\nduring reference (SD, Hz)')
    ax.set_title('Who carries the reference rhythm?', fontsize=10)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_H), plt.Rectangle((0, 0), 1, 1, color=C_R)],
              labels=[f'high-drive {pct_str} (top {n_stim})', f'random {pct_str} (example)'], fontsize=8, frameon=False)

    ref_n = nerve_readout(net, R_ref, mode='limb')
    nerve_off = 1.2 * np.abs(ref_n['upper']).max()
    for col, (R, idx, non, c, lab) in enumerate([
        (R_rand, rand_idx, non_rand, C_R, f'Random {pct_str} E'),
        (R_hi, hi_idx, non_hi, C_H, f'High-drive {pct_str} E'),
    ]):
        sl = slice(2 * col, 2 * col + 2)
        ax = fig.add_subplot(gs[1, sl])
        plot_raster(ax, R, net.phase_sort, proto, opsin_idx=idx, vmax=60.0, marker_color=c, marker_x=-0.1)
        ax.set_title(f'{lab}, ChR2 peak I = {i_max:g}, 5 s', fontsize=10, color=c)
        ax.set_ylabel('Neuron (phase-sorted)')

        ax = fig.add_subplot(gs[2, sl])
        shade_light(ax, proto); plot_light_trace(ax, proto, chr2)
        ax.plot(proto.tvec, R[idx].mean(0), c=c, label='lit E')
        ax.plot(proto.tvec, R[non].mean(0), c='#e09b22', label='unlit E')
        ax.plot(proto.tvec, R[net.N_exc:].mean(0), c='#3a6fd8', label='I')
        ax.set_xlim(0, proto.T / 1000.0); ax.set_ylim(0, 75); ax.set_ylabel('Mean rate (Hz)')
        ax.legend(fontsize=7, frameon=False, loc='upper right', ncol=3)

        nr = nerve_readout(net, R, mode='limb')
        ax = fig.add_subplot(gs[3, sl])
        plot_nerve(ax, proto.tvec, nr['upper'], nr['lower'], nerve_off, nr['labels'], proto, lw=0.5)
        ax.set_xlabel('Time (s)'); ax.set_ylabel(nr['ylabel'])

    ax = fig.add_subplot(gs[4, 0])
    bp = ax.boxplot([rand_sd, hi_sd, act_sd, best_sd],
                    tick_labels=['random', 'high-\ndrive', 'most\nactive', 'model-\nbest'],
                    patch_artist=True, widths=0.6)
    for p, c in zip(bp['boxes'], [C_R, C_H, C_A, C_B]):
        p.set_facecolor(c); p.set_alpha(0.6)
    ax.axhline(m_ref['osc_sd'], c='k', ls=':', label='reference rhythm')
    ax.axhline(m_rest['osc_sd'], c='.6', ls='--', label='no light')
    ax.set_ylabel('Rhythmicity of unlit cells\n(temporal SD, Hz)'); ax.set_ylim(0, 10)
    ax.set_title(f'{pct_str} E ({n_stim} cells)\n({n_rand} random draws; {n_noise} noise seeds)', fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc='upper left')

    ax = fig.add_subplot(gs[4, 1])
    bp = ax.boxplot([rand_lam, hi_lam, act_lam, best_lam],
                    tick_labels=['random', 'high-\ndrive', 'most\nactive', 'model-\nbest'],
                    patch_artist=True, widths=0.6)
    for p, c in zip(bp['boxes'], [C_R, C_H, C_A, C_B]):
        p.set_facecolor(c); p.set_alpha(0.6)
    ax.axhline(1, c='k', ls='--', lw=0.8)
    ax.axhline(lam_from_rates(net.W, R_ref, win, neuron).real, c='k', ls=':', label='reference')
    ax.set_ylabel('Re λ_max during light'); ax.set_title('Distance from rhythm threshold (Re λ = 1)', fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = fig.add_subplot(gs[4, 2])
    for k, c, lab in [('rand', C_R, 'random'), ('hi', C_H, 'high-drive')]:
        ax.errorbar(amps, sw[k].mean(1), sw[k].std(1), marker='o', c=c, label=lab, capsize=2)
    ax.axhline(m_ref['osc_sd'], c='k', ls=':'); ax.axhline(m_rest['osc_sd'], c='.6', ls='--')
    ax.axvspan(i_max * 0.9, i_max * 1.1, color='#4fa3ff', alpha=0.3)
    ax.text(i_max, 9.5, f'peak {i_max:g}', ha='center', fontsize=7)
    ax.set_xscale('log'); ax.set_ylim(0, 10)
    ax.set_xlabel('Peak photocurrent I'); ax.set_ylabel('Rhythmicity (SD)')
    ax.set_title('Light intensity', fontsize=9); ax.legend(fontsize=7, frameon=False, loc='center left')

    ax = fig.add_subplot(gs[4, 3])
    for j in range(conn_seeds.size):
        ax.plot([0, 1], net_res[j, :2], c='.6', lw=0.8, marker='o', ms=3)
    ax.scatter(np.zeros(conn_seeds.size), net_res[:, 0], c=C_R, zorder=3)
    ax.scatter(np.ones(conn_seeds.size), net_res[:, 1], c=C_H, zorder=3)
    ax.scatter(np.full(conn_seeds.size, 2), net_res[:, 2], c='k', zorder=3)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['random\n(mean of 10)', 'high-\ndrive', 'reference'])
    ax.set_ylabel('Rhythmicity (SD)'); ax.set_ylim(0, 12)
    ax.set_title(f'{conn_seeds.size} other networks\n(100:100, C = 0.1)', fontsize=9)
    save_fig(fig, os.path.join(out_dir, f'opto_random_vs_highdrive_{tag}'))

    q = lambda x: f'{np.mean(x):.2f} +- {np.std(x):.2f}'
    lines = [
        f'{pct_str} E cells ({n_stim}); I_OPTO = {i_max}; light {proto.stim_on}-{proto.stim_off} ms; analysis {win}',
        f'high-drive E (top SD in reference): {hi_idx.tolist()}',
        f'most-active E (top mean rate):      {act_idx.tolist()}  overlap w/ high-drive: {len(np.intersect1d(hi_idx, act_idx))}',
        f'model-best E (greedy):              {best_idx.tolist()}  overlap w/ high-drive: {len(np.intersect1d(hi_idx, best_idx))}',
        f'random example:                     {rand_idx.tolist()}',
        f'reference osc SD = {m_ref["osc_sd"]:.2f}, no light = {m_rest["osc_sd"]:.2f}',
        f'osc SD  random {q(rand_sd)} | high-drive {q(hi_sd)} | most-active {q(act_sd)} | model-best {q(best_sd)}',
        f'Re lam  random {q(rand_lam)} | high-drive {q(hi_lam)} | most-active {q(act_lam)} | model-best {q(best_lam)}',
        f'high-drive osc SD percentile among random draws: {(rand_sd < hi_sd.mean()).mean() * 100:.0f}%',
        'intensity sweep osc SD (amp: random, high-drive): ' + '; '.join(
            f'{A}: {sw["rand"][i].mean():.2f}, {sw["hi"][i].mean():.2f}' for i, A in enumerate(amps)),
        'other networks (random, high-drive, reference): ' + '; '.join(
            f's{cs}: {r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f}' for cs, r in zip(conn_seeds, net_res)),
    ]
    txt = '\n'.join(lines); print(txt)
    with open(os.path.join(out_dir, f'opto_random_vs_highdrive_{tag}.txt'), 'w') as fh:
        fh.write(txt + '\n')
