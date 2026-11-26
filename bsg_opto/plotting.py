"""Reusable Matplotlib plotting helpers shared across all experiments."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import ChR2Config, ProtocolConfig
from .opsin import chr2_fixed_waveform, chr2_current, chr2_params, draw_variable_light


def shade_light(ax, proto: ProtocolConfig, color: str = '#4fa3ff', alpha: float = 0.15):
    ax.axvspan(proto.stim_on / 1000.0, proto.stim_off / 1000.0,
               color=color, alpha=alpha, lw=0)


def plot_light_trace(ax, proto: ProtocolConfig, chr2: ChR2Config, scale: float = 70.0):
    wave = chr2_fixed_waveform(proto.T, proto.stim_on, proto.stim_off, chr2)
    ax.plot(proto.tvec, scale * wave, c='k', lw=0.8, ls=':', label='ChR2 current (norm.)')


def plot_raster(ax, R: np.ndarray, order: np.ndarray, proto: ProtocolConfig,
                opsin_idx=None, phi=None, split_ei: int | None = None,
                vmin: float = 0.0, vmax: float = 50.0, light_lines: bool = True,
                marker_color: str = '#ff4f4f', marker_x: float = -0.15,
                interpolation: str = 'none'):
    """Plot a neuron rate heatmap sorted by `order`, with optional E/I split and opsin markers."""
    N = R.shape[0]
    t_end = proto.T / 1000.0
    ax.imshow(R[order], aspect='auto', vmin=vmin, vmax=vmax, interpolation=interpolation,
              extent=[0, t_end, N, 0], cmap='viridis')
    if split_ei is not None:
        ax.axhline(split_ei, c='w', lw=1.2)
        ax.text(t_end * 1.01, split_ei / 2.0, 'E', va='center', fontsize=9,
                fontweight='bold', clip_on=False)
        ax.text(t_end * 1.01, split_ei + (N - split_ei) / 2.0, 'I', va='center',
                fontsize=9, fontweight='bold', clip_on=False)
    if opsin_idx is not None and len(opsin_idx):
        inv = np.empty(N, dtype=int)
        inv[order] = np.arange(N)
        rows = inv[np.asarray(opsin_idx, dtype=int)]
        if phi is not None:
            colors = plt.cm.Reds(0.3 + 0.7 * np.asarray(phi))
            ax.scatter(np.full(rows.size, marker_x), rows + 0.5, marker='>', s=16,
                       c=colors, edgecolors='k', linewidths=0.3, clip_on=False)
        else:
            ax.scatter(np.full(rows.size, marker_x), rows + 0.5, marker='>', s=14,
                       c=marker_color, clip_on=False)
    if light_lines:
        for x in (proto.stim_on, proto.stim_off):
            ax.axvline(x / 1000.0, c='w', ls='--', lw=0.8)


def plot_nerve(ax, tvec: np.ndarray, upper: np.ndarray, lower: np.ndarray,
               offset: float, labels: tuple[str, str], proto: ProtocolConfig,
               shade: bool = True, lw: float = 0.4):
    if shade:
        shade_light(ax, proto)
    ax.plot(tvec, lower, c='green', alpha=0.7, lw=lw)
    ax.plot(tvec, offset + upper, c='blue', alpha=0.7, lw=lw)
    ax.set_xlim(0, proto.T / 1000.0)
    ax.set_ylim(-offset, 2.0 * offset)
    ax.set_yticks([0.0, offset])
    ax.set_yticklabels([labels[1], labels[0]], fontsize=7)


def plot_heatmap_grid(ax, grid: np.ndarray, x_labels, y_labels,
                      xlabel: str, ylabel: str, title: str,
                      highlight_col: int | None = None,
                      cbar_label: str = 'Rhythmicity (SD, Hz)'):
    vmax = max(1.0, float(grid.max()))
    im = ax.imshow(grid, origin='lower', aspect='auto', cmap='magma', vmin=0, vmax=vmax)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    for i in range(grid.shape[0]):
        for k in range(grid.shape[1]):
            ax.text(k, i, f'{grid[i, k]:.2f}', ha='center', va='center', fontsize=6,
                    color='w' if grid[i, k] < 0.6 * vmax else 'k')
    if highlight_col is not None:
        ax.add_patch(plt.Rectangle((highlight_col - 0.5, -0.5), 1, grid.shape[0],
                                   fill=False, ec='#4fa3ff', lw=2))
    plt.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)


def plot_light_model(out_dir: str, proto: ProtocolConfig, chr2: ChR2Config,
                     N_exc: int, n_stim: int, i_max: float | None = None,
                     seed_offset: int = 0):
    """4-panel diagnostic figure of the intensity-dependent ChR2 model."""
    i_max = chr2.i_max if i_max is None else float(i_max)
    pct = f'{100.0 * n_stim / N_exc:g} %'
    fig, axs = plt.subplots(1, 4, figsize=(19, 3.8))
    phis = [0.1, 0.25, 0.5, 0.75, 1.0]
    cm = plt.cm.viridis(np.linspace(0.15, 0.95, len(phis)))
    Icur = chr2_current(proto.T, proto.stim_on, proto.stim_off, phis, i_max=i_max, cfg=chr2)
    ax = axs[0]
    for c, p, I in zip(cm, phis, Icur):
        ax.plot(proto.tvec - proto.stim_on / 1000.0, I, c=c, label=f'φ = {p:g}')
    ax.set_xlim(-0.05, 0.6)
    ax.set_xlabel('Time from light on (s)')
    ax.set_ylabel('Photocurrent I')
    ax.set_title(f'Onset (max I = {i_max:g}): brighter light →\nbigger peak, faster & deeper sag', fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axs[1]
    for c, p, I in zip(cm, phis, Icur):
        ax.plot(proto.tvec, I, c=c)
    shade_light(ax, proto)
    ax.set_xlim(0, proto.T / 1000.0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Photocurrent I')
    ax.set_title(f'Whole {(proto.stim_off - proto.stim_on) / 1000.0:g} s pulse', fontsize=9)

    ax = axs[2]
    pp = np.linspace(0.01, 1.0, 200)
    ss, td = chr2_params(pp, chr2)
    ax.plot(pp, ss, c='k', label='steady / peak')
    ax.set_xlabel('Relative light φ')
    ax.set_ylabel('Steady / peak ratio')
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.plot(pp, td, c='#ff4f4f')
    ax2.set_ylabel('τ_des (ms)', color='#ff4f4f')
    ax.set_title('Intensity-dependent desensitization', fontsize=9)

    ax = axs[3]
    op, phi = draw_variable_light(N_exc, n_stim, 0, seed_offset=seed_offset)
    I0 = chr2_current(proto.T, proto.stim_on, proto.stim_off, phi, i_max=i_max, cfg=chr2)
    o = np.argsort(phi)
    ax.bar(np.arange(op.size) - 0.2, I0.max(1)[o], 0.4, color='.6', label='peak')
    ax.bar(np.arange(op.size) + 0.2, I0[:, proto.stim_off - 1][o], 0.4, color='#ff4f4f', label='plateau')
    ax.set_xticks(range(op.size))
    ax.set_xticklabels([f'{x:.2f}' for x in phi[o]], fontsize=7, rotation=45)
    ax.set_xlabel('φ of each opsin cell (example draw)')
    ax.set_ylabel('Photocurrent I')
    ax.set_title(f'Random light, {pct} E cells (max I = {i_max:g})', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'light_model'), dpi=150)


def save_fig(fig, base_path: str, dpi: int = 150):
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    fig.savefig(base_path + '.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(base_path + '.pdf', bbox_inches='tight')
    plt.close(fig)
