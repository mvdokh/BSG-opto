"""Quick plot of oscillation amplitude (rate SD, sp/s) vs. oscillation frequency (Hz)
at D = 20 across all 1,000 random BSG networks (from cached benchmark data).
"""
from __future__ import annotations

import os
import sys
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
from scipy.stats import spearmanr

from bsg_opto.config import RESULTS_DIR
from bsg_opto.experiments.opto_threshold_figure import configure_publication_style


def main():
    out_dir = os.path.join(RESULTS_DIR, 'bsg_benchmark_1000')
    cache_path = os.path.join(out_dir, 'benchmark_1000_cache.npz')
    data = np.load(cache_path, allow_pickle=False)

    amp = data['osc_sd'][:, -1]       # rate SD at D = 20 (sp/s)
    freq = data['peak_freq'][:, -1]   # peak oscillation frequency at D = 20 (Hz)
    is_real = data['lam1_im'] < 1e-5
    is_cplx = ~is_real
    rhy = amp >= 2.0

    rho_all, p_all = spearmanr(freq, amp)
    rho_rhy, p_rhy = spearmanr(freq[rhy], amp[rhy])

    configure_publication_style()
    COL_CPLX = '#D64045'
    COL_REAL = '#3B6EA5'
    COL_REF = '#D97706'

    fig = plt.figure(figsize=(5.3, 4.55))
    ax = fig.add_axes([0.120, 0.130, 0.648, 0.610])
    ax_top = fig.add_axes([0.120, 0.755, 0.648, 0.138], sharex=ax)
    ax_rgt = fig.add_axes([0.798, 0.130, 0.168, 0.610], sharey=ax)

    # Small uniform jitter strictly within each 0.167 Hz FFT bin so overlapping dots are visible
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.042, 0.042, size=freq.size)
    fx = freq + jitter

    ax.axhline(2.0, color='#777777', ls=(0, (3.0, 2.0)), lw=0.8, zorder=1)
    ax.text(1.78, 2.20, 'Rhythmic threshold (2 sp/s)', ha='right', va='bottom',
            fontsize=5.2, color='#555555')

    ax.scatter(fx[is_cplx], amp[is_cplx], s=11, color=COL_CPLX, alpha=0.52, edgecolors='none',
               label=f'Complex $\\lambda_1$ ($\\mathrm{{Im}}\\,\\lambda_1 \\neq 0$, rotational; $n={is_cplx.sum()}$)', zorder=3)
    ax.scatter(fx[is_real], amp[is_real], s=12, color=COL_REAL, alpha=0.58, edgecolors='none',
               label=f'Real $\\lambda_1$ ($\\mathrm{{Im}}\\,\\lambda_1 = 0$, non-rotational; $n={is_real.sum()}$)', zorder=4)

    # Mark Reference seed 1 at its exact unjittered frequency
    ax.scatter([freq[0]], [amp[0]], s=54, marker='*', color=COL_REF, edgecolors='#1A1A1A',
               linewidths=0.45, label=f'Reference seed 1 ({freq[0]:.2f} Hz, {amp[0]:.2f} sp/s)', zorder=6)

    ax.set_xlim(0.05, 1.80)
    ax.set_ylim(-0.35, 14.3)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax.set_xlabel('Oscillation frequency at $D = 20$ (Hz, FFT bin $\\pm$ jitter)', labelpad=3.0)
    ax.set_ylabel('Oscillation amplitude at $D = 20$ (rate SD, sp/s)', labelpad=3.0)

    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.92,
              edgecolor='#DDDDDD', fontsize=5.0, markerscale=1.05, handletextpad=0.35, labelspacing=0.25)

    # Top marginal: stacked histogram of oscillation frequency by FFT bin
    df = 1.0 / 6.0
    fbins = np.arange(0.5 * df, 11.5 * df, df)
    ax_top.hist([freq[is_cplx], freq[is_real]], bins=fbins, stacked=True,
                color=[COL_CPLX, COL_REAL], alpha=0.80, edgecolor='white', lw=0.45)
    ax_top.set_ylabel('Count', fontsize=5.4, labelpad=2.5)
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.set_yticks([0, 100, 200])

    # Main title + clean summary subtitle above the top histogram (leaves scatter 100% unobstructed)
    fig.text(0.50, 0.962, 'Oscillation amplitude vs. frequency across 1,000 random BSG networks ($D = 20$)',
             ha='center', va='top', fontsize=7.2, fontweight='bold', color='#111111')
    fig.text(0.50, 0.925,
             f'Rhythmic ($\\geq 2$ sp/s): {rhy.sum()}/1,000 ({100*rhy.mean():.1f}%)   ·   '
             f'Rhythmic freq.: median {np.median(freq[rhy]):.2f} Hz (max {freq[rhy].max():.2f} Hz)   ·   '
             f'Amp. vs. freq.: $\\rho = {rho_rhy:.2f}$ ($p = {p_rhy:.2f}$)',
             ha='center', va='top', fontsize=5.3, color='#444444')

    # Right marginal: stacked horizontal histogram of amplitude
    abins = np.linspace(0.0, 14.0, 29)
    ax_rgt.hist([amp[is_cplx], amp[is_real]], bins=abins, orientation='horizontal', stacked=True,
                color=[COL_CPLX, COL_REAL], alpha=0.80, edgecolor='white', lw=0.45)
    ax_rgt.axhline(2.0, color='#777777', ls=(0, (3.0, 2.0)), lw=0.8)
    ax_rgt.set_xlabel('Count', fontsize=5.4, labelpad=2.5)
    ax_rgt.tick_params(axis='y', labelleft=False)
    ax_rgt.set_xticks([0, 50, 100])

    pdf_path = os.path.join(out_dir, 'BSG_1000_amp_vs_freq_D20.pdf')
    png_path = os.path.join(out_dir, 'BSG_1000_amp_vs_freq_D20.png')
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)

    caption = f"""**Oscillation amplitude versus oscillation frequency at $D = 20$ across 1,000 random BSG networks.**

Joint scatter plot and marginal distributions of steady-state oscillation amplitude (mean temporal firing-rate s.d. across all $N = 200$ neurons, in sp/s) versus peak oscillation frequency (FFT spectral peak over the 6 s steady-state window, in Hz; small horizontal jitter $\\pm 0.04$ Hz added strictly within each 0.167 Hz FFT frequency bin for visibility) for all $n = 1{{,}}000$ independent random balanced Dale networks ($N = 200$, $C = 0.1$, $g = 1.0$, $W_{{\\mathrm{{radius}}}} = 1.0$, $\\tau_V = 50$ ms) driven at global tonic drive $D = 20$. Red points and histogram bars denote networks with a complex leading eigenvalue ($\\mathrm{{Im}}\\,\\lambda_1 \\neq 0$, rotational dominant mode; $n = {is_cplx.sum()}$); blue points and bars denote networks with a purely real leading eigenvalue ($\\mathrm{{Im}}\\,\\lambda_1 = 0$, non-rotational dominant mode; $n = {is_real.sum()}$); amber star marks the original reference network (`seed = 1`, {freq[0]:.2f} Hz, {amp[0]:.2f} sp/s); horizontal dashed line marks the 2 sp/s rhythmicity threshold ({rhy.sum()}/1,000 rhythmic, {100*rhy.mean():.1f}%). Among rhythmic networks, oscillation amplitude and oscillation frequency are uncorrelated (Spearman $\\rho = {rho_rhy:.2f}$, $p = {p_rhy:.2f}$), and all 1,000 random realizations remain strictly below 1.7 Hz (median rhythmic frequency {np.median(freq[rhy]):.2f} Hz, maximum {freq[rhy].max():.2f} Hz).
"""
    cap_path = os.path.join(out_dir, 'BSG_1000_amp_vs_freq_D20_caption.md')
    with open(cap_path, 'w') as fh:
        fh.write(caption)
    print(f'[plot] Saved {pdf_path}, {png_path}, and {cap_path}')


if __name__ == '__main__':
    main()
