#!/usr/bin/env python
"""
:Summary:
All plotting routines for EISRAD: radar plot, metric bar plot,
manual-vs-auto scatter, Dice-vs-volume heatmap, and lesion detection curve.

:Authors:
    Markus D. Schirmer
    MGH / Harvard Medical School

:License: MIT
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, MaxNLocator
from math import pi
from scipy.stats import pearsonr, spearmanr, rankdata

BLUE = 'royalblue'
TEAL = '#2a9d8f'

def _corr(func, x, y):
    """Correlation coefficient, or NaN when undefined (< 2 or constant values)."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return func(x, y)[0]

def _intermediate_ticks(min_val, max_val, n=4):
    return list(np.linspace(min_val, max_val, num=n+2)[1:-1])

def setup_radar_axes(ax, measures):
    """
    Polar axes for a radar plot: 0-1 radius, one labelled spoke per measure
    (labels aligned away from the circle), ring numbers between the first
    two spokes. Returns the spoke angles.
    """
    N = len(measures)
    angles = [n/float(N)*2*pi for n in range(N)]
    ax.set_ylim(0, 1)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_xticks(angles)
    ax.set_xticklabels(measures, fontsize=12)
    # Align labels away from the circle so long names do not overlap it
    for lab, ang in zip(ax.get_xticklabels(), angles):
        c = np.cos(ang)
        lab.set_horizontalalignment('left' if c > 0.1 else
                                    'right' if c < -0.1 else 'center')
    ax.tick_params(axis='x', pad=2)
    # Ring numbers halfway between the first two spokes (Dice and Jaccard)
    ax.set_rlabel_position(180.0 / N)
    return angles

def plot_radar(values, colourmap, measures, info, outfile='polar_results.png',
               show_points=True, density=False, density_bin=0.1,
               density_label='% of cases'):
    """
    Radar plot of multi-metric comparisons: median line, IQR band, a table
    of median / quartiles and, with `show_points`, one dot per case coloured
    by manual lesion volume (with colorbar). With `density`, each spoke
    instead shows a heatmap strip: the % of cases (per spoke) in each value
    bin of width `density_bin`, under the median line and IQR band.
    Missing values (NaN, e.g. boundary measures of missed lesions) are left
    out of the statistics.
    """
    values = np.asarray(values, dtype=float)
    N = len(measures)
    angles = [n/float(N)*2*pi for n in range(N)]
    angles_closed = angles + [angles[0]]

    # Compute median and IQR
    medians = np.nanmedian(values, axis=0)
    q1      = np.nanpercentile(values, 25, axis=0)
    q3      = np.nanpercentile(values, 75, axis=0)
    med_closed = np.concatenate([medians, [medians[0]]])
    q1_closed  = np.concatenate([q1,      [q1[0]]])
    q3_closed  = np.concatenate([q3,      [q3[0]]])

    # Make the figure wider: 12" wide instead of 9"
    fig = plt.figure(figsize=(12, 6), dpi=300, constrained_layout=True)
    if show_points or density:
        gs = fig.add_gridspec(1, 3, width_ratios=[2, 0.45, 1.1])
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1.1])

    # ── Radar subplot (left 2/3) ──
    ax = fig.add_subplot(gs[0, 0], polar=True)
    setup_radar_axes(ax, measures)
    if density:
        # Small hole in the centre so the spokes' 0-bins do not overlap
        hole = 0.15
        ax.set_rorigin(-hole)
        # Extra unlabelled rings so every bin boundary sits on a ring
        ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(
            np.arange(density_bin, 1.0, density_bin)))
        ax.grid(True, which='minor', linestyle='--', linewidth=0.5,
                alpha=0.5)
        _radar_density(fig, gs, ax, values, angles, density_bin, density_label)
    ax.fill_between(angles_closed, q1_closed, q3_closed,
                    color='royalblue', alpha=0.2, zorder=2)
    ax.plot(angles_closed, med_closed, color='royalblue', linewidth=2,
            zorder=4)

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    if density:
        ax.set_axisbelow(True)     # rings under the segments
        ax.xaxis.grid(False)       # no spoke lines; the segments mark them
    if show_points:
        _radar_points(fig, gs, ax, values, colourmap, angles, info)

    # ── Table subplot (right) ──
    tax = fig.add_subplot(gs[0, -1])
    tax.axis('off')
    col_labels = ['Median','25%','75%']
    cell_text  = [
        [f"{m:.2f}", f"{l:.2f}", f"{h:.2f}"]
        for m,l,h in zip(medians, q1, q3)
    ]
    tbl = tax.table(
        cellText=cell_text,
        rowLabels=measures,
        colLabels=col_labels,
        cellLoc='center', rowLoc='center',
        loc='center',
        bbox=[0.25, 0.00, 0.75, 1.00]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)

    # Save/display
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    if info.get('display', False):
        plt.show()
    plt.close(fig)

def _radar_density(fig, gs, ax, values, angles, bin_width, label):
    """
    Heatmap along every spoke: % of the spoke's cases per value bin, drawn
    as ring segments (empty bins white), with a shared colorbar in the
    middle column. The centre hole (set_rorigin) keeps the inner segments
    from shrinking to a point.
    """
    N      = len(angles)
    edges  = np.linspace(0, 1, int(round(1 / bin_width)) + 1)
    pct    = []
    for j in range(N):
        y = values[:, j]
        y = np.clip(y[np.isfinite(y)], 0, 1)
        counts, _ = np.histogram(y, bins=edges)
        pct.append(100.0 * counts / max(len(y), 1))

    # Colour classes covering 0-100%: fine 5% steps where most segments are,
    # widening towards 100 so large shares keep their own class. EISRAD blue
    # gradient (as in the heatmaps); more lesions = darker
    bounds = [0, 5, 10, 15, 20, 30, 40, 60, 80, 100]
    blues = matplotlib.colors.LinearSegmentedColormap.from_list(
        'eisrad_blues', ['#c6d4f5', BLUE, '#0b1f66'])   # as in the heatmaps
    cmap = matplotlib.colors.ListedColormap(
        blues(np.linspace(0.0, 1.0, len(bounds) - 1)))
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Ring segments (annular sectors, drawn as polar bars so the edges follow
    # the circles) of 30% of the spoke spacing; the polar axes place them
    # correctly including the radial origin offset (hole)
    width = 0.3 * (2 * pi / N)
    for ang, p in zip(angles, pct):
        # empty bins (exactly 0%) are white, so the spoke keeps its shape
        colours = [cmap(norm(v)) if v > 0 else 'white' for v in p]
        ax.bar(np.full(len(p), ang), np.diff(edges), width=width,
               bottom=edges[:-1], color=colours,
               edgecolor='white', linewidth=1, zorder=1)

    cax = fig.add_subplot(gs[0, 1])
    pos = cax.get_position()
    new_h = pos.height * 0.8
    # shift right so the colorbar clears the labels on the right of the radar
    cax.set_position([pos.x0 + pos.width * 0.35,
                      pos.y0 + (pos.height - new_h) / 2,
                      pos.width * 0.35, new_h])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax, spacing='uniform')
    cb.set_ticks(bounds)
    cb.ax.minorticks_off()
    cb.set_label(label, fontsize=12)
    cax.tick_params(labelsize=10)

def _radar_points(fig, gs, ax, values, colourmap, angles, info):
    """
    Dots per case on the radar, coloured by volume, plus the colorbar. Each
    case's angular offset from the spoke follows its volume rank (smallest
    clockwise, largest counter-clockwise), so a case sits at the same
    position on every spoke and similar volumes sit together.
    """
    N = len(angles)
    if info.get('logplot', False):
        pos = colourmap[colourmap > 0]
        vmin = pos.min() if pos.size else 1e-6
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=info['maximum'])
        # Empty auto masks would be masked on a log scale; show them at the bottom
        colourmap = np.clip(colourmap, vmin, None)
    else:
        norm = matplotlib.colors.Normalize(vmin=info['minimum'],
                                           vmax=info['maximum'])

    n_cases = len(colourmap)
    rank    = (rankdata(colourmap) - 1) / max(n_cases - 1, 1)   # 0 .. 1
    offsets = (rank - 0.5) * 2 * 0.15 * (2 * pi / N)            # ±15% of spacing
    for i, y_vals in enumerate(values):
        mask = np.isfinite(y_vals)
        if not mask.any():
            continue
        angs_pt = np.array(angles)[mask] + offsets[i]
        cvals   = np.full(mask.sum(), colourmap[i])
        ax.scatter(angs_pt, y_vals[mask],
                   c=cvals, cmap='cividis', norm=norm,
                   s=20, zorder=3)

    # ── Colorbar subplot (middle, now wider) ──
    cax = fig.add_subplot(gs[0, 1])
    pos = cax.get_position()
    new_h = pos.height * 0.8
    new_y = pos.y0 + (pos.height - new_h) / 2
    cax.set_position([pos.x0, new_y, pos.width*0.5, new_h])

    cb  = matplotlib.colorbar.ColorbarBase(
              cax, cmap='cividis', norm=norm, orientation='vertical')
    if info.get('logplot', False):
        cax.set_yscale('log')
        cax.yaxis.set_major_locator(LogLocator(base=10))
        cax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    label = info.get('label','')
    if info.get('unit'):
        label += f" ({info['unit']})"
    cb.set_label(label, fontsize=12)
    cax.tick_params(labelsize=10)

def _log_edges(lo, hi, bins_per_decade=10):
    """Bin edges on a 10^(k/bins_per_decade) grid covering lo..hi (lo > 0)."""
    k_lo = int(np.floor(np.log10(lo) * bins_per_decade))
    k_hi = max(int(np.ceil(np.log10(hi) * bins_per_decade)), k_lo + 1)
    return 10.0 ** (np.arange(k_lo, k_hi + 1) / bins_per_decade)

def _heatmap_figure(nx, ny, cell, top_hist=False):
    """
    Fixed layout in inches so heatmap cells stay square and panels line up:
    heatmap, marginal histogram (shared y) to its right, and a stats panel on
    the far right. Without `top_hist` the horizontal count colorbar sits
    above the heatmap; with it, a second marginal histogram (shared x) sits
    above the heatmap and the colorbar moves into the empty top-right corner.
    Returns fig, ax, hax, tax (None without top_hist), cax, legend_ax.
    """
    W, Hh  = nx * cell, ny * cell
    L, B   = 0.8, 0.65                 # left / bottom margins
    gap    = 0.15
    hist_w = 1.2
    stat_w = 3.4
    cb_h   = 0.12
    top    = 0.65                      # room for the colorbar + its label
    fig_w  = L + W + gap + hist_w + 0.4 + stat_w
    if top_hist:
        fig_h = B + Hh + gap + hist_w + 0.2
    else:
        fig_h = B + Hh + gap + cb_h + top
    fig    = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    def box(x0, y0, w, h):
        return [x0 / fig_w, y0 / fig_h, w / fig_w, h / fig_h]

    ax        = fig.add_axes(box(L, B, W, Hh))
    hax       = fig.add_axes(box(L + W + gap, B, hist_w, Hh), sharey=ax)
    legend_ax = fig.add_axes(box(L + W + gap + hist_w + 0.4, B, stat_w, Hh))
    if top_hist:
        tax = fig.add_axes(box(L, B + Hh + gap, W, hist_w), sharex=ax)
        cax = fig.add_axes(box(L + W + gap, B + Hh + gap + 0.25,
                               hist_w, cb_h))
    else:
        tax = None
        cax = fig.add_axes(box(L, B + Hh + gap, W, cb_h))
    return fig, ax, hax, tax, cax, legend_ax

def _draw_heatmap(fig, ax, cax, xedges, yedges, H, small=False):
    """Count heatmap (empty cells white, linear scale) + colorbar on top."""
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'eisrad_blues', ['#c6d4f5', BLUE, '#0b1f66'])
    cmap.set_bad('white')
    norm = matplotlib.colors.Normalize(vmin=1, vmax=max(H.max(), 2))
    mesh = ax.pcolormesh(xedges, yedges, np.ma.masked_equal(H, 0).T,
                         cmap=cmap, norm=norm,
                         edgecolors='white', linewidth=0.3)
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    cb = fig.colorbar(mesh, cax=cax, orientation='horizontal')
    cax.xaxis.set_major_locator(MaxNLocator(integer=True,
                                            nbins=4 if small else 'auto'))
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    cb.set_label('Count', fontsize=12)
    cax.tick_params(labelsize=10)

def _draw_marginal(hax, y, yedges, top=False):
    """
    Marginal histogram sharing the heatmap's bins: to the right of it
    (shared y) by default, or above it (shared x) with `top`.
    """
    hax.hist(y, bins=yedges, orientation='vertical' if top else 'horizontal',
             color=BLUE, alpha=0.6)
    if top:
        hax.tick_params(labelbottom=False, labelsize=10)
        hax.set_ylabel('Count', fontsize=12)
    else:
        hax.tick_params(labelleft=False, labelsize=10)
        hax.set_xlabel('Count', fontsize=12)
    hax.spines['top'].set_visible(False)
    hax.spines['right'].set_visible(False)
    hax.grid(True, axis='y' if top else 'x',
             linestyle='--', linewidth=0.5, alpha=0.5)

def plot_volume_scatter(df, outfile, bins_per_decade=10, lesion=False):
    """
    Heatmap of manual vs. automated volume (count per cell) on log axes with
    square cells (`bins_per_decade` per factor of 10, same grid on both
    axes), marginal histograms of manual (top) and automated (right) volume,
    the identity line, and
    a manual/auto zero-vs-nonzero confusion table. Cases with a zero volume
    cannot sit on log axes; they are counted in the table (FN, FP, TN).

    With `lesion`, rows are lesions (manual lesions + false-positive blobs):
    TP / FN count manual lesions with / without overlap (the 'detected'
    column), FP counts false-positive blobs, and TN does not apply.
    """
    manual = df['vol_manual_ml']
    auto   = df['vol_auto_ml']

    # Confusion matrix components
    mpos = manual > 0
    apos = auto   > 0
    if lesion:
        det = (df['detected'] == True)  # NaN for false-positive blobs
        TP = int((mpos & det).sum())
        FN = int((mpos & ~det).sum())
        FP = int((~mpos).sum())
        TN = '–'
    else:
        TP = int((mpos & apos).sum())
        FN = int((mpos & ~apos).sum())
        FP = int((~mpos & apos).sum())
        TN = int((~mpos & ~apos).sum())

    # Pearson correlation
    pr = _corr(pearsonr, manual, auto)

    # Same log grid on both axes so the identity line crosses cell corners
    both = mpos & apos
    x = manual[both].to_numpy()
    y = auto[both].to_numpy()
    if both.any():
        edges = _log_edges(min(x.min(), y.min()), max(x.max(), y.max()),
                           bins_per_decade)
    else:
        edges = _log_edges(0.1, 1, bins_per_decade)
    # Guard against float rounding pushing min/max just outside the edges
    x = np.clip(x, edges[0], edges[-1])
    y = np.clip(y, edges[0], edges[-1])
    H, _, _ = np.histogram2d(x, y, [edges, edges])

    n = len(edges) - 1
    fig, ax, hax, tax, cax, legend_ax = _heatmap_figure(n, n, 3.6 / n,
                                                        top_hist=True)
    _draw_heatmap(fig, ax, cax, edges, edges, H, small=True)
    ax.plot(edges[[0, -1]], edges[[0, -1]],
            color='grey', linestyle='--', linewidth=1)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(edges[0], edges[-1])
    ax.set_xlabel('Manual Lesion Volume (mL)', fontsize=12)
    ax.set_ylabel('Automated Lesion Volume (mL)', fontsize=12)

    _draw_marginal(hax, y, edges)
    _draw_marginal(tax, x, edges, top=True)

    # Side‐panel
    legend_ax.axis('off')
    legend_ax.text(0, 0.95, f'Pearson r = {pr:.2f}',
                   transform=legend_ax.transAxes,
                   fontsize=11, va='top')
    if lesion:
        cell_text = [['',               'Overlap',    'No overlap'],
                     ['Manual lesion',  f'TP = {TP}', f'FN = {FN}'],
                     ['Auto only',      f'FP = {FP}', TN]]
    else:
        cell_text = [['',           'Auto > 0',   'Auto = 0'],
                     ['Manual > 0', f'TP = {TP}', f'FN = {FN}'],
                     ['Manual = 0', f'FP = {FP}', f'TN = {TN}']]
    tbl = legend_ax.table(cellText=cell_text, cellLoc='center',
                          loc='center', bbox=[0, 0.35, 1, 0.4])
    tbl[0, 0].set_visible(False)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    fig.savefig(outfile, dpi=300)
    plt.close(fig)

def plot_dice_vs_volume(df, outfile, dice_thresh=0.2, dice_bin=0.05,
                        bins_per_decade=10, unit='patients', extra_stats=None):
    """
    Heatmap of DICE vs. manual volume (count per cell), with a marginal
    Dice histogram sharing the Dice axis and the heatmap's Dice bins.

    Cells are square: volume bins are equally wide on the log axis
    (`bins_per_decade` per factor of 10, so 1 mL, 10 mL, ... are always edges)
    and run from the smallest to the largest volume; the panel widens with
    the volume range instead of changing the bin count.

    `unit` names the rows in the stats panel ('patients' or 'lesions');
    `extra_stats` replaces the 'Manual zeros' line when given.
    """
    manual = df['vol_manual_ml']
    dice   = df['Dice']

    # Mask out zero volumes for axis, but still count them
    pos = manual > 0
    x = manual[pos].to_numpy()
    y = dice[pos].to_numpy()
    zero_manual = int((manual == 0).sum())

    if not pos.any():
        print("⚠️  No lesions in manual masks → skipping Dice heatmap.")
        return

    pr = _corr(pearsonr, x, y)
    sr = _corr(spearmanr, x, y)
    n_hit  = int((dice >= dice_thresh).sum())
    dr     = n_hit / len(dice) * 100.0

    # Square bins: ny Dice rows, log-volume columns covering min..max volume
    ny     = int(round(1.0 / dice_bin))
    xedges = _log_edges(x.min(), x.max(), bins_per_decade)
    yedges = np.linspace(0, 1, ny + 1)
    # Guard against float rounding pushing min/max just outside the edges
    x = np.clip(x, xedges[0], xedges[-1])
    H, _, _ = np.histogram2d(x, y, [xedges, yedges])

    fig, ax, hax, _, cax, legend_ax = _heatmap_figure(len(xedges) - 1, ny,
                                                      3.6 / ny)
    _draw_heatmap(fig, ax, cax, xedges, yedges, H)
    ax.axhline(dice_thresh, color=BLUE, linestyle='--', linewidth=1)

    ax.set_xscale('log')
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_xlabel('Manual Lesion Volume (mL)', fontsize=12)
    ax.set_ylabel('DICE Score', fontsize=12)
    ax.set_ylim(0, 1)

    _draw_marginal(hax, y, yedges)
    hax.axhline(dice_thresh, color=BLUE, linestyle='--', linewidth=1)

    # Side‐panel
    legend_ax.axis('off')
    stats = [
        f'Pearson r = {pr:.2f}',
        f'Spearman r = {sr:.2f}',
        f'# {unit} (Dice≥{dice_thresh:g}) = {n_hit} ({dr:.1f}%)',
    ] + (extra_stats if extra_stats is not None
         else [f'Manual zeros = {zero_manual}'])
    for i, txt in enumerate(stats):
        legend_ax.text(0, 0.9 - i*0.15, txt,
                       transform=legend_ax.transAxes,
                       fontsize=11, va='top')

    fig.savefig(outfile, dpi=300)
    plt.close(fig)

def plot_metric_bars(values, colourmap, labels, info, outfile,
                     higher_better=None, bounded=None):
    """
    Bar-plot alternative to the radar, with one panel (and y axis) per
    metric so raw values in their own units can be shown (e.g. HD95 in mm).
    Solid bars show the median, error bars the IQR, and each case is drawn
    on top as a translucent point coloured by manual lesion volume (same
    colour scale as the radar). Its horizontal offset follows its volume rank
    (smallest left, largest right), as on the radar, so a case keeps the same
    position in every panel and similar volumes sit together.

    higher_better : per metric, True / False adds ↑ / ↓ to the panel title
    bounded       : per metric, True keeps a 0-1 axis (extended below 0 if
                    needed); False scales the axis to the data from 0
    """
    values  = np.asarray(values, dtype=float)
    n       = len(labels)
    higher_better = higher_better or [None] * n
    bounded       = bounded or [True] * n
    medians = np.nanmedian(values, axis=0)
    q1      = np.nanpercentile(values, 25, axis=0)
    q3      = np.nanpercentile(values, 75, axis=0)

    cmap = plt.get_cmap('cividis')
    if info.get('logplot', False):
        pos  = colourmap[colourmap > 0]
        vmin = pos.min() if pos.size else 1e-6
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=info['maximum'])
        # Zero volumes would be masked on a log scale; show them at the bottom
        colourmap = np.clip(colourmap, vmin, None)
    else:
        norm = matplotlib.colors.Normalize(vmin=info['minimum'],
                                           vmax=info['maximum'])

    fig, axes = plt.subplots(1, n, figsize=(1.25 * n + 1.5, 4.5), dpi=300,
                             constrained_layout=True, squeeze=False)
    axes = axes[0]
    rank   = (rankdata(colourmap) - 1) / max(len(colourmap) - 1, 1)   # 0 .. 1
    jitter = (rank - 0.5) * 2 * 0.2                                  # ±0.2

    for j, ax in enumerate(axes):
        y    = values[:, j]
        mask = np.isfinite(y)
        ax.bar(0, medians[j], width=0.6, color=TEAL, zorder=1)
        ax.scatter(jitter[mask], y[mask], c=colourmap[mask], cmap=cmap,
                   norm=norm, s=10, alpha=0.4, edgecolors='none', zorder=2)
        ax.errorbar(0, medians[j], yerr=[[medians[j] - q1[j]], [q3[j] - medians[j]]],
                    fmt='none', ecolor='black', elinewidth=1.5, capsize=5,
                    zorder=3)

        arrow = {True: ' ↑', False: ' ↓'}.get(higher_better[j], '')
        ax.set_title(labels[j] + arrow, fontsize=10)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        if bounded[j]:
            ax.set_ylim(min(0, np.nanmin(y)) if mask.any() else 0, 1.02)
        elif mask.any():
            ax.set_ylim(0, np.nanmax(y) * 1.05 or 1)
        ax.tick_params(labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=axes, shrink=0.8, pad=0.02)
    label = info.get('label','')
    if info.get('unit'):
        label += f" ({info['unit']})"
    cb.set_label(label, fontsize=11)
    cb.ax.tick_params(labelsize=9)

    fig.savefig(outfile, dpi=300)
    if info.get('display', False):
        plt.show()
    plt.close(fig)

def detection_curve(coverages, n_boot=1000, seed=0):
    """
    Detection rate (%) vs. coverage threshold, pooled over items, with a 95%
    bootstrap CI from resampling patients (lesions of a patient stay
    together). An item counts as detected at t if coverage >= t and > 0.

    coverages : list of non-empty 1-D arrays, one per patient
    Returns thresholds, pooled, ci_low, ci_high (each 101 values).
    """
    thresholds = np.linspace(0, 1, 101)
    # detected[p, t]: items of patient p with coverage >= thresholds[t]
    detected = np.array([((c[:, None] >= thresholds) & (c[:, None] > 0)).sum(axis=0)
                         for c in coverages])
    n_items  = np.array([len(c) for c in coverages])
    pooled   = detected.sum(axis=0) / n_items.sum() * 100.0

    # Patient-level bootstrap: multinomial counts = how often each is drawn
    rng = np.random.default_rng(seed)
    P   = len(coverages)
    W   = rng.multinomial(P, np.full(P, 1.0/P), size=n_boot)
    boot = (W @ detected) / (W @ n_items)[:, None] * 100.0
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return thresholds, pooled, lo, hi

def plot_detection_curve(coverages, outfile, n_boot=1000, seed=0,
                         unit='lesions'):
    """
    Detection curve. An item (lesion, or a patient's whole lesion mask)
    counts as detected at threshold t if the automated mask covers at least
    a fraction t of it (and overlaps it at all, so t = 0 means "any overlap"
    rather than 100%). The line is the pooled % of items detected; the band
    is a 95% bootstrap CI from resampling patients, so lesions of the same
    patient stay together.

    coverages : list of 1-D arrays, one per patient: the coverage of each of
                its lesions (unit='lesions'), or a single value, the patient's
                total lesion coverage (unit='patients')
    """
    coverages = [np.asarray(c) for c in coverages if len(c)]
    if not coverages:
        print("⚠️  No lesions in manual masks → skipping detection curve.")
        return

    thresholds, pooled, lo, hi = detection_curve(coverages, n_boot, seed)
    P = len(coverages)

    all_cov  = np.concatenate(coverages)
    any_hit  = (all_cov > 0).mean() * 100.0
    half_hit = (all_cov >= 0.5).mean() * 100.0

    fig, (ax, legend_ax) = plt.subplots(
        ncols=2,
        figsize=(9, 4.5),
        dpi=300,
        gridspec_kw={'width_ratios': [2, 1]},
        constrained_layout=True
    )

    ax.fill_between(thresholds, lo, hi, color=BLUE, alpha=0.2, linewidth=0)
    ax.plot(thresholds, pooled, color=BLUE, linewidth=2)

    ax.set_xlabel('Lesion Coverage Threshold', fontsize=12)
    ax.set_ylabel(f'{unit.capitalize()} Detected (%)', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # Side‐panel
    legend_ax.axis('off')
    stats = [
        (f'Lesions = {len(all_cov)} ({P} patients)' if unit == 'lesions'
         else f'Patients = {P}'),
        f'Any overlap = {any_hit:.1f}%',
        f'Coverage ≥ 50% = {half_hit:.1f}%',
        f'Median coverage = {np.median(all_cov):.2f}',
        f'Band: 95% CI ({n_boot} patient bootstraps)'
    ]
    for i, txt in enumerate(stats):
        legend_ax.text(0, 0.9 - i*0.15, txt,
                       transform=legend_ax.transAxes,
                       fontsize=11, va='top')

    fig.savefig(outfile, dpi=300)
    plt.close(fig)
