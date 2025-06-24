#!/usr/bin/env python
"""
:Summary:
All plotting routines for EISRAD: radar plot, manual-vs-auto scatter,
and Dice-vs-volume scatter.

:Authors:
    Markus D. Schirmer
    MGH / Harvard Medical School

:Version: 1.1
:Date: 2025-06-24
:License: MIT
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from math import pi
from scipy.stats import pearsonr, spearmanr

BLUE = 'royalblue'

def _intermediate_ticks(min_val, max_val, n=4):
    return list(np.linspace(min_val, max_val, num=n+2)[1:-1])

def plot_radar(values, colourmap, measures, info, outfile='polar_results.png'):
    """
    Wider radar plot of multi-metric comparisons:
      • Figure width increased to 12" so the middle colorbar can expand.
      • GridSpec cols: [2, 0.2, 1] width ratios (radar : cbar : table).
    """
    N = len(measures)
    angles = [n/float(N)*2*pi for n in range(N)]
    angles_closed = angles + [angles[0]]

    # Compute median and IQR
    medians = np.median(values, axis=0)
    q1      = np.percentile(values, 25, axis=0)
    q3      = np.percentile(values, 75, axis=0)
    med_closed = np.concatenate([medians, [medians[0]]])
    q1_closed  = np.concatenate([q1,      [q1[0]]])
    q3_closed  = np.concatenate([q3,      [q3[0]]])

    # Make the figure wider: 12" wide instead of 9"
    fig = plt.figure(figsize=(12, 6), dpi=300, constrained_layout=True)
    gs  = fig.add_gridspec(1, 3, width_ratios=[2, 0.2, 1])

    # ── Radar subplot (left 2/3) ──
    ax = fig.add_subplot(gs[0, 0], polar=True)
    ax.set_ylim(0, 1)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_xticks(angles)
    ax.set_xticklabels(measures, fontsize=12)
    ax.fill_between(angles_closed, q1_closed, q3_closed,
                    color='royalblue', alpha=0.2, zorder=1)
    ax.plot(angles_closed, med_closed, color='royalblue', linewidth=2)

    # Individual points
    if info.get('logplot', False):
        pos = colourmap[colourmap > 0]
        vmin = pos.min() if pos.size else 1e-6
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=info['maximum'])
    else:
        norm = None

    for i, y_vals in enumerate(values):
        mask = np.isfinite(y_vals)
        if not mask.any():
            continue
        jitter = np.random.randn(mask.sum()) * (2*pi/N) / 20
        angs_pt = np.array(angles)[mask] + jitter
        cvals   = np.full(mask.sum(), colourmap[i])
        ax.scatter(angs_pt, y_vals[mask],
                   c=cvals, cmap='cividis', norm=norm,
                   s=20, zorder=3)

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

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

    # ── Table subplot (right 1/3) ──
    tax = fig.add_subplot(gs[0, 2])
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
        bbox=[0.10, 0.00, 0.90, 1.00]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)

    # Save/display
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    if info.get('display', False):
        plt.show()
    plt.close(fig)

def plot_volume_scatter(df, outfile):
    """
    Publication-quality scatter of manual vs. automated volumes.
    Now in royalblue instead of black.
    """
    manual = df['vol_manual_cc']
    auto   = df['vol_auto_cc']

    # Confusion matrix components
    mpos = manual > 0
    apos = auto   > 0
    TP = int((mpos & apos).sum())
    FN = int((mpos & ~apos).sum())
    FP = int((~mpos & apos).sum())
    TN = int((~mpos & ~apos).sum())

    # Pearson correlation
    pr, _ = pearsonr(manual, auto)

    fig, (ax, legend_ax) = plt.subplots(
        ncols=2,
        figsize=(9, 4.5),
        dpi=300,
        gridspec_kw={'width_ratios': [2, 1]},
        constrained_layout=True
    )

    # Scatter in blue
    ax.scatter(
        manual, auto,
        color=BLUE, s=20, alpha=0.6, edgecolors='none'
    )
    lims = [min(manual.min(), auto.min()), max(manual.max(), auto.max())]
    ax.plot(lims, lims, color='grey', linestyle='--', linewidth=1)

    # Symlog for zeros
    ax.set_xscale('symlog')
    ax.set_yscale('symlog')
    ax.set_xlabel('Manual Lesion Volume (cc)', fontsize=12)
    ax.set_ylabel('Automated Lesion Volume (cc)', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # Side‐panel
    legend_ax.axis('off')
    legend_ax.text(0, 0.95, f'Pearson r = {pr:.2f}',
                   transform=legend_ax.transAxes,
                   fontsize=11, va='top')
    cell_text = [[f'TP = {TP}', f'FN = {FN}'],
                 [f'FP = {FP}', f'TN = {TN}']]
    tbl = legend_ax.table(cellText=cell_text,
                          rowLabels=['Manual > 0', 'Manual = 0'],
                          colLabels=['Auto > 0', 'Auto = 0'],
                          cellLoc='center', rowLoc='center',
                          loc='center', bbox=[0, 0.1, 1, 0.75])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    fig.savefig(outfile, dpi=300)
    plt.close(fig)

def plot_dice_vs_volume(df, outfile, dice_thresh=0.2):
    """
    Clean log–scatter of DICE vs. manual volume, now in royalblue.
    """
    manual = df['vol_manual_cc']
    dice   = df['Dice']

    # Mask out zero volumes for axis, but still count them
    pos = manual > 0
    x = manual[pos]
    y = dice[pos]
    zero_manual = int((manual == 0).sum())

    pr, _ = pearsonr(x, y)
    sr, _ = spearmanr(x, y)
    dr     = (dice > dice_thresh).mean() * 100.0

    fig, (ax, legend_ax) = plt.subplots(
        ncols=2,
        figsize=(9, 4.5),
        dpi=300,
        gridspec_kw={'width_ratios': [2, 1]},
        constrained_layout=True
    )

    # Scatter in royalblue
    ax.scatter(
        x, y,
        color=BLUE, s=20, alpha=0.6, edgecolors='none'
    )
    ax.axhline(dice_thresh, color=BLUE, linestyle='--', linewidth=1)

    ax.set_xscale('log')
    ax.set_xlabel('Manual Lesion Volume (cc)', fontsize=12)
    ax.set_ylabel('DICE Score', fontsize=12)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # Side‐panel
    legend_ax.axis('off')
    stats = [
        f'Pearson r = {pr:.2f}',
        f'Spearman r = {sr:.2f}',
        f'Detection rate = {dr:.1f}%',
        f'Manual zeros = {zero_manual}'
    ]
    for i, txt in enumerate(stats):
        legend_ax.text(0, 0.9 - i*0.15, txt,
                       transform=legend_ax.transAxes,
                       fontsize=11, va='top')

    fig.savefig(outfile, dpi=300)
    plt.close(fig)