#!/usr/bin/env python
"""
:Summary:
Plots for comparing algorithms (eisrad-compare): radar overlay, paired box
plots with significance brackets, and detection-curve overlays. One colour
per algorithm from the colour-blind-safe Okabe-Ito palette.

:Authors:
    Markus D. Schirmer
    MGH / Harvard Medical School

:License: MIT
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from eisrad.plots import setup_radar_axes, detection_curve
from eisrad.compare import (PATIENT_METRICS, PATIENT_RADAR, LESION_METRICS,
                            LESION_RADAR, patient_table, lesion_table, fp_counts)

# Okabe & Ito (2008): distinguishable with all common colour-vision deficiencies
OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#CC79A7',
             '#56B4E9', '#D55E00', '#F0E442', '#000000']

def _colours(names):
    return {n: OKABE_ITO[i % len(OKABE_ITO)] for i, n in enumerate(names)}

def _stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


# ─── Radar overlay ───

def _lesion_radar_values(runs, patients):
    """{algorithm: lesions x 6 array} of the lesion radar scores."""
    out = {}
    for name, (_, les) in runs.items():
        m = les[(les['type'] == 'manual') & les['manual'].isin(patients)]
        r = (3.0 * m['vol_manual_ml'].to_numpy() * 1000.0 / (4 * np.pi)) ** (1 / 3)
        out[name] = np.column_stack(
            [m[c].to_numpy(dtype=float) for c in ['Dice', 'Jaccard', 'TPR', 'VS']]
            + [1.0 / (1.0 + m['HD95_mm'].to_numpy(dtype=float) / r),
               m['SDice'].to_numpy(dtype=float)])
    return out

def plot_radar_overlay(runs, patients, outfile, lesion=False):
    """
    Median line and IQR band of every algorithm on one radar (0-1 scores,
    negative values clamped to 0 as in the single-algorithm radar).
    """
    names   = list(runs)
    colours = _colours(names)
    if lesion:
        measures = LESION_RADAR
        values   = _lesion_radar_values(runs, patients)
    else:
        measures = PATIENT_RADAR
        values   = {n: runs[n][0].loc[patients, measures].to_numpy(dtype=float)
                    for n in names}

    fig = plt.figure(figsize=(8, 7), dpi=300, constrained_layout=True)
    ax  = fig.add_subplot(111, polar=True)
    angles = setup_radar_axes(ax, measures)
    closed = lambda v: np.concatenate([v, v[:1]])
    ang_c  = angles + angles[:1]
    for n in names:
        v  = np.clip(values[n], 0, None)
        q1, med, q3 = (np.nanpercentile(v, q, axis=0) for q in (25, 50, 75))
        ax.fill_between(ang_c, closed(q1), closed(q3), color=colours[n],
                        alpha=0.15, linewidth=0, zorder=1)
        ax.plot(ang_c, closed(med), color=colours[n], linewidth=2.2,
                label=n, zorder=3)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06),
              ncol=min(len(names), 4), frameon=False, fontsize=11)
    ax.set_title('Lesions' if lesion else 'Patients', fontsize=13, pad=40)
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ─── Paired box plots with significance brackets ───

def plot_compare_boxes(runs, patients, pairs, outfile, lesion=False):
    """
    One panel per metric (own y axis, raw values), one box per algorithm with
    the individual patients / lesions as faint dots. Brackets mark pairwise
    differences that stay significant after Holm correction.
    """
    names   = list(runs)
    colours = _colours(names)
    level   = 'lesion' if lesion else 'patient'
    if lesion:
        specs = [(lab, lesion_table(runs, patients, col), hb, bd)
                 for lab, col, hb, bd in LESION_METRICS]
    else:
        specs = [(lab, patient_table(runs, patients, col), hb, bd)
                 for lab, col, hb, bd, _ in PATIENT_METRICS]
        if all(les is not None for _, les in runs.values()):
            specs.append(('False-positive lesions', fp_counts(runs, patients),
                          False, False))
    sig = pairs[(pairs['level'] == level)].set_index(['metric', 'A', 'B'])

    max_cols = 7          # wrap into rows so the figure fits a page
    k     = len(names)
    ncols = min(len(specs), max_cols)
    nrows = int(np.ceil(len(specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, dpi=300, constrained_layout=True,
                             figsize=(0.55 * k * ncols + 1.5, 4.2 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(specs):]:
        ax.set_visible(False)
    rng = np.random.default_rng(0)
    for ax, (label, df, higher, bounded) in zip(axes, specs):
        data = [df[n].dropna().to_numpy() for n in names]
        bp = ax.boxplot(data, positions=np.arange(k), widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'))
        for patch, n in zip(bp['boxes'], names):
            patch.set_facecolor(colours[n]); patch.set_alpha(0.55)
            patch.set_edgecolor('black')
        for i, d in enumerate(data):
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=4,
                       color=colours[names[i]], alpha=0.35, linewidths=0,
                       zorder=3)

        # y range, then brackets stacked above the data
        allv = np.concatenate([d for d in data if len(d)]) if any(len(d) for d in data) else np.array([0.0])
        lo = min(0.0, np.nanmin(allv)) if bounded else 0.0
        hi = 1.0 if bounded else (np.nanmax(allv) if allv.size else 1.0) or 1.0
        step, top = 0.08 * (hi - lo), hi
        for i, j in [(i, j) for i in range(k) for j in range(i + 1, k)]:
            key = (label, names[i], names[j])
            if key not in sig.index:
                continue
            p = sig.loc[key, 'p (Holm)']
            if not np.isfinite(p) or p >= 0.05:
                continue
            top += step
            ax.plot([i, i, j, j], [top - step * 0.3, top, top, top - step * 0.3],
                    color='black', linewidth=0.8)
            ax.text((i + j) / 2, top, _stars(p), ha='center', va='bottom',
                    fontsize=9)
        ax.set_ylim(lo, top + step)

        arrow = ' ↑' if higher else ' ↓'
        title = label.replace('False-positive lesions', 'False-positive\nlesions')
        ax.set_title(title + arrow, fontsize=10)
        ax.set_xticks([])
        ax.tick_params(labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

    handles = [matplotlib.patches.Patch(facecolor=colours[n], alpha=0.55,
                                        edgecolor='black', label=n)
               for n in names]
    fig.legend(handles=handles, loc='outside lower center', ncol=min(k, 6),
               frameon=False, fontsize=10)
    fig.suptitle(('Lesions' if lesion else 'Patients') +
                 '  (* p<0.05, ** p<0.01, *** p<0.001, Holm-corrected)',
                 fontsize=10)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


# ─── Detection curve overlay ───

def plot_detection_overlay(runs, patients, outfile, n_boot=1000, seed=0,
                           lesion=False):
    """Detection curves of all algorithms (95% bootstrap bands) on one plot."""
    names   = list(runs)
    colours = _colours(names)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300, constrained_layout=True)
    for n in names:
        met, les = runs[n]
        if lesion:
            m = les[(les['type'] == 'manual') & les['manual'].isin(patients)]
            cov = [g['TPR'].to_numpy(dtype=float)
                   for _, g in m.groupby('manual', sort=False)]
        else:
            sub = met.loc[patients]
            cov = [np.array([t]) for t, v in zip(sub['TPR'], sub['vol_manual_ml'])
                   if v > 0]
        if not cov:
            continue
        t, pooled, lo, hi = detection_curve(cov, n_boot, seed)
        ax.fill_between(t, lo, hi, color=colours[n], alpha=0.2, linewidth=0)
        ax.plot(t, pooled, color=colours[n], linewidth=2,
                label=f'{n} ({pooled[0]:.1f}% any overlap)')
    ax.set_xlabel('Lesion Coverage Threshold', fontsize=12)
    ax.set_ylabel(('Lesions' if lesion else 'Patients') + ' Detected (%)',
                  fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False, fontsize=10)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
