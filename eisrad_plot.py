#!/usr/bin/env python
"""
:Summary:
This module handles visualization for EISRAD by creating radar (polar) plots to display multiple segmentation similarity metrics across one or more image pairs.

:Function:
    plot_evaluation(values, info, measures, colourmap, outfile)

:Authors:
    Markus D. Schirmer
    MGH / Harvard Medical School

:Version: 1.0
:Date: 2024-05-13
:License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from matplotlib import gridspec
from math import pi

def _generate_intermediate_ticks(start, end, count=4):
    step = (end - start) / (count + 1)
    return [round(start + step * (i + 1), 2) for i in range(count)]
    
def plot_evaluation(values, info, measures, colourmap, outfile='polar_results.png'):
    """
    Generate a radar plot (polar plot) to visualize multi-metric segmentation comparisons.

    Parameters
    ----------
    values : np.ndarray
        A (N x M) array where each row contains metric values for a segmentation pair.
    info : dict
        Dictionary with plotting parameters:
            - minimum (float): min value for colorbar
            - maximum (float): max value for colorbar
            - label (str): label for colorbar
            - unit (str): unit string for colorbar
            - logplot (bool): whether to log-scale the colorbar
            - display (bool): whether to show the figure interactively
    measures : list of str
        Ordered list of metric names corresponding to the values.
    colourmap : np.ndarray
        A vector of volume or intensity values for color encoding.
    outfile : str
        File path to save the plot.
    """

    # Normalize colour values
    _min, _max = info['minimum'], info['maximum']
    if _max == _min:
        _max += 1e-6  # prevent divide-by-zero
    normed_colours = (colourmap - _min) / (_max - _min)
    colours = [cm.cividis(c) for c in normed_colours]

    # Angular positions around circle
    N = len(measures)
    angles = [n / float(N) * 2 * pi for n in range(N)]

    # Set up polar plot
    fig = plt.figure(figsize=(11, 9.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[17, 2, 1])
    ax = plt.subplot(gs[0], polar=True)

    ax.set_rlabel_position(0)
    ax.xaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)
    ax.yaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)
    # Force consistent radial scale from 0.0 to 1.0 in 0.2 steps
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=14)


    # Plot individual data points as scattered dots
    for i in range(values.shape[0]):
        jittered_angles = np.asarray(angles) + np.random.randn(N) * (2 * pi / N) / 15.
        color = [colours[i]] * N
        norm = matplotlib.colors.LogNorm(vmin=_min, vmax=_max) if info['logplot'] else None
        ax.scatter(jittered_angles, values[i, :], s=23, color=color, norm=norm, zorder=3)

    # Median + IQR shading
    med = np.median(values, axis=0).tolist() + [np.median(values, axis=0)[0]]
    upper = np.percentile(values, 75, axis=0).tolist() + [np.percentile(values, 75, axis=0)[0]]
    lower = np.percentile(values, 25, axis=0).tolist() + [np.percentile(values, 25, axis=0)[0]]
    angles += angles[:1]

    ax.plot(angles, med, color=[86 / 255., 180 / 255., 233 / 255.], zorder=5)
    ax.fill_between(angles, upper, lower, zorder=4, color=[86 / 255., 180 / 255., 233 / 255.], alpha=0.3)
    plt.xticks(angles[:-1], [])

    # Labels for each axis/metric
    for i in range(N):
        angle_rad = i / float(N) * 2 * pi - 0.05
        ax.text(angle_rad, 1.22, f"{measures[i]}\n(m={med[i]:.2f})",
                size=16, ha='center', va='center')

    # === Colorbar setup ===
    cbax = plt.subplot(gs[2])
    dummy = np.array([[_min, _max]])
    cmap_img = plt.imshow(dummy, aspect='auto', cmap="cividis", visible=False)
    norm = matplotlib.colors.LogNorm(vmin=_min, vmax=_max) if info['logplot'] else None
    cbar = plt.colorbar(cmap_img, cax=cbax)

    ticks = cbar.get_ticks()
    labels = ["" for _ in ticks]
    if _min < min(ticks):
        ticks = [_min] + list(ticks)
        labels = [f"{_min:.1f} {info['unit']}"] + labels
    else:
        labels[0] = f"{min(ticks):.1f} {info['unit']}"

    if _max > max(ticks):
        ticks = ticks + [_max]
        labels = labels + [f"{_max:.1f} {info['unit']}"]
    else:
        labels[-1] = f"{max(ticks):.1f} {info['unit']}"


    ticks = sorted(set([_min, *_generate_intermediate_ticks(_min, _max), _max]))
    labels = [f"{t:.1f} {info['unit']}" for t in ticks]

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    cbar.ax.set_ylabel(info['label'], labelpad=-20, size=16)
    cbar.ax.tick_params(labelsize=13)

    # === Save and optionally display ===
    plt.savefig(outfile, bbox_inches='tight')
    if info.get("display", False):
        plt.show()
    plt.clf()
    plt.close('all')
