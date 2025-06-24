#!/usr/bin/env python
"""
:Summary:
Command-line entrypoint for EISRAD. Parses arguments, loads segmentation pairs,
computes metrics, writes reports, and delegates all plotting to plots.py.

:Example:
    eisrad -f segmentations.csv -o radar.png -r metrics.csv -l

:Authors:
    Markus D. Schirmer
    MGH / Harvard Medical School

:Version: 2.0.1
:Date: 2025-06-24
:License: MIT
"""

import sys
import os
import numpy  as np
import pandas as pd
import nibabel as nib
from optparse import OptionParser

# Absolute imports to avoid shadowing std-lib io
from eisrad.io      import load_and_reorient_pair
from eisrad.metrics import get_values
from eisrad.reports import write_low_dice, write_high_vol_diff
from eisrad.plots   import plot_radar, plot_volume_scatter, plot_dice_vs_volume


def main(argv=None):
    """
    argv: list of command-line arguments (including script name). If None,
          OptionParser.parse_args will read from sys.argv.
    """
    parser = OptionParser()
    # Core options
    parser.add_option('-f','--file',     dest='f',    help='Input CSV (manual,auto)')
    parser.add_option('-o','--output',   dest='o',
                      default='polar_results.png', help='Radar plot output file')
    parser.add_option('-r','--results',  dest='r',    help='Metrics CSV output file')
    parser.add_option('-b','--binarize', action='store_true', default=False,
                      help='Threshold segmentations (>0→1)')
    parser.add_option('-d','--display',  action='store_true', default=False,
                      help='Show plots interactively')
    parser.add_option('-v','--verbose',  action='store_true', default=False,
                      help='Print per-pair details')
    parser.add_option('-m','--min',      dest='min',   default=None,
                      help='Colorbar minimum value')
    parser.add_option('-M','--max',      dest='max',   default=None,
                      help='Colorbar maximum value')
    parser.add_option('-L','--label',    dest='label', default='',
                      help='Colorbar label')
    parser.add_option('-u','--unit',     dest='unit',  default='',
                      help='Colorbar unit')
    parser.add_option('-l','--log',      action='store_true', default=False,
                      help='Use log-scaled colorbar')

    # New flags
    parser.add_option('--dice-threshold',      dest='dice_thresh',
                      type='float', default=0.2,
                      help='Threshold for low-Dice reporting')
    parser.add_option('--low-dice-csv',        dest='low_dice_csv',
                      help='Write low-Dice cases to CSV')
    parser.add_option('--vol-diff-threshold',  dest='vol_diff_thresh',
                      type='float',
                      help='Threshold for relative volume-difference reporting')
    parser.add_option('--high-vol-diff-csv',   dest='high_vol_diff_csv',
                      help='Write high-vol-diff cases to CSV')
    parser.add_option('--scatter-volume',      dest='sv',
                      help='Output manual vs auto volume scatter plot (PNG)')
    parser.add_option('--scatter-dice',        dest='sd',
                      help='Output Dice vs volume scatter plot (PNG)')

    # Parse arguments
    if argv is not None:
        opts, args = parser.parse_args(argv[1:])
    else:
        opts, args = parser.parse_args()

    # Validate inputs
    if not opts.f or not os.path.isfile(opts.f):
        print("❌ Input CSV not found.")
        sys.exit(1)
    odir = os.path.dirname(opts.o)
    if odir and not os.path.isdir(odir):
        print(f"❌ Output directory does not exist: {odir}")
        sys.exit(1)

    try:
        min_val = float(opts.min) if opts.min else None
        max_val = float(opts.max) if opts.max else None
    except ValueError:
        print("❌ Colorbar min/max must be numeric.")
        sys.exit(1)

    # Read pair list
    df_pairs = pd.read_csv(opts.f)
    if not {'manual','auto'}.issubset(df_pairs.columns):
        print("❌ CSV must contain 'manual' and 'auto' columns.")
        sys.exit(1)
    file_list   = df_pairs[['manual','auto']].values.tolist()
    total_pairs = len(file_list)

    # Prepare to collect metrics & volumes
    measures    = ['Dice','Jaccard','TPR','VS','MI','ARI','ICC',
                   '1/(1+PBD)','KAP','1-OER','1-DER']
    data        = []
    manual_cc   = []
    auto_cc     = []
    valid_pairs = []

    # Process each pair
    for idx,(mfile,afile) in enumerate(file_list):
        if not os.path.isfile(mfile) or not os.path.isfile(afile):
            print(f"⚠️  Skipping missing: {mfile}, {afile}")
            continue

        # Load & reorient both, optionally binarize
        A, zooms = load_and_reorient_pair(mfile, opts.binarize)
        B, _     = load_and_reorient_pair(afile, opts.binarize)

        if A.shape != B.shape:
            print(f"⚠️  Shape mismatch: {mfile} vs {afile}")
            continue

        # Compute similarity metrics
        vals_dict = get_values(A, B, measures)
        row       = [round(vals_dict[m],3) for m in measures]
        data.append(row)

        # Compute volumes (cc)
        vm = A.sum() * np.prod(zooms) / 1000.0
        va = B.sum() * np.prod(zooms) / 1000.0
        manual_cc.append(vm)
        auto_cc.append(va)

        valid_pairs.append((mfile, afile))
        if opts.verbose:
            print(f"✔️  Pair {idx}: Dice={row[0]}, vol_auto={va:.1f}cc")

    # Report skipped
    skipped = total_pairs - len(valid_pairs)
    print(f"ℹ️  Skipped {skipped}/{total_pairs} pairs.")

    if not data:
        print("⚠️  No valid pairs → exiting.")
        if opts.r:
            pd.DataFrame({"warning":["No valid pairs"]}).to_csv(opts.r, index=False)
        sys.exit(0)

    # Build DataFrame of metrics + volumes + relative diff
    df_out = pd.DataFrame(data, columns=measures)
    mf, af = zip(*valid_pairs)
    df_out.insert(0,'manual',mf)
    df_out.insert(1,'auto',  af)
    df_out['vol_manual_cc'] = manual_cc
    df_out['vol_auto_cc']   = auto_cc
    df_out['vol_rel_diff']  = (
        np.abs(df_out['vol_auto_cc'] - df_out['vol_manual_cc'])
        / df_out['vol_manual_cc'].replace({0:np.nan})
    )

    # Write metrics CSV
    if opts.r:
        df_out.to_csv(opts.r, index=False)

    # Reports: low-Dice & high-vol-diff
    write_low_dice(df_out, opts.dice_thresh,     opts.low_dice_csv)
    if opts.vol_diff_thresh is not None:
        write_high_vol_diff(df_out, opts.vol_diff_thresh, opts.high_vol_diff_csv)

    # Radar plot
    info = {
        'minimum': round(min(auto_cc),2) if min_val is None else min_val,
        'maximum': round(max(auto_cc),2) if max_val is None else max_val,
        'label':   opts.label,
        'unit':    opts.unit,
        'logplot': opts.log,
        'display': opts.display
    }
    if info['minimum'] == info['maximum']:
        info['maximum'] += 1e-6

    plot_radar(
        values    = np.asarray(data),
        colourmap = np.asarray(auto_cc),
        measures  = measures,
        info      = info,
        outfile   = opts.o
    )

    # Optional scatter plots
    if opts.sv:
        plot_volume_scatter(df_out, opts.sv)
    if opts.sd:
        plot_dice_vs_volume(df_out, opts.sd, opts.dice_thresh)


if __name__ == "__main__":
    # When run as: python -m eisrad or python __main__.py
    main()
