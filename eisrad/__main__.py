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

:License: MIT
"""

import sys
import os
import numpy  as np
import pandas as pd
import nibabel as nib
from optparse import OptionParser
from concurrent.futures import ProcessPoolExecutor

# Absolute imports to avoid shadowing std-lib io
from eisrad         import __version__
from eisrad.io      import load_and_reorient_pair
from eisrad.metrics import get_values, get_lesion_table, LESION_MEASURES
from eisrad.reports import write_low_dice, write_high_vol_diff
from eisrad.plots   import (plot_radar, plot_metric_bars, plot_volume_scatter,
                            plot_dice_vs_volume, plot_detection_curve)


def _threshold_callback(option, opt_str, value, parser):
    """-b [T]: take the next argument as threshold T if it is a number."""
    threshold = 0.0
    if parser.rargs:
        try:
            threshold = float(parser.rargs[0])
            del parser.rargs[0]
        except ValueError:
            pass
    setattr(parser.values, option.dest, threshold)


def _suffixed(path, suffix):
    """'out/plot.png', '_lesion' -> 'out/plot_lesion.png'"""
    root, ext = os.path.splitext(path)
    return root + suffix + ext


PLOT_FORMATS = ('png', 'webp', 'jpg', 'jpeg', 'pdf', 'svg', 'tif', 'tiff')


def _available_cpus():
    """CPUs this process may use, respecting CPU affinity limits."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:          # not available on macOS / Windows
        return os.cpu_count() or 1


def _process_pair(task):
    """
    Load one manual/auto pair and compute everything EISRAD needs from it.
    Runs in a worker process with -j > 1, so it returns plain data and the
    messages to print instead of printing itself.
    """
    idx, mfile, afile, binarize, measures, tolerance_mm = task
    res = {'skip': True, 'messages': [], 'mfile': mfile, 'afile': afile}
    if not os.path.isfile(mfile) or not os.path.isfile(afile):
        res['messages'].append(f"⚠️  Skipping missing: {mfile}, {afile}")
        return res
    res['messages'].append(f"ℹ️  Loading file pair: {mfile}, {afile}")

    # Load & reorient both, optionally binarize
    A, zooms = load_and_reorient_pair(mfile, binarize)
    B, _     = load_and_reorient_pair(afile, binarize)

    if A.shape != B.shape:
        res['messages'].append(f"⚠️  Shape mismatch: {mfile} vs {afile}")
        return res

    # Metrics and volumes assume 0/1 masks: binarize others at 0.5
    res['binarized'] = []
    if not np.isin(A, (0, 1)).all():
        A = (A > 0.5).astype(np.uint8)
        res['binarized'].append(mfile)
    if not np.isin(B, (0, 1)).all():
        B = (B > 0.5).astype(np.uint8)
        res['binarized'].append(afile)

    # Per-lesion table (manual lesions + false-positive automated blobs);
    # lesion TPR is the coverage used by the detection curve
    les, fp_vol = get_lesion_table(A, B, zooms, tolerance_mm)

    # Compute similarity metrics (lesion-averaged ones reuse the table)
    vals_dict  = get_values(A, B, measures + ['HD95_mm (L)'], zooms,
                            tolerance_mm, lesions=les)
    res['row'] = [round(vals_dict[m],3) for m in measures]
    res['hd95_mm'] = vals_dict['HD95_mm (L)']

    # Compute volumes (mL)
    res['vm'] = A.sum() * np.prod(zooms) / 1000.0
    res['va'] = B.sum() * np.prod(zooms) / 1000.0

    res['coverage'] = les['TPR']
    # Patient-level coverage: share of the whole manual lesion mask found
    res['pat_cov'] = (np.array([vals_dict['TPR']]) if les['TPR'].size
                      else np.zeros(0))
    rows = []
    for i in range(len(les['vol_manual_ml'])):
        lrow = {'manual': mfile, 'auto': afile, 'lesion': i + 1,
                'type': 'manual'}
        lrow.update({k: les[k][i] for k in les})
        rows.append(lrow)
    for j, v in enumerate(fp_vol):
        rows.append({'manual': mfile, 'auto': afile,
                     'lesion': j + 1, 'type': 'false_positive',
                     'vol_manual_ml': 0.0, 'vol_auto_ml': v,
                     'detected': np.nan, 'Dice': 0.0,
                     'Jaccard': 0.0, 'TPR': np.nan, 'VS': 0.0,
                     'HD95_mm': np.nan, 'SDice': np.nan})
    res['lesion_rows'] = rows
    res['skip'] = False
    return res


def main(argv=None):
    """
    argv: list of command-line arguments (including script name). If None,
          OptionParser.parse_args will read from sys.argv.
    """
    parser = OptionParser(version=f"eisrad {__version__}")
    # Core options
    parser.add_option('-f','--file',     dest='f',    help='Input CSV (manual,auto)')
    parser.add_option('-o','--output',   dest='o',
                      default='polar_results.png', help='Radar plot output file; format from the extension '
                           '(.png, .webp, .jpg, .pdf, .svg)')
    parser.add_option('-r','--results',  dest='r',    help='Metrics CSV output file')
    parser.add_option('-b','--binarize', action='callback', dest='binarize',
                      callback=_threshold_callback, default=None,
                      help='Usage: -b [T]. Binarize all masks at > T '
                           '(-b alone: T = 0). '
                           'Without -b, only non-binary masks are binarized, '
                           'at > 0.5')
    parser.add_option('-d','--display',  action='store_true', default=False,
                      help='Show plots interactively')
    parser.add_option('--surface-tolerance', dest='surf_tol', type='float',
                      default=None,
                      help='Surface Dice tolerance in mm (default: one voxel '
                           'in every direction, adapting to the resolution)')
    parser.add_option('--format',            dest='fmt',
                      help='Image format for all plots (png, webp, jpg, pdf, '
                           'svg); replaces the extension of every plot file')
    parser.add_option('-j','--jobs',     dest='jobs', type='int',
                      default=1,
                      help='Number of file pairs processed in parallel '
                           '(default: 1; this machine has '
                           f'{_available_cpus()} CPUs available)')
    parser.add_option('-v','--verbose',  action='store_true', default=False,
                      help='Print per-pair details')
    parser.add_option('-m','--min',      dest='min',   default=None,
                      help='Colorbar minimum value')
    parser.add_option('-M','--max',      dest='max',   default=None,
                      help='Colorbar maximum value')
    parser.add_option('-L','--label',    dest='label',
                      default='Manual lesion volume',
                      help='Colorbar label (default: Manual lesion volume)')
    parser.add_option('-u','--unit',     dest='unit',  default='mL',
                      help='Colorbar unit (default: mL)')
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
    parser.add_option('--volume-heatmap', '--scatter-volume', dest='sv',
                      help='Output manual vs auto volume heatmap')
    parser.add_option('--dice-heatmap', '--scatter-dice', dest='sd',
                      help='Output Dice vs volume heatmap')
    parser.add_option('--bar-plot',            dest='bar',
                      help='Output metric bar plot, alternative to radar')
    parser.add_option('--detection-curve',     dest='dc',
                      help='Output detection vs coverage curve')

    # Parse arguments
    if argv is not None:
        opts, args = parser.parse_args(argv[1:])
    else:
        opts, args = parser.parse_args()

    # --format: replace the extension of every plot output
    if opts.fmt:
        fmt = opts.fmt.lower().lstrip('.')
        if fmt not in PLOT_FORMATS:
            print(f"❌ Unsupported --format '{opts.fmt}'; "
                  f"use one of: {', '.join(PLOT_FORMATS)}")
            sys.exit(1)
        for attr in ('o', 'bar', 'sv', 'sd', 'dc'):
            path = getattr(opts, attr)
            if path:
                setattr(opts, attr, os.path.splitext(path)[0] + '.' + fmt)

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
                   'KAP','1-OER/2','1-DER/2','1/(1+HD95/r) (L)','SDice (L)']
    data        = []
    manual_ml   = []
    hd95_mm     = []
    auto_ml     = []
    coverages   = []
    pat_cov     = []
    lesion_rows = []
    valid_pairs = []
    binarized   = []

    # Process each pair (optionally in parallel; results keep input order)
    tasks = [(idx, mfile, afile, opts.binarize, measures, opts.surf_tol)
             for idx, (mfile, afile) in enumerate(file_list)]
    if opts.jobs < 1:
        print("❌ -j/--jobs must be at least 1.")
        sys.exit(1)
    if opts.jobs > 1:
        pool    = ProcessPoolExecutor(max_workers=opts.jobs)
        results = pool.map(_process_pair, tasks, chunksize=4)
    else:
        pool    = None
        results = map(_process_pair, tasks)

    for idx, res in enumerate(results):
        for msg in res['messages']:
            print(msg)
        if res['skip']:
            continue
        binarized.extend(res['binarized'])
        data.append(res['row'])
        hd95_mm.append(res['hd95_mm'])
        manual_ml.append(res['vm'])
        auto_ml.append(res['va'])
        coverages.append(res['coverage'])
        pat_cov.append(res['pat_cov'])
        lesion_rows.extend(res['lesion_rows'])
        valid_pairs.append((res['mfile'], res['afile']))
        if opts.verbose:
            print(f"✔️  Pair {idx}: Dice={res['row'][0]}, "
                  f"vol_auto={res['va']:.1f}mL")
    if pool is not None:
        pool.shutdown()

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
    df_out['HD95_mm (L)']   = hd95_mm
    df_out['vol_manual_ml'] = manual_ml
    df_out['vol_auto_ml']   = auto_ml
    df_out['vol_rel_diff']  = (
        np.abs(df_out['vol_auto_ml'] - df_out['vol_manual_ml'])
        / df_out['vol_manual_ml'].replace({0:np.nan})
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
        'minimum': round(min(manual_ml),2) if min_val is None else min_val,
        'maximum': round(max(manual_ml),2) if max_val is None else max_val,
        'label':   opts.label,
        'unit':    opts.unit,
        'logplot': opts.log,
        'display': opts.display
    }
    if info['minimum'] == info['maximum']:
        info['maximum'] += 1e-6

    # ARI / ICC / KAP can dip below 0 (worse than chance); clamp for the
    # plots so all spokes share the 0-1 scale. metrics.csv keeps raw values.
    raw_vals  = np.asarray(data, dtype=float)
    plot_vals = np.clip(raw_vals, 0, None)
    n_neg     = (raw_vals < 0).sum(axis=0)

    plot_radar(
        values    = plot_vals,
        colourmap = np.asarray(manual_ml),
        measures  = measures,
        info      = info,
        outfile   = opts.o
    )

    # Optional plots
    if opts.bar:
        col = {m: raw_vals[:, i] for i, m in enumerate(measures)}
        bar_cols = [  # label, values, higher is better, bounded to 0-1
            ('Dice',            col['Dice'],             True,  True),
            ('Jaccard',         col['Jaccard'],          True,  True),
            ('TPR',             col['TPR'],              True,  True),
            ('VS',              col['VS'],               True,  True),
            ('MI',              col['MI'],               True,  True),
            ('ARI',             col['ARI'],              True,  True),
            ('ICC',             col['ICC'],              True,  True),
            ('KAP',             col['KAP'],              True,  True),
            ('OER',             2 * (1 - col['1-OER/2']), False, False),
            ('DER',             2 * (1 - col['1-DER/2']), False, False),
            ('HD95 (L)\n[mm]', np.asarray(hd95_mm),      False, False),
            ('SDice (L)',       col['SDice (L)'],        True,  True),
        ]
        plot_metric_bars(np.column_stack([c[1] for c in bar_cols]),
                         np.asarray(manual_ml), [c[0] for c in bar_cols],
                         info, opts.bar,
                         higher_better=[c[2] for c in bar_cols],
                         bounded=[c[3] for c in bar_cols])
    if opts.sv:
        plot_volume_scatter(df_out, opts.sv)
    if opts.sd:
        plot_dice_vs_volume(df_out, opts.sd, opts.dice_thresh)
    if opts.dc:
        plot_detection_curve(pat_cov, opts.dc, unit='patients')
        plot_detection_curve(coverages, _suffixed(opts.dc, '_lesion'),
                             unit='lesions')

    # Per-lesion versions
    df_les = pd.DataFrame(lesion_rows, columns=[
        'manual', 'auto', 'lesion', 'type', 'vol_manual_ml',
        'vol_auto_ml', 'detected'] + LESION_MEASURES)
    if opts.r:
        df_les.to_csv(_suffixed(opts.r, '_lesions'), index=False)
    man   = df_les[df_les['type'] == 'manual']
    n_fp  = int((df_les['type'] == 'false_positive').sum())
    print(f"ℹ️  {len(man)} manual lesions, {n_fp} false-positive lesions.")

    if man.empty:
        print("⚠️  No lesions in manual masks → skipping per-lesion "
              "radar, bar plot and Dice heatmap.")
    else:
        # Lesion radar: median + IQR only (thousands of dots would hide them);
        # HD95 / SDice are NaN for missed lesions and left out of their spokes
        r_mm = (3.0 * man['vol_manual_ml'].to_numpy() * 1000.0
                / (4.0 * np.pi)) ** (1.0 / 3.0)
        les_radar = np.column_stack(
            [man[m].to_numpy(dtype=float) for m in ['Dice', 'Jaccard', 'TPR', 'VS']]
            + [1.0 / (1.0 + man['HD95_mm'].to_numpy(dtype=float) / r_mm),
               man['SDice'].to_numpy(dtype=float)])
        plot_radar(les_radar, None,
                   ['Dice', 'Jaccard', 'TPR', 'VS', '1/(1+HD95/r)', 'SDice'],
                   info, _suffixed(opts.o, '_lesion'), show_points=False,
                   density=True, density_label='% of lesions')

        if opts.bar:
            vols = man['vol_manual_ml'].to_numpy()
            info_les = dict(info,
                            minimum=round(vols.min(), 2),
                            maximum=round(vols.max(), 2),
                            label='Manual lesion volume', unit='mL')
            if info_les['minimum'] == info_les['maximum']:
                info_les['maximum'] += 1e-6
            les_labels = ['Dice', 'Jaccard', 'TPR', 'VS', 'HD95\n[mm]',
                          'SDice']
            plot_metric_bars(man[LESION_MEASURES].to_numpy(dtype=float),
                             vols, les_labels, info_les,
                             _suffixed(opts.bar, '_lesion'),
                             higher_better=[True]*4 + [False, True],
                             bounded=[True]*4 + [False, True])
        if opts.sd:
            plot_dice_vs_volume(
                man, _suffixed(opts.sd, '_lesion'), opts.dice_thresh,
                unit='lesions',
                extra_stats=[f'False-positive lesions = {n_fp}'])
    if opts.sv and not df_les.empty:
        plot_volume_scatter(df_les, _suffixed(opts.sv, '_lesion'),
                            lesion=True)

    if n_neg.any():
        counts = ', '.join(f"{m}: {c}" for m, c in zip(measures, n_neg) if c)
        print(f"⚠️  Negative metric values were clamped to 0 in the radar plot "
              f"(metrics CSV keeps raw values) — {counts}")
    if binarized:
        print(f"⚠️  {len(binarized)} non-binary mask(s) were binarized at 0.5 "
              f"(> 0.5 → 1); use -b [T] to binarize all masks at > T:")
        for f in binarized:
            print(f"     {f}")


if __name__ == "__main__":
    # When run as: python -m eisrad or python __main__.py
    main()
