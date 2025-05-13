#!/usr/bin/env python
"""
:Summary:
EISRAD (Evaluation of Image Segmentations using RADar plots) compares binary segmentations between a reference (manual) and a candidate (auto) using multiple similarity metrics.
It produces a polar radar plot of results and exports numerical metrics to CSV.

:Example:
    python eisrad.py -f segmentations.csv -o radar.png -r metrics.csv -b

:Input:
    CSV file with two columns (manual, auto) containing full paths to segmentation files in NIfTI format.

:Output:
    - PNG radar plot visualizing similarity metrics across all pairs
    - Optional CSV with numeric values for each comparison

:Options:
    -f / --file       CSV input file (required)
    -o / --output     Output radar plot image (default: polar_results.png)
    -r / --results    Output CSV with numeric metrics
    -b / --binarize   Binarize input segmentations
    -d / --display    Display plot interactively
    -v / --verbose    Print file and volume info per comparison
    -m / --min        Min value for colorbar (optional)
    -M / --max        Max value for colorbar (optional)
    -L / --label      Label for colorbar
    -u / --unit       Unit for colorbar values
    -l / --log        Use logarithmic colorbar scaling

:Authors:
    Markus D. Schirmer
    MGH / Harvard Medical School

:Version: 1.1
:Date: 2024-05-13
:License: MIT
:Contact: software@markus-schirmer.com
"""

# =============================================
# Imports
# =============================================
import os
import sys
import pandas as pd
import numpy as np
import nibabel as nib
from optparse import OptionParser
from matplotlib import pyplot as plt

# Local imports (modularized)
from eisrad_metrics import get_values
from eisrad_plot import plot_evaluation

# =============================================
# Main execution
# =============================================
def main(argv):
    parser = OptionParser()
    parser.add_option('-f', '--file', dest='f', help='Input CSV file: manual,auto')
    parser.add_option('-o', '--output', dest='o', default='polar_results.png', help='Output radar image')
    parser.add_option('-r', '--results', dest='r', help='Optional output CSV with numeric metrics')
    parser.add_option('-m', '--min', dest='min', default=None)
    parser.add_option('-M', '--max', dest='max', default=None)
    parser.add_option('-L', '--label', dest='label', default='')
    parser.add_option('-u', '--unit', dest='unit', default='')
    parser.add_option('-l', '--log', action='store_true', default=False)
    parser.add_option('-d', '--display', action='store_true', default=False)
    parser.add_option('-v', '--verbose', action='store_true', default=False)
    parser.add_option('-b', '--binarize', action='store_true', default=False)
    (options, args) = parser.parse_args()

    if not options.f or not os.path.isfile(options.f):
        print("❌ Input CSV file not found.")
        sys.exit(1)

    if not os.path.isdir(os.path.dirname(options.o)) and os.path.dirname(options.o) != '':
        print(f"❌ Output directory not found: {os.path.dirname(options.o)}")
        sys.exit(1)

    try:
        min_val = float(options.min) if options.min else None
        max_val = float(options.max) if options.max else None
    except ValueError:
        print("❌ Colorbar min/max must be numeric.")
        sys.exit(1)

    df = pd.read_csv(options.f)
    if not {'manual', 'auto'}.issubset(df.columns):
        print("❌ CSV must contain 'manual' and 'auto' column headers.")
        sys.exit(1)

    file_list = df[["manual", "auto"]].values.tolist()

    measures = ['Dice', 'Jaccard', 'TPR', 'VS', 'MI', 'ARI', 'ICC',
                '1/(1+PBD)', 'KAP', '1-OER', '1-DER']

    data = []
    colourmap = []

    for i, (manual_file, auto_file) in enumerate(file_list):
        if not os.path.isfile(manual_file) or not os.path.isfile(auto_file):
            print(f"⚠️ Skipping missing file pair: {manual_file}, {auto_file}")
            continue

        A = nib.load(manual_file).get_fdata()
        B_img = nib.load(auto_file)
        B = B_img.get_fdata()

        if options.binarize:
            A = (A > 0).astype(int)
            B = (B > 0).astype(int)

        if A.shape != B.shape:
            print(f"⚠️ Shape mismatch: {manual_file} vs {auto_file}")
            continue

        values_dict = get_values(A, B, measures)
        values = [round(values_dict[m], 3) for m in measures]
        data.append(values)

        vol_cc = np.sum(B) * np.prod(B_img.header.get_zooms()) / 1000.0
        colourmap.append(vol_cc)

        if options.verbose:
            print(f"✔️ Pair #{i}: Volume = {vol_cc:.1f}cc")

    if not data:
        print("⚠️ No valid segmentation pairs found. Exiting.")
        if options.r:
            pd.DataFrame({"warning": ["No valid segmentation pairs found."]}).to_csv(options.r, index=False)
        sys.exit(0)

    if options.r:
        df_out = pd.DataFrame(data, columns=measures)
        df_out.insert(0, "manual", [row[0] for row in file_list])
        df_out.insert(1, "auto", [row[1] for row in file_list])
        df_out.to_csv(options.r, index=False)


    info = {
        'minimum': round(min(colourmap), 2) if min_val is None else min_val,
        'maximum': round(max(colourmap), 2) if max_val is None else max_val,
        'label': options.label,
        'unit': options.unit,
        'logplot': options.log,
        'display': options.display
    }

    if info['minimum'] == info['maximum']:
        info['maximum'] += 1e-6

    plot_evaluation(
        values=np.asarray(data),
        info=info,
        measures=measures,
        colourmap=np.asarray(colourmap),
        outfile=options.o
    )

if __name__ == "__main__":
    main(sys.argv)
