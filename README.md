# EISRAD

**EISRAD** (Evaluation of Image Segmentations using RADar plots) is a Python toolkit designed to
quantitatively assess how well automated segmentation algorithms reproduce a manual (reference)
segmentation. It computes a variety of metrics and generates plots to help visualize algorithm
performance across a cohort.

---

## Scientific Overview

Below is an example **radar plot** summarizing multiple similarity metrics (e.g., Dice, Jaccard,
true positive rate) for each case in a cohort. Points are colored by the automated segmentation
volume on a logarithmic scale. The solid blue line indicates the median performance across cases,
and the shaded blue band represents the interquartile range (IQR), highlighting variability.

![Radar Plot Example](radar.png)

Next, a **volume agreement scatter plot** shows how manual and automated lesion volumes correlate.
Each point represents one case; perfect agreement would lie on the diagonal line. The Pearson
correlation coefficient quantifies overall agreement, while the inset confusion matrix reports the
number of cases where both volumes are zero (no lesion) or nonzero.

![Volume Scatter Example](vol_scatter.png)

Finally, the **DICE vs. lesion volume scatter plot** illustrates how algorithm accuracy (Dice)
depends on lesion size. The red dashed line marks the detection threshold (e.g., Dice = 0.2), and both
Pearson and Spearman correlations are reported.

![DICE vs Volume Scatter Example](dice_scatter.png)
---

## Installation

You can either install EISRAD into your Python environment or run it directly without installation.

### 1. Editable install

```bash
git clone https://github.com/mdschirmer/EISRAD.git
cd EISRAD
python3 -m venv venv
source venv/bin/activate
pip install -e .            # or: pip install -e ".[test]" to also run the tests
```

This installs the dependencies (listed in `pyproject.toml`) and makes the
`eisrad` command available in your shell. Check the installed version with
`eisrad --version`.

### 2. Run without installing

Install the dependencies listed in `pyproject.toml` (e.g. `pip install matplotlib nibabel numpy pandas scikit-learn scipy`),
then, from the project root (where the `eisrad/` folder lives), run:

```bash
# Using module invocation
PYTHONPATH=. python -m eisrad [OPTIONS]

# Or directly with the entrypoint script
python eisrad/__main__.py [OPTIONS]
```

No need for `pip install` if you prefer in-place execution.

---

## Usage

At minimum, provide an input CSV and output paths:

```bash
eisrad -f pair_list.csv -o radar.png -r metrics.csv
```

- **`-f, --file`**: CSV file with `manual,auto` columns pointing to NIfTI mask paths.  
- **`-o, --output`**: Output radar plot image (default `polar_results.png`).  
- **`-r, --results`**: Output CSV for numeric metrics.

Run `eisrad -h` for the full list of flags.

### Key options

- `-b, --binarize`: Threshold volumes `>0 → 1`.  
- `-l, --log`: Enable log color scale on radar.  
- `--dice-threshold FLOAT`: Low-Dice cutoff (default `0.2`).  
- `--low-dice-csv PATH`: Save cases with Dice < threshold.  
- `--vol-diff-threshold FLOAT`: Relative volume-diff cutoff.  
- `--high-vol-diff-csv PATH`: Save cases exceeding vol-diff.  
- `--volume-heatmap PATH`: Save manual vs auto volume heatmap with marginal volume histograms (`--scatter-volume` still works as an alias).  
- `--dice-heatmap PATH`: Save Dice vs volume heatmap with marginal Dice histogram (`--scatter-dice` still works as an alias).
- `--bar-plot PATH`: Save metric bar plot (median + IQR, cases as points), an alternative to the radar.
- `--detection-curve PATH`: Save lesion detection rate vs. coverage threshold curve.

---

## Examples

### Basic radar & metrics

```bash
eisrad -f segs.csv -o radar.png -r metrics.csv
```

### Full evaluation with all outputs

```bash
eisrad \
  -f comparison.csv \
  -o comparison/radar.png \
  -r comparison/metrics.csv \
  -l \
  --dice-threshold 0.2 \
  --low-dice-csv comparison/low_dice.csv \
  --vol-diff-threshold 0.5 \
  --high-vol-diff-csv comparison/high_vol_diff.csv \
  --volume-heatmap comparison/vol_heatmap.png \
  --dice-heatmap comparison/dice_heatmap.png \
  --bar-plot comparison/bars.png \
  --detection-curve comparison/detection.png
```

This generates:

- **Radar plot (log scale)**: `comparison/radar.png`  
- **Metrics CSV**: `comparison/metrics.csv`  
- **Low-Dice cases**: `comparison/low_dice.csv`  
- **High vol-diff cases**: `comparison/high_vol_diff.csv`  
- **Volume heatmap**: `comparison/vol_heatmap.png`  
- **DICE heatmap**: `comparison/dice_heatmap.png`  
- **Metric bar plot**: `comparison/bars.png`  
- **Detection curve**: `comparison/detection.png`  

---

## Project Structure

```
eisrad/
├─ __main__.py
├─ io.py
├─ metrics.py
├─ reports.py
└─ plots.py
README.md
pyproject.toml
```

---

## License

Released under the MIT License.
