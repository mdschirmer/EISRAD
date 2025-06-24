# EISRAD

**EISRAD** (Evaluation of Image Segmentations using RADar plots) is a flexible Python toolkit for
comparing binary segmentation masks against a reference standard. It computes a broad suite of
overlap- and agreement-based metrics, exports numeric results to CSV, and creates publication-quality
visualizations (radar plots, scatter plots) to help you diagnose strengths and failure modes
across your cohort.

*(Fun fact: “Eisrad” is the German word for “ice circle” — a rare swirling disk of ice seen on
Estonian rivers.)*

---

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/markus-schirmer/eisrad.git
   cd eisrad
   ```
2. Create and activate a Python 3 environment, e.g. using mamba/conda:
   ```bash
   mamba create -n eisrad python=3.11 pip
   mamba activate eisrad
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Key packages:* `numpy`, `pandas`, `nibabel`, `scikit-image`, `scikit-learn`, `matplotlib`.

---

## 📄 Usage

EISRAD’s command-line front-end is `eisrad` (or `python -m eisrad`). At minimum you need:

```bash
eisrad -f pair_list.csv -o radar.png -r metrics.csv
```

- **`-f, --file`**: CSV with two columns: `manual`,`auto` (full paths to NIfTI files).  
- **`-o, --output`**: Path to save the radar plot (default: `polar_results.png`).  
- **`-r, --results`**: Path to save numeric metrics as CSV.  

Run `eisrad -h` for the full list of options.

### Optional flags

- **`-b, --binarize`**: Threshold images `>0 → 1` before computing metrics.  
- **`-l, --log`**: Use logarithmic color scale on radar plot.  
- **`--dice-threshold`** *FLOAT* (default: 0.2): “low-Dice” cutoff for detection reporting.  
- **`--low-dice-csv`** *PATH*: Write CSV of all cases with Dice < threshold.  
- **`--vol-diff-threshold`** *FLOAT*: Relative volume-difference cutoff (e.g. 0.5 = 50%).  
- **`--high-vol-diff-csv`** *PATH*: Write CSV of all cases exceeding volume-diff cutoff.  
- **`--scatter-volume`** *PATH*: Save a manual-vs-auto volume log–log scatter plot.  
- **`--scatter-dice`** *PATH*: Save a Dice-vs-volume semi-log scatter plot.  

---

## 🔍 Examples

1. **Basic radar & metrics**  
   ```bash
   eisrad -f segmentations.csv -o cohort_radar.png -r cohort_metrics.csv
   ```

2. **Include log colorbar + binarization**  
   ```bash
   eisrad -f segmentations.csv -o radar_log.png -r metrics_log.csv -b -l
   ```

3. **Report low-Dice & high-volume-diff cases**  
   ```bash
   eisrad      -f segmentations.csv      -r cohort_metrics.csv      --dice-threshold 0.2      --low-dice-csv low_dice_cases.csv      --vol-diff-threshold 0.5      --high-vol-diff-csv high_vol_diff_cases.csv
   ```

4. **Generate scatter diagnostics**  
   ```bash
   eisrad      -f segmentations.csv      --scatter-volume vol_scatter.png      --scatter-dice dice_scatter.png
   ```

5. **Full evaluation with all plots & reports**  
   ```bash
   eisrad      -f bkupcomparison.csv      -o comparison/radar.png      -r comparison/metrics.csv      -l      --dice-threshold 0.2      --low-dice-csv comparison/low_dice.csv      --vol-diff-threshold 0.5      --high-vol-diff-csv comparison/high_vol_diff.csv      --scatter-volume comparison/vol_scatter.png      --scatter-dice comparison/dice_scatter.png
   ```
   This command produces the following in your `comparison/` folder:

   - **Radar plot (log scale)**  
     ![Radar Plot](comparison/radar.png)

   - **Metrics CSV**: all numeric metrics + volumes + relative differences  
     `comparison/metrics.csv`

   - **Low-Dice cases (Dice < 0.2)**  
     `comparison/low_dice.csv`

   - **High volume-difference cases (rel diff > 0.5)**  
     `comparison/high_vol_diff.csv`

   - **Volume scatter (manual vs auto)**  
     ![Volume Scatter](comparison/vol_scatter.png)

   - **DICE vs. volume scatter**  
     ![DICE Scatter](comparison/dice_scatter.png)

---

## 📂 Module Structure

```
eisrad/
├─ __main__.py        # CLI & orchestration
├─ io.py              # load + reorient NIfTIs
├─ metrics.py         # Dice, Jaccard, TPR, MI, ARI, ICC, etc.
├─ reports.py         # low-Dice & high-vol-diff CSV/report
└─ plots.py           # radar, volume-scatter, dice-scatter
```

---

## 🧊 Credits

Developed by Markus D. Schirmer  
Massachusetts General Hospital / Harvard Medical School

---

## 📝 License

Released under the MIT License.
