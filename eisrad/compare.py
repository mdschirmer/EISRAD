#!/usr/bin/env python
"""
:Summary:
Compare several algorithms from the outputs of separate EISRAD runs
(metrics.csv + metrics_lesions.csv per algorithm). Patients are matched by
their manual mask path, lesions by (manual mask path, lesion number), so all
comparisons are paired. Writes statistics tables and comparison plots.

:Example:
    eisrad-compare -i SALSA=salsa/metrics.csv -i nnUNet=nnunet/metrics.csv \\
                   -o comparison/ --format pdf

:Authors:
    Markus D. Schirmer
    MGH / Harvard Medical School

:License: MIT
"""

import os
import re
import sys
from itertools import combinations
from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy import stats

from eisrad import __version__

# label, CSV column or function of the metrics table, higher is better,
# bounded to 0-1 (for plot axes), shown on the radar overlay
PATIENT_METRICS = [
    ('Dice',             'Dice',             True,  True,  True),
    ('Jaccard',          'Jaccard',          True,  True,  True),
    ('TPR',              'TPR',              True,  True,  True),
    ('VS',               'VS',               True,  True,  True),
    ('MI',               'MI',               True,  True,  True),
    ('ARI',              'ARI',              True,  True,  True),
    ('ICC',              'ICC',              True,  True,  True),
    ('KAP',              'KAP',              True,  True,  True),
    ('OER',              lambda d: 2 * (1 - d['1-OER/2']), False, False, False),
    ('DER',              lambda d: 2 * (1 - d['1-DER/2']), False, False, False),
    ('HD95 (L) [mm]',    'HD95_mm (L)',      False, False, False),
    ('SDice (L)',        'SDice (L)',        True,  True,  True),
]
# radar overlay uses the 0-1 scores, as the single-algorithm radar does
PATIENT_RADAR = ['Dice', 'Jaccard', 'TPR', 'VS', 'MI', 'ARI', 'ICC', 'KAP',
                 '1-OER/2', '1-DER/2', '1/(1+HD95/r) (L)', 'SDice (L)']

LESION_METRICS = [
    ('Dice',        'Dice',    True,  True),
    ('Jaccard',     'Jaccard', True,  True),
    ('TPR',         'TPR',     True,  True),
    ('VS',          'VS',      True,  True),
    ('HD95 [mm]',   'HD95_mm', False, False),
    ('SDice',       'SDice',   True,  True),
]
LESION_RADAR = ['Dice', 'Jaccard', 'TPR', 'VS', '1/(1+HD95/r)', 'SDice']


# ─── Loading and matching ───

def _lesions_path(path):
    root, ext = os.path.splitext(path)
    return root + '_lesions' + ext

def load_runs(inputs):
    """
    inputs: list of (name, metrics CSV path). Returns {name: (metrics, lesions)}
    with lesions None if no *_lesions.csv exists next to the metrics CSV.
    """
    runs = {}
    for name, path in inputs:
        if not os.path.isfile(path):
            sys.exit(f"❌ Metrics CSV not found for {name}: {path}")
        met = pd.read_csv(path)
        if met['manual'].duplicated().any():
            n = int(met['manual'].duplicated().sum())
            print(f"⚠️  {name}: {n} duplicate manual paths, keeping the first.")
            met = met.drop_duplicates('manual')
        lp  = _lesions_path(path)
        les = pd.read_csv(lp) if os.path.isfile(lp) else None
        if les is None:
            print(f"⚠️  {name}: no lesion table ({lp}); lesion-level "
                  f"comparisons skipped.")
        runs[name] = (met.set_index('manual'), les)
    return runs

MATCH_KEYS = {
    'path':     lambda p: p,
    'filename': lambda p: os.path.basename(p),
    'BIDS sub-ID': lambda p: (re.search(r'sub-[A-Za-z0-9]+', p) or [None])[0],
}

def _key_func(pattern):
    """Patient ID from a manual path with a user regex (group 1 if present)."""
    rx = re.compile(pattern)
    def key(p):
        m = rx.search(p)
        return (m.group(1) if m.groups() else m.group(0)) if m else None
    return key

def choose_match_key(runs, id_pattern=None):
    """
    Decide how patients are identified across runs. With `id_pattern`, that
    regex is used. Otherwise the candidates in MATCH_KEYS are tried and the
    one matching the most patients is used; a key only qualifies if it is
    found for every patient and unique within each run. Returns (name, func).
    """
    if id_pattern:
        cands = {f"--id-pattern '{id_pattern}'": _key_func(id_pattern)}
    else:
        cands = MATCH_KEYS
    best = (None, None, -1)
    for name, f in cands.items():
        keysets, ok = [], True
        for met, _ in runs.values():
            keys = [f(p) for p in met.index]
            if any(k is None for k in keys) or len(set(keys)) != len(keys):
                ok = False
                break
            keysets.append(set(keys))
        if not ok:
            continue
        n = len(set.intersection(*keysets))
        if n > best[2]:
            best = (name, f, n)
        if name == 'path' and n > 0:
            break              # exact paths match: no need to guess
    if best[0] is None:
        sys.exit("❌ Could not derive a unique patient ID from the manual paths;"
                 " use --id-pattern REGEX (e.g. 'sub-[0-9]+').")
    return best[0], best[1]

def apply_match_key(runs, key):
    """Re-index metrics and lesion tables by the patient key."""
    out = {}
    for name, (met, les) in runs.items():
        met = met.copy()
        met.index = [key(p) for p in met.index]
        met.index.name = 'manual'
        if les is not None:
            les = les.copy()
            les['manual'] = [key(p) for p in les['manual']]
        out[name] = (met, les)
    return out

def match_patients(runs):
    """Patients present in every run (input order); reports the rest."""
    names  = list(runs)
    common = set.intersection(*(set(m.index) for m, _ in runs.values()))
    for n in names:
        extra = len(runs[n][0].index.difference(common))
        if extra:
            print(f"⚠️  {n}: {extra} patient(s) missing from other runs, "
                  f"left out of the comparison.")
    order = [p for p in runs[names[0]][0].index if p in common]
    print(f"ℹ️  {len(order)} patients matched across {len(names)} algorithms.")
    return order

def patient_table(runs, patients, column):
    """DataFrame (patients x algorithms) of one metric."""
    cols = {}
    for name, (met, _) in runs.items():
        sub = met.loc[patients]
        cols[name] = column(sub) if callable(column) else sub[column]
    return pd.DataFrame(cols, index=patients)

def lesion_table(runs, patients, column):
    """
    DataFrame (manual lesions x algorithms) of one lesion metric, indexed by
    (manual path, lesion number); only lesions of matched patients.
    """
    cols = {}
    for name, (_, les) in runs.items():
        m = les[(les['type'] == 'manual') & les['manual'].isin(patients)]
        cols[name] = m.set_index(['manual', 'lesion'])[column]
    return pd.DataFrame(cols)

def fp_counts(runs, patients):
    """False-positive lesions per patient (patients x algorithms)."""
    cols = {}
    for name, (_, les) in runs.items():
        fp = les[les['type'] == 'false_positive'].groupby('manual').size()
        cols[name] = fp.reindex(patients, fill_value=0)
    return pd.DataFrame(cols, index=patients)


# ─── Statistics ───

def holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaN kept as NaN)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size == 0:
        return out
    order = ok[np.argsort(p[ok])]
    m = len(order)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out[order] = np.minimum(adj, 1.0)
    return out

EFFECTS = {'median': ('median of differences', np.median),
           'mean':   ('mean difference',       np.mean)}

def bootstrap_ci(diff, stat=np.median, groups=None, n_boot=2000, seed=0):
    """
    95% bootstrap CI of `stat` (median or mean) of the differences. With
    `groups` (patient per value), whole patients are resampled so values of
    one patient stay together.
    """
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    if groups is None:
        idx  = rng.integers(0, len(diff), (n_boot, len(diff)))
        boot = stat(diff[idx], axis=1)
    else:
        groups = np.asarray(groups)
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
        boot = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            boot[b] = stat(diff[np.concatenate([members[g] for g in pick])])
    return tuple(np.percentile(boot, [2.5, 97.5]))

def _effect(row, diff, kind, groups, n_boot, seed, scale=1.0):
    """Fill effect name, difference (A-B) and its CI into a result row."""
    name, stat = EFFECTS[kind]
    row['effect'] = name
    if len(diff) >= 2:
        row['diff (A-B)'] = scale * float(stat(diff))
        lo, hi = bootstrap_ci(diff, stat, groups, n_boot, seed)
        row['CI low'], row['CI high'] = scale * lo, scale * hi
    else:
        row['diff (A-B)'] = row['CI low'] = row['CI high'] = np.nan

def _summaries(sub, a, b):
    """Median and mean of both algorithms on the paired subset."""
    f = lambda fn, c: float(fn(sub[c])) if len(sub) else np.nan
    return {'median A': f(np.median, a), 'median B': f(np.median, b),
            'mean A': f(np.mean, a), 'mean B': f(np.mean, b)}

def wilcoxon_p(diff):
    """Two-sided Wilcoxon signed-rank p-value (1.0 if all differences are 0)."""
    diff = np.asarray(diff, dtype=float)
    if not np.any(diff != 0):
        return 1.0
    return float(stats.wilcoxon(diff).pvalue)

def cluster_permutation_p(diff, groups, n_perm=5000, seed=0):
    """
    Two-sided paired permutation test for clustered data: the sign of all
    differences of one patient is flipped together, so lesions of the same
    patient are not treated as independent. Statistic: sum of differences.
    """
    diff = np.asarray(diff, dtype=float)
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(inv, weights=diff, minlength=len(uniq))
    obs  = abs(sums.sum())
    if obs == 0:
        return 1.0
    rng   = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(uniq)))
    null  = np.abs(signs @ sums)
    return float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1))

def mcnemar_p(a, b):
    """Exact McNemar test for paired binary outcomes."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
    if n01 + n10 == 0:
        return 1.0
    return float(stats.binomtest(n10, n01 + n10, 0.5).pvalue)

def cochran_q_p(X):
    """Cochran's Q test for k paired binary outcomes (rows = subjects)."""
    X = np.asarray(X, dtype=float)
    k = X.shape[1]
    col, row = X.sum(axis=0), X.sum(axis=1)
    N = X.sum()
    denom = k * N - np.sum(row ** 2)
    if denom == 0:
        return 1.0
    Q = (k - 1) * (k * np.sum(col ** 2) - N ** 2) / denom
    return float(stats.chi2.sf(Q, k - 1))


def compare_patient_metric(df, label, higher, n_boot, seed, effect='median'):
    """
    Omnibus (Friedman, k >= 3) and pairwise Wilcoxon rows for one metric.
    `effect`: 'median' (median of differences, matches Wilcoxon) or 'mean'
    (for counts with many ties, e.g. false-positive lesions).
    """
    names = list(df.columns)
    full  = df.dropna()
    omni = {'level': 'patient', 'metric': label, 'n': len(full),
            **{f'median {n}': np.median(full[n]) if len(full) else np.nan
               for n in names},
            'test': 'Friedman' if len(names) >= 3 else '', 'p': np.nan}
    if len(names) >= 3 and len(full) >= 2:
        if np.ptp(full.values, axis=1).any():      # some patient differs
            omni['p'] = float(stats.friedmanchisquare(
                *[full[n] for n in names]).pvalue)
        else:
            omni['p'] = 1.0
    pairs = []
    for a, b in combinations(names, 2):
        sub  = df[[a, b]].dropna()
        diff = (sub[a] - sub[b]).to_numpy()
        row = {'level': 'patient', 'metric': label, 'A': a, 'B': b,
               'n': len(sub), **_summaries(sub, a, b),
               'test': 'Wilcoxon signed-rank'}
        _effect(row, diff, effect, None, n_boot, seed)
        row['p'] = wilcoxon_p(diff) if len(sub) >= 2 else np.nan
        row['better'] = _better(row, higher)
        pairs.append(row)
    return omni, pairs

def compare_lesion_metric(df, label, higher, n_boot, n_perm, seed):
    """
    Pairwise patient-clustered permutation rows for one lesion metric. The
    effect is the mean difference (the quantity the permutation test tests),
    with a patient-bootstrap CI; medians of differences are often 0 here
    because many lesions get identical values from both algorithms.
    """
    names, pairs = list(df.columns), []
    for a, b in combinations(names, 2):
        sub    = df[[a, b]].dropna()
        diff   = (sub[a] - sub[b]).to_numpy()
        groups = sub.index.get_level_values('manual').to_numpy()
        row = {'level': 'lesion', 'metric': label, 'A': a, 'B': b,
               'n': len(sub), 'patients': len(np.unique(groups)),
               **_summaries(sub, a, b),
               'test': 'patient-clustered permutation'}
        _effect(row, diff, 'mean', groups, n_boot, seed)
        row['p'] = (cluster_permutation_p(diff, groups, n_perm, seed)
                    if len(sub) >= 2 else np.nan)
        row['better'] = _better(row, higher)
        pairs.append(row)
    return pairs

def _better(row, higher):
    """
    Better algorithm, following the reported effect `diff (A-B)` so the two
    columns never contradict each other. Only when that effect is exactly 0
    (e.g. a median of differences with most differences 0) are the means
    (or % detected) used to show the direction.
    """
    d = row.get('diff (A-B)', np.nan)
    if not np.isfinite(d) or d == 0:
        a = row.get('mean A', row.get('% detected A', np.nan))
        b = row.get('mean B', row.get('% detected B', np.nan))
        if not (np.isfinite(a) and np.isfinite(b)) or a == b:
            return ''
        d = a - b
    return row['A'] if (d > 0) == higher else row['B']


def compare_detection(runs, patients, n_boot, n_perm, seed):
    """
    Detection (any overlap) of patients with a manual lesion, and of lesions:
    Cochran's Q (k >= 3) + pairwise exact McNemar for patients, pairwise
    patient-clustered permutation for lesions.
    """
    names = list(runs)
    omni, pairs = [], []

    pat = patient_table(runs, patients, 'TPR')
    vol = runs[names[0]][0].loc[patients, 'vol_manual_ml']
    det = (pat[vol > 0] > 0)
    o = {'level': 'patient', 'metric': 'Detected (any overlap)', 'n': len(det),
         **{f'median {n}': 100.0 * det[n].mean() for n in names},
         'test': "Cochran's Q" if len(names) >= 3 else '', 'p': np.nan}
    if len(names) >= 3 and len(det):
        o['p'] = cochran_q_p(det.values)
    omni.append(o)
    for a, b in combinations(names, 2):
        diff = (det[a].astype(float) - det[b].astype(float)).to_numpy()
        row = {'level': 'patient', 'metric': 'Detected (any overlap)',
               'A': a, 'B': b, 'n': len(det),
               '% detected A': 100.0 * det[a].mean(),
               '% detected B': 100.0 * det[b].mean(),
               'test': 'exact McNemar', 'p': mcnemar_p(det[a], det[b])}
        _effect(row, diff, 'mean', None, n_boot, seed, scale=100.0)
        row['effect'] = 'difference in % detected'
        row['better'] = _better(row, True)
        pairs.append(row)

    if all(les is not None for _, les in runs.values()):
        ld = lesion_table(runs, patients, 'detected')
        ld = ld.apply(lambda c: c.astype(str).str.lower() == 'true')
        groups = ld.index.get_level_values('manual').to_numpy()
        for a, b in combinations(names, 2):
            diff = (ld[a].astype(float) - ld[b].astype(float)).to_numpy()
            row = {'level': 'lesion', 'metric': 'Detected (any overlap)',
                   'A': a, 'B': b, 'n': len(ld),
                   'patients': len(np.unique(groups)),
                   '% detected A': 100.0 * ld[a].mean(),
                   '% detected B': 100.0 * ld[b].mean(),
                   'test': 'patient-clustered permutation',
                   'p': cluster_permutation_p(diff, groups, n_perm, seed)}
            _effect(row, diff, 'mean', groups, n_boot, seed, scale=100.0)
            row['effect'] = 'difference in % detected'
            row['better'] = _better(row, True)
            pairs.append(row)
    return omni, pairs


def run_comparison(runs, patients, n_boot=2000, n_perm=5000, seed=0):
    """
    All statistics. Returns (omnibus DataFrame, pairwise DataFrame) with
    Holm-corrected p-values: omnibus tests across metrics, pairwise tests
    across all pairs and metrics, each within a level (patient / lesion).
    """
    omni, pairs = [], []
    for label, col, higher, _, _ in PATIENT_METRICS:
        o, p = compare_patient_metric(patient_table(runs, patients, col),
                                      label, higher, n_boot, seed)
        omni.append(o); pairs += p
    has_lesions = all(les is not None for _, les in runs.values())
    if has_lesions:
        o, p = compare_patient_metric(fp_counts(runs, patients),
                                      'False-positive lesions', False,
                                      n_boot, seed, effect='mean')
        omni.append(o); pairs += p
        for label, col, higher, _ in LESION_METRICS:
            pairs += compare_lesion_metric(lesion_table(runs, patients, col),
                                           label, higher, n_boot, n_perm, seed)
    o, p = compare_detection(runs, patients, n_boot, n_perm, seed)
    omni += o; pairs += p

    omni  = pd.DataFrame(omni)
    pairs = pd.DataFrame(pairs)
    omni['p (Holm)'] = holm(omni['p'])
    pairs['p (Holm)'] = np.nan
    for lvl in pairs['level'].unique():
        m = pairs['level'] == lvl
        pairs.loc[m, 'p (Holm)'] = holm(pairs.loc[m, 'p'])
    return omni, pairs


# ─── Command line ───

def _parse_input(option, opt_str, value, parser):
    if '=' not in value:
        raise SystemExit(f"❌ {opt_str} expects NAME=PATH, got '{value}'")
    name, path = value.split('=', 1)
    parser.values.inputs.append((name.strip(), path.strip()))

def main(argv=None):
    parser = OptionParser(
        usage='%prog -i NAME=metrics.csv -i NAME=metrics.csv [...] -o OUTDIR',
        version=f"eisrad-compare {__version__}")
    parser.set_defaults(inputs=[])
    parser.add_option('-i', '--input', type='string', action='callback',
                      callback=_parse_input,
                      help='Algorithm name and its EISRAD metrics CSV, as '
                           'NAME=PATH (repeat for each algorithm; the '
                           '*_lesions.csv next to it is used automatically)')
    parser.add_option('-o', '--outdir', dest='outdir', default='comparison',
                      help='Output folder (default: comparison)')
    parser.add_option('--format', dest='fmt', default='png',
                      help='Image format for plots (png, webp, pdf, svg, ...)')
    parser.add_option('--n-boot', dest='n_boot', type='int', default=2000,
                      help='Bootstrap resamples for CIs (default: 2000)')
    parser.add_option('--n-perm', dest='n_perm', type='int', default=5000,
                      help='Permutations for lesion-level tests (default: 5000)')
    parser.add_option('--id-pattern', dest='id_pattern', default=None,
                      help='Regex that extracts the patient ID from the manual '
                           'mask path (group 1 if given), e.g. "sub-[0-9]+". '
                           'Default: automatic (full path, file name or BIDS '
                           'sub-ID, whichever matches most patients)')
    parser.add_option('--seed', dest='seed', type='int', default=0,
                      help='Random seed (default: 0)')
    opts, _ = parser.parse_args(argv[1:] if argv is not None else None)

    if len(opts.inputs) < 2:
        parser.error('give at least two algorithms with -i NAME=PATH')
    names = [n for n, _ in opts.inputs]
    if len(set(names)) != len(names):
        parser.error('algorithm names must be unique')
    os.makedirs(opts.outdir, exist_ok=True)

    runs     = load_runs(opts.inputs)
    key_name, key = choose_match_key(runs, opts.id_pattern)
    print(f"ℹ️  Patients matched by {key_name}.")
    runs     = apply_match_key(runs, key)
    patients = match_patients(runs)
    if len(patients) < 2:
        sys.exit("❌ Fewer than two patients in common; nothing to compare.")

    # Tests follow from the number of algorithms: with two, the paired tests
    # are the comparison (an omnibus test would be redundant); with three or
    # more, omnibus tests across all algorithms come first, then pairs.
    if len(names) == 2:
        print("ℹ️  2 algorithms → paired tests only: Wilcoxon signed-rank "
              "(patient metrics), exact McNemar (patients detected), "
              "patient-clustered permutation (lesions).")
    else:
        print(f"ℹ️  {len(names)} algorithms → omnibus tests (Friedman, "
              f"Cochran's Q) across all, then pairwise tests for all "
              f"{len(names) * (len(names) - 1) // 2} pairs.")

    omni, pairs = run_comparison(runs, patients, opts.n_boot, opts.n_perm,
                                 opts.seed)
    omni_path = os.path.join(opts.outdir, 'comparison_omnibus.csv')
    if len(names) >= 3:
        omni.to_csv(omni_path, index=False)
    elif os.path.exists(omni_path):
        os.remove(omni_path)           # stale file from an earlier 3+ run
    pairs.to_csv(os.path.join(opts.outdir, 'comparison_pairwise.csv'), index=False)

    from eisrad import compare_plots as cp
    out = lambda name: os.path.join(opts.outdir, f'{name}.{opts.fmt.lstrip(".")}')
    cp.plot_radar_overlay(runs, patients, out('compare_radar'))
    cp.plot_compare_boxes(runs, patients, pairs, out('compare_boxes'))
    cp.plot_detection_overlay(runs, patients, out('compare_detection'),
                              opts.n_boot, opts.seed)
    if all(les is not None for _, les in runs.values()):
        cp.plot_radar_overlay(runs, patients, out('compare_radar_lesion'),
                              lesion=True)
        cp.plot_compare_boxes(runs, patients, pairs, out('compare_boxes_lesion'),
                              lesion=True)
        cp.plot_detection_overlay(runs, patients, out('compare_detection_lesion'),
                                  opts.n_boot, opts.seed, lesion=True)
    print(f"✅ Results written to {opts.outdir}/")


if __name__ == '__main__':
    main()
