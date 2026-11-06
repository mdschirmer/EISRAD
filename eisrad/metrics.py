#!/usr/bin/env python
"""
:Summary:
Defines segmentation comparison metrics between two binary numpy arrays,
including overlap metrics (Dice, Jaccard, TPR, VS),
agreement indices (ARI, ICC, Cohen’s Kappa),
distance measures (HD, AVD),
and information‐theoretic (NMI) metrics. PBD is not provided: for binary
masks 1/(1+PBD) equals Dice.

:Authors:
    Markus D. Schirmer
    MGH / Harvard Medical School

:License: MIT
"""

import numpy as np
from scipy import ndimage

# ─── Basic confusion‐matrix counts ───

def get_truepos(A, B):
    A_bin = (A > 0)
    B_bin = (B > 0)
    return float(np.sum(A_bin & B_bin))

def get_falsepos(A, B):
    A_bin = (A > 0)
    B_bin = (B > 0)
    return float(np.sum(~A_bin & B_bin))

def get_falseneg(A, B):
    A_bin = (A > 0)
    B_bin = (B > 0)
    return float(np.sum(A_bin & ~B_bin))

def get_trueneg(A, B):
    A_bin = (A > 0)
    B_bin = (B > 0)
    return float(np.sum(~A_bin & ~B_bin))


# ─── Overlap & similarity metrics ───

def get_dice(A, B):
    """Dice coefficient: 2|A∩B| / (|A| + |B|)."""
    TP = get_truepos(A, B)
    FP = get_falsepos(A, B)
    FN = get_falseneg(A, B)
    denom = 2*TP + FP + FN
    return 2*TP/denom if denom else 1.0

def get_jaccard(A, B):
    """Jaccard index: |A∩B| / |A∪B|."""
    TP = get_truepos(A, B)
    FP = get_falsepos(A, B)
    FN = get_falseneg(A, B)
    denom = TP + FP + FN
    return TP/denom if denom else 1.0

def get_tpr(A, B):
    """True positive rate: TP / (TP + FN); 1.0 if A is empty (nothing to miss)."""
    TP = get_truepos(A, B)
    FN = get_falseneg(A, B)
    return TP/(TP+FN) if (TP+FN) else 1.0

def get_vs(A, B):
    """Volumetric similarity: 1 − |FP − FN|/(2TP + FP + FN)."""
    TP = get_truepos(A, B)
    FP = get_falsepos(A, B)
    FN = get_falseneg(A, B)
    denom = 2*TP + FP + FN
    return 1 - abs(FP - FN)/denom if denom else 1.0

def _contingency(A, B):
    """2x2 voxel contingency table [[TN, FP], [FN, TP]] as Python ints."""
    TP, FP = int(get_truepos(A, B)), int(get_falsepos(A, B))
    FN, TN = int(get_falseneg(A, B)), int(get_trueneg(A, B))
    return [[TN, FP], [FN, TP]]

def _entropy(counts, n):
    p = np.array([c for c in counts if c > 0], dtype=float) / n
    return float(-(p * np.log(p)).sum())

def get_MI(A, B):
    """
    Normalized Mutual Information 2·I(A;B) / (H(A) + H(B)) between two binary
    masks, computed from the 2x2 contingency table (same values as
    sklearn.metrics.normalized_mutual_info_score); 1.0 if both are constant.
    """
    C = _contingency(A, B)
    n = sum(map(sum, C))
    rows = [sum(C[0]), sum(C[1])]
    cols = [C[0][0] + C[1][0], C[0][1] + C[1][1]]
    hA, hB = _entropy(rows, n), _entropy(cols, n)
    if hA == 0 and hB == 0:
        return 1.0
    mi = sum(C[i][j] / n * np.log(C[i][j] * n / (rows[i] * cols[j]))
             for i in range(2) for j in range(2) if C[i][j])
    return float(2 * max(mi, 0.0) / (hA + hB))


# ─── Agreement & correlation ───

def get_ARI(A, B):
    """
    Adjusted Rand Index between two binary masks, from voxel-pair counts of
    the 2x2 contingency table in exact integer arithmetic (same values as
    sklearn.metrics.adjusted_rand_score).
    """
    C = _contingency(A, B)
    n = sum(map(sum, C))
    rows = [sum(C[0]), sum(C[1])]
    cols = [C[0][0] + C[1][0], C[0][1] + C[1][1]]
    sum_sq = sum(c * c for r in C for c in r)
    tp = sum_sq - n
    fp = sum(C[i][j] * cols[j] for i in range(2) for j in range(2)) - sum_sq
    fn = sum(C[i][j] * rows[i] for i in range(2) for j in range(2)) - sum_sq
    tn = n * n - fp - fn - sum_sq
    if fn == 0 and fp == 0:
        return 1.0
    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))

def get_ICC(A, B):
    """Intraclass correlation coefficient, one-way ICC(1,1) for two raters."""
    A = (A > 0).astype(float)      # arithmetic below must not run on bools
    B = (B > 0).astype(float)
    n = A.size
    mean_img = (A + B) / 2.0
    MS_w = np.sum((A - mean_img)**2 + (B - mean_img)**2) / n
    MS_b = 2.0/(n - 1) * np.sum((mean_img - mean_img.mean())**2)
    denom = MS_b + MS_w
    return (MS_b - MS_w)/denom if denom else 1.0

def get_KAP(A, B):
    """
    Cohen’s kappa (p_o − p_e) / (1 − p_e) between two binary masks, from the
    2x2 contingency table (same values as sklearn.metrics.cohen_kappa_score);
    1.0 if both are empty.
    """
    (TN, FP), (FN, TP) = _contingency(A, B)
    if TP + FP + FN == 0:
        return 1.0
    n  = TP + FP + FN + TN
    po = (TP + TN) / n
    pe = ((TP + FN) * (TP + FP) + (TN + FN) * (TN + FP)) / (n * n)
    return float((po - pe) / (1 - pe)) if pe != 1 else float('nan')


# ─── Probabilistic & distance metrics ───

def _surface(M):
    """Surface voxels of a binary mask: voxels with a background face-neighbour."""
    inner = ndimage.binary_erosion(M, ndimage.generate_binary_structure(M.ndim, 1))
    return M & ~inner

def _surface_dists(A_bin, B_bin, zooms):
    """
    Distances from each surface voxel of A to the surface of B and vice
    versa, in mm and in voxel steps (chessboard: 1 = within the 26-neighbourhood),
    computed in the bounding box of A ∪ B. Both masks must be non-empty.
    """
    box = ndimage.find_objects((A_bin | B_bin).astype(np.uint8))[0]
    sA, sB = _surface(A_bin[box]), _surface(B_bin[box])
    mm  = (ndimage.distance_transform_edt(~sB, sampling=zooms)[sA],
           ndimage.distance_transform_edt(~sA, sampling=zooms)[sB])
    vox = (ndimage.distance_transform_cdt(~sB, metric='chessboard')[sA],
           ndimage.distance_transform_cdt(~sA, metric='chessboard')[sB])
    return mm, vox

def get_surface_metrics(A, B, zooms=None, tolerance_mm=None):
    """
    Boundary agreement between two binary masks:

    HD95   95th percentile of all surface-to-surface distances (both
           directions pooled), in mm.
    SDice  surface Dice: share of surface voxels of A and B lying within the
           tolerance of the other surface; by default within one voxel in
           every direction (26-neighbourhood, adapts to the image resolution),
           or within `tolerance_mm` mm if given.

    Returns (HD95, SDice): (0, 1) if both masks are empty, (inf, 0) if
    exactly one is.
    """
    A_bin = (A > 0)
    B_bin = (B > 0)
    if zooms is None:
        zooms = (1.0,) * A_bin.ndim
    if not A_bin.any() and not B_bin.any():
        return 0.0, 1.0
    if not A_bin.any() or not B_bin.any():
        return np.inf, 0.0

    (mm_ab, mm_ba), (vox_ab, vox_ba) = _surface_dists(A_bin, B_bin, zooms)
    hd95 = np.percentile(np.concatenate([mm_ab, mm_ba]), 95)
    if tolerance_mm is None:
        ok_ab, ok_ba = vox_ab <= 1, vox_ba <= 1
    else:
        ok_ab, ok_ba = mm_ab <= tolerance_mm, mm_ba <= tolerance_mm
    sdice = (ok_ab.sum() + ok_ba.sum()) / (ok_ab.size + ok_ba.size)
    return float(hd95), float(sdice)


# ─── Lesion-wise detection ───

def _label26(M):
    """Connected components with 26-connectivity (full in any ndim)."""
    return ndimage.label(M, structure=np.ones((3,) * M.ndim, dtype=bool))

def get_ODER(A, B):
    """
    Outline Error Rate (OER) and Detection Error Rate (DER), lesion-wise
    (Wack et al., 2012). Lesions are connected components (26-connectivity);
    a lesion is detected if it overlaps any lesion in the other mask.
      OER = (voxels of detected lesions in A and B − 2|A∩B|) / MTA
      DER = (voxels of lesions found in only one mask) / MTA
    with MTA = (|A| + |B|) / 2. Both are 0 when A and B are empty. Both lie
    in [0, 2] and OER + DER = 2 (1 − Dice), so EISRAD reports 1 − OER/2 and
    1 − DER/2, which lie in [0, 1].
    """
    A_bin = (A > 0)
    B_bin = (B > 0)
    MTA = (A_bin.sum() + B_bin.sum()) / 2.0
    if MTA == 0:
        return 0.0, 0.0
    inter = A_bin & B_bin

    OE, DE = -2.0 * inter.sum(), 0.0
    for M in (A_bin, B_bin):
        labels, n = _label26(M)
        size    = np.bincount(labels.ravel(), minlength=n+1)[1:]
        matched = np.zeros(n, dtype=bool)
        hit     = np.unique(labels[inter])
        matched[hit[hit > 0] - 1] = True
        OE += size[matched].sum()
        DE += size[~matched].sum()
    return float(OE / MTA), float(DE / MTA)

LESION_MEASURES = ['Dice', 'Jaccard', 'TPR', 'VS', 'HD95_mm', 'SDice']

def get_lesion_table(A, B, zooms=None, tolerance_mm=None):
    """
    Lesion-wise comparison with the manual mask A as reference.

    Lesions are the connected components (26-connectivity) of A. Automated
    blobs (components of B) that touch any manual lesion are shared out
    voxel by voxel to the nearest manual lesion (Euclidean distance in mm,
    using `zooms`), so a blob spanning two lesions is split rather than
    counted twice, and several blobs on one lesion are pooled. Automated
    blobs touching no manual lesion are false-positive lesions.

    Returns
    -------
    lesions : dict of 1-D arrays, one entry per manual lesion:
        vol_manual_ml, vol_auto_ml, detected (any overlap), and the
        LESION_MEASURES computed from the lesion vs. its assigned voxels.
        HD95_mm and SDice (see get_surface_metrics) are NaN for lesions
        that were not detected.
    fp_vol_ml : 1-D array, volume (mL) of each false-positive lesion.
    """
    A_bin = (A > 0)
    B_bin = (B > 0)
    if zooms is None:
        zooms = (1.0,) * A_bin.ndim
    voxvol = float(np.prod(zooms)) / 1000.0

    la, na = _label26(A_bin)
    lb, nb = _label26(B_bin)
    hit = np.unique(lb[A_bin & B_bin])
    B_touch = np.isin(lb, hit[hit > 0])

    fp_size = np.bincount(lb[B_bin & ~B_touch], minlength=nb+1)[1:]
    fp_vol_ml = fp_size[fp_size > 0] * voxvol

    if na == 0:
        empty = {k: np.zeros(0) for k in
                 ['vol_manual_ml', 'vol_auto_ml', 'detected'] + LESION_MEASURES}
        return empty, fp_vol_ml

    # Nearest manual lesion for every touching automated voxel, computed in
    # the bounding box of A ∪ touching blobs (contains every manual lesion)
    box = ndimage.find_objects((A_bin | B_touch).astype(np.uint8))[0]
    sub_la = la[box]
    _, idx = ndimage.distance_transform_edt(sub_la == 0, sampling=zooms,
                                            return_indices=True)
    owner = sub_la[tuple(idx)]

    size_a = np.bincount(la.ravel(), minlength=na+1)[1:].astype(float)
    size_b = np.bincount(owner[B_touch[box]], minlength=na+1)[1:].astype(float)
    tp     = np.bincount(la[A_bin & B_bin], minlength=na+1)[1:].astype(float)
    fp, fn = size_b - tp, size_a - tp

    # Boundary measures per detected lesion: lesion vs. its assigned voxels
    assigned = np.zeros(la.shape, dtype=la.dtype)
    assigned[box][B_touch[box]] = owner[B_touch[box]]
    objs_a = ndimage.find_objects(la)
    objs_b = ndimage.find_objects(assigned, max_label=na)
    hd95 = np.full(na, np.nan)
    sdice = np.full(na, np.nan)
    for i in np.flatnonzero(tp > 0):
        sl = tuple(slice(min(p.start, q.start), max(p.stop, q.stop))
                   for p, q in zip(objs_a[i], objs_b[i]))
        hd95[i], sdice[i] = get_surface_metrics(la[sl] == i + 1,
                                                assigned[sl] == i + 1,
                                                zooms, tolerance_mm)

    with np.errstate(divide='ignore', invalid='ignore'):
        lesions = {
            'vol_manual_ml': size_a * voxvol,
            'vol_auto_ml':   size_b * voxvol,
            'detected':      tp > 0,
            'Dice':          2*tp / (2*tp + fp + fn),
            'Jaccard':       tp / (tp + fp + fn),
            'TPR':           tp / size_a,
            'VS':            1 - np.abs(fp - fn) / (2*tp + fp + fn),
            'HD95_mm':       hd95,
            'SDice':         sdice,
        }
    return lesions, fp_vol_ml


def lesion_surface_summary(lesions, auto_empty):
    """
    Patient-level boundary measures from a lesion table, averaged over the
    detected manual lesions with lesion volume as weight (missed lesions are
    left out; they count in 1-DER/2 and the detection curves):

      '1/(1+HD95/r) (L)'          weighted mean of 1 / (1 + HD95_i / r_i), with r_i
                          the radius (mm) of a sphere with lesion i's volume
      'SDice (L)'         weighted mean of SDice_i
      'HD95_mm (L)'       weighted mean of HD95_i in mm

    Scores are 1 if neither mask has a lesion and 0 if no manual lesion was
    detected (or the manual mask is empty but the automated one is not);
    HD95_mm (L) is then 0 or NaN.
    """
    det = np.asarray(lesions['detected'], dtype=bool)
    if det.size == 0:
        perfect = bool(auto_empty)
        return {'1/(1+HD95/r) (L)': float(perfect), 'SDice (L)': float(perfect),
                'HD95_mm (L)': 0.0 if perfect else np.nan}
    if not det.any():
        return {'1/(1+HD95/r) (L)': 0.0, 'SDice (L)': 0.0, 'HD95_mm (L)': np.nan}
    w    = lesions['vol_manual_ml'][det]
    hd   = lesions['HD95_mm'][det]
    r    = (3.0 * w * 1000.0 / (4.0 * np.pi)) ** (1.0 / 3.0)   # mL -> mm³
    mean = lambda x: float(np.sum(w * x) / np.sum(w))
    return {'1/(1+HD95/r) (L)':    mean(1.0 / (1.0 + hd / r)),
            'SDice (L)':   mean(lesions['SDice'][det]),
            'HD95_mm (L)': mean(hd)}


# ─── Dispatcher ───

def get_values(A, B, measures, zooms=None, tolerance_mm=None, lesions=None):
    """
    Compute and return a dict of {metric_name: value} for each name in measures.
    """
    A, B = (A > 0), (B > 0)    # binarize once; every metric thresholds at > 0
    vals = {}
    oder = None
    dist = None
    for m in measures:
        if   m == 'Dice':        vals[m] = get_dice(A, B)
        elif m == 'Jaccard':     vals[m] = get_jaccard(A, B)
        elif m == 'TPR':         vals[m] = get_tpr(A, B)
        elif m == 'VS':          vals[m] = get_vs(A, B)
        elif m == 'MI':          vals[m] = get_MI(A, B)
        elif m == 'ARI':         vals[m] = get_ARI(A, B)
        elif m == 'ICC':         vals[m] = get_ICC(A, B)
        elif m == 'KAP':         vals[m] = get_KAP(A, B)
        elif m in ('1/(1+HD95/r) (L)', 'SDice (L)', 'HD95_mm (L)'):
            if dist is None:   # lesion-averaged; reuse a given lesion table
                if lesions is None:
                    lesions, _ = get_lesion_table(A, B, zooms, tolerance_mm)
                dist = lesion_surface_summary(lesions, auto_empty=not B.any())
            vals[m] = dist[m]
        elif m in ('1-OER/2', '1-DER/2'):
            if oder is None:
                oder = get_ODER(A, B)      # shared by both
            OER, DER = oder
            vals[m] = 1.0 - (OER if m == '1-OER/2' else DER) / 2.0
        else:
            raise ValueError(f"Unknown measure: {m}")
    return vals
