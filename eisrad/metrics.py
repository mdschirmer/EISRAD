#!/usr/bin/env python
"""
:Summary:
Defines segmentation comparison metrics between two binary numpy arrays,
including overlap metrics (Dice, Jaccard, TPR, VS),
agreement indices (ARI, ICC, Cohen’s Kappa),
distance measures (HD, AVD),
and information‐theoretic (NMI) and probabilistic (PBD) metrics.

:Authors:
    Markus D. Schirmer
    MGH / Harvard Medical School

:License: MIT
"""

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    cohen_kappa_score
)

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

def get_MI(A, B):
    """Normalized Mutual Information (NMI) between two binary masks."""
    flatA = (A > 0).ravel().astype(int)
    flatB = (B > 0).ravel().astype(int)
    return normalized_mutual_info_score(flatA, flatB)


# ─── Agreement & correlation ───

def get_ARI(A, B):
    """Adjusted Rand Index between two binary masks."""
    flatA = (A > 0).ravel().astype(int)
    flatB = (B > 0).ravel().astype(int)
    return adjusted_rand_score(flatA, flatB)

def get_ICC(A, B):
    """Intraclass correlation coefficient (two-way mixed ICC(3,1))."""
    n = A.size
    mean_img = (A + B) / 2.0
    MS_w = np.sum((A - mean_img)**2 + (B - mean_img)**2) / n
    MS_b = 2.0/(n - 1) * np.sum((mean_img - mean_img.mean())**2)
    denom = MS_b + MS_w
    return (MS_b - MS_w)/denom if denom else 1.0

def get_KAP(A, B):
    """Cohen’s kappa between two binary masks (1.0 if both are empty)."""
    flatA = (A > 0).ravel().astype(int)
    flatB = (B > 0).ravel().astype(int)
    if not flatA.any() and not flatB.any():
        return 1.0
    return cohen_kappa_score(flatA, flatB)


# ─── Probabilistic & distance metrics ───

def get_PBD(A, B):
    """
    Probabilistic distance: normalized voxel-wise disagreement.
    0 if both masks are empty, inf if they are non-empty but do not overlap.
    """
    A_bin = (A > 0)
    B_bin = (B > 0)
    intersection = np.sum(A_bin & B_bin)
    if intersection == 0:
        return 0.0 if not (A_bin.any() or B_bin.any()) else np.inf
    return float(np.sum(np.abs(A_bin.astype(int) - B_bin.astype(int))) 
                 / (2.0 * intersection))

def get_HD(A, B):
    """Symmetric Hausdorff distance between two binary masks."""
    A_pts = np.argwhere(A > 0)
    B_pts = np.argwhere(B > 0)
    if A_pts.size == 0 or B_pts.size == 0:
        return 0.0
    d1 = directed_hausdorff(A_pts, B_pts)[0]
    d2 = directed_hausdorff(B_pts, A_pts)[0]
    return max(d1, d2)

def get_AVD(A, B):
    """Average symmetric surface distance between two binary masks."""
    A_pts = np.argwhere(A > 0)
    B_pts = np.argwhere(B > 0)
    if A_pts.size == 0 or B_pts.size == 0:
        return 0.0
    tree_A = cKDTree(A_pts)
    tree_B = cKDTree(B_pts)
    dA, _ = tree_B.query(A_pts)
    dB, _ = tree_A.query(B_pts)
    return float((dA.mean() + dB.mean()) / 2.0)


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
    with MTA = (|A| + |B|) / 2. Both are 0 when A and B are empty. Either
    can exceed 1, so 1-OER / 1-DER can be negative.
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

def get_lesion_coverage(A, B):
    """
    Per-lesion coverage: for each connected component (26-connectivity) of the
    reference mask A, the fraction of its voxels that are also set in B.
    Returns a 1-D array with one value in [0, 1] per lesion (empty if A is empty).
    """
    A_bin = (A > 0)
    B_bin = (B > 0)
    labels, n = _label26(A_bin)
    if n == 0:
        return np.zeros(0)
    size    = np.bincount(labels.ravel(), minlength=n+1)[1:]
    covered = np.bincount(labels[B_bin], minlength=n+1)[1:]
    return covered / size


# ─── Dispatcher ───

def get_values(A, B, measures):
    """
    Compute and return a dict of {metric_name: value} for each name in measures.
    """
    vals = {}
    for m in measures:
        if   m == 'Dice':        vals[m] = get_dice(A, B)
        elif m == 'Jaccard':     vals[m] = get_jaccard(A, B)
        elif m == 'TPR':         vals[m] = get_tpr(A, B)
        elif m == 'VS':          vals[m] = get_vs(A, B)
        elif m == 'MI':          vals[m] = get_MI(A, B)
        elif m == 'ARI':         vals[m] = get_ARI(A, B)
        elif m == 'ICC':         vals[m] = get_ICC(A, B)
        elif m == '1/(1+PBD)':   vals[m] = 1.0/(1.0 + get_PBD(A, B))
        elif m == 'KAP':         vals[m] = get_KAP(A, B)
        elif m == '1-OER' or m=='1-DER':
            OER, DER = get_ODER(A, B)
            if m == '1-OER': vals[m] = 1.0 - OER
            else:            vals[m] = 1.0 - DER
        else:
            raise ValueError(f"Unknown measure: {m}")
    return vals
