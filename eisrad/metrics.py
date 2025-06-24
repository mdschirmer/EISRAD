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

:Version: 1.3
:Date: 2025-06-24
:License: MIT
"""

import numpy as np
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
    """True positive rate: TP / (TP + FN)."""
    TP = get_truepos(A, B)
    FN = get_falseneg(A, B)
    return TP/(TP+FN) if (TP+FN) else 0.0

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
    """Cohen’s kappa between two binary masks."""
    flatA = (A > 0).ravel().astype(int)
    flatB = (B > 0).ravel().astype(int)
    return cohen_kappa_score(flatA, flatB)


# ─── Probabilistic & distance metrics ───

def get_PBD(A, B):
    """Probabilistic distance: normalized voxel-wise disagreement."""
    A_bin = (A > 0)
    B_bin = (B > 0)
    intersection = np.sum(A_bin & B_bin)
    if intersection == 0:
        return 0.0
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
            # Outline & Detection Error Rates
            # OER = FP/(TP+FP), DER = FN/(TP+FN)
            TP = get_truepos(A,B)
            FP = get_falsepos(A,B)
            FN = get_falseneg(A,B)
            OER = FP/(TP+FP) if (TP+FP) else 0.0
            DER = FN/(TP+FN) if (TP+FN) else 0.0
            if m == '1-OER': vals[m] = 1.0 - OER
            else:            vals[m] = 1.0 - DER
        else:
            raise ValueError(f"Unknown measure: {m}")
    return vals
