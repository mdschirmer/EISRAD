#!/usr/bin/env python
"""
:Summary:
This module defines similarity and agreement metrics used to compare binary segmentations in EISRAD.

Each function computes a commonly used statistical or spatial metric, with supporting logic for comparing true/false positive/negative relationships between a reference (manual) and candidate (automated) segmentation.

This module is intended to be imported by the main `eisrad.py` evaluation pipeline.

:Authors:
    Markus D. Schirmer
    MGH / Harvard Medical School

:Version: 1.1
:Date: 2024-05-13
:License: MIT
:Contact: software@markus-schirmer.com
"""

import numpy as np
import skimage.measure as skm

# === Base classification components ===

def get_truepos(A, B):
    """True positives: voxels correctly labeled as positive"""
    return float(np.sum(np.logical_and(B == 1, A == 1)))

def get_trueneg(A, B):
    """True negatives: voxels correctly labeled as negative"""
    return float(np.sum(np.logical_and(B == 0, A == 0)))

def get_falsepos(A, B):
    """False positives: voxels labeled as positive in B but negative in A"""
    return float(np.sum(np.logical_and(B == 1, A == 0)))

def get_falseneg(A, B):
    """False negatives: voxels labeled as negative in B but positive in A"""
    return float(np.sum(np.logical_and(A == 1, B == 0)))


# === Similarity and overlap metrics ===

def get_dice(A, B):
    """
    Dice coefficient (F1 score):
    Harmonic mean of precision and recall. Measures overlap between A and B.
    """
    TP, FP, FN = get_truepos(A, B), get_falsepos(A, B), get_falseneg(A, B)
    return (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0

def get_jaccard(A, B):
    """
    Jaccard index (IoU):
    Ratio of intersection to union of A and B. Similar to Dice, but penalizes FP/FN more.
    """
    TP, FP, FN = get_truepos(A, B), get_falsepos(A, B), get_falseneg(A, B)
    return TP / (TP + FP + FN) if (TP + FP + FN) else 0

def get_sensitivity(A, B):
    """
    True Positive Rate (Sensitivity, Recall):
    Proportion of positives in A correctly identified in B.
    """
    TP, FN = get_truepos(A, B), get_falseneg(A, B)
    return TP / (TP + FN) if (TP + FN) else 0

def get_specificity(A, B):
    """
    True Negative Rate (Specificity):
    Proportion of negatives in A correctly identified in B.
    """
    TN, FP = get_trueneg(A, B), get_falsepos(A, B)
    return TN / (TN + FP) if (TN + FP) else 0

def get_volumetric_similarity(A, B):
    """
    Volumetric Similarity (VS):
    Measures how similar the total segmented volume is between A and B.
    """
    TP, FP, FN = get_truepos(A, B), get_falsepos(A, B), get_falseneg(A, B)
    denom = 2 * TP + FP + FN
    return 1. - abs(FN - FP) / denom if denom else 0


# === Consistency and entropy-based metrics ===

def get_global_consistency_error(A, B):
    """
    1 - GCE (Global Consistency Error):
    Measures how well one segmentation can be viewed as a refinement of the other.
    """
    n = float(A.size)
    TP, TN = get_truepos(A, B), get_trueneg(A, B)
    FP, FN = get_falsepos(A, B), get_falseneg(A, B)

    E1 = (FN * (FN + 2 * TP) / (TP + FN) + FP * (FP + 2 * TN) / (TN + FP)) / n
    E2 = (FP * (FP + 2 * TP) / (TP + FP) + FN * (FN + 2 * TN) / (TN + FN)) / n
    return min(E1, E2)

def get_probabilities(A, B):
    n = float(A.size)
    TP, TN = get_truepos(A, B), get_trueneg(A, B)
    FP, FN = get_falsepos(A, B), get_falseneg(A, B)
    return [
        (TP + FN) / n, (TN + FN) / n, (TP + FP) / n, (TN + FP) / n,
        TP / n, FN / n, FP / n, TN / n
    ]

def get_log(p): return 0. if p == 0 else p * np.log2(p)

def get_MI(A, B):
    """
    Normalized Mutual Information (NMI), Variation of Information (VOI):
    Measures information shared between A and B, and distance between distributions.
    """
    p = [get_log(x) for x in get_probabilities(A, B)]
    H1 = -(p[0] + p[1])
    H2 = -(p[2] + p[3])
    H12 = -sum(p[4:])
    MI = H1 + H2 - H12
    VOI = (H1 + H2 - 2 * MI) / (2 * np.log2(2.))
    return 2 * MI / (H1 + H2) if (H1 + H2) else 0, VOI


# === Agreement metrics ===

def get_abcd(A, B):
    n = float(A.size)
    TP, TN = get_truepos(A, B), get_trueneg(A, B)
    FP, FN = get_falsepos(A, B), get_falseneg(A, B)
    a = 0.5 * (TP*(TP-1) + FP*(FP-1) + TN*(TN-1) + FN*(FN-1))
    b = 0.5 * ((TP+FN)**2 + (TN+FP)**2 - (TP**2 + TN**2 + FP**2 + FN**2))
    c = 0.5 * ((TP+FP)**2 + (TN+FN)**2 - (TP**2 + TN**2 + FP**2 + FN**2))
    d = n*(n-1)/2 - (a + b + c)
    return a, b, c, d

def get_rand_idx(A, B):
    """
    Adjusted Rand Index (ARI):
    Measures similarity of the assignments, adjusted for chance.
    """
    a, b, c, d = get_abcd(A, B)
    denom = c**2 + b**2 + 2 * a * d + (a + d) * (c + b)
    ARI = 2 * (a * d - b * c) / denom if denom else 0
    return (a + b) / (a + b + c + d), ARI

def get_ICC(A, B):
    """
    Intraclass Correlation Coefficient (ICC):
    Measures the consistency or reproducibility of quantitative measurements.
    """
    n = float(A.size)
    mean_img = (A + B) / 2.
    MS_w = np.sum((A - mean_img)**2 + (B - mean_img)**2) / n
    MS_b = 2 / (n - 1) * np.sum((mean_img - np.mean(mean_img))**2)
    denom = MS_b + MS_w
    return (MS_b - MS_w) / denom if denom else 0

def get_PBD(A, B):
    """
    Probabilistic Distance (PBD):
    Measures normalized voxel-wise disagreement. Higher is worse.
    """
    combined = np.sum(np.multiply(A, B))
    return 1.0 if combined == 0 else np.sum(np.abs(A - B)) / (2.0 * combined)

def get_KAP(A, B):
    """
    Cohen's Kappa:
    Measures agreement corrected for expected chance agreement.
    """
    n = float(A.size)
    TP, TN = get_truepos(A, B), get_trueneg(A, B)
    FP, FN = get_falsepos(A, B), get_falseneg(A, B)
    fa = TP + TN
    fc = 1. / n * ((TN + FN)*(TN+FP)+(FP+TP)*(FN+TP))
    denom = n - fc
    return (fa - fc) / denom if denom else 0


# === Distance-based measures ===

def directed_HD(A, B):
    coords_A = np.argwhere(A)
    coords_B = np.argwhere(B)
    if len(coords_A) == 0 or len(coords_B) == 0:
        return [np.inf]
    return [np.min(np.linalg.norm(coords_B - a, axis=1)) for a in coords_A]

def get_HD(A, B):
    """
    Hausdorff Distance:
    Maximal surface mismatch. Sensitive to outliers.
    """
    return max(np.max(directed_HD(A, B)), np.max(directed_HD(B, A)))

def get_AVD(A, B):
    """
    Average Surface Distance (AVD):
    Mean mismatch along the surface. More robust than Hausdorff.
    """
    return max(np.mean(directed_HD(A, B)), np.mean(directed_HD(B, A)))


# === Detection and outline error ===

def get_ODER(A, B):
    """
    Outline Error Rate (OER) & Detection Error Rate (DER):
    Partition errors into boundary mismatch vs missed detections.
    """
    MTA = (np.sum(A) + np.sum(B)) / 2.0
    intersect = A * B
    labels_A = skm.label(A)
    labels_B = skm.label(B)
    labels_A_ids = np.unique(np.multiply(intersect, labels_A))
    labels_B_ids = np.unique(np.multiply(intersect, labels_B))
    labels_A_only = [ii for ii in np.unique(labels_A) if ii not in labels_A_ids and ii > 0]
    labels_B_only = [ii for ii in np.unique(labels_B) if ii not in labels_B_ids and ii > 0]
    OE = sum(np.sum(labels_A == i) for i in labels_A_ids if i > 0) + \
         sum(np.sum(labels_B == i) for i in labels_B_ids if i > 0) - 2 * np.sum(intersect)
    DE = sum(np.sum(labels_A == i) for i in labels_A_only) + \
         sum(np.sum(labels_B == i) for i in labels_B_only)
    return OE / MTA if MTA else 0, DE / MTA if MTA else 0


# === Master dispatcher ===

def get_values(A, B, measures):
    """
    Dispatches calls to all enabled metric functions and returns a dict of results.
    """
    values = {}
    if 'Dice' in measures: values['Dice'] = get_dice(A, B)
    if 'Jaccard' in measures: values['Jaccard'] = get_jaccard(A, B)
    if 'TPR' in measures: values['TPR'] = get_sensitivity(A, B)
    if 'TNR' in measures: values['TNR'] = get_specificity(A, B)
    if 'VS' in measures: values['VS'] = get_volumetric_similarity(A, B)
    if '1-GCE' in measures: values['1-GCE'] = 1. - get_global_consistency_error(A, B)
    if 'MI' in measures or '1-VOI' in measures:
        NMI, VOI = get_MI(A, B)
        if 'MI' in measures: values['MI'] = NMI
        if '1-VOI' in measures: values['1-VOI'] = 1. - VOI
    if 'RI' in measures or 'ARI' in measures:
        RI, ARI = get_rand_idx(A, B)
        if 'RI' in measures: values['RI'] = RI
        if 'ARI' in measures: values['ARI'] = ARI
    if 'ICC' in measures: values['ICC'] = get_ICC(A, B)
    if '1/(1+PBD)' in measures: values['1/(1+PBD)'] = 1. / (1. + get_PBD(A, B))
    if 'KAP' in measures: values['KAP'] = get_KAP(A, B)
    if '1/(1+HD)' in measures: values['1/(1+HD)'] = 1. / (1. + get_HD(A, B))
    if '1/(1+AVD)' in measures: values['1/(1+AVD)'] = 1. / (1. + get_AVD(A, B))
    if '1-OER' in measures or '1-DER' in measures:
        OER, DER = get_ODER(A, B)
        if '1-OER' in measures: values['1-OER'] = 1. - OER
        if '1-DER' in measures: values['1-DER'] = 1. - DER
    return values
