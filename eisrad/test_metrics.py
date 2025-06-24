# test_metrics.py

import numpy as np
import pytest

from eisrad.metrics import (
    get_truepos, get_falsepos, get_falseneg, get_trueneg,
    get_dice, get_jaccard, get_tpr, get_vs,
    get_MI, get_ARI, get_ICC, get_PBD, get_KAP,
    get_HD, get_AVD
)

@pytest.fixture
def simple_masks():
    """
    A = [[1, 0],
         [1, 0]]
    B = [[1, 1],
         [0, 0]]

    TP=1, FP=1, FN=1, TN=1
    """
    A = np.array([[1, 0],
                  [1, 0]], dtype=int)
    B = np.array([[1, 1],
                  [0, 0]], dtype=int)
    return A, B

@pytest.fixture
def identical_masks():
    """
    A = [[1, 0],
         [0, 0]]
    B = A
    """
    A = np.array([[1, 0],
                  [0, 0]], dtype=int)
    return A, A.copy()

@pytest.mark.parametrize("func,expected", [
    (get_truepos,  1),
    (get_falsepos, 1),
    (get_falseneg, 1),
    (get_trueneg,  1),
])
def test_confusion_counts(simple_masks, func, expected):
    A, B = simple_masks
    assert func(A, B) == expected

@pytest.mark.parametrize("func,expected", [
    (get_dice,    0.5),       # 2*1/(2+1+1)
    (get_jaccard, 1/3),       # 1/(1+1+1)
    (get_tpr,     0.5),       # 1/(1+1)
    (get_vs,      1.0),       # 1 - |1-1|/4
])
def test_overlap_metrics(simple_masks, func, expected):
    A, B = simple_masks
    assert pytest.approx(expected, rel=1e-6) == func(A, B)

@pytest.mark.parametrize("func,expected", [
    (get_HD,  1.0),   # Hausdorff = 1 voxel
    (get_AVD, 0.5),   # avg surface dist = (1+0)/2
])
def test_distance_metrics(simple_masks, func, expected):
    A, B = simple_masks
    assert pytest.approx(expected, rel=1e-6) == func(A, B)

@pytest.mark.parametrize("func,expected", [
    # confusion on identical
    (get_truepos,  1),
    (get_falsepos, 0),
    (get_falseneg, 0),
    (get_trueneg,  3),

    # overlap on identical = perfect
    (get_dice,     1.0),
    (get_jaccard,  1.0),
    (get_tpr,      1.0),
    (get_vs,       1.0),

    # information & agreement on identical = perfect
    (get_MI,       1.0),
    (get_ARI,      1.0),
    (get_ICC,      1.0),
    (get_KAP,      1.0),

    # probabilistic distance = 0
    (get_PBD,      0.0),

    # distance on identical = 0
    (get_HD,       0.0),
    (get_AVD,      0.0),
])
def test_identical_masks_metrics(identical_masks, func, expected):
    A, B = identical_masks
    assert pytest.approx(expected, rel=1e-6) == func(A, B)
