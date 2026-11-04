# test_metrics.py

import numpy as np
import pytest

from eisrad.metrics import (
    get_truepos, get_falsepos, get_falseneg, get_trueneg,
    get_dice, get_jaccard, get_tpr, get_vs,
    get_MI, get_ARI, get_ICC, get_PBD, get_KAP,
    get_HD, get_AVD, get_lesion_coverage
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


def test_pbd_no_overlap():
    """Non-empty masks without overlap are maximally distant, not perfect."""
    A = np.array([[1, 0], [0, 0]])
    B = np.array([[0, 0], [0, 1]])
    assert get_PBD(A, B) == np.inf
    assert get_PBD(np.zeros((2, 2)), np.zeros((2, 2))) == 0.0

def test_empty_mask_conventions():
    """Both masks empty is a perfect result for TPR / 1-DER / 1-OER."""
    from eisrad.metrics import get_values
    Z = np.zeros((3, 3))
    both_empty = get_values(Z, Z, ['TPR', '1-DER', '1-OER'])
    assert both_empty == {'TPR': 1.0, '1-DER': 1.0, '1-OER': 1.0}

def test_oder_lesionwise():
    """
    A: a 4-voxel lesion (half overlapped by B) and a missed 1-voxel lesion.
    B: overlaps the first lesion with 2 voxels + adds 2 extra contiguous
    voxels, and has a separate 1-voxel false positive.
    |A| = 5, |B| = 5, MTA = 5, |A∩B| = 2.
    OER = (4 + 4 - 2*2) / 5 = 0.8 ; DER = (1 + 1) / 5 = 0.4
    """
    from eisrad.metrics import get_ODER
    A = np.zeros((8, 8), dtype=int)
    B = np.zeros_like(A)
    A[0:2, 0:2] = 1          # lesion 1
    A[6, 6]     = 1          # missed lesion
    B[0:2, 1:3] = 1          # overlaps lesion 1 (2 voxels) + 2 extra
    B[6, 0]     = 1          # false positive
    OER, DER = get_ODER(A, B)
    assert OER == pytest.approx(0.8)
    assert DER == pytest.approx(0.4)

def test_oder_complete_miss():
    """Nothing detected: no outline error, all detection error."""
    from eisrad.metrics import get_ODER
    A = np.zeros((4, 4)); A[0, 0] = 1
    OER, DER = get_ODER(A, np.zeros_like(A))
    assert (OER, DER) == (0.0, 2.0)

def test_kap_both_empty():
    Z = np.zeros((3, 3))
    assert get_KAP(Z, Z) == 1.0

def test_lesion_coverage():
    """
    Two lesions in A: a 2x2 block (half covered by B) and a diagonal voxel
    pair (one voxel covered). The diagonal pair counts as one lesion under
    26-connectivity.
    """
    A = np.zeros((6, 6, 1), dtype=int)
    B = np.zeros_like(A)
    A[0:2, 0:2, 0] = 1
    B[0:2, 0, 0]   = 1
    A[4, 4, 0] = 1
    A[5, 5, 0] = 1
    B[4, 4, 0] = 1
    cov = np.sort(get_lesion_coverage(A, B))
    np.testing.assert_allclose(cov, [0.5, 0.5])

def test_lesion_coverage_empty():
    A = np.zeros((3, 3, 3))
    assert get_lesion_coverage(A, A).size == 0
