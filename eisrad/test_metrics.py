# test_metrics.py

import numpy as np
import pytest

from eisrad.metrics import (
    get_truepos, get_falsepos, get_falseneg, get_trueneg,
    get_dice, get_jaccard, get_tpr, get_vs,
    get_MI, get_ARI, get_ICC, get_KAP,
    get_surface_metrics, get_lesion_table, lesion_surface_summary
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

def test_hd95_ignores_single_outlier():
    """
    One stray voxel far away sets the maximum distance (plain Hausdorff) but
    is below the 95th percentile, so HD95 stays 0 for otherwise equal masks.
    """
    A = np.zeros((40, 40), dtype=int); A[5:25, 5:25] = 1
    B = A.copy(); B[38, 38] = 1
    assert get_surface_metrics(A, B)[0] == pytest.approx(0.0)

def test_hd95_mm():
    """A 5x5 square shifted by 2 columns, with 1 x 3 mm voxels (6 mm shift)."""
    A = np.zeros((9, 14), dtype=int); A[2:7, 2:7] = 1
    B = np.zeros_like(A);             B[2:7, 4:9] = 1
    assert get_surface_metrics(A, B)[0] == pytest.approx(2.0)
    assert get_surface_metrics(A, B, zooms=(1.0, 3.0))[0] == pytest.approx(6.0)

def test_surface_dice_tolerance():
    """
    Default tolerance = one voxel in every direction: a 1-column shift is
    fully within tolerance, a 3-column shift is not. With 1 x 3 mm voxels a
    1-column shift is 3 mm: outside a 2 mm tolerance, inside a 3 mm one.
    """
    A = np.zeros((9, 16), dtype=int); A[2:7, 2:7] = 1
    B1 = np.zeros_like(A); B1[2:7, 3:8] = 1
    B3 = np.zeros_like(A); B3[2:7, 5:10] = 1
    assert get_surface_metrics(A, B1)[1] == pytest.approx(1.0)
    assert get_surface_metrics(A, B3)[1] < 1.0
    z = (1.0, 3.0)
    assert get_surface_metrics(A, B1, z, tolerance_mm=2.0)[1] < 1.0
    assert get_surface_metrics(A, B1, z, tolerance_mm=3.0)[1] == pytest.approx(1.0)
    # resolution-based default ignores the voxel size
    assert get_surface_metrics(A, B1, z)[1] == pytest.approx(1.0)

def test_surface_metrics_empty():
    A = np.zeros((5, 5), dtype=int); A[2, 2] = 1
    Z = np.zeros_like(A)
    assert get_surface_metrics(Z, Z) == (0.0, 1.0)
    assert get_surface_metrics(A, Z) == (np.inf, 0.0)
    assert get_surface_metrics(Z, A) == (np.inf, 0.0)

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

    # boundary measures on identical: HD95 0 mm, surface Dice 1
    (lambda A, B: get_surface_metrics(A, B)[0], 0.0),
    (lambda A, B: get_surface_metrics(A, B)[1], 1.0),
])
def test_identical_masks_metrics(identical_masks, func, expected):
    A, B = identical_masks
    assert pytest.approx(expected, rel=1e-6) == func(A, B)


def test_empty_mask_conventions():
    """Both masks empty is a perfect result for TPR / 1-DER/2 / 1-OER/2."""
    from eisrad.metrics import get_values
    Z = np.zeros((3, 3))
    both_empty = get_values(Z, Z, ['TPR', '1-DER/2', '1-OER/2'])
    assert both_empty == {'TPR': 1.0, '1-DER/2': 1.0, '1-OER/2': 1.0}

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

@pytest.mark.parametrize("seed", range(5))
def test_oder_halves_sum_to_dice_error(seed):
    """OER/2 + DER/2 = 1 - Dice, so 1-OER/2 and 1-DER/2 lie in [0, 1]."""
    from eisrad.metrics import get_ODER, get_values
    rng = np.random.default_rng(seed)
    A = (rng.random((15, 15, 6)) < 0.08).astype(int)
    B = (rng.random((15, 15, 6)) < 0.12).astype(int)
    OER, DER = get_ODER(A, B)
    assert OER / 2 + DER / 2 == pytest.approx(1 - get_dice(A, B))
    v = get_values(A, B, ['1-OER/2', '1-DER/2'])
    assert 0 <= v['1-OER/2'] <= 1 and 0 <= v['1-DER/2'] <= 1

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
    cov = np.sort(get_lesion_table(A, B)[0]['TPR'])
    np.testing.assert_allclose(cov, [0.5, 0.5])

def test_lesion_coverage_empty():
    A = np.zeros((3, 3, 3))
    les, fp = get_lesion_table(A, A)
    assert les['TPR'].size == 0 and fp.size == 0


# ─── Lesion table (manual mask as reference) ───

def test_lesion_table_merge():
    """
    One automated blob spans two manual lesions: its voxels are shared out
    to the nearest lesion, so nothing is counted twice.
    A: lesions at cols 0-1 and 6-7; B: one bar over cols 0-7.
    Cols 0-3 go to lesion 1, cols 4-7 to lesion 2 -> each: |A|=2, |B|=4, TP=2.
    """
    from eisrad.metrics import get_lesion_table
    A = np.zeros((3, 10), dtype=int)
    B = np.zeros_like(A)
    A[0, 0:2] = 1
    A[0, 6:8] = 1
    B[0, 0:8] = 1
    les, fp = get_lesion_table(A, B)
    np.testing.assert_allclose(les['vol_auto_ml'] * 1000, [4, 4])
    np.testing.assert_allclose(les['Dice'], [2/3, 2/3])
    np.testing.assert_allclose(les['TPR'], [1, 1])
    assert fp.size == 0

def test_lesion_table_split_and_fp():
    """
    One manual lesion (cols 0-5) covered by two automated blobs (cols 0-1
    and 4-5) -> pooled: |A|=6, |B|=4, TP=4. A separate blob far away is a
    false-positive lesion of 3 voxels.
    """
    from eisrad.metrics import get_lesion_table
    A = np.zeros((5, 12), dtype=int)
    B = np.zeros_like(A)
    A[0, 0:6] = 1
    B[0, 0:2] = 1
    B[0, 4:6] = 1
    B[4, 9:12] = 1
    les, fp = get_lesion_table(A, B)
    np.testing.assert_allclose(les['Dice'], [0.8])
    np.testing.assert_allclose(les['TPR'], [4/6])
    np.testing.assert_allclose(fp * 1000, [3])

def test_lesion_table_anisotropic():
    """
    Nearest-lesion assignment uses mm. Lesion 1 at (0,0), lesion 2 at (3,1);
    an automated blob covers (0,0), (1,0), (2,0) and only overlaps lesion 1.
    Voxel (2,0) is 2 mm from lesion 1 either way, but lesion 2 is 1.41 mm
    away with 1 mm voxels and 5.1 mm away with 1 x 5 mm voxels.
    Lesion 2 is never detected (no overlap), even when voxels are assigned.
    """
    from eisrad.metrics import get_lesion_table
    A = np.zeros((4, 3), dtype=int)
    B = np.zeros_like(A)
    A[0, 0] = 1
    A[3, 1] = 1
    B[0:3, 0] = 1
    iso, _   = get_lesion_table(A, B, zooms=(1.0, 1.0))
    aniso, _ = get_lesion_table(A, B, zooms=(1.0, 5.0))
    np.testing.assert_allclose(iso['vol_auto_ml'] * 1000,   [2, 1])
    np.testing.assert_allclose(aniso['vol_auto_ml'] * 1000, [15, 0])  # 5 mm² voxels
    np.testing.assert_allclose(iso['Dice'],   [2/3, 0])
    np.testing.assert_allclose(aniso['Dice'], [0.5, 0])
    assert list(iso['detected']) == [True, False]


# ─── Closed-form MI / ARI / KAP agree with scikit-learn ───

@pytest.mark.parametrize("seed", range(6))
def test_closed_form_matches_sklearn(seed):
    """Random binary masks, incl. sparse, empty and full-overlap cases."""
    sk = pytest.importorskip("sklearn.metrics")
    from eisrad.metrics import get_MI, get_ARI, get_KAP
    rng = np.random.default_rng(seed)
    shape = (20, 20, 10)
    p = [0.3, 0.02, 0.001, 0.5, 0.05, 0.0][seed]
    A = (rng.random(shape) < p).astype(int)
    B = A.copy() if seed == 3 else (rng.random(shape) < p * 1.5).astype(int)
    a, b = A.ravel(), B.ravel()
    assert get_MI(A, B)  == pytest.approx(sk.normalized_mutual_info_score(a, b), abs=1e-12)
    assert get_ARI(A, B) == pytest.approx(sk.adjusted_rand_score(a, b), abs=1e-12)
    if a.any() or b.any():
        assert get_KAP(A, B) == pytest.approx(sk.cohen_kappa_score(a, b), abs=1e-12)


def test_values_independent_of_input_dtype():
    """Bool, int and float masks give identical metrics (ICC once broke on bools)."""
    from eisrad.metrics import get_values
    rng = np.random.default_rng(0)
    A = rng.random((12, 12, 6)) < 0.2
    B = rng.random((12, 12, 6)) < 0.25
    ms = ['Dice', 'Jaccard', 'TPR', 'VS', 'MI', 'ARI', 'ICC', 'KAP', '1-OER/2', '1-DER/2',
          '1/(1+HD95/r) (L)', 'SDice (L)', 'HD95_mm (L)']
    ref = get_values(A.astype(float), B.astype(float), ms)
    for dt in (bool, np.uint8, int):
        vals = get_values(A.astype(dt), B.astype(dt), ms)
        for m in ms:
            assert vals[m] == pytest.approx(ref[m], abs=1e-12), (dt, m)
    assert get_ICC(A, B) == pytest.approx(get_ICC(A.astype(float), B.astype(float)))


# ─── Lesion-wise boundary measures and their patient summary ───

def test_lesion_table_surface_measures():
    """
    Lesion 1 segmented exactly (HD95 0, SDice 1), lesion 2 missed (NaN):
    boundary measures are only defined for detected lesions.
    """
    A = np.zeros((20, 20), dtype=int); A[2:6, 2:6] = 1; A[14:17, 14:17] = 1
    B = np.zeros_like(A);             B[2:6, 2:6] = 1
    les, _ = get_lesion_table(A, B)
    assert les['HD95_mm'][0] == pytest.approx(0.0)
    assert les['SDice'][0] == pytest.approx(1.0)
    assert np.isnan(les['HD95_mm'][1]) and np.isnan(les['SDice'][1])

def test_lesion_surface_summary():
    """
    Volume-weighted mean over detected lesions; missed lesions are left out;
    0 if nothing is detected; 1 if neither mask has a lesion.
    Score per lesion: 1 / (1 + HD95 / r), r = radius of a sphere with the
    lesion volume (1 mL -> r = 6.2035 mm).
    """
    r1 = (3 * 1000 / (4 * np.pi)) ** (1 / 3)
    les = {'detected': np.array([True, True, False]),
           'vol_manual_ml': np.array([1.0, 3.0, 5.0]),
           'HD95_mm': np.array([0.0, r1, np.nan]),
           'SDice': np.array([1.0, 0.6, np.nan])}
    r2 = (3 * 3000 / (4 * np.pi)) ** (1 / 3)
    out = lesion_surface_summary(les, auto_empty=False)
    assert out['SDice (L)'] == pytest.approx((1 * 1.0 + 3 * 0.6) / 4)
    assert out['1/(1+HD95/r) (L)'] == pytest.approx((1 * 1.0 + 3 / (1 + r1 / r2)) / 4)
    assert out['HD95_mm (L)'] == pytest.approx(3 * r1 / 4)

    missed = dict(les, detected=np.array([False, False, False]))
    assert lesion_surface_summary(missed, False)['SDice (L)'] == 0.0
    none = {k: np.zeros(0) for k in les}
    assert lesion_surface_summary(none, auto_empty=True)['SDice (L)'] == 1.0
    assert lesion_surface_summary(none, auto_empty=False)['SDice (L)'] == 0.0
