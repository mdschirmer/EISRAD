# test_metrics.py

import numpy as np
import pytest

from eisrad.metrics import (
    get_truepos, get_falsepos, get_falseneg, get_trueneg,
    get_dice, get_jaccard, get_tpr, get_vs,
    get_MI, get_ARI, get_ICC, get_KAP,
    get_HD, get_AVD, get_lesion_table
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

    # distance on identical = 0
    (get_HD,       0.0),
    (get_AVD,      0.0),
])
def test_identical_masks_metrics(identical_masks, func, expected):
    A, B = identical_masks
    assert pytest.approx(expected, rel=1e-6) == func(A, B)


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
    np.testing.assert_allclose(les['vol_auto_cc'] * 1000, [4, 4])
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
    np.testing.assert_allclose(iso['vol_auto_cc'] * 1000,   [2, 1])
    np.testing.assert_allclose(aniso['vol_auto_cc'] * 1000, [15, 0])  # 5 mm² voxels
    np.testing.assert_allclose(iso['Dice'],   [2/3, 0])
    np.testing.assert_allclose(aniso['Dice'], [0.5, 0])
    assert list(iso['detected']) == [True, False]


# ─── CLI: -b [T] ───

@pytest.mark.parametrize("flags,expected_cc", [
    ([],                0.002),  # no -b: non-binary mask binarized at > 0.5
    (['-b'],            0.004),  # -b alone: > 0
    (['-b', '0.2'],     0.003),  # -b T
    (['-b', '-v'],      0.004),  # -b followed by another flag: T = 0
])
def test_binarize_threshold(tmp_path, flags, expected_cc):
    """Auto mask holds 0.1, 0.3, 0.7, 0.9 (1 mm voxels): count above T."""
    import nibabel as nib
    import pandas as pd
    from eisrad.__main__ import main
    A = np.zeros((4, 4, 4)); A[0, 0, 0:4] = 1
    B = np.zeros((4, 4, 4)); B[0, 0, 0:4] = [0.1, 0.3, 0.7, 0.9]
    for name, M in (('m', A), ('a', B)):
        nib.save(nib.Nifti1Image(M.astype(np.float32), np.eye(4)),
                 str(tmp_path / f'{name}.nii.gz'))
    (tmp_path / 'pairs.csv').write_text(
        f"manual,auto\n{tmp_path/'m.nii.gz'},{tmp_path/'a.nii.gz'}\n")
    main(['eisrad', '-f', str(tmp_path / 'pairs.csv'),
          '-o', str(tmp_path / 'r.png'), '-r', str(tmp_path / 'r.csv')] + flags)
    vol = pd.read_csv(tmp_path / 'r.csv')['vol_auto_cc'].iloc[0]
    assert vol == pytest.approx(expected_cc)


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
    ms = ['Dice', 'Jaccard', 'TPR', 'VS', 'MI', 'ARI', 'ICC', 'KAP', '1-OER', '1-DER']
    ref = get_values(A.astype(float), B.astype(float), ms)
    for dt in (bool, np.uint8, int):
        vals = get_values(A.astype(dt), B.astype(dt), ms)
        for m in ms:
            assert vals[m] == pytest.approx(ref[m], abs=1e-12), (dt, m)
    assert get_ICC(A, B) == pytest.approx(get_ICC(A.astype(float), B.astype(float)))
