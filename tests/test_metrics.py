import numpy as np
from sklearn import metrics as skm

from tc.metrics import accuracy, confusion_matrix, expected_calibration_error, macro_f1, precision_recall_f1


def test_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 300)
    p = np.where(rng.random(300) < 0.6, y, rng.integers(0, 5, 300))
    assert accuracy(y, p) == skm.accuracy_score(y, p)
    assert np.array_equal(confusion_matrix(y, p), skm.confusion_matrix(y, p))
    for ours, sk in zip(precision_recall_f1(y, p), skm.precision_recall_fscore_support(y, p, zero_division=0)):
        assert np.allclose(ours, sk)
    assert np.isclose(macro_f1(y, p), skm.f1_score(y, p, average="macro"))


def test_never_predicted_and_absent_classes():
    y, p = np.array([0, 0, 3, 3]), np.array([0, 3, 3, 3])
    prec, rec, f1 = precision_recall_f1(np.array([0, 1]), np.array([0, 0]))
    assert prec[1] == 0 and rec[1] == 0 and f1[1] == 0
    assert np.isclose(macro_f1(y, p), skm.f1_score(y, p, average="macro"))  # classes 1, 2 absent: excluded


def test_expected_calibration_error():
    P = np.tile([0.8, 0.2], (10, 1))
    assert np.isclose(expected_calibration_error(P, np.array([0] * 8 + [1] * 2)), 0)  # 80% sure, 80% right
    assert np.isclose(expected_calibration_error(P, np.zeros(10, int)), 0.2)  # 80% sure, always right
