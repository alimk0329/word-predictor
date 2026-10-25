import numpy as np


def accuracy(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def confusion_matrix(y, p, k=None):
    """cm[i, j] = samples of true class i predicted as j; labels are ints 0..k-1."""
    y, p = np.asarray(y), np.asarray(p)
    k = k or int(max(y.max(), p.max())) + 1
    return np.bincount(y * k + p, minlength=k * k).reshape(k, k)


def precision_recall_f1(y, p, k=None):
    """Per-class arrays; 0 where undefined (sklearn's zero_division=0)."""
    cm = confusion_matrix(y, p, k)
    tp, predicted, actual = np.diag(cm), cm.sum(axis=0), cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return tuple(np.nan_to_num(a / b) for a, b in [(tp, predicted), (tp, actual), (2 * tp, predicted + actual)])


def macro_f1(y, p):
    """Mean F1 over labels present in y or p (sklearn's default label set)."""
    return float(precision_recall_f1(y, p)[2][np.union1d(y, p)].mean())


def expected_calibration_error(P, y, bins=15):
    """Size-weighted mean |confidence - accuracy| over equal-width bins of the top probability."""
    conf, hit = P.max(axis=1), P.argmax(axis=1) == y
    b = np.minimum((conf * bins).astype(int), bins - 1)
    return float(np.abs(np.bincount(b, conf, bins) - np.bincount(b, hit, bins)).sum() / len(y))
