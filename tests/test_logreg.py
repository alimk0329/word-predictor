import numpy as np
import scipy.sparse as sp
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from tc.logreg import SoftmaxRegression, loss_and_grad, softmax


def _data(n=600, seed=0):
    X, y = make_classification(n, 20, n_informative=10, n_classes=4, random_state=seed)
    return (X - X.mean(0)) / X.std(0), y


def test_softmax_is_stable_for_huge_logits():
    p = softmax(np.array([[1000.0, 1000.0, -1000.0]]))
    assert np.allclose(p, [[0.5, 0.5, 0.0]])


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(20, 5)), rng.integers(0, 3, 20)
    W, b, lam, eps = rng.normal(size=(5, 3)), rng.normal(size=3), 0.1, 1e-6
    _, dW, db = loss_and_grad(W, b, X, y, lam)

    def numeric(P, f):
        g = np.zeros_like(P)
        for i in np.ndindex(P.shape):
            hi, lo = P.copy(), P.copy()
            hi[i] += eps
            lo[i] -= eps
            g[i] = (f(hi) - f(lo)) / (2 * eps)
        return g

    for analytic, num in [
        (dW, numeric(W, lambda w: loss_and_grad(w, b, X, y, lam)[0])),
        (db, numeric(b, lambda c: loss_and_grad(W, c, X, y, lam)[0])),
    ]:
        assert np.linalg.norm(num - analytic) / (np.linalg.norm(num) + np.linalg.norm(analytic)) < 1e-5


def test_full_batch_loss_decreases_monotonically():
    X, y = _data(200)
    m = SoftmaxRegression(lam=1e-3, lr=0.2, batch_size=len(y), max_epochs=60).fit(X, y)
    assert np.all(np.diff(m.loss_curve_) < 0)


def test_same_objective_as_sklearn():
    X, y = _data()
    Xtr, ytr, Xte, yte = X[:450], y[:450], X[450:], y[450:]
    lam = 1e-3  # sklearn minimizes C * sum(CE) + ||W||^2 / 2, i.e. lam = 1 / (C * n)
    sk = LogisticRegression(C=1 / (lam * len(ytr)), max_iter=5000, tol=1e-10).fit(Xtr, ytr)
    full = SoftmaxRegression(lam=lam, lr=0.5, batch_size=len(ytr), max_epochs=2000).fit(Xtr, ytr)
    assert np.allclose(full.predict_proba(Xte), sk.predict_proba(Xte), atol=1e-5)  # same optimum
    mini = SoftmaxRegression(lam=lam, lr=0.2, decay=0.1, batch_size=32, max_epochs=150).fit(Xtr, ytr)
    assert abs(np.mean(mini.predict(Xte) == yte) - sk.score(Xte, yte)) <= 0.02


def test_early_stopping_keeps_best_validation_weights():
    X, y = _data()
    m = SoftmaxRegression(lam=0, lr=0.5, max_epochs=500, patience=3).fit(sp.csr_array(X[:450]), y[:450], X[450:], y[450:])
    assert m.n_epochs_ < 500
    logp = np.log(m.predict_proba(X[450:]))
    assert np.isclose(-logp[np.arange(150), y[450:]].mean(), min(m.val_curve_), atol=1e-6)


def test_kept_weights_match_val_curve_minimum_for_sub_tolerance_gains():
    X, y = _data()  # lr this small improves val loss by ~5e-5 per epoch, under the 1e-4 patience tolerance
    m = SoftmaxRegression(lam=0, lr=1e-4, max_epochs=50, patience=3).fit(X[:450], y[:450], X[450:], y[450:])
    assert np.isclose(m.log_loss(X[450:], y[450:]), min(m.val_curve_), rtol=0, atol=1e-9)


def test_integer_features_train_float_weights():
    X, y = np.random.default_rng(0).poisson(1.0, (60, 5)), np.arange(60) % 3
    assert SoftmaxRegression(max_epochs=3).fit(X, y).W.dtype == np.float64
