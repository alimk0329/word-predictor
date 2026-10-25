import numpy as np
import scipy.sparse as sp
from sklearn.naive_bayes import MultinomialNB as SkNB

from tc.naive_bayes import MultinomialNB


def test_toy_known_answer():
    X = np.array([[3, 0], [2, 1], [0, 3], [1, 2]])
    nb = MultinomialNB(alpha=1).fit(X, np.array([0, 0, 1, 1]))
    # P(w|c) = (count + 1) / (total + V): class 0 -> (5+1)/8, (1+1)/8
    assert np.allclose(np.exp(nb.feature_log_prob_), [[0.75, 0.25], [0.25, 0.75]])
    assert np.allclose(np.exp(nb.class_log_prior_), [0.5, 0.5])
    assert list(nb.predict(np.array([[4, 0], [0, 4], [2, 1]]))) == [0, 1, 0]


def test_matches_sklearn_dense_and_sparse():
    rng = np.random.default_rng(0)
    X, Xt = rng.poisson(0.5, (300, 60)), rng.poisson(0.5, (100, 60))
    y = rng.choice(["atheism", "hockey", "space", "crypt"], 300)
    for alpha in (1.0, 0.1):
        ours, sk = MultinomialNB(alpha).fit(sp.csr_array(X), y), SkNB(alpha=alpha).fit(X, y)
        assert np.allclose(ours.feature_log_prob_, sk.feature_log_prob_)
        assert np.allclose(ours.class_log_prior_, sk.class_log_prior_)
        assert np.array_equal(ours.predict(Xt), sk.predict(Xt))
        assert np.array_equal(ours.predict(sp.csr_array(Xt)), sk.predict(Xt))
        assert np.allclose(ours.predict_proba(Xt), sk.predict_proba(Xt))


def test_long_documents_do_not_underflow():
    nb = MultinomialNB().fit(np.array([[5, 1], [1, 5]]), np.array([0, 1]))
    p = nb.predict_proba(np.array([[5000, 4000]]))
    assert np.all(np.isfinite(p)) and np.isclose(p.sum(), 1) and p[0, 0] > 0.99
