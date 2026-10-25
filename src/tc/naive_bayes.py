import numpy as np

from .logreg import softmax


class MultinomialNB:
    """log P(c|d) = log P(c) + sum_w count(w, d) log P(w|c) + const, with Laplace/Lidstone smoothing alpha.
    Everything stays in log space: multiplying raw probabilities underflows on long documents."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        Y = np.eye(len(self.classes_))[y]
        counts = (X.T @ Y).T + self.alpha  # k x V smoothed word counts per class
        self.feature_log_prob_ = np.log(counts) - np.log(counts.sum(axis=1, keepdims=True))
        self.class_log_prior_ = np.log(Y.sum(axis=0) / len(y))
        return self

    def joint_log_likelihood(self, X):
        return X @ self.feature_log_prob_.T + self.class_log_prior_  # one matrix multiply

    def predict_proba(self, X):
        return softmax(self.joint_log_likelihood(X))

    def predict(self, X):
        return self.classes_[self.joint_log_likelihood(X).argmax(axis=1)]
