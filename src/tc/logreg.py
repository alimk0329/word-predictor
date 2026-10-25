import numpy as np


def log_softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)  # stable: exp() never sees a positive number
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))


def softmax(Z):
    return np.exp(log_softmax(Z))


def loss_and_grad(W, b, X, y, lam):
    """Mean cross-entropy + lam/2 ||W||^2 for integer labels y, and its gradient:
    dL/dW = X.T @ (P - Y) / n + lam * W,  dL/db = mean(P - Y)."""
    n = len(y)
    logP = log_softmax(X @ W + b)
    loss = -logP[np.arange(n), y].mean() + lam / 2 * np.vdot(W, W)
    dZ = np.exp(logP)
    dZ[np.arange(n), y] -= 1
    dZ /= n
    return loss, X.T @ dZ + lam * W, dZ.sum(axis=0)


class SoftmaxRegression:
    """Multiclass logistic regression trained by mini-batch gradient descent with lr / (1 + decay * epoch),
    early-stopped on validation loss (best weights are kept)."""

    def __init__(self, lam=1e-4, lr=1.0, decay=0.0, batch_size=256, max_epochs=100, patience=5, seed=0):
        self.lam, self.lr, self.decay, self.batch_size = lam, lr, decay, batch_size
        self.max_epochs, self.patience, self.seed = max_epochs, patience, seed

    def fit(self, X, y, X_val=None, y_val=None):
        self.classes_, y = np.unique(y, return_inverse=True)
        n, dtype = X.shape[0], np.result_type(X.dtype, np.float32)  # int counts -> float weights
        self.W = W = np.zeros((X.shape[1], len(self.classes_)), dtype)
        self.b = b = np.zeros(len(self.classes_), dtype)
        rng = np.random.default_rng(self.seed)
        self.loss_curve_, self.val_curve_ = [], []
        best, wait = np.inf, 0
        for epoch in range(self.max_epochs):
            lr, total, order = self.lr / (1 + self.decay * epoch), 0.0, rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = order[s : s + self.batch_size]
                loss, dW, db = loss_and_grad(W, b, X[idx], y[idx], self.lam)
                W -= lr * dW
                b -= lr * db
                total += loss * len(idx)
            self.loss_curve_.append(total / n)
            self.n_epochs_ = epoch + 1
            if X_val is None:
                continue
            loss = self.log_loss(X_val, y_val, W, b)
            self.val_curve_.append(loss)
            wait = 0 if loss < best - 1e-4 else wait + 1  # patience needs a real gain...
            if loss < best:  # ...but any gain updates the kept weights
                best, self.W, self.b = loss, W.copy(), b.copy()
            if wait >= self.patience:
                break
        return self

    def log_loss(self, X, y, W=None, b=None):
        W, b = (self.W, self.b) if W is None else (W, b)
        logP = log_softmax(X @ W + b)
        return -logP[np.arange(len(y)), np.searchsorted(self.classes_, y)].mean()

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.classes_[(X @ self.W + self.b).argmax(axis=1)]
