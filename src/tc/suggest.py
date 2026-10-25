"""The dashboard's model: calibrated topic probabilities plus topic-aware word completion."""
import bisect
import pickle
import re
from itertools import product
from pathlib import Path

import numpy as np

from .logreg import SoftmaxRegression, log_softmax, softmax
from .naive_bayes import MultinomialNB
from .vectorize import CountVectorizer, TfidfVectorizer

TFIDF = dict(min_df=2, max_features=30000)
LOGREG = dict(lam=1e-6, lr=100, decay=0.1, batch_size=256, max_epochs=300, patience=5)  # picked on validation
COMPLETION_ALPHA = 0.1
SNIPPET_WORDS = (3, 5, 10, 25, 50, None)  # calibrate on what people actually type: short texts too
PARTIAL = re.compile(r"[^\W\d_]+(?:'[^\W\d_]*)*\Z")  # the word being typed at the very end


def load_20ng(subset):
    from sklearn.datasets import fetch_20newsgroups  # sklearn only downloads the data

    d = fetch_20newsgroups(subset=subset, remove=("headers", "footers", "quotes"))
    return d.data, d.target, d.target_names, np.array([int(f.rsplit("/", 1)[1]) for f in d.filenames])


def split(y, article, val_frac=0.1):
    """(train, val) indices; val = each newsgroup's latest posts (highest article numbers). The test set is
    later posts too, so early stopping and calibration see the same drift a deployed model does."""
    val = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        val.extend(idx[np.argsort(article[idx])][len(idx) - int(len(idx) * val_frac) :])
    return np.setdiff1d(np.arange(len(y)), val), np.sort(val)


def snippets(texts, y, lengths=SNIPPET_WORDS):
    """Every text cut to its first k words, for each k (None = whole text)."""
    cut = [" ".join(t.split()[:k]) if k else t for k in lengths for t in texts]
    return cut, np.tile(y, len(lengths))


def fit_temperature(Z, n, y):
    """(t0, c) minimizing the NLL of softmax(Z / (t0 * (1 + c / n))), n = known words per row (> 0)."""
    rows = np.arange(len(y))
    return min(
        product(np.geomspace(0.1, 3, 50), np.r_[0, np.geomspace(0.25, 32, 15)]),
        key=lambda p: -log_softmax(Z / (p[0] * (1 + p[1] / n))[:, None])[rows, y].mean(),
    )


def train(texts, y, labels, val_texts, val_y):
    tfidf = TfidfVectorizer(**TFIDF)
    clf = SoftmaxRegression(**LOGREG).fit(tfidf.fit_transform(texts), y, tfidf.transform(val_texts), val_y)
    return Suggester(tfidf, clf, labels, texts, y, val_texts, val_y)


class Suggester:
    def __init__(self, tfidf, clf, labels, texts, y, val_texts, val_y):
        """tfidf and clf come fitted; texts/y fit the per-topic word distributions, val_* the calibration."""
        self.tfidf, self.clf, self.labels = tfidf, clf, list(labels)
        self.prior = np.bincount(np.searchsorted(clf.classes_, y), minlength=len(clf.classes_)) / len(y)
        # Length-aware temperature: logits / (t0 * (1 + c / known_words)). One temperature can't fit both a
        # 3-word and a 300-word text, and the dashboard shows probabilities from the first keystroke.
        cut, cut_y = snippets(val_texts, val_y)
        X = tfidf.transform(cut)
        n, ys = np.diff(X.indptr), np.searchsorted(clf.classes_, cut_y)
        keep = n > 0  # no known words -> prior; nothing to calibrate
        self.t0, self.c = fit_temperature((X @ clf.W + clf.b)[keep], n[keep], ys[keep])
        cv = CountVectorizer(min_df=2, stopwords=frozenset())  # function words are what people type most
        nb = MultinomialNB(COMPLETION_ALPHA).fit(cv.fit_transform(texts), y)
        self.words = list(cv.vocab_)  # alphabetical, so all words with a prefix form one slice
        self.word_probs = np.exp(nb.feature_log_prob_.T).astype(np.float32)  # V x k: P(word | topic)

    def topics(self, texts):
        """Calibrated P(topic | text), one row per text (columns follow clf.classes_), and known-word counts."""
        X = self.tfidf.transform(texts)
        n = np.diff(X.indptr)
        P = softmax((X @ self.clf.W + self.clf.b) / (self.t0 * (1 + self.c / np.maximum(n, 1)))[:, None])
        P[n == 0] = self.prior  # no evidence: the training class frequencies, not whatever the bias says
        return P, n

    def complete(self, prefix, topic_probs, k=3):
        """Top-k known words starting with prefix: P(w | prefix, text) ∝ sum_c P(c | text) P(w | c)."""
        lo = bisect.bisect_left(self.words, prefix)
        hi = bisect.bisect_left(self.words, prefix + "\U0010ffff", lo)
        p = self.word_probs[lo:hi] @ topic_probs.astype(np.float32)
        top = np.argsort(-p, kind="stable")[:k]
        return [(self.words[lo + i], float(p[i] / p.sum())) for i in top]

    def suggest(self, text, k_topics=5, k_words=3):
        text = text.replace("\u2019", "'")
        m = PARTIAL.search(text)
        # the half-typed word must not vote on its own completion, so completion uses the text before it
        (probs, context), known = self.topics([text, text[: m.start()] if m else ""])
        top = np.argsort(-probs, kind="stable")[:k_topics]
        out = {"topics": [(self.labels[self.clf.classes_[i]], float(probs[i])) for i in top], "known": int(known[0])}
        out["prefix"] = prefix = m.group().lower() if m else ""
        out["words"] = self.complete(prefix, context, k_words) if m else []
        return out

    def save(self, path):
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path):
        return pickle.loads(Path(path).read_bytes())  # only ever our own file, written by save()
