from itertools import chain

import numpy as np
import scipy.sparse as sp

from .tokenize import STOPWORDS, build_vocab, tokenize


class CountVectorizer:
    """Bag of words: docs -> sparse float32 term counts (scipy.sparse for X only, see README)."""

    def __init__(self, min_df=2, max_features=None, stopwords=STOPWORDS):
        self.min_df, self.max_features, self.stopwords = min_df, max_features, stopwords

    def fit(self, docs):
        self.fit_transform(docs)
        return self

    def fit_transform(self, docs):
        toks = [tokenize(d, self.stopwords) for d in docs]
        self.vocab_ = build_vocab(toks, self.min_df, self.max_features)
        return self._count(toks)

    def transform(self, docs):
        return self._count([tokenize(d, self.stopwords) for d in docs])

    def _count(self, toks):
        get = self.vocab_.get
        ids = [[i for i in map(get, doc) if i is not None] for doc in toks]
        lens = [len(i) for i in ids]
        rows = np.repeat(np.arange(len(ids)), lens)
        cols = np.fromiter(chain.from_iterable(ids), np.int32, sum(lens))
        # duplicate (row, col) pairs are summed into counts when COO converts to CSR
        return sp.csr_array((np.ones(len(cols), np.float32), (rows, cols)), shape=(len(ids), len(self.vocab_)))


class TfidfVectorizer(CountVectorizer):
    """sklearn defaults: raw counts * smooth idf, idf = ln((1 + n) / (1 + df)) + 1, rows L2-normalized."""

    def fit_transform(self, docs):
        X = super().fit_transform(docs)
        df = np.bincount(X.indices, minlength=X.shape[1])
        self.idf_ = np.log((1 + X.shape[0]) / (1 + df)) + 1
        return self._tfidf(X)

    def transform(self, docs):
        return self._tfidf(super().transform(docs))

    def _tfidf(self, X):
        X.data *= self.idf_[X.indices]
        norms = np.sqrt(X.multiply(X).sum(axis=1))
        X.data /= np.repeat(norms, np.diff(X.indptr))  # empty rows repeat 0 times: no 0/0
        return X
