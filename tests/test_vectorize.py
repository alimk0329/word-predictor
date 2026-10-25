import numpy as np
from sklearn.feature_extraction.text import CountVectorizer as SkCount
from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidf

from tc.tokenize import tokenize
from tc.vectorize import CountVectorizer, TfidfVectorizer

NO_STOP = frozenset()
CORPUS = [
    "The rocket reached orbit and the crew cheered.",
    "Hockey fans cheered when the team scored in overtime.",
    "The crew repaired the rocket engine before launch.",
    "Overtime goal! The team won the hockey game.",
    "Launch delayed: engine trouble on the rocket.",
    "",
]


def test_tfidf_hand_computed_three_docs():
    X = TfidfVectorizer(min_df=1, stopwords=NO_STOP).fit_transform(["the cat sat", "the cat cat ran", "dog ran"])
    # vocab: cat dog ran sat the; df = 2 1 2 1 2; idf = ln((1+n)/(1+df)) + 1
    counts = np.array([[1, 0, 0, 1, 1], [2, 0, 1, 0, 1], [0, 1, 1, 0, 0]], float)
    expected = counts * (np.log(4 / (1 + np.array([2, 1, 2, 1, 2]))) + 1)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(X.toarray(), expected, atol=1e-6)
    assert np.allclose(X[[2]].toarray(), [[0, 0.795960, 0.605349, 0, 0]], atol=1e-6)


def test_counts_match_sklearn():
    ours = CountVectorizer(min_df=1).fit_transform(CORPUS).toarray()
    sk = SkCount(tokenizer=tokenize, lowercase=False, token_pattern=None).fit_transform(CORPUS).toarray()
    assert np.array_equal(ours, sk)


def test_tfidf_matches_sklearn_on_same_tokens():
    unseen = ["rocket rocket hockey", "nothing known here", "the crew won"]
    for min_df in (1, 2):
        ours = TfidfVectorizer(min_df=min_df)
        sk = SkTfidf(tokenizer=tokenize, lowercase=False, token_pattern=None, min_df=min_df)
        X, Xs = ours.fit_transform(CORPUS), sk.fit_transform(CORPUS)
        assert ours.vocab_ == sk.vocabulary_
        assert np.allclose(ours.idf_, sk.idf_, atol=1e-6)
        assert np.allclose(X.toarray(), Xs.toarray(), atol=1e-6)
        assert np.allclose(ours.transform(unseen).toarray(), sk.transform(unseen).toarray(), atol=1e-6)


def test_document_without_known_words_is_a_zero_row():
    X = TfidfVectorizer(min_df=1).fit(CORPUS).transform(["zzz qqq", ""])
    assert X.shape[0] == 2 and X.nnz == 0
