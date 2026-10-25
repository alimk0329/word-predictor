from tc.tokenize import build_vocab, tokenize


def test_lowercases_splits_and_drops_stopwords_digits_short_tokens():
    assert tokenize("The GPU's 3D driver, x11 & a NASA-launch!") == ["gpu's", "driver", "nasa", "launch"]


def test_curly_apostrophe_and_unicode_letters():
    assert tokenize("Don’t café", stopwords=frozenset()) == ["don't", "café"]


def test_build_vocab_min_df_is_document_frequency():
    docs = [["b", "a", "a", "a"], ["a", "c"], ["b", "d"]]
    assert build_vocab(docs, min_df=2) == {"a": 0, "b": 1}


def test_build_vocab_max_features_keeps_most_frequent_sorted():
    docs = [["z", "z", "z"], ["y", "y"], ["x"]]
    assert build_vocab(docs, min_df=1, max_features=2) == {"y": 0, "z": 1}


def test_build_vocab_breaks_frequency_ties_alphabetically():
    for docs in ([["b"], ["a"], ["c", "c"]], [["a"], ["b"], ["c", "c"]]):
        assert build_vocab(docs, min_df=1, max_features=2) == {"a": 0, "c": 1}
