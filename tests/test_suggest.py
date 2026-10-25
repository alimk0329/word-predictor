import numpy as np

from tc.logreg import SoftmaxRegression
from tc.suggest import Suggester
from tc.vectorize import TfidfVectorizer

SPACE = [
    "nasa launched the shuttle into orbit",
    "the shuttle crew reached orbit around the moon",
    "rocket engines and the shuttle launch schedule",
    "satellite orbit decayed and the shuttle returned",
]
HOCKEY = [
    "the goalie stopped the puck and the team won",
    "hockey fans saw a late score in overtime",
    "the team needed one more score to win the game",
    "the goalie gave up a score in the third period",
]
TEXTS, Y = SPACE + HOCKEY, np.array([0] * 4 + [1] * 4)


def _model():
    tfidf = TfidfVectorizer(min_df=1)
    clf = SoftmaxRegression(lam=1e-3, lr=5, batch_size=8, max_epochs=300).fit(tfidf.fit_transform(TEXTS), Y)
    return Suggester(tfidf, clf, ["sci.space", "rec.sport.hockey"], TEXTS, Y, TEXTS, Y)


def test_topics_and_topic_aware_completion():
    m = _model()
    space, hockey = m.suggest("rocket launch and s"), m.suggest("goalie puck and s")
    assert space["topics"][0][0] == "sci.space" and hockey["topics"][0][0] == "rec.sport.hockey"
    assert space["prefix"] == "s" and space["words"][0][0] == "shuttle"
    assert hockey["words"][0][0] == "score"
    probs = [p for _, p in space["words"]]
    assert probs == sorted(probs, reverse=True) and 0 < sum(probs) <= 1 + 1e-6
    assert abs(sum(p for _, p in space["topics"]) - 1) < 1e-6


def test_prefix_extraction():
    m = _model()
    assert m.suggest("The ORB")["prefix"] == "orb" and m.suggest("The ORB")["words"][0][0] == "orbit"
    assert m.suggest("Don’")["prefix"] == "don'"
    for done in ("the orbit ", "the orbit.", "the orbit\n", "", "x2"):
        assert m.suggest(done)["words"] == [], done
    assert m.suggest("zzzz")["words"] == []


def test_no_known_words_gives_class_prior(tmp_path):
    m = _model()
    s = m.suggest("qwerty")
    assert s["known"] == 0 and [p for _, p in s["topics"]] == [0.5, 0.5]
    m.save(tmp_path / "m.pkl")
    assert Suggester.load(tmp_path / "m.pkl").suggest("rocket s") == m.suggest("rocket s")
