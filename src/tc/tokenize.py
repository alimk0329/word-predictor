import re
from collections import Counter

STOPWORDS = frozenset("""
a about above after again against all also am an and any are aren't as at be because been before being below
between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each few
for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him
himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself just let's me more most mustn't
my myself no nor not now of off on once only or other ought our ours ourselves out over own same shan't she
she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there
there's these they they'd they'll they're they've this those through to too under until up very was wasn't we
we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's
will with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

# Runs of letters (any script), keeping inner apostrophes: "don't", "o'clock". Digits and "_" split tokens.
WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")


def tokenize(text, stopwords=STOPWORDS):
    words = WORD.findall(text.lower().replace("\u2019", "'"))
    return [w for w in words if len(w) > 1 and w not in stopwords]


def build_vocab(token_lists, min_df=2, max_features=None):
    """term -> column, alphabetical like sklearn; terms in >= min_df docs, capped to the most frequent."""
    df, tf = Counter(), Counter()
    for toks in token_lists:
        tf.update(toks)
        df.update(set(toks))
    terms = [t for t, n in df.items() if n >= min_df]
    if max_features and len(terms) > max_features:
        terms = sorted(terms, key=lambda t: (-tf[t], t))[:max_features]  # ties by term: same vocab every run
    return {t: i for i, t in enumerate(sorted(terms))}
