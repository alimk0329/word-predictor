# Text classifier from scratch (NumPy)

[![CI](https://github.com/alimk0329/word-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/alimk0329/word-predictor/actions/workflows/ci.yml)

TF-IDF, multinomial Naive Bayes and softmax logistic regression written from scratch in NumPy and checked against
scikit-learn on 20 Newsgroups. Naive Bayes predictions are identical to scikit-learn's. Logistic regression reaches
67.6% test accuracy (scikit-learn on the same objective: 65.8%) and trains several times faster (timings vary by run). A live dashboard shows
calibrated topic probabilities and topic-aware word completions that update on every keystroke; Tab accepts the
top word.

![Dashboard: "the shuttle crew reached orbit around the moon and the gold" → golden 42%, goldin 32%; sci.space >99%](dashboard/screenshot.png)

## Run it

```bash
pip install -e ".[dev]"
pytest -q                       # 27 tests, ~2 s
python benchmarks/compare.py    # ~1 min: prints every table below, writes model.pkl, exits 1 if a check fails
python dashboard/server.py      # http://127.0.0.1:8000 (trains in seconds if model.pkl is missing)
```

Seeded and deterministic: two runs with different `PYTHONHASHSEED`s print identical numbers apart from timings, which vary from run to run.
Measured on Python 3.14, NumPy 2.3.5, a 12-core WSL2 machine.

## Results

20 Newsgroups with headers, footers and quotes removed (otherwise e-mail addresses and signatures leak the label).
Train 10,191 / validation 1,123 / test 7,532 posts; 30,000-term vocabulary.

| Model | Impl | Test accuracy | Macro-F1 | Train s |
|---|---|---|---|---|
| Multinomial NB (counts, α=1) | ours | 0.6529 | 0.6352 | 0.01 |
| Multinomial NB (counts, α=1) | sklearn | 0.6529 | 0.6352 | 0.02 |
| Softmax LogReg (TF-IDF, L2) | ours | 0.6757 | 0.6704 | 1.54 |
| Softmax LogReg (TF-IDF, L2) | sklearn lbfgs | 0.6581 | 0.6498 | 17.73 |

- Naive Bayes agrees with scikit-learn on 100% of test predictions. Both score 0.653. α=0.01 scores 2 points higher on validation.
- Both logistic regressions minimize the same objective (sklearn's `C` = 1/(λn)). With full-batch descent ours lands
  on sklearn's optimum, probabilities within 1e-5 (`tests/test_logreg.py`). In the table ours is ahead because early
  stopping on validation adds regularization.
- **Performance.** One epoch on 2,000 posts takes 178 ms with dense float64 X and 10 ms with sparse float32 (18x).
  Full training would take about 32 s dense (extrapolated) against 1.5 s measured. Dense X alone would be 2.4 GB.

## Does it generalize?

- **Validation looks like the future.** Validation is each newsgroup's latest 10% of training posts, by article
  number. The test set is later posts, so early stopping and calibration see the same drift a deployed model does.
  Hyperparameters were picked on validation only: λ=1e-6, lr=100 with 1/(1+0.1·epoch) decay, batch 256.
- **Regularized twice.** L2 and early stopping, which kept epoch 30 of 35. Accuracy is 0.969 on train, 0.723 on
  validation and 0.676 on test. A train gap like that is normal for a linear model with 30k sparse features; held-out
  and out-of-domain data are the real test.
- **The percentages mean what they say.** Expected calibration error (ECE) on test posts cut to their first k words,
  which is what the dashboard sees while you type:

  | Words typed | 3 | 5 | 10 | 25 | 50 | full post |
  |---|---|---|---|---|---|---|
  | Accuracy | 0.170 | 0.224 | 0.362 | 0.543 | 0.631 | 0.696 |
  | ECE, raw model | 0.171 | 0.116 | 0.033 | 0.088 | 0.102 | 0.078 |
  | ECE, calibrated | **0.027** | **0.022** | **0.028** | **0.050** | **0.058** | **0.060** |

  Full posts are slightly overconfident because of drift: a calibration fit on the test posts themselves would only
  reach 0.058.
- **Out of domain.** 60 sentences written for this benchmark (3 per class, present-day phrasing, none from the
  dataset): top-1 0.917, top-3 1.000, 6-way super-topic 0.967. Four of the five misses are neighbouring topics (the fifth sends
  "The second amendment protects the right to keep and bear arms." to sci.crypt), e.g.
  "Was the Book of Mormon really inspired by God?" → alt.atheism instead of talk.religion.misc.
- **Word completion.** Tested on 3,071 held-out words sampled from 400 test posts. "No context" weights the same word model by
  class frequency only.

  | Letters typed | Tab correct | In top 3 | No-context Tab correct |
  |---|---|---|---|
  | 1 | 0.178 | 0.277 | 0.173 |
  | 2 | 0.329 | 0.531 | 0.316 |
  | 3 | 0.469 | 0.721 | 0.450 |

`benchmarks/compare.py` gates all of this: NB identical to sklearn, LogReg within 2 points of sklearn and ≥ 0.65,
ECE ≤ 0.08 at every length and within 0.01 of the test-fit ceiling, out-of-domain top-1 ≥ 0.70 / top-3 ≥ 0.90 /
super-topic ≥ 0.85, topic context beating no-context completion, and p95 latency per keystroke < 10 ms (measured:
0.38 ms).

## The dashboard

- **Topics.** Logistic regression on TF-IDF, with logits divided by `t0·(1 + c/known_words)`. `t0` and `c` are fitted
  on validation posts cut to 3–50 words plus whole posts. A single temperature can't serve both a 3-word and a
  300-word text: it gave ECE 0.26 at 3 words. With no known words the odds are the training class frequencies.
- **Completion.** `P(w | prefix, text) ∝ Σ_c P(c | text before the word) · P(w | c)`. `P(w | c)` comes from a
  Naive Bayes word model (α=0.1) over 31,760 words that keeps function words. The vocabulary is sorted, so a prefix
  is one `bisect` slice. The half-typed word doesn't vote on its own completion.
- **Typing.** Every keystroke POSTs to `/predict`, and responses that arrive out of order are dropped. Tab inserts
  the rest of the grey word plus a space, but only when the suggestion answers the exact current text and the caret
  is at the end; otherwise Tab moves focus as usual. The server is stdlib `http.server` with keep-alive and Nagle
  disabled, which had been adding ~40 ms per keystroke; a round trip takes ~0.5 ms.

## Math

**TF-IDF.** tf is the raw count and idf(t) = ln((1+N)/(1+df(t))) + 1; each row is then L2-normalized. The +1s
inside the log act like one extra document containing every term, so no term divides by zero and rare terms don't
explode. The +1 outside keeps a term that appears in every document from being zeroed out entirely. Normalizing
stops long posts from dominating. The output matches sklearn's `TfidfVectorizer` to 1e-6 when both are given the same tokens.

**Naive Bayes.** P(c|d) ∝ P(c) Πw P(w|c)^count(w,d), so log P(c|d) = log P(c) + Σw count(w,d) log P(w|c) +
const. For all documents at once that is one matrix product, X · log P(w|c)ᵀ. With Laplace smoothing,
P(w|c) = (N_cw + α)/(N_c + αV); without it, one unseen word gives log 0 = −∞ and vetoes the class. A product of
thousands of probabilities underflows float64, while a sum of logs does not.

**Softmax regression.** P = softmax(XW + b) and L = −(1/n) Σi log P(i, yi) + (λ/2)‖W‖². For one sample,
∂Li/∂zk = pk − 1[k = yi], because the softmax Jacobian and the log's derivative cancel. Hence
∇W = Xᵀ(P − Y)/n + λW and ∇b = mean(P − Y). A finite-difference check agrees to a relative error below 1e-5.
Softmax is shift-invariant, so subtracting each row's max before `exp` changes nothing except preventing overflow.

**Why L2 on sparse text.** There are 600k weights for 10k documents. Many rare words occur in only one class, and
without a penalty their weights grow without bound to separate the training set perfectly. L2, a Gaussian prior on
the weights, keeps them finite and spreads weight across correlated words.

## Error analysis (LogReg, test)

Most confused pairs: talk.politics.guns ↔ talk.politics.misc (108), soc.religion.christian ↔ talk.religion.misc
(105), alt.atheism ↔ soc.religion.christian (77). These are neighbouring discussions with shared vocabulary.
talk.religion.misc is the weakest class (F1 0.32).

| Class | F1 | Highest-weight words |
|---|---|---|
| alt.atheism | 0.476 | atheism god bible religion atheists motto bobby islamic atheist islam |
| comp.graphics | 0.674 | graphics image images animation pov tiff cview polygon points computer |
| comp.os.ms-windows.misc | 0.621 | windows cica file win ini files risc memory driver ms |
| comp.sys.ibm.pc.hardware | 0.629 | dx scsi pc monitor vlb irq bios ide controller card |
| comp.sys.mac.hardware | 0.698 | mac apple lc se quadra centris powerbook duo simms nubus |
| comp.windows.x | 0.756 | motif server window widget xterm mit widgets hi sun windows |
| misc.forsale | 0.779 | sale offer sell shipping new dos interested condition includes call |
| rec.autos | 0.712 | car cars dealer toyota ford oil engine honda vw autos |
| rec.motorcycles | 0.757 | bike dod ride bikes motorcycle helmet riding bmw harley dog |
| rec.sport.baseball | 0.825 | baseball team players braves cubs stadium game runs year phillies |
| rec.sport.hockey | 0.875 | hockey team game nhl season play leafs playoff puck mask |
| sci.crypt | 0.755 | clipper encryption key nsa government keys privacy security crypto chip |
| sci.electronics | 0.459 | circuit electronics power chips hc ground radio dial cci mc |
| sci.med | 0.767 | msg doctor disease medical pain treatment health patients cancer effects |
| sci.space | 0.717 | space orbit launch nasa moon shuttle solar spacecraft lunar flight |
| soc.religion.christian | 0.675 | god church christians jesus christ christianity christian faith truth marriage |
| talk.politics.guns | 0.610 | gun guns weapons firearms weapon fire fbi law nra government |
| talk.politics.mideast | 0.780 | israel israeli arab jews armenian turkish armenians jewish turkey arabs |
| talk.politics.misc | 0.518 | tax government clinton president homosexuals state bush libertarians health drugs |
| talk.religion.misc | 0.322 | god christian jesus christians koresh kent objective morality fbi bible |

## Lessons from building it

- **Step size depends on feature scale.** Unit-norm sparse TF-IDF rows give a tiny curvature. lr ≤ 10 had not
  converged after 200 epochs; lr=100 reaches sklearn's validation loss in 2 s.
- **Accuracy and calibration are separate questions.** The raw model was underconfident on full posts (60% sure,
  76% right) and overconfident on 3-word snippets.
- **Determinism has to be designed in.** The vocabulary depended on `PYTHONHASHSEED`, because ties at the
  `max_features` cutoff followed set iteration order. Ties now break by term.

## Deliberate exceptions and known limits

- `scipy.sparse` is used for X only, and nowhere else. scikit-learn only downloads the data and serves as the
  reference.
- The topics are 1993 newsgroups. Modern words ("iPhone") are unknown and carry no evidence; the dashboard shows
  how many words it knows.
- Completion is a topic-weighted unigram model. It ignores the previous word (a bigram would turn "new yo" into
  "york"), and 4% of held-out words are outside its vocabulary.
- Ghost text needs the caret at the end of the text. The server is a local tool: localhost only, no auth.
