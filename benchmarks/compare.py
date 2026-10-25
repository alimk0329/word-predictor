"""Ours vs scikit-learn on 20 Newsgroups, plus the checks the dashboard model must pass.

    python benchmarks/compare.py    # prints markdown, writes model.pkl, exits 1 if any check fails
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB as SkNB

from tc.logreg import SoftmaxRegression, softmax
from tc.metrics import accuracy, confusion_matrix, expected_calibration_error, macro_f1, precision_recall_f1
from tc.naive_bayes import MultinomialNB
from tc.suggest import LOGREG, SNIPPET_WORDS, TFIDF, Suggester, fit_temperature, load_20ng, snippets, split
from tc.tokenize import WORD
from tc.vectorize import CountVectorizer, TfidfVectorizer

MODEL = Path(__file__).resolve().parents[1] / "model.pkl"

# Written for this benchmark before any model saw them: short, present-day, everyday phrasing.
OOD = [
    ("I don't believe in any god, and morality doesn't need religion.", "alt.atheism"),
    ("Atheists are often asked how they can have ethics without faith.", "alt.atheism"),
    ("There is no evidence that a deity exists, so why should I believe?", "alt.atheism"),
    ("What is the best algorithm for rendering 3D polygons with texture mapping?", "comp.graphics"),
    ("I need software to convert JPEG images to GIF without losing quality.", "comp.graphics"),
    ("How do I implement ray tracing and shading to make a realistic image?", "comp.graphics"),
    ("Windows keeps crashing after I installed the new driver. How do I fix it?", "comp.os.ms-windows.misc"),
    ("My Windows desktop icons disappeared after the update.", "comp.os.ms-windows.misc"),
    ("How do I change the default fonts in the Windows control panel?", "comp.os.ms-windows.misc"),
    ("My PC won't boot after I installed a second IDE hard drive on the motherboard.", "comp.sys.ibm.pc.hardware"),
    ("Should I buy more RAM or a faster CPU for my PC?", "comp.sys.ibm.pc.hardware"),
    ("The BIOS does not detect the SCSI controller card I just installed.", "comp.sys.ibm.pc.hardware"),
    ("My Apple Macintosh won't start after I upgraded the memory.", "comp.sys.mac.hardware"),
    ("Is this monitor compatible with my older Mac?", "comp.sys.mac.hardware"),
    ("The battery in my Apple laptop drains really fast. Is that normal?", "comp.sys.mac.hardware"),
    ("How do I configure the X server so remote clients can open windows on my display?", "comp.windows.x"),
    ("My Motif widget doesn't redraw when the window is resized.", "comp.windows.x"),
    ("I can't get xterm to use a different font under the X Window System.", "comp.windows.x"),
    ("For sale: barely used laptop, asking $300 or best offer, shipping included.", "misc.forsale"),
    ("Selling my old textbooks and a bike. Email me if interested.", "misc.forsale"),
    ("Brand new in the box, price is negotiable, will ship anywhere.", "misc.forsale"),
    ("My car's engine makes a strange noise when I accelerate on the highway.", "rec.autos"),
    ("Which sedan gets better gas mileage, the Toyota or the Honda?", "rec.autos"),
    ("I need new tires and brakes for my truck before winter.", "rec.autos"),
    ("I just bought my first motorcycle and need a good helmet.", "rec.motorcycles"),
    ("Riding in the rain is dangerous without proper gear for the bike.", "rec.motorcycles"),
    ("Which Harley or Kawasaki is best for a beginner rider?", "rec.motorcycles"),
    ("The pitcher threw a no-hitter and the team won in the ninth inning.", "rec.sport.baseball"),
    ("Who has the best batting average in the league this season?", "rec.sport.baseball"),
    ("The Yankees need better hitting if they want to make the playoffs.", "rec.sport.baseball"),
    ("The goalie made forty saves and the team won in overtime.", "rec.sport.hockey"),
    ("Who will win the Stanley Cup this year, the Penguins or the Rangers?", "rec.sport.hockey"),
    ("He scored a hat trick in the third period of the hockey game.", "rec.sport.hockey"),
    ("Is this encryption algorithm secure against brute force attacks on the key?", "sci.crypt"),
    ("The government wants a backdoor in every encryption chip.", "sci.crypt"),
    ("How does public key cryptography like RSA work?", "sci.crypt"),
    ("What resistor value do I need to power an LED from a 9 volt battery?", "sci.electronics"),
    ("I'm designing a circuit with an op amp and need help with the voltage gain.", "sci.electronics"),
    ("How do I solder a capacitor onto this circuit board?", "sci.electronics"),
    ("My doctor prescribed antibiotics for the infection but I still have symptoms.", "sci.med"),
    ("Is there any evidence that this diet helps with migraine headaches?", "sci.med"),
    ("What are the side effects of this medication for patients with diabetes?", "sci.med"),
    ("NASA plans to launch a new satellite into orbit next year.", "sci.space"),
    ("The shuttle astronauts will repair the telescope during the mission.", "sci.space"),
    ("How much fuel does a rocket need to reach the moon?", "sci.space"),
    ("Jesus taught that we should love our neighbors and forgive those who sin against us.", "soc.religion.christian"),
    ("Our church is studying the Bible and what salvation through Christ means.", "soc.religion.christian"),
    ("How should Christians understand prayer and the grace of God?", "soc.religion.christian"),
    ("Gun control laws will not stop criminals from getting firearms.", "talk.politics.guns"),
    ("The second amendment protects the right to keep and bear arms.", "talk.politics.guns"),
    ("Should assault weapons be banned to reduce violent crime?", "talk.politics.guns"),
    ("The conflict between Israel and the Palestinians has lasted for decades.", "talk.politics.mideast"),
    ("Armenians and Turks still disagree about the history of the genocide.", "talk.politics.mideast"),
    ("Peace talks in the Middle East broke down again over the occupied territories.", "talk.politics.mideast"),
    ("The president's new tax plan will raise taxes on the middle class.", "talk.politics.misc"),
    ("Congress is debating health care reform and the federal budget deficit.", "talk.politics.misc"),
    ("The government should not spend more money on welfare programs.", "talk.politics.misc"),
    ("The cult leader claimed to be a prophet and his followers believed him.", "talk.religion.misc"),
    ("Many religions disagree about morality, and the Bible is not the only source.", "talk.religion.misc"),
    ("Was the Book of Mormon really inspired by God?", "talk.religion.misc"),
]


def timed(fn):
    t = time.perf_counter()
    return fn(), time.perf_counter() - t


def table(header, rows):
    print("| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(map(str, row)) + " |")
    print()


def group(label):
    """The six usual 20 Newsgroups super-topics."""
    if "relig" in label or "atheism" in label:
        return "religion"
    return "politics" if label.startswith("talk.politics") else label.split(".")[0]


def main():
    texts, y, labels, article = load_20ng("train")
    test_texts, yt, _, _ = load_20ng("test")
    tr, va = split(y, article)
    tr_texts, va_texts, ytr, yva = [texts[i] for i in tr], [texts[i] for i in va], y[tr], y[va]
    checks = []

    # Naive Bayes on counts, Laplace alpha = 1; logistic regression on TF-IDF with the *same* objective
    # for both implementations (sklearn's C * sum(CE) + ||W||^2 / 2 is our mean(CE) + lam/2 ||W||^2, C = 1/(lam n)).
    cv = CountVectorizer(**TFIDF)
    Xc, Xct = cv.fit_transform(tr_texts), cv.transform(test_texts)
    nb, t_nb = timed(lambda: MultinomialNB(1.0).fit(Xc, ytr))
    sknb, t_sknb = timed(lambda: SkNB(alpha=1.0).fit(Xc, ytr))
    tfidf = TfidfVectorizer(**TFIDF)
    (X, Xv, Xt), t_vec = timed(lambda: (tfidf.fit_transform(tr_texts), tfidf.transform(va_texts), tfidf.transform(test_texts)))
    lr, t_lr = timed(lambda: SoftmaxRegression(**LOGREG).fit(X, ytr, Xv, yva))
    sklr, t_sklr = timed(lambda: LogisticRegression(C=1 / (LOGREG["lam"] * X.shape[0]), max_iter=3000).fit(X, ytr))
    preds = {name: m.predict(Xm) for name, m, Xm in [("nb", nb, Xct), ("sknb", sknb, Xct), ("lr", lr, Xt), ("sklr", sklr, Xt)]}
    acc = {name: accuracy(yt, p) for name, p in preds.items()}

    print(f"## Ours vs scikit-learn\n\n20 Newsgroups, headers/footers/quotes removed. Train {len(tr)}, "
          f"validation {len(va)} (each newsgroup's latest train posts), test {len(yt)}. Vocabulary {X.shape[1]} terms; "
          f"TF-IDF of all three sets took {t_vec:.1f} s.\n")
    table(["Model", "Impl", "Test accuracy", "Macro-F1", "Train s"], [
        [model, impl, f"{acc[k]:.4f}", f"{macro_f1(yt, preds[k]):.4f}", f"{t:.2f}"]
        for model, impl, k, t in [
            ("Multinomial NB (counts, α=1)", "ours", "nb", t_nb), ("Multinomial NB (counts, α=1)", "sklearn", "sknb", t_sknb),
            ("Softmax LogReg (TF-IDF, L2)", "ours", "lr", t_lr), ("Softmax LogReg (TF-IDF, L2)", "sklearn lbfgs", "sklr", t_sklr),
        ]
    ])
    agree = accuracy(preds["nb"], preds["sknb"])
    checks += [
        (f"NB predictions identical to sklearn ({agree:.2%} agree)", agree == 1.0),
        (f"LogReg within 2 points of sklearn ({acc['lr'] - acc['sklr']:+.4f})", abs(acc["lr"] - acc["sklr"]) <= 0.02),
        (f"LogReg test accuracy >= 0.65 ({acc['lr']:.4f})", acc["lr"] >= 0.65),
    ]

    model = Suggester(tfidf, lr, labels, tr_texts, ytr, va_texts, yva)
    model.save(MODEL)

    # --- generalization: held-out splits, every typed length, calibration ---
    print("## Generalization\n")
    print(f"Early stopping kept epoch {int(np.argmin(lr.val_curve_)) + 1} of {lr.n_epochs_}. Calibration "
          f"(fit on validation snippets): logits / ({model.t0:.3f} * (1 + {model.c:.2f} / known words)).\n")
    table(["Split", "Accuracy"], [
        ["train", f"{accuracy(ytr, lr.predict(X)):.4f}"],
        ["validation (early stopping + calibration)", f"{accuracy(yva, lr.predict(Xv)):.4f}"],
        ["test (held out, posted later)", f"{acc['lr']:.4f}"],
    ])
    rows, eces = [], {}
    for k in SNIPPET_WORDS:
        st, sy = snippets(test_texts, yt, (k,))
        P, n = model.topics(st)
        keep = n > 0
        raw = softmax(tfidf.transform(st)[np.flatnonzero(keep)] @ lr.W + lr.b)
        eces[k] = expected_calibration_error(P[keep], sy[keep])
        rows.append([k or "full post", int(keep.sum()), f"{accuracy(sy[keep], P[keep].argmax(1)):.4f}",
                     f"{P[keep].max(1).mean():.3f}", f"{expected_calibration_error(raw, sy[keep]):.3f}", f"{eces[k]:.3f}"])
    print("Test posts cut to their first k words (what the dashboard sees while you type):\n")
    table(["Words typed", "Texts", "Accuracy", "Mean confidence", "ECE raw", "ECE calibrated"], rows)
    # Ceiling for this calibration family: fit it on the test posts themselves (diagnostic only, never used).
    n = np.diff(Xt.indptr)
    Z, n, ys = (Xt @ lr.W + lr.b)[n > 0], n[n > 0], yt[n > 0]
    t0, c = fit_temperature(Z, n, ys)
    oracle = expected_calibration_error(softmax(Z / (t0 * (1 + c / n))[:, None]), ys)
    print(f"Test posts were written after every training post, so some overconfidence is drift. Even a calibration "
          f"fit on the test posts themselves only reaches ECE {oracle:.3f} on full posts.\n")
    checks += [
        (f"Calibrated test ECE <= 0.08 at every length, full posts included ({max(eces.values()):.3f} worst)",
         max(eces.values()) <= 0.08),
        (f"Full-post ECE within 0.01 of a calibration fit on test itself ({eces[None]:.3f} vs {oracle:.3f})",
         eces[None] - oracle <= 0.01),
    ]

    # --- error analysis ---
    cm = confusion_matrix(yt, preds["lr"], len(labels))
    pair = np.triu(cm + cm.T, 1)
    print("## Error analysis\n\nMost confused class pairs (LogReg, test):\n")
    table(["Class A", "Class B", "Confusions"],
          [[labels[i], labels[j], pair[i, j]] for i, j in zip(*np.unravel_index(np.argsort(-pair, axis=None)[:3], pair.shape))])
    vocab = np.array(list(tfidf.vocab_))
    f1 = precision_recall_f1(yt, preds["lr"], len(labels))[2]
    print("Per-class F1 and highest-weight words (LogReg):\n")
    table(["Class", "F1", "Top-10 words"],
          [[labels[c], f"{f1[c]:.3f}", " ".join(vocab[np.argsort(-lr.W[:, c])[:10]])] for c in range(len(labels))])

    # --- out-of-domain sentences ---
    P, _ = model.topics([s for s, _ in OOD])
    truth = np.array([labels.index(label) for _, label in OOD])
    top3 = np.argsort(-P, axis=1)[:, :3]
    o1, o3 = accuracy(truth, top3[:, 0]), float(np.mean([t in row for t, row in zip(truth, top3)]))
    og = float(np.mean([group(labels[p]) == group(labels[t]) for p, t in zip(top3[:, 0], truth)]))
    print(f"## Out-of-domain sentences\n\n{len(OOD)} hand-written sentences (3 per class), not from the dataset.\n")
    table(["Top-1", "Top-3", "Super-topic (6-way)", "Mean confidence"], [[f"{o1:.3f}", f"{o3:.3f}", f"{og:.3f}", f"{P.max(1).mean():.3f}"]])
    print("Misses:\n")
    table(["Sentence", "Truth", "Predicted (p)"],
          [[s, label, f"{labels[top3[i, 0]]} ({P[i, top3[i, 0]]:.2f})"] for i, (s, label) in enumerate(OOD) if top3[i, 0] != truth[i]])
    checks += [
        (f"Out-of-domain top-1 >= 0.70 ({o1:.3f})", o1 >= 0.70),
        (f"Out-of-domain top-3 >= 0.90 ({o3:.3f})", o3 >= 0.90),
        (f"Out-of-domain super-topic >= 0.85 ({og:.3f})", og >= 0.85),
    ]

    # --- word completion: type the first L letters of held-out words, does Tab give the right word? ---
    rng = np.random.default_rng(0)
    hits = np.zeros((3, 4))  # per L: ours top-1, ours top-3, no-context top-1, no-context top-3
    counts, oov, known = np.zeros(3), 0, set(model.words)
    for d in rng.choice(len(test_texts), 400, replace=False):
        doc = test_texts[d].replace("\u2019", "'")
        spans = [m for m in WORD.finditer(doc) if len(m.group()) >= 3]
        for m in [spans[i] for i in rng.choice(len(spans), min(8, len(spans)), replace=False)]:
            word, ctx = m.group().lower(), model.topics([doc[: m.start()]])[0][0]
            oov += word not in known
            for L in (1, 2, 3):
                if L >= len(word):
                    continue
                ours, base = (model.complete(word[:L], w) for w in (ctx, model.prior))
                top1 = [r[0][0] if r else None for r in (ours, base)]  # [] when no known word has the prefix
                hits[L - 1] += [top1[0] == word, word in dict(ours), top1[1] == word, word in dict(base)]
                counts[L - 1] += 1
    rate = hits / counts[:, None]
    print(f"## Word completion\n\n{int(counts[0])} held-out words (>= 3 letters) from 400 test posts; "
          f"{oov / counts[0]:.1%} are outside the {len(model.words)}-word completion vocabulary and count as misses. "
          "No-context = same word model weighted by class frequencies only.\n")
    table(["Letters typed", "Tab correct (top-1)", "In top 3", "No-context top-1", "No-context top-3"],
          [[L + 1, *(f"{v:.3f}" for v in rate[L])] for L in range(3)])
    checks.append((f"Topic context beats no-context completion at 1-3 letters (top-1 +{(rate[:, 0] - rate[:, 2]).min():.3f} worst)",
                   bool(np.all(rate[:, 0] > rate[:, 2]))))

    # --- per-keystroke latency of the exact call the dashboard server makes ---
    lat = []
    for d in rng.choice(len(test_texts), 1000):
        text = test_texts[d][: rng.integers(1, 2000)]
        lat.append(timed(lambda: model.suggest(text))[1] * 1000)
    p50, p95, worst_ms = np.percentile(lat, 50), np.percentile(lat, 95), max(lat)
    print(f"## Latency\n\n`Suggester.suggest()` on 1000 random test-post prefixes up to 2000 characters: "
          f"p50 {p50:.2f} ms, p95 {p95:.2f} ms, max {worst_ms:.2f} ms.\n")
    checks.append((f"Per-keystroke p95 latency < 10 ms ({p95:.2f} ms)", p95 < 10))

    # --- performance: what sparse float32 bought over dense float64 (same code, same data) ---
    n = 2000
    one_epoch = {**LOGREG, "max_epochs": 1}
    _, t_sparse = timed(lambda: SoftmaxRegression(**one_epoch).fit(X[:n], ytr[:n]))
    dense = X[:n].toarray().astype(np.float64)
    _, t_dense = timed(lambda: SoftmaxRegression(**one_epoch).fit(dense, ytr[:n]))
    scale = X.shape[0] / n * lr.n_epochs_
    print(f"## Performance\n\nOne epoch on {n} posts: dense float64 {t_dense * 1000:.0f} ms vs sparse float32 "
          f"{t_sparse * 1000:.0f} ms ({t_dense / t_sparse:.0f}x). Full training ({lr.n_epochs_} epochs) extrapolates to "
          f"{t_dense * scale:.0f} s dense vs {t_lr:.1f} s measured sparse; dense X alone would be "
          f"{X.shape[0] * X.shape[1] * 8 / 1e9:.1f} GB.\n")

    print("## Checks\n")
    for name, ok in checks:
        print(f"- [{'x' if ok else ' '}] {name}")
    print(f"\nModel saved to {MODEL.name}.")
    return all(ok for _, ok in checks)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
