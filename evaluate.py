"""
evaluate.py — Evaluation against SciFact gold labels
=====================================================
1. Retrieval:  dev claims as queries -> P@1, P@5, Recall@5, Recall@10, MRR@10 of the gold
               evidence abstract. BM25 vs tf-idf cosine baseline.
2. Verifier:   does the system call a claim SUPPORTED exactly when SciFact's experts do?
               Thresholds are tuned on claims_train and reported on claims_dev.
               Baseline = raw BM25 score > threshold (the naive "high score means supported").
3. Live LLM answers (optional, --live): verdict shares over demo_questions.json answers.

Run:  python evaluate.py [--live]
Out:  outputs/*.csv, outputs/*.png
"""

import argparse
import json
import os
import random
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk import sent_tokenize

from data_loader import RAW_DIR, read_jsonl
from ir_engine import (AdvancedIREngine, HALLUCINATION_COVERAGE, MIN_NORM_BM25, NUMBER_GATE_COVERAGE,
                       extract_numbers)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")


def gold_label(claim):
    labels = {e["label"] for evs in claim["evidence"].values() for e in evs}
    if "SUPPORT" in labels:
        return "SUPPORT"
    if "CONTRADICT" in labels:
        return "CONTRADICT"
    return "NEI"


def doc_ranking(hits, chunks, depth=10):
    """Collapse a chunk ranking into a ranking of distinct abstracts."""
    seen = []
    for idx, _ in hits:
        d = str(chunks[idx]["doc_id"])
        if d not in seen:
            seen.append(d)
        if len(seen) == depth:
            break
    return seen


# --------------------------------------------------------------------------- retrieval
def evaluate_retrieval(engine, claims):
    rows = []
    with_evidence = [c for c in claims if c["evidence"]]
    for name, fn in [("tf-idf cosine (baseline)", engine.search_tfidf), ("BM25", engine.search)]:
        p1 = p5 = r5 = r10 = mrr = 0.0
        for c in with_evidence:
            relevant = set(c["evidence"].keys())
            ranked = doc_ranking(fn(c["claim"], k=60), engine.chunks)
            # IR Concept: Precision@k = relevant in top k / k; Recall@k = relevant in top k / all relevant.
            p1 += sum(d in relevant for d in ranked[:1]) / 1
            p5 += sum(d in relevant for d in ranked[:5]) / 5
            r5 += sum(d in relevant for d in ranked[:5]) / len(relevant)
            r10 += sum(d in relevant for d in ranked[:10]) / len(relevant)
            # IR Concept: Mean Reciprocal Rank — 1/rank of the first relevant abstract.
            mrr += next((1 / (i + 1) for i, d in enumerate(ranked) if d in relevant), 0.0)
        n = len(with_evidence)
        rows.append({"retriever": name, "P@1": p1 / n, "P@5": p5 / n, "Recall@5": r5 / n,
                     "Recall@10": r10 / n, "MRR@10": mrr / n, "queries": n})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- verifier
def claim_features(engine, claims):
    """Run verify_claim once per claim; thresholds are applied afterwards."""
    feats = []
    for c in claims:
        r = engine.verify_claim(c["claim"])
        feats.append({"id": c["id"], "claim": c["claim"], "gold": gold_label(c),
                      "coverage": r["coverage"], "best_bm25": r["score"], "top_score": r["top_score"],
                      "norm_top": r["norm_top"],
                      "number_mismatch": r["number_mismatch_raw"], "doc_id": r["doc_id"],
                      "gold_docs": list(c["evidence"].keys())})
    return pd.DataFrame(feats)


def prf(pred, gold):
    """Precision / recall / F1 for the positive class (SUPPORTED)."""
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def ours_pred(df, support_cov, min_bm25, gate=NUMBER_GATE_COVERAGE):
    """Vectorised copy of ir_engine.decide_verdict (incl. the number gate) over a feature table."""
    nm = df.number_mismatch.to_numpy() & (df.coverage.to_numpy() >= gate)
    sup = (df.coverage.to_numpy() >= support_cov) & (df.best_bm25.to_numpy() >= min_bm25) & ~nm
    hal = nm | ((df.norm_top.to_numpy() >= MIN_NORM_BM25) & (df.coverage.to_numpy() < HALLUCINATION_COVERAGE))
    return pd.Series(np.where(sup, "SUPPORTED", np.where(hal, "HALLUCINATED", "UNVERIFIED")), index=df.index)


def tune(train):
    gold = train.gold == "SUPPORT"
    best_base = max(((t, prf(train.top_score > t, gold)[2]) for t in range(0, 61)), key=lambda x: x[1])
    best_ours = (None, None, -1.0)
    for cov in [x / 100 for x in range(30, 96, 5)]:
        for mb in range(0, 31, 2):
            f = prf(ours_pred(train, cov, mb) == "SUPPORTED", gold)[2]
            if f > best_ours[2]:
                best_ours = (cov, mb, f)
    return best_base[0], best_ours[0], best_ours[1]


def evaluate_verifier(train, dev):
    base_t, cov, mb = tune(train)
    print(f"Tuned on train: baseline BM25 threshold={base_t} | ours SUPPORT_COVERAGE={cov} MIN_BM25={mb}")
    gold = dev.gold == "SUPPORT"
    dev = dev.copy()
    dev["verdict"] = ours_pred(dev, cov, mb)
    rows = []
    for name, pred in [("Baseline: raw BM25 > threshold", dev.top_score > base_t),
                       ("Ours: coverage + number check", dev.verdict == "SUPPORTED")]:
        p, r, f = prf(pred, gold)
        rows.append({"method": name, "precision": p, "recall": r, "F1": f})
    # How often each gold class gets flagged green (lower is better for CONTRADICT / NEI).
    false_green = (dev[dev.gold != "SUPPORT"].groupby("gold").apply(
        lambda g: pd.Series({"baseline_green": (g.top_score > base_t).mean(),
                             "ours_green": (g.verdict == "SUPPORTED").mean(),
                             "n": len(g)}), include_groups=False))
    confusion = pd.crosstab(dev.gold, dev.verdict)
    params = {"THRESHOLD": base_t, "SUPPORT_COVERAGE": cov, "MIN_BM25": mb}
    return pd.DataFrame(rows), false_green, confusion, dev, params


# --------------------------------------------------------------------------- live answers
def evaluate_live(engine):
    """Verify the cached Groq answers clause by clause (long sentences are split first)."""
    from llm_agent import ask
    with open(os.path.join(HERE, "demo_questions.json"), encoding="utf-8") as f:
        questions = json.load(f)
    rows = []
    for q in questions:
        answer, _ = ask(q)
        for si, sentence in enumerate(sent_tokenize(answer), start=1):
            for r in engine.check_sentence(sentence):
                rows.append({"question": q, "sentence_no": si, "clause": r["sentence"],
                             "verdict": r["verdict"], "bm25": r["score"], "coverage": r["coverage"],
                             "number_mismatch": r["number_mismatch"], "doc_id": r["doc_id"],
                             "title": r["title"], "unsupported_terms": " ".join(r["unsupported_terms"][:5])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- targeted tests
def evaluate_compound(engine, dev, seed=0):
    """Does clause splitting rescue true facts hidden in a long sentence?

    Build 124 synthetic long sentences "<SUPPORT claim> <connector> <NEI claim>". The first
    part is true (gold SUPPORT), the second is not backed by the corpus (gold NEI). We compare
    checking the whole sentence at once with splitting it first.
    """
    rng = random.Random(seed)
    sup = [c for c in dev if gold_label(c) == "SUPPORT"]
    nei = [c for c in dev if gold_label(c) == "NEI"]
    connectors = ["; ", ", while ", ", but "]
    n = split_ok = 0
    whole_green = a_green = a_alone_green = b_green = b_alone_green = 0
    for i, a in enumerate(sup):
        b = rng.choice(nei)
        text = a["claim"].rstrip(".") + connectors[i % 3] + b["claim"]
        n += 1
        whole_green += engine.verify_claim(text)["verdict"] == "SUPPORTED"
        a_alone_green += engine.verify_claim(a["claim"])["verdict"] == "SUPPORTED"
        b_alone_green += engine.verify_claim(b["claim"])["verdict"] == "SUPPORTED"
        parts = engine.check_sentence(text)
        if len(parts) == 2:
            split_ok += 1
            a_green += parts[0]["verdict"] == "SUPPORTED"
            b_green += parts[1]["verdict"] == "SUPPORTED"
    return pd.DataFrame([{
        "compound_sentences": n,
        "split_into_2_clauses": split_ok / n,
        "TRUE part shown green: whole sentence": whole_green / n,
        "TRUE part shown green: after splitting": a_green / max(split_ok, 1),
        "TRUE part shown green: atomic (upper bound)": a_alone_green / n,
        "NEI part falsely green: after splitting": b_green / max(split_ok, 1),
        "NEI part falsely green: atomic": b_alone_green / n}]).T.rename(columns={0: "value"})


def evaluate_number_check(engine, dev):
    """Does the number check catch a changed figure?

    Take SUPPORT claims that state a quantity (number + unit) and are correctly SUPPORTED, change one
    number to a wrong value, and see whether the verdict leaves SUPPORTED (and goes red).
    """
    # Quantities only: a number followed by a unit ("12%", "2.5 mg", "52 weeks", "3-fold") or
    # preceded by "$". Digits inside names ("ALDH1", "Th17", "BRCA 1") are left alone, because
    # changing them alters the entity, not a statistic.
    num_re = re.compile(
        r"(?:(?<=\$)|(?<![A-Za-z0-9\-./]))\d+(?:\.\d+)?"
        r"(?=\s?(?:%|percent|-?fold|mg|g\b|kg|ml|mmhg|iu|units?\b|years?|months?|weeks?|days?|hours?|h\b"
        r"|times|patients|participants|subjects|women|men|people|children|million|billion))", re.I)
    rows = []
    for c in dev:
        if gold_label(c) != "SUPPORT" or not num_re.search(c["claim"]):
            continue
        orig = engine.verify_claim(c["claim"])
        if orig["verdict"] != "SUPPORTED":
            continue
        m = num_re.search(c["claim"])
        wrong = f"{float(m.group()) * 3 + 7:g}"
        bad = c["claim"][:m.start()] + wrong + c["claim"][m.end():]
        with_check = engine.verify_claim(bad)
        no_check = engine.verify_claim(bad, number_gate=9.0)     # gate > 1: number check never fires
        rows.append({"claim": c["claim"], "altered": bad,
                     "still_green_without_check": no_check["verdict"] == "SUPPORTED",
                     "still_green_with_check": with_check["verdict"] == "SUPPORTED",
                     "red_with_check": with_check["verdict"] == "HALLUCINATED"})
    df = pd.DataFrame(rows)
    return df


# --------------------------------------------------------------------------- charts
def plot_bars(df, index_col, cols, title, path):
    ax = df.set_index(index_col)[cols].T.plot.bar(figsize=(8, 4.5), rot=0)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def red_zone_sweep(train, dev, min_bm25=14, support_cov=0.55):
    """How the length-normalised 'on topic' threshold trades red count against red precision."""
    rows = []
    for t in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        row = {"MIN_NORM_BM25": t}
        for name, df in (("train", train), ("dev", dev)):
            nm = df.number_mismatch.to_numpy() & (df.coverage.to_numpy() >= NUMBER_GATE_COVERAGE)
            sup = (df.coverage.to_numpy() >= support_cov) & (df.best_bm25.to_numpy() >= min_bm25) & ~nm
            red = (nm | ((df.norm_top.to_numpy() >= t) & (df.coverage.to_numpy() < HALLUCINATION_COVERAGE))) & ~sup
            r = df[red]
            row[f"{name}_red"] = len(r)
            row[f"{name}_red_truly_unsupported"] = (r.gold != "SUPPORT").mean() if len(r) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_sweep(dev, min_bm25):
    """Precision/recall of SUPPORTED on dev as the coverage threshold moves (operating points)."""
    gold = dev.gold == "SUPPORT"
    rows = []
    for cov in [x / 100 for x in range(30, 96, 5)]:
        p, r, f = prf(ours_pred(dev, cov, min_bm25) == "SUPPORTED", gold)
        rows.append({"SUPPORT_COVERAGE": cov, "precision": p, "recall": r, "F1": f})
    return pd.DataFrame(rows)


def plot_sweep(sweep, chosen, path):
    plt.figure(figsize=(8, 4.5))
    for col in ["precision", "recall", "F1"]:
        plt.plot(sweep.SUPPORT_COVERAGE, sweep[col], marker="o", label=col)
    plt.axvline(chosen, color="grey", ls="--", label=f"tuned on train ({chosen})")
    plt.xlabel("SUPPORT_COVERAGE threshold")
    plt.ylim(0, 1)
    plt.title("Precision / recall trade-off of the SUPPORTED verdict (SciFact dev)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_coverage(dev, path):
    plt.figure(figsize=(8, 4.5))
    for label in ["SUPPORT", "CONTRADICT", "NEI"]:
        plt.hist(dev[dev.gold == label].coverage, bins=20, range=(0, 1), alpha=0.5, label=label)
    plt.xlabel("idf-weighted term coverage of best evidence chunk")
    plt.ylabel("claims")
    plt.title("Coverage by SciFact gold label (dev)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="also verify Groq answers to demo_questions.json")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.set_option("display.width", 140)

    engine = AdvancedIREngine()
    train = read_jsonl(os.path.join(RAW_DIR, "claims_train.jsonl"))
    dev = read_jsonl(os.path.join(RAW_DIR, "claims_dev.jsonl"))

    print("\n=== 1. Retrieval (SciFact dev, claims with gold evidence) ===")
    retrieval = evaluate_retrieval(engine, dev)
    print(retrieval.round(3).to_string(index=False))
    retrieval.to_csv(os.path.join(OUT_DIR, "retrieval_results.csv"), index=False)
    plot_bars(retrieval, "retriever", ["P@1", "P@5", "Recall@5", "Recall@10", "MRR@10"],
              "Retrieval of gold evidence abstracts (SciFact dev)", os.path.join(OUT_DIR, "retrieval.png"))

    print("\n=== 2. Verifier (tuned on train, reported on dev) ===")
    verifier, false_green, confusion, dev_df, params = evaluate_verifier(
        claim_features(engine, train), claim_features(engine, dev))
    print(verifier.round(3).to_string(index=False))
    print("\nShare of non-supported claims wrongly shown GREEN:")
    print(false_green.round(3).to_string())
    print("\nConfusion (gold rows x our verdict columns):")
    print(confusion.to_string())
    verifier.to_csv(os.path.join(OUT_DIR, "verifier_results.csv"), index=False)
    false_green.to_csv(os.path.join(OUT_DIR, "verifier_false_green.csv"))
    confusion.to_csv(os.path.join(OUT_DIR, "verifier_confusion.csv"))
    dev_df.to_csv(os.path.join(OUT_DIR, "verifier_dev_predictions.csv"), index=False)
    with open(os.path.join(OUT_DIR, "tuned_thresholds.json"), "w") as f:
        json.dump(params, f, indent=2)
    plot_bars(verifier, "method", ["precision", "recall", "F1"],
              "Is the claim supported? (SciFact dev)", os.path.join(OUT_DIR, "verifier.png"))
    plot_coverage(dev_df, os.path.join(OUT_DIR, "coverage_by_label.png"))
    train_feats = claim_features(engine, train)
    red_sweep = red_zone_sweep(train_feats, dev_df)
    print("\nRed-zone ('on topic' = length-normalised BM25) sweep; chosen MIN_NORM_BM25 =", MIN_NORM_BM25)
    print(red_sweep.round(2).to_string(index=False))
    red_sweep.to_csv(os.path.join(OUT_DIR, "red_zone_sweep.csv"), index=False)
    sweep = coverage_sweep(dev_df, params["MIN_BM25"])
    print("\nCoverage-threshold sweep on dev:")
    print(sweep.round(3).to_string(index=False))
    sweep.to_csv(os.path.join(OUT_DIR, "verifier_sweep.csv"), index=False)
    plot_sweep(sweep, params["SUPPORT_COVERAGE"], os.path.join(OUT_DIR, "verifier_sweep.png"))

    print("\n=== 2b. Targeted test: long compound sentences (clause splitting) ===")
    compound = evaluate_compound(engine, dev)
    print(compound.round(3).to_string())
    compound.to_csv(os.path.join(OUT_DIR, "compound_test.csv"))

    print("\n=== 2c. Targeted test: one quantity changed in a correct claim (number check, train+dev) ===")
    perturb = evaluate_number_check(engine, train + dev)   # quantity claims are rare: pool both splits
    summary = pd.DataFrame([{
        "claims tested": len(perturb),
        "still green, NO number check": perturb.still_green_without_check.mean(),
        "still green, WITH number check": perturb.still_green_with_check.mean(),
        "flagged red, WITH number check": perturb.red_with_check.mean()}]).T.rename(columns={0: "value"})
    print(summary.round(3).to_string())
    perturb.to_csv(os.path.join(OUT_DIR, "number_check_test.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, "number_check_summary.csv"))

    if args.live:
        print("\n=== 3. Live LLM answers, checked clause by clause (demo_questions.json) ===")
        live = evaluate_live(engine)
        print(f"{len(live)} clauses from {live.sentence_no.size and live.question.nunique()} questions")
        print(live.verdict.value_counts().to_string())
        print(live.verdict.value_counts(normalize=True).round(3).to_string())
        live.to_csv(os.path.join(OUT_DIR, "live_answers.csv"), index=False)

    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
