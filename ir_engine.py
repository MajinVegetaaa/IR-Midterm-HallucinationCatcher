"""
ir_engine.py — The Backend Core
================================
Loads the chunked SciFact corpus, builds a BM25 index over it and checks whether a
single sentence (one claim from the LLM's answer) is backed by a source chunk.

Pipeline for one sentence:
    tokenize -> case-fold -> stop-word removal -> Porter stem        (query processing)
    -> BM25 over the inverted index -> heap top-K                       (retrieval)
    -> idf-weighted term coverage + number check on the top-K           (support test)
    -> verdict: SUPPORTED / UNVERIFIED / HALLUCINATED
"""

import heapq
import json
import math
import os
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ----------------------------------------------------------------------------
# Tunable thresholds (defaults were tuned on SciFact claims_train by evaluate.py)
# ----------------------------------------------------------------------------
# Raw-BM25 baseline: a sentence is "supported" if its top BM25 score exceeds this.
THRESHOLD = 18.0
# Minimum BM25 score for the corpus to count as "on topic" for this sentence at all.
# Below it, the corpus simply has nothing to say -> UNVERIFIED rather than HALLUCINATED.
MIN_BM25 = 14.0
# idf-weighted share of the claim's terms that must appear in the evidence chunk.
SUPPORT_COVERAGE = 0.55
# On-topic but this little of the claim is in the evidence -> likely HALLUCINATED.
HALLUCINATION_COVERAGE = 0.35
# "On topic" for the red verdict is judged on a length-normalised BM25: top score divided by the
# most this query could score (every term at tf -> infinity: sum(idf) * (k1 + 1)). Raw BM25 grows
# with query length, so on long LLM clauses it called unrelated chunks "on topic". 0.15 is the
# largest value that keeps >= 80% of red verdicts truly unsupported on SciFact train.
MIN_NORM_BM25 = 0.15
# How many BM25 candidates are re-checked for coverage.
TOP_K = 5
# A number mismatch only counts when the evidence chunk covers the claim at least this much:
# figures can only contradict evidence that is about the same thing. Same value as the
# support threshold, i.e. "this chunk would support the claim if the figures agreed".
NUMBER_GATE_COVERAGE = 0.55
# Claim splitting: sentences with more distinct terms than SPLIT_ABOVE_TERMS are cut into
# clauses; a clause needs at least CLAUSE_MIN_TERMS terms or it is merged into a neighbour.
# SciFact claims average ~10 terms, which is the scale the thresholds were tuned on.
SPLIT_ABOVE_TERMS = 12
CLAUSE_MIN_TERMS = 4

CORPUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "corpus.json")

_STEMMER = PorterStemmer()
_STOPWORDS = None
# IR Concept: Tokenization — alphanumeric runs; decimals like "2.5" stay one token,
# hyphenated words ("covid-19") are split so "covid 19" and "covid-19" match.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _stopwords():
    global _STOPWORDS
    if _STOPWORDS is None:
        # NLTK lists contraction fragments ("d" from "I'd", "t" from "don't") as stop words,
        # but in biomedical text they are content: "vitamin D", "T cells", "protein S".
        _STOPWORDS = set(stopwords.words("english")) - {"d", "t", "s", "m", "o", "y"}
    return _STOPWORDS


@lru_cache(maxsize=200_000)
def _stem(word):
    return _STEMMER.stem(word)


# "20 000", "20,000" and "20\u202f000" are all the number 20000, not "20" and "000".
_THOUSANDS_RE = re.compile(r"\b(\d{1,3})((?:[ ,\u202f\u00a0]\d{3})+)\b")


def normalize_text(text):
    """Unicode (NFKC: no-break spaces and hyphens, ligatures) and thousands-separator clean-up."""
    text = unicodedata.normalize("NFKC", text)
    return _THOUSANDS_RE.sub(lambda m: m.group(1) + re.sub(r"\D", "", m.group(2)), text)


def analyze(text):
    """Run the full normalisation pipeline and return every intermediate stage."""
    # IR Concept: Case-folding — "BRCA1", "Brca1" and "brca1" become the same term.
    folded = normalize_text(text).lower().replace(",", "")
    raw = _TOKEN_RE.findall(folded)
    # IR Concept: Stop-word removal — function words ("the", "of", "is") carry no
    # topical signal and would match almost every chunk.
    no_stop = [t for t in raw if t not in _stopwords()]
    # IR Concept: Stemming (Porter) — "infections", "infected", "infect" -> "infect",
    # so a claim and its evidence match despite different word forms.
    stemmed = [_stem(t) for t in no_stop]
    return {"raw": raw, "no_stop": no_stop, "stemmed": stemmed}


def tokenize(text):
    return analyze(text)["stemmed"]


def extract_numbers(text):
    """Numbers in the text, normalised ("1,000" -> "1000", "2.50" -> "2.5")."""
    out = set()
    for n in _NUMBER_RE.findall(normalize_text(text).replace(",", "")):
        out.add(n.rstrip("0").rstrip(".") if "." in n else n)
    return out


# Parentheticals that carry statistics ("(HR 0.88, 95% CI 0.78-0.99)", "(n = 25)") are checked
# on their own terms in the text; as part of a clause they only dilute coverage.
_STAT_PAREN_RE = re.compile(r"\s*[\(\[][^()\[\]]*\d[^()\[\]]*[\)\]]")
_CLAUSE_BOUNDARY_RE = re.compile(
    r"\s*;\s*|\s*[\u2014\u2013]\s+|\s*:\s+"
    r"|,\s+(?:but|while|whereas|although|though|yet|and|which|thereby|suggesting|indicating)\s+"
    r"|\s+(?:but|whereas|although)\s+", re.I)


def split_claims(sentence):
    """Cut a long sentence into short checkable clauses (rule-based, FActScore-style).

    IR Concept: the *query* needs the same "what is a unit?" decision as the corpus. Coverage
    is a share of the query's terms, so a 22-term sentence can almost never be covered by one
    3-sentence chunk even when every fact in it is true. Short sentences are returned as-is.
    """
    if len(set(tokenize(sentence))) <= SPLIT_ABOVE_TERMS:
        return [{"text": sentence.strip(), "dropped": []}]
    dropped = [m.strip() for m in _STAT_PAREN_RE.findall(sentence)]
    text = _STAT_PAREN_RE.sub("", sentence).strip()
    clauses = []
    for piece in _CLAUSE_BOUNDARY_RE.split(text):
        piece = (piece or "").strip(" ,.;:")
        if not piece:
            continue
        if clauses and len(set(tokenize(piece))) < CLAUSE_MIN_TERMS:
            clauses[-1] += ", " + piece          # fragment: glue to the previous clause
        else:
            clauses.append(piece)
    if len(clauses) > 1 and len(set(tokenize(clauses[0]))) < CLAUSE_MIN_TERMS:
        clauses[1] = clauses[0] + ", " + clauses[1]
        clauses = clauses[1:]
    if not clauses:
        return [{"text": sentence.strip(), "dropped": []}]
    return [{"text": c if c.endswith(".") else c + ".", "dropped": dropped if i == 0 else []}
            for i, c in enumerate(clauses)]


def decide_verdict(coverage, best_bm25, norm_top, number_mismatch,
                   support_cov=SUPPORT_COVERAGE, halluc_cov=HALLUCINATION_COVERAGE, min_bm25=MIN_BM25,
                   min_norm=MIN_NORM_BM25):
    """Three-way decision from the retrieval evidence for one sentence."""
    if coverage >= support_cov and best_bm25 >= min_bm25 and not number_mismatch:
        return "SUPPORTED"
    # On topic (the corpus does discuss this) but the specifics are absent or the figures
    # differ -> the sentence most likely adds something the sources never said.
    if number_mismatch or (norm_top >= min_norm and coverage < halluc_cov):
        return "HALLUCINATED"
    # Nothing relevant enough in the corpus to judge either way.
    return "UNVERIFIED"


class AdvancedIREngine:
    def __init__(self, corpus_path=CORPUS_PATH):
        with open(corpus_path, encoding="utf-8") as f:
            self.chunks = json.load(f)

        # IR Concept: Zones — title and body are both indexed, so a claim that names the
        # paper's subject (often only in the title) still reaches the right chunk.
        self.chunk_tokens = [tokenize(c["title"] + " " + c["text"]) for c in self.chunks]
        self.chunk_token_sets = [set(toks) for toks in self.chunk_tokens]

        # IR Concept: BM25 (Okapi) — beyond the syllabus. tf-idf with tf saturation (k1)
        # and document-length normalisation (b), so long chunks don't win just by size.
        self.bm25 = BM25Okapi(self.chunk_tokens, k1=1.5, b=0.75)

        # IR Concept: Inverted index — term -> postings list of chunk ids (sorted).
        # Built from the same term frequencies BM25 uses, so the X-ray can show postings.
        self.postings_index = defaultdict(list)
        for chunk_idx, freqs in enumerate(self.bm25.doc_freqs):
            for term in freqs:
                self.postings_index[term].append(chunk_idx)
        # Terms never seen in the corpus get the highest idf: an unseen word is maximally
        # specific, so failing to find it should weigh heavily against the claim.
        self.max_idf = max(self.bm25.idf.values())

        self._tfidf = None  # built lazily for the baseline comparison

    # ------------------------------------------------------------------ helpers
    def idf(self, term):
        # IR Concept: Inverse document frequency — log((N - df + 0.5) / (df + 0.5)).
        # Rare terms ("osteosarcoma") get a high idf; common ones ("patient") a low one.
        return self.bm25.idf.get(term, self.max_idf)

    def postings(self, term):
        """Postings list for a (stemmed) term: the chunk ids that contain it."""
        return self.postings_index.get(term, [])

    def term_breakdown(self, query_terms, chunk_idx):
        """Per-term BM25 contribution to one chunk's score (the 'X-ray' table)."""
        freqs = self.bm25.doc_freqs[chunk_idx]
        dl = self.bm25.doc_len[chunk_idx]
        k1, b, avgdl = self.bm25.k1, self.bm25.b, self.bm25.avgdl
        rows = []
        for term in dict.fromkeys(query_terms):  # unique, order kept
            tf = freqs.get(term, 0)
            idf = self.bm25.idf.get(term, 0.0)
            # IR Concept: BM25 term score = idf * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl))
            score = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl)) if tf else 0.0
            rows.append({"term": term, "tf": tf, "df": len(self.postings(term)),
                         "idf": round(idf, 3), "bm25": round(score, 3)})
        return rows

    def coverage(self, query_terms, chunk_idx):
        """idf-weighted share of the claim's terms present in the chunk."""
        terms = set(query_terms)
        if not terms:
            return 0.0, []
        chunk_terms = self.chunk_token_sets[chunk_idx]
        total = sum(self.idf(t) for t in terms)
        covered = sum(self.idf(t) for t in terms if t in chunk_terms)
        missing = sorted((t for t in terms if t not in chunk_terms), key=self.idf, reverse=True)
        return (covered / total if total > 0 else 0.0), missing

    # ------------------------------------------------------------------ retrieval
    def search(self, query, k=TOP_K):
        """BM25 ranked retrieval. Returns [(chunk_idx, score)] best first."""
        terms = tokenize(query) if isinstance(query, str) else query
        if not terms:
            return []
        scores = self.bm25.get_scores(terms)
        # IR Concept: Heap-based top-K — O(N log K) instead of sorting all N scores.
        top = heapq.nlargest(k, range(len(scores)), key=scores.__getitem__)
        return [(i, float(scores[i])) for i in top if scores[i] > 0]

    def search_tfidf(self, query, k=TOP_K):
        """Baseline: lnc-style tf-idf vectors + cosine similarity (sklearn)."""
        if self._tfidf is None:
            # IR Concept: Vector space model — log-tf (sublinear_tf), idf, L2 length
            # normalisation, so the dot product of two vectors is their cosine.
            self._tfidf = TfidfVectorizer(analyzer=lambda toks: toks, sublinear_tf=True, norm="l2")
            self._tfidf_matrix = self._tfidf.fit_transform(self.chunk_tokens)
        terms = tokenize(query) if isinstance(query, str) else query
        if not terms:
            return []
        q_vec = self._tfidf.transform([terms])
        sims = linear_kernel(q_vec, self._tfidf_matrix).ravel()
        top = heapq.nlargest(k, range(len(sims)), key=sims.__getitem__)
        return [(i, float(sims[i])) for i in top if sims[i] > 0]

    # ------------------------------------------------------------------ verification
    def verify_claim(self, sentence, support_cov=None, halluc_cov=None, min_bm25=None, threshold=None,
                     number_gate=None, min_norm=None):
        """Check one (short) claim against the corpus and explain the decision."""
        number_gate = NUMBER_GATE_COVERAGE if number_gate is None else number_gate
        min_norm = MIN_NORM_BM25 if min_norm is None else min_norm
        support_cov = SUPPORT_COVERAGE if support_cov is None else support_cov
        halluc_cov = HALLUCINATION_COVERAGE if halluc_cov is None else halluc_cov
        min_bm25 = MIN_BM25 if min_bm25 is None else min_bm25
        threshold = THRESHOLD if threshold is None else threshold

        stages = analyze(sentence)
        terms = stages["stemmed"]
        result = {
            "sentence": sentence,
            "score": 0.0,
            "top_score": 0.0,
            "norm_top": 0.0,
            "source_chunk": "",
            "tokens_matched": terms,
            "is_supported": False,
            "tokens_raw": stages["raw"],
            "tokens_no_stop": stages["no_stop"],
            "coverage": 0.0,
            "unsupported_terms": terms,
            "number_mismatch": False,
            "number_mismatch_raw": False,
            "claim_numbers": sorted(extract_numbers(sentence)),
            "verdict": "UNVERIFIED",
            "bm25_only_supported": False,
            "top_k": [],
            "term_breakdown": [],
            "doc_id": None,
            "title": "",
            "chunk_rank": None,
        }
        hits = self.search(terms, TOP_K)
        if not hits:
            return result

        # IR Concept: Two-stage ranking — BM25 finds the K most relevant chunks, then each
        # is re-scored by how much of the claim it actually contains (coverage). Topical
        # relevance alone is not support: "vitamin D cures cancer" matches any chunk about
        # vitamin D and cancer.
        candidates = []
        for rank, (idx, bm25_score) in enumerate(hits, start=1):
            cov, missing = self.coverage(terms, idx)
            candidates.append({"idx": idx, "rank": rank, "bm25": bm25_score,
                               "coverage": cov, "missing": missing})
        best = max(candidates, key=lambda c: (c["coverage"], c["bm25"]))
        chunk = self.chunks[best["idx"]]

        # Number check: if the claim states a figure and the evidence states other figures
        # but not this one, the claim most likely misquotes the source.
        claim_nums = extract_numbers(sentence)
        chunk_nums = extract_numbers(chunk["title"] + " " + chunk["text"])
        number_mismatch_raw = bool(claim_nums and chunk_nums and not claim_nums <= chunk_nums)
        number_mismatch = number_mismatch_raw and best["coverage"] >= number_gate

        top_score = hits[0][1]
        max_possible = (self.bm25.k1 + 1) * sum(self.idf(t) for t in set(terms))
        norm_top = top_score / max_possible if max_possible else 0.0
        verdict = decide_verdict(best["coverage"], best["bm25"], norm_top, number_mismatch,
                                 support_cov, halluc_cov, min_bm25, min_norm)

        result.update({
            "top_score": round(top_score, 3),
            "norm_top": round(norm_top, 3),
            "score": round(best["bm25"], 3),
            "source_chunk": chunk["text"],
            "is_supported": verdict == "SUPPORTED",
            "coverage": round(best["coverage"], 3),
            "unsupported_terms": best["missing"],
            "number_mismatch": number_mismatch,
            "number_mismatch_raw": number_mismatch_raw,
            "verdict": verdict,
            "bm25_only_supported": top_score > threshold,
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "chunk_rank": best["rank"],
            "term_breakdown": self.term_breakdown(terms, best["idx"]),
            "top_k": [{"rank": c["rank"], "chunk_id": self.chunks[c["idx"]]["chunk_id"],
                       "doc_id": self.chunks[c["idx"]]["doc_id"],
                       "title": self.chunks[c["idx"]]["title"],
                       "bm25": round(c["bm25"], 3), "coverage": round(c["coverage"], 3)}
                      for c in candidates],
        })
        return result

    def check_sentence(self, sentence, **thresholds):
        """Split a sentence into clauses and verify each; returns one result dict per clause."""
        results = []
        for clause in split_claims(sentence):
            r = self.verify_claim(clause["text"], **thresholds)
            r["parent_sentence"] = sentence
            r["dropped"] = clause["dropped"]
            results.append(r)
        return results


if __name__ == "__main__":
    engine = AdvancedIREngine()
    for s in ["Vitamin D deficiency is associated with an increased risk of tuberculosis.",
              "ISRO built a secret alien base on the far side of the moon.",
              "Vitamin D supplementation cures all cancers within a week."]:
        r = engine.verify_claim(s)
        print(f"\n{s}\n  verdict={r['verdict']} bm25={r['score']} coverage={r['coverage']} "
              f"missing={r['unsupported_terms'][:5]} doc={r['doc_id']}")
