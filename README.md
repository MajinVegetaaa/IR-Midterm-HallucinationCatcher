# Fact-Check RAG: The Hallucination Catcher

CSD358 IR Hackathon — **Track 1: Retrieval-Augmented Generation and trustworthy answers**

An LLM answers a science or health question from memory. The answer is split into
sentences and **each sentence is run as a query against a BM25 index of 5,183 scientific abstracts**
(SciFact). Every sentence is coloured:

| Verdict | Meaning |
|---|---|
| 🟢 SUPPORTED | a retrieved chunk contains most of the sentence's informative (high-idf) terms |
| 🟡 UNVERIFIED | the corpus has nothing relevant enough to judge |
| 🔴 HALLUCINATED | the corpus covers the topic but not the specifics, or the figures don't match |

The **IR X-Ray** panel shows each step for every sentence: raw → stop-word-free → stemmed tokens,
per-term BM25 contributions (tf, df, idf, score), postings lists, the top-K candidates, idf coverage,
the claim terms missing from the evidence, and the source chunk.

📄 **Full technical report** (architecture diagrams, algorithms, code walkthrough, evaluation):
[`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md)

## Setup

```bash
cd midproject/factcheck_rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # GROQ_API_KEYS=key1,key2 (free at console.groq.com/keys)

python data_loader.py         # downloads SciFact, writes data/corpus.json (21,651 chunks)
streamlit run app.py          # the dashboard
python evaluate.py            # metrics + charts -> outputs/
python evaluate.py --live     # also verifies Groq answers to demo_questions.json
```

Put your Groq api keys in `.env`, and/or paste 
comma-separated keys into the UI sidebar.
The answer header shows which key was used (masked). `GROQ_API_KEY` (single key) still works.

No key? Choose **"Paste text to check"** in the sidebar to verify any paragraph offline.

## Data

**SciFact** — Wadden et al., *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020
(allenai, CC BY-NC 2.0). The corpus has 5,183 abstracts. The labelled claims are
`claims_train` (809, used for tuning) and `claims_dev` (300, used for reporting). Each labelled claim
lists its evidence abstracts and whether they SUPPORT or CONTRADICT it.

## Pipeline and where each IR concept lives

```
LLM answer ─► sent_tokenize ─► split_claims (long sentences → short clauses) ─► for each clause:
   tokenize ─► case-fold ─► stop words ─► Porter stem          ir_engine.analyze()
   ─► BM25 over inverted index ─► heap top-K (K=5)             ir_engine.search()
   ─► idf-weighted coverage per candidate, pick best           ir_engine.coverage()
   ─► number-mismatch check (only if evidence covers claim)    ir_engine.verify_claim()
   ─► verdict                                                  ir_engine.decide_verdict()
```

| Concept | Where |
|---|---|
| Chunking ("what is a document"): 3-sentence windows, stride 2 | `data_loader.chunk_abstract` |
| Zones (title + body indexed, title shown as source) | `data_loader`, `AdvancedIREngine.__init__` |
| Tokenisation, case-folding, stop words, Porter stemming | `ir_engine.analyze` |
| Inverted index / postings lists | `AdvancedIREngine.postings_index`, `postings()` |
| df / idf | `AdvancedIREngine.idf`, `term_breakdown` |
| BM25 (beyond syllabus) | `rank_bm25.BM25Okapi`, per-term formula in `term_breakdown` |
| Heap-based top-K | `AdvancedIREngine.search` (`heapq.nlargest`) |
| tf-idf + cosine (baseline) | `AdvancedIREngine.search_tfidf` |
| Query-side "what is a unit?" (clause splitting) | `ir_engine.split_claims`, `check_sentence` |
| Length-normalised BM25 ("on topic" test) | `ir_engine.verify_claim` (`norm_top`) |
| Precision, recall, P@k, MRR | `evaluate.py` |

Search for `IR Concept:` in the code to find each one.

## Results (`python evaluate.py`)

**Retrieval** — SciFact dev claims with gold evidence (188 queries), abstract-level:

| Retriever | P@1 | P@5 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|
| tf-idf cosine (baseline) | 0.660 | 0.180 | 0.857 | 0.899 | 0.746 |
| **BM25** | **0.718** | **0.189** | **0.891** | **0.931** | **0.793** |

**Verifier** — is the claim SUPPORTED? Thresholds tuned on train, reported on dev (300 claims):

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Baseline: raw BM25 score > threshold | 0.458 | 0.968 | 0.622 |
| **Ours: idf coverage + number check** | **0.580** | 0.823 | **0.680** |

Share of claims wrongly shown green, by gold label:
- **NOT ENOUGH INFO:** baseline 79.5% → ours **30.4%**.
- **CONTRADICT:** baseline 82.8% → ours 62.5%.

**Targeted tests** (labelled, built from SciFact dev):

| Test | Result |
|---|---|
| **Clause splitting.** A true claim joined to an unbacked one in a long sentence, shown green | whole sentence **8.1%** → after splitting **81.5%** (checked alone: 82.3%) |
| **Number check.** A correct claim with one quantity changed (e.g. 50% → 157%), still shown green | without the check **8/10** → with it **0/10** (small sample: SciFact has few quantity claims) |

**Live LLM answers** (`python evaluate.py --live`, 15 questions): 97 clauses → 🔴 29% · 🟡 66% · 🟢 5%.
**This is not a hallucination rate.** There are no gold labels for LLM text, and most remaining red
clauses are real findings (e.g. named statin trials) that SciFact's 5,183 abstracts don't contain. Red
on general LLM text mostly means "specifics not found in our corpus". Details: `docs/TECHNICAL_REPORT.md`
§11.6.

Charts: `outputs/retrieval.png`, `verifier.png`, `verifier_sweep.png`, `coverage_by_label.png`.

## Demo and how its examples were chosen

The recorded demo follows [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md). To be transparent:

- **Live questions were picked for corpus coverage.** We tested 12 candidate questions built from
  SciFact's most frequent topics and show the ones the corpus can actually check. Even the best
  (DNA methylation) has 3 of 7 clauses green; 5 of the 12 had none. Real LLM answers come out mostly 🟡.
- **Paste-mode examples have known answers**: SciFact claims with expert labels, or one quantity edited.
  They are one-click buttons in the app.
- **The evaluation was not curated.** The 15 questions in `demo_questions.json` were fixed before tuning and
  are reported as-is, and the headline metrics use SciFact's labelled dev split.

## What works / limitations / planned

**Works:** the full closed-book pipeline (Groq → sentence and clause split → BM25 verification → coloured
clauses + X-ray), multi-key failover, paste mode, threshold sliders, cached LLM answers, and the
reproducible evaluation.

**Limitations (shown in the demo):**
- **Negation and contradiction.** "X increases Y" and "X does not increase Y" share almost every term,
  and "no"/"not" are stop words, so they are deleted before retrieval. Term overlap can't tell the two
  apart, which is why 62.5% of CONTRADICT claims still show green.
- **Paraphrase.** A sentence that is correct but worded differently from the abstract gets low coverage.
- **Coverage is an average.** "Vitamin D supplementation cures all cancers within a week" can still
  pass, because the rare terms "vitamin D", "supplementation" and "cancer" outweigh the missing "cure"
  and "week". Raising the coverage slider trades recall for precision (see `verifier_sweep.png`).
- **Corpus scope.** True facts that aren't in the 5,183 abstracts come out UNVERIFIED or red, not
  "wrong". For general LLM text this dominates the result.
- **Rule-based clause splitting.** No parser: some clauses are cut badly, and statistics in parentheses
  are set aside rather than checked.

**Planned:** dense or hybrid retrieval for paraphrases, a negation-aware support check, feeding the
retrieved chunks to the LLM (RAG generation mode), and per-sentence citations.

## Work division
- **Poorab Mishra:** `data_loader.py`, `ir_engine.py`, retrieval evaluation.
- **Vishu Vardhan Chundu:** `llm_agent.py`, `demo_questions.json`, verifier evaluation.
- **Shreyas Achal:** `app.py` (Streamlit dashboard), report and video.

## AI-use declaration
Claude Code (Anthropic) was used to help design the system and write the code in this repository.
The answers being fact-checked are generated by an LLM served through the Groq API.
