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
