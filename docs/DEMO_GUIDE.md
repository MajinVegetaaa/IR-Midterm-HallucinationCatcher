# Demo guide — Fact-Check RAG (5–8 min video)

This is the script for the recorded demo: what to show, in which order, and what each example should
produce. Every verdict below was checked against the current code (`ir_engine.py`, thresholds as
committed). If you change a threshold slider during recording, the colours may change.

## How the demo examples were chosen (say this in the video and keep it in the report)

We are open about this so the demo is not mistaken for typical performance:

1. **Live LLM questions were chosen for corpus coverage.** SciFact is 5,183 research abstracts, not a
   general medical encyclopedia. We generated 12 candidate questions from SciFact's most frequent title
   topics (cancer, stem cells, DNA, diabetes, HIV, …), asked the LLM each one, and kept the ones whose
   answers the corpus can actually check. Even on these, most clauses come out 🟡 UNVERIFIED: the model
   writes general textbook statements, while the corpus holds specific findings. **The best candidate (DNA
   methylation) had 3 of 7 clauses green, 5 of the 12 candidates had none, and the median was 1.**
2. **Paste-mode examples have known answers.** They are SciFact claims with expert gold labels, or
   controlled edits of them (one quantity changed), so the audience can see the system is right or wrong
   against ground truth, not against our opinion.
3. **The evaluation set was not curated.** The 15 questions in `demo_questions.json` were fixed before
   the system was tuned and are reported as-is (`outputs/live_answers.csv`). The headline metrics come from
   the labelled SciFact dev split, which we did not choose.

## Pre-flight (do once before recording)

```bash
cd midproject/factcheck_rag && source .venv/bin/activate
python evaluate.py --live      # refreshes outputs/ and caches every LLM answer used below
streamlit run app.py
```

- Leave **"Use cached answers"** on, so the live questions return instantly and identically every take.
- Default thresholds (don't touch the sliders unless you are demonstrating them).
- Close other tabs. Make sure no API key is visible (the key field is a password box; keys show masked).

## Running order (matches the brief's "What the video must show")

| # | Time | What | Who |
|---|---|---|---|
| 1 | 0:00–0:50 | Problem and why it is Track 1 | Member 3 |
| 2 | 0:50–3:00 | System running end to end (live + paste examples, incl. limitations) | Member 3 |
| 3 | 3:00–5:00 | Pipeline with real intermediate output (X-ray, code) | Member 1 |
| 4 | 5:00–6:30 | Evaluation vs baseline | Member 2 |
| 5 | 6:30–7:30 | Limitations, next steps, each member's part | all |

### 1. Problem (≤ 1 min)

> "LLMs answer health questions confidently even when they are wrong. Track 1 asks for a RAG system where
> every generated claim can be traced to a ranked source using an IR component you can inspect. Ours
> splits an LLM answer into claims and checks each one against 5,183 research abstracts with BM25."

### 2. System running end to end

**2a. Live LLM question** (sidebar: *Ask the LLM*). Click the example pill:

| Question | What you should see |
|---|---|
| *How does DNA methylation regulate gene expression in cancer?* | 7 clauses: **3 🟢**, 2 🟡, 2 🔴. Point at a green clause and its source paper, then at the 🔴 "≈30% of colorectal cancers show MLH1 promoter methylation" clause: the corpus discusses this topic but not that figure. |

Say: *"Most real LLM sentences come out amber. That is the honest result: amber means our corpus of
5,183 abstracts does not contain the claim, not that it is false."*

Optional second question: *How does smoking affect lung cancer risk?* (1 🟢, 2 🟡, 3 🔴).

**2b. Paste mode with known answers** (sidebar: *Paste text to check*, then click the pill and **Check**):

| Pill | Expected | Point to make |
|---|---|---|
| Correct claim — *Activation of PPM1D suppresses p53 function.* | 🟢 coverage 1.00, BM25 28.5, source = the gold SciFact abstract | Every claim term is in the evidence |
| Real figure (50%) — *…quit smoking reduce lung cancer risk by approximately 50%.* | 🟢 coverage 0.87, source "Effect of smoking reduction on lung cancer risk" | — |
| Changed figure (157%) — same sentence, 157% | 🔴 **number mismatch**, **same source paper** | Same evidence, one figure changed → caught. Term overlap alone would still say green. |
| Compound sentence | Split into 2 clauses: 🟢 ALDH1 clause + 🔴 biomaterials clause | Long sentences are split before checking (query-side "what is a unit") |
| Not in corpus — *ISRO built a secret alien base on the moon.* | 🟡 UNVERIFIED, missing: isro, moon, built… | The system declines to judge rather than guess |

**2c. Limitations (required by the brief)** — show at least one:

| Pill | What happens | Why (say this) |
|---|---|---|
| Limitation: negation — *LDL cholesterol has **no** involvement in the development of cardiovascular disease.* | 🟢 coverage 0.87 — **wrong**: SciFact's experts label it CONTRADICT | "no" and "not" are **stop words**, so they are deleted before retrieval. Term matching cannot see negation; 62.5% of CONTRADICT claims are shown green. |
| Limitation: averaging — *Vitamin D deficiency is associated with an increased risk of tuberculosis.* | 🟢 coverage 0.72 — **wrong**: the source is about fractures | Coverage is an average; the X-ray lists **tuberculosi** first under "not found in evidence". |

### 3. Pipeline with intermediate output

Open the X-ray expander of the **Correct claim** and walk down it:

1. Raw → stop-words removed → stemmed tokens (`activ ppm1d suppress p53 function`).
2. **Per-term BM25 table**: tf, df, idf, contribution. `ppm1d` has df = 8 → idf 7.842 → 15.14 of the 28.51
   score. The five contributions add up to the score shown.
3. **Postings lists**: `ppm1d df=8 -> [...]`.
4. **Top-K table**: the 5 BM25 candidates and their coverage; the evidence is the best by coverage.

Then briefly show code (`ir_engine.py`): `analyze()`, `search()` (heapq top-K), `coverage()`,
`split_claims()`, `decide_verdict()`. Search for `IR Concept:` comments.

### 4. Evaluation (show `outputs/`)

| Show | Number to say |
|---|---|
| `retrieval.png` | BM25 beats tf-idf cosine: P@1 0.718 vs 0.660, MRR 0.793 vs 0.746 (188 SciFact dev claims) |
| `verifier.png` | Verifier F1 0.680 vs 0.622 for raw-BM25-threshold baseline; NEI claims wrongly green 79.5% → 30.4% |
| `verifier_sweep.png` | Threshold tuned on train (0.55) is also the dev optimum |
| `compound_test.csv` | True claim inside a long sentence shown green: 8.1% whole → 81.5% after splitting |
| `number_check_summary.csv` | Correct claim with one quantity changed, still green: 8/10 without the check → 0/10 with it (small sample) |

### 5. Limitations and next steps

- Negation (stop-word removal deletes "no/not"); contradictions often shown green.
- Coverage is an average (tuberculosis example).
- Corpus scope: on general LLM text most clauses are amber, and 29% red, where red mostly means "specific
  figures/studies not in our 5,183 abstracts". This is not a hallucination rate.
- Next: negation cues and proximity (positional index), a larger corpus (PubMed), dense/hybrid retrieval,
  RAG generation mode.
- Each member: one sentence on the component they own.

## Things not to claim

- ❌ "It detects hallucinations with X% accuracy on LLM output." There are no gold labels for LLM text.
- ❌ "Red means the model lied." Red means the corpus covers the topic but not these specifics.
- ❌ "It detects contradictions." It mostly cannot (negation).
- ✅ "On SciFact's labelled claims, it identifies supported claims with F1 0.68 vs 0.62 for the baseline,
  and every verdict comes with a ranked, inspectable source."
