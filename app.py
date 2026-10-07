"""
app.py — Fact-Check RAG: The Hallucination Catcher (Streamlit "X-Ray Dashboard")
=================================================================================
Column 1 (user side): ask a question -> LLM answer -> every sentence coloured by verdict.
Column 2 (IR X-ray):  per-sentence diagnostics: tokens, BM25 term weights, postings,
                      coverage, unsupported terms and the retrieved source chunk.

Run:  streamlit run app.py
"""

import html
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from nltk import sent_tokenize

import ir_engine
from data_loader import CORPUS_PATH, ensure_nltk
from ir_engine import AdvancedIREngine
from llm_agent import DEFAULT_MODEL, ask, load_keys, mask

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
st.set_page_config(page_title="Fact-Check RAG", page_icon="🔬", layout="wide")

VERDICT_STYLE = {
    "SUPPORTED": ("🟢", "rgba(46, 160, 67, 0.22)", "#2ea043"),
    "UNVERIFIED": ("🟡", "rgba(210, 153, 34, 0.22)", "#d29922"),
    "HALLUCINATED": ("🔴", "rgba(248, 81, 73, 0.22)", "#f85149"),
}

# Demo questions on topics SciFact covers well (chosen by testing 12 candidates; docs/DEMO_GUIDE.md).
EXAMPLES = [
    "How does DNA methylation regulate gene expression in cancer?",
    "How does smoking affect lung cancer risk?",
    "Does vitamin D supplementation reduce the risk of cancer?",
]

# Paste-mode examples whose correct answer is known from SciFact gold labels (docs/DEMO_GUIDE.md).
PASTE_EXAMPLES = {
    "Correct claim": "Activation of PPM1D suppresses p53 function.",
    "Real figure (50%)": "Participants who quit smoking reduce lung cancer risk by approximately 50%.",
    "Changed figure (157%)": "Participants who quit smoking reduce lung cancer risk by approximately 157%.",
    "Compound sentence": "ALDH1 expression is associated with poorer prognosis in breast cancer, "
                         "while 0-dimensional biomaterials show inductive properties.",
    "Not in corpus": "ISRO built a secret alien base on the moon.",
    "Limitation: negation": "LDL cholesterol has no involvement in the development of cardiovascular disease.",
    "Limitation: averaging": "Vitamin D deficiency is associated with an increased risk of tuberculosis.",
}

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.answer-box {line-height: 2.0; font-size: 1.02rem; padding: 0.9rem 1rem;
             border: 1px solid rgba(128,128,128,0.25); border-radius: 10px;}
.claim {padding: 2px 4px; border-radius: 4px; border-bottom: 2px solid;}
.pill {display: inline-block; padding: 1px 8px; margin: 2px 3px 2px 0; border-radius: 999px;
       font-family: monospace; font-size: 0.85rem; border: 1px solid rgba(128,128,128,0.35);}
.pill-miss {background: rgba(248, 81, 73, 0.15); border-color: #f85149;}
.source {font-size: 0.92rem; padding: 0.6rem 0.8rem; border-left: 3px solid #58a6ff;
         background: rgba(88, 166, 255, 0.08); border-radius: 4px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Building BM25 index over SciFact chunks…")
def load_engine():
    ensure_nltk()
    return AdvancedIREngine()


def pills(tokens, cls=""):
    if not tokens:
        return "<i>none</i>"
    return "".join(f'<span class="pill {cls}">{html.escape(t)}</span>' for t in tokens)


def render_answer(results):
    """Rebuild the paragraph with every checked clause wrapped in its verdict colour."""
    parts = []
    for i, r in enumerate(results, start=1):
        _, bg, border = VERDICT_STYLE[r["verdict"]]
        tip = f'C{i} · {r["verdict"]} · BM25 {r["score"]} · coverage {r["coverage"]}'
        parts.append(f'<span class="claim" style="background:{bg};border-color:{border}" '
                     f'title="{html.escape(tip)}">{html.escape(r["sentence"])}</span>')
    st.markdown(f'<div class="answer-box">{" ".join(parts)}</div>', unsafe_allow_html=True)


def render_xray(engine, i, r):
    icon = VERDICT_STYLE[r["verdict"]][0]
    label = f'{icon} C{i} · {r["verdict"]} · BM25 {r["score"]} · coverage {r["coverage"]:.2f}'
    with st.expander(label, expanded=(r["verdict"] != "SUPPORTED")):
        st.caption(r["sentence"])
        if r["dropped"]:
            st.caption("Set aside before checking (statistics in parentheses): " + "  ".join(r["dropped"]))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("BM25 (evidence)", r["score"])
        m2.metric("Top-1 BM25 (normalised)", f'{r["top_score"]} ({r["norm_top"]:.2f})')
        m3.metric("idf coverage", f'{r["coverage"]:.2f}')
        m4.metric("Evidence rank", r["chunk_rank"] or "—")

        st.markdown("**Query processing** (tokenize → stop-words removed → Porter stem)")
        st.markdown(f'Raw: {pills(r["tokens_raw"])}', unsafe_allow_html=True)
        st.markdown(f'After stop-words: {pills(r["tokens_no_stop"])}', unsafe_allow_html=True)
        st.markdown(f'Stemmed (queried): {pills(r["tokens_matched"])}', unsafe_allow_html=True)
        st.markdown(f'Not found in evidence (highest idf first): '
                    f'{pills(r["unsupported_terms"], "pill-miss")}', unsafe_allow_html=True)
        if r["number_mismatch"]:
            st.error(f'Number mismatch: the claim states {", ".join(r["claim_numbers"])}, '
                     f'which the evidence chunk does not contain.')

        if r["term_breakdown"]:
            st.markdown("**Per-term BM25 contribution to the evidence chunk**")
            st.dataframe(pd.DataFrame(r["term_breakdown"]), hide_index=True, width="stretch")

        if r["source_chunk"]:
            st.markdown(f'**Source chunk** — doc `{r["doc_id"]}` · *{html.escape(r["title"])}*')
            st.markdown(f'<div class="source">{html.escape(r["source_chunk"])}</div>', unsafe_allow_html=True)
            st.markdown("**Top-K BM25 candidates** (re-checked for coverage)")
            st.dataframe(pd.DataFrame(r["top_k"]), hide_index=True, width="stretch")
        else:
            st.info("No chunk shares any term with this sentence.")

        terms = list(dict.fromkeys(r["tokens_matched"]))[:6]
        if terms:
            st.markdown("**Postings lists** (df · first chunk ids)")
            st.code("\n".join(
                f'{t:<14} df={len(engine.postings(t)):<6} -> '
                f'{[engine.chunks[j]["chunk_id"] for j in engine.postings(t)[:5]]}'
                for t in terms), language="text")


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    source = st.radio("Answer source", ["Ask the LLM (Groq)", "Paste text to check"],
                      help="Paste mode checks any text without calling the API.")
    ui_keys = st.text_input("Extra Groq API keys", type="password", placeholder="gsk_…, gsk_…",
                            help="Optional. Several keys separated by commas. Tried before the "
                                 "keys in .env; if one is rate-limited, the next is used.")
    all_keys = load_keys(ui_keys)
    st.caption(f"{len(all_keys)} key(s) available: " + ", ".join(mask(k) for k in all_keys)
               if all_keys else "No keys yet — add GROQ_API_KEYS to .env or paste above.")
    model = st.text_input("Model", value=DEFAULT_MODEL)
    use_cache = st.toggle("Use cached answers", value=True,
                          help="Re-asking the same question reuses the stored answer.")
    st.divider()
    st.subheader("Thresholds")
    support_cov = st.slider("SUPPORTED: min coverage", 0.0, 1.0, ir_engine.SUPPORT_COVERAGE, 0.05)
    halluc_cov = st.slider("HALLUCINATED: max coverage", 0.0, 1.0, ir_engine.HALLUCINATION_COVERAGE, 0.05)
    min_bm25 = st.slider("SUPPORTED: min BM25", 0.0, 40.0, ir_engine.MIN_BM25, 1.0)
    min_norm = st.slider("HALLUCINATED: min normalised BM25 (on topic)", 0.0, 0.5, ir_engine.MIN_NORM_BM25, 0.01)
    st.divider()
    st.caption("Corpus: SciFact (Wadden et al., 2020) — 5,183 abstracts in 3-sentence chunks.")
    st.caption("🟢 supported · 🟡 not in corpus · 🔴 likely hallucinated")

# ----------------------------------------------------------------------------- main
st.title("🔬 Fact-Check RAG: The Hallucination Catcher")
st.caption("An LLM answers from memory; a BM25 index over scientific abstracts checks every claim in the answer.")

if not os.path.exists(CORPUS_PATH):
    st.error("Corpus not found. Run `python data_loader.py` first.")
    st.stop()
engine = load_engine()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("💬 Ask")
    if source == "Ask the LLM (Groq)":
        ex = st.pills("Examples", EXAMPLES, selection_mode="single")
        question = st.chat_input("Ask a science or health question…") or ex
        if question and question != st.session_state.get("last_input"):
            st.session_state.last_input = question
            try:
                with st.spinner("Asking the LLM…"):
                    answer, cached, used_key = ask(question, api_key=ui_keys, model=model,
                                                   use_cache=use_cache, return_key=True)
                st.session_state.run = {"question": question, "answer": answer, "cached": cached,
                                        "key": used_key}
            except Exception as e:
                st.session_state.run = None
                st.error(f"LLM call failed: {e}")
    else:
        def _load_example():
            pick = st.session_state.get("paste_example")
            if pick:
                st.session_state.paste_text = PASTE_EXAMPLES[pick]

        st.pills("Load an example", list(PASTE_EXAMPLES), selection_mode="single",
                 key="paste_example", on_change=_load_example)
        pasted = st.text_area("Text to check", height=160, key="paste_text",
                              placeholder="Paste a paragraph, e.g. an LLM answer…")
        if st.button("Check", type="primary") and pasted.strip():
            st.session_state.run = {"question": None, "answer": pasted.strip(), "cached": False}

    run = st.session_state.get("run")
    if run:
        if run["question"]:
            st.markdown(f"**Question:** {run['question']}")
            st.markdown("**Raw LLM answer**" + (" *(cached)*" if run["cached"] else "")
                        + (f" · key `{run['key']}`" if run.get("key") else ""))
            st.info(run["answer"])

        # IR Concept: each sentence is split into short clauses (the query-side "what is a unit?"
        # decision) and every clause becomes one query against the index.
        sentences = [s for s in sent_tokenize(run["answer"]) if s.strip()]
        results = [r for s in sentences
                   for r in engine.check_sentence(s, support_cov=support_cov, halluc_cov=halluc_cov,
                                                  min_bm25=min_bm25, min_norm=min_norm)]

        st.markdown("**Fact-checked answer**")
        render_answer(results)
        counts = pd.Series([r["verdict"] for r in results]).value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Supported", int(counts.get("SUPPORTED", 0)))
        c2.metric("🟡 Unverified", int(counts.get("UNVERIFIED", 0)))
        c3.metric("🔴 Hallucinated", int(counts.get("HALLUCINATED", 0)))

with col2:
    st.subheader("🩻 IR X-Ray")
    if run:
        for i, r in enumerate(results, start=1):
            render_xray(engine, i, r)
    else:
        st.caption("Diagnostics for every sentence will appear here.")
