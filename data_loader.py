"""
data_loader.py — Corpus & Chunking Engine
==========================================
Downloads the SciFact dataset (Wadden et al., EMNLP 2020, allenai) and turns its
~5,183 scientific abstracts into overlapping sentence-window chunks.

Run:  python data_loader.py
Out:  data/corpus.json   (list of chunks)
      data/raw/*.jsonl   (original SciFact files: corpus + labelled claims)
"""

import io
import json
import os
import tarfile

import nltk
import requests

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.json")

# IR Concept: Chunking — deciding "what is a document".
# A whole abstract is too coarse to support one sentence-level claim, and a single
# sentence loses context. We use a sliding window of WINDOW sentences moving STRIDE
# sentences at a time, so neighbouring chunks overlap by one sentence and a fact that
# spans a sentence boundary is never split across two chunks.
WINDOW = 3
STRIDE = 2


def ensure_nltk():
    """Download the NLTK resources used across the project (sentence splitter + stop words)."""
    for pkg, path in [("punkt", "tokenizers/punkt"),
                      ("punkt_tab", "tokenizers/punkt_tab"),
                      ("stopwords", "corpora/stopwords")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def download_scifact():
    """Fetch and extract the SciFact release tarball into data/raw (skipped if already present)."""
    needed = ["corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl"]
    if all(os.path.exists(os.path.join(RAW_DIR, f)) for f in needed):
        print("SciFact already downloaded.")
        return
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Downloading SciFact from {SCIFACT_URL} ...")
    resp = requests.get(SCIFACT_URL, timeout=120)
    resp.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = os.path.basename(member.name)
            if name in needed:
                with open(os.path.join(RAW_DIR, name), "wb") as out:
                    out.write(tar.extractfile(member).read())
    print("Extracted:", ", ".join(needed))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_abstract(doc):
    """Split one SciFact abstract (already a list of sentences) into overlapping windows."""
    sentences = [s.strip() for s in doc["abstract"] if s.strip()]
    chunks = []
    start = 0
    while True:
        window = sentences[start:start + WINDOW]
        if not window:
            break
        chunks.append({
            "chunk_id": f"{doc['doc_id']}_{start}",
            "doc_id": doc["doc_id"],
            # IR Concept: Zones — the title is kept as its own field so the engine can
            # index it alongside the body and the UI can show where evidence came from.
            "title": doc["title"].strip(),
            "text": " ".join(window),
            "sent_ids": list(range(start, start + len(window))),
        })
        if start + WINDOW >= len(sentences):
            break
        start += STRIDE
    return chunks


def build_corpus():
    ensure_nltk()
    download_scifact()
    docs = read_jsonl(os.path.join(RAW_DIR, "corpus.jsonl"))
    corpus = [chunk for doc in docs for chunk in chunk_abstract(doc)]
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f)
    print(f"{len(docs)} abstracts -> {len(corpus)} chunks saved to {CORPUS_PATH}")
    return corpus


if __name__ == "__main__":
    build_corpus()
