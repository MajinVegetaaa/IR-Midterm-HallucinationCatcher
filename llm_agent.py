"""
llm_agent.py — The Groq API Connection
=======================================
Sends the user's question to an LLM hosted on Groq (OpenAI-compatible API) and returns
the answer text. Closed-book: the model answers from its own memory and sees none of our
corpus; ir_engine.py checks the answer afterwards.

Responses are cached on disk (keyed by model + prompt) so the demo and the evaluation are
reproducible and don't burn free-tier rate limits on repeated questions.
"""

import hashlib
import json
import os

from dotenv import load_dotenv
from openai import AuthenticationError, NotFoundError, OpenAI, PermissionDeniedError, RateLimitError

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "llm_cache")

SYSTEM_PROMPT = (
    "You are a knowledgeable science and health assistant. Answer the user's question in "
    "one short paragraph of 4-6 factual sentences. State specific findings, figures and "
    "study results where you know them. Do not use bullet points or headings."
)


def _cache_path(model, question):
    key = hashlib.sha256(f"{model}\n{SYSTEM_PROMPT}\n{question}".encode()).hexdigest()[:24]
    return os.path.join(CACHE_DIR, f"{key}.json")


def load_keys(extra=None):
    """All usable Groq keys, in order: keys typed in the UI, then GROQ_API_KEYS, then GROQ_API_KEY.

    GROQ_API_KEYS takes several keys separated by commas or newlines; duplicates are dropped.
    """
    raw = []
    if extra:
        raw += [extra] if isinstance(extra, str) else list(extra)
    raw += [os.getenv("GROQ_API_KEYS", ""), os.getenv("GROQ_API_KEY", "")]
    keys = []
    for chunk in raw:
        for k in chunk.replace(",", "\n").split("\n"):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    return keys


def mask(key):
    """Show a key safely in the UI, e.g. 'gsk_…9f3a'."""
    return f"{key[:4]}…{key[-4:]}" if len(key) > 8 else "…"


# Index of the key that last succeeded, so the next call starts there instead of
# hammering a key that just hit its rate limit.
_last_good = 0


def ask(question, api_key=None, model=None, use_cache=True, return_key=False):
    """Return the Groq-hosted LLM's closed-book answer to `question`, and whether it came from cache.

    `api_key` may be one key or a list of keys. If a key is rate-limited (HTTP 429), invalid
    (401) or out of quota, the next key is tried automatically.
    """
    global _last_good
    model = model or DEFAULT_MODEL
    path = _cache_path(model, question)
    if use_cache and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            answer = json.load(f)["answer"]
        return (answer, True, None) if return_key else (answer, True)

    keys = load_keys(api_key)
    if not keys:
        raise RuntimeError("No Groq API key: set GROQ_API_KEYS in .env or paste keys in the sidebar.")

    start = _last_good % len(keys)
    order = keys[start:] + keys[:start]
    errors = []
    for key in order:
        client = OpenAI(api_key=key, base_url=GROQ_BASE_URL, max_retries=0)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": question}],
                temperature=0.3,
            )
        except NotFoundError:
            # Same for every key: the model name itself is wrong or retired. Don't burn the other keys.
            raise RuntimeError(f"Groq has no model called '{model}'. Pick a current one from "
                               "https://console.groq.com/docs/models and set LLM_MODEL in .env "
                               "(or the Model box in the sidebar).") from None
        except (RateLimitError, AuthenticationError, PermissionDeniedError) as e:
            errors.append(f"{mask(key)}: {type(e).__name__}")
            continue  # this key is exhausted or invalid -> try the next one
        _last_good = keys.index(key)
        break
    else:
        raise RuntimeError("All Groq keys failed — " + "; ".join(errors))

    answer = response.choices[0].message.content.strip()
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": model, "question": question, "answer": answer}, f, indent=2)
    return (answer, False, mask(key)) if return_key else (answer, False)

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Does vitamin D supplementation reduce cancer risk?"
    text, cached = ask(q)
    print(("[cached] " if cached else "") + text)
