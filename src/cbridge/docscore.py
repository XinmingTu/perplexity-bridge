"""Whole-document exact BPB: every token of every document, in one pass per document.

This is the standard perplexity measurement (as in BPBench): the model reads the full document and we keep the
log-probability of every token given everything before it. Speaker labels are kept apart so results can be
reported with and without them. Per-token log-probabilities and character offsets are saved for later analyses
(e.g. which kinds of words separate models).

Backends: Fireworks completions with echo, and Tinker compute_logprobs (tokenizer from the sampler itself).
The first token of each document has no prediction and is not scored.
"""

import asyncio
import gzip
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from .text import Document, utf8_len

FIREWORKS = "https://api.fireworks.ai/inference/v1"
LN2 = math.log(2)


def summarize(doc: Document, text: str, labels: list[tuple[int, int]], offs: list[int], lps: list[float],
              stop: int | None = None) -> dict:
    """Bits and bytes over scored tokens (index >= 1), split into speaker-label tokens and the rest.

    `offs[0]` is the document's first real token, which is never scored, on every backend. `stop` is where the
    scored text ends (the start of the first dropped token if the input was truncated). A token touching a
    speaker label counts as label, whatever the tokenizer merged it with.
    """
    ends = offs[1:] + [len(text) if stop is None else stop]
    bits = {"all": 0.0, "speech": 0.0}
    nbytes = {"all": 0, "speech": 0}
    for i in range(1, len(offs)):
        b, n = -lps[i] / LN2, utf8_len(text[offs[i]:ends[i]])
        in_label = any(offs[i] < c and ends[i] > a for a, c in labels if c > a)
        bits["all"] += b
        nbytes["all"] += n
        if not in_label:
            bits["speech"] += b
            nbytes["speech"] += n
    return dict(doc_id=doc.id, kind=doc.meta.get("kind", ""), source=doc.source, n_tokens=len(offs),
                bits=bits["all"], bytes=nbytes["all"], bits_speech=bits["speech"], bytes_speech=nbytes["speech"])


def _rows(path: Path) -> list[dict]:
    """Rows of a results file, ignoring a damaged tail left by an interrupted write."""
    rows = []
    if path.exists():
        try:
            with gzip.open(path, "rt") as f:
                for line in f:
                    rows.append(json.loads(line))
        except (EOFError, OSError, json.JSONDecodeError):
            pass
    return rows


def _done(path: Path) -> set:
    return {r["doc_id"] for r in _rows(path)}


def _clean_logprobs(doc_id: str, raw: list) -> list[float] | None:
    """Only position 0 may lack a log-probability; anything else means the backend dropped a token."""
    if any(x is None for x in raw[1:]):
        print(f"  {doc_id}: missing log-probability inside the document; skipped")
        return None
    return [0.0] + [float(x) for x in raw[1:]]


async def fireworks_docs(model: str, docs: list[Document], out: Path, concurrency: int = 6) -> None:
    m = model if "/" in model else f"accounts/fireworks/models/{model}"
    done, sem = _done(out), asyncio.Semaphore(concurrency)
    out.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=600, headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}) as client:
        async def one(doc: Document) -> dict | None:
            text, labels = doc.render()
            body = {"model": m, "prompt": text, "max_tokens": 1, "echo": True, "logprobs": 1, "temperature": 0}
            async with sem:
                for attempt in range(10):
                    try:
                        r = await client.post(f"{FIREWORKS}/completions", json=body)
                        if r.status_code == 200:
                            break
                        if r.status_code not in (429, 500, 502, 503, 504):
                            print(f"  {doc.id}: HTTP {r.status_code} {r.text[:120]}")
                            return None
                    except httpx.TransportError:
                        pass
                    await asyncio.sleep(min(60, 2 ** attempt))
                else:
                    return None
            d = r.json()
            lp = d["choices"][0]["logprobs"]
            # Prompt tokens are those starting inside the text; the generated token (if echoed) starts at its end.
            n = sum(o < len(text) for o in lp["text_offset"])
            offs, lps = lp["text_offset"][:n], _clean_logprobs(doc.id, lp["token_logprobs"][:n])
            if lps is None:
                return None
            row = summarize(doc, text, labels, offs, lps)
            row.update(offsets=offs, logprobs=[round(x, 4) for x in lps])
            return row

        todo = [d for d in docs if d.id not in done]
        for i in range(0, len(todo), 24):
            rows = await asyncio.gather(*(one(d) for d in todo[i:i + 24]))
            with gzip.open(out, "at") as f:
                for row in rows:
                    if row:
                        f.write(json.dumps(row) + "\n")


def tokenizer_for(client, model: str):
    """The sampler's own tokenizer if it offers a fast one; otherwise the model's tokenizer from Hugging Face
    (Thinking Machines models do not expose get_tokenizer)."""
    try:
        tok = client.get_tokenizer()
        if getattr(tok, "is_fast", False):  # offsets need a fast tokenizer
            return tok
    except Exception:  # noqa: BLE001
        pass
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def tinker_docs(model: str, docs: list[Document], out: Path, workers: int = 8, max_tokens: int = 32000) -> None:
    import tinker

    client = tinker.ServiceClient().create_sampling_client(base_model=model)
    tok = tokenizer_for(client, model)
    done = _done(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def one(doc: Document) -> dict | None:
        text, labels = doc.render()
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = list(enc["input_ids"]), [a for a, _ in enc["offset_mapping"]]
        # The model gets its BOS token if it has one, but the first real token stays unscored, as on Fireworks.
        bos = getattr(tok, "bos_token_id", None)
        lead = [bos] if bos is not None else []
        stop = offs[max_tokens - len(lead)] if len(ids) > max_tokens - len(lead) else None
        ids, offs = ids[:max_tokens - len(lead)], offs[:max_tokens - len(lead)]
        for attempt in range(6):
            try:
                lps = client.compute_logprobs(tinker.types.ModelInput.from_ints(lead + ids)).result()[len(lead):]
                break
            except Exception as e:  # noqa: BLE001 - transient service errors; give up after retries
                if attempt == 5:
                    print(f"  {doc.id}: {str(e)[:120]}")
                    return None
        lps = _clean_logprobs(doc.id, lps)
        if lps is None:
            return None
        row = summarize(doc, text, labels, offs, lps, stop)
        row.update(offsets=offs, logprobs=[round(x, 4) for x in lps])
        return row

    todo = [d for d in docs if d.id not in done]
    with ThreadPoolExecutor(workers) as pool:
        for i in range(0, len(todo), 32):
            rows = list(pool.map(one, todo[i:i + 32]))
            with gzip.open(out, "at") as f:
                for row in rows:
                    if row:
                        f.write(json.dumps(row) + "\n")


def load(out: Path) -> list[dict]:
    return _rows(out)
