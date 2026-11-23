"""Does an instruct model still work as a plain-text language model?

Greedy continuation of a few document openings. A model fails if its continuations loop (most 4-grams repeated)
or emit chat/agent control tokens. Such a model's BPB is reported as "not comparable", not ranked.
"""

import os
import re

import httpx

from .text import Document

CONTROL = re.compile(r"<tool_call>|<invoke|</?think>|<\|im_(start|end)\|>|<\|assistant\|>|<\|end\|>")


def repeated_4grams(text: str) -> float:
    w = text.split()
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def verdict(continuations: list[str]) -> dict:
    rep = sum(repeated_4grams(c) for c in continuations) / len(continuations)
    ctrl = sum(bool(CONTROL.search(c)) for c in continuations)
    return dict(repeated_4grams=round(rep, 3), control_tokens=ctrl, ok=rep < 0.7 and ctrl == 0,
                samples=[c[:200] for c in continuations])


def openings(docs: list[Document], n: int = 3, words: int = 300) -> list[str]:
    return [" ".join(d.render()[0].split(" ")[:words]) for d in docs[:n]]


def fireworks_check(model: str, docs: list[Document], max_tokens: int = 80) -> dict:
    m = model if "/" in model else f"accounts/fireworks/models/{model}"
    outs = []
    for prompt in openings(docs):
        r = httpx.post("https://api.fireworks.ai/inference/v1/completions", timeout=120,
                       headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"},
                       json={"model": m, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0})
        r.raise_for_status()
        outs.append(r.json()["choices"][0]["text"])
    return verdict(outs)


def tinker_check(model: str, docs: list[Document], max_tokens: int = 80) -> dict:
    import tinker

    client = tinker.ServiceClient().create_sampling_client(base_model=model)
    from .docscore import tokenizer_for

    tok = tokenizer_for(client, model)
    outs = []
    for prompt in openings(docs):
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        if getattr(tok, "bos_token_id", None) is not None:
            ids = [tok.bos_token_id] + ids
        res = client.sample(tinker.types.ModelInput.from_ints(ids), num_samples=1,
                            sampling_params=tinker.types.SamplingParams(max_tokens=max_tokens, temperature=0)).result()
        outs.append(tok.decode(res.sequences[0].tokens, skip_special_tokens=False))
    return verdict(outs)
