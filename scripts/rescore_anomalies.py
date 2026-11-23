"""Find (model, document) scores far out of line with other models, re-request them three times, and keep the
median of the retries. Fireworks occasionally returns corrupted echo log-probabilities (BPB of 4 or 0 for a
document that other runs score at 0.5); retries of the same request usually come back normal.

A score is flagged when, after removing the model's typical level, it departs from the per-document median
across models by more than a factor of 1.6. The rule is the same for every model and does not look at which
model is better.
"""

import asyncio
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from cbridge.corpus import read_jsonl  # noqa: E402
from cbridge.docscore import fireworks_docs, load  # noqa: E402
from docs_meta import CHECKS, MODELS  # noqa: E402

load_dotenv(".env")
BASE = Path("results/docs/truth/fresh")
FIREWORKS = {"kimi-k3", "kimi-k3p1-pro", "ember-1", "glm-5p3", "glm-5p3-flash", "minimax-m3", "minimax-m3p1-flash",
             "qwen3p8-max", "nemotron-3-ultra-nvfp4", "nemotron-lightning-3p5-30b-a3b", "muse-spark-open"}


def flags() -> dict[str, list[str]]:
    bpb = {}
    for stem in list(MODELS) + list(CHECKS):
        p = BASE / f"{stem}.jsonl.gz"
        if p.exists():
            rows = load(p)
            if len(rows) >= 380:
                bpb[stem] = {r["doc_id"]: r["bits"] / r["bytes"] for r in rows}
    docs = sorted(set.intersection(*(set(v) for v in bpb.values())))
    m = np.array([[bpb[s][d] for d in docs] for s in bpb])
    rel = m / np.median(m, 0)
    rel = rel / np.median(rel, 1, keepdims=True)
    bad = (rel < 0.6) | (rel > 1.6)
    return {s: [docs[j] for j in np.where(bad[i])[0]] for i, s in enumerate(bpb) if bad[i].any()}


def main() -> None:
    found = flags()
    print({s: len(v) for s, v in found.items()})
    corpus = {d.id: d for d in read_jsonl(Path("data/corpus/truth/fresh.jsonl"))}
    for stem, ids in found.items():
        if stem not in FIREWORKS:
            print(f"{stem}: {len(ids)} flagged, not a Fireworks run; left as is for inspection")
            continue
        retries = []
        for k in range(3):
            tmp = Path(f"cache/rescore/{stem}.{k}.jsonl.gz")
            tmp.unlink(missing_ok=True)
            asyncio.run(fireworks_docs(stem, [corpus[i] for i in ids], tmp))
            retries.append({r["doc_id"]: r for r in load(tmp)})
        main_path = BASE / f"{stem}.jsonl.gz"
        rows = {r["doc_id"]: r for r in load(main_path)}
        changed = []
        for i in ids:
            got = sorted((r[i] for r in retries if i in r), key=lambda r: r["bits"] / r["bytes"])
            if got:
                new = got[len(got) // 2]
                changed.append((i, rows[i]["bits"] / rows[i]["bytes"], [round(g["bits"] / g["bytes"], 3) for g in got]))
                rows[i] = new
        shutil.copy(main_path, main_path.with_suffix(".orig.gz"))
        with gzip.open(main_path, "wt") as f:
            for r in rows.values():
                f.write(json.dumps(r) + "\n")
        for i, old, vals in changed:
            print(f"  {stem} {i[:16]}: {old:.3f} -> retries {vals}")
    print("still flagged after retries:", {s: len(v) for s, v in flags().items()})


if __name__ == "__main__":
    main()
