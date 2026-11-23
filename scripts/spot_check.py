"""Re-score a random 10% of documents for each Fireworks model and compare with the stored scores, to see whether
Fireworks also returns errors too small for the anomaly rule to catch."""

import asyncio
import random
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, "src")
from cbridge.corpus import read_jsonl  # noqa: E402
from cbridge.docscore import fireworks_docs, load  # noqa: E402

load_dotenv(".env")
BASE = Path("results/docs/truth/fresh")
MODELS = ["kimi-k3", "kimi-k3p1-pro", "glm-5p3", "glm-5p3-flash", "minimax-m3", "qwen3p8-max", "nemotron-3-ultra-nvfp4",
          "nemotron-lightning-3p5-30b-a3b"]
corpus = read_jsonl(Path("data/corpus/truth/fresh.jsonl"))
sample = random.Random(7).sample(corpus, 40)


async def main():
    async def one(m):
        tmp = Path(f"cache/spot/{m}.jsonl.gz")
        tmp.unlink(missing_ok=True)
        await fireworks_docs(m, sample, tmp)
        return m, tmp
    for m, tmp in await asyncio.gather(*(one(m) for m in MODELS)):
        old = {r["doc_id"]: r["bits"] / r["bytes"] for r in load(BASE / f"{m}.jsonl.gz")}
        new = {r["doc_id"]: r["bits"] / r["bytes"] for r in load(tmp)}
        d = np.array([new[i] - old[i] for i in new if i in old])
        print(f"{m:32s} n={len(d)}  mean diff {d.mean():+.4f}  |diff|>0.02: {(abs(d) > 0.02).sum()}  max |diff| {abs(d).max():.3f}")

asyncio.run(main())
