"""`cb`: build the corpus, then score every configured open model on every whole document.

Nothing that costs money runs without --execute.
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from dotenv import load_dotenv

from . import corpus as corpus_mod

DATA, RESULTS = Path("data"), Path("results")


def corpus_path(cfg, window):
    return DATA / "corpus" / cfg["name"] / f"{window}.jsonl"


def cmd_corpus(args, cfg):
    """Fetch transcripts for each window and keep a fixed, seeded sample per source and tier."""
    for window, span in cfg["windows"].items():
        if args.window and window != args.window:
            continue
        chosen = []
        for source in cfg["sources"]:
            docs = corpus_mod.SOURCES[source](span["start"], span["end"])
            by_kind = defaultdict(list)
            for d in docs:
                by_kind[d.meta["kind"]].append(d)
            rng = random.Random(f"{cfg['seed']}|{window}|{source}")
            picked = []
            for tier, kinds in cfg["tiers"].items():
                # Round-robin over kinds so no one kind of proceeding dominates a tier.
                pools = [rng.sample(by_kind[k], len(by_kind[k])) for k in kinds if by_kind.get(k)]
                quota, got = cfg["docs_per_source"][tier], []
                while len(got) < quota and any(pools):
                    for pool in pools:
                        if pool and len(got) < quota:
                            got.append(pool.pop())
                picked += got
            print(f"{window} {source}: fetched {len(docs)}, kept {len(picked)} "
                  f"({dict(sorted((k, sum(d.meta['kind'] == k for d in picked)) for k in by_kind))})")
            chosen += picked
        path = corpus_path(cfg, window)
        corpus_mod.write_jsonl(sorted(chosen, key=lambda d: d.id), path)
        print(f"  -> {path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()[:16]}")


def cmd_docs(args, cfg):
    """Whole-document exact BPB for every configured open model, plus the plain-text sanity check."""
    from .docscore import fireworks_docs, tinker_docs
    from .sanity import fireworks_check, tinker_check

    window = args.window or "fresh"
    docs = corpus_mod.read_jsonl(corpus_path(cfg, window))[: args.limit]
    base = RESULTS / "docs" / cfg["name"] / window
    plan = [(prov, m) for prov, ms in cfg["doc_models"].items() for m in ms if not args.model or m == args.model]
    tok = sum(len(d.render()[0]) for d in docs) // 4
    print(f"{len(plan)} models x {len(docs)} documents, ~{tok:,} tokens per model, ~{tok * len(plan):,} in all")
    if not args.execute:
        return
    sane = base / "sanity.jsonl"
    checked = {json.loads(x)["model"] for x in sane.open()} if sane.exists() else set()
    lock = threading.Lock()

    def run(prov, m):
        out = base / (m.replace("/", "__") + ".jsonl.gz")
        if m not in checked:
            try:  # the sanity check never blocks scoring
                v = (fireworks_check if prov == "fireworks" else tinker_check)(m, docs)
                with lock, sane.open("a") as f:
                    f.write(json.dumps(dict(model=m, provider=prov, **v)) + "\n")
                print(f"  {m}: sanity ok={v['ok']} repeated={v['repeated_4grams']} control={v['control_tokens']}",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  {m}: sanity check failed: {str(e)[:150]}", flush=True)
        try:
            if prov == "fireworks":
                asyncio.run(fireworks_docs(m, docs, out))
            else:
                tinker_docs(m, docs, out)
            print(f"  {m}: scored", flush=True)
        except Exception as e:  # noqa: BLE001 - one unavailable model must not stop the others
            print(f"  {m}: FAILED: {str(e)[:200]}", flush=True)

    base.mkdir(parents=True, exist_ok=True)
    # Every model runs at the same time; each model also sends its own requests concurrently.
    with ThreadPoolExecutor(max_workers=len(plan)) as pool:
        list(pool.map(lambda pm: run(*pm), plan))


def main():
    ap = argparse.ArgumentParser(prog="cb", description=__doc__)
    ap.add_argument("command", choices=["corpus", "docs"])
    ap.add_argument("config", nargs="?", default="configs/truth.yaml")
    ap.add_argument("--model")
    ap.add_argument("--window")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--execute", action="store_true", help="actually call models (otherwise: plan only)")
    ap.add_argument("--env-file", default=os.environ.get("CB_ENV_FILE", ".env"))
    args = ap.parse_args()
    load_dotenv(args.env_file)
    cfg = yaml.safe_load(Path(args.config).read_text())
    globals()[f"cmd_{args.command}"](args, cfg)
