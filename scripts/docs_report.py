"""Whole-document BPB for every scored model: table with document-bootstrap intervals, on the documents that
every included model scored. Writes results/docs_summary.json for the plots."""

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from docs_meta import CHECKS, MODELS  # noqa: E402
from per_doc import PER_DOC, has, rows as doc_rows  # noqa: E402

BASE = PER_DOC
TIERS = {"spontaneous": {"oral_questions", "urgent_question"}, "prepared": {"debate", "statement", "business", "petition"}}
sanity = {json.loads(x)["model"].replace("/", "__"): json.loads(x) for x in (BASE / "sanity.jsonl").open()} \
    if (BASE / "sanity.jsonl").exists() else {}

rows = {}
for stem in list(MODELS) + list(CHECKS):
    if has(stem):
        rows[stem] = doc_rows(stem)
n_docs = max(len(v) for v in rows.values())
included = {s: r for s, r in rows.items() if len(r) >= 0.97 * n_docs}  # skip runs that are missing documents
missing = sorted(set(rows) - set(included))
common = set.intersection(*(set(r) for r in included.values()))
out = {"n_docs": len(common), "missing_runs": missing, "tiers": {}}
rng = random.Random(0)
for tier, kinds in TIERS.items():
    docs = sorted(d for d in common if next(iter(included.values()))[d]["kind"] in kinds)
    samples = [rng.choices(range(len(docs)), k=len(docs)) for _ in range(1000)]
    res = {}
    for stem, r in included.items():
        for key in ("all", "speech"):
            b = np.array([r[d]["bits" if key == "all" else "bits_speech"] for d in docs])
            n = np.array([r[d]["bytes" if key == "all" else "bytes_speech"] for d in docs])
            boots = [b[s].sum() / n[s].sum() for s in samples]
            res.setdefault(stem, {})[key] = [float(b.sum() / n.sum()), *map(float, np.percentile(boots, [2.5, 97.5]))]
        res[stem]["per_doc"] = {d: r[d]["bits"] / r[d]["bytes"] for d in docs}
    out["tiers"][tier] = {"docs": len(docs), "models": res}
    print(f"\n## {tier}: {len(docs)} documents (all text; speech only)")
    for stem in sorted(res, key=lambda s: res[s]["all"][0]):
        meta = MODELS.get(stem, ("check", f"{stem} (check of {CHECKS.get(stem)})", "", None, None, ""))
        flag = "" if sanity.get(stem, {}).get("ok", True) else "  [plain-text check failed]"
        a, s = res[stem]["all"], res[stem]["speech"]
        print(f"{meta[1]:28s} {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]   speech {s[0]:.3f}{flag}")
print("\nruns left out (incomplete):", missing)
Path("results/docs_summary.json").write_text(json.dumps(out))
