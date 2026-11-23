"""Per-document scores for each model, as committed in results/per_doc/<model>.csv.

The full per-token log-probabilities (results/docs/, not in git) are written by `cb docs` and scripts/modal_vllm.py;
`scripts/export_per_doc.py` turns them into these summaries, which are all the charts need.
"""

import csv
from pathlib import Path

PER_DOC = Path("results/per_doc")
NUMERIC = {"n_tokens": int, "bytes": int, "bytes_speech": int, "bits": float, "bits_speech": float}


def has(stem: str) -> bool:
    return (PER_DOC / f"{stem}.csv").exists()


def rows(stem: str) -> dict[str, dict]:
    """doc_id -> {source, kind, n_tokens, bits, bytes, bits_speech, bytes_speech} for one model."""
    with open(PER_DOC / f"{stem}.csv", newline="") as f:
        return {r["doc_id"]: {k: NUMERIC.get(k, str)(v) for k, v in r.items()} for r in csv.DictReader(f)}


def bpb(r: dict[str, dict], docs: list[str]) -> float:
    """Bits per byte over a set of documents: total bits over total bytes."""
    return sum(r[d]["bits"] for d in docs) / sum(r[d]["bytes"] for d in docs)
