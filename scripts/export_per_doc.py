"""Write results/per_doc/<model>.csv (one row per document) from the full per-token results in results/docs/."""

import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "src")
from cbridge.docscore import load  # noqa: E402

SRC, DST = Path("results/docs/truth/fresh"), Path("results/per_doc")
COLS = ["doc_id", "source", "kind", "n_tokens", "bits", "bytes", "bits_speech", "bytes_speech"]

DST.mkdir(parents=True, exist_ok=True)
for path in sorted(SRC.glob("*.jsonl.gz")):
    if path.name.endswith(".orig.gz"):  # pre-rescore backups (scripts/rescore_anomalies.py)
        continue
    with open(DST / (path.name.removesuffix(".jsonl.gz") + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in sorted(load(path), key=lambda r: r["doc_id"]):
            w.writerow([f"{r[c]:.4f}" if isinstance(r[c], float) else r[c] for c in COLS])
shutil.copy(SRC / "sanity.jsonl", DST / "sanity.jsonl")
print(f"wrote {len(list(DST.glob('*.csv')))} models to {DST}")
