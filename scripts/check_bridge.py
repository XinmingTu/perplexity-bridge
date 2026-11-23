"""Out-of-sample check of the second bridge, and how much MAI's four sets agree with each other.

The second bridge maps our parliament BPB to MAI's code BPB with a line through Kimi K3 and Qwen 3.8. MiMo-V2.6-Pro-RL
is also in MAI's table but was not used to draw that line, so predicting its MAI score from our measurement tests the
bridge. Both sides are post-trained checkpoints, so no base/instruct correction is involved.
"""

import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
from per_doc import bpb, rows  # noqa: E402

mai = json.load(open("data/anchors/mai_2026-11.json"))
o1, o2 = mai["openai_logppl"]["MAI-Base-1"], mai["openai_logppl"]["MAI-Base-2"]
k3, qw, mimo = rows("kimi-k3"), rows("qwen3p8-max"), rows("XiaomiMiMo__MiMo-V2.6-Pro-RL")
docs = sorted(d for d in set(k3) & set(qw) & set(mimo) if k3[d]["kind"] in {"oral_questions", "urgent_question"})
x_k, x_q, x_m = bpb(k3, docs), bpb(qw, docs), bpb(mimo, docs)
print(f"{len(docs)} spontaneous documents. Our BPB: Kimi K3 {x_k:.4f}, Qwen 3.8 {x_q:.4f}, MiMo {x_m:.4f}")
print("MiMo through bridge 2, by MAI set (error converted to OpenAI log-perplexity through bridge 1):")
for name, v in mai["bpb"].items():
    b2 = (v["Qwen3.8"] - v["Kimi-K3"]) / (x_q - x_k)
    pred = v["Kimi-K3"] + b2 * (x_m - x_k)
    b1 = (o1 - o2) / (v["MAI-Base-1"] - v["MAI-Base-2"])
    print(f"  {name:5s} MAI {v['MiMo-V2.6-Pro-RL']:.4f}  predicted {pred:.4f}  "
          f"error {b1 * (pred - v['MiMo-V2.6-Pro-RL']):+.3f}")

print("Correlation between MAI's sets across their six models:")
sets = list(mai["bpb"])
for i, a in enumerate(sets):
    for b in sets[i + 1:]:
        models = list(mai["bpb"][a])
        r = np.corrcoef([mai["bpb"][a][m] for m in models], [mai["bpb"][b][m] for m in models])[0, 1]
        print(f"  {a} vs {b}: r = {r:.3f}")
