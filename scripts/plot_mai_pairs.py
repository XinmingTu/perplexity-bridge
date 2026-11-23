"""Pairwise scatter of MAI's four held-out BPB sets (Figure 1 of their Nov 2026 pre-training post), with Pearson r.

Shows how closely a model's score on one domain tracks its score on another, across the six models MAI reports.
"""

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

BG, INK, MUTED, GRID = "#FAF9F5", "#141413", "#6B6A64", "#E8E6DC"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED})

bpb = json.load(open("data/anchors/mai_2026-11.json"))["bpb"]
SETS = {"code": "Code (MAI's codebase)", "qa": "QA", "stem": "STEM", "math": "Math"}
MODELS = {  # name in MAI's table: (label, colour, marker); * = post-trained checkpoint
    "MAI-Base-1": ("MAI-Base-1", "#D62F2F", "o"), "MAI-Base-2": ("MAI-Base-2", "#8E1B1B", "o"),
    "DeepSeek-V4-Pro-Base": ("DeepSeek-V4-Pro-Base", "#4D6BFE", "s"), "Kimi-K3": ("Kimi K3*", "#2a78d6", "D"),
    "Qwen3.8": ("Qwen 3.8*", "#eb6834", "D"), "MiMo-V2.6-Pro-RL": ("MiMo-V2.6-Pro-RL*", "#E8A400", "D")}

fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), facecolor=BG)
fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.1, wspace=0.32, hspace=0.42)
for ax, (a, b) in zip(axes.flat, itertools.combinations(SETS, 2), strict=True):
    x = np.array([bpb[a][m] for m in MODELS])
    y = np.array([bpb[b][m] for m in MODELS])
    r = np.corrcoef(x, y)[0, 1]
    k, c = np.polyfit(x, y, 1)
    pad = (x.max() - x.min()) * 0.12
    xs = np.linspace(x.min() - pad, x.max() + pad, 20)
    ax.set_facecolor(BG)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0, labelsize=11.5)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.plot(xs, k * xs + c, color=MUTED, lw=1.6, ls="--", zorder=1)
    for m, (label, colour, marker) in MODELS.items():
        ax.scatter(bpb[a][m], bpb[b][m], s=190, color=colour, marker=marker, edgecolor=BG, linewidth=1.6, zorder=3)
    ax.set_xlabel(SETS[a], size=13.5)
    ax.set_ylabel(SETS[b], size=13.5)
    ax.set_title(f"r = {r:.2f}", loc="left", size=19, weight="bold", color=INK)
    print(f"{a:5s} vs {b:5s}  r = {r:.3f}")

fig.text(0.03, 0.955, "A model's score on one domain tracks its score on another", family="DejaVu Serif", size=23,
         weight="bold")
fig.text(0.03, 0.912, "Bits per byte on Microsoft AI's four held-out sets, for the six models in their Figure 1 "
         "(each point is one model; lower is better).", size=13.5, color=MUTED)
handles = [Line2D([], [], marker=mk, ls="", color=col, markersize=12, label=lab) for lab, col, mk in MODELS.values()]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.885), ncol=6, frameon=False, fontsize=12.5,
           handletextpad=0.3, columnspacing=1.3)
fig.text(0.03, 0.02, "Source: microsoft.ai/news/frontier-pre-training-at-microsoft-ai (Figure 1). * post-trained "
         "checkpoint. Correlations computed here from MAI's published numbers; six models only.", size=10.5,
         color=MUTED)
out = Path("figures/mai_domain_correlation.png")
fig.savefig(out, dpi=170, facecolor=BG)
print(out)
