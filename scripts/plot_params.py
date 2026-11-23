"""BPB (log2 perplexity per byte) against total parameters for every scored open model."""

import textwrap
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, "scripts")
from docs_meta import MODELS  # noqa: E402
from per_doc import has, rows  # noqa: E402

BG, INK, MUTED, GRID = "#FAF9F5", "#141413", "#6B6A64", "#E8E6DC"
LAB = {"Moonshot": "#2a78d6", "Alibaba": "#eb6834", "NVIDIA": "#1baf7a", "Zhipu": "#eda100", "MiniMax": "#e87ba4",
       "Google": "#008300", "Thinking Machines": "#6250d6", "Meta": "#e34948", "OpenAI": "#A8A69D"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})
SPONT = {"oral_questions", "urgent_question"}
OFF = 0.80

data = {}
for stem in MODELS:
    if has(stem) and len(r := rows(stem)) >= 380:
        data[stem] = r
docs = sorted(d for d in set.intersection(*(set(v) for v in data.values())) if next(iter(data.values()))[d]["kind"] in SPONT)
bpb = {s: sum(r[d]["bits"] for d in docs) / sum(r[d]["bytes"] for d in docs) for s, r in data.items()
       if MODELS[s][5] == "instruct" and MODELS[s][2] >= "2026-01-01"}
# Instruct models only (every lab releases them; closed models are only these), released in 2026.
shown = {s: v for s, v in bpb.items() if v <= OFF}
off = {s: v for s, v in bpb.items() if v > OFF}

plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(figsize=(14, 11), facecolor=BG)
fig.subplots_adjust(left=0.09, right=0.97, top=0.79, bottom=0.17)
ax.set_facecolor(BG)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_color(MUTED)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(length=0, labelsize=13)
ax.grid(color=GRID, lw=0.9)
ax.set_axisbelow(True)

# trend over instruct models: BPB linear in log10(total parameters)
xs = np.array([np.log10(MODELS[s][3]) for s in shown if MODELS[s][5] == "instruct"])
ys = np.array([shown[s] for s in shown if MODELS[s][5] == "instruct"])
slope, icpt = np.polyfit(xs, ys, 1)
r = np.corrcoef(xs, ys)[0, 1]
grid = np.linspace(xs.min() - 0.15, xs.max() + 0.15, 50)
ax.plot(10 ** grid, slope * grid + icpt, color=MUTED, lw=1.8, ls="--", zorder=1)
ax.text(4300, 0.765, f"Trend (dashed): {slope:+.3f} bits per byte\nfor every 10x more parameters, "
        f"r = {r:.2f}", ha="right", va="top", size=13, color=MUTED)

OFFSET = {  # label placement in points: (dx, dy); negative dx puts the label on the left
    "Qwen__Qwen3.5-4B": (13, 0), "Qwen__Qwen3.5-9B": (-13, 0),
    "nemotron-lightning-3p5-30b-a3b": (13, 6), "Qwen__Qwen3.8-27B": (-13, 8), "Qwen__Qwen3.6-35B-A3B": (13, 0),
    "thinkingmachines__Inkling-2-Mini": (-13, -6),
    "nvidia__NVIDIA-Nemotron-3-Super-120B-A12B-BF16": (-13, 7),
    "nvidia__NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16": (-13, -8), "minimax-m3": (-13, 9),
    "Qwen__Qwen3.5-397B-A17B": (-13, 10), "thinkingmachines__Inkling": (-13, 9), "glm-5p3": (12, 8),
    "glm-5p3-flash": (-13, -7), "nemotron-3-ultra-nvfp4": (12, -12), "moonshotai__Kimi-K2.6": (12, -11),
    "thinkingmachines__Inkling-2-Small": (12, -11), "qwen3p8-max": (-13, 13), "kimi-k3p1-pro": (-14, 12),
    "kimi-k3": (-14, -12), "muse-spark-open": (-13, -8), "meta-models__Muse-Glimmer-30B": (-13, -4)}
for s, v in shown.items():
    lab, name, date, total, active, kind = MODELS[s]
    dense = active is not None and abs(active - total) < 0.5
    c = LAB[lab]
    ax.scatter(total, v, s=190, marker="s" if dense else "o", color=c, edgecolor=BG, linewidth=1.8, zorder=3)
    dx, dy = OFFSET.get(s, (12, 0))
    ax.annotate(name, (total, v), xytext=(dx, dy), textcoords="offset points", size=12.5, color=c, weight="bold",
                zorder=4, ha="right" if dx < 0 else ("center" if dx == 0 else "left"), va="center",
                bbox=dict(boxstyle="square,pad=0.12", fc=BG, ec="none", alpha=0.9))

ax.set_xscale("log")
ticks = [3, 10, 30, 100, 300, 1000, 3000]
ax.set_xticks(ticks, ["3B", "10B", "30B", "100B", "300B", "1T", "3T"])
ax.minorticks_off()
ax.set_xlim(3, 4500)
ax.set_ylim(0.49, 0.77)
ax.set_xlabel("total parameters (log scale)", size=14)
ax.set_ylabel("bits per byte (lower is better)", size=14)

fig.text(0.03, 0.955, "Bigger open models predict fresh parliament speech better", family="DejaVu Serif", size=23,
         weight="bold")
fig.text(0.03, 0.915, f"Exact bits per byte over every token of {len(docs)} whole documents: spontaneous UK and Canadian\n"
         "parliamentary questions (not prepared speeches), Oct–Nov 2026, after every model's training cutoff.",
         size=13.5, color=MUTED, linespacing=1.4, va="top")
labs_present = [l for l in LAB if any(MODELS[s][0] == l for s in shown)]
handles = [Line2D([], [], marker="o", ls="", color=LAB[l], markersize=11, label=l) for l in labs_present]
handles += [Line2D([], [], marker="o", ls="", color=MUTED, markersize=11, label="mixture of experts"),
            Line2D([], [], marker="s", ls="", color=MUTED, markersize=11, label="dense")]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.855), ncol=5, frameon=False,
           fontsize=12.5, handletextpad=0.3, columnspacing=1.4)
notes = "; ".join(f"{MODELS[s][1]} {v:.2f}" for s, v in sorted(off.items(), key=lambda kv: kv[1]))
r_small = rows("thinkingmachines__Inkling-Small") if has("thinkingmachines__Inkling-Small") else {}
small_bpb = sum(r_small[d]["bits"] for d in docs) / sum(r_small[d]["bytes"] for d in docs) if r_small else float("nan")
fig.text(0.03, 0.02, textwrap.fill("Instruct models released in 2026. Left out, because they answer as a chat assistant or loop "
         f"instead of continuing plain text: {notes}. Also left out: Inkling 1 Small {small_bpb:.2f} (first generation, "
         "cause unknown). Not shown: base models, and models released before 2026 (Qwen 3 8B, gpt-oss-120b and 20b). "
         "Each point's 95% interval is about ±0.005. Total parameters from the Fireworks and Hugging Face model pages.", 150),
         size=9.5, color=MUTED, linespacing=1.45)
out = Path("figures/params_vs_bpb.png")
fig.savefig(out, dpi=170, facecolor=BG)
print(out, f"slope {slope:.4f} r {r:.3f}")
