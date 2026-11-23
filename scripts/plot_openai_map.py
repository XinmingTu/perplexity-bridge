"""Place open instruct models on OpenAI's base-model log-perplexity axis through two bridges.

Step 1: MAI eval-set BPB -> OpenAI log-perplexity, through MAI-Base-1 (0.878) and MAI-Base-2 (0.819), the only models
        with scores on both (Microsoft AI, Nov 2026; same tokenizer as OpenAI).
Step 2: our parliament BPB -> MAI eval-set BPB, through Kimi K3 and Qwen3.8-2.4T, the only models in both tables
        (instruct checkpoints on both sides).
Each step is a straight line through two points, so it is exact for the bridge models and untested for the rest.
The main estimate uses MAI's held-out code set (closest to OpenAI's internal code); the bar spans the estimates
through MAI's four sets (code, QA, STEM, math).
"""

import datetime as dt
import json
import textwrap
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "scripts")
from docs_meta import MODELS  # noqa: E402
from per_doc import has, rows  # noqa: E402

BG, INK, MUTED, GRID = "#FAF9F5", "#141413", "#6B6A64", "#E8E6DC"
OPENAI, MAI = "#2B4C8C", "#D62F2F"
LAB = {"Moonshot": "#2a78d6", "Alibaba": "#eb6834", "NVIDIA": "#1baf7a", "Zhipu": "#eda100", "MiniMax": "#e87ba4",
       "Google": "#008300", "Thinking Machines": "#6250d6", "Meta": "#e34948"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})

mai = json.load(open("data/anchors/mai_2026-11.json"))
o1, o2 = mai["openai_logppl"]["MAI-Base-1"], mai["openai_logppl"]["MAI-Base-2"]

# our whole-document BPB, spontaneous questions, instruct models released in 2026 that work on plain text
data = {s: rows(s) for s in MODELS if has(s)}
data = {s: r for s, r in data.items() if len(r) >= 380}
docs = sorted(d for d in set.intersection(*(set(v) for v in data.values()))
              if next(iter(data.values()))[d]["kind"] in {"oral_questions", "urgent_question"})
all_bpb = {s: sum(r[d]["bits"] for d in docs) / sum(r[d]["bytes"] for d in docs) for s, r in data.items()}
# Post-training cost measured on our own base/instruct pairs of the same model (healthy instruct models only).
PAIRS = [("Qwen__Qwen3.5-9B-Base", "Qwen__Qwen3.5-9B"),
         ("nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16", "nemotron-lightning-3p5-30b-a3b")]
ratios = {i: all_bpb[i] / all_bpb[b] for b, i in PAIRS if b in all_bpb and i in all_bpb}
OUR_FACTOR = float(np.exp(np.mean(np.log(list(ratios.values())))))
# Post-training cost used for the chart: Thinking Machines' own pair, Inkling 2 Small post-trained 0.2080 vs pre-trained
# 0.2030 bits per byte on their held-out code (thinkingmachines.ai/news/inkling-2-small, Nov 2026): a frontier-scale
# model on code, the bridge's domain. Our two small-model pairs (OUR_FACTOR) are printed for comparison.
FACTOR = 0.2080 / 0.2030
print("post-training ratios:", {k: round(v, 4) for k, v in ratios.items()}, "ours", round(OUR_FACTOR, 4), "used", round(FACTOR, 4))
ours = {s: v for s, v in all_bpb.items() if MODELS[s][5] == "instruct" and MODELS[s][2] >= "2026-01-01" and v <= 0.8}


def maps(x, base_equivalent=False):
    """OpenAI log-perplexity for our BPB x, through each MAI eval set.

    With base_equivalent, every instruct score (ours and MAI's Kimi K3 / Qwen3.8) is divided by FACTOR, the
    post-training cost measured on our base/instruct pairs; the result is where the model's base would sit.
    """
    out = {}
    for name, v in mai["bpb"].items():
        b1 = (o1 - o2) / (v["MAI-Base-1"] - v["MAI-Base-2"])
        a1 = o2 - b1 * v["MAI-Base-2"]
        b2 = (v["Qwen3.8"] - v["Kimi-K3"]) / (ours["qwen3p8-max"] - ours["kimi-k3"])
        a2 = v["Kimi-K3"] - b2 * ours["kimi-k3"]
        out[name] = a1 + b1 * (a2 + b2 * x) / (FACTOR if base_equivalent else 1.0)
    return out


raw = {s: maps(v) for s, v in ours.items()}
proj = {s: maps(v, base_equivalent=True) for s, v in ours.items()}
json.dump({MODELS[s][1]: {"our_bpb": round(ours[s], 4), "raw_instruct": {k: round(x, 4) for k, x in raw[s].items()},
                          "base_equivalent": {k: round(x, 4) for k, x in proj[s].items()}, "factor": round(FACTOR, 4)}
           for s in proj},
          open("results/openai_axis_projection.json", "w"), indent=1)

# Uncertainty (1 sd, OpenAI units), from the independent review's error budget: which MAI set stands in for OpenAI's
# code (0.0125), the post-training correction (0.015), reading MAI's OpenAI-set numbers (0.005), plus the
# speech-to-code bridge, which grows with distance from the bridge models (d: half-spans from their midpoint).
mid, half = (ours["kimi-k3"] + ours["qwen3p8-max"]) / 2, (ours["qwen3p8-max"] - ours["kimi-k3"]) / 2
dist = {s: (v - mid) / half for s, v in ours.items()}
sd = {s: float(np.sqrt(0.0125**2 + 0.015**2 + 0.005**2 + (0.015 * max(abs(dist[s]), 0.5)) ** 2)) for s in ours}
EXTRAP = 1.5  # beyond this many half-spans a model is extrapolated
shown = {s: p for s, p in proj.items() if abs(dist[s]) <= EXTRAP}  # far-out models are not plotted
hidden = sorted((s for s in proj if s not in shown), key=lambda s: ours[s])

# MAI's own open-model entries, placed through bridge 1 only (no second bridge needed)
cv = mai["bpb"]["code"]
b1c = (o1 - o2) / (cv["MAI-Base-1"] - cv["MAI-Base-2"])
a1c = o2 - b1c * cv["MAI-Base-2"]
REF = [("DeepSeek-V4-Pro-Base", dt.date(2026, 4, 22), a1c + b1c * cv["DeepSeek-V4-Pro-Base"]),
       ("MiMo-V2.6-Pro-RL", dt.date(2026, 9, 21), a1c + b1c * cv["MiMo-V2.6-Pro-RL"] / FACTOR)]

# --gpt35 draws the chart from GPT-3.5 (March 2022) into a separate file
WITH_GPT35 = "--gpt35" in sys.argv
FIRST_YEAR = "2022" if WITH_GPT35 else "2023"
rows = [r.split(",") for r in Path("data/anchors/openai_2026-11-22.csv").read_text().splitlines()[1:]]
pts = [(dt.datetime.strptime(m, "%B %Y").date().replace(day=15), float(v), lab) for lab, m, v, _ in rows if lab != "Internal model" and m[-4:] >= FIRST_YEAR]

T0, T2 = (dt.date(2022, 1, 20) if WITH_GPT35 else dt.date(2023, 1, 20)).toordinal(), dt.date(2027, 1, 10).toordinal()


def X(d):
    return (d.toordinal() - T0) / (T2 - T0)



# --vertical: a 3:4 portrait version (for phone feeds) with labels in two columns right beside the 2026 cluster.
if "--vertical" in sys.argv:
    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(figsize=(12, 16), facecolor=BG)
    fig.subplots_adjust(left=0.1, right=0.985, top=0.835, bottom=0.1)
    ax.set_facecolor(BG)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(length=0, labelsize=15)
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1.52)
    ax.set_ylim(0.745, 1.0 if WITH_GPT35 else 0.925)
    ax.spines["bottom"].set_bounds(0, 1.0)
    ax.plot([X(p[0]) for p in pts], [p[1] for p in pts], color=OPENAI, lw=3, marker="o", ms=11, zorder=3,
            markeredgecolor=BG, markeredgewidth=1.6)
    MAI_PTS = [("MAI-Base-1", dt.date(2026, 4, 15), o1), ("MAI-Base-2", dt.date(2026, 9, 15), o2)]
    for name, d, v in MAI_PTS:
        ax.scatter(X(d), v, s=200, color=MAI, zorder=6, edgecolor=BG, linewidth=1.8)
    for name, d, v in REF:
        ax.scatter(X(d), v, s=150, marker="D", color=BG, edgecolor=INK, linewidth=2.2, zorder=6)
    for s_, p in shown.items():
        lab, date, v, e = MODELS[s_][0], dt.date.fromisoformat(MODELS[s_][2]), p["code"], sd[s_]
        ax.plot([X(date)] * 2, [v - e, v + e], color=LAB[lab], alpha=0.28, lw=4, solid_capstyle="round")
        ax.scatter(X(date), v, s=150, color=LAB[lab], edgecolor=BG, linewidth=1.6, zorder=5)
    # milestones
    lo, hi = ax.get_ylim()
    for lab_, _, val, milestone in rows:
        if milestone and lo < float(val) < hi:
            right = milestone.startswith(("Fluent", "Benefits", "Solved"))
            text = "Benefits from thinking\nbefore answering" if milestone.startswith("Benefits") else milestone
            text = "Solved a Millennium\nPrize math problem" if milestone.startswith("Solved") else text
            ax.plot([0, 1.0], [float(val)] * 2, color="#9DB3D9", lw=1.4, ls=(0, (2, 2.5)), zorder=0)
            x_at = 0.9 if milestone.startswith("Solved") else (0.74 if right else 0.01)
            ax.text(x_at, float(val) + 0.0013, text, color="#4F6799", size=14, style="italic",
                    ha="right" if right else "left", va="bottom", zorder=1, linespacing=1.1)
    # OpenAI labels up to 2025 beside their points
    for d, v, lab in pts:
        if d < dt.date(2026, 1, 1):
            dx, dy, ha = {"GPT-3.5": (14, 4, "left"), "GPT-4": (0, -20, "center"), "GPT-4o / o1": (-8, -20, "right"),
                          "GPT-4.5": (-14, 10, "right"), "GPT-5 / 5.1": (6, 18, "left"),
                          "GPT-5.2": (14, 0, "left")}.get(lab, (0, 18, "center"))
            ax.annotate(lab, (X(d), v), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", size=15,
                        color=INK, weight="bold", zorder=7,
                        bbox=dict(boxstyle="square,pad=0.08", fc=BG, ec="none", alpha=0.85))
    # 2026 labels: a short column on each side of the cluster, in the points' vertical order
    left_keys = {"GPT-5.3-Codex / 5.4", "GPT-5.5", "Kimi K2.6", "Nemotron 3 Ultra"}
    items = [(X(d), v, lab, INK) for d, v, lab in pts if d >= dt.date(2026, 1, 1)]
    items += [(X(d), v, n, MAI) for n, d, v in MAI_PTS] + [(X(d), v, n + "*", INK) for n, d, v in REF]
    items += [(X(dt.date.fromisoformat(MODELS[s_][2])), p["code"], MODELS[s_][1], LAB[MODELS[s_][0]])
              for s_, p in shown.items()]
    GAP = 0.0068 if WITH_GPT35 else 0.0052
    for side, col in (("left", X(dt.date(2025, 12, 1))), ("right", X(dt.date(2027, 1, 20)))):
        group = sorted((it for it in items if (it[2] in left_keys) == (side == "left")), key=lambda it: -it[1])
        ys = [it[1] for it in group]
        for _ in range(400):
            for i in range(1, len(ys)):
                if ys[i - 1] - ys[i] < GAP:
                    m_ = (ys[i - 1] + ys[i]) / 2
                    ys[i - 1], ys[i] = m_ + GAP / 2, m_ - GAP / 2
        for (x, v, text, color), y in zip(group, ys):
            end = col + (0.008 if side == "left" else -0.008)
            ax.plot([x, end], [v, y], color=color, lw=1, alpha=0.5, zorder=2)
            ax.text(col, y, text, color=color, weight="bold", size=15, va="center",
                    ha="right" if side == "left" else "left", zorder=7,
                    bbox=dict(boxstyle="square,pad=0.1", fc=BG, ec="none", alpha=0.9))
    ticks = [dt.date(y, 1, 1) for y in range(2023 if WITH_GPT35 else 2024, 2028)]
    ax.set_xticks([X(t) for t in ticks], [str(t.year) for t in ticks])
    ax.set_ylabel("log-perplexity on OpenAI's code set (lower is better)", size=16)
    handles = [plt.Line2D([], [], color=OPENAI, marker="o", lw=3, markersize=10, label="OpenAI"),
               plt.Line2D([], [], color=MAI, marker="o", ls="", markersize=12, label="Microsoft AI"),
               plt.Line2D([], [], color=BG, marker="D", ls="", markeredgecolor=INK, markeredgewidth=2.2,
                          markersize=10, label="* open model, MAI's own number")]
    handles += [plt.Line2D([], [], color=c, marker="o", ls="", markersize=11, label=f"{l}")
                for l, c in LAB.items() if any(MODELS[s_][0] == l for s_ in shown)]
    leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=True, fontsize=14,
                    facecolor=BG, edgecolor="none", framealpha=1)
    leg.set_zorder(8)
    fig.text(0.04, 0.965, "Open models on OpenAI's\nbase-model log-perplexity chart", family="DejaVu Serif", size=30,
             weight="bold", va="top", linespacing=1.15)
    fig.text(0.04, 0.895, "Bridged through Microsoft AI's base models and fresh, off-the-cuff\n"
             "UK and Canadian parliament questions (Oct–Nov 2026). Bars: ±1 sd.", size=15.5, color=MUTED, va="top",
             linespacing=1.4)
    fig.text(0.04, 0.02, textwrap.fill("Rough: the second bridge is a line through two models (Kimi K3, Qwen 3.8), about "
             "±0.025. Open models are instruct versions, nudged to their base-model equivalent using the base vs instruct "
             "gap Thinking Machines reported for Inkling 2 Small. "
             "Models far from the bridge are not shown. Dates approximate.", 120), size=12, color=MUTED, linespacing=1.4)
    out = Path("figures/openai_axis_projection_vertical" + ("_with_gpt35" if WITH_GPT35 else "") + ".png")
    fig.savefig(out, dpi=170, facecolor=BG)
    print(out)
    sys.exit(0)


plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(figsize=(14, 12) if WITH_GPT35 else (14, 11), facecolor=BG)
fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.17)
ax.set_facecolor(BG)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_color(MUTED)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(length=0, labelsize=13)
ax.grid(axis="y", color=GRID, lw=0.9)
ax.set_axisbelow(True)
ax.plot([X(p[0]) for p in pts], [p[1] for p in pts], color=OPENAI, lw=2.6, marker="o", ms=10, zorder=3,
        markeredgecolor=BG, markeredgewidth=1.5)
MAI_PTS = [("MAI-Base-1", dt.date(2026, 4, 15), o1), ("MAI-Base-2", dt.date(2026, 9, 15), o2)]
for name, d, v in MAI_PTS:
    ax.scatter(X(d), v, s=170, color=MAI, zorder=6, edgecolor=BG, linewidth=1.8)
for name, d, v in REF:
    ax.scatter(X(d), v, s=130, marker="D", color=BG, edgecolor=INK, linewidth=2, zorder=6)
for s, p in shown.items():
    lab, date, v, e = MODELS[s][0], dt.date.fromisoformat(MODELS[s][2]), p["code"], sd[s]
    ax.plot([X(date)] * 2, [v - e, v + e], color=LAB[lab], alpha=0.3, lw=3.5, solid_capstyle="round")
    ax.scatter(X(date), v, s=130, color=LAB[lab], edgecolor=BG, linewidth=1.6, zorder=5)

# Early OpenAI points: labels beside the points.
COLUMN_FROM = dt.date(2025, 7, 1)
BELOW = ("GPT-4", "GPT-4o / o1", "GPT-4.5") if WITH_GPT35 else ("GPT-4", "GPT-4 Turbo", "GPT-4.5")
for d, v, lab in pts:
    if d < COLUMN_FROM:
        dy = -17 if lab in BELOW else 15
        dx = -30 if lab == "GPT-4o / o1" and dy < 0 else 0  # the line falls to the right of GPT-4o
        if lab == "GPT-4.5":  # left of the point: clear of the lines on both sides and of the milestone line below
            dx, dy = -12, 0
        ax.annotate(lab, (X(d), v), xytext=(dx, dy), textcoords="offset points", ha="right" if lab == "GPT-4.5" else "center", va="center", size=13,
                    color=INK, weight="bold")

# Everything from mid-2025 on: a label column right of the data, in the points' vertical order, with leader lines.
items = [(X(d), v, lab, INK) for d, v, lab in pts if d >= COLUMN_FROM]
items += [(X(d), v, n, MAI) for n, d, v in MAI_PTS] + [(X(d), v, n + "*", INK) for n, d, v in REF]
items += [(X(dt.date.fromisoformat(MODELS[s][2])), p["code"], MODELS[s][1], LAB[MODELS[s][0]]) for s, p in shown.items()]
items.sort(key=lambda it: -it[1])
GAP, COL = (0.0088 if WITH_GPT35 else 0.0071), 1.04
ys = [it[1] for it in items]
for _ in range(400):  # spread the label heights to at least GAP apart, staying close to the points
    for i in range(1, len(ys)):
        if ys[i - 1] - ys[i] < GAP:
            mid_ = (ys[i - 1] + ys[i]) / 2
            ys[i - 1], ys[i] = mid_ + GAP / 2, mid_ - GAP / 2
for (x, v, text, color), y in zip(items, ys):
    ax.plot([x + 0.008, COL - 0.015, COL - 0.005], [v, y, y], color=color, lw=0.9, alpha=0.5, zorder=2)
    ax.text(COL, y, text, color=color, weight="bold", size=12.5, va="center", ha="left")

ticks = [dt.date(y, 1, 1) for y in range(2023 if WITH_GPT35 else 2024, 2028)]
ax.set_xticks([X(t) for t in ticks], [str(t.year) for t in ticks])
ax.set_xlim(0, 1.36)
ax.spines["bottom"].set_bounds(0, 1.0)
ax.set_ylim(0.74, 1.0 if WITH_GPT35 else 0.925)
ax.set_ylabel("log-perplexity on OpenAI's code set (lower is better)", size=14)
# OpenAI's capability milestones (from their chart): a dotted line at the level of the model that reached each one.
MILESTONE_AT = {"Useful coder": (0.01, "left"), "Cybersecurity expert": (0.01, "left")}  # others end at x=0.74
lo, hi = ax.get_ylim()
for lab_, _, val, milestone in rows:
    if milestone and lo < float(val) < hi:
        x0, ha = MILESTONE_AT.get(milestone, (0.74, "right"))
        ax.plot([0, 1.0], [float(val)] * 2, color="#9DB3D9", lw=1.3, ls=(0, (2, 2.5)), zorder=0)
        if WITH_GPT35 and milestone.startswith("Benefits"):
            milestone = "Benefits from thinking\nbefore answering"
        ax.text(x0, float(val) + 0.0012, milestone, linespacing=1.15, color="#4F6799", size=12.5, style="italic", ha=ha, va="bottom",
                zorder=1)

handles = [plt.Line2D([], [], color=OPENAI, marker="o", lw=2.6, markersize=9, label="OpenAI (published)"),
           plt.Line2D([], [], color=MAI, marker="o", ls="", markersize=11, label="Microsoft AI (published)"),
           plt.Line2D([], [], color=BG, marker="D", ls="", markeredgecolor=INK, markeredgewidth=2, markersize=9,
                      label="* open model in MAI's table")]
handles += [plt.Line2D([], [], color=c, marker="o", ls="", markersize=10, label=f"{l} (projected)")
            for l, c in LAB.items() if any(MODELS[s][0] == l for s in shown)]
leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=True, fontsize=12, facecolor=BG,
                edgecolor="none", framealpha=1)
leg.set_zorder(8)

fig.text(0.03, 0.955, "Open models on OpenAI's base-model log-perplexity axis", family="DejaVu Serif", size=23,
         weight="bold")
fig.text(0.03, 0.915, "Projected through two bridges: MAI-Base-1/2, and Kimi K3 / Qwen 3.8 scored on spontaneous\n"
         "parliamentary questions (not prepared speeches), Oct–Nov 2026. Bars: ±1 sd.", va="top", size=13.5,
         color=MUTED, linespacing=1.4)
fig.text(0.03, 0.02, textwrap.fill("Read with care: the second bridge is a line through two points; typical error near "
         "the bridge models is ±0.025 (1 sd). Open models are instruct checkpoints shown as base-equivalent "
         f"(BPB ÷ {FACTOR:.3f}: Inkling 2 Small's post-trained vs pre-trained BPB, from Thinking Machines; our own two "
         f"pairs give {OUR_FACTOR:.3f}). "
         "Not shown (too far from the bridge models): " + ", ".join(MODELS[x][1] for x in hidden)
         + (". Dates approximate." if WITH_GPT35 else ". GPT-3.5 (2022) not shown. Dates approximate."), 150), size=9.5, color=MUTED, linespacing=1.45)
out = Path("figures/openai_axis_projection" + ("_with_gpt35" if WITH_GPT35 else "") + ".png")
fig.savefig(out, dpi=170, facecolor=BG)
for s in sorted(proj, key=lambda s: proj[s]["code"]):
    p = proj[s]
    print(f"{MODELS[s][1]:24s} ours {ours[s]:.3f} -> OpenAI {p['code']:.3f}  [{min(p.values()):.3f}, {max(p.values()):.3f}]")
