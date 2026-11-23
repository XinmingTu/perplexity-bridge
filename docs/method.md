# Method

## The score

Every model reads each document whole, and the score is the sum of `-log2 p(token | everything before it)` over all
tokens, divided by the document's UTF-8 bytes: bits per byte (BPB). This is ordinary perplexity, normalised by bytes
instead of tokens so that different tokenizers compare fairly. The first token of a document has no context and is
not scored, on every backend. Speaker labels ("Mr Speaker:") are scored too; `bits_speech` and `bytes_speech` in
`results/per_doc/` leave them out, and the ranking is the same either way.

Backends, all giving the log-probability of every prompt token in one pass:

- Fireworks completions with `echo` and `logprobs`;
- Tinker `compute_logprobs`;
- vLLM 0.33 on Modal with `prompt_logprobs` (`scripts/modal_vllm.py`), for models no hosted API scores. Tokenization
  is done locally with the model's own tokenizer, BOS first when the model has one.

Documents longer than 32,000 tokens are cut there, and only the bytes before the cut count.

## The text

UK and Canadian House of Commons transcripts from Oct 1 to Nov 20, 2026, English-original paragraphs only, sampled
with a fixed seed (`configs/truth.yaml`): 311 spontaneous questions (oral questions and urgent questions) and 80
prepared items (debates, statements, business, petitions). The charts use the spontaneous tier. Off-the-cuff
questions are unlikely to be AI-written, and the whole window is after every scored model's training cutoff. Across
models, BPB on spontaneous and prepared speech agree at r ≈ 0.97–0.98.

## Which models

Open instruct models released in 2026 that continue plain text. A short greedy continuation of each document checks
this: a model fails if most of its 4-grams repeat or it emits chat or tool-call tokens. Gemma 4 instruct, gpt-oss and
MiniMax M3.1 Flash answer as a chat assistant or loop instead, and score 1.5–2.8 BPB, so they are left out. Inkling 1
Small scores far worse than its siblings while continuing text normally, cause unknown; it is left out too. Base
models are scored but not charted, since most labs release only instruct models.

## Data quality

Fireworks occasionally returns corrupted echo log-probabilities for a document. A document is flagged when its BPB,
relative to that model's typical level, departs from the median across models by more than a factor of 1.6;
`scripts/rescore_anomalies.py` re-requests each flagged document three times and keeps the median. Re-scoring a
random 10% of documents agrees within ±0.0006 BPB (`scripts/spot_check.py`); Tinker and vLLM are deterministic.
A model's BPB over the 311 documents has a 95% interval of about ±0.005 (document bootstrap,
`scripts/docs_report.py`); differences between two models on the same documents are tighter, about ±0.0015.

## The two bridges

1. **MAI code BPB → OpenAI log-perplexity**, a line through MAI-Base-1 (0.2401 → 0.878) and MAI-Base-2
   (0.2203 → 0.819). MAI's code set (its own codebase) is the closest of its four sets to OpenAI's internal code.
2. **Our BPB → MAI code BPB**, a line through Kimi K3 (0.509 → 0.2264) and Qwen 3.8 (0.565 → 0.2400).

Why text can stand in for code: across the six models in MAI's table, BPB on QA text and on code correlate at
r = 0.985 (`figures/mai_domain_correlation.png`). Math agrees least.

MAI's own open-model entries (DeepSeek-V4-Pro-Base, MiMo-V2.6-Pro-RL) are placed through bridge 1 alone.

## Base versus instruct

OpenAI and MAI chart base models; most open models are instruct checkpoints, which predict plain text a little worse.
Every instruct score (ours, and MAI's for Kimi K3 and Qwen 3.8) is divided by 1.025, the gap Thinking Machines
reported for Inkling 2 Small (0.2080 post-trained against 0.2030 pre-trained BPB on their held-out code). Our own
pairs give 2.7% (Qwen 3.5 9B) and 7.0% (Nemotron 3.5 Lightning 30B). Without the correction every open model sits
about 0.015 higher on OpenAI's axis; the ranking among open models does not change.

## Error

One standard deviation on OpenAI's axis combines: which MAI set stands in for OpenAI's (0.0125), the base/instruct
correction (0.015), reading MAI's published values (0.005), and the speech-to-code bridge, 0.015 per half-span of
distance from the Kimi–Qwen midpoint (at least half a half-span). Models more than 1.5 half-spans away are
extrapolated and not charted. Inside MAI's own table, a two-point Kimi–Qwen line between sets misses the other models
by RMS 0.018.

Out-of-sample check (`scripts/check_bridge.py`): MiMo-V2.6-Pro-RL is in MAI's table but not in bridge 2. Predicting
its MAI code BPB from our measurement misses by 0.023 on OpenAI's axis, within the stated error.

## Not attempted

Closed models give no token probabilities, so they are not scored. Inverting the size trend to estimate closed models'
parameter counts was tried and rejected: on MAI's models of known size it was off by 2–5×.

## Files

- `results/per_doc/<model>.csv`: bits and bytes for every document, all the charts need.
- `results/docs/truth/fresh/<model>.jsonl.gz`: one line per document, with `doc_id`, `source` and `kind` (matching
  `data/corpus/truth/fresh.jsonl`); `offsets`, each token's start as a character index into the document's rendered
  text (`Document.render()`); `logprobs`, each token's natural-log probability given everything before it (the first
  token is never scored and is 0); and the totals `bits`, `bytes`, `bits_speech`, `bytes_speech`. Tokens are each
  model's own, so offsets differ between models. If a document appears more than once, the last line is the one used.
- `scripts/`: charts (`plot_*.py`), the bridge check, the per-model report, Modal scoring, re-scoring, and
  `export_per_doc.py`, which rebuilds `results/per_doc/` from the per-token files.
