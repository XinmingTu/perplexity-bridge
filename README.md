# Open models on OpenAI's perplexity chart

OpenAI's [Perplexity and Intelligence](https://openai.com/index/perplexity-and-intelligence/) charts the base-model
log-perplexity of its own models, from GPT-3.5 to GPT-6.2 Astra. This repository puts 2026's open models on the same
chart.

The open models are tested on spontaneous questions asked by real people in the UK and Canadian parliaments in
October and November 2026. Off-the-cuff speech is not AI-written, and all of it comes after every model's training
cutoff, so no model can have seen it.

OpenAI's test set is private, so open models get onto its chart through two bridges:

1. **Parliament questions.** Every open model is scored on these questions. Kimi K3 and Qwen 3.8 also appear in
   Microsoft AI's published table, which links this scale to Microsoft AI's.
2. **Microsoft AI.** MAI-Base-1 and MAI-Base-2 have scores on both Microsoft AI's test set and OpenAI's, which links
   Microsoft AI's scale to OpenAI's.

Each model reads every document in full and is scored on every token, in bits per byte, so models with different
tokenizers compare fairly.

![Open models on OpenAI's base-model log-perplexity chart](figures/openai_axis_projection_with_gpt35.png)

On the parliament questions alone, before any bridging, bigger open models do better:

![Bits per byte against total parameters](figures/params_vs_bpb.png)

## Read with care

- The second bridge rests on two models, so placements are rough: about ±0.025 on OpenAI's chart near those models.
  A model the bridge never saw (MiMo-V2.6-Pro) lands within that.
- Most open models only come as instruct versions, which predict plain text a little worse than base models. Scores
  are adjusted by the base-vs-instruct gap Thinking Machines reported for Inkling 2 Small.

## What's here

The charts, the per-document scores behind them, every token's log-probability for every model, the 391 transcripts,
and the code that scored them. The full method is in [docs/method.md](docs/method.md). To reproduce the charts or dig
into the data, let your coding agent explore the repository.

## Sources

- OpenAI, [Perplexity and Intelligence](https://openai.com/index/perplexity-and-intelligence/)
- Microsoft AI, [Frontier Pre-training at Microsoft AI](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/),
  and their November 2026 chart of MAI-Base-1/2 on OpenAI's test set
- Thinking Machines, [Introducing Inkling 2 Small](https://thinkingmachines.ai/news/inkling-2-small/)
- UK House of Commons Hansard (Open Parliament Licence v3.0); House of Commons of Canada debates via
  [openparliament.ca](https://openparliament.ca)
