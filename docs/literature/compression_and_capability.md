# Compression / perplexity as a capability measure, and comparing it across models

Literature survey, 2026-11-22. Context: a study placing open and closed LLMs on one
bits-per-byte (BPB) axis on fresh human text, motivated by OpenAI's "Perplexity and
Intelligence" (2026-11-22).

Verification legend:
- **[V]** I fetched the abstract, page or repo, or saw the arXiv listing, and the claim matches it.
- **[S]** The claim comes from search-result snippets or secondary coverage, not the primary text.
- **[M]** The claim is from memory; it is a well-known paper and I did not re-check it this session.
- **[U]** Unverified or conflicting. Do not rely on it without checking.

---

## 0. The motivating post and how it was received

**Perplexity and Intelligence.** Jakub Pachocki (OpenAI chief scientist), 2026-11-22.
https://openai.com/index/perplexity-and-intelligence/ **[V]**

- **What it reports.** Base-model log-perplexity (cross-entropy) on a fixed snapshot of a held-out
  dataset drawn from OpenAI's internal codebase, the dataset OpenAI has used since GPT-3.5. Values
  named in the text: GPT-3.5 ≈ 0.99, GPT-4 ≈ 0.91, "below 0.8" for emerging capabilities in cyber and
  math, ≈ 0.76 for the model that produced open-math results, and a forecast window of 0.68–0.70.
- **What it leaves out.** The text does not give the log base (nats or bits). It implies the loss is per
  token. It does not say whether the tokenizer was held fixed across models, which matters because
  OpenAI moved from cl100k to o200k-family tokenizers [M]. It gives no snapshot date and no dataset size.
- **Caveats the author states.** Labs use different eval sets and data mixes, so cross-lab comparison
  is hard. Chain-of-thought and RL gains do not lower perplexity. The entropy floor is unknown, so small
  loss gains near the floor may mean large capability jumps. Footnote 4 says code is a weak proxy for the
  abilities of GPT-2 and GPT-3.
- **The proposal.** Shared transparency norms and "common limits": labs would not cross a perplexity
  threshold until a milestone is met, such as a successor to chain-of-thought monitors.
- **[U] How many models are plotted.** The chart is an image, so the page text does not show the count.
  Cellcog's secondary write-up reads 8 points off the image: GPT-3.5 0.99, GPT-4 0.91, GPT-4.5 0.88,
  GPT-5.5 0.85, GPT-5.6 Sol 0.83, GPT-6 Astra 0.78, GPT-6.2 Astra 0.76, internal 0.73. It also lists
  capability lines at 0.99, 0.91, 0.87, 0.83 and 0.76, and a shaded band below 0.71
  (https://cellcog.ai/blog/openai-perplexity-and-intelligence/). The brief for this survey says 14
  models. Check the figure itself.

**Reactions (as of 2026-11-22).** I found only news and aggregator coverage, mostly restating the post:
FourWeekMBA, startuphub.ai, mlq.ai, lavx, digg, dealroom, cryptobriefing, explainx (which looks
machine-generated). **[S]**

- Altman called it "the most important thing OpenAI has published in awhile". **[S]** (cellcog)
- At the time of writing, no response had appeared from Anthropic, Google DeepMind, Meta or xAI, and I
  found no analysis on LessWrong or the Alignment Forum, from Zvi, or from Epoch. **[S]** Expect these
  within days.
- **Related context:** Dario Amodei, "We Must Pace the Frontier" (2026; the post links to it).
  https://darioamodei.com/post/we-must-pace-the-frontier **[S]** Zvi wrote about that essay:
  https://thezvi.substack.com/p/we-must-pace-the-frontier **[S]**
- **Also related:** Pachocki's earlier OpenAI post "An Alien Mind" and LessWrong discussion of it,
  https://www.lesswrong.com/posts/8GYhKdbEHs3vQZFv9/ **[S]**

**Prior lab disclosures of internal loss.**

- **GPT-4 Technical Report.** OpenAI, 2023. https://arxiv.org/abs/2303.08774 **[V]** Figure 1 predicts
  GPT-4's final loss on a dataset "derived from our internal codebase" (not in the training set). It
  uses a power law with an irreducible term, fit to models trained with ≤1/1000 of GPT-4's compute. An
  early draft labelled the y-axis "bits per word". This is the same dataset family as the 2026 post, so
  the 2026 post extends a curve OpenAI first showed in 2023.
- **Gopher (2112.11446) and Chinchilla (2203.15556).** DeepMind, 2021–22. **[M]** Both report Pile BPB
  per subset. Earlier lab-published BPB on public text.
- **DeepSeek-V2 and V3 reports.** **[V]** for V3: https://arxiv.org/abs/2412.19437. Pile-test BPB:
  DeepSeek-V3-Base 0.548, LLaMA-3.1-405B-Base 0.542, Qwen2.5-72B-Base 0.638, DeepSeek-V2-Base 0.606. A
  rare same-pipeline, cross-lab BPB comparison. Qwen2.5-72B trails by about 0.09 BPB despite
  competitive benchmark scores, a sign of data-mix and domain effects. Pile is old and likely
  contaminated.
- **OLMo / Paloma.** AI2. https://arxiv.org/abs/2402.00838 **[S]** Reports BPB on 11 Paloma sources.
- **Llama 3 Herd (2407.21783).** Meta, 2024. **[M]** Predicts benchmark accuracy in two steps: compute
  to the NLL of the correct answer, then NLL to accuracy. This links loss to capability, but on task
  NLL, not raw-text BPB.
- **Gemini 1.5 report (2403.05530).** **[M]** Cumulative NLL against context position, on long
  documents and code. Loss is shown as a function of context, not compared across labs.

---

## 1. Loss / BPB versus downstream capability

**Compression Represents Intelligence Linearly.** Huang, Zhang, Shan, He (HKUST, Tencent), COLM 2024.
https://arxiv.org/abs/2404.09937 **[V]**

- **Setup.** 31 base models. BPC on fresh-for-2023 corpora: Common Crawl Sep–Oct 2023, GitHub Python
  Jun–Sep 2023, arXiv math Sep–Oct 2023. Context fixed at 1,900 tokens with stride 512.
- **Result.** Pearson ρ ≈ −0.93 with the 12-benchmark average, and about −0.94 to −0.95 per domain.
- **Where it breaks.** The correlation needs the corpus domain to match the benchmark; off-domain pairs
  are weak. Qwen sat far above the line on GSM8K and MATH, apparently from benchmark exposure, and was
  excluded from the fit. MMLU format changes moved some models by about 20 points. The correlation
  saturates at about 3M characters of corpus.
- **Scope.** Base models only; fine-tuned models are explicitly out of scope. 31 models, many from the
  same families.

**Is Compression Really Linear with Code Intelligence?** Xuyang, Luo, Chu, …, Zhu, Zhou, 2025, revised
to v5 in Mar 2026. https://arxiv.org/abs/2505.11441 **[V]** On code, code capability vs BPC is
**logarithmic**, not linear. The linear fit holds only on the tail of that curve. Benchmark format also
matters, and the authors introduce "Format Annealing" to measure intrinsic code capability. This is the
domain OpenAI uses.

**Language Modeling Is Compression.** Delétang, Ruoss, et al. (DeepMind), ICLR 2024.
https://arxiv.org/abs/2309.10668 **[V]** Formal prediction–compression equivalence through arithmetic
coding. Chinchilla-70B beats PNG and FLAC on image and audio. The paper discusses tokenization as
pre-compression, and it notes that counting the parameter cost reverses the gains at scale. It is the
theoretical basis for "BPB = compression".

**Observational Scaling Laws.** Ruan, Maddison, Hashimoto, NeurIPS 2024.
https://arxiv.org/abs/2405.10938 **[V]** Fits about 100 public models from many labs on a
low-dimensional capability space derived from benchmarks. Model families differ mainly in how
efficiently they turn compute into that capability. This is a cross-lab template that does not need a
shared loss. A BPB axis could be added as another observable.

**Loss-to-Loss Prediction.** Brandfonbrener, Anand, Vyas, Malach, Kakade, 2024.
https://arxiv.org/abs/2411.12925 **[V]** Train-to-train, train-to-test and test-to-test losses across
datasets follow shifted power laws, so a scaling law can be translated between datasets. The basis for
mapping one lab's held-out loss onto another corpus, but only within controlled model families.

**LLMs on the Line: Data Determines Loss-to-Loss Scaling Laws.** Mayilvahanan, Wiedemer, Mallick,
Bethge, Brendel, ICML 2025. https://arxiv.org/abs/2502.12120 **[V]**

- **Finding.** Over more than 6,000 checkpoints, the **pretraining data** sets the loss-to-downstream
  line. Tokenizer, architecture, size and optimizer have little effect; v1 had also named the
  tokenizer.
- **Why it matters here.** Models from different labs, trained on different data, sit on different
  lines. The same BPB need not mean the same capability across labs.

**Understanding Emergent Abilities from the Loss Perspective.** Du, Zeng, Dong, Tang (Zhipu,
Tsinghua), NeurIPS 2024. https://arxiv.org/abs/2403.15796 **[V]** Abilities appear once pretraining
loss falls below task-specific thresholds; above them, performance is at chance. This is the academic
version of OpenAI's "capability at 0.76" framing. The thresholds are defined only for one
corpus–tokenizer setup [M].

**Language Models Scale Reliably with Over-training and on Downstream Tasks.** Gadre et al., ICLR
2025. https://arxiv.org/abs/2403.08540 **[V]** A power law links perplexity to average top-1 error
across 104 models. It holds **for models trained on the same data**. Single tasks are noisier.

**Establishing Task Scaling Laws via Compute-Efficient Model Ladders.** Bhagia et al. (AI2), 2024.
https://arxiv.org/abs/2412.04403 **[V]** Two steps: compute to task BPB (NLL of the correct answer per
byte), then BPB to accuracy. Within about 2 points on 4 multiple-choice tasks. BPB is used so that
tokenizers can be compared.

**Why Has Predicting Downstream Capabilities … Remained Elusive?** Schaeffer et al., ICML 2025.
https://arxiv.org/abs/2406.04391 **[V]** Accuracy depends on probability mass on specific wrong
answers, not just loss on the correct one. This is why loss is predictable and benchmarks are not.

**Are Emergent Abilities of LLMs a Mirage?** Schaeffer, Miranda, Koyejo, NeurIPS 2023.
https://arxiv.org/abs/2304.15004 **[M]** Emergence is partly an artifact of discontinuous metrics.
Under continuous metrics, gains track loss smoothly. A counterpoint to threshold claims.

**Scaling Laws for Downstream Task Performance of LLMs.** Isik et al., 2024.
https://arxiv.org/abs/2402.04177 **[M]** Under distribution mismatch between pretraining and the
downstream task, downstream metrics can move non-monotonically with pretraining loss. Alignment between
the corpus and the task matters.

**Ranking LLMs by Compression.** Guo et al., 2024. https://arxiv.org/abs/2406.14171 **[V]** Five LLMs.
Compression ratio correlates with NLP task performance. A small sample.

**Mahoney, "Text Compression as a Test for AI".** 1999.
https://gwern.net/doc/cs/algorithm/information/compression/1999-mahoney.pdf **[S]** Historical
precursor, alongside the Hutter Prize.

---

## 2. Comparing models with different tokenizers

**BPB definition (The Pile, Gao et al. 2020).** https://arxiv.org/abs/2101.00027 **[M]**
BPB = (L_T / L_B) · NLL_token / ln 2, where L_T is the token count and L_B the UTF-8 byte count. Paloma,
HELM and OLMo use it. Count bytes from the raw UTF-8 text, never from decoded tokens; SentencePiece "▁"
and leading spaces inflate the count (BPBench README) **[V]**.

**Tokenizer Myths and Where To Find Them.** 2026-11. https://arxiv.org/abs/2611.04215 **[V]**
(abstract only)

- **Myth #1: "BPB is tokenizer-agnostic."** BPB = tokens/byte × bits/token, and the two factors do not
  cancel. Untrained models already show BPB preferences between vocabularies.
- On trained models with the same data, BPB correlates only weakly with downstream performance.
- **Why it matters here.** BPB differences between labs partly reflect tokenizer–model pairing, not
  just model quality.

**Exact Byte-Level Probabilities from Tokenized LMs for FIM-Tasks and Model Ensembles.** Phan, Amos,
Gat, Havasi, Muckley, Ullrich (Meta), ICLR 2025. https://arxiv.org/abs/2410.09303 **[V]**
Byte-Token Representation Lemma: an exact map from a token LM to a byte LM at inference time, with no
retraining. It removes "tokenization bias", under which two statistically equivalent models can give
different next-byte predictions. Code: facebookresearch/Exact-Byte-Level-Probabilities-from-Tokenized-LMs.

**Understanding and Mitigating Tokenization Bias in LMs.** Phan, Havasi, Muckley, Ullrich, 2024.
https://arxiv.org/abs/2406.16829 **[V]** Maximum prefix encoding and BPE conditioning cause a bias
that more data cannot fix. A correction algorithm simulates token-free behaviour.

**From Language Models over Tokens to Language Models over Characters.** Vieira, Cotterell, et al.
(ETH), ICML 2025. https://arxiv.org/abs/2412.03719 **[V]**

- Exact and approximate token-to-character conversion. On Llama-3.1-8B, excess is under 0.00021 bits
  per character at 46.3 characters/s.
- Marginalizing over tokenizations gives a **significant BPB improvement**. Canonical-tokenization BPB
  overstates the true character-level loss, by an amount that differs by model.
- Code: genlm/genlm-bytes.

**Sampling from Your Language Model One Byte at a Time.** Hayase et al., 2025.
https://arxiv.org/abs/2506.14123 **[S]** Handles the prompt-boundary problem at byte level.

**TokSuite: Measuring the Impact of Tokenizer Choice on LM Behavior.** 2025.
https://arxiv.org/abs/2512.20757 **[S]** (not read)

**Benchmarking BPE Tokenizers …** MELLM workshop 2026.
https://aclanthology.org/2026.mellm-1.27.pdf **[S][U]** Raw perplexity rankings shift by up to 21.6%
across 19 models. It proposes tokenizer-normalized evaluation; the method is unverified.

**Lessons from the Trenches on Reproducible Evaluation of LMs.** Biderman et al. (EleutherAI), 2024.
https://arxiv.org/abs/2405.14782 **[S]** Practical guidance on BPB, perplexity normalization and
lm-eval-harness pitfalls.

**Byte Latent Transformer.** Pagnoni et al. (Meta), 2024. https://arxiv.org/abs/2412.09871 **[S]**
Reports BPB because per-token perplexity only has meaning under a fixed tokenizer. Byte-level models
are a tokenizer-free reference point.

**BOS and context details.** **[V]** UncheatableEval provides `bos_mode_finder.py` because the official
BOS setting is often wrong. BPBench scores some models with an explicit-BOS HF scorer because vLLM
"mis-serves" them. Huang et al. fixed context at 1,900 tokens. A token-based window gives
different-tokenizer models different amounts of byte context, so fix the context in bytes.

---

## 3. Fresh, contamination-free perplexity benchmarks

**BPBench.** Trelis Research, Jul 2026. https://github.com/TrelisResearch/bpbench and
https://research.trelis.com/bpbench **[V]** This is the closest prior work to the planned study.

- **Data.** Exact BPB on fresh, provenance-verified human text: US Congressional Record (259 docs) and
  UK Hansard (147 docs), both July 2026, plus a private held-out prose slice (51 docs). Spoken-channel
  transcripts give provenance instead of AI detectors. Fresh code was dropped because its provenance
  could not be vetted in time.
- **Leaderboard (macro BPB).** A tied top cluster at 0.615–0.652: DeepSeek-V4-Pro 0.615,
  Nemotron-3-Ultra 0.620 (NVFP4), Kimi K2.6 0.621, GLM-5.2, Qwen3.7-Plus and others. MiniMax-M2.7 0.738.
  Qwen3.6-27B 0.758.
- **Outliers.** GPT-OSS-120B 1.51 and GPT-OSS-20B 1.60. Gemma-4-31B-it about 2.36, provisional. The
  authors suggest synthetic- or chat-heavy training. Capable models can compress fresh human text very
  badly.
- **Post-training raises BPB.** Llama-3-8B moves from 0.624 (base) to 0.712 (instruct) on Hansard, a
  gap larger than the whole frontier spread. Only one pair was tested.
- **Closed models** (GPT-5.6 Sol, Claude Fable, Gemini Flash 3.6) are **not placed**. APIs do not
  return prompt logprobs, and three sampling estimators were tried:
  - completions resampling (n=32) has a floor of about 6 bits/word;
  - plain chat adds about +2.7 bits/word of reply-start bias that differs by model;
  - echo chat ("Begin exactly with: …") reaches 92–100% adherence but still has model-dependent bias.
  None was reliable enough to report.
- **Rules.** Reasoning must be off and verified through API telemetry. Read "tiers, not fine
  rankings".

**Uncheatable Eval: Dynamic Compression-Based Evaluation of LMs.** Tan, Li, Shen, 2026-09.
https://arxiv.org/abs/2609.27510; code https://github.com/Jellyfish042/uncheatable_eval; HF Space
Jellyfish042/UncheatableEval **[V]**

- **Data.** Regularly crawled new text from the past 1–2 weeks in 14 categories: arXiv by field, GitHub
  by language, BBC News, AO3, Wikipedia in English and other languages.
- **Scope.** 80 base models. Reports compression rate, BPC and BPB. Compression scales with model size;
  attention, hybrid and recurrent models behave differently as context grows. Strongly linked to
  zero-shot MMLU (coefficient not seen).
- Base models only. No API models.

**Evaluating LLMs for Generalization and Robustness via Data Compression.** Li, Guo, Guerin, Lin,
2024. https://arxiv.org/abs/2402.00861 **[V]** The predecessor of UncheatableEval. 14 LLMs on 83
months of data (2017–2023), split at each model's cutoff. Compression worsens after the cutoff for
many models; news and code are hardest. **Context size and tokenization strongly affect results.**

**LatestEval.** Li, Guerin, Lin, AAAI 2024. https://arxiv.org/abs/2312.12343 **[V]** Uses only text
inside a recent time window (arXiv, BBC, GitHub) to avoid overlap with training data. It is QA rather
than BPB, but it shares the time-window method.

**Dated Data: Tracing Knowledge Cutoffs in LLMs.** Cheng et al. (JHU), COLM 2024 Outstanding Paper.
https://arxiv.org/abs/2403.12958 **[V]** Effective cutoff = the minimum of perplexity over dated
snapshots of Wikipedia and NYT. Effective cutoffs often differ from reported ones. Use this to check
whether "fresh" text is really fresh for each model.

**Is Your LLM Outdated? (FreshBench).** Zhu et al., NAACL 2025. https://arxiv.org/abs/2405.08460
**[V]** Character-normalized BPC on fresh text. Temporal bias, and decline over time after the cutoff.

**Paloma.** Magnusson et al. (AI2), NeurIPS 2024 D&B. https://arxiv.org/abs/2312.10523 **[V]** BPB on
546 English and code domains from 16 sources (585 domains and 18 sources in v1). Supports
decontamination and controlled submissions. It shows rankings change by domain, so macro-averaging
choices matter.

**Algorithmic Progress in Language Models.** Ho, Besiroglu, et al. (Epoch), NeurIPS 2024.
https://arxiv.org/abs/2403.05812 **[V]** Over 200 cross-lab perplexity evaluations on WikiText and PTB,
2012–2023. Compute to reach a fixed perplexity halves about every 8 months. This is the main
third-party cross-lab analysis using loss, but on old, contaminated benchmarks.

**[S] Epoch on compute per fixed loss.** A Redwood summary says Epoch found the compute needed for a
fixed next-token loss falls about 3× per year. Not confirmed on Epoch's site.
https://blog.redwoodresearch.org/p/whats-going-on-with-ai-progress-and

I found no Epoch analysis comparing frontier labs on a shared perplexity set.

---

## 4. Getting probabilities from closed APIs

**Logits of API-Protected LLMs Leak Proprietary Information.** Finlayson, Ren, Swayamdipta, COLM 2024.
https://arxiv.org/abs/2403.09539 **[V]** Through the softmax bottleneck, full-vocabulary logprobs can be
recovered from restricted API outputs, for under $1,000 for gpt-3.5-turbo. This is a possible route to
exact closed-model BPB where top-k logprobs plus logit bias exist. Providers have since restricted
logit bias [M].

**Stealing Part of a Production LM.** Carlini et al., 2024. https://arxiv.org/abs/2403.06634 **[M]**
Same bottleneck. It led OpenAI and Google to limit logit-bias and logprob combinations.

**API logprob status.** **[S][U]** No logprobs on Anthropic or on OpenAI reasoning models
(github.com/Robby955/logprobe). OpenAI non-reasoning chat models return output-token logprobs only
(top ≤ 20). OpenAI's `echo` prompt scoring was retired in 2023 [M]. Gemini has `responseLogprobs` on
output tokens [M]. Re-check all of this at study time.

---

## Implications for the study design

1. **Closed models cannot be placed on an exact-BPB axis through public APIs.** BPBench, the closest
   prior work, tried three estimators and declined to report closed models. Either secure lab
   cooperation (scoring by the lab, a lab-run harness, or a prompt-logprob endpoint), or treat
   closed-model values as estimates with stated bias. If estimates are used, calibrate the estimator on
   open models where exact BPB is known, and report tiers.
2. **Base versus deployed checkpoint.** Closed models are only available post-trained, and
   post-training raises BPB, by +0.09 in one pair, larger than the frontier spread. Score open models in
   both base and instruct form to measure the gap. Compare closed models to open instruct models, or
   state the confound. Huang et al. and UncheatableEval both restrict to base models for this reason.
3. **BPB is not fully tokenizer-neutral.** Count bytes from raw text and fix context in bytes, not
   tokens. Audit BOS handling per model. As a robustness check, run byte-level marginalization (Vieira
   et al. via genlm-bytes, or Phan et al.) on a subset to see whether rankings change.
4. **Loss and capability are not one cross-lab line.** The pretraining data mix sets the line ("LLMs on
   the Line"). The correlation is domain-matched and breaks for benchmark-tuned models (Huang). It is
   logarithmic on code (2505.11441). Synthetic-heavy models such as GPT-OSS and Gemma-4 are capable but
   compress fresh human text badly. Report per-domain slices (prose, code, math or science), not just a
   macro mean, and expect lab-level offsets.
5. **Freshness has to be shown, not assumed.** Use text dated after every model's reported cutoff.
   Check effective cutoffs with perplexity-by-month curves (Dated Data, Li et al.). Prefer provenance
   such as transcripts or dated records over AI detectors, because new web text increasingly contains
   LLM output, which favours some models. Keep a private held-out slice.
6. **OpenAI's numbers cannot be converted to the study's axis.** They are per-token, with no stated log
   base, on internal code, with tokenizers that may change across models. The 0.76 and 0.68–0.70
   thresholds cannot be mapped to BPB on public text. At most, compare orderings of OpenAI models if any
   become scorable.
