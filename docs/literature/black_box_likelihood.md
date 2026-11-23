# Black-box likelihood estimation and next-word guessing games: annotated list

Survey date 2026-11-22. Context: putting closed chat models on a bits-per-byte (BPB) axis with a
Shannon-style top-20 guessing game, calibrated against exact BPB on open base and instruct models.

Verification markers:
- **[V]**: title, authors, year and URL confirmed this session by search or fetch; the summary comes from the abstract or page.
- **[V-abs]**: existence and abstract confirmed, details below the abstract not checked.
- **[M]**: from prior knowledge; not fetched this session. Check before citing.
- **[U]**: secondary or community source (forum, blog) or a detail I could not confirm.

Relevance tags: **SUPPORTS**, **UNDERMINES**, **CAVEAT** (a bias the design must control), **TOOL** (a usable technique).

---

## 1. Shannon's guessing game and its modern uses

1. **Prediction and Entropy of Printed English**: C. E. Shannon, 1951, *Bell System Technical Journal* 30:50–64. https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf **[V]**
   A human guesses each next letter in turn. The number of guesses needed gives a rank sequence, from which Shannon derived *upper and lower bounds* (about 0.6–1.3 bits/char), not a point estimate. He also described the variant your design uses: stop after a fixed number of misses and reveal the symbol.
   *Relevance*: SUPPORTS the idea. CAVEAT: the rank entropy H(R) is a valid code length, but for an imperfect guesser it is only an upper bound. It throws away the guesser's confidence (see the "design implications" section).

2. **A Convergent Gambling Estimate of the Entropy of English**: T. Cover & R. King, 1978, *IEEE Trans. Inf. Theory* 24(4):413–421. https://www.gwern.net/doc/cs/algorithm/information/1978-cover.pdf **[V]**
   Subjects bet money on the next symbol. The bets recover a full probability distribution, which guessing alone cannot. Individual subjects measured 1.3–1.7 bpc; combining them gave about 1.3 bpc. The authors state explicitly that the guessing procedure gives only partial information about next-symbol probabilities.
   *Relevance*: the direct precedent for adding *verbalized probabilities* to the top-20 list. A bet-based code is a proper scoring rule, so it can approach the guesser's real cross-entropy. A rank code cannot.

3. **Entropy Rate Estimation for English via a Large Cognitive Experiment Using Mechanical Turk**: G. Ren, S. Takahashi, K. Tanaka-Ishii, 2019, *Entropy* 21(12):1201. https://www.mdpi.com/1099-4300/21/12/1201 **[V]**
   A crowdsourced replication of Shannon's game, extrapolated to about 1.22 bpc. The authors find Shannon's single-subject sample too small to converge. They also cite Moradi et al. (about 1.6 bpc).
   *Relevance*: shows that guessing-game estimates are sample-hungry and depend on the extrapolation. Budget enough positions per document.

4. **Language Models Are Better Than Humans at Next-Token Prediction**: B. Shlegeris, F. Roger, L. Chan, E. McLean, 2022 (TMLR 2024). https://arxiv.org/abs/2212.11281 ; blog post https://www.lesswrong.com/posts/htrZrxduciZ5QaCjw **[V]**
   Two experiments on OpenWebText: a top-1 guessing game (60 participants, about 18.5k guesses) and a human perplexity estimate from pairwise relative-probability judgments. Humans did worse than GPT-Neo-1.3B and GPT-2-large on both. The authors argue that Shannon-style bpc numbers are not comparable with per-token LM perplexity.
   *Relevance*: methodological precedent for scoring a *player* operationally rather than reading its internals. It also shows top-1 or rank metrics and perplexity can tell different stories.

5. **Large Language Models and the Entropy of English**: C. Scheibner, L. M. Smith, W. Bialek, 2025-12-31. https://arxiv.org/abs/2512.24969 **[V-abs]**
   Uses open LLMs as the Shannon "player", but reads off the full distribution, so the conditional entropy is computed directly. Conditional entropy keeps falling as context grows, up to about 10^4 characters, and a growing fraction of characters becomes near-certain at long context.
   *Relevance*: CAVEAT. Context length strongly changes BPB. Fix the context length, and keep it identical between the guessing game and the exact-BPB calibration runs. The many near-certain positions are where a rank code costs the most relative to true log-loss.

6. **Express Your Doubts: Probabilistic World Modeling Should not be Based on Token logprobs**: E. Wagner & O. Abend, 2025 (ICML 2026 position track). https://arxiv.org/abs/2505.02072 **[V-abs]**
   Position paper. It argues that the Shannon game is a *response-prediction* task: a guesser names its best guess, which is not the same as sampling from the language model. Treating the game as autoregressive language modeling, as Zouhar et al. 2024 do **[U: Zouhar paper not located]**, conflates the two. It argues for verbalized distributions.
   *Relevance*: CAVEAT, central to the design. What a chat model lists as its "top-20 guesses" is a response policy, not necessarily the argsort of its next-token distribution.

7. **Predict the Next Word: Humans exhibit uncertainty in this task and language models ___**: E. Ilia & W. Aziz, 2024, EACL (short). https://arxiv.org/abs/2402.17527 **[V-abs]**
   On the Provo cloze corpus, with about 40 human completions per prefix, GPT-2, BLOOM and ChatGPT are poorly calibrated to the spread of human next-word responses. ChatGPT reportedly does worst **[U: how ChatGPT's distribution was obtained, probably by sampling, not verified]**. The authors also argue that expected calibration error (ECE) is the wrong metric here.
   *Relevance*: early evidence that chat models' next-word distributions are distorted relative to base models.

8. **Large-scale cloze evaluation reveals that token prediction tasks are neither lexically nor semantically aligned**: 2024. https://arxiv.org/abs/2410.12057 **[V-abs; authors not checked]**
   GPT-2, RoBERTa and Pythia under-estimate human cloze probabilities by orders of magnitude and over-rank rare responses.
   *Relevance*: background only. It concerns cloze, not compression.

9. **LLMZip: Lossless Text Compression using Large Language Models**: C. S. K. Valmeekam et al., 2023. https://arxiv.org/abs/2306.04050 **[V-abs]**
   Codes the *rank* of the true token under LLaMA-7B, compressed with zip, Huffman or arithmetic coding. Reports about 0.71 bpc on text8. Rank coding is slightly worse than arithmetic coding on the probabilities.
   *Relevance*: SUPPORTS. A rank code is a legitimate compressor whose cost is close to log-loss coding when the ranks come from the true distribution. TOOL: a reference for the size of the rank-code penalty versus arithmetic coding.

10. **LLM-based Source Code Compression via Thresholded Symbol Ranking**: 2026. https://arxiv.org/abs/2607.24192 **[V-abs; authors not checked]**
    A top-T rank code with exceptions for out-of-threshold symbols, the same structure as your "rank code + escape". With T=63 it compressed better than unlimited rank. A related rank-tier paper notes that fixed-tier codes cost more bits per byte than arithmetic coding.
    *Relevance*: SUPPORTS the escape structure. The truncation point K matters little once K is a few dozen.

11. **Humans Outperform Machines at the Bilingual Shannon Game**: (ResearchGate listing) https://www.researchgate.net/publication/312250823 **[U: details not read]**. A bilingual variant of the guessing game.

---

## 2. Black-box estimation of LM probabilities

### 2a. Sampling and completion resampling

12. **BPBench**: Trelis Research, 2026. Repository https://github.com/TrelisResearch/bpbench ; write-up https://research.trelis.com/bpbench **[V]**
    *The closest prior work to the study.* It measures exact BPB on fresh, post-cutoff transcripts of parliamentary speech (US Congressional Record, UK Hansard, and a private slice), using echo plus logprobs. Thirteen open models are on the board; DeepSeek-V4-Pro is best at 0.615 BPB. Three black-box estimators were tried against exact BPB on 10 open models:
    - **Completions resampling** (n=32 samples on the raw string): close for strong models, but has a "6-bit floor", so it under-reads weak models (Laguna reads 0.11–0.17 too low). It is unavailable for chat-only APIs.
    - **Plain chat** ("next word?"): fails. The first assistant token carries mass for reply-openers, adding about 2.7 bits/word on Inkling. The bias varies by model (−0.01 to −0.19 BPB), so it cannot be calibrated out.
    - **Echo chat** (begin by repeating exactly the quoted last 8 words, then continue): quoting the string gives 92–100% compliance, against about 2% unquoted. It restores the conditional (median 2.34 bits/word against 2.31 raw; 4.74 cold) and leaves a near-uniform offset, but it stays noisy.
    Conclusion: they "did not find a robust way" to estimate commercial-model BPB, and placed **no closed models**. They advise reading the board as *tiers*. Base versus instruct: Llama-3-8B scores 0.624 base and 0.712 instruct on Hansard, a gap larger than the whole frontier spread. The 95% CIs are about ±0.015.
    *Relevance*: CRITICAL. (a) Use the echo framing, or something equivalent, so the guess is conditioned on the text and not on "start of a reply". (b) The +2.7 bits/word first-token effect and its model dependence would contaminate a naive guessing prompt the same way. (c) Model-dependent bias is the failure mode that defeated them. Your calibration must show the guessing-game-to-BPB mapping is *shared* across models, including across chat models, not just fit on average.

13. **Blackbox Model Provenance via Palimpsestic Membership Inference**: R. Kuditipudi, J. Huang, S. Zhu, D. Yang, C. Potts, P. Liang, 2025 (NeurIPS 2025). https://arxiv.org/abs/2510.19796 **[V-abs]**
    Search snippets report a sampling-based "empirical log-perplexity" that correlates with true log-likelihood ranks: Spearman 0.71, 0.85 and 0.92 at N = 1, 4 and 16 samples, on OLMo-7B **[U: numbers from the search snippet, not checked in the PDF]**.
    *Relevance*: TOOL. Sampling frequencies give usable *rankings* with few samples, but absolute values carry a floor of log2(N).

14. **Sampling-based Pseudo-Likelihood for Membership Inference Attacks (SaMIA)**: M. Kaneko, Y. Ma, Y. Wata, N. Okazaki, 2024 (Findings of ACL 2025). https://arxiv.org/abs/2404.11262 **[V-abs]**
    Generates continuations of a prefix and scores ROUGE-N overlap with the true suffix as a pseudo-likelihood. It matches likelihood-based membership inference on Wikipedia. "Membership Inference on LLMs in the Wild" (2026, https://arxiv.org/abs/2601.11314, **[V-abs]**) proposes SimMIA, which samples word by word.
    *Relevance*: SUPPORTS. Generation-only signals track likelihood well enough to rank texts.

### 2b. Logit-bias and logprob extraction (needs an API knob most closed models no longer expose)

15. **Language Model Inversion**: J. X. Morris, W. Zhao, J. T. Chiu, V. Shmatikov, A. M. Rush, 2023 (ICLR 2024). https://arxiv.org/abs/2311.13647 **[V-abs]**
    Recovers full next-token probabilities through an API that offers only argmax plus `logit_bias`, by binary search for the smallest bias that flips the argmax.
    *Relevance*: TOOL, but only where `logit_bias` exists. That covers older OpenAI chat models; Anthropic offers none, and the newest OpenAI and Gemini models are unclear (see item 33).

16. **Logits of API-Protected LLMs Leak Proprietary Information**: M. Finlayson, X. Ren, S. Swayamdipta, 2024 (COLM 2024). https://arxiv.org/abs/2403.09539 **[V]**
    The softmax bottleneck puts the outputs in a low-dimensional subspace. With a few full outputs (via logit bias), you can recover full-vocabulary distributions cheaply, estimate the hidden size (about 4096 for gpt-3.5-turbo), and fingerprint the model.
    *Relevance*: TOOL where logit bias exists. Otherwise it is background on why providers removed logprobs and logit bias.

17. **Stealing Part of a Production Language Model**: N. Carlini et al., 2024 (ICML 2024 best paper). https://arxiv.org/abs/2403.06634 **[V-abs]**
    Recovers the final projection matrix (hidden sizes 1024 and 2048 for Ada and Babbage) from top-k logprobs plus logit bias, for under $20. After it, providers restricted logprob and logit-bias access.
    *Relevance*: explains why the guessing game is needed at all.

### 2c. Verbalized confidence and distributions

18. **Just Ask for Calibration**: K. Tian, E. Mitchell, A. Zhou, A. Sharma, R. Rafailov, H. Yao, C. Finn, C. D. Manning, 2023 (EMNLP 2023). https://arxiv.org/abs/2305.14975 **[V]**
    For ChatGPT, GPT-4 and Claude, verbalized confidences, especially when the model lists its top-k answers with probabilities, are better calibrated than the RLHF model's own conditional probabilities. ECE often falls by about 50% relative. RLHF worsens Llama-70B's logprob calibration.
    *Relevance*: SUPPORTS asking for *probabilities* on the top-k list, not just an order. Caveat: this is QA-answer calibration, not next-word log-loss.

19. **Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs**: M. Xiong et al., 2023 (ICLR 2024). https://arxiv.org/abs/2306.13063 **[V-abs]**
    A framework of verbalized prompts (CoT, Top-K, multi-step), sampling, and consistency aggregation. Verbalized confidence is *overconfident*. Consistency across samples beats verbalized confidence, and hybrids do best. Black-box methods trail white-box ones only slightly.
    *Relevance*: TOOL. Combining a few samples (agreement on the top guess) with a verbalized list should beat either alone.

20. **Teaching Models to Express Their Uncertainty in Words**: S. Lin, J. Hilton, O. Evans, 2022 (TMLR). https://arxiv.org/abs/2205.14334 **[M]**. Fine-tuned GPT-3 to state calibrated verbal probabilities. Background.

21. **Language Models (Mostly) Know What They Know**: S. Kadavath et al. (Anthropic), 2022. https://arxiv.org/abs/2207.05221 **[M]**. Base models' token probabilities are well calibrated on multiple-choice questions, and RLHF degrades this unless a temperature is applied. Background for item 26.

22. **Prompting Is Not a Substitute for Probability Measurements in Large Language Models**: J. Hu & R. Levy, 2023 (EMNLP 2023). https://arxiv.org/abs/2305.13264 **[V]**
    Metalinguistic prompts ("which word is more likely?") agree less with the model's own string probabilities the further they move from a direct next-word measurement. Prompted judgments understate what the model knows.
    *Relevance*: CAVEAT. A "list your top-20 guesses" prompt is metalinguistic. Expect the listed ranking to be noisier than the true argsort, and keep the task as close to plain continuation as possible (the echo framing helps here).

23. **Language Models Fail to Introspect About Their Knowledge of Language**: S. Song, J. Hu, K. Mahowald, 2025 (COLM 2025). https://arxiv.org/abs/2503.07513 **[V-abs]**
    Across 21 open models, on grammaticality and *word prediction*, prompted answers predict the model's own string probabilities no better than another similar model's probabilities do. There is no privileged self-access.
    *Relevance*: CAVEAT, important. The guessing game measures the model's *prompted* predictive knowledge, which is related to, but distinct from, its token distribution. That is fine if the study defines the quantity operationally ("guessing-game bits per byte"), and problematic if it claims to estimate the closed model's true BPB.

24. **Speak vs. Know: Balancing Behavior and Knowledge Distributions of Large Language Models**: Y. Liu, M. Kaneko, C. Chu, 2026-10. https://arxiv.org/abs/2610.27648 **[V-abs]**
    Verbalized and logprob distributions disagree on the argmax for 37% of World Values Survey items, averaged over three LLMs.
    *Relevance*: CAVEAT. The same gap, in a different domain.

25. **Forcing Diffuse Distributions out of Language Models**: Y. Zhang et al., 2024. https://arxiv.org/abs/2404.10859 **[M]**. Instruct models are badly non-uniform when asked to produce random or diverse outputs. Relevant to whether a "top-20 list" covers the tail.

### 2d. Membership and contamination detection without logprobs (methods transferable to "likelihood ranking")

26. **Detecting Pretraining Data from Large Language Models (Min-K% Prob)**: W. Shi, A. Ajith, M. Xia, Y. Huang, D. Liu, T. Blevins, D. Chen, L. Zettlemoyer, 2023 (ICLR 2024). https://arxiv.org/abs/2310.16789 **[V-abs]**
    Averages the log-probability of the k% lowest-probability tokens; the WikiMIA benchmark splits data by date relative to training. It needs logprobs.
    *Relevance*: the *tail* tokens carry the membership and novelty signal, and they are exactly the tokens a top-20 game sends to escape. Your escape coder decides how much of the signal survives.

27. **DE-COP: Detecting Copyrighted Content in Language Models Training Data**: A. V. Duarte, X. Zhao, A. L. Oliveira, L. Li, 2024 (ICML 2024). https://arxiv.org/abs/2402.09910 **[V]**
    A multiple-choice quiz: pick the verbatim passage among paraphrases. It reaches 72% accuracy on fully black-box models, where earlier methods were near 0%.
    *Relevance*: a quiz framing works on closed models. Also a contamination check you could run on your "fresh" corpus.

28. **Investigating Data Contamination in Modern Benchmarks for LLMs (TS-Guessing)**: C. Deng, Y. Zhao, X. Tang, M. Gerstein, A. Cohan, 2023 (NAACL 2024). https://arxiv.org/abs/2311.09783 **[V]**
    Mask a wrong option or a word and ask the model to fill it in. GPT-4 recovers masked MMLU options at 57% exact match.
    *Relevance*: a guessing-style probe for memorization. It shows closed-model exact-guess rates are a strong contamination signal, so fresh text is essential.

29. **Time Travel in LLMs: Tracing Data Contamination in Large Language Models**: S. Golchin & M. Surdeanu, 2023 (ICLR 2024). https://arxiv.org/abs/2308.08493 **[V]**
    "Guided instruction" gives the dataset name and partition plus the start of an instance, and compares the overlap of the completion with a general prompt.
    *Relevance*: a contamination check.

30. **Speak, Memory: An Archaeology of Books Known to ChatGPT/GPT-4**: K. Chang, M. Cramer, S. Soni, D. Bamman, 2023 (EMNLP 2023). https://arxiv.org/abs/2305.00118 **[V]**
    A name-cloze probe (human baseline 0%) reveals memorized books. Memorization tracks how often the book appears on the web.
    *Relevance*: single-word guessing on closed models as a likelihood proxy. Its success is a reminder that guessing accuracy on *seen* text measures memorization, not compression.

---

## 3. Instruct and RLHF models versus base models on natural text

31. **GPT-4 Technical Report**: OpenAI, 2023. https://arxiv.org/abs/2303.08774 **[V-abs]**. The calibration figure (MMLU) shows the pre-trained model is highly calibrated and the post-trained model much less so. OpenAI's research page says the same **[V]**; the figure itself was not re-read.
    *Relevance*: the standard citation that post-training distorts probabilities. It concerns answer calibration, not raw-text log-loss.

32. **Mysteries of Mode Collapse**: janus, 2022, LessWrong. https://www.lesswrong.com/posts/t9svvNPNmFf5Qa3TA **[V]** Corrections: https://www.lesswrong.com/posts/mbGjzyy6eJXT4gFpm (text-davinci-002 was FeedME/SFT, not RLHF) and https://www.lesswrong.com/posts/pjesEx526ngE6dnmr (the effect does not reproduce on RLHF'd davinci-003).
    *Relevance*: CAVEAT. Post-trained models can put most of their mass on "attractor" continuations. The cause, SFT or RLHF, is disputed.

33. **Understanding the Effects of RLHF on LLM Generalisation and Diversity**: R. Kirk et al., 2023 (ICLR 2024). https://arxiv.org/abs/2310.06452 **[V]**
    RLHF generalizes better out of distribution than SFT but sharply reduces output diversity.
    *Relevance*: chat models are sharper. A rank code and a log-loss code react differently to sharpness, so the base-to-instruct gap in guessing-game bits per byte may not equal the gap in BPB.

34. **Base Models Beat Aligned Models at Randomness and Creativity**: P. West & C. Potts, 2025. https://arxiv.org/abs/2505.00047 **[V-abs]**
    Aligned models are biased (for example toward "7" in random numbers) and more predictable in mixed-strategy games. Stronger benchmark scores go with weaker unpredictability.

35. **Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity**: J. Zhang, S. Yu, D. Chong, A. Sicilia, M. Tomz, C. D. Manning, W. Shi, 2025 (ICML 2026). https://arxiv.org/abs/2510.01171 **[V-abs]**
    Attributes mode collapse to typicality bias in preference data. Asking the model for "k responses *with their probabilities*" recovers diversity closer to the base model's.
    *Relevance*: SUPPORTS your format. A list-with-probabilities prompt is the published remedy for mode collapse and may move a chat model's listed distribution toward its pretrained one. TOOL: borrow the prompt.

36. **The Unlocking Spell on Base LLMs (URIAL; token distribution shift)**: B. Y. Lin et al., 2023 (ICLR 2024). https://arxiv.org/abs/2312.01552 ; analysis https://allenai.github.io/re-align/tds.html **[V]**
    Llama-2-chat and Llama-2 agree on the decoded token at most positions. The shifts (about 5–8%) are stylistic tokens and disclaimers, concentrated at the *start* of a response, and the aligned model's token is usually in the base model's top 5.
    *Relevance*: SUPPORTS. On natural-text continuation, the chat model's top-20 should mostly contain the base model's top candidates. It also explains BPBench's first-token problem: alignment's shift is largest at response start.

37. **Psychometric Predictive Power of Large Language Models**: T. Kuribayashi, Y. Oseki, T. Baldwin, 2023 (Findings of NAACL 2024). https://arxiv.org/abs/2311.07484 **[V-abs]**
    Instruction-tuned models' next-word probabilities predict human reading times worse than base models' at *equal perplexity*. Prompting helps but does not close the gap.
    *Relevance*: CAVEAT. Instruct and base distributions differ in shape, not only in overall loss.

38. **BPBench base-versus-instruct gap** (see item 12): Llama-3-8B scores 0.624 base and 0.712 instruct on Hansard **[V]**. The authors tentatively link GPT-OSS's poor raw-prose BPB (1.5–1.6) to synthetic and chat-heavy training.
    *Relevance*: your instruct-model calibration points are essential. Closed models are chat checkpoints, so the comparable open quantity is *instruct* exact BPB, not base.

39. **Alignment tax**: InstructGPT (L. Ouyang et al., 2022, https://arxiv.org/abs/2203.02155) reports regressions on public NLP tasks after RLHF, mitigated by PPO-ptx (mixing in pretraining gradients). Y. Lin et al., 2023, "Mitigating the Alignment Tax of RLHF" (https://arxiv.org/abs/2309.06256). **[M]**

### Recovering base-like distributions from chat models

40. **Prefilling**: as of the Claude 4.6 generation, Anthropic returns **HTTP 400 for a prefilled assistant turn**, per the official migration guide (https://platform.claude.com/docs/en/about-claude/models/migration-guide) **[V via search snippet]**. Secondary sources say the same holds for Opus 4.7+, Sonnet 4.6+ and Fable 5 **[U]**.
    *Relevance*: CHANGES THE DESIGN. You cannot prefill "…last words of the text" to force a raw continuation on current Claude models. Use BPBench's echo instruction, or a structured-output schema, instead.

41. **Echo framing** (BPBench, item 12): an instruction to begin with an exact quoted copy of the last 8 words, then read the word after the echo. It restores the conditional distribution. Discard positions where fewer than 16 of the 32 samples follow the format.

42. **Emulated fine-tuning** (E. Mitchell et al., 2023, https://arxiv.org/abs/2310.12962) and **proxy-tuning** (A. Liu et al., 2024, https://arxiv.org/abs/2401.08565) **[M]**: combine logits from a small base/chat pair with a large model to emulate large-base or large-chat behavior. They need logits, so they apply only to the open calibration side, for example as a sanity check on how the base-to-chat shift scales.

43. **Logprob availability on closed APIs (late 2026)** **[U: community reports, not official docs]**
    - OpenAI: GPT-5-series reasoning models reject or omit logprobs; GPT-5.1 and 5.2 reportedly return them only with reasoning effort "none", inconsistently. https://community.openai.com/t/gpt-5-2-logprobs-support-removed/1378114
    - Gemini: logprobs on 2.5 via Vertex; 3.x reportedly returns "logprobs is not supported for this model". https://discuss.ai.google.dev/t/missing-logprobs-support-in-the-newest-gemini-models-3-1-pro-3-6-flash-on-vertex-ai-and-ai-studio/176557
    - Anthropic: none.
    *Relevance*: where a closed model still gives output-token logprobs, for example GPT-5.x with reasoning "none" or Gemini 2.5, it is a free *within-closed-model* check. Compare the guessing-game ranks with the logprob ranks of the first guess.

---

## 4. Prior attempts to rank closed models by a perplexity-like quantity

44. **BPBench** (item 12) **[V]**: the only explicit attempt found. It tried completions resampling, plain chat and echo chat on Fable, Gemini Flash 3.6 and GPT-5.6 Sol, judged none reliable, and placed *no* closed models. It did not try a top-k guessing game.

45. **Compression Represents Intelligence Linearly**: Y. Huang, J. Zhang, Z. Shan, J. He, 2024 (COLM 2024). https://arxiv.org/abs/2404.09937 **[V-abs]**
    Across 31 open base models, BPC on fresh corpora correlates linearly with averaged benchmark scores (Pearson about −0.95). Open models only. A 2025 follow-up finds a log, not linear, relationship for code (https://arxiv.org/abs/2505.11441).
    *Relevance*: the motivation for the axis. Note it is base models only.

46. **Quantifying LLM Base Model Quality through Compression** (Danish/English/Python leaderboard): R. Larsen, 2025-04-29. https://rlrs.dk/blog/leaderboard.html **[V]**. Base models only, open models only, needs logprobs.

47. **Language Modeling Is Compression**: G. Delétang et al., 2023 (ICLR 2024). https://arxiv.org/abs/2309.10668 **[M]**. Open models (Chinchilla) as arithmetic-coding compressors. Background.

48. **UncheatableEval** (Jellyfish042, GitHub) **[M]**: a compression rate on fresh text (arXiv, news, GitHub), open models only.

49. **Sampling-based perplexity of Gemini and Claude**: a search snippet mentions a paper that computes "empirical log-perplexity" from sampled next-token frequencies for Gemini and Claude, which expose no logprobs **[U: I could not pin down which paper; possibly item 13 or a related provenance or attack paper]**.

No peer-reviewed work was found that ranks GPT, Claude and Gemini on a BPB-like axis without logprobs. The guessing-game approach appears novel.

---

## Design implications

1. **The rank code is a model-dependent upper bound, not an estimate of BPB.** For a guesser q, both the rank entropy H(R) and the cross-entropy CE(q) upper-bound the true entropy, but H(R) ≥ CE(q) roughly when q is sharp and well calibrated. A rank code charges every rank-1 hit −log P(R=1), even when the model was 99.9% sure. The gap therefore grows with sharpness and with the fraction of near-certain positions (Scheibner et al.; Kirk et al.). Base models (calibrated) and chat models (sharp, mode-collapsed) will have different gaps. Fit and *report* the mapping from guessing-game bits per byte to exact BPB separately for base and instruct models; a single line that fits base models may not transfer.
2. **Ask for probabilities, not just ranks, following Cover & King.** Have the model attach a probability to each of its 20 guesses (the Verbalized Sampling / Tian Top-K format). Score with a mixture code: verbalized probability, smoothed, with escape mass for misses. Report both the rank code and the probability code. This is the closest black-box analogue to log-loss. Verbalized probabilities are overconfident (Xiong et al.), so smooth or temperature-scale them with parameters fit on open models.
3. **Fix the conditioning first; it is the bias that defeated BPBench.** A plain "list the next words" prompt conditions on "start of an assistant reply" (+2.7 bits/word, model-dependent). Use the echo instruction or a structured-output field that repeats the last words. On current Claude, prefill returns 400.
4. **Validate the shape, not only the mean.** BPBench failed because the bias differed by model. Show that the residuals of the guessing-game-to-BPB map are small and uncorrelated with model family, base versus instruct, and size on held-out open models. Report tiers with CIs, not fine ranks.
5. **Define the claim operationally.** Song, Hu & Mahowald and Hu & Levy show that prompted predictions are not the model's string probabilities. Call the closed-model number "guessing-game bits per byte" and present the open-model calibration as evidence that it tracks BPB. Do not present it as "Claude's BPB".
6. **Escape coding decides tail sensitivity.** The information that separates models lives in low-probability tokens (Min-K%). Spell-out escape costs should come from a fixed external character model, identical for every model, so that a weak model is not flattered (compare BPBench's 6-bit floor).
7. **Hold context length and reasoning fixed.** Entropy keeps falling up to about 10^4 characters of context. Use the same context window for open and closed runs, and disable or verify-off reasoning, as BPBench does through telemetry.
8. **Run contamination checks on the fresh corpus.** Name cloze, DE-COP or TS-Guessing style probes are cheap on closed models. Exact-guess rates far above the open-model trend would flag memorization.
