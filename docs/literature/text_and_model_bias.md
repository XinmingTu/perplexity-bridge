# Biases in the evaluation text and in the models: annotated literature list

Survey date: 2026-11-22. Study context: bits-per-byte (BPB) on fresh UK Commons Hansard and Canadian Commons Hansard (Oct-Nov 2026), plus a 2020 control window.

Verification marks:
- **[V]**: I opened the primary source, or a fetched page that quotes it, this session.
- **[S]**: secondary coverage only (news, aggregator, blog). Check the primary source before citing.
- **[M]**: cited from memory or from search snippets only. The work is well known, but I did not re-read it this session.
- **[U]**: unverified or conflicting claims.

---

## 1. How much LLM-written text there is after 2023

### 1a. Parliaments and legislative speech (directly relevant)

1. **The Economist, AI-written speeches in English-speaking legislatures (Pangram analysis)**, 23 Sep 2026. The Economist paywall blocked access, so these figures come from coverage in the Globe and Mail (Marie Woolf, 24 and 26 Sep 2026), Canadian Press/CTV and others. [S]
   - URLs: https://www.theglobeandmail.com/politics/article-one-fifth-mp-speeches-house-of-commons-written-ai-help-analysis-finds/ ; https://www.theglobeandmail.com/politics/article-international-trade-minister-among-mps-who-used-ai-to-help-write/
   - Method: Hansard speeches, questions and other interventions of **100 words or more**, run through the Pangram detector, for the year from **Sep 2025**.
   - Canada: about **20% of English words** spoken in the Commons were AI-assisted, second only to Australia (about 21%). **Question Period rarely used AI. Private members' business was about 20% AI-assisted.** Eleven English-speaking MPs had at least half of their words flagged; one MP had about 75%. By party, NDP > Liberal > Conservative.
   - UK: about **1 in 10 words** in Commons and Lords debates was flagged, and more than half of MPs rarely or never used AI. Coverage describes UK use as near zero in 2024 and rising sharply since.
   - The coverage reports no false-positive control on pre-2022 Hansard. The Globe's lede says "one-fifth of the words" while its body says "20% of contributions" [U].
   - **Implication:** both of the study's fresh corpora contain an AI-assisted share in the high single digits to about 20%, and that share is concentrated in prepared speech rather than spontaneous exchanges. This is the single most important finding for the design.

2. **Queen's Park Observer, "1 in 7 words at Queen's Park flagged as AI"**, 12 Nov 2026. https://www.qpobserver.ca/p/exclusive-1-in-7-words-at-queens [V]
   - Data: Ontario legislature Hansard, about 400k words from 27 Oct to 5 Nov 2026, run through Pangram.
   - Results: **14.8%** of words were flagged as AI or likely AI. **Question Period scored about 4%.** A **fall-2022 baseline of 7 sitting days scored 0%.**
   - **Implication:** an independent replication on a Canadian Westminster-style Hansard, in the same weeks as the study's fresh window. It shows the same split (spontaneous exchanges low, set speeches high) and gives a usable pre-ChatGPT null baseline. The study's 2020 control can serve the same purpose.

3. **Suvanto, McGlinchey, Barclay, Wahde, "Detecting undisclosed LLM-generated content in parliamentary texts"**, Chalmers and Edinburgh Napier, arXiv 2606.14209, Jun 2026. https://arxiv.org/abs/2606.14209 [V]
   - Data: **UK Hansard written statements** (not debate transcripts) and Swedish Riksdag motions, 2021 to Apr 2026.
   - Method: ICON, an interpretable n-gram linear classifier. It was trained on pre-2022 texts and on Gemini and GPT-5-mini rewrites of them.
   - UK texts with at least one flagged paragraph: 2.35% (2021), 4.44% (2023), 8.09% (2025), **15.45% (2026, partial year)**. Whole texts flagged in 2026: 2.12%.
   - The base false-positive rate is about 6% at paragraph level without a threshold, and close to 0 with a confidence threshold. None of the flagged texts disclosed AI use.
   - The authors deliberately avoided spoken debates.
   - **Implication:** confirms a steep rise in 2025-26 in UK parliamentary writing. Written statements and written answers should be excluded or analysed separately.

4. **Apple Tokamak (pseudonym), "MPs are almost certainly using ChatGPT to generate Commons speeches"**, Pimlico Journal (Substack), 1 Sep 2025. https://www.pimlicojournal.co.uk/p/mps-are-almost-certainly-using-chatgpt [V]
   - Method: Z-scores of AI-associated words and phrases in Hansard from 2007 to 2025.
   - Results: "I rise to speak" reached Z = 3.60 in 2025. "Delve" was **not** significant (Z about 0).
   - The author says there is no proof. The phrase may also spread through new MPs copying it.
   - **Implication:** lexical tells such as "delve" are weak in speech. Use detector-based or distributional estimates instead of word lists.

5. **James Titcomb, The Telegraph, "ChatGPT triggers surge in MPs using…"**, 11 Sep 2025, syndicated on Yahoo. https://www.yahoo.com/news/articles/chatgpt-triggers-surge-mps-using-132959577.html [V]
   - "I rise to speak" appeared 635 times in 2025 so far, against 231 in all of 2024. PoliticsHome gives different counts (601 vs 131, both Houses) [S].
   - Tom Tugendhat MP complained in the chamber that members' speeches sounded like ChatGPT.
   - **Implication:** a datable marker that can serve as a cheap sanity check on the share of AI-assisted speech by period.

6. **Luke Evans MP reads a ChatGPT-written speech in the Commons**, ITV/PA, 20 Dec 2022. https://itv.com/news/central/2022-12-20/tory-mp-reads-out-ai-generated-speech-in-commons-first [S]
   - **Implication:** the first known AI-written Commons speech, in Dec 2022. A 2020 control window is safely before any such use.

7. **Canadian House of Commons institutional AI use**. [S]/[U]
   - The Hub, 24 Oct 2026 (https://thehub.ca/2026/10/24/house-officials-tell-committee-ai-strategy-unpublished-and-u-s-courts-could-reach-commons-data/) reports that the House **"has been using AI to produce a first draft of Hansard for more than a year"**, and that a Copilot pilot is running for MPs chosen by the party whips.
   - An IPU use case from Jul 2024 describes ASR drafts that staff then edit (https://www.ipu.org/ai-use-cases/automated-speech-recognition-support-transcription-parliamentary-proceedings).
   - UK Hansard also uses ASR "to support transcription" and is trialling AI tools (IPU use cases; parallelparliament written question).
   - The sources do not say whether this is ASR only or also LLM post-editing [U].
   - **Implication:** the transcription pipeline itself changed between 2020 and 2026. Part of any shift in BPB between the windows may come from how the record is produced, not from what MPs said.

8. **Simard, Leal-Wyss, Bernier-Colborne, Knowles (NRC Canada), "A Longitudinal Study of the Adoption of Specialized MT Systems in Canadian Parliamentary Translation"**, EAMT 2026. https://aclanthology.org/2026.eamt-2.32.pdf [V]
   - **Leal-Wyss et al., "All those in favour will please say yea…"**, EAMT 2026, is a companion user study. https://aclanthology.org/2026.eamt-1.36.pdf [V]
   - Since 2023, Parliament of Canada translators have optionally used the Hawkeye NMT system, built on Sockeye and trained on parliamentary text.
   - Use rose from 16% of texts (May 2023) to about **50-60% (fall 2024 to Dec 2025)**. Use is **56.4% for the Debates**, 50.0% for translation into English and 57.8% into French.
   - **Implication:** the English Hansard renders interventions made in French as translations, and about half of those are now MT post-edits. **Keep only interventions originally spoken in English.** This is a second kind of machine-written text that is distinct from LLM drafting by MPs.

9. **UK Parliament, AI guidance for Members**, 2025. https://webtest.parliament.uk/mps-lords-and-offices/offices/bicameral/ai-guidance-for-members [S]
   - **Canadian House of Commons GenAI resources for MPs**, May 2025, from the IPU innovation tracker [S].
   - Neither house bans AI drafting.

### 1b. Other domains: background on base rates

10. **Liang et al. (Stanford), "Quantifying large language model usage in scientific papers"**, *Nature Human Behaviour*, 4 Aug 2025, DOI 10.1038/s41562-025-02273-8. Preprint: "Mapping the Increasing Use of LLMs in Scientific Papers", arXiv 2404.01268. [V via search]
    - Data: 1.12M papers. Method: the distributional framework described in item 12.
    - The estimated share of LLM-modified sentences in CS abstracts reached **22.5%** by Sep 2024.
    - **Implication:** the standard corpus-level estimator. It works on populations, not on individual documents.

11. **Liang et al., "The Widespread Adoption of Large Language Model-Assisted Writing Across Society"**, *Patterns* 2025, arXiv 2502.09747. https://arxiv.org/abs/2502.09747 [S]
    - Corporate press releases: up to about 24%. **UN press releases: about 3% (early 2023) rising to about 14% (late 2024).** Consumer complaints: about 18%. Job postings from small firms: about 10%.
    - Adoption plateaued after the initial jump.
    - **Implication:** institutional and government writing reached 10-25% by 2024. Prepared parliamentary speech should be expected in that range.

12. **Liang, Izzo, Zhang, …, Zou, "Monitoring AI-Modified Content at Scale"**, ICML 2024, arXiv 2403.07183. https://arxiv.org/abs/2403.07183 [V abstract]
    - Method: maximum-likelihood estimation of a mixture fraction α over word-occurrence distributions, using human and AI reference corpora.
    - Result: 6.5-16.9% of ML-conference peer reviews were substantially LLM-modified.
    - **Implication:** this estimator can be run per stratum of Hansard, for example by debate type, with the 2020 window serving as the human reference corpus.

13. **Kobak, González-Márquez, Horvát, Lause, "Delving into LLM-assisted writing in biomedical publications through excess vocabulary"**, *Science Advances* 11(27), Jul 2025, arXiv 2406.07016. [V via search]
    - At least 13.5% of 2024 PubMed abstracts were LLM-processed, estimated by word frequencies relative to a counterfactual extrapolation from pre-LLM years.
    - **Implication:** a model-free "excess words" method. It is confounded where humans themselves adopt LLM vocabulary (see items 15-16).

14. **Russell et al. (UMD) with Pangram, "AI use in American newspapers is widespread, uneven, and rarely disclosed"**, arXiv 2510.18774, ACL 2026. https://arxiv.org/abs/2510.18774 [S]
    - Data: 186k articles from summer 2025. **9.1%** were flagged as AI or mixed. The rate was **1.7% at papers with circulation over 100k**. Opinion pages at the NYT, WaPo and WSJ were flagged 6.4 times more often than news articles.
    - **Implication:** if news is added as a corpus later, large-outlet news reporting is among the cleanest sources of recent English.

15. **Graphite (Surfer detector), AI share of new web articles**, reported by Axios, Oct 2025 and May 2026. https://axios.com/2026/05/15/human-vs-ai-written-articles [S]
    - About 50% of new English web articles since late 2024 are mainly AI-generated. The detector has about 4.2% false positives.
    - **Implication:** generic fresh web text is the least trustworthy kind of "human" text.

16. **Yakura et al. (MPI for Human Development), "Empirical evidence of Large Language Model's influence on human spoken communication"**, arXiv 2409.01754 (v3, Jul 2025). [V via search]
    - **Anderson, Galpin, Juzek (Florida State), "Model Misalignment and Language Change: Traces of AI-Associated Language in Unscripted Spoken English"**, arXiv 2508.00238, AIES 2025. [S]
    - Both find abrupt rises in "delve", "meticulous", "boast" and similar words in unscripted talks and podcasts after 2022.
    - **Implication:** human speech itself has drifted toward LLM vocabulary. (a) Word-frequency detectors produce false positives on genuine post-2022 speech. (b) Even "fully human" 2026 speech may be slightly easier for LLMs to predict than 2020 speech. This is principle (1) in a soft form.

---

## 2. Detecting AI-written text at corpus level, and false positives on speech

17. **Liang et al. 2024**, the distributional MLE estimator (item 12). It works on populations and is robust because it never classifies individual documents. Its known weakness is sensitivity to the choice of AI reference corpus (which model and which prompt). [M]

18. **Hans et al., "Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text"**, ICML 2024, arXiv 2401.12070. [M]
    - The score is the ratio of perplexity to cross-perplexity between two related models, such as Falcon-7B and Falcon-7B-instruct. The authors claim a 0.01% false-positive rate.
    - **Mozilla Foundation evaluation**, 2024 (https://www.mozillafoundation.org/en/blog/who-wrote-that-evaluating-tools-to-detect-ai-generated-text/): the measured false-positive rate was about **0.7%**, roughly 70 times the authors' figure. [S]

19. **Bao et al., "Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature"**, ICLR 2024, arXiv 2310.05130. [M]
    - Both Fast-DetectGPT and Binoculars collapse when the generating model is prompted to avoid AI phrasing (jack-ford/AI-Detector-Evaluation, GitHub). [S]
    - **Implication:** zero-shot detectors are themselves likelihood-based. Using one to filter a BPB benchmark is circular: it removes exactly the text that LMs find easy. Prefer a classifier detector (Pangram, ICON) or the distributional estimator for stratification, and say so.

20. **Pangram Labs technical reports** (2024, and v4 in 2026), plus **Jabarian and Imas, "Artificial Writing and Automated Detection"**, UChicago BFI working paper 2025-116 / NBER w34223. https://bfi.uchicago.edu/wp-content/uploads/2025/10/Artificial-Writing-and-Automated-Detection.pdf [S]
    - The vendor reports a false-positive rate of about 0.004% (v4). The BFI study finds false positives and false negatives near zero on medium and long passages, but higher on short ones.
    - Critiques: LessWrong, "We underestimate the weaknesses of Pangram"; Buttondown/MAIHT3K, "Pangram is not an oracle". Both say the definition of a false positive for mixed text is underspecified. [S]
    - **No published evaluation of Pangram on edited parliamentary transcripts was found.** The best evidence available is the 0% result on the 2022 Ontario baseline (item 2).

21. **Liang, Yuksekgonul, Mao, Wu, Zou, "GPT detectors are biased against non-native English writers"**, *Patterns* 2023, arXiv 2304.02819. [M]
    - Seven detectors misclassified about 61% of TOEFL essays as AI-written.
    - **Implication:** simple, formulaic or low-perplexity human text triggers detectors. Formulaic parliamentary language, such as procedural set phrases and read-out answers, is at risk. **Calibrate any detector on the study's own 2020 control window and report its false-positive rate there.**

22. **Spoken transcripts.** No study was found that measures false-positive rates of AI-text detectors on speech transcripts, whether ASR output or edited Hansard. This is a gap the study should acknowledge. [U]
    - Indirect evidence: detectors are unreliable below about 80-100 words (sloptotal README; the Economist's 100-word floor). Edited Hansard is closer to written prose than raw ASR is (section 5).

---

## 3. Self-preference, self-recognition and synthetic-data effects

23. **Panickssery, Bowman, Feng, "LLM Evaluators Recognize and Favor Their Own Generations"**, NeurIPS 2024, arXiv 2404.13076. [M]
    - Self-recognition ability correlates linearly with self-preference. Fine-tuning that strengthens self-recognition increases self-preference.

24. **Wataoka, Takahashi, Ri (SB Intuitions), "Self-Preference Bias in LLM-as-a-Judge"**, NeurIPS 2024 SafeGenAI workshop, arXiv 2410.21819. https://arxiv.org/abs/2410.21819 [V via search]
    - LLM judges rate **lower-perplexity** outputs higher than humans do, whether or not the judge wrote them. The authors conclude that self-preference is a familiarity, or perplexity, effect.
    - **Implication for BPB:** the mechanism that may drive judge bias is literally the study's metric. Text written by model family F will have low perplexity under F. **Principle (2) is therefore a mechanical effect on BPB, not only a hypothesis.**

25. **Chae et al., "Self- and Other-Labels Induce Bidirectional Bias in LLM Judges"**, Findings of EMNLP 2026, pp. 33466-33482, arXiv 2608.18091. https://aclanthology.org/2026.findings-emnlp.1571/ [V via search]
    - This is the EMNLP 2026 Findings critique. After controlling for quality and judge severity, most of the raw self-preference effect disappears. On originality, judges show self-deprecation.
    - It cites Roytburg et al. (2026), who attribute about 90% of measured self-preference to evaluator uncertainty, and Chen et al. (2025) on the quality confound. [S for the cited works]
    - **Implication:** self-preference in judging is weaker than claimed. That finding does not transfer to likelihood: a BPB advantage on a family's own text needs no judgment. Test it directly with a model-family × AI-flag interaction (see design notes).

26. **Sun, Yin, Xu, Kolter, Liu, "Idiosyncrasies in Large Language Models"**, ICML 2025, arXiv 2502.12150. [V via search]
    - A fine-tuned embedding classifier identifies which of ChatGPT, Claude, Grok, Gemini or DeepSeek produced a text with **97.1%** accuracy. The signal is rooted in word-level distributions and survives paraphrasing and translation by another LLM.
    - **Implication:** AI-assisted Hansard text carries family-specific fingerprints, so a contamination advantage would differ by family. ChatGPT is likely the dominant tool for MPs [U]: OpenAI-family models may gain most, and Microsoft Copilot runs on GPT models.

27. **Laurito et al., "AI–AI bias: Large language models favor communications generated by large language models"**, PNAS 2025, arXiv 2407.12856. [M]
    - LLMs choose items described by LLM-written text over human-written text.

28. **Dai et al., "Neural Retrievers are Biased Towards LLM-Generated Content"** (source bias), KDD 2024, arXiv 2310.20501. [V via search]
    - **Wang et al., "Perplexity Trap: PLM-Based Retrievers Overrate Low Perplexity Documents"**, ICLR 2025, arXiv 2503.08684. [V via search]
    - LLM-generated documents have lower perplexity and are ranked higher. A 2026 L3S paper (arXiv 2602.10833) argues that supervised fine-tuning, not perplexity alone, drives the effect. [S]
    - **Implication:** independent evidence that LLM-written text is generically lower-perplexity across models, not only under its own family.

29. **Self-perplexity is not always lower.** Llama3-8B-Instruct self-recognition study, arXiv 2410.02064, and an LLM self-recognition activation-signature study, arXiv 2606.06315. [S]
    - Self-written text usually has lower perplexity under the model that wrote it, but not always: for Ministral-3-8B-Instruct, human summaries had lower perplexity.
    - **Implication:** measure the effect rather than assume it.

30. **Mireshghallah et al., "Smaller Language Models are Better Zero-shot Machine-Generated Text Detectors"**, EACL 2024, arXiv 2305.09859. [V via search]
    - Small models pick up the likelihood "curvature" of generations from other models. Shared training data or size between detector and generator matters little.
    - **Implication:** LLM text is easy for LMs in general. The family-specific component is a second-order effect on top of that.

31. **Shumailov et al., "AI models collapse when trained on recursively generated data"**, *Nature* 631, 2024. [V via search]
    - Recursive training on a model's own outputs loses the tails of the distribution. In the OPT-125m experiments, perplexity on the original human test data rose over generations [M].
    - **Gerstgrasser et al., "Is Model Collapse Inevitable?"**, arXiv 2404.01413, and **Kazdan et al., "Collapse or Thrive?"**, arXiv 2410.16713: collapse is avoided when synthetic data accumulates alongside real data instead of replacing it. Adding synthetic data can still raise loss on real data when real data is plentiful. [V via search]
    - **Implication:** models trained heavily on synthetic or distilled data may be worse on the tails of human text (rare names, idioms, heckles) while scoring better on AI-like text. This argues for reporting BPB on low-AI strata separately.

---

## 4. Training-data cutoffs and contamination

### 4a. Published cutoffs as of Nov 2026

| Model | Training / knowledge cutoff | Source | Mark |
|---|---|---|---|
| Claude Fable 5.2, Opus 5.5, Sonnet 5.5, Haiku 5.5 | **Jun 2026** (reliable and training) | platform.claude.com models overview | [V] |
| Claude Sonnet 5 | Jan 2026 | Anthropic support article (search) | [S] |
| GPT-6 Sol (API) | **20 Apr 2026** | developers.openai.com model page (search) | [V via search] |
| GPT-6.1 Sol | **30 Apr 2026** | developers.openai.com model page (search) | [V via search] |
| GPT-6 Luna (API) | 18 May 2026 | rankscope.ai aggregator | [S] |
| GPT-6 Astra | 30 Apr 2026 | rankscope.ai | [S] |
| GPT-5.6 Sol / Luna | 16 Feb 2026 | rankscope.ai | [S] |
| GPT-5.5 | 1 Dec 2025 | aggregators citing OpenAI | [S] |
| GPT-5.2 to 5.4 | 31 Aug 2025 | aggregators citing OpenAI | [S] |
| ChatGPT "October" GPT-6 Sol / Luna | not published | rankscope.ai | [S] |
| Gemini 3 Pro, 3.1 Pro, 3.x Flash | **Jan 2025** | ai.google.dev; Gemini 3 Pro model card | [V via search] |
| Gemini 3.5 Pro | not found | — | [U] |
| DeepSeek V4 (Flash) | about May 2025 from the model card description; a behavioural audit suggests about Jul 2025 | arXiv 2611.14487, Table 3 | [S] |
| Qwen 3.5 / 3.6 / 3.7 | not published (year only); about Dec 2024 estimated for Qwen3.6-35B-A3B | arXiv 2611.14487; metehan.ai | [U] |
| Kimi K2 / K2.6 | Dec 2024 to Apr 2025 (inferred); K3 (Jul 2026) not published | aggregators | [U] |
| GLM-4.x / GLM-5 (Feb 2026) | not published | aggregators | [U] |
| Llama 4 | Aug 2024 | HF model cards via aggregator | [S] |
| Grok 4.x | Nov 2024 | docs.x.ai via aggregator | [S] |

**Implication:** every documented cutoff is on or before **Jun 2026**, so an Oct-Nov 2026 window is post-cutoff for all listed models. Cutoffs for Chinese open models are mostly unpublished, and some served chat products (the ChatGPT "October" versions) may have later or undisclosed data. Prefer API snapshots with documented cutoffs, and record the snapshot ID used.

### 4b. Checking cutoffs empirically

32. **Cheng et al., "Dated Data: Tracing Knowledge Cutoffs in Large Language Models"**, COLM 2024, arXiv 2403.12958. [V via search]
    - The **effective** cutoff is located at the minimum of perplexity across dated versions of a resource. It often differs sharply from the reported cutoff, because old pages reappear in new Common Crawl dumps and deduplication misses near-duplicates.
    - **Implication:** the study can apply this directly. Compute monthly BPB on Hansard from Jan 2025 to Nov 2026 for each model. A rise after month m locates that model's effective cutoff, and the result doubles as a contamination check.

33. **Li, Wan, Hou, Deng, Yuan, "FinCutBench: A Continually Updated Benchmark for Measuring the Temporal Boundaries of LLM Financial Knowledge"**, arXiv 2611.11588, Nov 2026. [V abstract]
    - Uses dated multiple-choice facts and a fitted monthly knowledge profile to estimate a "boundary" month. GPT-6 Sol and Luna are among the models evaluated.
    - **Implication:** an alternative check based on knowledge rather than likelihood.

34. **Duan et al., "Do Membership Inference Attacks Work on Large Language Models?"**, COLM 2024, arXiv 2402.07841. **Meeus et al., "SoK: Membership Inference Attacks on LLMs are Rushing Nowhere"**, arXiv 2406.17975. [V via search]
    - Membership inference is near random. Apparent successes often come from **temporal distribution shift**: members and non-members drawn from different periods.
    - **Implication:** this confound applies to the study's own 2020-vs-2026 contrast. A BPB gap between windows mixes memorization, topical and temporal drift (new names, bills, events), AI-share drift and changes in the transcription pipeline. A bag-of-words "which window" classifier gives a cheap estimate of how far the windows differ.

35. **Lazaridou et al., "Mind the Gap: Assessing Temporal Generalization in Neural Language Models"**, NeurIPS 2021, arXiv 2102.01951. [V via search]
    - Perplexity worsens steadily on future data, and fastest in **politics** and sports. A model trained up to the test period scores up to 16% better in perplexity.
    - **Implication:** expect a substantial and model-dependent penalty on fresh 2026 politics text that has nothing to do with AI text. Models with later cutoffs will have an advantage (Jun 2026 Claude against Jan 2025 Gemini). **Report each model's cutoff gap beside its score, or add cutoff as a covariate.**

36. **Li et al., "Evaluating Large Language Models for Generalization and Robustness via Data Compression"**, arXiv 2402.00861, 2024. **"Uncheatable Eval: Dynamic Compression-Based Evaluation of Language Models"**, arXiv 2609.27510, 2026; GitHub Jellyfish042/uncheatable_eval. [V via search]
    - Compression-rate evaluation on data before and after each model's cutoff. The 2024 paper found models generalize well on arXiv but degrade on news and code.
    - **Implication:** the closest prior work on the study's axis. Cite it, and differentiate the study by its corpus choice and its control for AI-text share.

### 4c. Is Hansard in training corpora?

37. **Kandpal et al. (EleutherAI and others), "The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text"**, NeurIPS 2025 D&B, arXiv 2506.05209. [V via search]
    - The collection includes a **`common-pile/uk_hansard`** subset (HF collection, updated 6 Jun 2025), and Comma v0.1 was trained on it.
    - **Implication:** UK Hansard is an explicit pretraining source for at least some open models, and almost certainly enters others through Common Crawl and mirrors.

38. **Crawlability**. [V, checked this session]
    - openparliament.ca robots.txt disallows only `/search`.
    - ourcommons.ca disallows only search, embed and error paths.
    - TheyWorkForYou (a UK Hansard mirror) disallows only `/pwdata`, `/user`, `/vote` and `/search`.
    - hansard.parliament.uk sits behind a Cloudflare challenge, which blocked my scripted fetch. The official UK site is partly crawler-hostile, but its mirrors are open.
    - The Canadian Hansard has been a standard English-French MT corpus since the IBM work around 1990 (Brown et al.; Gale and Church 1991), and is in OPUS [M].
    - The Pile includes EuroParl (European Parliament), not the UK or Canadian Hansard [M].
    - **Implication:** the 2020 control is very likely **in the training data of every model**, possibly with repeated copies. It measures in-distribution and possibly memorized performance, not "human text the model hasn't seen". It is a good null for AI detectors, but not a clean baseline for compression. Check verbatim memorization, for example with continuation-match on 2020 passages.

---

## 5. Spoken and written text in LM evaluation

39. **The UK Hansard editorial standard.** Erskine May and Hansard's "About" page describe a "substantially verbatim" report: "repetitions and redundancies omitted and obvious mistakes corrected", but with nothing left out that adds to the meaning. The standard comes from a select committee of 1893 or 1907; the date conflicts between sources [U]. https://hansard.parliament.uk/about [S]

40. **The Canadian Hansard editorial standard.** House of Commons *Procedure and Practice* (3rd ed.) calls it "the report in extenso … with due regard to necessary grammatical or editorial changes", and the House calls it an "edited, revised, and corrected transcript". Speaker Milliken ruled that it is not a verbatim transcript. Members may suggest corrections but may not make material changes. https://www.ourcommons.ca/procedure/procedure-and-practice-3/ch_24_1-e.html ; https://ourcommons.ca/procedure/speakers-decisions/peter-milliken/ch02/decision11-e.html [V via search]

41. **Mollin, "The Hansard hazard: gauging the accuracy of British parliamentary transcripts"**, *Corpora* 2(2):187-210, 2007. [S]
    - Hansard makes significant lexical and grammatical changes and removes performance features such as disfluencies, false starts and the force of interjections. Slembrouck (1992) makes a similar argument about the "written construction of spoken discourse" [M].
    - **Implication:** Hansard is closer to written prose than to speech. Expect BPB between that of conversational transcripts (higher) and edited news (lower).
    - Editing practice also varies by era and by editor, so editing is a time-varying confound between 2020 and 2026. Editors might also tidy AI-ish text, or leave it untouched. No study addresses this [U].

42. **Spoken versus written text and perplexity.** I found no modern study of LLM perplexity on spontaneous-speech transcripts versus written text. [U]
    - Older work finds that disfluent transcripts have higher n-gram perplexity, and that removing disfluencies brings them closer to written text. One example is "Improving spoken language translation by automatic disfluency removal" (ResearchGate); a 2025 punctuation study (arXiv 2506.03793) shows that performance gaps persist on spontaneous speech. [S]
    - **Implication:** as a reference point, raw ASR or unedited parliamentary video captions would be a "more spoken" corpus with a different exposure to AI. MP-drafted text that is read aloud is still prepared, written text after Hansard's editing.

43. **ParliaBench**, arXiv 2511.08247, 2025. A benchmark for LLM-generated UK parliamentary speech. It is a potential source of AI-written parliamentary reference text for calibrating the distributional estimator. [S]

---

## Design notes that follow from the literature

1. **Stratify the fresh window by proceeding type and trust spontaneous exchanges most.** Question Period / PMQs, oral-question supplementaries, urgent-question exchanges, interventions and points of order show low AI use (Canada about 4% QP in Ontario; "rarely" in the Economist analysis). Prepared material is flagged most: private members' business, adjournment and Westminster Hall debates, SO31 Members' statements, maiden speeches, and ministerial opening speeches read from briefs. Written statements and answers should be dropped or kept as a separate high-AI stratum (UK 15% with at least one flagged paragraph in 2026).
2. **Canada: keep only interventions originally spoken in English.** About half of French-to-English Debates translation is NMT post-edited (Simard et al. 2026). A per-statement language field should exist in the ourcommons.ca XML and the openparliament.ca data [U, check the field name].
3. **Estimate the AI share inside the study, and model it rather than only filtering it out.**
   - Run the Liang distributional estimator per stratum, with 2020 as the human reference. Optionally also run a classifier detector such as Pangram at speech level with a 100-word floor.
   - Calibrate false positives on the 2020 window.
   - Avoid filtering with Binoculars or Fast-DetectGPT: both are likelihood-based, which makes the filter circular for a BPB study.
4. **Test principle (2) directly.** Use per-speech BPB with a model-family × AI-flag interaction. If OpenAI-family models gain more on flagged speeches than other families do, contamination by their own style is shifting the ranking. Report rankings on unflagged, spontaneous-only text as the headline, and on everything as a secondary result.
5. **Read the 2020 control as "seen, human, pre-AI", not "unseen human".** It is in training data (Common Pile uk_hansard, crawlable mirrors), it predates ASR and NMT in the Hansard pipeline, and it differs in topic. The 2020-vs-2026 gap therefore conflates memorization, temporal drift, AI share and pipeline change (Duan et al. 2024; Lazaridou et al. 2021). A cleaner within-2026 contrast is low-AI against high-AI strata. Monthly BPB from Jan 2025 to Nov 2026 also gives effective cutoffs (Cheng et al. 2024).
6. **Report each model's cutoff gap.** Documented cutoffs range from Jan 2025 (Gemini 3.x) to Jun 2026 (Claude 5.5). Politics text punishes staleness most (Lazaridou et al. 2021), so a ranking on fresh politics text partly reflects cutoff recency.
7. **Treat human vocabulary drift as a floor.** Post-2022 human speech has itself moved toward LLM vocabulary (Yakura et al.; Juzek et al.). Even the "human" 2026 stratum will be slightly LLM-like. No corpus fully separates this, so state it as a limitation.
