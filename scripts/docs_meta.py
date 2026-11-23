"""Model metadata for the whole-document charts: lab, label, approximate release date, total and active
parameters (billions; None if unpublished), instruct or base, and the results file stem.

Dates: Fireworks listing date for models scored there, Hugging Face repository creation date otherwise
(repositories can be created before a public release, so dates are approximate).
"""

MODELS = {
    # file stem: (lab, label, date, total B, active B, kind)
    "kimi-k3":            ("Moonshot", "Kimi K3", "2026-07-19", 2781, 104, "instruct"),
    "kimi-k3p1-pro":      ("Moonshot", "Kimi K3.1 Pro", "2026-11-02", 2780, 104, "instruct"),
    "moonshotai__Kimi-K2.6": ("Moonshot", "Kimi K2.6", "2026-04-14", 1027, 32, "instruct"),
    "qwen3p8-max":        ("Alibaba", "Qwen 3.8 (2.4T)", "2026-08-05", 2420, 95, "instruct"),
    "Qwen__Qwen3.5-397B-A17B": ("Alibaba", "Qwen 3.5 397B", "2026-02-16", 403, 17, "instruct"),
    "Qwen__Qwen3.8-27B":  ("Alibaba", "Qwen 3.8 27B", "2026-08-05", 27.8, 27.8, "instruct"),
    "Qwen__Qwen3.6-35B-A3B": ("Alibaba", "Qwen 3.6 35B-A3B", "2026-04-15", 36, 3, "instruct"),
    "Qwen__Qwen3.5-35B-A3B-Base": ("Alibaba", "Qwen 3.5 35B-A3B Base", "2026-02-24", 36, 3, "base"),
    "Qwen__Qwen3.5-9B":   ("Alibaba", "Qwen 3.5 9B", "2026-02-27", 9.7, 9.7, "instruct"),
    "Qwen__Qwen3.5-9B-Base": ("Alibaba", "Qwen 3.5 9B Base", "2026-02-26", 9.7, 9.7, "base"),
    "Qwen__Qwen3.5-4B":   ("Alibaba", "Qwen 3.5 4B", "2026-02-27", 4.7, 4.7, "instruct"),
    "Qwen__Qwen3-8B":     ("Alibaba", "Qwen 3 8B", "2025-04-27", 8.2, 8.2, "instruct"),
    "glm-5p3":            ("Zhipu", "GLM-5.3", "2026-08-28", 743, None, "instruct"),
    "glm-5p3-flash":      ("Zhipu", "GLM-5.3 Flash", "2026-08-26", 320, 18, "instruct"),
    "nemotron-3-ultra-nvfp4": ("NVIDIA", "Nemotron 3 Ultra", "2026-06-02", 549, 55, "instruct"),
    "nvidia__NVIDIA-Nemotron-3-Super-120B-A12B-BF16": ("NVIDIA", "Nemotron 3 Super", "2026-03-10", 124, 12, "instruct"),
    "nvidia__NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16": ("NVIDIA", "Nemotron 3.5 Super", "2026-10-12", 124, 12, "instruct"),
    "nemotron-lightning-3p5-30b-a3b": ("NVIDIA", "Nemotron Lightning 30B", "2026-08-07", 31.6, 3, "instruct"),
    "nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16": ("NVIDIA", "Nemotron Lightning 30B Base", "2026-08-05", 31.6, 3,
                                                                "base"),
    "minimax-m3":         ("MiniMax", "MiniMax M3", "2026-06-11", 428, 23, "instruct"),
    "minimax-m3p1-flash": ("MiniMax", "MiniMax M3.1 Flash", "2026-10-17", 428, 23, "instruct"),
    "openai__gpt-oss-120b": ("OpenAI", "gpt-oss-120b", "2025-08-04", 117, 5.1, "instruct"),
    "openai__gpt-oss-20b": ("OpenAI", "gpt-oss-20b", "2025-08-04", 21, 3.6, "instruct"),
    "thinkingmachines__Inkling": ("Thinking Machines", "Inkling 1", "2026-07-14", 952, None, "instruct"),
    "thinkingmachines__Inkling-2-Small": ("Thinking Machines", "Inkling 2 Small", "2026-11-04", 594, None, "instruct"),
    "thinkingmachines__Inkling-2-Mini": ("Thinking Machines", "Inkling 2 Mini", "2026-11-17", 26, None, "instruct"),
    "muse-spark-open":    ("Meta", "Muse Spark Open", "2026-10-14", 816, 42, "instruct"),
    "meta-models__Muse-Glimmer-30B": ("Meta", "Muse Glimmer 30B", "2026-08-09", 29.8, None, "instruct"),
    "google__gemma-4-31B-it": ("Google", "Gemma 4 31B", "2026-03-11", 31.3, 31.3, "instruct"),
    "google__gemma-4-31B": ("Google", "Gemma 4 31B Base", "2026-03-12", 31.3, 31.3, "base"),
    "google__gemma-4-26B-A4B-it": ("Google", "Gemma 4 26B-A4B", "2026-03-11", 25.8, 4, "instruct"),
    "google__gemma-4-26B-A4B": ("Google", "Gemma 4 26B-A4B Base", "2026-03-12", 25.8, 4, "base"),
    "google__gemma-4-12B-it": ("Google", "Gemma 4 12B", "2026-05-23", 12.0, 12.0, "instruct"),
    "google__gemma-4-12B": ("Google", "Gemma 4 12B Base", "2026-05-23", 12.0, 12.0, "base"),
}
# Scored but not plotted as separate models: a fine-tune of Kimi K3, and the BF16 copy of Nemotron 3 Ultra on
# Tinker (a cross-check of the Fireworks NVFP4 copy).
# Left out: Inkling Small (first generation, superseded by Inkling 2) scores 1.8x worse than its siblings on
# every document while continuing text normally; cause unknown.
CHECKS = {"ember-1": "kimi-k3", "nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": "nemotron-3-ultra-nvfp4"}
