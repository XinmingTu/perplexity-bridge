"""Whole-document exact log-probabilities on Modal with vLLM, for open models that no hosted API scores.

The tokenization happens locally (same tokenizer, with offsets, BOS first when the model has one); Modal only
returns the log-probability of every prompt token. Results go through cbridge.docscore.summarize, so they are
scored exactly like the Fireworks and Tinker runs.

    uv run modal run scripts/modal_vllm.py --models google/gemma-4-12B,google/gemma-4-12B-it
"""

import gzip
import json
import sys
from pathlib import Path

import modal

app = modal.App("compression-bridge-score")
image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm==0.33.0", "huggingface_hub[hf_transfer]")
image = image.env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
# Large FP8/MXFP4 models JIT-compile CUDA kernels at load time, so they need nvcc: use NVIDIA's CUDA 13.0 devel image
# (vLLM 0.33 ships torch built for CUDA 13.0).
big_image = (modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.12")
             .pip_install("vllm==0.33.0", "huggingface_hub[hf_transfer]")
             .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "CUDA_HOME": "/usr/local/cuda", "VLLM_USE_DEEP_GEMM": "0",
                   "VLLM_MOE_USE_DEEP_GEMM": "0"}))
cache = modal.Volume.from_name("compression-bridge-hf-cache", create_if_missing=True)
MAX_TOKENS = 32000


def _score(model: str, token_lists: list[list[int]], tp: int = 1) -> list[list[float]]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, max_model_len=MAX_TOKENS + 8, gpu_memory_utilization=0.88, enable_chunked_prefill=True,
              max_num_batched_tokens=4096, trust_remote_code=True, enable_prefix_caching=False, tensor_parallel_size=tp)
    cache.commit()
    params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    outs = llm.generate([{"prompt_token_ids": ids} for ids in token_lists], params)
    result = []
    for ids, out in zip(token_lists, outs, strict=True):
        lps = [None]
        for tok, pl in zip(ids[1:], out.prompt_logprobs[1:], strict=True):
            lps.append(pl[tok].logprob)
        result.append(lps)
    return result


@app.function(image=image, gpu="H200", volumes={"/root/.cache/huggingface": cache}, timeout=3 * 3600)
def score(model: str, token_lists: list[list[int]]) -> list[list[float]]:
    return _score(model, token_lists)


# For ~1T-parameter models (FP8 weights around 1 TB). The container stops when the call returns.
@app.function(image=big_image, gpu="B200:8", volumes={"/root/.cache/huggingface": cache}, timeout=int(2.5 * 3600))
def score_big(model: str, token_lists: list[list[int]]) -> list[list[float]]:
    return _score(model, token_lists, tp=8)


@app.local_entrypoint()
def main(models: str, limit: int = 0, big: bool = False):
    sys.path.insert(0, "src")
    from transformers import AutoTokenizer

    from cbridge.corpus import read_jsonl
    from cbridge.docscore import _clean_logprobs, _done, summarize

    docs = read_jsonl(Path("data/corpus/truth/fresh.jsonl"))
    docs = docs[:limit] if limit else docs
    jobs = []
    for model in models.split(","):
        out = Path("results/docs/truth/fresh") / (model.replace("/", "__") + ".jsonl.gz")
        todo = [d for d in docs if d.id not in _done(out)]
        if not todo:
            continue
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        lead = [tok.bos_token_id] if tok.bos_token_id is not None else []
        prepared = []
        for d in todo:
            text, labels = d.render()
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = list(enc["input_ids"]), [a for a, _ in enc["offset_mapping"]]
            keep = MAX_TOKENS - len(lead)
            stop = offs[keep] if len(ids) > keep else None
            prepared.append((d, text, labels, ids[:keep], offs[:keep], stop))
        jobs.append((model, out, lead, prepared))
    calls = [(model, [lead + p[3] for p in prepared]) for model, out, lead, prepared in jobs]
    for (model, out, lead, prepared), lps_all in zip(jobs, (score_big if big else score).starmap(calls, return_exceptions=True), strict=True):
        if isinstance(lps_all, BaseException):  # one model failing must not lose the others
            print(f"{model}: FAILED {str(lps_all)[:300]}")
            continue
        with gzip.open(out, "at") as f:
            for (d, text, labels, ids, offs, stop), lps in zip(prepared, lps_all, strict=True):
                lps = _clean_logprobs(d.id, lps[len(lead):] if lead else lps)
                if lps is None:
                    continue
                row = summarize(d, text, labels, offs, lps, stop)
                row.update(offsets=offs, logprobs=[round(x, 4) for x in lps])
                f.write(json.dumps(row) + "\n")
        print(f"{model}: {len(prepared)} documents written to {out}")
