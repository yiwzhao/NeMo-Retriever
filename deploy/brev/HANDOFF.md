# AIDP × TensorMesh CacheBlend — Development Handoff

This document is the single source of truth for continuing development on this project.
It is committed to GitHub so it survives instance deletion and session loss.

---

## Project Goal

One-click Brev Launchable that demonstrates **NeMo Retriever + LMCache CacheBlend**
side-by-side against a vLLM APC baseline, with live performance metrics.

Core narrative: APC handles contiguous prefix reuse; CacheBlend adds non-prefix KV
reuse for retrieved document chunks that appear at different positions across requests.
They are **complementary, not competing**.

---

## Architecture

```
Enterprise Documents (T2-RAGBench FinQA, 299 PDFs)
         │
         ▼
NeMo Retriever (k3s, port 7670)
  OCR NIM (nemotron-ocr-v2)
  Layout NIM (nemotron-page-elements-v3)
  Table NIM (nemotron-table-structure-v1)
  Embed NIM (llama-nemotron-embed-vl-1b-v2)
  VectorDB (LanceDB)
         │
    shared retrieved chunks
    ┌────┴────┐
    ▼         ▼
Baseline      Optimized
port 8001     port 8002
vLLM + APC    vLLM + APC + LMCache CacheBlend
    │               │
    └──────┬─────────┘
           ▼
   Deploy UI (port 8000)
   /compare  — dual-panel streaming comparison
   /benchmark — batch statistical evaluation
```

---

## Current State (commit f5a97ddb, 2026-07-23)

### Services
| Service | Port | Notes |
|---------|------|-------|
| NeMo Retriever | 7670 | k3s, all NIMs Running |
| vLLM Baseline | 8001 | meta-llama/Llama-3.1-8B-Instruct |
| vLLM CacheBlend | 8002 | + LMCache 0.5.1, blender initialized |
| Deploy UI | 8000 | INFERENCE_MODE=host |

### Brev URLs (may change if instance is recreated)
- Deploy UI: https://deploy-ui-gh0lyurwe.apps.run.brev.nvidia.com
- Compare: https://deploy-ui-gh0lyurwe.apps.run.brev.nvidia.com/compare
- JupyterLab: https://jupyter-gh0lyurwe.apps.run.brev.nvidia.com

### GitHub
- Fork: https://github.com/yiwzhao/NeMo-Retriever (branch: brev-launchable)
- Collaboration: https://github.com/Tensormesh-Collaboration/aidp-rag (branch: main)

---

## Key Technical Decisions & Why

### Why host pip install instead of NGC Dynamo container?
NGC key `nvapi-...` lacks Container Registry access for the `ai-dynamo` org → 401 on
`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1`. Host pip install bypasses NGC entirely.

### Why sequential vLLM startup?
Two Llama-3.1-8B instances loading simultaneously peak at ~38GB GPU + NIMs' 51GB = 89GB.
If both load at the same time, the instantaneous peak during KV cache allocation can
exceed 96GB and trigger k8s OOM eviction of NIM pods. Sequential startup (baseline READY
first, then CacheBlend) keeps peak below limit.

### Why VLLM_USE_FLASHINFER_SAMPLER=0?
Blackwell SM 12.x requires CUDA ≥ 12.9 for architecture detection; PyTorch ships with
CUDA 12.8. FlashInfer JIT compilation fails to detect the arch → empty arch list → abort.
Setting this env var switches vLLM to FLASH_ATTN backend (PyTorch native FlashAttention 2).

### Why patch vLLM model_runner.py?
vLLM 0.25.1 doesn't call `VLLMModelTracker.register_model()` after loading the model.
LMCache CacheBlend needs this to find the model instance during blending. Without it:
`ValueError: vllm model for vllm-instance not found`. The patch is 5 lines inserted
after `load_model()` in `vllm/v1/worker/gpu/model_runner.py`. `start-inference.sh`
applies it automatically (idempotent, grep-guarded) after every pip install.

### Why gpu-memory-utilization=0.20?
NIMs occupy ~51GB. Each vLLM gets 0.20 × 96GB = 19.2GB. Two vLLMs = 38.4GB.
51 + 38.4 = 89.4GB < 96GB. Raising to 0.25 would cause OOM.

### Why max-model-len=4096?
KV cache blocks scale with max_model_len. At 0.20 util, 4096 leaves ~3.2GB for KV cache.
Raising to 8192 would halve the available KV blocks, hurting concurrent throughput.

### Why LMCache connector name is LMCacheConnectorV1?
The vLLM `KVConnectorFactory` registry (in lmcache 0.5.x) registers under
`LMCacheConnectorV1`, not `LMCacheConnector`. Found by querying
`KVConnectorFactory._registry.keys()` at runtime.

### Why are metrics computed from counters instead of gauges?
vLLM 0.25 exposes only Prometheus counters (`prefix_cache_hits_total`,
`external_prefix_cache_hits_total`, etc.), not pre-computed rate gauges.
LMCache registers no separate `lmcache:*` Prometheus metrics — its KV activity
appears as `vllm:external_prefix_cache_*`. `get_vllm_metrics()` in inference.py
derives all ratios from these counters.

### Why HF cache on /ephemeral?
Brev instances have a larger `/ephemeral` disk partition separate from the root `/`.
Root disk fills up quickly (currently 93%). Model weights (16GB) must go to `/ephemeral`
to avoid disk-pressure taint on the k8s node. `app.py` creates the path and sets
`HUGGINGFACE_HUB_CACHE` before running start-inference.sh.

---

## Versions

| Component | Version |
|-----------|---------|
| vLLM | 0.25.1 |
| LMCache | 0.5.1 (git: 979719d7) |
| LLM Model | meta-llama/Llama-3.1-8B-Instruct |
| Embed NIM | llama-nemotron-embed-vl-1b-v2 |
| OCR NIM | nemotron-ocr-v2 |
| Dataset | T2-RAGBench @ adf7fe1541ac, FinQA dev (299 PDFs, 883 QA) |

---

## lmcache.yaml (runtime file, generated by start-inference.sh)

```yaml
chunk_size: 256        # tokens per KV cache chunk
local_cpu: true        # store KV states in CPU RAM
max_local_cpu_size: 20 # max 20GB CPU RAM for KV cache
enable_blending: true  # activate CacheBlend
blend_min_tokens: 128  # minimum tokens in a chunk to trigger blending
```

KV shape for Llama-3.1-8B: `(32 layers, 2 K/V, 256 tokens, 8 heads, 128 dim)` = 32KB per chunk.
20GB CPU RAM ≈ 640 cached chunks ≈ 163,840 tokens.

---

## Restart Command (after instance stop/start)

```bash
export HF_TOKEN="hf_..."   # your HuggingFace token (needs Llama-3.1-8B-Instruct access)
cd ~/NeMo-Retriever && git pull && ./deploy/brev/start-inference.sh 2>&1 | tee ~/inference.log
```

The script handles all 10 steps automatically:
0. Wait for k3s NIM pods (≥4 Running)
1. pip install vllm + lmcache into ~/inference-venv
2. Patch vLLM model_runner.py (idempotent)
3. GPU sanity check
4. Pre-download Llama-3.1-8B-Instruct to /ephemeral HF cache
5. Write ~/lmcache.yaml
6. Kill old vLLM processes
7. Start baseline (8001) → wait for READY
8. Start CacheBlend (8002) → wait for READY (sequential!)
9. Restart Deploy UI (8000) — webui thread auto-manages :7670 port-forward
10. Check vectordb → auto re-ingest T2-RAGBench if empty

---

## Rebuild from Scratch (after instance deletion)

1. Wait for Brev to run `setup.sh` automatically (~5 min → "script: Completed")
2. Open Deploy UI → enter NGC key + HF Token → click Deploy Launchable
3. Wait ~60-75 min (both phases run automatically, no SSH needed)

---

## Metrics Explanation

| Metric | Source | Formula |
|--------|--------|---------|
| TTFT | client timer | time from POST to first delta token |
| E2E | client timer | retrieval_ms + generation_ms |
| tok/s | client | token_count / gen_time_s |
| APC hit rate | vLLM /metrics | prefix_cache_hits_total / prefix_cache_queries_total |
| LMCache hit ratio | vLLM /metrics | external_prefix_cache_hits / external_prefix_cache_queries |
| Blend ratio | vLLM /metrics | external_prefix_cache_hits / prefix_cache_queries_total |
| GPU KV usage | vLLM /metrics | vllm:kv_cache_usage_perc |

All metrics are **cumulative since server start**, not per-request.

---

## Known Issues / Remaining Work

| Priority | Issue | Notes |
|----------|-------|-------|
| P1 | Blend ratio ~0.4% in demo | Need ≥50 queries with overlapping docs + ≥10 warmup. Select questions about the same company across years. |
| P1 | lmcache_hit_ratio not in benchmark aggregation | `_agg_path()` in inference.py collects blend_ratio but not hit_ratio. 2-line fix. |
| P1 | Cold/warm cache UI narrative | Demo should explicitly show first query (cold) vs subsequent (warm). No UI indicator exists yet. |
| P2 | Answer quality evaluation | `qa_eval()` in benchmark.py is implemented. Needs `NVIDIA_API_KEY` for Nemotron-49B scoring. |
| P3 | No-cache ablation path | Third vLLM without --enable-prefix-caching as lower bound. May not fit in 96GB. |
| P3 | GPU compute utilization in compare UI | Data collected in InferenceSampler but not displayed. |
| P3 | Disk at 93% | /ephemeral has 16GB free. pip/vllm compile cache + model weights are the main consumers. |

---

## Coding Conventions

- All code and documentation: **English only** (no Chinese in code/comments)
- Commit message suffix: `Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>`
- After each change: `bash -n` (shell) or `python3 -c "import ast; ast.parse(...)"` (Python) → commit → push all 3 remotes
- Push command:
  ```bash
  git push origin brev-launchable
  git push origin brev-launchable:main
  git push tensormesh brev-launchable:main
  ```

---

## FAQ Quick Reference

**Q: Why does CacheBlend add TTFT overhead instead of reducing it?**
With low blend ratio (<5%), every request incurs the CPU cache lookup + blending overhead
without finding anything to reuse. Only when blend ratio is 10%+ does TTFT start dropping.

**Q: Why p95 is unreliable with 10 queries?**
p95 of 10 samples = the 9th-slowest value. A single slow request dominates.
Run ≥50 queries for statistically meaningful p50/p95.

**Q: Where does `lmcache_hit_ratio` differ from `lmcache_blend_ratio`?**
- `hit_ratio` = ext_hits / ext_queries (of tokens sent to LMCache, how many were cached)
- `blend_ratio` = ext_hits / total_queries (of ALL prompt tokens, how many CacheBlend served)
blend_ratio is the headline demo metric; hit_ratio measures LMCache's internal efficiency.
