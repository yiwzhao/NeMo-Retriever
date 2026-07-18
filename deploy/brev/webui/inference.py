# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
NeMo Retriever — inference layer module.

Provides streaming RAG over two inference paths (baseline APC vs CacheBlend)
and a batch benchmark harness that measures TTFT, E2E latency, throughput,
KV cache hit rate, and answer quality for both.

Consumed by app.py (FastAPI routes) and directly from the CLI.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import threading
import time
import uuid
from collections import Counter
from typing import Generator, Iterator

import requests

RETRIEVER_URL = os.environ.get("RETRIEVER_URL", "http://localhost:7670")
BASELINE_LLM_URL = os.environ.get("BASELINE_LLM_URL", "http://localhost:8001")
CACHEBLEND_LLM_URL = os.environ.get("CACHEBLEND_LLM_URL", "http://localhost:8002")
LLM_MODEL = os.environ.get("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
LLM_TOP_K = int(os.environ.get("LLM_TOP_K", "5"))

PATHS = {
    "baseline": BASELINE_LLM_URL,
    "cacheblend": CACHEBLEND_LLM_URL,
}

_RAG_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using only the provided context. "
    "Be concise and factual. If the context does not contain the answer, say so."
)


# ── vLLM / LMCache metrics ────────────────────────────────────────────────────

def get_vllm_metrics(base_url: str) -> dict:
    """Scrape vLLM Prometheus /metrics for KV cache statistics."""
    want = {
        "vllm:gpu_cache_usage_perc",
        "vllm:cpu_cache_usage_perc",
        "vllm:prefix_cache_hit_rate",
        # LMCache adds these when CacheBlend is active
        "lmcache:hit_ratio",
        "lmcache:blend_ratio",
        "lmcache:local_cpu_cache_usage",
    }
    out: dict = {}
    try:
        text = requests.get(f"{base_url}/metrics", timeout=5).text
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            for key in want:
                if line.startswith(key + " ") or line.startswith(key + "{"):
                    try:
                        out[key] = float(line.split()[-1])
                    except (IndexError, ValueError):
                        pass
    except Exception:  # noqa: BLE001
        pass
    return out


def check_health(path: str) -> dict:
    """Return health status dict for one inference path."""
    url = PATHS.get(path, "")
    if not url:
        return {"ok": False, "error": f"unknown path: {path}"}
    try:
        r = requests.get(f"{url}/health", timeout=5)
        if r.status_code == 200:
            models = []
            try:
                models = [m["id"] for m in requests.get(f"{url}/v1/models", timeout=5)
                          .json().get("data", [])]
            except Exception:  # noqa: BLE001
                pass
            return {"ok": True, "url": url, "models": models}
        return {"ok": False, "url": url, "http_status": r.status_code}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "url": url, "error": str(e)}


# ── retrieval helper ──────────────────────────────────────────────────────────

def retrieve_context(question: str, top_k: int = LLM_TOP_K) -> tuple[list[dict], float]:
    """Query the NeMo Retriever and return (deduplicated hits, retrieval_ms)."""
    t0 = time.time()
    r = requests.post(
        f"{RETRIEVER_URL}/v1/query",
        json={"query": question, "top_k": top_k * 4, "format": "hits"},
        timeout=120,
    )
    retrieval_ms = (time.time() - t0) * 1000
    hits_raw = (r.json().get("results") or [{}])[0].get("hits", [])
    out: list[dict] = []
    seen: set[str] = set()
    for h in hits_raw:
        text = h.get("text") or (h.get("metadata", {}) or {}).get("content", "")
        key = " ".join(str(text).split())[:200]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({
            "text": text,
            "score": h.get("score") or h.get("_distance"),
            "source": (h.get("metadata", {}) or {}).get("source") or h.get("source"),
        })
        if len(out) >= top_k:
            break
    return out, retrieval_ms


# ── streaming RAG generator ───────────────────────────────────────────────────

def rag_stream(
    question: str,
    llm_url: str,
    model: str = LLM_MODEL,
    top_k: int = LLM_TOP_K,
    pre_fetched_hits: list[dict] | None = None,
) -> Generator[dict, None, None]:
    """
    Streaming RAG generator. Yields dicts in this order:
      {"type": "context",  "hits": [...], "retrieval_ms": float}
      {"type": "token",    "text": str}  (repeated)
      {"type": "metrics",  ...}          (final)
      {"type": "error",    "text": str}  (on failure)

    pre_fetched_hits: pass already-retrieved hits to share a single retrieval
    call between both paths (keeps the comparison fair).
    """
    # 1. Retrieval
    if pre_fetched_hits is not None:
        hits = pre_fetched_hits
        retrieval_ms = 0.0
    else:
        try:
            hits, retrieval_ms = retrieve_context(question, top_k)
        except Exception as e:  # noqa: BLE001
            yield {"type": "error", "text": f"Retrieval failed: {e}"}
            return

    yield {"type": "context", "hits": hits, "retrieval_ms": round(retrieval_ms, 1)}

    context = "\n\n---\n\n".join(h["text"] for h in hits if h.get("text"))[:8192]
    messages = [
        {"role": "system", "content": _RAG_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    # 2. Streaming generation
    t_llm_start = time.time()
    ttft_ms: float | None = None
    token_count = 0
    error: str | None = None

    try:
        with requests.post(
            f"{llm_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": 0,
                "max_tokens": 512,
            },
            stream=True,
            timeout=300,
        ) as resp:
            if resp.status_code != 200:
                error = f"LLM HTTP {resp.status_code}: {resp.text[:200]}"
            else:
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode() if isinstance(raw, bytes) else raw
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                    except Exception:  # noqa: BLE001
                        continue
                    if delta:
                        if ttft_ms is None:
                            ttft_ms = (time.time() - t_llm_start) * 1000
                        token_count += 1
                        yield {"type": "token", "text": delta}
    except Exception as e:  # noqa: BLE001
        error = f"LLM stream error: {e}"

    if error:
        yield {"type": "error", "text": error}

    t_end = time.time()
    gen_ms = (t_end - t_llm_start) * 1000
    e2e_ms = retrieval_ms + gen_ms
    throughput = round(token_count / (gen_ms / 1000), 1) if gen_ms > 0 and token_count > 0 else 0.0

    vllm_metrics = get_vllm_metrics(llm_url)

    yield {
        "type": "metrics",
        "ttft_ms": round(ttft_ms or 0.0, 1),
        "e2e_ms": round(e2e_ms, 1),
        "gen_ms": round(gen_ms, 1),
        "retrieval_ms": round(retrieval_ms, 1),
        "token_count": token_count,
        "throughput_tok_s": throughput,
        "prefix_cache_hit_rate": vllm_metrics.get("vllm:prefix_cache_hit_rate"),
        "gpu_cache_usage_perc": vllm_metrics.get("vllm:gpu_cache_usage_perc"),
        "lmcache_hit_ratio": vllm_metrics.get("lmcache:hit_ratio"),
        "lmcache_blend_ratio": vllm_metrics.get("lmcache:blend_ratio"),
        "error": error,
    }


def rag_stream_sse(question: str, llm_url: str, **kwargs) -> Iterator[str]:
    """Wrap rag_stream() as Server-Sent Events (data: <json>\\n\\n)."""
    for event in rag_stream(question, llm_url, **kwargs):
        yield f"data: {json.dumps(event)}\n\n"
    yield "event: end\ndata: {}\n\n"


# ── inference benchmark harness ───────────────────────────────────────────────

import pathlib
BENCH_DIR = pathlib.Path(os.environ.get("BENCH_DIR",
                          str(pathlib.Path.home() / "benchmark-results")))


def _pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return round(xs[f] + (xs[c] - xs[f]) * (k - f), 2)


def _agg_latency(samples: list[float]) -> dict:
    if not samples:
        return {}
    return {
        "min": _pct(samples, 0),
        "avg": round(statistics.mean(samples), 2),
        "p50": _pct(samples, 50),
        "p90": _pct(samples, 90),
        "p95": _pct(samples, 95),
        "p99": _pct(samples, 99),
        "max": _pct(samples, 100),
    }


class InferenceSampler:
    """1 Hz GPU/CPU resource sampler, phase-tagged."""
    def __init__(self, path: pathlib.Path):
        self.phase = "init"
        self._stop = threading.Event()
        self._f = open(path, "w", encoding="utf-8")
        self._t: threading.Thread | None = None

    def set_phase(self, p: str):
        self.phase = p

    def _sample(self) -> dict:
        out: dict = {"ts": time.time(), "phase": self.phase}
        try:
            raw = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip().splitlines()
            for i, row in enumerate(raw):
                gu, mu, mt, pw = [x.strip() for x in row.split(",")]
                out[f"gpu{i}_util"] = float(gu)
                out[f"gpu{i}_vram_used_mb"] = float(mu)
                out[f"gpu{i}_vram_total_mb"] = float(mt)
                out[f"gpu{i}_power_w"] = float(pw)
        except Exception:  # noqa: BLE001
            pass
        return out

    def _loop(self):
        while not self._stop.is_set():
            self._f.write(json.dumps(self._sample()) + "\n")
            self._f.flush()
            self._stop.wait(1.0)

    def start(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=3)
        self._f.close()


def run_inference_benchmark(
    questions: list[str],
    model: str = LLM_MODEL,
    top_k: int = LLM_TOP_K,
    cold_cache_warmup: int = 0,
    log=None,
) -> dict:
    """
    Run questions against both inference paths and compare:
      - TTFT, E2E latency, generation throughput
      - vLLM prefix cache hit rate
      - LMCache blend ratio (CacheBlend path only)
      - Shared retrieval (same context for both — isolates inference delta)

    cold_cache_warmup: if > 0, submit the first N questions once to both paths
    before the timed run (simulates warm-cache steady-state).

    Returns a summary dict saved to BENCH_DIR/<run_id>/inference-summary.json.
    """
    run_id = f"{int(time.time())}-inference-{uuid.uuid4().hex[:6]}"
    artifacts = BENCH_DIR / run_id
    artifacts.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def _log(m: str):
        log_lines.append(str(m))
        if log:
            log(str(m))
        else:
            print(m, flush=True)

    _log(f"Inference benchmark: {len(questions)} questions, model={model}, top_k={top_k}")
    _log(f"Artifacts: {artifacts}")

    sampler = InferenceSampler(artifacts / "resource-samples.jsonl")
    sampler.start()

    path_results: dict[str, list[dict]] = {"baseline": [], "cacheblend": []}

    try:
        # Optional warm-up pass
        if cold_cache_warmup > 0:
            sampler.set_phase("warmup")
            _log(f"Warming up cache with first {cold_cache_warmup} questions…")
            for q in questions[:cold_cache_warmup]:
                hits, _ = retrieve_context(q, top_k)
                for path, url in PATHS.items():
                    # consume the full stream silently to fill the KV cache
                    for _ in rag_stream(q, url, model=model, pre_fetched_hits=hits):
                        pass
            _log("Warm-up done.")

        # Timed evaluation
        sampler.set_phase("eval")
        result_f = open(artifacts / "inference-results.jsonl", "w", encoding="utf-8")

        for n, question in enumerate(questions, 1):
            _log(f"[{n}/{len(questions)}] {question[:80]}")

            # Shared retrieval — both paths get the exact same context chunks.
            try:
                hits, retrieval_ms = retrieve_context(question, top_k)
            except Exception as e:  # noqa: BLE001
                _log(f"  retrieval failed: {e}")
                continue

            for path, url in PATHS.items():
                rec: dict = {
                    "n": n, "question": question, "path": path,
                    "retrieval_ms": round(retrieval_ms, 1),
                }
                full_response = []
                for event in rag_stream(question, url, model=model,
                                        pre_fetched_hits=hits):
                    if event["type"] == "token":
                        full_response.append(event["text"])
                    elif event["type"] == "metrics":
                        rec.update(event)
                    elif event["type"] == "error":
                        rec["error"] = event["text"]

                rec["response_preview"] = "".join(full_response)[:200]
                path_results[path].append(rec)
                result_f.write(json.dumps(rec) + "\n")
                result_f.flush()

                status = (f"  {path}: TTFT={rec.get('ttft_ms')}ms "
                          f"E2E={rec.get('e2e_ms')}ms "
                          f"throughput={rec.get('throughput_tok_s')}tok/s "
                          f"prefix_hit={rec.get('prefix_cache_hit_rate')}")
                if path == "cacheblend":
                    status += f" blend={rec.get('lmcache_blend_ratio')}"
                _log(status)

        result_f.close()

    finally:
        sampler.stop()

    # Aggregate per-path
    def _agg_path(records: list[dict]) -> dict:
        ttfts = [r["ttft_ms"] for r in records if r.get("ttft_ms")]
        e2es = [r["e2e_ms"] for r in records if r.get("e2e_ms")]
        throughputs = [r["throughput_tok_s"] for r in records if r.get("throughput_tok_s")]
        prefix_hits = [r["prefix_cache_hit_rate"] for r in records
                       if r.get("prefix_cache_hit_rate") is not None]
        blend_ratios = [r["lmcache_blend_ratio"] for r in records
                        if r.get("lmcache_blend_ratio") is not None]
        return {
            "n_queries": len(records),
            "ttft_ms": _agg_latency(ttfts),
            "e2e_ms": _agg_latency(e2es),
            "throughput_tok_s": {
                "avg": round(statistics.mean(throughputs), 1) if throughputs else None,
                "p50": _pct(throughputs, 50),
                "p95": _pct(throughputs, 95),
            },
            "prefix_cache_hit_rate": {
                "avg": round(statistics.mean(prefix_hits), 4) if prefix_hits else None,
                "samples": len(prefix_hits),
            },
            "lmcache_blend_ratio": {
                "avg": round(statistics.mean(blend_ratios), 4) if blend_ratios else None,
                "samples": len(blend_ratios),
            },
            "errors": sum(1 for r in records if r.get("error")),
        }

    baseline_agg = _agg_path(path_results["baseline"])
    cacheblend_agg = _agg_path(path_results["cacheblend"])

    # Delta summary — how much CacheBlend improved over baseline
    def _delta(b_val, c_val, lower_is_better=True):
        if b_val is None or c_val is None:
            return None
        delta = c_val - b_val
        pct = round(delta / b_val * 100, 1) if b_val else None
        better = (delta < 0) if lower_is_better else (delta > 0)
        return {"absolute": round(delta, 2), "pct": pct, "cacheblend_better": better}

    ttft_b = (baseline_agg.get("ttft_ms") or {}).get("avg")
    ttft_c = (cacheblend_agg.get("ttft_ms") or {}).get("avg")
    e2e_b = (baseline_agg.get("e2e_ms") or {}).get("avg")
    e2e_c = (cacheblend_agg.get("e2e_ms") or {}).get("avg")
    tput_b = (baseline_agg.get("throughput_tok_s") or {}).get("avg")
    tput_c = (cacheblend_agg.get("throughput_tok_s") or {}).get("avg")

    summary = {
        "run_id": run_id,
        "model": model,
        "top_k": top_k,
        "questions": len(questions),
        "cold_cache_warmup": cold_cache_warmup,
        "baseline": baseline_agg,
        "cacheblend": cacheblend_agg,
        "delta": {
            "ttft_ms": _delta(ttft_b, ttft_c, lower_is_better=True),
            "e2e_ms": _delta(e2e_b, e2e_c, lower_is_better=True),
            "throughput_tok_s": _delta(tput_b, tput_c, lower_is_better=False),
        },
    }

    (artifacts / "inference-summary.json").write_text(json.dumps(summary, indent=2))
    (artifacts / "log.txt").write_text("\n".join(log_lines))

    _log(f"\n=== Inference benchmark complete ===")
    _log(f"TTFT: baseline={ttft_b}ms, cacheblend={ttft_c}ms")
    _log(f"E2E:  baseline={e2e_b}ms, cacheblend={e2e_c}ms")
    _log(f"Summary saved: {artifacts / 'inference-summary.json'}")

    return summary
