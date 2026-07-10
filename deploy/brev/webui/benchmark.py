# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
NeMo Retriever — benchmark harness (phase 1: retrieval-only, raw metrics).

One run = one `run_id`. Pipeline:
    ensure dataset -> SHA-256 canonical manifest (dedup) -> dedup ingest with
    canonical IDs -> resource sampling (1 Hz, phase-tagged) -> retrieval eval
    (concurrency=1, all evaluable queries) -> summary.json + JSONL artifacts.

Identity is the content **SHA-256**; each PDF is ingested once with filename
`<sha>.pdf`, so a query hit's `source` maps straight back to a canonical doc and
Hit@k / MRR are well-defined regardless of colliding basenames or duplicate
aliases. Answer-quality (LLM) eval is intentionally out of scope in v1.

CLI (validate in a terminal first):
    python benchmark.py quick --max-qa 50      # fast smoke
    python benchmark.py quick                  # full FinQA dev retrieval eval

Artifacts: $BENCH_DIR/<run_id>/{run.json,manifest.jsonl,ingestion-results.jsonl,
retrieval-results.jsonl,resource-samples.jsonl,summary.json,log.txt}

NOTE: for meaningful numbers run against a FRESH index (a just-deployed stack).
Re-ingesting on an index that already holds playground duplicates won't create
false gold matches (non-canonical sources are ignored), but stale rows can
occupy result slots and depress Hit@k.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import threading
import time
import uuid

import requests

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import download_dataset as dd  # noqa: E402

RETRIEVER_URL = os.environ.get("RETRIEVER_URL", "http://localhost:7670")
BENCH_DIR = pathlib.Path(os.environ.get("BENCH_DIR", str(pathlib.Path.home() / "benchmark-results")))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2")


def _pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return round(xs[f] + (xs[c] - xs[f]) * (k - f), 2)


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _all_docs(jid: str):
    items, off = [], 0
    while True:
        page = requests.get(f"{RETRIEVER_URL}/v1/ingest/job/{jid}/documents",
                            params={"limit": 1000, "offset": off}, timeout=180).json()
        batch = page.get("items", [])
        items.extend(batch)
        total = int(page.get("total_filtered", page.get("total", len(items))))
        off += len(batch)
        if not batch or off >= total:
            break
    return items


# ── 1 Hz resource sampler, tagged with the current phase ─────────────────────
class ResourceSampler:
    def __init__(self, path: pathlib.Path):
        self.phase = "init"
        self._stop = threading.Event()
        self._f = open(path, "w", encoding="utf-8")
        self._t = None

    def set_phase(self, p: str):
        self.phase = p

    def _gpu(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
            u, mu, mt, pw, tp = [x.strip() for x in out.split(",")]
            return {"gpu_util": float(u), "vram_used_mb": float(mu), "vram_total_mb": float(mt),
                    "gpu_power_w": float(pw), "gpu_temp_c": float(tp)}
        except Exception:  # noqa: BLE001
            return {}

    def _host(self):
        try:
            import psutil
            vm = psutil.virtual_memory()
            io = psutil.disk_io_counters()
            net = psutil.net_io_counters()
            return {"cpu_pct": psutil.cpu_percent(), "ram_used_mb": round(vm.used / 1e6, 1),
                    "disk_read": io.read_bytes, "disk_write": io.write_bytes,
                    "net_recv": net.bytes_recv, "net_sent": net.bytes_sent}
        except Exception:  # noqa: BLE001
            return {}

    def _loop(self):
        while not self._stop.is_set():
            rec = {"ts": time.time(), "phase": self.phase, **self._gpu(), **self._host()}
            self._f.write(json.dumps(rec) + "\n")
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


# ── manifest: SHA-canonical dedup + QA gold mapping ──────────────────────────
def build_manifest(mode: str, log):
    t0 = time.time()
    qa = []
    path_sha, sha_rep, sha_alias = {}, {}, {}
    bytes_total, sha_time = 0, 0.0
    for subset, split in dd.SPLITS[mode]:
        base = dd.split_dir(subset, split)
        subset_dir = dd.DATASET_DIR / "data" / subset
        meta = base / "metadata.jsonl"
        if not meta.is_file():
            continue
        for line in meta.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            fn = r.get("file_name")
            p = next((c for c in (base / fn, subset_dir / fn) if fn and c.is_file()), None) if fn else None
            if p is None:
                qa.append({"id": r.get("id"), "subset": subset, "split": split,
                           "question": r.get("question"), "gold_sha": None})
                continue
            sp = str(p)
            if sp not in path_sha:
                ts = time.time()
                sha = _sha256(p)
                sha_time += time.time() - ts
                path_sha[sp] = sha
                if sha not in sha_rep:
                    sha_rep[sha] = sp
                    sha_alias[sha] = []
                    try:
                        bytes_total += p.stat().st_size
                    except OSError:
                        pass
                sha_alias[sha].append(sp)
            qa.append({"id": r.get("id"), "subset": subset, "split": split,
                       "question": r.get("question"), "gold_sha": path_sha[sp],
                       "answer": r.get("original_answer") or r.get("program_answer")})
    manifest = {"sha_rep": sha_rep, "sha_alias": sha_alias, "path_sha": path_sha,
                "bytes_total": bytes_total, "sha_time": round(sha_time, 2),
                "build_time": round(time.time() - t0, 2)}
    log(f"manifest: {len(path_sha)} paths -> {len(sha_rep)} unique SHAs, {len(qa)} QA records")
    return qa, manifest


# ── dedup ingest with canonical filenames (<sha>.pdf) ────────────────────────
def ingest(manifest, log, artifacts):
    shas = list(manifest["sha_rep"].keys())
    job = requests.post(f"{RETRIEVER_URL}/v1/ingest/job",
                        json={"expected_documents": len(shas), "label": "benchmark",
                              "metadata": {}, "retain_results": False}, timeout=120).json()
    jid = job["job_id"]
    did_sha, up_failed = {}, 0
    for i, sha in enumerate(shas, 1):
        p = pathlib.Path(manifest["sha_rep"][sha])
        meta = {"filename": f"{sha}.pdf", "content_type": "application/pdf",
                "metadata": {"canonical_id": sha}}
        try:
            with open(p, "rb") as fh:
                up = requests.post(f"{RETRIEVER_URL}/v1/ingest/job/{jid}/document",
                                   files={"file": (f"{sha}.pdf", fh, "application/pdf")},
                                   data={"metadata": json.dumps(meta)}, timeout=300).json()
            did_sha[up["document_id"]] = sha
        except Exception as exc:  # noqa: BLE001
            up_failed += 1
            log(f"upload failed {sha[:12]}: {exc}")
        if i % 50 == 0:
            log(f"  uploaded {i}/{len(shas)}")
    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        items = _all_docs(jid)
        if sum(1 for d in items if d.get("status") in ("completed", "failed")) >= len(did_sha):
            break
        time.sleep(3)
    items = _all_docs(jid)
    with open(artifacts / "ingestion-results.jsonl", "w", encoding="utf-8") as f:
        for d in items:
            f.write(json.dumps({
                "document_id": d.get("document_id"), "sha": did_sha.get(d.get("document_id")),
                "status": d.get("status"), "chunks": d.get("result_rows"),
                "elapsed_s": d.get("elapsed_s"), "submitted_at": d.get("submitted_at"),
                "completed_at": d.get("completed_at"), "error": d.get("error")}) + "\n")
    return items, did_sha, up_failed


def agg_ingestion(items, up_failed, wall_s):
    ok = [d for d in items if d.get("status") == "completed"]
    fail = [d for d in items if d.get("status") == "failed"]
    chunks = sum((d.get("result_rows") or 0) for d in items)
    el = [d.get("elapsed_s") for d in items if d.get("elapsed_s") is not None]
    return {
        "documents_attempted": len(items), "documents_succeeded": len(ok),
        "documents_failed": len(fail), "upload_failed": up_failed,
        "chunks_created": chunks, "avg_chunks_per_doc": round(chunks / len(ok), 2) if ok else None,
        "ingest_wall_s": round(wall_s, 1),
        "docs_per_min": round(len(ok) / (wall_s / 60), 1) if wall_s > 0 else None,
        "per_doc_latency_s": {"min": _pct(el, 0), "avg": round(statistics.mean(el), 2) if el else None,
                              "p50": _pct(el, 50), "p90": _pct(el, 90), "p95": _pct(el, 95),
                              "p99": _pct(el, 99), "max": _pct(el, 100)},
    }


# ── retrieval evaluation (concurrency=1) ─────────────────────────────────────
def retrieval_eval(qa, log, artifacts, top_k=10, max_q=None):
    evaluable = [q for q in qa if q.get("gold_sha")]
    if max_q:
        evaluable = evaluable[:max_q]
    lat, results, fails = [], [], 0
    f = open(artifacts / "retrieval-results.jsonl", "w", encoding="utf-8")
    for n, q in enumerate(evaluable, 1):
        rank, shas, ms = None, [], None
        t = time.time()
        try:
            r = requests.post(f"{RETRIEVER_URL}/v1/query",
                              json={"query": q["question"], "top_k": top_k, "format": "hits"}, timeout=120)
            ms = (time.time() - t) * 1000
            if r.status_code != 200:
                fails += 1
            else:
                hits = (r.json().get("results") or [{}])[0].get("hits", [])
                for h in hits:
                    src = h.get("source") or (h.get("metadata", {}) or {}).get("source") or ""
                    s = src[:-4] if src.endswith(".pdf") else src
                    if len(s) == 64 and s not in shas:
                        shas.append(s)
                if q["gold_sha"] in shas:
                    rank = shas.index(q["gold_sha"]) + 1
            lat.append(ms)
        except Exception as exc:  # noqa: BLE001
            fails += 1
            log(f"query failed: {exc}")
        rec = {"id": q["id"], "subset": q["subset"], "split": q["split"], "gold_sha": q["gold_sha"],
               "rank": rank, "latency_ms": round(ms, 1) if ms else None, "n_returned": len(shas)}
        results.append(rec)
        f.write(json.dumps(rec) + "\n")
        if n % 100 == 0:
            log(f"  evaluated {n}/{len(evaluable)} queries")
    f.close()
    return results, lat, fails


def agg_retrieval(results, lat, fails):
    n = len(results)
    ranked = [r for r in results if r["rank"]]

    def hit(k):
        return round(sum(1 for r in results if r["rank"] and r["rank"] <= k) / n, 4) if n else None

    ranks = [r["rank"] for r in ranked]

    def group(key):
        g = {}
        for r in results:
            g.setdefault(r[key], []).append(r)
        return {k: {"n": len(v),
                    "hit@1": round(sum(1 for x in v if x["rank"] == 1) / len(v), 4),
                    "hit@5": round(sum(1 for x in v if x["rank"] and x["rank"] <= 5) / len(v), 4),
                    "mrr@10": round(sum(1 / x["rank"] for x in v if x["rank"]) / len(v), 4)}
                for k, v in g.items()}

    return {
        "queries_total": n, "queries_completed": n - fails, "queries_failed": fails,
        "hit@1": hit(1), "hit@3": hit(3), "hit@5": hit(5), "hit@10": hit(10),
        "mrr@10": round(sum(1 / r["rank"] for r in ranked) / n, 4) if n else None,
        "mean_gold_rank": round(statistics.mean(ranks), 2) if ranks else None,
        "median_gold_rank": statistics.median(ranks) if ranks else None,
        "no_hit@10_rate": round(sum(1 for r in results if not r["rank"]) / n, 4) if n else None,
        "latency_ms": {"min": _pct(lat, 0), "avg": round(statistics.mean(lat), 2) if lat else None,
                       "p50": _pct(lat, 50), "p90": _pct(lat, 90), "p95": _pct(lat, 95),
                       "p99": _pct(lat, 99), "max": _pct(lat, 100)},
        "retrieval_qps_seq": round(len(lat) / (sum(lat) / 1000), 2) if lat and sum(lat) > 0 else None,
        "by_subset": group("subset"), "by_split": group("split"),
    }


def agg_resources(path: pathlib.Path):
    byphase = {}
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        byphase.setdefault(r.get("phase", "?"), []).append(r)
    out = {}
    for ph, rows in byphase.items():
        col = lambda k: [x[k] for x in rows if x.get(k) is not None]  # noqa: E731
        gu, vr, pw, tp = col("gpu_util"), col("vram_used_mb"), col("gpu_power_w"), col("gpu_temp_c")
        cpu, ram = col("cpu_pct"), col("ram_used_mb")
        out[ph] = {
            "samples": len(rows),
            "gpu_util": {"avg": round(statistics.mean(gu), 1) if gu else None, "p95": _pct(gu, 95), "max": max(gu) if gu else None},
            "vram_used_mb": {"avg": round(statistics.mean(vr), 1) if vr else None, "p95": _pct(vr, 95), "peak": max(vr) if vr else None},
            "gpu_power_w": {"avg": round(statistics.mean(pw), 1) if pw else None, "max": max(pw) if pw else None},
            "gpu_temp_c": {"max": max(tp) if tp else None},
            "cpu_pct": {"avg": round(statistics.mean(cpu), 1) if cpu else None, "max": max(cpu) if cpu else None},
            "ram_used_mb": {"avg": round(statistics.mean(ram), 1) if ram else None, "peak": max(ram) if ram else None},
        }
    return out


# ── orchestration ────────────────────────────────────────────────────────────
def run(mode="quick", top_k=10, max_qa=None, log=None):
    run_id = f"{int(time.time())}-{mode}-{uuid.uuid4().hex[:6]}"
    artifacts = BENCH_DIR / run_id
    artifacts.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def _log(m):
        log_lines.append(str(m))
        if log:
            log(str(m))
        else:
            print(m, flush=True)

    started = time.time()
    sampler = ResourceSampler(artifacts / "resource-samples.jsonl")
    sampler.start()
    try:
        gpu_name = ""
        try:
            gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                      capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        except Exception:  # noqa: BLE001
            pass
        cfg = {"run_id": run_id, "status": "running", "dataset": "T2-RAGBench", "revision": dd.REVISION,
               "subset": mode, "started": started, "embed_model": EMBED_MODEL, "top_k": top_k,
               "distance_metric": "lancedb-default", "vector_db": "LanceDB", "query_concurrency": 1,
               "gpu": gpu_name, "external_llm": None}

        sampler.set_phase("download")
        _log("Ensuring dataset present…")
        dd.hf_download(mode, log=_log)

        sampler.set_phase("manifest")
        _log("Building SHA-256 manifest…")
        qa, manifest = build_manifest(mode, _log)
        with open(artifacts / "manifest.jsonl", "w", encoding="utf-8") as f:
            for sha, rep in manifest["sha_rep"].items():
                f.write(json.dumps({"sha": sha, "rep": rep, "aliases": manifest["sha_alias"][sha]}) + "\n")
        corpus = {
            "qa_records_total": len(qa),
            "qa_records_evaluable": sum(1 for q in qa if q.get("gold_sha")),
            "qa_records_missing_pdf": sum(1 for q in qa if not q.get("gold_sha")),
            "pdf_paths_total": len(manifest["path_sha"]),
            "unique_pdfs_sha256": len(manifest["sha_rep"]),
            "duplicate_aliases": len(manifest["path_sha"]) - len(manifest["sha_rep"]),
            "total_pdf_bytes": manifest["bytes_total"],
            "sha_calc_time_s": manifest["sha_time"], "manifest_build_time_s": manifest["build_time"],
        }

        sampler.set_phase("ingestion")
        _log("Ingesting unique documents (canonical <sha>.pdf)…")
        ti = time.time()
        items, did_sha, up_failed = ingest(manifest, _log, artifacts)
        ingestion = agg_ingestion(items, up_failed, time.time() - ti)
        _log(f"ingested {ingestion['documents_succeeded']}/{ingestion['documents_attempted']}, "
             f"{ingestion['chunks_created']} chunks")

        sampler.set_phase("retrieval")
        _log("Retrieval evaluation (concurrency=1)…")
        results, lat, fails = retrieval_eval(qa, _log, artifacts, top_k=top_k, max_q=max_qa)
        retrieval = agg_retrieval(results, lat, fails)
        _log(f"Hit@1={retrieval['hit@1']} Hit@5={retrieval['hit@5']} MRR@10={retrieval['mrr@10']}")

        sampler.set_phase("done")
    finally:
        sampler.stop()

    resources = agg_resources(artifacts / "resource-samples.jsonl")
    finished = time.time()
    cfg.update(status="completed", finished=finished, wall_time_s=round(finished - started, 1))
    summary = {
        "run_config": cfg, "dataset_corpus": corpus, "ingestion": ingestion, "retrieval": retrieval,
        "answer_quality": {"status": "not run (v1 retrieval-only)"},
        "resources": resources,
        "errors": {"query_failures": fails, "ingestion_failed": ingestion["documents_failed"],
                   "upload_failed": ingestion["upload_failed"], "qa_missing_pdf": corpus["qa_records_missing_pdf"]},
    }
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2))
    (artifacts / "run.json").write_text(json.dumps(cfg, indent=2))
    (artifacts / "log.txt").write_text("\n".join(log_lines))
    _log(f"\nSaved -> {artifacts / 'summary.json'}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="NeMo Retriever retrieval benchmark (v1)")
    ap.add_argument("mode", nargs="?", default="quick", choices=list(dd.MODES))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-qa", type=int, default=None, help="cap eval queries (default: all evaluable)")
    args = ap.parse_args()
    s = run(args.mode, args.top_k, args.max_qa)
    print("\n=== retrieval ===")
    print(json.dumps(s["retrieval"], indent=2))
    print("\n=== corpus ===")
    print(json.dumps(s["dataset_corpus"], indent=2))


if __name__ == "__main__":
    main()
