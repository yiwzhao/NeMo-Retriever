# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
NeMo Retriever — one-click deploy web UI.

A small FastAPI app that lets a user deploy the Core RAG stack on a Brev node
without touching a terminal:

  1. Connect  — paste an NGC API key.
  2. Setup    — the app runs `deploy/brev/bootstrap.sh` and streams the log.
  3. Try it   — a built-in ingest + query playground against the live service,
                plus a link to the full Jupyter notebook.

Run (the Brev setup script does this for you):
    pip install fastapi uvicorn requests
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time

import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]  # .../NeMo-Retriever
BOOTSTRAP = REPO_ROOT / "deploy" / "brev" / "bootstrap.sh"
START_INFERENCE = REPO_ROOT / "deploy" / "brev" / "start-inference.sh"
DATA_DIR = REPO_ROOT / "data"
INDEX_HTML = HERE / "index.html"

RETRIEVER_URL = os.environ.get("RETRIEVER_URL", "http://localhost:7670")
# Inference layer — set by bootstrap-inference.sh when the LLM pods are up.
os.environ.setdefault("RETRIEVER_URL", RETRIEVER_URL)  # propagate to inference module
# Optional: a Jupyter URL for the "Open notebook" button. If unset, the UI shows
# the notebook path instead.
NOTEBOOK_URL = os.environ.get("NOTEBOOK_URL", "")

# A small, fast, diverse multi-doc corpus for the "scale" demo (all in data/).
SAMPLE_PDF = "multimodal_test.pdf"
MINI_CORPUS = [
    "multimodal_test.pdf",   # text + table + chart
    "table_test.pdf",        # dense table
    "test-shapes.pdf",       # graphic elements
    "woods_frost.pdf",       # prose
    "functional_validation.pdf",
    "embedded_table.pdf",
]

app = FastAPI(title="NeMo Retriever Deploy")


# ── keep localhost:7670 wired to the in-cluster service ──────────────────────
# The retriever service is a ClusterIP; the host reaches it via `kubectl
# port-forward`. Run a self-healing forward so the playground (and any local
# client) can hit http://localhost:7670 without a manual step.
_PF_PORT = "7670"
# Inference ports — forwarded by bootstrap-inference.sh; the manager
# keeps them alive across k3s restarts just like the retriever forward.
_INFERENCE_FORWARDS = [
    ("svc/inference-baseline",   "8001", "8001"),
    ("svc/inference-cacheblend", "8002", "8002"),
]


def _portforward_manager() -> None:
    # Self-healing: the UI starts before k3s exists and k3s restarts a few times
    # during bootstrap, so a plain "respawn if the process died" loop can get
    # wedged. Instead, actively probe /v1/health and force-respawn whenever the
    # forward isn't actually serving — even if the process looks alive.
    if not ("localhost" in RETRIEVER_URL or "127.0.0.1" in RETRIEVER_URL):
        return
    kubeconfig = os.environ.get("KUBECONFIG", "/etc/rancher/k3s/k3s.yaml")
    proc = None
    bad = 0
    while True:
        try:
            kubectl = shutil.which("kubectl") or "/usr/local/bin/kubectl"
            ready = os.path.exists(kubectl) and os.path.exists(kubeconfig)
            try:
                serving = requests.get(f"{RETRIEVER_URL}/v1/health", timeout=2).status_code == 200
            except Exception:  # noqa: BLE001
                serving = False
            bad = 0 if serving else bad + 1
            # respawn if the process is gone, or health has failed a few times in a row
            if ready and ((proc is None or proc.poll() is not None) or bad >= 3):
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except Exception:  # noqa: BLE001
                        proc.kill()
                env = dict(os.environ)
                env["KUBECONFIG"] = kubeconfig
                proc = subprocess.Popen(
                    [kubectl, "port-forward", "-n", "retriever",
                     "svc/retriever-nemo-retriever", f"{_PF_PORT}:{_PF_PORT}"],
                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                bad = 0
        except Exception:  # noqa: BLE001 - kubectl may not exist until k3s is installed
            proc = None
        time.sleep(5)


def _detect_existing() -> None:
    """If the stack is already live (e.g. after a webui restart or reopening the
    launchable), mark the deploy done so the UI lands on the playground."""
    while True:
        with _lock:
            skip = _state["running"] or _state["phase"] == "done"
        if not skip:
            try:
                if requests.get(f"{RETRIEVER_URL}/v1/health", timeout=3).status_code == 200:
                    with _lock:
                        if not _state["running"]:
                            _state.update(phase="done", running=False, done=True, ok=True)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(6)


# ── deploy state (single run at a time) ──────────────────────────────────────
# Defined BEFORE the background threads start — _detect_existing uses _lock/_state,
# so starting the threads earlier races the module import (NameError on some hosts).
_state = {"phase": "idle", "running": False, "done": False, "ok": False}
_log: list[str] = []
_lock = threading.Lock()
_KEY_RE = re.compile(r"nvapi-[A-Za-z0-9_\-]+")

def _inference_portforward_manager() -> None:
    """Keep port-forwards alive for k8s-deployed inference paths.

    Skipped when INFERENCE_MODE=host (vLLM runs directly on the host at ports
    8001/8002 — no kubectl port-forward needed in that case).
    """
    if os.environ.get("INFERENCE_MODE") == "host":
        return  # host-based vLLM; ports are already on localhost
    kubeconfig = os.environ.get("KUBECONFIG", "/etc/rancher/k3s/k3s.yaml")
    procs: dict[str, object] = {}
    while True:
        try:
            kubectl = shutil.which("kubectl") or "/usr/local/bin/kubectl"
            ready = os.path.exists(kubectl) and os.path.exists(kubeconfig)
            if ready:
                for svc, local_port, remote_port in _INFERENCE_FORWARDS:
                    proc = procs.get(svc)
                    if proc is None or proc.poll() is not None:
                        env = dict(os.environ)
                        env["KUBECONFIG"] = kubeconfig
                        procs[svc] = subprocess.Popen(
                            [kubectl, "port-forward", "-n", "retriever",
                             svc, f"{local_port}:{remote_port}"],
                            env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        )
        except Exception:  # noqa: BLE001
            pass
        time.sleep(10)


threading.Thread(target=_portforward_manager, daemon=True).start()
threading.Thread(target=_inference_portforward_manager, daemon=True).start()
threading.Thread(target=_detect_existing, daemon=True).start()


def _redact(line: str) -> str:
    return _KEY_RE.sub("nvapi-***", line)


def _append(line: str) -> None:
    with _lock:
        _log.append(_redact(line))
        if len(_log) > 8000:
            del _log[: len(_log) - 8000]


def _run_script(cmd: list, cwd: str, env: dict, label: str) -> bool:
    """Run a subprocess, streaming its output to the shared log. Returns True on success."""
    _append(f"==> {label}")
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            _append(line.rstrip("\n"))
        rc = proc.wait()
        _append(f"==> {label} exited with code {rc}")
        return rc == 0
    except Exception as exc:  # noqa: BLE001
        _append(f"ERROR running {label}: {exc}")
        return False


def _run_bootstrap(ngc_key: str, hf_token: str = "") -> None:
    env = dict(os.environ)
    env["NGC_API_KEY"] = ngc_key
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    env.setdefault("KUBECONFIG", "/etc/rancher/k3s/k3s.yaml")
    # Pin HF cache to /ephemeral (large disk on Brev) so model files don't land
    # in the repo directory. Create the path if it doesn't exist yet.
    if not env.get("HUGGINGFACE_HUB_CACHE"):
        hf_cache = pathlib.Path("/ephemeral/cache/huggingface/hub")
        try:
            hf_cache.mkdir(parents=True, exist_ok=True)
        except OSError:
            hf_cache = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
            hf_cache.mkdir(parents=True, exist_ok=True)
        env["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)

    with _lock:
        _log.clear()
        _state.update(phase="retriever", running=True, done=False, ok=False)

    # ── Phase 1: NeMo Retriever (k3s + NIMs) ──────────────────────────────────
    ok = _run_script(
        ["bash", str(BOOTSTRAP)], str(REPO_ROOT), env,
        f"Phase 1/2 — NeMo Retriever bootstrap ({BOOTSTRAP.name})",
    )
    if not ok:
        _append("==> bootstrap.sh failed — skipping inference layer startup")
        with _lock:
            _state.update(running=False, done=True, ok=False, phase="failed")
        return

    with _lock:
        _state.update(phase="inference")

    # ── Phase 2: Inference layer (vLLM baseline + CacheBlend + webui) ─────────
    if not START_INFERENCE.is_file():
        _append(f"WARNING: {START_INFERENCE} not found — skipping inference layer")
    else:
        _append("==> Phase 2/2 — Inference layer (vLLM + LMCache)")
        _append("    This restarts the current webui process at the end.")
        _append("    The page may reload — that is expected.")
        ok = _run_script(
            ["bash", str(START_INFERENCE)], str(REPO_ROOT), env,
            f"Phase 2/2 — Inference layer ({START_INFERENCE.name})",
        )

    with _lock:
        _state.update(running=False, done=True, ok=ok, phase="done" if ok else "failed")


# ── pages ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


@app.get("/api/config")
def config() -> dict:
    return {"notebook_url": NOTEBOOK_URL, "sample_pdf": SAMPLE_PDF, "corpus_size": len(MINI_CORPUS)}


@app.get("/THIRD_PARTY_DATA.md", response_class=HTMLResponse)
def third_party_data() -> HTMLResponse:
    md = HERE.parent / "THIRD_PARTY_DATA.md"
    text = md.read_text(encoding="utf-8") if md.is_file() else "Not found."
    return HTMLResponse(f"<pre style='white-space:pre-wrap;font-family:monospace;padding:24px;"
                        f"background:#0b0d0b;color:#e8ece6'>{text}</pre>")


# ── deploy ───────────────────────────────────────────────────────────────────
@app.post("/api/reset")
async def reset_state() -> JSONResponse:
    """Reset deploy state so the UI returns to Step 1 (Connect)."""
    with _lock:
        if _state.get("running"):
            return JSONResponse({"error": "A deployment is running — wait for it to finish."}, status_code=409)
        _log.clear()
        _state.update(phase="idle", running=False, done=False, ok=False)
    return JSONResponse({"reset": True})


@app.post("/api/deploy")
async def deploy(request: Request) -> JSONResponse:
    body = await request.json()
    key = (body or {}).get("ngc_api_key", "").strip()
    hf_token = (body or {}).get("hf_token", "").strip()
    if not key:
        return JSONResponse({"error": "Enter your NGC API key (nvapi-...)."}, status_code=400)
    try:
        key.encode("ascii")
    except UnicodeEncodeError:
        return JSONResponse(
            {"error": "Key contains non-ASCII characters — paste your real nvapi-... key."},
            status_code=400,
        )
    if hf_token:
        try:
            hf_token.encode("ascii")
        except UnicodeEncodeError:
            return JSONResponse(
                {"error": "HuggingFace token contains non-ASCII characters."},
                status_code=400,
            )
    with _lock:
        if _state["running"]:
            return JSONResponse({"error": "A deployment is already running."}, status_code=409)
    threading.Thread(target=_run_bootstrap, args=(key, hf_token), daemon=True).start()
    return JSONResponse({"started": True})


@app.get("/api/status")
def status() -> dict:
    with _lock:
        return {**_state, "log_lines": len(_log)}


@app.get("/api/logs")
def logs() -> StreamingResponse:
    def gen():
        idx = 0
        while True:
            with _lock:
                new = _log[idx:]
                idx = len(_log)
                done = _state["done"]
                phase = _state["phase"]
                ok = _state["ok"]
            for ln in new:
                yield f"data: {json.dumps({'line': ln})}\n\n"
            if done and idx >= len(_log):
                yield f"event: end\ndata: {json.dumps({'phase': phase, 'ok': ok})}\n\n"
                return
            time.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


# ── live service proxy (playground) ──────────────────────────────────────────
@app.get("/api/health")
def health() -> JSONResponse:
    try:
        r = requests.get(f"{RETRIEVER_URL}/v1/health", timeout=5)
        return JSONResponse({"ok": r.status_code == 200, "body": r.json()})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)})


def _ingest_files(filenames: list[str]) -> dict:
    paths = [DATA_DIR / f for f in filenames if (DATA_DIR / f).is_file()]
    if not paths:
        return {"error": "No sample files found on disk."}
    job = requests.post(
        f"{RETRIEVER_URL}/v1/ingest/job",
        json={"expected_documents": len(paths), "label": "webui", "metadata": {}, "retain_results": False},
        timeout=60,
    ).json()
    jid = job["job_id"]
    ids = []
    for p in paths:
        meta = {"filename": p.name, "content_type": "application/pdf", "metadata": {}}
        with open(p, "rb") as fh:
            up = requests.post(
                f"{RETRIEVER_URL}/v1/ingest/job/{jid}/document",
                files={"file": (p.name, fh, "application/pdf")},
                data={"metadata": json.dumps(meta)},
                timeout=120,
            ).json()
        ids.append(up["document_id"])
    deadline = time.monotonic() + 900
    items: dict[str, dict] = {}
    while time.monotonic() < deadline:
        docs = requests.get(f"{RETRIEVER_URL}/v1/ingest/job/{jid}/documents", params={"limit": 1000}, timeout=60).json()
        items = {d["document_id"]: d for d in docs.get("items", [])}
        if all(items.get(i, {}).get("status") in ("completed", "failed") for i in ids):
            break
        time.sleep(2)
    docs_out = [
        {"filename": items.get(i, {}).get("filename"), "status": items.get(i, {}).get("status"),
         "rows": items.get(i, {}).get("result_rows")}
        for i in ids
    ]
    total = sum((d["rows"] or 0) for d in docs_out)
    return {"job_id": jid, "documents": docs_out, "total_rows": total}


@app.post("/api/ingest")
async def ingest(request: Request) -> JSONResponse:
    body = await request.json()
    which = (body or {}).get("which", "sample")
    files = MINI_CORPUS if which == "corpus" else [SAMPLE_PDF]
    try:
        return JSONResponse(_ingest_files(files))
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)


# ── T2-RAGBench dataset: download + batch ingest with progress ───────────────
sys.path.insert(0, str(HERE.parent))
import download_dataset as dd  # noqa: E402  (deploy/brev/download_dataset.py)

_ds = {"phase": "idle", "mode": None, "downloaded": 0, "uploaded": 0, "ingested": 0,
       "failed": 0, "chunks": 0, "total": 0, "elapsed": 0, "sample_q": None}
_ds_log: list[str] = []
_ds_lock = threading.Lock()


def _dlog(msg: str) -> None:
    with _ds_lock:
        _ds_log.append(str(msg))
        if len(_ds_log) > 4000:
            del _ds_log[: len(_ds_log) - 4000]


def _collect_records(mode: str):
    """Return [(pdf_path, question, answer)] — metadata-driven. Each split's
    metadata.jsonl has a `file_name` (e.g. 'pdf/V/2008/page_17.pdf') relative to
    the split dir; resolve it to a PDF and take the QA from the same record.
    Deduped by resolved path (a document can have several questions)."""
    records, seen = [], set()
    for subset, split in dd.SPLITS[mode]:
        base = dd.split_dir(subset, split)             # data/<subset>/<split>
        subset_dir = dd.DATASET_DIR / "data" / subset  # data/<subset>
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
            if not fn:
                continue
            # file_name is relative to the split dir; fall back to the subset dir.
            p = next((c for c in (base / fn, subset_dir / fn) if c.is_file()), None)
            if p is None or str(p) in seen:
                continue
            seen.add(str(p))
            records.append((p, r.get("question"),
                            r.get("original_answer") or r.get("program_answer")))
    return records


def _run_dataset(mode: str) -> None:
    t0 = time.monotonic()
    with _ds_lock:
        _ds_log.clear()
        _ds.update(phase="downloading", mode=mode, downloaded=0, uploaded=0, ingested=0,
                   failed=0, chunks=0, total=0, elapsed=0, sample_q=None)
    _dlog(f"Downloading T2-RAGBench [{dd.MODES[mode]['desc']}] @ {dd.REVISION[:12]}")
    try:
        dd.hf_download(mode, log=_dlog)
    except Exception as exc:  # noqa: BLE001
        _dlog(f"download failed: {exc}")
        with _ds_lock:
            _ds.update(phase="failed", elapsed=int(time.monotonic() - t0))
        return

    records = _collect_records(mode)
    with _ds_lock:
        _ds.update(phase="ingesting", downloaded=len(records), total=len(records))
        for _p, q, _a in records:
            if q:
                _ds["sample_q"] = q
                break
    _dlog(f"Downloaded {len(records)} PDFs. Ingesting…")
    if not records:
        with _ds_lock:
            _ds.update(phase="failed")
        _dlog("No PDFs found after download.")
        return

    upload_failed = 0
    try:
        job = requests.post(
            f"{RETRIEVER_URL}/v1/ingest/job",
            json={"expected_documents": len(records), "label": f"t2ragbench-{mode}",
                  "metadata": {}, "retain_results": False},
            timeout=120,
        ).json()
        jid = job["job_id"]
        wanted = set()

        def _refresh() -> int:
            """Pull job doc statuses into the counters; return terminal count.
            Pages the endpoint (server caps `limit` at 1000)."""
            items = []
            offset = 0
            try:
                while True:
                    page = requests.get(
                        f"{RETRIEVER_URL}/v1/ingest/job/{jid}/documents",
                        params={"limit": 1000, "offset": offset}, timeout=180,
                    ).json()
                    batch = page.get("items", [])
                    items.extend(batch)
                    total = int(page.get("total_filtered", page.get("total", len(items))))
                    offset += len(batch)
                    if not batch or offset >= total:
                        break
            except Exception:  # noqa: BLE001
                return 0
            with _ds_lock:
                _ds.update(
                    ingested=sum(1 for d in items if d.get("status") == "completed"),
                    failed=sum(1 for d in items if d.get("status") == "failed") + upload_failed,
                    chunks=sum((d.get("result_rows") or 0) for d in items),
                    elapsed=int(time.monotonic() - t0),
                )
            return sum(1 for d in items
                       if d.get("document_id") in wanted and d.get("status") in ("completed", "failed"))

        # Upload (progress shown live). The service ingests docs as they arrive,
        # so we refresh the counters during upload too — not just after.
        for idx, (p, q, a) in enumerate(records, 1):
            meta = {"filename": p.name, "content_type": "application/pdf",
                    "metadata": {"dataset": "t2-ragbench", "question": q, "answer": a}}
            try:
                with open(p, "rb") as fh:
                    up = requests.post(
                        f"{RETRIEVER_URL}/v1/ingest/job/{jid}/document",
                        files={"file": (p.name, fh, "application/pdf")},
                        data={"metadata": json.dumps(meta)}, timeout=300,
                    ).json()
                wanted.add(up["document_id"])
            except Exception as exc:  # noqa: BLE001
                upload_failed += 1
                _dlog(f"upload failed {p.name}: {exc}")
            with _ds_lock:
                _ds.update(uploaded=idx, elapsed=int(time.monotonic() - t0))
            if idx % 10 == 0:
                _refresh()

        deadline = time.monotonic() + (3600 if mode == "quick" else 6 * 3600)
        while time.monotonic() < deadline:
            if _refresh() >= len(wanted):
                break
            time.sleep(3)
        with _ds_lock:
            _ds.update(phase="done", elapsed=int(time.monotonic() - t0))
        _dlog(f"Done — ingested {_ds['ingested']}, failed {_ds['failed']}, {_ds['chunks']} chunks in {_ds['elapsed']}s")
    except Exception as exc:  # noqa: BLE001
        _dlog(f"ingest error: {exc}")
        with _ds_lock:
            _ds.update(phase="failed", elapsed=int(time.monotonic() - t0))


@app.post("/api/dataset")
async def dataset(request: Request) -> JSONResponse:
    body = await request.json()
    mode = (body or {}).get("mode", "quick")
    if mode not in dd.MODES:
        return JSONResponse({"error": f"unknown mode {mode}"}, status_code=400)
    with _ds_lock:
        if _ds["phase"] in ("downloading", "ingesting"):
            return JSONResponse({"error": "A dataset run is already in progress."}, status_code=409)
    threading.Thread(target=_run_dataset, args=(mode,), daemon=True).start()
    return JSONResponse({"started": True, "mode": mode})


@app.get("/api/dataset/status")
def dataset_status() -> dict:
    with _ds_lock:
        return dict(_ds)


@app.get("/api/dataset/logs")
def dataset_logs() -> StreamingResponse:
    def gen():
        idx = 0
        while True:
            with _ds_lock:
                new = _ds_log[idx:]
                idx = len(_ds_log)
                phase = _ds["phase"]
            for ln in new:
                yield f"data: {json.dumps({'line': ln})}\n\n"
            if phase in ("done", "failed") and idx >= len(_ds_log):
                yield f"event: end\ndata: {json.dumps({'phase': phase})}\n\n"
                return
            time.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


# ── benchmark harness (retrieval-only v1) ────────────────────────────────────
import benchmark as bench  # noqa: E402  (deploy/brev/webui/benchmark.py)
import inference as inf  # noqa: E402  (deploy/brev/webui/inference.py)

_bench = {"phase": "idle", "run_id": None}
_bench_log: list[str] = []
_bench_lock = threading.Lock()


def _bench_logfn(msg: str) -> None:
    with _bench_lock:
        _bench_log.append(str(msg))
        if len(_bench_log) > 4000:
            del _bench_log[: len(_bench_log) - 4000]


def _run_bench(mode: str, max_qa, qa_sample=100) -> None:
    with _bench_lock:
        _bench_log.clear()
        _bench.update(phase="running", run_id=None)
    try:
        summary = bench.run(mode, top_k=10, max_qa=max_qa, qa_sample=qa_sample, log=_bench_logfn)
        with _bench_lock:
            _bench.update(phase="done", run_id=summary["run_config"]["run_id"])
    except Exception as exc:  # noqa: BLE001
        _bench_logfn(f"ERROR: {exc}")
        with _bench_lock:
            _bench.update(phase="failed")


@app.get("/benchmark", response_class=HTMLResponse)
def benchmark_page() -> str:
    return (HERE / "benchmark.html").read_text(encoding="utf-8")


@app.post("/api/benchmark")
async def benchmark_start(request: Request) -> JSONResponse:
    body = await request.json()
    mode = (body or {}).get("mode", "quick")
    max_qa = (body or {}).get("max_qa")
    qa_sample = (body or {}).get("qa_sample", 100)
    if mode not in bench.dd.MODES:
        return JSONResponse({"error": f"unknown mode {mode}"}, status_code=400)
    with _bench_lock:
        if _bench["phase"] == "running":
            return JSONResponse({"error": "A benchmark is already running."}, status_code=409)
    threading.Thread(target=_run_bench, args=(mode, max_qa, qa_sample), daemon=True).start()
    return JSONResponse({"started": True})


@app.get("/api/benchmark/status")
def benchmark_status() -> dict:
    with _bench_lock:
        return dict(_bench)


@app.get("/api/benchmark/logs")
def benchmark_logs() -> StreamingResponse:
    def gen():
        idx = 0
        while True:
            with _bench_lock:
                new = _bench_log[idx:]
                idx = len(_bench_log)
                phase = _bench["phase"]
            for ln in new:
                yield f"data: {json.dumps({'line': ln})}\n\n"
            if phase in ("done", "failed") and idx >= len(_bench_log):
                yield f"event: end\ndata: {json.dumps({'phase': phase})}\n\n"
                return
            time.sleep(0.5)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


@app.get("/api/benchmark/runs")
def benchmark_runs() -> dict:
    runs = []
    if bench.BENCH_DIR.is_dir():
        for d in sorted(bench.BENCH_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if (d / "summary.json").is_file():
                runs.append(d.name)
    return {"runs": runs}


@app.get("/api/benchmark/summary")
def benchmark_summary(run_id: str = "") -> JSONResponse:
    d = (bench.BENCH_DIR / run_id) if run_id else None
    if not d or not (d / "summary.json").is_file():
        cand = ([p for p in bench.BENCH_DIR.iterdir() if (p / "summary.json").is_file()]
                if bench.BENCH_DIR.is_dir() else [])
        if not cand:
            return JSONResponse({"error": "no completed runs"}, status_code=404)
        d = max(cand, key=lambda p: p.stat().st_mtime)
    return JSONResponse(json.loads((d / "summary.json").read_text(encoding="utf-8")))


# ── inference comparison UI ───────────────────────────────────────────────────

@app.get("/compare", response_class=HTMLResponse)
def compare_page() -> str:
    return (HERE / "compare.html").read_text(encoding="utf-8")


@app.get("/api/inference/health/{path}")
def inference_health(path: str) -> JSONResponse:
    """Health check for one inference path (baseline or cacheblend)."""
    return JSONResponse(inf.check_health(path))


@app.get("/api/inference/health")
def inference_health_all() -> JSONResponse:
    return JSONResponse({
        "baseline": inf.check_health("baseline"),
        "cacheblend": inf.check_health("cacheblend"),
    })


@app.post("/api/rag/stream")
async def rag_stream(request: Request) -> StreamingResponse:
    """
    Streaming RAG endpoint.  Accepts JSON body:
      { "question": str, "path": "baseline"|"cacheblend",
        "shared_hits": [...] (optional, avoids re-retrieval for warm runs) }

    Yields SSE events: context → token* → metrics | error
    """
    body = await request.json()
    question = (body or {}).get("question", "").strip()
    path = (body or {}).get("path", "baseline")
    shared_hits = (body or {}).get("shared_hits")  # pre-fetched context from prior call
    top_k = int((body or {}).get("top_k", inf.LLM_TOP_K))

    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)
    if path not in inf.PATHS:
        return JSONResponse({"error": f"unknown path: {path}"}, status_code=400)

    llm_url = inf.PATHS[path]
    return StreamingResponse(
        inf.rag_stream_sse(question, llm_url, top_k=top_k,
                           model=path,
                           pre_fetched_hits=shared_hits or None),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── inference benchmark ───────────────────────────────────────────────────────

_ibench: dict = {"phase": "idle"}
_ibench_log: list[str] = []
_ibench_lock = threading.Lock()


def _ibench_logfn(msg: str) -> None:
    with _ibench_lock:
        _ibench_log.append(str(msg))
        if len(_ibench_log) > 4000:
            del _ibench_log[: len(_ibench_log) - 4000]


def _run_ibench(questions: list[str], top_k: int, cold_cache_warmup: int) -> None:
    with _ibench_lock:
        _ibench_log.clear()
        _ibench.update(phase="running", run_id=None)
    try:
        summary = inf.run_inference_benchmark(
            questions, top_k=top_k, cold_cache_warmup=cold_cache_warmup,
            log=_ibench_logfn,
        )
        with _ibench_lock:
            _ibench.update(phase="done", run_id=summary["run_id"])
    except Exception as exc:  # noqa: BLE001
        _ibench_logfn(f"ERROR: {exc}")
        with _ibench_lock:
            _ibench.update(phase="failed")


@app.post("/api/inference/benchmark/start")
async def ibench_start(request: Request) -> JSONResponse:
    body = await request.json()
    questions = (body or {}).get("questions", [])
    if not questions:
        return JSONResponse({"error": "questions list is required"}, status_code=400)
    top_k = int((body or {}).get("top_k", inf.LLM_TOP_K))
    warmup = int((body or {}).get("cold_cache_warmup", 0))
    with _ibench_lock:
        if _ibench["phase"] == "running":
            return JSONResponse({"error": "A benchmark is already running."}, status_code=409)
    threading.Thread(
        target=_run_ibench, args=(questions, top_k, warmup), daemon=True
    ).start()
    return JSONResponse({"started": True})


@app.get("/api/inference/benchmark/status")
def ibench_status() -> dict:
    with _ibench_lock:
        return dict(_ibench)


@app.get("/api/inference/benchmark/logs")
def ibench_logs() -> StreamingResponse:
    def gen():
        idx = 0
        while True:
            with _ibench_lock:
                new = _ibench_log[idx:]
                idx = len(_ibench_log)
                phase = _ibench["phase"]
            for ln in new:
                yield f"data: {json.dumps({'line': ln})}\n\n"
            if phase in ("done", "failed") and idx >= len(_ibench_log):
                yield f"event: end\ndata: {json.dumps({'phase': phase})}\n\n"
                return
            time.sleep(0.5)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


@app.get("/api/inference/benchmark/summary")
def ibench_summary(run_id: str = "") -> JSONResponse:
    bench_dir = inf.BENCH_DIR
    if not bench_dir.is_dir():
        return JSONResponse({"error": "no benchmark runs found"}, status_code=404)
    cands = [p for p in bench_dir.iterdir() if (p / "inference-summary.json").is_file()]
    if not cands:
        return JSONResponse({"error": "no completed inference runs"}, status_code=404)
    if run_id:
        d = bench_dir / run_id
    else:
        d = max(cands, key=lambda p: p.stat().st_mtime)
    sf = d / "inference-summary.json"
    if not sf.is_file():
        return JSONResponse({"error": "summary not found"}, status_code=404)
    return JSONResponse(json.loads(sf.read_text(encoding="utf-8")))


@app.post("/api/query")
async def query(request: Request) -> JSONResponse:
    body = await request.json()
    q = (body or {}).get("query", "").strip()
    top_k = int((body or {}).get("top_k", 5))
    if not q:
        return JSONResponse({"error": "Enter a query."}, status_code=400)
    try:
        # Over-fetch, then dedupe by text so repeated ingests (the vectordb is
        # append-only) don't show up as identical hits.
        r = requests.post(
            f"{RETRIEVER_URL}/v1/query",
            json={"query": q, "top_k": max(top_k * 4, 20), "format": "hits"},
            timeout=120,
        )
        if r.status_code != 200:
            return JSONResponse({"error": f"HTTP {r.status_code}: {r.text[:300]}"}, status_code=502)
        results = r.json().get("results", [])
        hits = results[0].get("hits", []) if results else []
        out, seen = [], set()
        for h in hits:
            text = h.get("text") or (h.get("metadata", {}) or {}).get("content", "")
            key = " ".join(str(text).split())[:200]
            if not key or key in seen:
                continue
            seen.add(key)
            out.append({
                "score": h.get("score") or h.get("_distance"),
                "source": (h.get("metadata", {}) or {}).get("source") or h.get("source"),
                "text": text,
            })
            if len(out) >= top_k:
                break
        return JSONResponse({"hits": out})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)
