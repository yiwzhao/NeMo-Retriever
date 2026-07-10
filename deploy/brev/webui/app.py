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
DATA_DIR = REPO_ROOT / "data"
INDEX_HTML = HERE / "index.html"

RETRIEVER_URL = os.environ.get("RETRIEVER_URL", "http://localhost:7670")
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


def _portforward_manager() -> None:
    if not ("localhost" in RETRIEVER_URL or "127.0.0.1" in RETRIEVER_URL):
        return
    env = dict(os.environ)
    env.setdefault("KUBECONFIG", "/etc/rancher/k3s/k3s.yaml")
    kubectl = shutil.which("kubectl") or "/usr/local/bin/kubectl"
    proc = None
    while True:
        try:
            if proc is None or proc.poll() is not None:
                proc = subprocess.Popen(
                    [kubectl, "port-forward", "-n", "retriever",
                     "svc/retriever-nemo-retriever", f"{_PF_PORT}:{_PF_PORT}"],
                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
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


threading.Thread(target=_portforward_manager, daemon=True).start()
threading.Thread(target=_detect_existing, daemon=True).start()

# ── deploy state (single run at a time) ──────────────────────────────────────
_state = {"phase": "idle", "running": False, "done": False, "ok": False}
_log: list[str] = []
_lock = threading.Lock()
_KEY_RE = re.compile(r"nvapi-[A-Za-z0-9_\-]+")


def _redact(line: str) -> str:
    return _KEY_RE.sub("nvapi-***", line)


def _append(line: str) -> None:
    with _lock:
        _log.append(_redact(line))
        if len(_log) > 8000:
            del _log[: len(_log) - 8000]


def _run_bootstrap(ngc_key: str) -> None:
    env = dict(os.environ)
    env["NGC_API_KEY"] = ngc_key
    env.setdefault("KUBECONFIG", "/etc/rancher/k3s/k3s.yaml")
    with _lock:
        _log.clear()
        _state.update(phase="running", running=True, done=False, ok=False)
    _append(f"==> Starting bootstrap.sh ({BOOTSTRAP})")
    try:
        proc = subprocess.Popen(
            ["bash", str(BOOTSTRAP)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            _append(line.rstrip("\n"))
        rc = proc.wait()
        ok = rc == 0
        _append(f"==> bootstrap.sh exited with code {rc}")
        with _lock:
            _state.update(running=False, done=True, ok=ok, phase="done" if ok else "failed")
    except Exception as exc:  # noqa: BLE001
        _append(f"ERROR: {exc}")
        with _lock:
            _state.update(running=False, done=True, ok=False, phase="failed")


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
@app.post("/api/deploy")
async def deploy(request: Request) -> JSONResponse:
    body = await request.json()
    key = (body or {}).get("ngc_api_key", "").strip()
    if not key:
        return JSONResponse({"error": "Enter your NGC API key (nvapi-...)."}, status_code=400)
    try:
        key.encode("ascii")
    except UnicodeEncodeError:
        return JSONResponse(
            {"error": "Key contains non-ASCII characters — paste your real nvapi-... key."},
            status_code=400,
        )
    with _lock:
        if _state["running"]:
            return JSONResponse({"error": "A deployment is already running."}, status_code=409)
    threading.Thread(target=_run_bootstrap, args=(key,), daemon=True).start()
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

    return StreamingResponse(gen(), media_type="text/event-stream")


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

_ds = {"phase": "idle", "mode": None, "downloaded": 0, "ingested": 0,
       "failed": 0, "chunks": 0, "total": 0, "elapsed": 0, "sample_q": None}
_ds_log: list[str] = []
_ds_lock = threading.Lock()


def _dlog(msg: str) -> None:
    with _ds_lock:
        _ds_log.append(str(msg))
        if len(_ds_log) > 4000:
            del _ds_log[: len(_ds_log) - 4000]


def _collect_records(mode: str):
    """Return [(pdf_path, question, answer)] for the mode, reading QA + target
    file_name from each split's metadata.jsonl."""
    records, seen = [], set()
    for subset, split in dd.SPLITS[mode]:
        base = dd.split_dir(subset, split)
        qa = {}
        meta = base / "metadata.jsonl"
        if meta.is_file():
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
                key = os.path.basename(str(fn))
                qa.setdefault(key, {"question": r.get("question"),
                                    "answer": r.get("original_answer") or r.get("program_answer")})
                qa.setdefault(os.path.splitext(key)[0], qa[key])
        pdfdir = base / "pdf"
        if not pdfdir.is_dir():
            continue
        for p in sorted(pdfdir.glob("*.pdf")):
            if str(p) in seen:
                continue
            seen.add(str(p))
            m = qa.get(p.name) or qa.get(p.stem) or {}
            records.append((p, m.get("question"), m.get("answer")))
    return records


def _run_dataset(mode: str) -> None:
    t0 = time.monotonic()
    with _ds_lock:
        _ds_log.clear()
        _ds.update(phase="downloading", mode=mode, downloaded=0, ingested=0,
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
        for p, q, a in records:
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
                _ds["elapsed"] = int(time.monotonic() - t0)

        deadline = time.monotonic() + (3600 if mode == "quick" else 6 * 3600)
        while time.monotonic() < deadline:
            docs = requests.get(f"{RETRIEVER_URL}/v1/ingest/job/{jid}/documents",
                                params={"limit": 20000}, timeout=180).json()
            items = docs.get("items", [])
            ing = sum(1 for d in items if d.get("status") == "completed")
            fail = sum(1 for d in items if d.get("status") == "failed")
            ch = sum((d.get("result_rows") or 0) for d in items)
            with _ds_lock:
                _ds.update(ingested=ing, failed=fail + upload_failed, chunks=ch,
                           elapsed=int(time.monotonic() - t0))
            terminal = sum(1 for d in items
                           if d.get("document_id") in wanted and d.get("status") in ("completed", "failed"))
            if terminal >= len(wanted):
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

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/query")
async def query(request: Request) -> JSONResponse:
    body = await request.json()
    q = (body or {}).get("query", "").strip()
    top_k = int((body or {}).get("top_k", 5))
    if not q:
        return JSONResponse({"error": "Enter a query."}, status_code=400)
    try:
        r = requests.post(
            f"{RETRIEVER_URL}/v1/query",
            json={"query": q, "top_k": top_k, "format": "hits"},
            timeout=120,
        )
        if r.status_code != 200:
            return JSONResponse({"error": f"HTTP {r.status_code}: {r.text[:300]}"}, status_code=502)
        results = r.json().get("results", [])
        hits = results[0].get("hits", []) if results else []
        out = [
            {"score": h.get("score") or h.get("_distance"),
             "source": (h.get("metadata", {}) or {}).get("source") or h.get("source"),
             "text": h.get("text") or (h.get("metadata", {}) or {}).get("content", "")}
            for h in hits
        ]
        return JSONResponse({"hits": out})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)
