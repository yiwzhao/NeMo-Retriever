<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Retriever on Brev — Deployment Notes & Troubleshooting Log

A complete, archival reference for the single-node **Core RAG** launchable in
this folder: what it deploys, how the GPU is shared, the exact environment it was
verified on, and a blow-by-blow log of every issue hit while bringing it up on a
**Blackwell RTX PRO 6000 + k3s (containerd 2.x)** node — with root cause and fix.

For the short "how to run it" version, see [`README.md`](./README.md). This
document is the deep dive you reach for when something breaks or you're adapting
the launchable to a new environment.

---

## 1. What this launchable is

A one-command bootstrap ([`bootstrap.sh`](./bootstrap.sh)) that stands up the
**core** NeMo Retriever RAG pipeline on a single Brev GPU node using
single-node Kubernetes (k3s) + Helm, plus a notebook
([`notebooks/nemo_retriever_quickstart.ipynb`](./notebooks/nemo_retriever_quickstart.ipynb))
that drives the running service over its REST API (ingest → query → answer).

**Scope:** the four core NIMs + the FastAPI service + LanceDB. Reranking, the
in-cluster answer LLM, audio/video (Parakeet), captioning (Omni), and Nemotron
Parse are **off by default**. Answer generation is expected to run against an
external OpenAI-compatible endpoint (e.g. build.nvidia.com).

---

## 2. Verified environment

This launchable was brought up end-to-end on the following. Other GPUs / k3s
versions should work, but the GPU-runtime handling in `bootstrap.sh` was shaped
by the containerd 2.x behavior below.

| Component | Version / value |
|-----------|-----------------|
| GPU | **NVIDIA RTX PRO 6000 Blackwell Server Edition**, 97,887 MiB (~96 GB) |
| NVIDIA driver (host) | **580.126.09** (Brev image; reused, `driver.enabled=false`) |
| CUDA (inside NIM images) | **13.0.1** |
| OS / kernel | Ubuntu 24.04.4 LTS / 6.8.0 |
| Kubernetes | **k3s v1.36.2+k3s1** |
| Container runtime | **containerd 2.3.2-k3s2** (config **version 3**) |
| GPU Operator | latest (**v26.3.3** at time of writing) — device-plugin only |
| NVIDIA device plugin | v0.19.3 |
| NIM Operator | latest |
| Retriever service / chart | **26.5.0** |
| Core NIM images | page-elements `1.8.0`, table-structure `1.8.0`, ocr-v2, `llama-nemotron-embed-vl-1b-v2:1.12.0` |
| GPU sharing | time-slicing, **8 replicas on 1 physical GPU** |

**End-to-end result:** ingesting the repo's `data/multimodal_test.pdf` produced
8 extracted chunks (text + table + chart); a query for *"Which animal is jumping
onto a laptop?"* returned the correct row `| Cat | Jumping onto a laptop | In a
home office |`.

---

## 3. Components & services

NeMo Retriever = a **FastAPI orchestrator** (CPU) + a set of **NVIDIA NIM
microservices** (GPU), reconciled on Kubernetes by the **NIM Operator**, with a
**LanceDB** pod for the vector index.

| RAG stage | Service / component | GPU? | In this launchable |
|-----------|--------------------|------|--------------------|
| Orchestration | `retriever-service` (FastAPI + Ray workers) | No — CPU/RAM heavy | Yes |
| Ingestion → layout | `page-elements` NIM | Yes | Yes (core) |
| Ingestion → tables | `table-structure` NIM | Yes | Yes (core) |
| Ingestion → OCR | `ocr` NIM | Yes | Yes (core) |
| Embedding | `vlm_embed` NIM (`llama-nemotron-embed-vl-1b-v2`) | Yes | Yes (core) |
| Indexing | LanceDB `vectordb` pod (+ PVC) | No — CPU | Yes (core) |
| Query / retrieval | `/v1/query` on the service → embeds query (reuses `vlm_embed`) → LanceDB search | Yes (reuses embed NIM) | Yes (core) |
| Reranking | `rerankqa` NIM (`llama-nemotron-rerank-vl-1b-v2`) | Yes | Off (optional) |
| Answer / LLM serving | `answer_llm` NIM (Super-49B) **or** external API | Yes (2 GPU) / — | External by default |
| Audio / video | `audio` Parakeet ASR NIM | Yes | Off (optional) |
| Captioning | Omni 30B NIM | Yes | Off (optional) |
| Alt. PDF parse | `nemotron_parse` NIM | Yes | Off (optional) |

**Kubernetes objects the stack creates** (namespace `retriever`):

- Deployments: `retriever-nemo-retriever` (service), `…-vectordb` (LanceDB),
  `…-otel`, `…-zipkin`.
- `NIMCache` + `NIMService` (CRDs, `apps.nvidia.com/v1alpha1`) for each of the 4
  core NIMs, reconciled by the NIM Operator into Deployments + Services.
- Service `retriever-nemo-retriever` on port **7670** (the single API entry
  point; it proxies `/v1/query` and `/v1/answer` to the vectordb pod on 7671).

**REST API surface** (all on port 7670):
`/v1/health`, `/v1/ingest/job` (create), `/v1/ingest/job/{id}/document`
(upload, multipart), `/v1/ingest/job/{id}/documents` (poll),
`/v1/ingest/job/{id}/document/{doc}` (result), `/v1/query`, `/v1/answer`.

---

## 4. GPU budget & sizing

The four core NIMs total only **~4.8 GiB** of weights, so they fit on one GPU.
`bootstrap.sh` enables **GPU time-slicing** (default 8 replicas) so a single
physical GPU advertises multiple schedulable `nvidia.com/gpu` units and hosts all
four NIMs.

| Layout | GPUs used | How |
|--------|-----------|-----|
| All 4 core NIMs on 1 GPU (time-slicing) — **default** | 1 | automatic; best on a large card like the 96 GB RTX PRO 6000 |
| 1 physical GPU per core NIM | 4 | `GPU_TIMESLICING_REPLICAS=1` before running bootstrap; needs ≥4 GPUs |

Optional additions (per the [support matrix](../../docs/docs/extraction/prerequisites-support-matrix.md#model-hardware-requirements)):
reranker +1 GPU (or 0 on an ≥80 GB card), in-cluster answer LLM (Super-49B)
**+2 GPUs**, Omni captioning +1 GPU (≥80 GB), Parakeet ASR +1 GPU (not on
Blackwell/H200 NVL). Even "everything on" fits inside 8 GPUs. **Minimum to run
RAG retrieval: 1 GPU.**

> Time-slicing shares a GPU with **no memory isolation** — fine here because the
> core NIMs are tiny. Do not rely on it to isolate large co-tenant models.

---

## 5. What `bootstrap.sh` does (in order)

1. **Pre-flight:** require `NGC_API_KEY`; reject a non-ASCII key (see issue #6);
   require `nvidia-smi`.
2. **k3s + Helm** — single-node cluster (skips install if a cluster is already
   present).
3. **Host GPU runtime** — install `nvidia-container-toolkit` via apt, restart
   k3s so it **natively detects** the `nvidia` runtime, then set `nvidia` as the
   **default** runtime via a `config-v3.toml.d` drop-in. (See issue #4 for why we
   don't use the GPU Operator's toolkit.)
4. **GPU Operator** — latest, `driver.enabled=false`, **`toolkit.enabled=false`**,
   device-plugin with the time-slicing ConfigMap. Then create the
   `/run/nvidia/validations/toolkit-ready` marker so the device-plugin init can
   proceed (see issue #5).
5. **NIM Operator** — installs the `apps.nvidia.com/v1alpha1` CRDs.
6. **NeMo Retriever** — Helm-installs the chart with
   [`values-brev-core.yaml`](./values-brev-core.yaml) + the NGC secrets.
7. **Status** — prints pods, NIMCache/NIMService, and the port-forward/health
   commands.

---

## 6. Troubleshooting log (root cause → fix)

Every issue hit during the first end-to-end bring-up, in order. Fixes marked
**[in script]** are already baked into `bootstrap.sh` / the chart; **[runtime]**
were one-off actions on the live node.

### #1 — Service pod CrashLoop: `ValidationError … embed_model_provider_prefix`
- **Symptom:** `retriever-service` crashlooped with pydantic
  `Extra inputs are not permitted [extra_forbidden]` for
  `nim_endpoints.embed_model_provider_prefix` / `vectordb.embed_model_provider_prefix`.
- **Root cause:** the chart on `main` is ahead of the GA service image `26.5.0`
  and always rendered `embed_model_provider_prefix: null`, a field the older
  image's config model rejects.
- **Fix [in script]:** chart templates render the key only when non-empty
  (`nemo_retriever/helm/templates/configmap.yaml`).

### #2 — NIM pods CrashLoop: `NVIDIA Driver was not detected`
- **Symptom:** all 4 NIM pods crashlooped; logs showed
  `WARNING: The NVIDIA Driver was not detected` and `libnvidia-ml.so.1 not found`.
  The GPU Operator's container-toolkit pod was itself crashlooping with
  `unable to load containerd config: unsupported config version: 3`.
- **Root cause:** GPU Operator **v24.9.1**'s bundled toolkit is too old to parse
  k3s's **containerd 2.x** config (version 3), so it never configured the nvidia
  runtime; NIM containers ran without driver injection.
- **Fix [in script]:** install the **latest** GPU Operator (leave
  `GPU_OPERATOR_VERSION` empty).

### #3 — `defaultRuntimeName: runc` → GPU pods get no GPU
- **Symptom:** even with the nvidia runtime registered, `crictl info` showed the
  default runtime was still `runc`; pods without an explicit runtime got no GPU.
- **Root cause:** k3s registers the `nvidia` runtime but does not make it default.
- **Fix [in script]:** set `default_runtime_name = "nvidia"` via a k3s
  `config-v3.toml.d` drop-in (see #4/#5 for why default runtime, not per-pod
  `runtimeClassName`, is what the device-plugin needs).

### #4 — Node `NotReady` / CNI broken (the big one)
- **Symptom:** `kubectl get nodes` → `NotReady`,
  `container runtime network not ready: cni plugin not initialized`; every pod
  stuck `Pending` (even CPU-only ones).
- **Root cause:** the GPU Operator's toolkit wrote a **stub** `config.toml.tmpl`
  containing only `imports = [...]` + `version = 3` — with no
  `{{ template "base" . }}`, so k3s regenerated a `config.toml` **missing its
  base config**, and the toolkit's runtime drop-in set the CNI paths to the
  generic `/etc/cni/net.d` + `/opt/cni/bin` instead of k3s's
  `/var/lib/rancher/k3s/agent/etc/cni/net.d` + `/var/lib/rancher/k3s/data/cni`.
  containerd then looked for CNI config in an empty directory. The toolkit also
  rewrote the file whenever its pod restarted.
- **Fix [in script]:** **do not use the operator toolkit at all.** Install
  `nvidia-container-toolkit` on the host, let k3s natively detect the `nvidia`
  runtime (k3s writes a correct config with its own CNI paths), and disable the
  operator toolkit (`toolkit.enabled=false`).

> **k3s facts learned here (containerd 2.x):** the template is
> `config-v3.toml.tmpl` (not `config.toml.tmpl`); the import dir is
> `/var/lib/rancher/k3s/agent/etc/containerd/config-v3.toml.d/*.toml`; CNI conf
> lives in `/var/lib/rancher/k3s/agent/etc/cni/net.d` and CNI binaries in
> `/var/lib/rancher/k3s/data/cni`. To customize, add a drop-in — never replace
> the generated `config.toml`.

### #5 — device-plugin advertises 0 GPUs / stuck `Init:0/2`
- **Symptom:** `nvidia.com/gpu` allocatable = `0`; the device-plugin pod hung in
  `Init:0/2` on its `toolkit-validation` init container:
  `until [ -f /run/nvidia/validations/toolkit-ready ]; do … done`.
- **Root cause:** that ready marker is normally written by the operator toolkit,
  which we disabled in #4. Also, with the default runtime still `runc` the
  device-plugin pod itself couldn't see the GPU.
- **Fix [in script]:** (a) set the default runtime to `nvidia` (#3), and (b)
  create `/run/nvidia/validations/toolkit-ready` — the host toolkit + nvidia
  default runtime already *is* the ready runtime stack.

### #6 — Ingestion fails: `UnicodeEncodeError` sending requests to NIMs
- **Symptom:** ingest job `failed` with
  `GraphIngestionError … 'ascii'/'latin-1' codec can't encode characters in
  position 13-14`, from both the page-elements and embedding stages. NIMs
  themselves returned 200 and embedded ASCII input fine.
- **Root cause:** the secret's `NGC_API_KEY` was a **literal placeholder with
  non-ASCII characters** (a copied `nvapi-<non-ASCII>key`). Non-ASCII can't go in
  an HTTP `Authorization` header, so every NIM call the service made crashed.
  (Image/weight pulls still worked because the image-pull secret is base64, which
  tolerates the bytes.)
- **Fix [in script]:** reject a non-ASCII `NGC_API_KEY` in pre-flight. **[runtime]**
  recreate the `ngc-api` secret with the real ASCII key and restart the service.

### #7 — Query returns `502 … ConnectTimeout` to the vectordb
- **Symptom:** after fixing #6, ingestion completed (8 chunks) but `/v1/query`
  returned `502 Failed to reach VectorDB service: ConnectTimeout`.
- **Root cause:** the separate `vectordb` pod (which embeds the query text) still
  held the bad key from #6, and there was a brief connection transient right after
  restarting it.
- **Fix [runtime]:** restart the `vectordb` deployment and retry; the query then
  returned the correct hits.

---

## 7. Key takeaways (for adapting to a new environment)

- **GPU Operator toolkit + k3s (containerd 2.x) = don't.** Use the host
  `nvidia-container-toolkit` + k3s-native runtime detection; keep the operator for
  the device-plugin + validators only (`toolkit.enabled=false`).
- **Make `nvidia` the default runtime** via a `config-v3.toml.d` drop-in so the
  operator's own GPU pods (device-plugin) get the GPU, not just your NIM pods.
- **When you disable the operator toolkit,** create
  `/run/nvidia/validations/toolkit-ready` or the device-plugin init blocks forever.
- **Never replace k3s's generated `config.toml`;** only add drop-ins that include
  the base. A stub template silently strips CNI and bricks the node.
- **The NGC key must be pure ASCII** — a placeholder with non-ASCII characters
  passes image pulls but breaks every NIM call with a `UnicodeEncodeError`.
- **`/run` is tmpfs:** the `toolkit-ready` marker does not survive a reboot. On a
  long-lived node, re-create it (or add a systemd unit) if the box reboots.

---

## 8. End-to-end verification

With the service port-forwarded (`kubectl port-forward -n retriever
svc/retriever-nemo-retriever 7670:7670`):

```bash
cd <repo-root>
pip install -q requests
export RETRIEVER_URL=http://localhost:7670
python3 - <<'PY'
import json, os, time, requests
base = os.environ["RETRIEVER_URL"]
job = requests.post(f"{base}/v1/ingest/job",
    json={"expected_documents":1,"label":"smoke","metadata":{},"retain_results":True}).json()
jid = job["job_id"]
meta = {"filename":"multimodal_test.pdf","content_type":"application/pdf","metadata":{}}
with open("data/multimodal_test.pdf","rb") as f:
    up = requests.post(f"{base}/v1/ingest/job/{jid}/document",
        files={"file":("multimodal_test.pdf",f,"application/pdf")},
        data={"metadata":json.dumps(meta)}).json()
did = up["document_id"]
for _ in range(160):
    docs = requests.get(f"{base}/v1/ingest/job/{jid}/documents", params={"limit":10}).json()
    rec = {d["document_id"]: d for d in docs.get("items", [])}.get(did, {})
    if rec.get("status") in ("completed","failed"): break
    time.sleep(3)
print("ingest:", rec.get("status"), "rows:", rec.get("result_rows"))
q = requests.post(f"{base}/v1/query",
    json={"query":"Which animal is jumping onto a laptop?","top_k":3,"format":"hits"}).json()
for h in (q.get("results") or [{}])[0].get("hits", [])[:3]:
    print("-", str(h.get("text",""))[:120])
PY
```

Expected: `ingest: completed rows: 8` and a hit containing
`Cat | Jumping onto a laptop`.

---

## 9. Teardown & cost

```bash
helm uninstall retriever -n retriever
kubectl delete nimservice,nimcache -n retriever --all   # NIMCaches are kept by default
```

Stop the Brev instance when you're done — an RTX PRO 6000 runs roughly
$2–2.6/hr depending on provider.

---

## 10. Files

| File | Purpose |
|------|---------|
| [`bootstrap.sh`](./bootstrap.sh) | End-to-end single-node install. |
| [`values-brev-core.yaml`](./values-brev-core.yaml) | Helm override: core RAG only. |
| [`notebooks/nemo_retriever_quickstart.ipynb`](./notebooks/nemo_retriever_quickstart.ipynb) | Drives the deployed service: ingest → query → answer. |
| [`README.md`](./README.md) | Short "how to run it" guide. |
| `DEPLOYMENT_NOTES.md` | This document — full reference + troubleshooting log. |

> Reference launchable provided "as is". The GPU-runtime steps were shaped on the
> environment in §2; other Brev images / k3s versions may need adjustment. Secure
> the service (AuthN/AuthZ) before exposing it beyond the node.