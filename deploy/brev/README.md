<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Retriever on Brev — single-node launchable (Core RAG)

A one-command bootstrap that stands up the **core** NeMo Retriever stack on a
single Brev GPU node (single-node Kubernetes), plus a notebook that drives the
running service over its REST API.

Everything on Brev is **single node**, so you are capped at the node's GPUs
(up to **8**). This launchable runs the core stack on a **single GPU** (via
time-slicing) and leaves headroom to add the reranker / answer LLM later.

---

## 0. Deploy in one click (recommended)

No terminal needed. The launchable ships a small **Deploy UI** that runs the
whole bootstrap for you and gives you an ingest + query playground when it's done.

**Set up the Brev launchable (VM Mode):**

1. **Code files** → point at this repo (`https://github.com/<you>/NeMo-Retriever.git`)
   so Brev clones everything, *or* leave it and let the setup script clone it.
2. **Setup script** → paste the contents of [`setup.sh`](./setup.sh) (it clones the
   repo and starts the Deploy UI + Jupyter — no secret needed here).
3. **Expose ports** → `8000` (Deploy UI) and `8888` (Jupyter).

**Then, as a user:**

1. Open the **Deploy UI** (Brev Secure Link for port `8000`).
2. Paste your **NGC API key** and click **Deploy Launchable**.
3. Watch the live activity log while it installs k3s + the GPU stack + NeMo
   Retriever (first run pulls weights, 10–30 min).
4. When it's live, use the built-in **ingest + query playground**, or open the
   full notebook for the single-doc + scaled-corpus walkthrough.

Everything below is the **manual/terminal path** and the reference details behind
what the Deploy UI does.

## 1. Components & services

NeMo Retriever = a **FastAPI orchestrator** (CPU) + a set of **NVIDIA NIM
microservices** (GPU), reconciled on Kubernetes by the **NIM Operator**, with a
**LanceDB** pod for the vector index. Mapping the RAG stages to services:

| Stage | Service / component | GPU? | In this launchable |
|-------|--------------------|------|--------------------|
| **Orchestration** | `retriever-service` (FastAPI + Ray workers) | ❌ CPU (CPU/RAM heavy) | ✅ |
| **Ingestion → extract** | `page-elements` NIM (layout) | ✅ | ✅ core |
| | `table-structure` NIM | ✅ | ✅ core |
| | `ocr` NIM | ✅ | ✅ core |
| **Embedding** | `vlm_embed` NIM (`llama-nemotron-embed-vl-1b-v2`) | ✅ | ✅ core |
| **Indexing** | LanceDB `vectordb` pod (+ PVC) | ❌ CPU | ✅ core |
| **Query / retrieval** | `/v1/query` on the service → embeds query (vlm_embed) → LanceDB search | ✅ (reuses embed NIM) | ✅ core |
| **Reranking** | `rerankqa` NIM (`llama-nemotron-rerank-vl-1b-v2`) | ✅ | ⛔ optional (off) |
| **Answer / LLM serving** | `answer_llm` NIM (Super-49B) **or** external API | ✅✅ (2 GPU) / — | ⛔ external by default |
| Audio/video | `audio` Parakeet ASR NIM | ✅ | ⛔ optional (off) |
| Captioning | Omni 30B NIM | ✅ | ⛔ optional (off) |
| Alt. PDF parse | `nemotron_parse` NIM | ✅ | ⛔ optional (off) |

> **LLM serving is not part of the Retriever core.** Retrieval returns chunks;
> generation is a separate concern. This launchable expects you to call an
> external OpenAI-compatible LLM (e.g. build.nvidia.com) from the client — the
> notebook shows the pattern. You can instead host it in-cluster (see below),
> which costs ~2 GPUs.

Sources: [`nemo_retriever/helm/README.md`](../../nemo_retriever/helm/README.md),
[`docs/docs/extraction/prerequisites-support-matrix.md`](../../docs/docs/extraction/prerequisites-support-matrix.md),
[`docs/docs/extraction/deployment-options.md`](../../docs/docs/extraction/deployment-options.md).

## 2. GPU budget (8-GPU single node)

The four core NIMs total only **~4.8 GiB** of weights combined, so they fit on a
single GPU. `bootstrap.sh` configures **GPU time-slicing** by default so one
physical GPU advertises several schedulable `nvidia.com/gpu` units and hosts all
four NIMs:

| Layout | GPUs used | When |
|--------|-----------|------|
| **All 4 core NIMs on 1 GPU** via time-slicing (**default**) | **1 / 8** | Best value on a large card (e.g. one 96 GiB RTX PRO 6000). Configured automatically. |
| **1 GPU per core NIM** | **4 / 8** | Set `GPU_TIMESLICING_REPLICAS=1` before running bootstrap; needs ≥4 physical GPUs. |

Adding optional pieces later (per the [support matrix](../../docs/docs/extraction/prerequisites-support-matrix.md#model-hardware-requirements)):

- **Reranker**: +1 GPU (or 0 if co-located on an ≥80 GB GPU with the core NIMs).
- **In-cluster answer LLM (Super-49B)**: **+2 GPUs**.
- **Omni captioning**: +1 GPU (≥80 GB).
- **Parakeet ASR** (audio/video): +1 GPU (not supported on Blackwell/H200 NVL).

So even the "everything on" configuration (4 core + 1 rerank + 2 LLM + caption + ASR) fits
inside 8 GPUs. **Minimum to run RAG retrieval: 1 GPU.**

## 3. Prerequisites

- A Brev GPU instance — **one large GPU is enough** (e.g. a single 96 GiB
  RTX PRO 6000; time-slicing packs all four core NIMs onto it) — with the
  **NVIDIA driver installed** (`nvidia-smi` works) and outbound network to
  `nvcr.io`, `helm.ngc.nvidia.com`, and (for hosted answers) `build.nvidia.com`.
- An **NGC API key** — <https://org.ngc.nvidia.com/setup/api-keys>. It pulls the
  service image and the NIM model weights.
- ~150 GB free disk for NIM model caches (see support matrix).

## 4. Launch

```bash
# from the repo root
export NGC_API_KEY=nvapi-xxxxxxxx
./deploy/brev/bootstrap.sh
```

`bootstrap.sh` installs, in order: **k3s** (single-node k8s) → **host
nvidia-container-toolkit** + k3s-native runtime detection + `nvidia` set as the
default containerd runtime → **NVIDIA GPU Operator** (device-plugin +
time-slicing; its own toolkit disabled) → **NVIDIA NIM Operator** (CRDs) → the
repo **Helm chart** with [`values-brev-core.yaml`](./values-brev-core.yaml).

> **Why the host toolkit instead of the GPU Operator's?** On k3s (containerd
> 2.x) the GPU Operator's bundled container-toolkit misconfigures containerd and
> breaks CNI. This launchable installs the toolkit on the host, lets k3s detect
> the `nvidia` runtime natively, and sets it as the default runtime. Verified on
> **RTX PRO 6000 (Blackwell) + k3s v1.36 / containerd 2.x**.

First run downloads model weights to PVCs and can take 10–30 min. Watch:

```bash
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
kubectl get pods,nimcache,nimservice -n retriever -w
```

When the service and the four `nimservice` resources are Ready, expose it:

```bash
kubectl port-forward -n retriever svc/retriever-nemo-retriever 7670:7670
curl -s http://localhost:7670/v1/health
```

Then open **[`notebooks/nemo_retriever_quickstart.ipynb`](./notebooks/nemo_retriever_quickstart.ipynb)**
and run it top-to-bottom: health → ingest a multimodal PDF → inspect extraction →
query → (optional) generate an answer.

```bash
export RETRIEVER_URL=http://localhost:7670
# optional, for the answer step:
export NVIDIA_API_KEY=nvapi-xxxxxxxx
```

## 5. Variations

**Tune or disable GPU time-slicing.** `bootstrap.sh` advertises
`GPU_TIMESLICING_REPLICAS` (default **8**) schedulable units per physical GPU via
a `time-slicing-config` ConfigMap consumed by the GPU Operator device plugin.
Raise it if you add GPU-needing pods, or set `GPU_TIMESLICING_REPLICAS=1` to give
each core NIM its own physical GPU (then use a ≥4-GPU instance). Time-slicing
shares a GPU without memory isolation — fine here because the core NIMs are tiny.
See the [GPU Operator time-slicing docs](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html).

**Host the answer LLM in-cluster (+2 GPUs).** Re-run the last helm step (or
`helm upgrade`) adding:

```bash
--set nimOperator.answer_llm.enabled=true
```

The service then serves `POST /v1/answer` end-to-end (Option B in the notebook).

**Add the reranker.** `--set nimOperator.rerankqa.enabled=true` (needs an ≥80 GB
GPU to co-reside with the core pipeline, else a dedicated GPU).

**Enable audio/video ingestion.** `--set nimOperator.audio.enabled=true`
`--set service.installFfmpeg=true`
`--set serviceConfig.nimEndpoints.audioGrpcEndpoint=audio:50051` (H100/A100 GPU;
not supported on Blackwell/H200 NVL).

## 6. Teardown

```bash
helm uninstall retriever -n retriever
kubectl delete nimservice,nimcache -n retriever --all   # NIMCaches are kept by default
```

## Files

| File | Purpose |
|------|---------|
| [`setup.sh`](./setup.sh) | Brev setup script: clones the repo and starts the Deploy UI + Jupyter (no secret). |
| [`webui/`](./webui/) | One-click Deploy UI (FastAPI): NGC key → runs `bootstrap.sh` with a live log → ingest + query playground. |
| [`bootstrap.sh`](./bootstrap.sh) | End-to-end single-node install (k3s + GPU runtime + GPU Operator + NIM Operator + chart). |
| [`values-brev-core.yaml`](./values-brev-core.yaml) | Helm override: core RAG only, optional NIMs off. |
| [`notebooks/nemo_retriever_quickstart.ipynb`](./notebooks/nemo_retriever_quickstart.ipynb) | Drives the deployed service: single doc → query → answer → scaled multi-doc corpus. |
| [`DEPLOYMENT_NOTES.md`](./DEPLOYMENT_NOTES.md) | Full reference: component/GPU details, verified environment, and the complete troubleshooting log. |

> This is a reference launchable provided "as is". Bootstrap steps (k8s distro,
> GPU runtime) may need adjustment for your specific Brev image. Secure the
> service (AuthN/AuthZ) before exposing it beyond the node.
