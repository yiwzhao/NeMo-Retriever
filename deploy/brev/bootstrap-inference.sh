#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# NeMo Retriever — inference layer bootstrap
# -----------------------------------------------------------------------------
# Deploys two LLM serving paths on top of the existing k3s retriever stack:
#
#   Baseline path  : vLLM + Automatic Prefix Caching (APC)
#                    svc/inference-baseline → localhost:8001
#   CacheBlend path: vLLM + APC + LMCache + CacheBlend (NVIDIA Dynamo image)
#                    svc/inference-cacheblend → localhost:8002
#
# Run after bootstrap.sh, using the same pattern:
#
#   cd ~/NeMo-Retriever
#   git pull
#   export NGC_API_KEY='nvapi-...'
#   export HF_TOKEN='hf_...'          # required for Llama 3.1 (HF-gated model)
#   ./deploy/brev/bootstrap-inference.sh 2>&1 | tee ~/bootstrap-inference.log
#
# Optional env vars:
#   LLM_MODEL     HuggingFace model ID  (default: meta-llama/Llama-3.1-8B-Instruct)
#   DYNAMO_IMAGE  NGC image ref         (default: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1)
#   NS            Kubernetes namespace  (default: retriever)
# -----------------------------------------------------------------------------
set -euo pipefail

NS="${NS:-retriever}"
LLM_MODEL="${LLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
# NGC catalog: https://catalog.ngc.nvidia.com/orgs/nvidia/ai-dynamo/containers/vllm
# Latest tag as of 2026-07-17: 1.2.1
DYNAMO_IMAGE="${DYNAMO_IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DYNAMO_DIR="${REPO_ROOT}/deploy/brev/dynamo"
WEBUI_DIR="${REPO_ROOT}/deploy/brev/webui"
VENV="${HOME}/deploy-ui-venv"

log()  { echo -e "\n\033[1;32m==> $*\033[0m"; }
warn() { echo -e "\n\033[1;33mWARN: $*\033[0m"; }
die()  { echo -e "\n\033[1;31mERROR: $*\033[0m" >&2; exit 1; }

export KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"

# ── pre-flight checks ─────────────────────────────────────────────────────────
[[ -n "${NGC_API_KEY:-}" ]] || die "NGC_API_KEY is not set. export NGC_API_KEY=nvapi-..."
if printf '%s' "${NGC_API_KEY}" | LC_ALL=C grep -q '[^ -~]'; then
  die "NGC_API_KEY contains non-ASCII characters. Use your real nvapi-... key."
fi

[[ -n "${HF_TOKEN:-}" ]] || \
  warn "HF_TOKEN is not set. Llama 3.1 is a gated model — set HF_TOKEN=hf_... or the pods will fail to download weights."

command -v kubectl >/dev/null   || die "kubectl not found — run bootstrap.sh first."
kubectl get nodes >/dev/null 2>&1 || die "Kubernetes not reachable (KUBECONFIG=${KUBECONFIG})."

log "Inference layer bootstrap"
log "  model  : ${LLM_MODEL}"
log "  image  : ${DYNAMO_IMAGE}"
log "  ns     : ${NS}"

# ── 1. Check GPU availability ─────────────────────────────────────────────────
AVAILABLE_GPUS=$(kubectl get nodes \
  -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}' 2>/dev/null \
  | tr ' ' '\n' | awk '{s+=$1}END{print s+0}')
log "nvidia.com/gpu units schedulable: ${AVAILABLE_GPUS}"
if [[ "${AVAILABLE_GPUS}" -lt 2 ]]; then
  warn "Only ${AVAILABLE_GPUS} GPU unit(s) available; each inference path needs 1."
  warn "If GPU_TIMESLICING_REPLICAS<2 in bootstrap.sh, re-run it with a higher value."
fi

# ── 2. Pull the Dynamo image directly into k3s containerd ────────────────────
# k3s bundles its own containerd; `k3s ctr` is the authoritative CLI for it.
# Pulling here means the pods start immediately without waiting for an image pull.
log "Pulling Dynamo image into k3s containerd k8s.io namespace (12 GB; may take 5-15 min first time)"
# Pods run in the k8s.io containerd namespace; pull there so pods start instantly.
if sudo k3s ctr --namespace k8s.io images ls 2>/dev/null | grep -q "${DYNAMO_IMAGE}"; then
  log "  Image already present in k8s.io namespace — skipping pull."
else
  sudo k3s ctr --namespace k8s.io images pull \
    --user "\$oauthtoken:${NGC_API_KEY}" \
    "${DYNAMO_IMAGE}" \
    || warn "Image pull failed — pods will pull it themselves on startup (slower first start)."
fi

# ── 3. Secrets ────────────────────────────────────────────────────────────────
log "Ensuring NGC image pull secret"
# The core stack already created this secret; re-apply is a no-op if it exists.
kubectl create secret docker-registry retriever-ngc-image-pull-secret \
  -n "${NS}" \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="${NGC_API_KEY}" \
  --dry-run=client -o yaml | kubectl apply -f -

log "Creating inference-config secret (model + HF token)"
kubectl create secret generic inference-config \
  -n "${NS}" \
  --from-literal=llm_model="${LLM_MODEL}" \
  --from-literal=hf_token="${HF_TOKEN:-}" \
  --dry-run=client -o yaml | kubectl apply -f -

# ── 4. K8s manifests ──────────────────────────────────────────────────────────
log "Applying LMCache ConfigMap"
kubectl apply -f "${DYNAMO_DIR}/lmcache-configmap.yaml"

log "Deploying baseline path (vLLM + APC) → svc/inference-baseline:8001"
sed "s|nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1|${DYNAMO_IMAGE}|g" \
  "${DYNAMO_DIR}/baseline-deployment.yaml" | kubectl apply -f -

log "Deploying CacheBlend path (vLLM + APC + LMCache) → svc/inference-cacheblend:8002"
sed "s|nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1|${DYNAMO_IMAGE}|g" \
  "${DYNAMO_DIR}/cacheblend-deployment.yaml" | kubectl apply -f -

# ── 5. Wait for rollout ───────────────────────────────────────────────────────
# Model weights are downloaded inside the pod on first start (~10-30 min for 8B).
# --timeout=60m lets the script block until ready; Ctrl-C to bail out early
# and check with `kubectl get pods -n retriever -w`.
log "Waiting for inference-baseline to be Ready (model download on first run may take a while)"
kubectl rollout status deployment/inference-baseline -n "${NS}" --timeout=60m \
  || warn "inference-baseline rollout timed out — check: kubectl logs -n ${NS} -l app=inference-baseline"

log "Waiting for inference-cacheblend to be Ready"
kubectl rollout status deployment/inference-cacheblend -n "${NS}" --timeout=60m \
  || warn "inference-cacheblend rollout timed out — check: kubectl logs -n ${NS} -l app=inference-cacheblend"

# ── 6. Port-forwards (self-healing via the webUI; also start one-shot here) ──
log "Starting port-forwards: 8001 → inference-baseline, 8002 → inference-cacheblend"
# Kill stale forwards from a previous run
pkill -f "kubectl port-forward.*inference-baseline"   2>/dev/null || true
pkill -f "kubectl port-forward.*inference-cacheblend" 2>/dev/null || true
sleep 2

nohup kubectl port-forward -n "${NS}" svc/inference-baseline   8001:8001 \
  >"${HOME}/pf-inference-baseline.log"   2>&1 &
nohup kubectl port-forward -n "${NS}" svc/inference-cacheblend 8002:8002 \
  >"${HOME}/pf-inference-cacheblend.log" 2>&1 &

# ── 7. Restart Deploy UI with inference env vars ──────────────────────────────
# setup.sh starts the UI without knowing about the inference endpoints.
# Kill it and re-launch with the new env vars so /compare and /api/rag/stream work.
log "Restarting Deploy UI with inference endpoints wired in"
pkill -f "uvicorn app:app" 2>/dev/null || true
sleep 2

cd "${WEBUI_DIR}"
BASELINE_LLM_URL="http://localhost:8001" \
CACHEBLEND_LLM_URL="http://localhost:8002" \
LLM_MODEL="${LLM_MODEL}" \
RETRIEVER_URL="${RETRIEVER_URL:-http://localhost:7670}" \
nohup "${VENV}/bin/python" -m uvicorn app:app \
  --host 0.0.0.0 --port 8000 \
  >"${HOME}/deploy-ui.log" 2>&1 &
cd - >/dev/null

# ── 8. Quick health check ─────────────────────────────────────────────────────
log "Waiting for port-forwards to settle (15s)…"
sleep 15
BASELINE_OK=false; CACHEBLEND_OK=false
for _ in $(seq 1 6); do
  curl -sf http://localhost:8001/health >/dev/null 2>&1 && BASELINE_OK=true
  curl -sf http://localhost:8002/health >/dev/null 2>&1 && CACHEBLEND_OK=true
  ${BASELINE_OK} && ${CACHEBLEND_OK} && break
  sleep 5
done

# ── 9. Summary ────────────────────────────────────────────────────────────────
log "Inference pods:"
kubectl get pods -n "${NS}" -l 'inference-path in (baseline,cacheblend)'

cat <<EOF

------------------------------------------------------------------------------
Inference layer bootstrap complete.

  Baseline   (vLLM + APC)       : ${BASELINE_OK}   → http://localhost:8001
  CacheBlend (vLLM + APC + LMCache): ${CACHEBLEND_OK}  → http://localhost:8002

Side-by-side comparison UI:
  http://localhost:8000/compare      (open via Brev Secure Link for port 8000)

Quick smoke test:
  curl http://localhost:8001/v1/models
  curl http://localhost:8002/v1/models

KV cache metrics (Prometheus):
  curl -s http://localhost:8001/metrics | grep -E 'prefix_cache|gpu_cache'
  curl -s http://localhost:8002/metrics | grep -E 'prefix_cache|lmcache'

Logs:
  tail -f ${HOME}/pf-inference-baseline.log
  tail -f ${HOME}/pf-inference-cacheblend.log
  tail -f ${HOME}/deploy-ui.log

If pods are still pulling model weights (~10–30 min for Llama-3.1-8B):
  kubectl get pods -n ${NS} -w -l 'inference-path in (baseline,cacheblend)'
  kubectl logs -n ${NS} -l app=inference-cacheblend -f

Cold vs warm cache:
  The first 1–2 requests are cold — KV states not yet cached.
  Send the same (or overlapping) question 2–3 times to warm the cache,
  then compare TTFT and latency between the two panels.
------------------------------------------------------------------------------
EOF
