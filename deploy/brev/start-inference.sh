#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# NeMo Retriever — host-based inference layer (vLLM + LMCache)
# -----------------------------------------------------------------------------
# Alternative to bootstrap-inference.sh that runs vLLM directly on the host
# instead of k8s pods. Use this when:
#   - The NGC key lacks Container Registry access for the ai-dynamo org, OR
#   - You want a faster / simpler setup without k8s image pulls.
#
# Both inference paths run as host processes (no k8s pods needed).
# The webui /compare page connects to them via localhost:8001 and localhost:8002.
#
# Usage:
# Run this every time the instance restarts:
#   cd ~/NeMo-Retriever
#   git pull                            # pick up any updates
#   export HF_TOKEN=hf_...             # required for Llama (gated model)
#   ./deploy/brev/start-inference.sh 2>&1 | tee ~/inference.log
#
# What this script does end-to-end:
#   0. Wait for NeMo Retriever k3s pods Running (auto after reboot, ~5 min)
#   1. Create/update ~/inference-venv with vllm + lmcache
#   2. Patch vLLM model_runner.py for CacheBlend (VLLMModelTracker)
#   3. GPU sanity check
#   4. Pre-download LLM model weights to HuggingFace cache
#   5. Write ~/lmcache.yaml (LMCache V1 config)
#   6. Kill any old vLLM processes on ports 8001/8002
#   7. Start baseline vLLM (port 8001, APC only) — wait for READY
#   8. Start CacheBlend vLLM (port 8002, APC+LMCache) — sequential after 7
#   9. Restart Deploy UI (port 8000) — its thread auto-manages :7670 port-forward
#  10. Check vectordb; auto re-ingest T2-RAGBench FinQA if empty
#
# Optional env vars:
#   LLM_MODEL          HuggingFace model ID  (default: meta-llama/Llama-3.1-8B-Instruct)
#   GPU_MEM_UTIL       fraction of GPU for each vLLM (default: 0.20)
#   MAX_MODEL_LEN      max context length (default: 4096)
#   HF_TOKEN           HuggingFace token for gated models (e.g. Llama)
#   RETRIEVER_URL      NeMo Retriever URL (default: http://localhost:7670)
# -----------------------------------------------------------------------------
set -euo pipefail

LLM_MODEL="${LLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.20}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
RETRIEVER_URL="${RETRIEVER_URL:-http://localhost:7670}"
# HF cache may live on /ephemeral on Brev instances (larger disk)
HF_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE:-}}"
[[ -z "${HF_HUB_CACHE}" && -d "/ephemeral/cache/huggingface/hub" ]] && \
  HF_HUB_CACHE="/ephemeral/cache/huggingface/hub"
# Only export if non-empty — exporting "" causes huggingface_hub to treat
# CWD as the cache root and dump model files into the repo directory.
[[ -n "${HF_HUB_CACHE}" ]] && export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
[[ -n "${HF_TOKEN:-}" ]] && export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEBUI_DIR="${REPO_ROOT}/deploy/brev/webui"
VENV="${HOME}/inference-venv"
DEPLOY_VENV="${HOME}/deploy-ui-venv"

LMCACHE_CONFIG="${HOME}/lmcache.yaml"
BASELINE_LOG="${HOME}/vllm-baseline.log"
CACHEBLEND_LOG="${HOME}/vllm-cacheblend.log"

log()  { echo -e "\n\033[1;32m==> $*\033[0m"; }
warn() { echo -e "\n\033[1;33mWARN: $*\033[0m"; }
die()  { echo -e "\n\033[1;31mERROR: $*\033[0m" >&2; exit 1; }

log "Inference layer bootstrap (host-based)"
log "  model          : ${LLM_MODEL}"
log "  gpu-mem-util   : ${GPU_MEM_UTIL} per path"
log "  max-model-len  : ${MAX_MODEL_LEN}"
[[ -n "${HUGGINGFACE_HUB_CACHE:-}" ]] && log "  hf cache       : ${HUGGINGFACE_HUB_CACHE}"
[[ -n "${HF_TOKEN:-}" ]]              && log "  hf token       : set"

# ── 0. Wait for NeMo Retriever k3s pods to be Running ────────────────────────
# On instance restart k3s comes back automatically but pods take a few minutes.
# vLLM needs the GPU budget that NIMs occupy to be stable before we calculate
# gpu-memory-utilization headroom.
log "Waiting for NeMo Retriever k3s pods to be Running (max 15 min)"
KUBECTL="kubectl"
[[ -f /etc/rancher/k3s/k3s.yaml ]] && export KUBECONFIG="/etc/rancher/k3s/k3s.yaml"
PODS_DEADLINE=$(( SECONDS + 900 ))

_retriever_ready() {
    local running
    running=$("${KUBECTL}" get pods -n retriever 2>/dev/null \
        | grep -c "1/1.*Running" || true)
    [[ "${running}" -ge 4 ]]   # embed + ocr + vectordb + retriever-main
}

if ! _retriever_ready; then
    log "  Pods not yet ready — waiting…"
    while [[ $SECONDS -lt $PODS_DEADLINE ]]; do
        if _retriever_ready; then
            break
        fi
        RUNNING=$("${KUBECTL}" get pods -n retriever 2>/dev/null | grep -c "1/1.*Running" || true)
        echo "  [$(date +%H:%M:%S)] ${RUNNING} pods Running (need ≥4)…"
        sleep 20
    done
    if ! _retriever_ready; then
        warn "NeMo Retriever pods did not reach Running in 15 min — proceeding anyway"
        "${KUBECTL}" get pods -n retriever 2>/dev/null | grep -v Completed | grep -v Evicted || true
    fi
fi
log "  NeMo Retriever pods: READY"

# ── 1. Python venv ────────────────────────────────────────────────────────────
log "Setting up inference venv at ${VENV}"
if [[ ! -d "${VENV}" ]]; then
    python3 -m venv "${VENV}"
fi

# ── 2. Install vLLM and LMCache ───────────────────────────────────────────────
log "Installing vLLM and LMCache (first run: 5-15 min; subsequent runs are instant)"
"${VENV}/bin/pip" install --quiet --upgrade pip
"${VENV}/bin/pip" install --quiet vllm lmcache

# Print versions for debugging
log "Installed versions:"
"${VENV}/bin/python" -c "import vllm; print(f'  vLLM     : {vllm.__version__}')" 2>/dev/null || true
"${VENV}/bin/python" -c "import lmcache; print(f'  LMCache  : {lmcache.__version__}')" 2>/dev/null || true

# ── 2.5. Patch vLLM model_runner for CacheBlend ──────────────────────────────
# vLLM 0.25 never calls VLLMModelTracker.register_model(), which CacheBlend
# requires to find the model instance. Apply a one-time idempotent patch after
# every pip install so reinstalls don't silently break port 8002.
log "Patching vLLM model_runner.py for CacheBlend (VLLMModelTracker)"
RUNNER="$(find "${VENV}/lib" -name "model_runner.py" \
          -path "*/vllm/v1/worker/gpu/model_runner.py" 2>/dev/null | head -1)"
if [[ -z "${RUNNER}" ]]; then
    warn "model_runner.py not found — skipping patch (CacheBlend may fail)"
elif grep -q "VLLMModelTracker" "${RUNNER}" 2>/dev/null; then
    log "  patch already present, skipping"
else
    "${VENV}/bin/python" - "${RUNNER}" <<'PYEOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text(encoding="utf-8")
ANCHOR = "        if self.lora_config:"
PATCH = (
    "        # Register model with LMCache so CacheBlend can find it.\n"
    "        # vLLM 0.25 never calls this; without it CacheBlend raises\n"
    "        # ValueError: vllm model for vllm-instance not found.\n"
    "        try:\n"
    "            from lmcache.v1.compute.models.utils import VLLMModelTracker\n"
    "            VLLMModelTracker.register_model(\"vllm-instance\", self.model)\n"
    "        except Exception:\n"
    "            pass\n"
    "        if self.lora_config:"
)
if ANCHOR not in src:
    print(f"WARNING: patch anchor not found in {p} — vLLM version may have changed", flush=True)
    sys.exit(0)
p.write_text(src.replace(ANCHOR, PATCH, 1), encoding="utf-8")
print(f"  patch applied to {p}", flush=True)
PYEOF
    log "  patch applied"
fi

# ── 3. GPU sanity check ────────────────────────────────────────────────────────
log "GPU memory status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
  --format=csv,noheader 2>/dev/null || warn "nvidia-smi not available"

# ── 4. Pre-download model into HuggingFace cache ──────────────────────────────
# Downloading before starting both vLLM instances avoids race conditions and
# ensures both pick up from the same local cache directory.
log "Pre-downloading model ${LLM_MODEL} to HuggingFace cache"
HF_HUB_ARGS=(
    "${LLM_MODEL}"
    --local-dir-use-symlinks False
)
[[ -n "${HF_TOKEN:-}" ]] && export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
"${VENV}/bin/python" -c "
from huggingface_hub import snapshot_download
import os
token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
print(f'Downloading {\"${LLM_MODEL}\"} (token: {\"yes\" if token else \"no — using public access\"})')
path = snapshot_download(
    '${LLM_MODEL}',
    ignore_patterns=['*.bin', 'original/*'],  # prefer safetensors
    token=token,
)
print(f'Model cached at: {path}')
" 2>&1 || warn "Model pre-download failed — vLLM will download on first start (slower)"

# ── 5. LMCache configuration ──────────────────────────────────────────────────
log "Writing LMCache V1 config to ${LMCACHE_CONFIG}"
cat > "${LMCACHE_CONFIG}" << 'YAML_EOF'
# LMCache V1 engine configuration (lmcache 0.5+).
#
# CacheBlend reuses cached KV chunks for retrieved document segments even when
# they appear at different positions across requests — complementing APC which
# only handles contiguous prefix matches.
chunk_size: 256

# CPU offload: store KV chunk states in host RAM between requests.
# Enables warm-cache reuse across multiple queries.
local_cpu: true
max_local_cpu_size: 20

# Enable CacheBlend: reuse cached KV at non-prefix positions.
enable_blending: true

# Minimum tokens in a chunk before blending is considered.
blend_min_tokens: 128
YAML_EOF

# ── 6. Kill any existing vLLM processes ───────────────────────────────────────
log "Stopping any existing vLLM processes on ports 8001 and 8002"
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
# Also kill by port in case the process name differs
fuser -k 8001/tcp 2>/dev/null || true
fuser -k 8002/tcp 2>/dev/null || true
sleep 2

# ── 7. Start baseline path: vLLM + APC ────────────────────────────────────────
log "Starting baseline path: vLLM + APC → port 8001"
log "  log: ${BASELINE_LOG}"
VLLM_USE_FLASHINFER_SAMPLER=0 \
HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
nohup "${VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "${LLM_MODEL}" \
    --host 0.0.0.0 \
    --port 8001 \
    --enable-prefix-caching \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --served-model-name baseline \
    --max-num-seqs 8 \
    --no-enable-log-requests \
    > "${BASELINE_LOG}" 2>&1 &
BASELINE_PID=$!
echo "  PID: ${BASELINE_PID}"

# ── 7.5. Wait for baseline to be ready before starting CacheBlend ─────────────
# IMPORTANT: do NOT start both vLLM instances simultaneously.
# Both load ~16 GB of model weights; concurrent loading exhausts GPU memory
# and causes the k8s NIMs to be evicted (disk/memory pressure).
log "Waiting for baseline (port 8001) to finish loading before starting CacheBlend…"
WAIT_DEADLINE=$(( SECONDS + 1200 ))  # 20 min max
while [[ $SECONDS -lt $WAIT_DEADLINE ]]; do
    if curl -sf http://localhost:8001/health >/dev/null 2>&1; then
        log "  baseline READY — starting CacheBlend"
        break
    fi
    sleep 10
done
if ! curl -sf http://localhost:8001/health >/dev/null 2>&1; then
    die "Baseline did not become ready within 20 min. Check ${BASELINE_LOG}"
fi

# ── 8. Start CacheBlend path: vLLM + APC + LMCache ───────────────────────────
log "Starting CacheBlend path: vLLM + APC + LMCache → port 8002"
log "  log: ${CACHEBLEND_LOG}"
log "  lmcache config: ${LMCACHE_CONFIG}"
VLLM_USE_FLASHINFER_SAMPLER=0 \
LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG}" \
HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
nohup "${VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "${LLM_MODEL}" \
    --host 0.0.0.0 \
    --port 8002 \
    --enable-prefix-caching \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --served-model-name cacheblend \
    --max-num-seqs 8 \
    --no-enable-log-requests \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    > "${CACHEBLEND_LOG}" 2>&1 &
CACHEBLEND_PID=$!
echo "  PID: ${CACHEBLEND_PID}"

# ── 9. Restart Deploy UI with inference endpoints ─────────────────────────────
log "Restarting Deploy UI with inference env vars (INFERENCE_MODE=host)"
pkill -f "uvicorn app:app" 2>/dev/null || true
sleep 2
cd "${WEBUI_DIR}"
INFERENCE_MODE=host \
BASELINE_LLM_URL="http://localhost:8001" \
CACHEBLEND_LLM_URL="http://localhost:8002" \
LLM_MODEL="${LLM_MODEL}" \
RETRIEVER_URL="${RETRIEVER_URL}" \
KUBECONFIG="/etc/rancher/k3s/k3s.yaml" \
nohup "${DEPLOY_VENV}/bin/python" -m uvicorn app:app \
    --host 0.0.0.0 --port 8000 \
    > "${HOME}/deploy-ui.log" 2>&1 &
cd - >/dev/null
sleep 3

# ── 10. Wait for models to load ───────────────────────────────────────────────
log "Waiting for both vLLM instances to load (model download + GPU warm-up)"
log "  First run: 5-20 min (model download). Subsequent runs: 1-3 min."
log "  Monitor progress:"
log "    tail -f ${BASELINE_LOG}"
log "    tail -f ${CACHEBLEND_LOG}"
log ""
log "Polling for readiness (max 30 min) — Ctrl-C to bail out early"

BASELINE_OK=false
CACHEBLEND_OK=false
DEADLINE=$(( SECONDS + 1800 ))  # 30 min

while [[ $SECONDS -lt $DEADLINE ]]; do
    if ! ${BASELINE_OK}; then
        if curl -sf http://localhost:8001/health >/dev/null 2>&1; then
            BASELINE_OK=true
            log "  baseline (port 8001): READY"
        fi
    fi
    if ! ${CACHEBLEND_OK}; then
        if curl -sf http://localhost:8002/health >/dev/null 2>&1; then
            CACHEBLEND_OK=true
            log "  cacheblend (port 8002): READY"
        fi
    fi
    ${BASELINE_OK} && ${CACHEBLEND_OK} && break
    sleep 15
    # Print progress from logs (baseline and cacheblend)
    B_SHARDS=$(grep -c "Loading safetensors checkpoint" "${BASELINE_LOG}" 2>/dev/null || echo 0)
    C_SHARDS=$(grep -c "Loading safetensors checkpoint" "${CACHEBLEND_LOG}" 2>/dev/null || echo 0)
    C_STATUS="loading"
    grep -q "Application startup complete" "${CACHEBLEND_LOG}" 2>/dev/null && C_STATUS="READY"
    grep -qE "ERROR|failed|Traceback" "${CACHEBLEND_LOG}" 2>/dev/null && C_STATUS="ERROR"
    echo "  [$(date +%H:%M:%S)] baseline:${BASELINE_OK} cb:${C_STATUS}(shards=${C_SHARDS}/${B_SHARDS})"
done

# ── 10.5. Check vectordb and re-ingest if empty ───────────────────────────────
# The vectordb PVC uses local-path storage → data survives pod restarts on the
# same instance. But if the instance was rebuilt or the PVC was lost, the index
# is empty and retrieval returns 0 hits. Detect this and offer to re-ingest.
if ${BASELINE_OK} && ${CACHEBLEND_OK}; then
    log "Checking vectordb for ingested documents…"
    # Wait for webui port-forward to the retriever to come up (managed by app.py)
    RETRIEVER_WAIT=0
    while [[ ${RETRIEVER_WAIT} -lt 60 ]]; do
        if curl -sf "${RETRIEVER_URL}/v1/health" >/dev/null 2>&1; then break; fi
        sleep 5; RETRIEVER_WAIT=$(( RETRIEVER_WAIT + 5 ))
    done

    if curl -sf "${RETRIEVER_URL}/v1/health" >/dev/null 2>&1; then
        # A quick retrieval probe — if we get 0 hits the index is probably empty
        HITS=$(curl -sf -X POST "${RETRIEVER_URL}/v1/query" \
            -H "Content-Type: application/json" \
            -d '{"query":"test","top_k":1,"format":"hits"}' 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(len((d.get('results') or [{}])[0].get('hits',[])))" 2>/dev/null || echo "0")
        if [[ "${HITS}" == "0" ]]; then
            warn "Vectordb appears empty (0 hits). Re-ingesting T2-RAGBench FinQA dataset…"
            if [[ -d "${HOME}/t2-ragbench/data/FinQA" ]]; then
                RETRIEVER_URL="${RETRIEVER_URL}" \
                "${DEPLOY_VENV}/bin/python" \
                    "${WEBUI_DIR}/benchmark.py" quick --no-qa 2>&1 \
                    | grep -E "ingested|Ingesting|chunks|Hit@|failed|ERROR" || true
                log "  Re-ingest complete."
            else
                warn "Dataset not found at ${HOME}/t2-ragbench — download with:"
                warn "  ${DEPLOY_VENV}/bin/python ${REPO_ROOT}/deploy/brev/download_dataset.py quick"
                warn "  RETRIEVER_URL=${RETRIEVER_URL} ${DEPLOY_VENV}/bin/python ${WEBUI_DIR}/benchmark.py quick --no-qa"
            fi
        else
            log "  Vectordb OK (${HITS} hits on probe query) — skipping re-ingest."
        fi
    else
        warn "Retriever not reachable at ${RETRIEVER_URL} — skipping vectordb check."
        warn "The webui port-forward manager will retry in the background."
    fi
fi

# ── 11. Final status ──────────────────────────────────────────────────────────
echo ""
echo "============================================================================="
echo "Inference layer — status"
echo "  Baseline   (vLLM + APC)       : ${BASELINE_OK}  → http://localhost:8001"
echo "  CacheBlend (vLLM+APC+LMCache) : ${CACHEBLEND_OK}  → http://localhost:8002"
echo ""
echo "Side-by-side comparison UI:"
echo "  http://localhost:8000/compare"
echo ""
echo "Quick smoke tests:"
echo "  curl http://localhost:8001/v1/models"
echo "  curl http://localhost:8002/v1/models"
echo ""
echo "KV cache metrics (after a few requests):"
echo "  curl -s http://localhost:8001/metrics | grep -E 'prefix_cache|gpu_cache'"
echo "  curl -s http://localhost:8002/metrics | grep -E 'prefix_cache|lmcache'"
echo ""
echo "Logs:"
echo "  tail -f ${BASELINE_LOG}"
echo "  tail -f ${CACHEBLEND_LOG}"
echo "  tail -f ${HOME}/deploy-ui.log"
echo ""
if ${BASELINE_OK} && ${CACHEBLEND_OK}; then
    echo "Both paths READY. Open /compare in your browser."
else
    echo "Still loading — check the logs above."
    echo "Run: tail -f ${BASELINE_LOG} ${CACHEBLEND_LOG}"
fi
echo "============================================================================="