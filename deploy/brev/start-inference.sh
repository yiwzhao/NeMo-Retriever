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
#   cd ~/NeMo-Retriever
#   git pull
#   # Optional: set a different model (default: Qwen/Qwen2.5-7B-Instruct)
#   # export LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct HF_TOKEN=hf_...
#   ./deploy/brev/start-inference.sh 2>&1 | tee ~/inference.log
#
# Optional env vars:
#   LLM_MODEL          HuggingFace model ID  (default: Qwen/Qwen2.5-7B-Instruct)
#   GPU_MEM_UTIL       fraction of GPU for each vLLM (default: 0.20)
#   MAX_MODEL_LEN      max context length (default: 4096)
#   HF_TOKEN           HuggingFace token for gated models (optional)
#   RETRIEVER_URL      NeMo Retriever URL (default: http://localhost:7670)
# -----------------------------------------------------------------------------
set -euo pipefail

LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.20}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
RETRIEVER_URL="${RETRIEVER_URL:-http://localhost:7670}"

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

# ── 8. Start CacheBlend path: vLLM + APC + LMCache ───────────────────────────
log "Starting CacheBlend path: vLLM + APC + LMCache → port 8002"
log "  log: ${CACHEBLEND_LOG}"
log "  lmcache config: ${LMCACHE_CONFIG}"
VLLM_USE_FLASHINFER_SAMPLER=0 \
LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG}" \
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
    # Print progress from logs
    echo -n "  [$(date +%H:%M:%S)] loading"
    grep -c "Loading safetensors checkpoint" "${BASELINE_LOG}" 2>/dev/null | xargs printf " baseline:shards=%s" || true
    echo ""
done

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