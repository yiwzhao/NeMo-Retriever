#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# NeMo Retriever — Brev single-node bootstrap (Core RAG only)
# -----------------------------------------------------------------------------
# Brings up a single-node Kubernetes cluster on a Brev GPU instance and installs
# the NeMo Retriever "core" stack: retriever-service + 4 core NIMs + LanceDB.
#
# Assumes a Brev instance that already has:
#   * NVIDIA GPU driver installed (nvidia-smi works)
#   * Ubuntu/Debian with sudo, curl
#   * Outbound network to nvcr.io / build.nvidia.com / helm repos
#
# Requires:
#   export NGC_API_KEY=<your NGC key>   # from https://org.ngc.nvidia.com/setup/api-keys
#
# Usage:
#   export NGC_API_KEY=nvapi-xxxxx
#   ./deploy/brev/bootstrap.sh
#
# This script is idempotent-ish (safe to re-run) but is a TEMPLATE: on some
# Brev images you may need to adjust the k8s distro or GPU runtime step.
# -----------------------------------------------------------------------------
set -euo pipefail

NS="${NS:-retriever}"
RELEASE="${RELEASE:-retriever}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHART_DIR="${REPO_ROOT}/nemo_retriever/helm"
VALUES="${REPO_ROOT}/deploy/brev/values-brev-core.yaml"

# Leave empty to install the LATEST GPU Operator. Required so its bundled
# container-toolkit understands k3s's containerd 2.x (config version 3).
# Old pins (e.g. v24.9.1) crash the toolkit with "unsupported config version: 3".
GPU_OPERATOR_VERSION="${GPU_OPERATOR_VERSION:-}"
NIM_OPERATOR_VERSION="${NIM_OPERATOR_VERSION:-}"   # empty = latest
# GPU time-slicing: how many schedulable nvidia.com/gpu units each PHYSICAL GPU
# advertises. Lets all 4 core NIMs (≈4.8 GiB combined) share ONE large card
# (e.g. a single 96 GiB RTX PRO 6000) instead of claiming a card each. Set to 1
# to disable time-slicing (then you need one physical GPU per core NIM).
GPU_TIMESLICING_REPLICAS="${GPU_TIMESLICING_REPLICAS:-8}"

log() { echo -e "\n\033[1;32m==> $*\033[0m"; }
die() { echo -e "\n\033[1;31mERROR: $*\033[0m" >&2; exit 1; }

[[ -n "${NGC_API_KEY:-}" ]] || die "NGC_API_KEY is not set. export NGC_API_KEY=nvapi-..."
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — this node has no usable GPU driver."

log "GPUs visible on this node:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# -----------------------------------------------------------------------------
# 1. Single-node Kubernetes (k3s) + Helm
# -----------------------------------------------------------------------------
if ! command -v kubectl >/dev/null || ! kubectl get nodes >/dev/null 2>&1; then
  log "Installing k3s (single-node Kubernetes)"
  curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644
  export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
  echo "export KUBECONFIG=/etc/rancher/k3s/k3s.yaml" >> "${HOME}/.bashrc"
else
  log "Existing Kubernetes detected — skipping k3s install"
fi
export KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"

if ! command -v helm >/dev/null; then
  log "Installing Helm"
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
fi

log "Waiting for the node to be Ready"
kubectl wait --for=condition=Ready node --all --timeout=180s

# -----------------------------------------------------------------------------
# 2. NVIDIA GPU Operator — exposes nvidia.com/gpu to the scheduler
#    (driver.enabled=false: reuse the host driver already on the Brev image)
# -----------------------------------------------------------------------------
log "Installing NVIDIA GPU Operator (${GPU_OPERATOR_VERSION}) with GPU time-slicing x${GPU_TIMESLICING_REPLICAS}"
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia >/dev/null 2>&1 || true
helm repo update >/dev/null

# The device-plugin reads this ConfigMap to advertise each physical GPU as
# ${GPU_TIMESLICING_REPLICAS} schedulable nvidia.com/gpu units. Must exist
# before/at operator install so the plugin picks it up on first start.
kubectl create namespace gpu-operator --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator
data:
  any: |-
    version: v1
    flags:
      migStrategy: none
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: ${GPU_TIMESLICING_REPLICAS}
EOF

# k3s ships its own containerd at non-standard paths, so the GPU Operator's
# container-toolkit must be pointed at them and told to register the `nvidia`
# runtime as the DEFAULT — otherwise NIM pods start without driver injection and
# crashloop with "NVIDIA Driver was not detected / libnvidia-ml.so.1 not found".
# The toolkit writes the runtime into k3s's containerd template; k3s merges it
# into the live config on (re)start. This k3s uses `config.toml`, so the
# template is `config.toml.tmpl`.
GPU_VERSION_FLAG=()
[[ -n "${GPU_OPERATOR_VERSION}" ]] && GPU_VERSION_FLAG=(--version "${GPU_OPERATOR_VERSION}")
helm upgrade --install gpu-operator nvidia/gpu-operator \
  -n gpu-operator "${GPU_VERSION_FLAG[@]}" \
  --set driver.enabled=false \
  --set toolkit.enabled=true \
  --set devicePlugin.config.name=time-slicing-config \
  --set devicePlugin.config.default=any \
  --set toolkit.env[0].name=CONTAINERD_CONFIG \
  --set-string toolkit.env[0].value=/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl \
  --set toolkit.env[1].name=CONTAINERD_SOCKET \
  --set-string toolkit.env[1].value=/run/k3s/containerd/containerd.sock \
  --set toolkit.env[2].name=CONTAINERD_RUNTIME_CLASS \
  --set-string toolkit.env[2].value=nvidia \
  --set toolkit.env[3].name=CONTAINERD_SET_AS_DEFAULT \
  --set-string toolkit.env[3].value=true \
  --wait --timeout 15m

log "Waiting for time-sliced nvidia.com/gpu units to be advertised on the node"
for i in $(seq 1 60); do
  if kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}' | grep -qE '[1-9]'; then
    kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}'; echo
    break
  fi
  sleep 10
  [[ $i -eq 60 ]] && die "Node never advertised nvidia.com/gpu — check GPU Operator pods in ns gpu-operator."
done

# -----------------------------------------------------------------------------
# 3. NVIDIA NIM Operator — reconciles NIMCache / NIMService CRDs
# -----------------------------------------------------------------------------
log "Installing NVIDIA NIM Operator"
kubectl create namespace nim-operator --dry-run=client -o yaml | kubectl apply -f -
VERSION_FLAG=()
[[ -n "${NIM_OPERATOR_VERSION}" ]] && VERSION_FLAG=(--version "${NIM_OPERATOR_VERSION}")
helm upgrade --install nim-operator nvidia/k8s-nim-operator \
  -n nim-operator "${VERSION_FLAG[@]}" \
  --wait --timeout 10m

log "Waiting for the apps.nvidia.com CRDs to register"
for i in $(seq 1 30); do
  if kubectl get crd nimservices.apps.nvidia.com >/dev/null 2>&1 \
     && kubectl get crd nimcaches.apps.nvidia.com >/dev/null 2>&1; then
    break
  fi
  sleep 5
  [[ $i -eq 30 ]] && die "NIM Operator CRDs (apps.nvidia.com/v1alpha1) never registered."
done

# -----------------------------------------------------------------------------
# 4. NeMo Retriever — core stack via the repo Helm chart
# -----------------------------------------------------------------------------
kubectl create namespace "${NS}" --dry-run=client -o yaml | kubectl apply -f -

log "Installing NeMo Retriever (core RAG only) into ns/${NS}"
helm upgrade --install "${RELEASE}" "${CHART_DIR}" \
  -n "${NS}" \
  -f "${VALUES}" \
  --set ngcImagePullSecret.create=true \
  --set ngcImagePullSecret.password="${NGC_API_KEY}" \
  --set ngcApiSecret.create=true \
  --set ngcApiSecret.password="${NGC_API_KEY}" \
  --wait --timeout 30m || {
    echo "helm --wait timed out; NIM model downloads can take a while. Continuing to status." ; }

# -----------------------------------------------------------------------------
# 5. Status + how to reach the service
# -----------------------------------------------------------------------------
log "NIM reconciliation status (weights download to PVCs on first run):"
kubectl get nimcache,nimservice -n "${NS}" || true
log "Pods:"
kubectl get pods -n "${NS}"

cat <<EOF

------------------------------------------------------------------------------
Done. First-run model downloads may still be in progress — watch with:

  kubectl get pods -n ${NS} -w

Port-forward the retriever service (port 7670) to use the quickstart notebook:

  kubectl port-forward -n ${NS} svc/${RELEASE}-nemo-retriever 7670:7670

Then set in the notebook / your shell:

  export RETRIEVER_URL=http://localhost:7670

Health check:

  curl -s http://localhost:7670/v1/health | jq .
------------------------------------------------------------------------------
EOF
