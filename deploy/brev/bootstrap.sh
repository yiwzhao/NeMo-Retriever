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

# Leave empty to install the LATEST GPU Operator — needed so its device-plugin
# supports this k3s's containerd 2.x. (We disable the operator's own toolkit and
# use the host toolkit instead; see section 2.)
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
# A non-ASCII key breaks the HTTP Authorization header the service sends to the
# NIMs, so ingestion fails even though images and weights pull fine. Reject early.
if printf '%s' "${NGC_API_KEY}" | LC_ALL=C grep -q '[^ -~]'; then
  die "NGC_API_KEY contains non-ASCII characters. Use your real nvapi-... key."
fi
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
# 2. GPU runtime on k3s — host container-toolkit + k3s NATIVE detection
# -----------------------------------------------------------------------------
# On k3s (containerd 2.x) the GPU Operator's own container-toolkit misconfigures
# containerd and breaks CNI, so we don't use it. Instead install
# nvidia-container-toolkit on the HOST and let k3s detect the `nvidia` runtime
# natively (this keeps CNI intact), then make it the default runtime via a k3s
# containerd drop-in so the device-plugin and every GPU pod get the GPU.
CONTAINERD_DIR=/var/lib/rancher/k3s/agent/etc/containerd

log "Installing nvidia-container-toolkit on the host (apt)"
if ! command -v nvidia-container-runtime >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
fi
command -v nvidia-container-runtime >/dev/null \
  || die "nvidia-container-runtime not on PATH after apt install."

log "Restarting k3s so it natively detects the nvidia runtime"
sudo systemctl restart k3s
sleep 20

log "Setting nvidia as the DEFAULT containerd runtime (k3s config-v3.toml.d drop-in)"
sudo mkdir -p "${CONTAINERD_DIR}/config-v3.toml.d"
sudo tee "${CONTAINERD_DIR}/config-v3.toml.d/99-nvidia-default.toml" >/dev/null <<'EOF'
version = 3
[plugins.'io.containerd.cri.v1.runtime'.containerd]
  default_runtime_name = "nvidia"
EOF
sudo systemctl restart k3s
sleep 20
kubectl wait --for=condition=Ready node --all --timeout=180s

# -----------------------------------------------------------------------------
# 3. NVIDIA GPU Operator — device-plugin + time-slicing ONLY (toolkit DISABLED)
# -----------------------------------------------------------------------------
log "Installing NVIDIA GPU Operator (device-plugin + time-slicing x${GPU_TIMESLICING_REPLICAS}; toolkit disabled)"
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia >/dev/null 2>&1 || true
helm repo update >/dev/null

# The device-plugin reads this ConfigMap to advertise each physical GPU as
# ${GPU_TIMESLICING_REPLICAS} schedulable nvidia.com/gpu units.
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

# driver.enabled=false: reuse the host driver. toolkit.enabled=false: the
# runtime is installed on the host (above); the operator only runs the
# device-plugin + validators here.
GPU_VERSION_FLAG=()
[[ -n "${GPU_OPERATOR_VERSION}" ]] && GPU_VERSION_FLAG=(--version "${GPU_OPERATOR_VERSION}")
helm upgrade --install gpu-operator nvidia/gpu-operator \
  -n gpu-operator "${GPU_VERSION_FLAG[@]}" \
  --set driver.enabled=false \
  --set toolkit.enabled=false \
  --set devicePlugin.config.name=time-slicing-config \
  --set devicePlugin.config.default=any \
  --wait --timeout 15m || \
  echo "GPU Operator --wait timed out (device-plugin waits for the toolkit-ready marker below); continuing."

# With toolkit.enabled=false, the device-plugin's `toolkit-validation` init
# container still blocks on /run/nvidia/validations/toolkit-ready — normally
# written by the operator toolkit. Our host toolkit + nvidia default runtime IS
# the ready stack, so create the marker to unblock the device-plugin.
log "Creating toolkit-ready marker so the device-plugin init can proceed"
sudo mkdir -p /run/nvidia/validations
sudo touch /run/nvidia/validations/toolkit-ready

log "Waiting for time-sliced nvidia.com/gpu units to be advertised on the node"
for i in $(seq 1 60); do
  if kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}' | grep -qE '[1-9]'; then
    kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}'; echo
    break
  fi
  sleep 10
  [[ $i -eq 60 ]] && die "Node never advertised nvidia.com/gpu — check the device-plugin pod in ns gpu-operator (is /run/nvidia/validations/toolkit-ready present, default runtime nvidia?)."
done

# -----------------------------------------------------------------------------
# 4. NVIDIA NIM Operator — reconciles NIMCache / NIMService CRDs
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
# 5. NeMo Retriever — core stack via the repo Helm chart
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
# 6. Status + how to reach the service
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
