#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# NeMo Retriever — Brev launchable setup script
# -----------------------------------------------------------------------------
# Paste this into the Brev launchable "setup script" box (VM Mode). It runs once
# when the VM is ready and needs NO secrets: it clones this repo and starts the
# one-click Deploy UI. The user then opens the UI (port 8000 via Secure Link),
# pastes their NGC key, and clicks "Deploy Launchable" — no terminal needed.
#
# Expose (Brev "TCP/UDP Ports" or auto Jupyter):
#   - 8000  Deploy UI            (the page users interact with)
#   - 8888  Jupyter (optional)   (for the full notebook walkthrough)
# -----------------------------------------------------------------------------
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/yiwzhao/NeMo-Retriever.git}"
REPO_BRANCH="${REPO_BRANCH:-brev-launchable}"
REPO_DIR="${HOME}/NeMo-Retriever"
UI_PORT="${UI_PORT:-8000}"

echo "==> Cloning ${REPO_URL} (${REPO_BRANCH})"
if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone -b "${REPO_BRANCH}" "${REPO_URL}" "${REPO_DIR}"
else
  git -C "${REPO_DIR}" fetch origin "${REPO_BRANCH}" && git -C "${REPO_DIR}" checkout "${REPO_BRANCH}" && git -C "${REPO_DIR}" pull
fi

# Use a dedicated venv so we don't need write access to the system Python
# (the Brev image's /opt/python-venv is not user-writable, and --user is not
# allowed inside a venv). --system-site-packages lets us reuse anything already
# installed (e.g. requests, jupyter).
VENV="${HOME}/deploy-ui-venv"
echo "==> Installing Deploy UI dependencies into ${VENV}"
python3 -m venv --system-site-packages "${VENV}"
"${VENV}/bin/pip" install --quiet --upgrade pip
"${VENV}/bin/pip" install --quiet fastapi "uvicorn[standard]" requests huggingface_hub psutil

# Jupyter for the notebook walkthrough — reuse the system one if present, else
# install into the venv.
JUPYTER="$(command -v jupyter || true)"
if [ -z "${JUPYTER}" ]; then
  "${VENV}/bin/pip" install --quiet jupyterlab || true
  JUPYTER="${VENV}/bin/jupyter"
fi

WEBUI_DIR="${REPO_DIR}/deploy/brev/webui"

echo "==> Starting the Deploy UI on :${UI_PORT}"
if ! curl -s "http://localhost:${UI_PORT}/api/config" >/dev/null 2>&1; then
  cd "${WEBUI_DIR}"
  nohup "${VENV}/bin/python" -m uvicorn app:app --host 0.0.0.0 --port "${UI_PORT}" \
    > "${HOME}/deploy-ui.log" 2>&1 &
  echo "    Deploy UI log: ${HOME}/deploy-ui.log"
fi

# Optional: start Jupyter Lab rooted at the repo so the notebook opens directly.
if [ -x "${JUPYTER}" ] && ! curl -s http://localhost:8888 >/dev/null 2>&1; then
  echo "==> Starting Jupyter Lab on :8888 (rooted at the repo)"
  cd "${REPO_DIR}"
  nohup "${JUPYTER}" lab --ip=0.0.0.0 --port=8888 --no-browser \
    --ServerApp.token='' --ServerApp.root_dir="${REPO_DIR}" \
    > "${HOME}/jupyter.log" 2>&1 &
fi

cat <<EOF

------------------------------------------------------------------------------
Setup done.

  • Open the Deploy UI (Brev Secure Link for port ${UI_PORT}) and click
    "Deploy Launchable" after pasting your NGC key.
  • The full notebook is at deploy/brev/notebooks/nemo_retriever_quickstart.ipynb
    (Jupyter on port 8888 if enabled).
------------------------------------------------------------------------------
EOF
