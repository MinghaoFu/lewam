#!/usr/bin/env bash
# Start the remote inference server (src/lewam_real_robot/scripts/serve.py) on a GPU box.
#
# One command from a fresh checkout: creates .venv if it's missing, installs the
# `serve` extra if the imports aren't there yet, then runs the server. Nothing
# about a policy or a robot is configured here -- the server learns which run to
# serve from the rollout client's handshake, so this stays up across checkpoints
# and across embodiments.
#
#   ./src/lewam_real_robot/scripts/serve.sh                      # 0.0.0.0:8080
#   ./src/lewam_real_robot/scripts/serve.sh --port 9000 --device cuda:1
#
# Any other flag is passed straight through to serve.py (--help lists them).
#
# --reinstall  Reinstall the `serve` extra before starting, for when deps have
#              drifted (a changed pyproject, a half-finished install).
set -euo pipefail

REINSTALL=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --reinstall) REINSTALL=1 ;;
        *) ARGS+=("$arg") ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# uv is what pyproject pins the lerobot source through; pip is the fallback
if command -v uv > /dev/null 2>&1; then
    VENV_CMD=(uv venv)
    INSTALL_CMD=(uv pip install -e ".[serve]")
else
    VENV_CMD=(python3 -m venv .venv)
    INSTALL_CMD=(pip install -e ".[serve]")
fi

if [[ ! -d .venv ]]; then
    echo "No .venv found, creating one..."
    "${VENV_CMD[@]}"
    REINSTALL=1
fi
source .venv/bin/activate

# checking imports rather than a marker file means a half-finished install is caught
if [[ "$REINSTALL" -eq 1 ]] || ! python -c "import torch, wandb, cv2" > /dev/null 2>&1; then
    echo "Installing the 'serve' extra..."
    "${INSTALL_CMD[@]}"
fi

# the checkpoint download is a wandb API call, so an unauthenticated box would
# otherwise fail at the first client handshake rather than at startup
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2> /dev/null; then
    echo
    echo "WARNING: wandb does not look authenticated on this machine, so downloading a"
    echo "         run's checkpoint will fail. Run 'wandb login', or set WANDB_API_KEY."
fi

echo
exec python src/lewam_real_robot/scripts/serve.py "${ARGS[@]}"
