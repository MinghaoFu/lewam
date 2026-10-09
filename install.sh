#!/usr/bin/env bash
# One-step install of the LeWAM environment: every task in configs/eval/ (and swm's TwoRoom) runs in it.
# Usage: bash install.sh [env_name] [cuda_tag]        e.g. bash install.sh lewam cu128
# Set CONDA=mamba or CONDA=micromamba to use another conda front end.
set -euo pipefail
ENV_NAME=${1:-lewam}; CUDA=${2:-cu128}; CONDA=${CONDA:-conda}
ROOT=$(cd "$(dirname "$0")" && pwd)

"$CONDA" create -y -n "$ENV_NAME" -c conda-forge python=3.11 "cmake<4" make
PY=$("$CONDA" run -n "$ENV_NAME" python -c "import sys; print(sys.executable)")
export PATH="$(dirname "$PY"):$PATH"
PIP="$PY -m pip"; export PYTHONNOUSERSITE=1      # keep ~/.local packages out of the env

$PIP install torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA"
$PIP install -e "$ROOT"
# robomimic and DexMimicGen pin old numpy/h5py/protobuf/...; installed without their dependencies (see README)
$PIP install --no-deps "robomimic @ git+https://github.com/ARISE-Initiative/robomimic@9ce0651"
[ -d "$ROOT/third_party/dexmimicgen" ] || git clone https://github.com/NVlabs/dexmimicgen "$ROOT/third_party/dexmimicgen"
$PIP install --no-deps -e "$ROOT/third_party/dexmimicgen"

echo "installed into conda env '$ENV_NAME'; check every env with:"
echo "  conda activate $ENV_NAME && MUJOCO_GL=egl python scripts/check_envs.py"
