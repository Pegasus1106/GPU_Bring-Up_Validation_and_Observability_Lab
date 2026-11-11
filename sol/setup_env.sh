#!/bin/bash
# One-time environment setup on ASU Sol (no Docker, no root).
#   ./sol/setup_env.sh            # run from the repo root; do the pip step in an interactive session if
#                                 # login-node limits bite:  interactive -c 4 -t 0-01:00
#
# Uses the module-provided Mamba (RC asks users not to install their own package manager).
set -eo pipefail
cd "$(dirname "$0")/.."
ENV_NAME="${LAB_ENV:-gpulab}"
# CUDA wheel flavour: any runtime the node driver supports works (Sol's driver supports CUDA 13.x at time of
# writing, so cu128 is fine). Change if `nvidia-smi` on your target node reports an older maximum.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

# requirements-app.txt pins numpy 2.5.x, which needs Python >= 3.12
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

module load mamba/latest
if ! mamba env list | grep -q "^${ENV_NAME} "; then
  mamba create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi
source activate "${ENV_NAME}"

if ! python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "ERROR: environment '${ENV_NAME}' uses Python $(python -V 2>&1) but the pinned dependencies need >= 3.12." >&2
  echo "Remove it and re-run:   mamba env remove -n ${ENV_NAME} -y && ./sol/setup_env.sh" >&2
  exit 1
fi

pip install --upgrade pip
pip install torch torchvision --index-url "${TORCH_INDEX}"
pip install -r requirements-dev.txt        # pinned service/test dependencies

# verified (sha256) monitoring binaries for this architecture -> .tools/<os>-<arch>/
python -m sol.lab setup ${WITH_GRAFANA:+--grafana}

python - <<'EOF'
import torch
print("torch", torch.__version__, "built for CUDA", torch.version.cuda, "| cuda available here:", torch.cuda.is_available(), "(False is expected on a login node)")
EOF
echo
echo "environment '${ENV_NAME}' ready. Next: ./sol/submit.sh bringup   (see docs/sol.md)"
