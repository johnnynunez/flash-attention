#!/usr/bin/env bash
set -euo pipefail

TORCH_VERSION="${TORCH_VERSION:?TORCH_VERSION not set}"
CUDA_VERSION="${CUDA_VERSION:-}"

# Convert CUDA version to the short form expected by PyTorch wheels (e.g. 12.9.1 -> 129).
short_cuda() {
  echo "${1//./}" | head -c 3
}

if [[ -n "${CUDA_VERSION}" ]]; then
  tag="$(short_cuda "${CUDA_VERSION}")"
  echo "Installing torch ${TORCH_VERSION} with CUDA ${tag}"
  pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/cu${tag}"
else
  echo "Installing torch ${TORCH_VERSION} (CPU only)"
  pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
fi

python --version
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA:', torch.version.cuda)"
python -c "from torch.utils import cpp_extension; print (cpp_extension.CUDA_HOME)"