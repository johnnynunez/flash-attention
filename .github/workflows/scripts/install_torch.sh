#!/usr/bin/env bash
set -euo pipefail

TORCH_VERSION="${TORCH_VERSION:?TORCH_VERSION not set}"
TORCH_CUDA_VERSION="${TORCH_CUDA_VERSION:-}"
CUDA_VERSION="${CUDA_VERSION:-}"

# Prefer explicitly provided TORCH_CUDA_VERSION, otherwise derive from CUDA_VERSION.
if [[ -z "${TORCH_CUDA_VERSION}" && -n "${CUDA_VERSION}" ]]; then
  TORCH_CUDA_VERSION="${CUDA_VERSION}"
fi

# Convert CUDA version to the short form expected by PyTorch wheels (e.g. 12.9 -> 129).
if [[ -n "${TORCH_CUDA_VERSION}" ]]; then
  SHORT_CUDA_VERSION="$(echo "${TORCH_CUDA_VERSION}" | tr -d '.' | head -c 3)"
  echo "Installing torch ${TORCH_VERSION} with CUDA ${SHORT_CUDA_VERSION}"
  pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/cu${SHORT_CUDA_VERSION}"
else
  echo "Installing torch ${TORCH_VERSION} (CPU only)"
  pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
fi