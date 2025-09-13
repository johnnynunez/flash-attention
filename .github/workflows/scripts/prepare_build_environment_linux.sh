#!/usr/bin/env bash
set -euo pipefail

echo "=== Linux build environment prep ==="
cat /etc/os-release || true
ARCH="$(uname -m)"
echo "ARCH=${ARCH}"

# CUDA version comes from the workflow via CIBW_ENVIRONMENT (defaults to 13.0.0)
RAW_CUDA_VERSION="${CUDA_VERSION:-13.0.0}"
# Convert 13.0.0 -> 13-0 (dnf package suffix) and 13.0 (useful for paths if needed)
CUDA_VERSION_DNF="$(echo "${RAW_CUDA_VERSION}" | awk -F. '{print $1"-"$2}')"
CUDA_VERSION_SHORT="$(echo "${RAW_CUDA_VERSION}" | awk -F. '{print $1"."$2}')"
echo "CUDA_VERSION(raw)=${RAW_CUDA_VERSION}  dnf=${CUDA_VERSION_DNF}  short=${CUDA_VERSION_SHORT}"

# Repos & update
dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
dnf -y update

if [[ "${ARCH}" == "x86_64" ]]; then
  dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
elif [[ "${ARCH}" == "aarch64" ]]; then
  dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo
else
  echo "Unsupported architecture: ${ARCH}" >&2
  exit 1
fi

dnf clean expire-cache

# Retry logic to make network hiccups less fatal
retries=5
n=0
until dnf install -y "cuda-toolkit-${CUDA_VERSION_DNF}"; do
  n=$((n+1))
  if [[ ${n} -ge ${retries} ]]; then
    echo "Failed after ${retries} attempts installing cuda-toolkit-${CUDA_VERSION_DNF}" >&2
    exit 1
  fi
  echo "Retrying CUDA install (${n}/${retries})..."
  sleep 10
done

# Sanity checks
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || true
else
  echo "nvcc not on PATH; temporarily adding /usr/local/cuda/bin"
  export PATH="/usr/local/cuda/bin:${PATH}"
  nvcc --version || echo "nvcc still not available (continuing)"
fi

echo "=== Linux build environment ready ==="
