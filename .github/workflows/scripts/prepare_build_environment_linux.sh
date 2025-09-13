#!/bin/bash

# Print release information
cat /etc/os-release

# Detect system architecture
ARCH=$(uname -m)
CUDA_VERSION="${CUDA_VERSION:-13.0.0}"
echo "Using CUDA $CUDA_VERSION"

# Install Vulkan and other dependencies
dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
dnf update -y

if [[ "$ARCH" == "x86_64" ]]; then
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
    dnf clean expire-cache
    dnf install -y cuda-toolkit-${CUDA_VERSION:-13-0} # Default to CUDA 13.0 for x86_64
elif [[ "$ARCH" == "aarch64" ]]; then
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo
    dnf clean expire-cache
    dnf install -y cuda-toolkit-${CUDA_VERSION:-13-0} # Default to CUDA 13.0 for aarch64
fi