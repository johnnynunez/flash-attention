#!/usr/bin/env bash
set -euo pipefail

echo "=== Windows (Git Bash) CUDA/MSVC/CMake/Git build prep ==="

ps() {
  powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$@"
}

# 1) OS / arch info (diagnostic)
cmd.exe /c ver || true
uname -a || true
echo "PROCESSOR_ARCHITECTURE=${PROCESSOR_ARCHITECTURE:-unknown}"

# 2) Visual Studio Build Tools (C++ workload)
echo "--- Installing Visual Studio 2022 Build Tools (C++ workload) ---"
ps '$ov = "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --norestart"; \
    Start-Process winget -ArgumentList @("install","--exact","--id","Microsoft.VisualStudio.2022.BuildTools","--accept-package-agreements","--accept-source-agreements","--disable-interactivity","--override",$ov) -Wait -NoNewWindow'

# 3) CMake + Git
echo "--- Installing CMake & Git ---"
ps 'Start-Process winget -ArgumentList @("install","--exact","--id","Kitware.CMake","--accept-package-agreements","--accept-source-agreements","--disable-interactivity") -Wait -NoNewWindow'
ps 'Start-Process winget -ArgumentList @("install","--exact","--id","Git.Git","--accept-package-agreements","--accept-source-agreements","--disable-interactivity") -Wait -NoNewWindow'

# 4) CUDA Toolkit
CUDA_VERSION="${CUDA_VERSION:-13.0.0}"   # picked up from env or defaults
SHORT_VER="${CUDA_VERSION%.*}"           # 13.0.0 -> 13.0 ; if already 12.9, stays 12.9
echo "Using CUDA $CUDA_VERSION"

if [[ "$SHORT_VER" == "$CUDA_VERSION" ]]; then SHORT_VER="$CUDA_VERSION"; fi

echo "--- Installing NVIDIA CUDA Toolkit $CUDA_VERSION ---"
ps "try {
       Start-Process winget -ArgumentList @('install','--exact','--id','NVIDIA.CUDA','--accept-package-agreements','--accept-source-agreements','--disable-interactivity','-v','$CUDA_VERSION') -Wait -NoNewWindow
     } catch {
       Write-Warning 'Exact CUDA version not found; installing latest NVIDIA.CUDA…'
       Start-Process winget -ArgumentList @('install','--exact','--id','NVIDIA.CUDA','--accept-package-agreements','--accept-source-agreements','--disable-interactivity') -Wait -NoNewWindow
     }"

CUDA_PATH_MIXED="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${SHORT_VER}"
export CUDA_HOME="$CUDA_PATH_MIXED"
export CUDA_PATH="$CUDA_PATH_MIXED"
export PATH="$CUDA_PATH_MIXED/bin:$PATH"

echo "CUDA_PATH=$CUDA_PATH"
if [[ -x "$CUDA_PATH_MIXED/bin/nvcc.exe" ]]; then
  "$CUDA_PATH_MIXED/bin/nvcc.exe" --version || true
else
  echo "WARN: nvcc.exe not found at $CUDA_PATH_MIXED/bin (it may appear after install completes)."
fi

# 5) Quick sanity hints (non-fatal)
command -v cl.exe   >/dev/null 2>&1 && echo "cl.exe available."   || echo "NOTE: cl.exe not on PATH yet (MSVC will still be auto-located by tools)."
command -v cmake    >/dev/null 2>&1 && cmake --version  || echo "NOTE: cmake not visible yet."
command -v git      >/dev/null 2>&1 && git --version    || echo "NOTE: git not visible yet."

echo "=== Windows build prep complete ==="