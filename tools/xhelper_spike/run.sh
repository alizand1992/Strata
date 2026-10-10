#!/usr/bin/env bash
# tools/xhelper_spike/run.sh - build the two halves of the Phase 1 spike and run them.
#
# Phase 1 of docs/plans/OPTION_A_EXCLUSIVE_RAM.md: nothing in the engine is touched.  This builds spike_cuda.cu with
# nvcc and spike_hip.cpp with hipcc, runs the CUDA parent (which forks the HIP child with a memfd fd), and prints the
# five tables Gate 1 is read from.  Run it on the box the plan names: RTX 5060 Ti primary, RX 9060 XT (gfx1200) helper,
# Linux.
#
#   ./run.sh                       build both, run with the defaults (10000 pings, 500 work rounds)
#   ./run.sh --rounds 30000        more pings for a better tail
#   ./run.sh --cuda-arch=sm_120    name the NVIDIA arch (nvcc picks one itself if you leave it out)
#   ./run.sh --hip-arch=gfx1201    the other AMD card
#   ./run.sh --parent-only         build and run only the CUDA half, to check this script works before the HIP half builds
#   ./run.sh --host-only           build and run spike_doorbell.cpp with plain g++: the memfd/futex plumbing, no toolkit
#
# Anything this script does not understand goes to the parent program.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVCC="${NVCC:-nvcc}"
HIPCC="${HIPCC:-hipcc}"
HIP_ARCH="${HIP_ARCH:-gfx1200}"
CUDA_ARCH="${CUDA_ARCH:-}"
PARENT_ONLY=0
HOST_ONLY=0
PASS=()

for a in "$@"; do
  case "$a" in
    --hip-arch=*) HIP_ARCH="${a#--hip-arch=}" ;;
    --cuda-arch=*) CUDA_ARCH="${a#--cuda-arch=}" ;;
    --parent-only) PARENT_ONLY=1 ;;
    --host-only) HOST_ONLY=1 ;;
    *) PASS+=("$a") ;;
  esac
done

say() { printf '\033[1m%s\033[0m\n' "$*"; }

if [ "$HOST_ONLY" -eq 1 ]; then
  say "building the host-only doorbell check (no CUDA, no HIP)"
  "${CXX:-g++}" -O2 -std=c++20 -Wall -Wextra "$HERE/spike_doorbell.cpp" -o "$HERE/strata-xh-doorbell"
  exec "$HERE/strata-xh-doorbell" ${PASS[@]+"${PASS[@]}"}
fi

command -v "$NVCC" >/dev/null 2>&1 || { echo "nvcc not found. Phase 1 needs the CUDA toolkit."; exit 1; }

say "building the CUDA parent"
# nvcc with no -arch builds for the oldest arch it still supports, which then JITs.  Ask the card what it is.
if [ -z "$CUDA_ARCH" ] && command -v nvidia-smi >/dev/null 2>&1; then
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)
  [ -n "$cap" ] && CUDA_ARCH="sm_$cap"
  [ -n "$CUDA_ARCH" ] && echo "detected the primary card's arch: $CUDA_ARCH"
fi
NVCC_ARGS=(-O3 -std=c++20 -Xcompiler '-Wall,-Wextra' "$HERE/spike_cuda.cu" -o "$HERE/strata-xh-spike-cuda")
[ -n "$CUDA_ARCH" ] && NVCC_ARGS+=(-arch="$CUDA_ARCH")
"$NVCC" "${NVCC_ARGS[@]}"

if [ "$PARENT_ONLY" -eq 1 ]; then
  say "parent only (--parent-only)"
  exec "$HERE/strata-xh-spike-cuda" "${PASS[@]}"
fi

command -v "$HIPCC" >/dev/null 2>&1 || {
  echo "hipcc not found. Phase 1 needs ROCm on the same box: the child is built for the AMD card and is forked by"
  echo "the CUDA parent, so the two halves cannot be built on different machines."
  exit 1
}

# Same for the AMD arch: read it off the card when rocminfo is there, otherwise believe the default.
if [ "$HIP_ARCH" = gfx1200 ] && command -v rocminfo >/dev/null 2>&1; then
  found=$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a]+' | sort -u | tail -1)
  [ -n "$found" ] && HIP_ARCH="$found"
fi

say "building the HIP child for $HIP_ARCH"
"$HIPCC" -O3 -std=c++20 -x hip --offload-arch="$HIP_ARCH" "$HERE/spike_hip.cpp" -o "$HERE/strata-xh-spike-hip"

say "running"
# The parent prints the gate lines.  Keep its output, and the child's stderr, in a log next to the tables.
set +e
"$HERE/strata-xh-spike-cuda" --child "$HERE/strata-xh-spike-hip" ${PASS[@]+"${PASS[@]}"} | tee "$HERE/results-$(date +%Y%m%d-%H%M%S).txt"
rc=${PIPESTATUS[0]}
set -e
exit "$rc"
