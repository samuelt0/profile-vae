#!/usr/bin/env bash
# Profile the Wan 2.2 VAE across three backends with Nsight Systems + Compute.
#
# Usage:
#   ./run_profiles.sh nsys diffusers     # nsys profile of diffusers+compile
#   ./run_profiles.sh nsys tensorrt      # nsys profile of torch_tensorrt
#   ./run_profiles.sh nsys sglang        # nsys profile of SGLang
#   ./run_profiles.sh nsys all           # all three
#   ./run_profiles.sh ncu diffusers      # kernel-level ncu profile (small res)
#   ./run_profiles.sh ncu sglang
#   ./run_profiles.sh ncu all
#
# Open .nsys-rep in /home/samuel/nsight-systems-2025.6.1/bin/nsys-ui
# Open .ncu-rep in ncu-ui (ships with CUDA: /usr/local/cuda/bin/ncu-ui)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTS="${REPORTS:-${HERE}/reports}"
mkdir -p "${REPORTS}"

NSYS=/home/samuel/nsight-systems-2025.6.1/bin/nsys
NCU=/usr/local/cuda/bin/ncu

# Expect the `nunchaku` conda env to be active, or override PY explicitly:
#   conda activate nunchaku && ./run_profiles.sh nsys diffusers
#   PY=/path/to/python ./run_profiles.sh nsys diffusers
PY="${PY:-python}"

# Full-size profiling settings (Wan 2.2 I2V defaults).
FULL_ARGS="--dtype bf16 --num-frames 81 --height 720 --width 1280 --warmup 10 --repeat 3"

# ncu settings: small tensor so kernel replay does not take hours.
NCU_ARGS="--dtype bf16 --num-frames 5 --height 256 --width 448 --warmup 1 --repeat 1"

# Kernels to skip at launch (warmup + compile) and number to record. These
# values focus the capture on a single measured encode or decode after warmup.
NCU_SKIP=200
NCU_COUNT=40

mode="${1:-}"
target="${2:-}"

if [[ -z "${mode}" || -z "${target}" ]]; then
  sed -n '2,16p' "${BASH_SOURCE[0]}"
  exit 1
fi

run_nsys () {
  local name="$1"; shift
  local out="${REPORTS}/${name}"
  echo "=== nsys: ${name} ==="
  # --capture-range=cudaProfilerApi + cudaProfilerStart/Stop in Python limits
  # the trace to the post-warmup measurement region.
  "${NSYS}" profile \
    --trace=cuda,nvtx,osrt \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --output="${out}" \
    "$@"
  echo "wrote ${out}.nsys-rep"
}

run_ncu () {
  local name="$1"; shift
  local out="${REPORTS}/${name}_ncu"
  echo "=== ncu: ${name} ==="
  # --set full captures every metric; consider --set detailed or --section
  # for faster runs.
  "${NCU}" \
    --set full \
    --target-processes all \
    --kernel-name-base function \
    --launch-skip "${NCU_SKIP}" \
    --launch-count "${NCU_COUNT}" \
    --force-overwrite \
    --export "${out}" \
    "$@"
  echo "wrote ${out}.ncu-rep"
}

nsys_diffusers () {
  run_nsys "diffusers_compile" \
    "${PY}" "${HERE}/profile_diffusers_compile.py" ${FULL_ARGS}
}

nsys_tensorrt () {
  run_nsys "tensorrt" \
    "${PY}" "${HERE}/profile_tensorrt.py" ${FULL_ARGS}
}

nsys_sglang () {
  run_nsys "sglang" \
    "${PY}" "${HERE}/profile_sglang.py" ${FULL_ARGS}
}

ncu_diffusers () {
  run_ncu "diffusers_compile" \
    "${PY}" "${HERE}/profile_diffusers_compile.py" ${NCU_ARGS} --no-compile
}

ncu_tensorrt () {
  run_ncu "tensorrt" \
    "${PY}" "${HERE}/profile_tensorrt.py" ${NCU_ARGS}
}

ncu_sglang () {
  run_ncu "sglang" \
    "${PY}" "${HERE}/profile_sglang.py" ${NCU_ARGS}
}

case "${mode}:${target}" in
  nsys:diffusers) nsys_diffusers ;;
  nsys:tensorrt)  nsys_tensorrt ;;
  nsys:sglang)    nsys_sglang ;;
  nsys:all)       nsys_diffusers; nsys_tensorrt; nsys_sglang ;;
  ncu:diffusers)  ncu_diffusers ;;
  ncu:tensorrt)   ncu_tensorrt ;;
  ncu:sglang)     ncu_sglang ;;
  ncu:all)        ncu_diffusers; ncu_tensorrt; ncu_sglang ;;
  *)
    echo "unknown combination: mode='${mode}' target='${target}'" >&2
    exit 1
    ;;
esac
