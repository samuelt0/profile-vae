#!/usr/bin/env bash
# Profile the Wan 2.2 VAE in 4 fp32 configurations with Nsight Systems:
#   1. encoder, no channels_last, no compile  (baseline)
#   2. decoder, no channels_last, no compile  (baseline)
#   3. encoder, channels_last_3d + torch.compile
#   4. decoder, channels_last_3d + torch.compile
#
# Usage:
#   ./run_fp32_profiles.sh            # run all 4
#   ./run_fp32_profiles.sh 1          # run only config 1
#   ./run_fp32_profiles.sh 1 3        # run configs 1 and 3
#
# Open .nsys-rep in /home/samuel/nsight-systems-2025.6.1/bin/nsys-ui

set -euo pipefail

# Triton JIT-compiles a small CUDA helper on first torch.compile use; it needs
# cuda.h on the gcc include path. The conda env ships nvcc but no headers, so
# point at the system CUDA toolkit (matches torch.version.cuda = 13.0).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CPATH="${CUDA_HOME}/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${CUDA_HOME}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTS="${REPORTS:-${HERE}/reports-fp32}"
mkdir -p "${REPORTS}"

NSYS="${NSYS:-$(command -v nsys || echo /home/samuel/nsight-systems-2025.6.1/bin/nsys)}"
PY="${PY:-python}"

# Same shape as the bf16 full-size run; warmup is bumped because compile
# configs need more iterations for inductor to settle.
COMMON_ARGS="--dtype fp32 --num-frames 81 --height 720 --width 1280 --warmup 10 --repeat 3"

run_nsys () {
  local name="$1"; shift
  local out="${REPORTS}/${name}"
  echo "=== nsys: ${name} ==="
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

cfg1_encoder_baseline () {
  run_nsys "fp32_encoder_baseline" \
    "${PY}" "${HERE}/profile_diffusers_compile.py" ${COMMON_ARGS} \
    --skip-decode --no-channels-last --no-compile
}

cfg2_decoder_baseline () {
  run_nsys "fp32_decoder_baseline" \
    "${PY}" "${HERE}/profile_diffusers_compile.py" ${COMMON_ARGS} \
    --skip-encode --no-channels-last --no-compile
}

cfg3_encoder_compile () {
  run_nsys "fp32_encoder_compile_cl3d" \
    "${PY}" "${HERE}/profile_diffusers_compile.py" ${COMMON_ARGS} \
    --skip-decode
}

cfg4_decoder_compile () {
  run_nsys "fp32_decoder_compile_cl3d" \
    "${PY}" "${HERE}/profile_diffusers_compile.py" ${COMMON_ARGS} \
    --skip-encode
}

run_one () {
  case "$1" in
    1) cfg1_encoder_baseline ;;
    2) cfg2_decoder_baseline ;;
    3) cfg3_encoder_compile ;;
    4) cfg4_decoder_compile ;;
    *) echo "unknown config: $1 (expected 1-4)" >&2; exit 1 ;;
  esac
}

if [[ $# -eq 0 ]]; then
  run_one 1
  run_one 2
  run_one 3
  run_one 4
else
  for cfg in "$@"; do
    run_one "${cfg}"
  done
fi
