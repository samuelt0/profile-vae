#!/usr/bin/env bash
# Profile the Wan 2.2 VAE under vllm-omni VAE Patch Parallel (size=8 by default)
# on this 8x B200 host, in 4 fp32 configurations:
#   1. encoder, no channels_last, no compile
#   2. decoder, no channels_last, no compile
#   3. encoder, channels_last_3d + torch.compile
#   4. decoder, channels_last_3d + torch.compile
#
# Usage:
#   ./run_vllm_omni_profiles.sh            # all 4
#   ./run_vllm_omni_profiles.sh 1          # only config 1
#   ./run_vllm_omni_profiles.sh 1 3        # configs 1 and 3
#
# Overrides:
#   NPROC=4 PP_SIZE=4 ./run_vllm_omni_profiles.sh   # smaller world
#   MASTER_PORT=29501 ./run_vllm_omni_profiles.sh
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
DTYPE="${DTYPE:-fp32}"
NPROC="${NPROC:-8}"
PP_SIZE="${PP_SIZE:-${NPROC}}"
REPORTS="${REPORTS:-${HERE}/reports-${DTYPE}-pp${PP_SIZE}}"
mkdir -p "${REPORTS}"

NSYS="${NSYS:-$(command -v nsys || echo /home/samuel/nsight-systems-2025.6.1/bin/nsys)}"
PY="${PY:-python}"
MASTER_PORT="${MASTER_PORT:-29500}"

COMMON_ARGS="--dtype ${DTYPE} --num-frames 81 --height 720 --width 1280 \
  --warmup 10 --repeat 3 --vae-patch-parallel-size ${PP_SIZE}"

run_nsys () {
  local name="$1"; shift
  local out="${REPORTS}/${name}"
  echo "=== nsys: ${name} ==="
  # nsys traces the torchrun parent + all 8 python child ranks into one
  # .nsys-rep. cudaProfilerStart/Stop in each rank's Python code limits the
  # capture to the post-warmup measurement region.
  "${NSYS}" profile \
    --trace=cuda,nvtx,osrt \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --output="${out}" \
    "${PY}" -m torch.distributed.run \
      --nproc-per-node="${NPROC}" \
      --master-port="${MASTER_PORT}" \
      "${HERE}/profile_vllm_omni.py" "$@"
  echo "wrote ${out}.nsys-rep"
}

cfg1_encoder_baseline () {
  run_nsys "vllm_omni_pp${PP_SIZE}_encoder_baseline" \
    ${COMMON_ARGS} --skip-decode --no-channels-last --no-compile
}

cfg2_decoder_baseline () {
  run_nsys "vllm_omni_pp${PP_SIZE}_decoder_baseline" \
    ${COMMON_ARGS} --skip-encode --no-channels-last --no-compile
}

cfg3_encoder_compile () {
  run_nsys "vllm_omni_pp${PP_SIZE}_encoder_compile_cl3d" \
    ${COMMON_ARGS} --skip-decode
}

cfg4_decoder_compile () {
  run_nsys "vllm_omni_pp${PP_SIZE}_decoder_compile_cl3d" \
    ${COMMON_ARGS} --skip-encode
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
