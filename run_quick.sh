#!/usr/bin/env bash
# fp32, 2 GPU, encoder, torch.compile + channels_last_3d.
# nsys: cuda graph tracing OFF, cuda + python backtraces ON.
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CPATH="${CUDA_HOME}/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${CUDA_HOME}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NSYS="${NSYS:-$(command -v nsys || echo /home/samuel/nsight-systems-2025.6.1/bin/nsys)}"
PY="${PY:-python}"

"${NSYS}" profile \
  --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=none \
  --cudabacktrace=all \
  --python-backtrace=cuda \
  --python-sampling=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="${HERE}/vllm_omni_pp2_encoder_compile_cl3d_bt" \
  "${PY}" -m torch.distributed.run \
    --nproc-per-node=2 \
    --master-port=29500 \
    "${HERE}/profile_vllm_omni.py" \
    --dtype fp32 --num-frames 81 --height 720 --width 1280 \
    --warmup 10 --repeat 3 --vae-patch-parallel-size 2 \
    --skip-decode
