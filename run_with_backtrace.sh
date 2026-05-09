#!/usr/bin/env bash
# Profile profile_vllm_omni.py with nsys, full CUDA + Python backtraces, and
# per-rank timings written to a txt file alongside the .nsys-rep.
#
# Usage:
#   ./run_with_backtrace.sh                              # decoder, pp=8, fp32 compile cl3d
#   MODE=encoder NPROC=2 PP_SIZE=2 ./run_with_backtrace.sh
#   DTYPE=bf16 ./run_with_backtrace.sh
#
# Env vars:
#   MODE       encoder|decoder            (default: decoder)
#   NPROC      ranks (== GPUs)            (default: 8)
#   PP_SIZE    --vae-patch-parallel-size  (default: NPROC)
#   DTYPE      fp32|bf16|fp16             (default: fp32)
#   COMPILE    1|0                        (default: 1 — torch.compile + cl3d)
#   REPORTS    output dir                 (default: reports-${DTYPE}-pp${PP_SIZE}-bt)
#   MASTER_PORT                           (default: 29500)
#
# Per-GPU breakdowns from the resulting .sqlite:
#   sqlite3 <out>.sqlite "SELECT deviceId, COUNT(*) AS n, ROUND(SUM(end-start)/1e6,1) AS ms \
#       FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY deviceId ORDER BY deviceId;"
#   sqlite3 <out>.sqlite "SELECT (globalTid/(1<<24)) AS rank, text, ROUND((end-start)/1e6,1) AS ms \
#       FROM NVTX_EVENTS WHERE text LIKE 'decode_iter_%' ORDER BY rank, start;"
#
# In nsys-ui, the Timeline view shows per-GPU rows by default. CUDA + Python
# backtrace flags below populate the Events View's "CUDA backtrace" column so
# you can see which Python line issued each kernel.

set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CPATH="${CUDA_HOME}/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${CUDA_HOME}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${MODE:-decoder}"
NPROC="${NPROC:-8}"
PP_SIZE="${PP_SIZE:-${NPROC}}"
DTYPE="${DTYPE:-fp32}"
COMPILE="${COMPILE:-1}"
REPORTS="${REPORTS:-${HERE}/reports-${DTYPE}-pp${PP_SIZE}-bt}"
MASTER_PORT="${MASTER_PORT:-29500}"

NSYS="${NSYS:-$(command -v nsys || echo /usr/local/cuda/bin/nsys)}"
PY="${PY:-python}"

mkdir -p "${REPORTS}"

case "${MODE}" in
  encoder) SKIP_FLAG="--skip-decode" ;;
  decoder) SKIP_FLAG="--skip-encode" ;;
  both)    SKIP_FLAG="" ;;
  *)       echo "MODE must be encoder|decoder|both" >&2; exit 1 ;;
esac

if [[ "${COMPILE}" == "1" ]]; then
  COMPILE_FLAGS=""
  TAG="compile_cl3d"
else
  COMPILE_FLAGS="--no-compile --no-channels-last"
  TAG="baseline"
fi

OUT_BASE="vllm_omni_pp${PP_SIZE}_${MODE}_${TAG}_bt"
OUT_PATH="${REPORTS}/${OUT_BASE}"

# TIMINGS_DIR tells profile_vllm_omni.py where to drop per-rank txt files.
export TIMINGS_DIR="${REPORTS}"

echo "=== nsys + backtrace: ${OUT_BASE} ==="
echo "    NPROC=${NPROC} PP_SIZE=${PP_SIZE} DTYPE=${DTYPE} COMPILE=${COMPILE}"
echo "    REPORTS=${REPORTS}"
echo

"${NSYS}" profile \
  --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --cudabacktrace=all \
  --python-backtrace=cuda \
  --python-sampling=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="${OUT_PATH}" \
  "${PY}" -m torch.distributed.run \
    --nproc-per-node="${NPROC}" \
    --master-port="${MASTER_PORT}" \
    "${HERE}/profile_vllm_omni.py" \
    --dtype "${DTYPE}" --num-frames 81 --height 720 --width 1280 \
    --warmup 10 --repeat 3 --vae-patch-parallel-size "${PP_SIZE}" \
    --timings-dir "${REPORTS}" \
    ${SKIP_FLAG} ${COMPILE_FLAGS}

echo
echo "wrote ${OUT_PATH}.nsys-rep"
echo "per-rank timings: ${REPORTS}/timings_pp${PP_SIZE}_world${NPROC}_rank*.txt"

# Auto-export sqlite so the per-GPU SQL queries above work without manual export.
echo
echo "=== exporting sqlite ==="
"${NSYS}" export --type=sqlite --force-overwrite=true \
  -o "${OUT_PATH}" "${OUT_PATH}.nsys-rep" >/dev/null
# Newer nsys writes the file without a .sqlite extension; rename if needed.
if [[ -f "${OUT_PATH}" && ! -s "${OUT_PATH}.sqlite" ]]; then
  mv -f "${OUT_PATH}" "${OUT_PATH}.sqlite"
fi
echo "wrote ${OUT_PATH}.sqlite"
