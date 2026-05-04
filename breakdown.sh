#!/usr/bin/env bash
# Per-rank kernel breakdown of one NVTX iteration in a multi-rank nsys trace.
#
# Usage:
#   ./breakdown.sh <report-name> [<nvtx-range>] [<rank>]
# Examples:
#   ./breakdown.sh vllm_omni_pp8_encoder_compile_cl3d
#   ./breakdown.sh vllm_omni_pp8_encoder_compile_cl3d encode_iter_1 0
#   ./breakdown.sh vllm_omni_pp8_decoder_compile_cl3d decode_iter_2 3
#
# The kernel-time column sums to that rank's GPU-busy time within the iter
# (not the iter's wall time — the difference is GPU-idle gaps).

set -euo pipefail

NAME="${1:?usage: breakdown.sh <report-name> [nvtx-range] [rank]}"
RANGE="${2:-encode_iter_1}"
RANK="${3:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REP="${HERE}/reports-fp32/${NAME}.nsys-rep"
SQ="${HERE}/reports-fp32/${NAME}.sqlite"

[[ -f "${REP}" ]] || { echo "no such report: ${REP}" >&2; exit 1; }
[[ -f "${SQ}"  ]] || nsys export --type sqlite --force-overwrite=true --output "${SQ}" "${REP}" >/dev/null

echo "=== ${NAME} :: ${RANGE} :: rank ${RANK} ==="
sqlite3 -column "${SQ}" "
WITH ranks AS (
  SELECT globalTid AS gtid, start AS s, end AS e,
         ROW_NUMBER() OVER (ORDER BY globalTid) - 1 AS rank
  FROM NVTX_EVENTS WHERE text = '${RANGE}'
),
me AS (SELECT * FROM ranks WHERE rank = ${RANK}),
kk AS (
  SELECT k.shortName AS sid, k.end - k.start AS dur
  FROM CUPTI_ACTIVITY_KIND_KERNEL k, me
  WHERE k.start >= me.s AND k.end <= me.e
    AND (k.globalPid >> 24) = (me.gtid >> 24)
)
SELECT
  printf('wall:     %.2f ms', (SELECT (e - s) / 1e6 FROM me)),
  printf('GPU busy: %.2f ms', (SELECT SUM(dur) / 1e6 FROM kk)),
  printf('idle gap: %.2f ms', (SELECT (e - s) / 1e6 - SUM(k.end - k.start) / 1e6
                                FROM me, CUPTI_ACTIVITY_KIND_KERNEL k
                                WHERE k.start >= me.s AND k.end <= me.e
                                  AND (k.globalPid >> 24) = (me.gtid >> 24)));
"
echo
echo "Top 15 kernels (sums to GPU-busy ms above):"
sqlite3 -header -column "${SQ}" "
WITH ranks AS (
  SELECT globalTid AS gtid, start AS s, end AS e,
         ROW_NUMBER() OVER (ORDER BY globalTid) - 1 AS rank
  FROM NVTX_EVENTS WHERE text = '${RANGE}'
),
me AS (SELECT * FROM ranks WHERE rank = ${RANK}),
kk AS (
  SELECT k.shortName AS sid, k.end - k.start AS dur
  FROM CUPTI_ACTIVITY_KIND_KERNEL k, me
  WHERE k.start >= me.s AND k.end <= me.e
    AND (k.globalPid >> 24) = (me.gtid >> 24)
),
totals AS (SELECT SUM(dur) AS t FROM kk)
SELECT
  ROUND(SUM(dur) / 1e6, 2) AS ms,
  ROUND(100.0 * SUM(dur) / (SELECT t FROM totals), 1) AS pct,
  COUNT(*) AS n,
  substr(s.value, 1, 75) AS kernel
FROM kk JOIN StringIds s ON s.id = kk.sid
GROUP BY kk.sid
ORDER BY SUM(dur) DESC LIMIT 15;
"
