#!/usr/bin/env bash
# Wrapper: invoke breakdown_full.py with the vae conda env's python (it has
# nsys + sqlite3 + statistics on PYTHONPATH).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/data/samuel/miniconda3/envs/vae/bin/python}"
exec "${PY}" "${HERE}/breakdown_full.py" "$@"
