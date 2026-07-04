#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${MIMIC_VIDEO_REPO_DIR:-/XYFS02/HDD_POOL/nju_shklu/nju_shklu_1/Video-as-action-generation-source/mimic-video}
RUN_ROOT=${MIMIC_LIBERO_RUN_ROOT:-"${REPO_DIR}/runs/libero_action_queue"}
STATE_FILE="${RUN_ROOT}/queue_state.tsv"
LOG_DIR="${RUN_ROOT}/logs"

echo "== processes =="
ps -ef | grep -E 'scripts.train|torch.distributed.run|train_libero_action_queue' | grep -v grep || true

echo
echo "== gpu =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

echo
echo "== queue state =="
if [[ -f "${STATE_FILE}" ]]; then
  tail -n 20 "${STATE_FILE}"
else
  echo "no state file: ${STATE_FILE}"
fi

echo
echo "== latest log =="
latest_log=$(find "${LOG_DIR}" -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
if [[ -n "${latest_log}" ]]; then
  echo "${latest_log}"
  tail -n 80 "${latest_log}"
else
  echo "no logs under ${LOG_DIR}"
fi
