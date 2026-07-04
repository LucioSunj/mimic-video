#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${MIMIC_VIDEO_REPO_DIR:-/XYFS02/HDD_POOL/nju_shklu/nju_shklu_1/Video-as-action-generation-source/mimic-video}
MODEL_DIR="${REPO_DIR}/model"
PYTHON=${MIMIC_VIDEO_PYTHON:-/XYFS02/HDD_POOL/nju_shklu/nju_shklu_1/mixture-of-horizon-for-world-action-model/mimic-video/model/.venv/bin/python}
RUN_ROOT=${MIMIC_LIBERO_RUN_ROOT:-"${REPO_DIR}/runs/libero_action_queue"}
LOG_DIR="${RUN_ROOT}/logs"

GPUS=${GPUS:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29653}
MAX_ITER=${MAX_ITER:-50000}
SWITCH_ITER=${SWITCH_ITER:-5000}
SAVE_ITER=${SAVE_ITER:-1000}
LOGGING_ITER=${LOGGING_ITER:-100}
VALIDATION_ITER=${VALIDATION_ITER:-1000000}
POLL_SECONDS=${POLL_SECONDS:-60}

SOURCE_GROUP=${SOURCE_GROUP:-vlsp_source_condition}
SOURCE_NAME=${SOURCE_NAME:-vlsp_source_condition_goal_20260629_183112}
SOURCE_CKPT_DIR="${MODEL_DIR}/checkpoints/vam/${SOURCE_GROUP}/${SOURCE_NAME}/checkpoints"
SOURCE_LATEST="${SOURCE_CKPT_DIR}/latest_checkpoint.txt"

GOAL_DATA_CONFIG=${GOAL_DATA_CONFIG:-libero_goal_full_no_rgb}
GOAL_EXP="w2a_${GOAL_DATA_CONFIG}_v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused_lr1.000e-04_layer20_bsz128"
SHM_VIDEO_LATENT_DIR=${MIMIC_LIBERO_SHM_GOAL_VIDEO_LATENT_DIR:-/dev/shm/mimic_video_latents/libero_goal_full_v2w_libero_goal_tokenizer_latents}

CONTINUATION_SCRIPT="${REPO_DIR}/scripts/continue_goal_after_source_condition.sh"
CONTINUATION_PID_FILE="${RUN_ROOT}/goal_continuation_watcher.pid"

mkdir -p "${LOG_DIR}"

ts() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

log() {
  echo "[$(ts)] $*"
}

latest_iter() {
  if [[ ! -f "${SOURCE_LATEST}" ]]; then
    echo 0
    return
  fi
  local latest
  latest=$(tr -d '[:space:]' < "${SOURCE_LATEST}")
  if [[ "${latest}" =~ iter_0*([0-9]+)\.pt ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo 0
  fi
}

source_running() {
  pgrep -f "job.group=${SOURCE_GROUP}.*job.name=${SOURCE_NAME}" >/dev/null 2>&1 \
    || pgrep -f "job.name=${SOURCE_NAME}.*job.group=${SOURCE_GROUP}" >/dev/null 2>&1
}

require_file() {
  [[ -f "$1" ]] || { log "ERROR missing file: $1" >&2; exit 1; }
}

preflight() {
  [[ -x "${PYTHON}" ]] || { log "ERROR Python is not executable: ${PYTHON}" >&2; exit 1; }
  [[ -x "${CONTINUATION_SCRIPT}" ]] || chmod +x "${CONTINUATION_SCRIPT}"
  require_file "${SHM_VIDEO_LATENT_DIR}/metadata.json"
  require_file "${SHM_VIDEO_LATENT_DIR}/video_latent.fp16.memmap"
}

stop_continuation_watcher() {
  if [[ ! -f "${CONTINUATION_PID_FILE}" ]]; then
    return
  fi
  local pid
  pid=$(tr -d '[:space:]' < "${CONTINUATION_PID_FILE}" || true)
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" >/dev/null 2>&1; then
    return
  fi
  log "Stopping continuation watcher pid=${pid}."
  pkill -TERM -P "${pid}" >/dev/null 2>&1 || true
  kill -TERM "${pid}" >/dev/null 2>&1 || true
  for _ in $(seq 1 30); do
    kill -0 "${pid}" >/dev/null 2>&1 || return
    sleep 1
  done
  log "Continuation watcher pid=${pid} did not exit after TERM; sending KILL."
  pkill -KILL -P "${pid}" >/dev/null 2>&1 || true
  kill -KILL "${pid}" >/dev/null 2>&1 || true
}

stop_source_training() {
  log "Stopping current source+cond training."
  pkill -TERM -f "job.group=${SOURCE_GROUP}.*job.name=${SOURCE_NAME}" >/dev/null 2>&1 || true
  pkill -TERM -f "job.name=${SOURCE_NAME}.*job.group=${SOURCE_GROUP}" >/dev/null 2>&1 || true
  for _ in $(seq 1 60); do
    source_running || return
    sleep 1
  done
  log "source+cond did not exit after TERM; sending KILL."
  pkill -KILL -f "job.group=${SOURCE_GROUP}.*job.name=${SOURCE_NAME}" >/dev/null 2>&1 || true
  pkill -KILL -f "job.name=${SOURCE_NAME}.*job.group=${SOURCE_GROUP}" >/dev/null 2>&1 || true
}

start_source_training_from_shm() {
  local stamp log_path
  stamp=$(date -u +%Y%m%d_%H%M%S)
  log_path="${LOG_DIR}/main_source_condition_goal_resume32_shm_${stamp}.log"
  log "Restarting source+cond from latest checkpoint with /dev/shm cache."
  log "log=${log_path}"
  (
    cd "${MODEL_DIR}"
    export CUDA_VISIBLE_DEVICES="${GPUS}"
    export PYTHONPATH="${MODEL_DIR}:${PYTHONPATH:-}"
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
    export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
    export NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-0}
    export DEBUG_STEP_TIMING=0
    export DEBUG_W2A_TIMING=0

    "${PYTHON}" -m torch.distributed.run \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_port="${MASTER_PORT}" \
      -m scripts.train \
      --config=cosmos_predict2/configs/config.py -- \
      "experiment=${GOAL_EXP}" \
      "job.group=${SOURCE_GROUP}" \
      "job.name=${SOURCE_NAME}" \
      "trainer.max_iter=${MAX_ITER}" \
      "trainer.logging_iter=${LOGGING_ITER}" \
      "trainer.validation_iter=${VALIDATION_ITER}" \
      "trainer.grad_accum_iter=1" \
      "checkpoint.save_iter=${SAVE_ITER}" \
      "dataloader_train.batch_size.global_bsz=32" \
      "dataloader_train.num_workers=1" \
      "dataloader_train.prefetch_factor=1" \
      "dataloader_train.persistent_workers=false" \
      "dataloader_train.pin_memory=false" \
      "dataloader_train.in_order=true" \
      "model.config.offline_video_embedding_dir=" \
      "model.config.offline_video_embedding_required=false" \
      "model.config.offline_video_latent_dir=${SHM_VIDEO_LATENT_DIR}" \
      "model.config.offline_video_latent_required=true" \
      "model.config.pipe_config.action_source_prior.enabled=true" \
      "model.config.pipe_config.action_source_prior.mode=video_prior_sample" \
      "model.config.pipe_config.action_conditioning.mode=normal"
  ) >"${log_path}" 2>&1 &
  echo "$!" > "${RUN_ROOT}/source_condition_shm.pid"
  log "source+cond shm pid=$(cat "${RUN_ROOT}/source_condition_shm.pid")"
}

start_continuation_watcher_from_shm() {
  local stamp log_path
  stamp=$(date -u +%Y%m%d_%H%M%S)
  log_path="${LOG_DIR}/goal_continuation_watcher_shm_${stamp}.log"
  log "Restarting continuation watcher with /dev/shm cache."
  (
    export MIMIC_LIBERO_GOAL_VIDEO_LATENT_DIR="${SHM_VIDEO_LATENT_DIR}"
    export GPUS="${GPUS}"
    export NPROC_PER_NODE="${NPROC_PER_NODE}"
    export MASTER_PORT=29661
    export MAX_ITER="${MAX_ITER}"
    export SAVE_ITER="${SAVE_ITER}"
    export LOGGING_ITER="${LOGGING_ITER}"
    export VALIDATION_ITER="${VALIDATION_ITER}"
    export POLL_SECONDS=300
    exec "${CONTINUATION_SCRIPT}"
  ) >"${log_path}" 2>&1 &
  echo "$!" > "${CONTINUATION_PID_FILE}"
  echo "${log_path}" > "${RUN_ROOT}/goal_continuation_watcher_latest_log.txt"
  log "continuation watcher shm pid=$(cat "${CONTINUATION_PID_FILE}") log=${log_path}"
}

main() {
  preflight
  log "Waiting for ${SOURCE_GROUP}/${SOURCE_NAME} to reach checkpoint iter ${SWITCH_ITER}."
  while true; do
    local iter
    iter=$(latest_iter)
    if (( iter >= MAX_ITER )); then
      log "Already at iter ${iter}; source+cond is complete, no switch needed."
      return 0
    fi
    if (( iter >= SWITCH_ITER )); then
      log "Found checkpoint iter ${iter}; switching source+cond to /dev/shm cache."
      stop_continuation_watcher
      stop_source_training
      start_source_training_from_shm
      start_continuation_watcher_from_shm
      log "Switch complete."
      return 0
    fi
    if ! source_running; then
      log "ERROR source+cond is not running and latest checkpoint is only iter ${iter}." >&2
      return 1
    fi
    log "latest checkpoint iter ${iter}; waiting ${POLL_SECONDS}s."
    sleep "${POLL_SECONDS}"
  done
}

main "$@"
