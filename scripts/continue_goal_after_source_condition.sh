#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${MIMIC_VIDEO_REPO_DIR:-/XYFS02/HDD_POOL/nju_shklu/nju_shklu_1/Video-as-action-generation-source/mimic-video}
MODEL_DIR="${REPO_DIR}/model"
PYTHON=${MIMIC_VIDEO_PYTHON:-/XYFS02/HDD_POOL/nju_shklu/nju_shklu_1/mixture-of-horizon-for-world-action-model/mimic-video/model/.venv/bin/python}
RUN_ROOT=${MIMIC_LIBERO_RUN_ROOT:-"${REPO_DIR}/runs/libero_action_queue"}
LOG_DIR="${RUN_ROOT}/logs"
STATE_FILE="${RUN_ROOT}/goal_continuation_state.tsv"

GPUS=${GPUS:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29661}
MAX_ITER=${MAX_ITER:-50000}
SAVE_ITER=${SAVE_ITER:-1000}
LOGGING_ITER=${LOGGING_ITER:-100}
VALIDATION_ITER=${VALIDATION_ITER:-1000000}
POLL_SECONDS=${POLL_SECONDS:-300}

SOURCE_GROUP=${SOURCE_GROUP:-vlsp_source_condition}
SOURCE_NAME=${SOURCE_NAME:-vlsp_source_condition_goal_20260629_183112}
SOURCE_CKPT_DIR="${MODEL_DIR}/checkpoints/vam/${SOURCE_GROUP}/${SOURCE_NAME}/checkpoints"
SOURCE_LATEST="${SOURCE_CKPT_DIR}/latest_checkpoint.txt"

GOAL_DATA_CONFIG=${GOAL_DATA_CONFIG:-libero_goal_full_no_rgb}
GOAL_EXP="w2a_${GOAL_DATA_CONFIG}_v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused_lr1.000e-04_layer20_bsz128"
GOAL_VIDEO_LATENT_NAME="${GOAL_DATA_CONFIG%_no_rgb}_v2w_libero_goal_tokenizer_latents"
SHM_GOAL_VIDEO_LATENT_DIR="/dev/shm/mimic_video_latents/${GOAL_VIDEO_LATENT_NAME}"
DEFAULT_GOAL_VIDEO_LATENT_DIR="${REPO_DIR}/data/libero_video_latents/${GOAL_VIDEO_LATENT_NAME}"
if [[ -n "${MIMIC_LIBERO_GOAL_VIDEO_LATENT_DIR:-}" ]]; then
  GOAL_VIDEO_LATENT_DIR="${MIMIC_LIBERO_GOAL_VIDEO_LATENT_DIR}"
elif [[ -f "${SHM_GOAL_VIDEO_LATENT_DIR}/metadata.json" && -f "${SHM_GOAL_VIDEO_LATENT_DIR}/video_latent.fp16.memmap" ]]; then
  GOAL_VIDEO_LATENT_DIR="${SHM_GOAL_VIDEO_LATENT_DIR}"
else
  GOAL_VIDEO_LATENT_DIR="${DEFAULT_GOAL_VIDEO_LATENT_DIR}"
fi

mkdir -p "${LOG_DIR}"

ts() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

log() {
  echo "[$(ts)] $*"
}

record_state() {
  local status=$1
  local variant=$2
  local job_name=$3
  local log_path=$4
  printf '%s\t%s\t%s\t%s\t%s\n' "$(ts)" "${status}" "${variant}" "${job_name}" "${log_path}" >> "${STATE_FILE}"
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
  require_file "${REPO_DIR}/checkpoints/video_backbone/v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused.pt"
  require_file "${GOAL_VIDEO_LATENT_DIR}/metadata.json"
  require_file "${GOAL_VIDEO_LATENT_DIR}/video_latent.fp16.memmap"
}

wait_for_source_condition() {
  log "Waiting for ${SOURCE_GROUP}/${SOURCE_NAME} to reach iter ${MAX_ITER}."
  while true; do
    local iter
    iter=$(latest_iter)
    if (( iter >= MAX_ITER )); then
      log "source+cond latest checkpoint is iter ${iter}. Waiting for source process to exit before continuing."
      while source_running; do
        sleep 60
      done
      log "source+cond completed. Continuing goal queue."
      return 0
    fi
    if ! source_running; then
      log "ERROR source+cond is not running and latest checkpoint is only iter ${iter}; refusing to launch downstream experiments." >&2
      return 1
    fi
    log "source+cond latest checkpoint iter ${iter}; still running. Sleeping ${POLL_SECONDS}s."
    sleep "${POLL_SECONDS}"
  done
}

run_variant() {
  local variant=$1
  local source_enabled=$2
  local source_mode=$3
  local conditioning_mode=$4
  local master_port=$5

  local stamp
  stamp=$(date -u +%Y%m%d_%H%M%S)
  local job_name="${variant}_goal_${stamp}"
  local log_path="${LOG_DIR}/${job_name}.log"

  record_state "START" "${variant}" "${job_name}" "${log_path}"
  log "START ${job_name}" | tee -a "${log_path}"
  log "log=${log_path}" | tee -a "${log_path}"
  log "experiment=${GOAL_EXP}" | tee -a "${log_path}"
  log "offline_video_latent_dir=${GOAL_VIDEO_LATENT_DIR}" | tee -a "${log_path}"

  set +e
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
      --master_port="${master_port}" \
      -m scripts.train \
      --config=cosmos_predict2/configs/config.py -- \
      "experiment=${GOAL_EXP}" \
      "job.group=${variant}" \
      "job.name=${job_name}" \
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
      "model.config.offline_video_latent_dir=${GOAL_VIDEO_LATENT_DIR}" \
      "model.config.offline_video_latent_required=true" \
      "model.config.pipe_config.action_source_prior.enabled=${source_enabled}" \
      "model.config.pipe_config.action_source_prior.mode=${source_mode}" \
      "model.config.pipe_config.action_conditioning.mode=${conditioning_mode}"
  ) 2>&1 | tee -a "${log_path}"
  local status=${PIPESTATUS[0]}
  set -e

  if [[ "${status}" -ne 0 ]]; then
    record_state "FAIL:${status}" "${variant}" "${job_name}" "${log_path}"
    log "FAIL ${job_name} status=${status}" | tee -a "${log_path}"
    return "${status}"
  fi

  record_state "DONE" "${variant}" "${job_name}" "${log_path}"
  log "DONE ${job_name}" | tee -a "${log_path}"
}

main() {
  preflight
  wait_for_source_condition
  run_variant "vlsp_source_only" true video_prior_sample zero_video "${MASTER_PORT}"
  run_variant "baseline_gaussian" false gaussian normal "$((MASTER_PORT + 1))"
  log "Goal source-only and baseline continuation completed."
}

main "$@"
