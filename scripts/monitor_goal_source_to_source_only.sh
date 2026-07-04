#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${MIMIC_VIDEO_REPO_DIR:-/XYFS02/HDD_POOL/nju_shklu/nju_shklu_1/Video-as-action-generation-source/mimic-video}
MODEL_DIR="${REPO_DIR}/model"
PYTHON=${MIMIC_VIDEO_PYTHON:-/XYFS02/HDD_POOL/nju_shklu/nju_shklu_1/mixture-of-horizon-for-world-action-model/mimic-video/model/.venv/bin/python}
RUN_ROOT=${MIMIC_LIBERO_RUN_ROOT:-"${REPO_DIR}/runs/libero_action_queue"}
LOG_DIR="${RUN_ROOT}/logs"
MONITOR_LOG=${MONITOR_LOG:-"${LOG_DIR}/goal_source_to_source_only_monitor_$(date -u +%Y%m%d_%H%M%S).log"}

SOURCE_GROUP=${SOURCE_GROUP:-vlsp_source_condition}
SOURCE_NAME=${SOURCE_NAME:-vlsp_source_condition_goal_20260629_183112}
SOURCE_LOG=${SOURCE_LOG:-"${LOG_DIR}/main_source_condition_goal_resume32_shm_setsid_20260630_054249.log"}
SOURCE_CKPT_DIR="${MODEL_DIR}/checkpoints/vam/${SOURCE_GROUP}/${SOURCE_NAME}/checkpoints"
SOURCE_LATEST="${SOURCE_CKPT_DIR}/latest_checkpoint.txt"

GPUS=${GPUS:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29661}
MAX_ITER=${MAX_ITER:-50000}
SAVE_ITER=${SAVE_ITER:-1000}
LOGGING_ITER=${LOGGING_ITER:-100}
VALIDATION_ITER=${VALIDATION_ITER:-1000000}
NORMAL_SLEEP=${NORMAL_SLEEP:-3600}
FINAL_SLEEP=${FINAL_SLEEP:-600}
FINAL_WINDOW=${FINAL_WINDOW:-500}
WATCHER_GRACE_SECONDS=${WATCHER_GRACE_SECONDS:-1800}

GOAL_DATA_CONFIG=${GOAL_DATA_CONFIG:-libero_goal_full_no_rgb}
GOAL_EXP="w2a_${GOAL_DATA_CONFIG}_v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused_lr1.000e-04_layer20_bsz128"
GOAL_VIDEO_LATENT_NAME="${GOAL_DATA_CONFIG%_no_rgb}_v2w_libero_goal_tokenizer_latents"
SHM_GOAL_VIDEO_LATENT_DIR="/dev/shm/mimic_video_latents/${GOAL_VIDEO_LATENT_NAME}"
DEFAULT_GOAL_VIDEO_LATENT_DIR="${REPO_DIR}/data/libero_video_latents/${GOAL_VIDEO_LATENT_NAME}"
GOAL_VIDEO_LATENT_DIR=${MIMIC_LIBERO_GOAL_VIDEO_LATENT_DIR:-}
if [[ -z "${GOAL_VIDEO_LATENT_DIR}" ]]; then
  if [[ -f "${SHM_GOAL_VIDEO_LATENT_DIR}/metadata.json" && -f "${SHM_GOAL_VIDEO_LATENT_DIR}/video_latent.fp16.memmap" ]]; then
    GOAL_VIDEO_LATENT_DIR="${SHM_GOAL_VIDEO_LATENT_DIR}"
  else
    GOAL_VIDEO_LATENT_DIR="${DEFAULT_GOAL_VIDEO_LATENT_DIR}"
  fi
fi

mkdir -p "${LOG_DIR}"

ts() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

log() {
  echo "[$(ts)] $*" | tee -a "${MONITOR_LOG}"
}

latest_checkpoint_iter() {
  local latest
  [[ -f "${SOURCE_LATEST}" ]] || { echo 0; return; }
  latest=$(tr -d '[:space:]' < "${SOURCE_LATEST}")
  if [[ "${latest}" =~ iter_0*([0-9]+)\.pt ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo 0
  fi
}

latest_progress_step() {
  [[ -f "$1" ]] || { echo 0; return; }
  perl -0777 -ne 'while(/Training:.*?\|\s*(\d+)\/50000/g){$s=$1} END{print $s || 0}' "$1" 2>/dev/null || echo 0
}

latest_loss_line() {
  [[ -f "$1" ]] || { echo "loss_step=0 loss=NA speed=NA"; return; }
  perl -0777 -ne 'while(/(\d+)\s+:\s+iter_speed\s+([0-9.]+).*?Loss:\s*([0-9.]+)/g){$s=$1;$v=$2;$l=$3} END{if($s){print "loss_step=$s loss=$l speed=${v}s"}else{print "loss_step=0 loss=NA speed=NA"}}' "$1" 2>/dev/null \
    || echo "loss_step=0 loss=NA speed=NA"
}

source_running() {
  pgrep -f "job.group=${SOURCE_GROUP}.*job.name=${SOURCE_NAME}" >/dev/null 2>&1 \
    || pgrep -f "job.name=${SOURCE_NAME}.*job.group=${SOURCE_GROUP}" >/dev/null 2>&1
}

continuation_watcher_running() {
  pgrep -f "scripts/continue_goal_after_source_condition.sh|continue_goal_after_source_condition.sh" >/dev/null 2>&1
}

source_only_running() {
  pgrep -f "job.group=vlsp_source_only.*goal" >/dev/null 2>&1
}

latest_source_only_log() {
  local latest
  latest=$(find "${LOG_DIR}" -maxdepth 1 -type f -name 'vlsp_source_only_goal_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1{print $2}')
  echo "${latest:-}"
}

source_only_entered_training() {
  local log_path=$1
  [[ -n "${log_path}" && -f "${log_path}" ]] || return 1
  grep -qE 'Training:|Iteration [0-9]+: Hit counter|[0-9]+ : iter_speed' "${log_path}"
}

launch_source_only_fallback() {
  local lock_dir="${RUN_ROOT}/source_only_launch.lock"
  if ! mkdir "${lock_dir}" 2>/dev/null; then
    log "source-only launch lock exists; not launching a duplicate."
    return 0
  fi

  local stamp job_name log_path
  stamp=$(date -u +%Y%m%d_%H%M%S)
  job_name="vlsp_source_only_goal_${stamp}"
  log_path="${LOG_DIR}/${job_name}.log"

  log "Fallback launching source-only: ${job_name}"
  log "Fallback source-only log: ${log_path}"
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

    exec "${PYTHON}" -m torch.distributed.run \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_port="${MASTER_PORT}" \
      -m scripts.train \
      --config=cosmos_predict2/configs/config.py -- \
      "experiment=${GOAL_EXP}" \
      "job.group=vlsp_source_only" \
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
      "model.config.pipe_config.action_source_prior.enabled=true" \
      "model.config.pipe_config.action_source_prior.mode=video_prior_sample" \
      "model.config.pipe_config.action_conditioning.mode=zero_video"
  ) >> "${log_path}" 2>&1 &
  log "Fallback source-only launcher pid=$!"
}

monitor_source_condition() {
  local source_done_at=0
  while true; do
    local ckpt_iter progress loss sleep_for remaining
    ckpt_iter=$(latest_checkpoint_iter)
    progress=$(latest_progress_step "${SOURCE_LOG}")
    loss=$(latest_loss_line "${SOURCE_LOG}")
    remaining=$(( MAX_ITER - progress ))
    (( remaining < 0 )) && remaining=0

    log "source+cond progress=${progress}/${MAX_ITER} ckpt=${ckpt_iter} ${loss}"

    if (( ckpt_iter >= MAX_ITER )) && ! source_running; then
      log "source+cond completed and process exited."
      return 0
    fi

    if (( ckpt_iter >= MAX_ITER )) && source_running; then
      log "source+cond reached final checkpoint; waiting for process exit."
      sleep "${FINAL_SLEEP}"
      continue
    fi

    if ! source_running; then
      log "WARNING source+cond process is not running before final checkpoint; latest ckpt=${ckpt_iter}."
      if (( source_done_at == 0 )); then
        source_done_at=$(date +%s)
      fi
    fi

    sleep_for="${NORMAL_SLEEP}"
    if (( remaining <= FINAL_WINDOW )); then
      sleep_for="${FINAL_SLEEP}"
    fi
    log "sleeping ${sleep_for}s before next source+cond check."
    sleep "${sleep_for}"
  done
}

ensure_source_only_started() {
  local done_at now so_log
  done_at=$(date +%s)
  while true; do
    so_log=$(latest_source_only_log)
    if source_only_running || source_only_entered_training "${so_log}"; then
      log "source-only detected. log=${so_log}"
      return 0
    fi

    if continuation_watcher_running; then
      now=$(date +%s)
      if (( now - done_at < WATCHER_GRACE_SECONDS )); then
        log "waiting for existing continuation watcher to start source-only. elapsed=$((now - done_at))s"
        sleep "${FINAL_SLEEP}"
        continue
      fi
      log "existing watcher has not started source-only after grace period; checking fallback."
    fi

    if ! source_only_running; then
      launch_source_only_fallback
    fi

    sleep "${FINAL_SLEEP}"
  done
}

monitor_source_only() {
  local so_log progress loss sleep_for remaining
  while true; do
    so_log=$(latest_source_only_log)
    progress=$(latest_progress_step "${so_log}")
    loss=$(latest_loss_line "${so_log}")
    remaining=$(( MAX_ITER - progress ))
    (( remaining < 0 )) && remaining=0
    log "source-only progress=${progress}/${MAX_ITER} ${loss} log=${so_log:-NA}"

    if source_only_entered_training "${so_log}"; then
      log "source-only training loop confirmed."
    else
      log "source-only not yet in training loop; sleeping ${FINAL_SLEEP}s."
      sleep "${FINAL_SLEEP}"
      continue
    fi

    if (( progress >= MAX_ITER )) && ! source_only_running; then
      log "source-only appears complete. monitor exiting."
      return 0
    fi

    sleep_for="${NORMAL_SLEEP}"
    if (( remaining <= FINAL_WINDOW )); then
      sleep_for="${FINAL_SLEEP}"
    fi
    log "sleeping ${sleep_for}s before next source-only check."
    sleep "${sleep_for}"
  done
}

main() {
  log "monitor started pid=$$"
  log "source=${SOURCE_GROUP}/${SOURCE_NAME}"
  log "source_log=${SOURCE_LOG}"
  log "goal_latent_dir=${GOAL_VIDEO_LATENT_DIR}"
  monitor_source_condition
  ensure_source_only_started
  monitor_source_only
}

main "$@"
