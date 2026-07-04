#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${MIMIC_VIDEO_REPO_DIR:-/XYFS02/HDD_POOL/nju_shklu/nju_shklu_1/Video-as-action-generation-source/mimic-video}
MODEL_DIR="${REPO_DIR}/model"
PYTHON=${MIMIC_VIDEO_PYTHON:-/XYFS02/HDD_POOL/nju_shklu/nju_shklu_1/mixture-of-horizon-for-world-action-model/mimic-video/model/.venv/bin/python}
DATA_ROOT=${MIMIC_LIBERO_DATA_ROOT:-"${REPO_DIR}/data"}
RUN_ROOT=${MIMIC_LIBERO_RUN_ROOT:-"${REPO_DIR}/runs/libero_action_queue"}
LOG_DIR="${RUN_ROOT}/logs"
STATE_FILE="${RUN_ROOT}/queue_state.tsv"

GPUS=${GPUS:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29631}
MAX_ITER=${MAX_ITER:-50000}
SAVE_ITER=${SAVE_ITER:-1000}
LOGGING_ITER=${LOGGING_ITER:-100}
VALIDATION_ITER=${VALIDATION_ITER:-1000}
RESUME=${RESUME:-0}
SUITE_FILTER=${SUITE_FILTER:-}
DATALOADER_TRAIN_GLOBAL_BSZ=${DATALOADER_TRAIN_GLOBAL_BSZ:-32}
DATALOADER_TRAIN_NUM_WORKERS=${DATALOADER_TRAIN_NUM_WORKERS:-1}
DATALOADER_TRAIN_PREFETCH_FACTOR=${DATALOADER_TRAIN_PREFETCH_FACTOR:-1}
DATALOADER_TRAIN_PERSISTENT_WORKERS=${DATALOADER_TRAIN_PERSISTENT_WORKERS:-false}
DATALOADER_TRAIN_PIN_MEMORY=${DATALOADER_TRAIN_PIN_MEMORY:-false}
DATALOADER_TRAIN_IN_ORDER=${DATALOADER_TRAIN_IN_ORDER:-true}
USE_OFFLINE_VIDEO_EMBEDDINGS=${USE_OFFLINE_VIDEO_EMBEDDINGS:-false}
USE_OFFLINE_VIDEO_LATENTS=${USE_OFFLINE_VIDEO_LATENTS:-true}
VIDEO_EMBEDDING_ROOT=${MIMIC_LIBERO_VIDEO_EMBEDDING_ROOT:-"${DATA_ROOT}/libero_video_embeddings"}
VIDEO_LATENT_ROOT=${MIMIC_LIBERO_VIDEO_LATENT_ROOT:-"${DATA_ROOT}/libero_video_latents"}

GOAL_DATA_CONFIG=${GOAL_DATA_CONFIG:-libero_goal_full_no_rgb}
SPATIAL_DATA_CONFIG=${SPATIAL_DATA_CONFIG:-libero_spatial_full_no_rgb}
OBJECT_DATA_CONFIG=${OBJECT_DATA_CONFIG:-libero_object_full_no_rgb}

GOAL_DATA_DIR=${MIMIC_LIBERO_GOAL_DIR:-"${DATA_ROOT}/${GOAL_DATA_CONFIG%_no_rgb}"}
SPATIAL_DATA_DIR=${MIMIC_LIBERO_SPATIAL_DIR:-"${DATA_ROOT}/${SPATIAL_DATA_CONFIG%_no_rgb}"}
OBJECT_DATA_DIR=${MIMIC_LIBERO_OBJECT_DIR:-"${DATA_ROOT}/${OBJECT_DATA_CONFIG%_no_rgb}"}

GOAL_VIDEO_EMBEDDING_DIR=${MIMIC_LIBERO_GOAL_VIDEO_EMBEDDING_DIR:-"${VIDEO_EMBEDDING_ROOT}/${GOAL_DATA_CONFIG%_no_rgb}_v2w_libero_goal_layer20"}
SPATIAL_VIDEO_EMBEDDING_DIR=${MIMIC_LIBERO_SPATIAL_VIDEO_EMBEDDING_DIR:-"${VIDEO_EMBEDDING_ROOT}/${SPATIAL_DATA_CONFIG%_no_rgb}_v2w_libero_spatial_layer20"}
OBJECT_VIDEO_EMBEDDING_DIR=${MIMIC_LIBERO_OBJECT_VIDEO_EMBEDDING_DIR:-"${VIDEO_EMBEDDING_ROOT}/${OBJECT_DATA_CONFIG%_no_rgb}_v2w_libero_object_layer20"}

GOAL_VIDEO_LATENT_DIR=${MIMIC_LIBERO_GOAL_VIDEO_LATENT_DIR:-"${VIDEO_LATENT_ROOT}/${GOAL_DATA_CONFIG%_no_rgb}_v2w_libero_goal_tokenizer_latents"}
SPATIAL_VIDEO_LATENT_DIR=${MIMIC_LIBERO_SPATIAL_VIDEO_LATENT_DIR:-"${VIDEO_LATENT_ROOT}/${SPATIAL_DATA_CONFIG%_no_rgb}_v2w_libero_spatial_tokenizer_latents"}
OBJECT_VIDEO_LATENT_DIR=${MIMIC_LIBERO_OBJECT_VIDEO_LATENT_DIR:-"${VIDEO_LATENT_ROOT}/${OBJECT_DATA_CONFIG%_no_rgb}_v2w_libero_object_tokenizer_latents"}

GOAL_EXP="w2a_${GOAL_DATA_CONFIG}_v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused_lr1.000e-04_layer20_bsz128"
SPATIAL_EXP="w2a_${SPATIAL_DATA_CONFIG}_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128"
OBJECT_EXP="w2a_${OBJECT_DATA_CONFIG}_v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused_lr1.000e-04_layer20_bsz128"

mkdir -p "${LOG_DIR}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

first_zarr() {
  find -L "$1" -type d -name '*.zarr' -print -quit 2>/dev/null || true
}

require_dataset() {
  local label=$1
  local path=$2
  [[ -d "${path}" ]] || die "${label} data_dir does not exist: ${path}"

  local sample
  sample=$(first_zarr "${path}")
  [[ -n "${sample}" ]] || die "${label} data_dir has no .zarr episodes: ${path}"

  [[ -e "${sample}/language_embedding/.zarray" ]] || die "${label} sample zarr is missing precomputed language_embedding: ${sample}"
}

require_file() {
  [[ -f "$1" ]] || die "required file is missing: $1"
}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_video_embedding_cache() {
  local label=$1
  local path=$2
  [[ -d "${path}" ]] || die "${label} offline video embedding dir does not exist: ${path}"
  require_file "${path}/metadata.json"
  require_file "${path}/crossattn_emb.fp16.memmap"
  require_file "${path}/video_sigma.npy"
}

require_video_latent_cache() {
  local label=$1
  local path=$2
  [[ -d "${path}" ]] || die "${label} offline video latent dir does not exist: ${path}"
  require_file "${path}/metadata.json"
  require_file "${path}/video_latent.fp16.memmap"
}

preflight() {
  [[ -x "${PYTHON}" ]] || die "Python is not executable: ${PYTHON}"
  require_file "${REPO_DIR}/checkpoints/video_backbone/v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused.pt"
  require_file "${REPO_DIR}/checkpoints/video_backbone/v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused.pt"
  require_file "${REPO_DIR}/checkpoints/video_backbone/v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused.pt"
  require_file "${REPO_DIR}/checkpoints/video_backbone/tokenizer/tokenizer.pth"
  require_file "${REPO_DIR}/checkpoints/text_encoder/t5-11b/pytorch_model.bin"
  require_dataset "goal (${GOAL_DATA_CONFIG})" "${GOAL_DATA_DIR}"
  require_dataset "spatial (${SPATIAL_DATA_CONFIG})" "${SPATIAL_DATA_DIR}"
  require_dataset "object (${OBJECT_DATA_CONFIG})" "${OBJECT_DATA_DIR}"
}

record_state() {
  local status=$1
  local variant=$2
  local suite=$3
  local job_name=$4
  local log_path=$5
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" "${variant}" "${suite}" "${job_name}" "${log_path}" >> "${STATE_FILE}"
}

run_one() {
  local variant=$1
  local suite=$2
  local experiment=$3
  local data_dir=$4
  local video_embedding_dir=$5
  local video_latent_dir=$6
  local source_enabled=$7
  local source_mode=$8
  local conditioning_mode=$9

  local offline_video_embedding_dir=""
  local offline_video_embedding_required=false
  local offline_video_latent_dir=""
  local offline_video_latent_required=false
  if is_true "${USE_OFFLINE_VIDEO_EMBEDDINGS}" && is_true "${USE_OFFLINE_VIDEO_LATENTS}"; then
    die "USE_OFFLINE_VIDEO_EMBEDDINGS and USE_OFFLINE_VIDEO_LATENTS are mutually exclusive"
  fi
  if is_true "${USE_OFFLINE_VIDEO_EMBEDDINGS}"; then
    offline_video_embedding_dir="${video_embedding_dir}"
    offline_video_embedding_required=true
    require_video_embedding_cache "${suite}" "${offline_video_embedding_dir}"
  fi
  if is_true "${USE_OFFLINE_VIDEO_LATENTS}"; then
    offline_video_latent_dir="${video_latent_dir}"
    offline_video_latent_required=true
    require_video_latent_cache "${suite}" "${offline_video_latent_dir}"
  fi

  local stamp
  stamp=$(date -u +%Y%m%d_%H%M%S)
  local job_name="${variant}_${suite}_${stamp}"
  local log_path="${LOG_DIR}/${job_name}.log"

  record_state "START" "${variant}" "${suite}" "${job_name}" "${log_path}"
  echo "=== START ${job_name} ===" | tee -a "${log_path}"
  echo "experiment=${experiment}" | tee -a "${log_path}"
  echo "data_dir=${data_dir}" | tee -a "${log_path}"
  echo "offline_video_embedding_dir=${offline_video_embedding_dir}" | tee -a "${log_path}"
  echo "offline_video_embedding_required=${offline_video_embedding_required}" | tee -a "${log_path}"
  echo "offline_video_latent_dir=${offline_video_latent_dir}" | tee -a "${log_path}"
  echo "offline_video_latent_required=${offline_video_latent_required}" | tee -a "${log_path}"
  echo "dataloader_train.batch_size.global_bsz=${DATALOADER_TRAIN_GLOBAL_BSZ}" | tee -a "${log_path}"
  echo "dataloader_train.num_workers=${DATALOADER_TRAIN_NUM_WORKERS}" | tee -a "${log_path}"
  echo "dataloader_train.prefetch_factor=${DATALOADER_TRAIN_PREFETCH_FACTOR}" | tee -a "${log_path}"
  echo "dataloader_train.persistent_workers=${DATALOADER_TRAIN_PERSISTENT_WORKERS}" | tee -a "${log_path}"
  echo "dataloader_train.pin_memory=${DATALOADER_TRAIN_PIN_MEMORY}" | tee -a "${log_path}"
  echo "dataloader_train.in_order=${DATALOADER_TRAIN_IN_ORDER}" | tee -a "${log_path}"

  set +e
  (
    cd "${MODEL_DIR}"
    export CUDA_VISIBLE_DEVICES="${GPUS}"
    export PYTHONPATH="${MODEL_DIR}:${PYTHONPATH:-}"
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}
    export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
    export NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-0}
    export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}

    "${PYTHON}" -m torch.distributed.run \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_port="${MASTER_PORT}" \
      -m scripts.train \
      --config=cosmos_predict2/configs/config.py -- \
      "experiment=${experiment}" \
      "job.group=${variant}" \
      "job.name=${job_name}" \
      "trainer.max_iter=${MAX_ITER}" \
      "trainer.logging_iter=${LOGGING_ITER}" \
      "trainer.validation_iter=${VALIDATION_ITER}" \
      "checkpoint.save_iter=${SAVE_ITER}" \
      "dataloader_train.batch_size.global_bsz=${DATALOADER_TRAIN_GLOBAL_BSZ}" \
      "dataloader_train.num_workers=${DATALOADER_TRAIN_NUM_WORKERS}" \
      "dataloader_train.prefetch_factor=${DATALOADER_TRAIN_PREFETCH_FACTOR}" \
      "dataloader_train.persistent_workers=${DATALOADER_TRAIN_PERSISTENT_WORKERS}" \
      "dataloader_train.pin_memory=${DATALOADER_TRAIN_PIN_MEMORY}" \
      "dataloader_train.in_order=${DATALOADER_TRAIN_IN_ORDER}" \
      "model.config.offline_video_embedding_dir=${offline_video_embedding_dir}" \
      "model.config.offline_video_embedding_required=${offline_video_embedding_required}" \
      "model.config.offline_video_latent_dir=${offline_video_latent_dir}" \
      "model.config.offline_video_latent_required=${offline_video_latent_required}" \
      "model.config.pipe_config.action_source_prior.enabled=${source_enabled}" \
      "model.config.pipe_config.action_source_prior.mode=${source_mode}" \
      "model.config.pipe_config.action_conditioning.mode=${conditioning_mode}"
  ) 2>&1 | tee -a "${log_path}"
  local train_status=${PIPESTATUS[0]}
  set -e

  if [[ "${train_status}" -ne 0 ]]; then
    record_state "FAIL:${train_status}" "${variant}" "${suite}" "${job_name}" "${log_path}"
    echo "=== FAIL ${job_name} status=${train_status} ===" | tee -a "${log_path}"
    return "${train_status}"
  fi

  record_state "DONE" "${variant}" "${suite}" "${job_name}" "${log_path}"
  echo "=== DONE ${job_name} ===" | tee -a "${log_path}"
}

already_done() {
  local variant=$1
  local suite=$2
  [[ "${RESUME}" == "1" ]] || return 1
  [[ -f "${STATE_FILE}" ]] || return 1
  awk -F '\t' -v v="${variant}" -v s="${suite}" '$2 == "DONE" && $3 == v && $4 == s {found=1} END {exit !found}' "${STATE_FILE}"
}

run_queue() {
  preflight
  : > "${STATE_FILE}"

  local variants=(
    "vlsp_source_condition video_prior_sample normal true"
    "vlsp_source_only video_prior_sample zero_video true"
    "baseline_gaussian gaussian normal false"
  )
  local suites=(
    "goal ${GOAL_EXP} ${GOAL_DATA_DIR} ${GOAL_VIDEO_EMBEDDING_DIR} ${GOAL_VIDEO_LATENT_DIR}"
    "spatial ${SPATIAL_EXP} ${SPATIAL_DATA_DIR} ${SPATIAL_VIDEO_EMBEDDING_DIR} ${SPATIAL_VIDEO_LATENT_DIR}"
    "object ${OBJECT_EXP} ${OBJECT_DATA_DIR} ${OBJECT_VIDEO_EMBEDDING_DIR} ${OBJECT_VIDEO_LATENT_DIR}"
  )

  for suite_spec in "${suites[@]}"; do
    read -r suite experiment data_dir video_embedding_dir video_latent_dir <<< "${suite_spec}"
    if [[ -n "${SUITE_FILTER}" && "${suite}" != "${SUITE_FILTER}" ]]; then
      continue
    fi
    for variant_spec in "${variants[@]}"; do
      read -r variant source_mode conditioning_mode source_enabled <<< "${variant_spec}"
      if already_done "${variant}" "${suite}"; then
        echo "skip completed ${variant}/${suite}"
        continue
      fi
      run_one "${variant}" "${suite}" "${experiment}" "${data_dir}" "${video_embedding_dir}" "${video_latent_dir}" "${source_enabled}" "${source_mode}" "${conditioning_mode}"
    done
  done
}

case "${1:-run}" in
  preflight)
    preflight
    echo "preflight ok"
    ;;
  run)
    run_queue
    ;;
  *)
    echo "usage: $0 [preflight|run]" >&2
    exit 2
    ;;
esac
