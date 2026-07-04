#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${MIMIC_VIDEO_REPO_DIR:-/XYFS02/HDD_POOL/nju_shklu/nju_shklu_1/Video-as-action-generation-source/mimic-video}
MODEL_DIR="${REPO_DIR}/model"
PYTHON=${MIMIC_VIDEO_PYTHON:-/XYFS02/HDD_POOL/nju_shklu/nju_shklu_1/mixture-of-horizon-for-world-action-model/mimic-video/model/.venv/bin/python}
DATA_ROOT=${MIMIC_LIBERO_DATA_ROOT:-"${REPO_DIR}/data"}
RUN_ROOT=${MIMIC_LIBERO_RUN_ROOT:-"${REPO_DIR}/runs/libero_action_queue"}
LOG_DIR="${RUN_ROOT}/logs"
LAUNCH_DIR="${RUN_ROOT}/launchers"

SUITE=${SUITE:-goal}
VARIANT=${VARIANT:-baseline_gaussian}
JOB_PREFIX=${JOB_PREFIX:-"${VARIANT}_${SUITE}_mig4_nccl"}
MAX_ITER=${MAX_ITER:-50000}
SAVE_ITER=${SAVE_ITER:-1000}
LOGGING_ITER=${LOGGING_ITER:-100}
VALIDATION_ITER=${VALIDATION_ITER:-1000000}
RUN_VALIDATION=${RUN_VALIDATION:-false}
MASTER_PORT=${MASTER_PORT:-29675}
DATALOADER_TRAIN_GLOBAL_BSZ=${DATALOADER_TRAIN_GLOBAL_BSZ:-16}
DATALOADER_TRAIN_NUM_WORKERS=${DATALOADER_TRAIN_NUM_WORKERS:-1}
DATALOADER_TRAIN_PREFETCH_FACTOR=${DATALOADER_TRAIN_PREFETCH_FACTOR:-1}
DATALOADER_TRAIN_PERSISTENT_WORKERS=${DATALOADER_TRAIN_PERSISTENT_WORKERS:-false}
DATALOADER_TRAIN_PIN_MEMORY=${DATALOADER_TRAIN_PIN_MEMORY:-false}
DATALOADER_TRAIN_IN_ORDER=${DATALOADER_TRAIN_IN_ORDER:-true}
USE_OFFLINE_VIDEO_EMBEDDINGS=${USE_OFFLINE_VIDEO_EMBEDDINGS:-false}
USE_OFFLINE_VIDEO_LATENTS=${USE_OFFLINE_VIDEO_LATENTS:-true}
VIDEO_EMBEDDING_ROOT=${MIMIC_LIBERO_VIDEO_EMBEDDING_ROOT:-"${DATA_ROOT}/libero_video_embeddings"}
VIDEO_LATENT_ROOT=${MIMIC_LIBERO_VIDEO_LATENT_ROOT:-"${DATA_ROOT}/libero_video_latents"}

MIGS=(
  "${MIG0:-MIG-1a53acd0-03f0-5f5e-963c-dc4fa053bb92}"
  "${MIG1:-MIG-bc0e4429-8bc2-505b-ae08-0db43b4beae2}"
  "${MIG2:-MIG-00393e6d-8dc3-5e12-be9e-4081cdad9af4}"
  "${MIG3:-MIG-260dab4f-0956-5c1d-8549-7e88490ef3a3}"
)

GOAL_DATA_CONFIG=${GOAL_DATA_CONFIG:-libero_goal_full_no_rgb}
SPATIAL_DATA_CONFIG=${SPATIAL_DATA_CONFIG:-libero_spatial_full_no_rgb}
OBJECT_DATA_CONFIG=${OBJECT_DATA_CONFIG:-libero_object_full_no_rgb}

GOAL_EXP="w2a_${GOAL_DATA_CONFIG}_v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused_lr1.000e-04_layer20_bsz128"
SPATIAL_EXP="w2a_${SPATIAL_DATA_CONFIG}_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128"
OBJECT_EXP="w2a_${OBJECT_DATA_CONFIG}_v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused_lr1.000e-04_layer20_bsz128"

GOAL_VIDEO_EMBEDDING_DIR=${MIMIC_LIBERO_GOAL_VIDEO_EMBEDDING_DIR:-"${VIDEO_EMBEDDING_ROOT}/${GOAL_DATA_CONFIG%_no_rgb}_v2w_libero_goal_layer20"}
SPATIAL_VIDEO_EMBEDDING_DIR=${MIMIC_LIBERO_SPATIAL_VIDEO_EMBEDDING_DIR:-"${VIDEO_EMBEDDING_ROOT}/${SPATIAL_DATA_CONFIG%_no_rgb}_v2w_libero_spatial_layer20"}
OBJECT_VIDEO_EMBEDDING_DIR=${MIMIC_LIBERO_OBJECT_VIDEO_EMBEDDING_DIR:-"${VIDEO_EMBEDDING_ROOT}/${OBJECT_DATA_CONFIG%_no_rgb}_v2w_libero_object_layer20"}

GOAL_VIDEO_LATENT_DIR=${MIMIC_LIBERO_GOAL_VIDEO_LATENT_DIR:-"${VIDEO_LATENT_ROOT}/${GOAL_DATA_CONFIG%_no_rgb}_v2w_libero_goal_tokenizer_latents"}
SPATIAL_VIDEO_LATENT_DIR=${MIMIC_LIBERO_SPATIAL_VIDEO_LATENT_DIR:-"${VIDEO_LATENT_ROOT}/${SPATIAL_DATA_CONFIG%_no_rgb}_v2w_libero_spatial_tokenizer_latents"}
OBJECT_VIDEO_LATENT_DIR=${MIMIC_LIBERO_OBJECT_VIDEO_LATENT_DIR:-"${VIDEO_LATENT_ROOT}/${OBJECT_DATA_CONFIG%_no_rgb}_v2w_libero_object_tokenizer_latents"}

mkdir -p "${LOG_DIR}" "${LAUNCH_DIR}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_file() {
  [[ -f "$1" ]] || die "required file is missing: $1"
}

require_cache_dir() {
  local label=$1
  local path=$2
  local payload=$3
  [[ -d "${path}" ]] || die "${label} cache dir does not exist: ${path}"
  require_file "${path}/metadata.json"
  require_file "${path}/${payload}"
}

select_suite() {
  case "${SUITE}" in
    goal)
      EXPERIMENT="${GOAL_EXP}"
      VIDEO_EMBEDDING_DIR="${GOAL_VIDEO_EMBEDDING_DIR}"
      VIDEO_LATENT_DIR="${GOAL_VIDEO_LATENT_DIR}"
      ;;
    spatial)
      EXPERIMENT="${SPATIAL_EXP}"
      VIDEO_EMBEDDING_DIR="${SPATIAL_VIDEO_EMBEDDING_DIR}"
      VIDEO_LATENT_DIR="${SPATIAL_VIDEO_LATENT_DIR}"
      ;;
    object)
      EXPERIMENT="${OBJECT_EXP}"
      VIDEO_EMBEDDING_DIR="${OBJECT_VIDEO_EMBEDDING_DIR}"
      VIDEO_LATENT_DIR="${OBJECT_VIDEO_LATENT_DIR}"
      ;;
    *)
      die "unknown SUITE=${SUITE}; expected goal, spatial, or object"
      ;;
  esac
}

select_variant() {
  case "${VARIANT}" in
    vlsp_source_condition|source_condition)
      JOB_GROUP="vlsp_source_condition"
      SOURCE_ENABLED=true
      SOURCE_MODE=video_prior_sample
      CONDITIONING_MODE=normal
      ;;
    vlsp_source_only|source_only)
      JOB_GROUP="vlsp_source_only"
      SOURCE_ENABLED=true
      SOURCE_MODE=video_prior_sample
      CONDITIONING_MODE=zero_video
      ;;
    baseline_gaussian|baseline)
      JOB_GROUP="baseline_gaussian"
      SOURCE_ENABLED=false
      SOURCE_MODE=gaussian
      CONDITIONING_MODE=normal
      ;;
    *)
      die "unknown VARIANT=${VARIANT}; expected source_condition, source_only, or baseline"
      ;;
  esac
}

preflight() {
  [[ -x "${PYTHON}" ]] || die "Python is not executable: ${PYTHON}"
  require_file "${REPO_DIR}/checkpoints/video_backbone/tokenizer/tokenizer.pth"
  require_file "${REPO_DIR}/checkpoints/text_encoder/t5-11b/pytorch_model.bin"

  if is_true "${USE_OFFLINE_VIDEO_EMBEDDINGS}" && is_true "${USE_OFFLINE_VIDEO_LATENTS}"; then
    die "USE_OFFLINE_VIDEO_EMBEDDINGS and USE_OFFLINE_VIDEO_LATENTS are mutually exclusive"
  fi
  if is_true "${USE_OFFLINE_VIDEO_EMBEDDINGS}"; then
    require_cache_dir "${SUITE} video embedding" "${VIDEO_EMBEDDING_DIR}" "crossattn_emb.fp16.memmap"
    OFFLINE_VIDEO_EMBEDDING_DIR="${VIDEO_EMBEDDING_DIR}"
    OFFLINE_VIDEO_EMBEDDING_REQUIRED=true
    OFFLINE_VIDEO_LATENT_DIR=""
    OFFLINE_VIDEO_LATENT_REQUIRED=false
  elif is_true "${USE_OFFLINE_VIDEO_LATENTS}"; then
    require_cache_dir "${SUITE} video latent" "${VIDEO_LATENT_DIR}" "video_latent.fp16.memmap"
    OFFLINE_VIDEO_EMBEDDING_DIR=""
    OFFLINE_VIDEO_EMBEDDING_REQUIRED=false
    OFFLINE_VIDEO_LATENT_DIR="${VIDEO_LATENT_DIR}"
    OFFLINE_VIDEO_LATENT_REQUIRED=true
  else
    OFFLINE_VIDEO_EMBEDDING_DIR=""
    OFFLINE_VIDEO_EMBEDDING_REQUIRED=false
    OFFLINE_VIDEO_LATENT_DIR=""
    OFFLINE_VIDEO_LATENT_REQUIRED=false
  fi
}

select_suite
select_variant
preflight

STAMP=$(date -u +%Y%m%d_%H%M%S)
JOB_NAME="${JOB_PREFIX}_${STAMP}"
RANK_PID_FILE="${LAUNCH_DIR}/${JOB_NAME}.ranks"
DONE_FILE="${LAUNCH_DIR}/${JOB_NAME}.done"
META_FILE="${LAUNCH_DIR}/${JOB_NAME}.meta"

cat > "${META_FILE}" <<EOF
JOB_NAME=${JOB_NAME}
SUITE=${SUITE}
VARIANT=${VARIANT}
JOB_GROUP=${JOB_GROUP}
EXPERIMENT=${EXPERIMENT}
MODEL_DIR=${MODEL_DIR}
PYTHON=${PYTHON}
OFFLINE_VIDEO_EMBEDDING_DIR=${OFFLINE_VIDEO_EMBEDDING_DIR}
OFFLINE_VIDEO_LATENT_DIR=${OFFLINE_VIDEO_LATENT_DIR}
MIMIC_DISTRIBUTED_BACKEND=nccl
NCCL_SHM_DISABLE=1
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
MAX_ITER=${MAX_ITER}
DATALOADER_TRAIN_GLOBAL_BSZ=${DATALOADER_TRAIN_GLOBAL_BSZ}
EOF

COMMON_OPTS=(
  --
  "experiment=${EXPERIMENT}"
  "job.group=${JOB_GROUP}"
  "job.name=${JOB_NAME}"
  "trainer.max_iter=${MAX_ITER}"
  "trainer.logging_iter=${LOGGING_ITER}"
  "trainer.validation_iter=${VALIDATION_ITER}"
  "trainer.run_validation=${RUN_VALIDATION}"
  "trainer.grad_accum_iter=1"
  "checkpoint.save_iter=${SAVE_ITER}"
  "dataloader_train.batch_size.global_bsz=${DATALOADER_TRAIN_GLOBAL_BSZ}"
  "dataloader_train.num_workers=${DATALOADER_TRAIN_NUM_WORKERS}"
  "dataloader_train.prefetch_factor=${DATALOADER_TRAIN_PREFETCH_FACTOR}"
  "dataloader_train.persistent_workers=${DATALOADER_TRAIN_PERSISTENT_WORKERS}"
  "dataloader_train.pin_memory=${DATALOADER_TRAIN_PIN_MEMORY}"
  "dataloader_train.in_order=${DATALOADER_TRAIN_IN_ORDER}"
  "model.config.offline_video_embedding_dir=${OFFLINE_VIDEO_EMBEDDING_DIR}"
  "model.config.offline_video_embedding_required=${OFFLINE_VIDEO_EMBEDDING_REQUIRED}"
  "model.config.offline_video_latent_dir=${OFFLINE_VIDEO_LATENT_DIR}"
  "model.config.offline_video_latent_required=${OFFLINE_VIDEO_LATENT_REQUIRED}"
  "model.config.pipe_config.action_source_prior.enabled=${SOURCE_ENABLED}"
  "model.config.pipe_config.action_source_prior.mode=${SOURCE_MODE}"
  "model.config.pipe_config.action_conditioning.mode=${CONDITIONING_MODE}"
)

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) starting ${JOB_NAME}"
: > "${RANK_PID_FILE}"
pids=()
for rank in 0 1 2 3; do
  (
    cd "${MODEL_DIR}"
    export CUDA_VISIBLE_DEVICES="${MIGS[$rank]}"
    export NVIDIA_VISIBLE_DEVICES="${MIGS[$rank]}"
    export NCCL_MIG_ID="${MIGS[$rank]}"
    export RANK="${rank}"
    export LOCAL_RANK="${rank}"
    export WORLD_SIZE=4
    export LOCAL_WORLD_SIZE=4
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT}"
    export PYTHONPATH="${MODEL_DIR}:${PYTHONPATH:-}"
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
    export MIMIC_DISTRIBUTED_BACKEND=nccl
    export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
    export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
    export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
    export DEBUG_STEP_TIMING="${DEBUG_STEP_TIMING:-0}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export DEBUG_W2A_TIMING="${DEBUG_W2A_TIMING:-0}"
    "${PYTHON}" -m scripts.train --config cosmos_predict2/configs/config.py "${COMMON_OPTS[@]}"
  ) > "${LOG_DIR}/${JOB_NAME}.rank${rank}.log" 2>&1 &
  pid=$!
  echo "rank${rank} pid=${pid} mig=${MIGS[$rank]} log=${LOG_DIR}/${JOB_NAME}.rank${rank}.log" | tee -a "${RANK_PID_FILE}"
  pids+=("${pid}")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) JOB=${JOB_NAME} STATUS=${status}" | tee "${DONE_FILE}"
exit "${status}"
