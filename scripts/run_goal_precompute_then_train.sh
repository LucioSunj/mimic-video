#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${MIMIC_VIDEO_REPO_DIR:-/XYFS02/HDD_POOL/nju_shklu/nju_shklu_1/Video-as-action-generation-source/mimic-video}
PYTHON=${MIMIC_VIDEO_PYTHON:-/XYFS02/HDD_POOL/nju_shklu/nju_shklu_1/mixture-of-horizon-for-world-action-model/mimic-video/model/.venv/bin/python}
DATA_ROOT=${MIMIC_LIBERO_DATA_ROOT:-"${REPO_DIR}/data"}
VIDEO_LATENT_ROOT=${MIMIC_LIBERO_VIDEO_LATENT_ROOT:-"${DATA_ROOT}/libero_video_latents"}
GOAL_DATA_DIR=${MIMIC_LIBERO_GOAL_DIR:-"${DATA_ROOT}/libero_goal_full"}
GOAL_VIDEO_LATENT_DIR=${MIMIC_LIBERO_GOAL_VIDEO_LATENT_DIR:-"${VIDEO_LATENT_ROOT}/libero_goal_full_v2w_libero_goal_tokenizer_latents"}

GPUS=${GPUS:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
PRECOMPUTE_MASTER_PORT=${PRECOMPUTE_MASTER_PORT:-29671}
MASTER_PORT=${MASTER_PORT:-29631}
PRECOMPUTE_BATCH_SIZE=${PRECOMPUTE_BATCH_SIZE:-8}
PRECOMPUTE_NUM_WORKERS=${PRECOMPUTE_NUM_WORKERS:-1}
PRECOMPUTE_PREFETCH_FACTOR=${PRECOMPUTE_PREFETCH_FACTOR:-1}

MAX_ITER=${MAX_ITER:-50000}
SAVE_ITER=${SAVE_ITER:-1000}
LOGGING_ITER=${LOGGING_ITER:-100}
VALIDATION_ITER=${VALIDATION_ITER:-1000000}
DATALOADER_TRAIN_GLOBAL_BSZ=${DATALOADER_TRAIN_GLOBAL_BSZ:-32}
DATALOADER_TRAIN_NUM_WORKERS=${DATALOADER_TRAIN_NUM_WORKERS:-1}
DATALOADER_TRAIN_PREFETCH_FACTOR=${DATALOADER_TRAIN_PREFETCH_FACTOR:-1}
DATALOADER_TRAIN_PERSISTENT_WORKERS=${DATALOADER_TRAIN_PERSISTENT_WORKERS:-false}
DATALOADER_TRAIN_PIN_MEMORY=${DATALOADER_TRAIN_PIN_MEMORY:-false}
DATALOADER_TRAIN_IN_ORDER=${DATALOADER_TRAIN_IN_ORDER:-true}

mkdir -p "${VIDEO_LATENT_ROOT}"

printf '[%s] goal tokenizer latent precompute starting\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'repo=%s\n' "${REPO_DIR}"
printf 'goal_data_dir=%s\n' "${GOAL_DATA_DIR}"
printf 'goal_video_latent_dir=%s\n' "${GOAL_VIDEO_LATENT_DIR}"
printf 'gpus=%s nproc=%s precompute_batch_size=%s\n' "${GPUS}" "${NPROC_PER_NODE}" "${PRECOMPUTE_BATCH_SIZE}"

cd "${REPO_DIR}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${REPO_DIR}/model:${PYTHONPATH:-}"

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${PRECOMPUTE_MASTER_PORT}" \
  tools/precompute_libero_video_latents.py \
  --config libero_goal_full \
  --data-dir "${GOAL_DATA_DIR}" \
  --output-dir "${GOAL_VIDEO_LATENT_DIR}" \
  --batch-size "${PRECOMPUTE_BATCH_SIZE}" \
  --num-workers "${PRECOMPUTE_NUM_WORKERS}" \
  --prefetch-factor "${PRECOMPUTE_PREFETCH_FACTOR}" \
  --pin-memory \
  --overwrite

"${PYTHON}" - "${GOAL_VIDEO_LATENT_DIR}" <<'PYVERIFY'
import json
import pathlib
import sys
cache_dir = pathlib.Path(sys.argv[1])
meta_path = cache_dir / "metadata.json"
if not meta_path.exists():
    raise SystemExit(f"missing metadata: {meta_path}")
meta = json.loads(meta_path.read_text())
latent_file = cache_dir / meta.get("latent_file", "video_latent.fp16.memmap")
if not latent_file.exists():
    raise SystemExit(f"missing latent memmap: {latent_file}")
if int(meta["processed"]) != int(meta["cache_len"]):
    raise SystemExit(f"incomplete cache: processed={meta['processed']} cache_len={meta['cache_len']}")
print(f"verified tokenizer latent cache: processed={meta['processed']} cache_len={meta['cache_len']} world_size={meta.get('world_size')}")
PYVERIFY

printf '[%s] goal tokenizer latent precompute complete; starting goal training trio\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

export MIMIC_VIDEO_REPO_DIR="${REPO_DIR}"
export MIMIC_VIDEO_PYTHON="${PYTHON}"
export MIMIC_LIBERO_DATA_ROOT="${DATA_ROOT}"
export MIMIC_LIBERO_VIDEO_LATENT_ROOT="${VIDEO_LATENT_ROOT}"
export MIMIC_LIBERO_GOAL_VIDEO_LATENT_DIR="${GOAL_VIDEO_LATENT_DIR}"
export USE_OFFLINE_VIDEO_EMBEDDINGS=false
export USE_OFFLINE_VIDEO_LATENTS=true
export SUITE_FILTER=goal
export GPUS="${GPUS}"
export NPROC_PER_NODE="${NPROC_PER_NODE}"
export MASTER_PORT="${MASTER_PORT}"
export MAX_ITER="${MAX_ITER}"
export SAVE_ITER="${SAVE_ITER}"
export LOGGING_ITER="${LOGGING_ITER}"
export VALIDATION_ITER="${VALIDATION_ITER}"
export DATALOADER_TRAIN_GLOBAL_BSZ="${DATALOADER_TRAIN_GLOBAL_BSZ}"
export DATALOADER_TRAIN_NUM_WORKERS="${DATALOADER_TRAIN_NUM_WORKERS}"
export DATALOADER_TRAIN_PREFETCH_FACTOR="${DATALOADER_TRAIN_PREFETCH_FACTOR}"
export DATALOADER_TRAIN_PERSISTENT_WORKERS="${DATALOADER_TRAIN_PERSISTENT_WORKERS}"
export DATALOADER_TRAIN_PIN_MEMORY="${DATALOADER_TRAIN_PIN_MEMORY}"
export DATALOADER_TRAIN_IN_ORDER="${DATALOADER_TRAIN_IN_ORDER}"
export DEBUG_STEP_TIMING=${DEBUG_STEP_TIMING:-1}
export DEBUG_STEP_TIMING_INTERVAL=${DEBUG_STEP_TIMING_INTERVAL:-100}
export DEBUG_W2A_TIMING=${DEBUG_W2A_TIMING:-1}
export DEBUG_W2A_TIMING_INTERVAL=${DEBUG_W2A_TIMING_INTERVAL:-100}

"${REPO_DIR}/scripts/train_libero_action_queue.sh" run

printf '[%s] goal training trio complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
