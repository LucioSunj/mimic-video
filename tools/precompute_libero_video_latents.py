#!/usr/bin/env python3
"""Precompute LIBERO video tokenizer latents for world2action training.

The cache is indexed by MimicDataset sample_idx. It stores the frozen Cosmos
video tokenizer output used by Video2WorldPipeline.get_mimic_data_and_condition,
so training can skip RGB zarr reads and tokenizer.encode while still sampling a
fresh video sigma/epsilon and running the frozen video DiT online.

The script can also be launched with torchrun. In that mode each rank owns a
non-overlapping sample_idx shard and writes to disjoint rows of the same memmap.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm


def repo_paths() -> tuple[Path, Path]:
    this = Path(__file__).resolve()
    repo = this.parents[1]
    return repo, repo / "model"


def move_to_cuda(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.to(device="cuda", non_blocking=True)
    if isinstance(value, dict):
        return {k: move_to_cuda(v) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_cuda(v) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_cuda(v) for v in value)
    return value


def encode_mimic_latent(tokenizer, batch: dict[str, torch.Tensor], *, sigma_data: float) -> tuple[np.ndarray, np.ndarray]:
    sample_idx = batch["sample_idx"].detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    obs_rgb = move_to_cuda(batch["obs/workspace_rgb"])
    action_rgb = move_to_cuda(batch["action/workspace_rgb"])
    raw_state = torch.concat((obs_rgb, action_rgb), dim=2)
    latent = tokenizer.encode(raw_state) * sigma_data
    if latent.shape[2] != 16:
        padded = torch.zeros((latent.shape[0], 16, 16, 60, 80), device=latent.device, dtype=latent.dtype)
        padded[:, :, : latent.shape[2], :, :] = latent
        latent = padded
    return sample_idx, latent.detach().to(dtype=torch.float16, device="cpu").numpy()


def init_distributed_from_env() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    elif torch.cuda.is_available():
        torch.cuda.set_device(0)
    return rank, world_size, local_rank


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="RGB LIBERO dataloading config, e.g. libero_goal_full")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override dataset.dataset.data_dir")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="Debug only: precompute the first N samples")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed_from_env()
    is_rank0 = rank == 0

    repo, model_dir = repo_paths()
    sys.path.insert(0, str(model_dir))

    from cosmos_predict2.configs.config_video2world import get_cosmos_predict2_video2world_pipeline
    from cosmos_predict2.configs.defaults.data_action import get_data_config, get_dataset
    from imaginaire.lazy_config import instantiate

    if is_rank0:
        if args.output_dir.exists():
            if not args.overwrite:
                raise SystemExit(f"output dir already exists; pass --overwrite to replace: {args.output_dir}")
            shutil.rmtree(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)

    cfg = get_data_config(args.config)
    if args.data_dir is not None:
        cfg.dataset.dataset.data_dir = str(args.data_dir)
    dataset = get_dataset(cfg, is_train=True)
    cache_len = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    if cache_len <= 0:
        raise SystemExit("empty dataset/cache length")

    shard_indices = list(range(rank, cache_len, world_size))
    shard_dataset = torch.utils.data.Subset(dataset, shard_indices)
    print(
        f"rank={rank} local_rank={local_rank} world_size={world_size} "
        f"shard_samples={len(shard_indices)} cache_len={cache_len}",
        flush=True,
    )

    loader_kwargs = dict(
        dataset=shard_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(**loader_kwargs)

    pipe_cfg = copy.deepcopy(get_cosmos_predict2_video2world_pipeline(model_size="2B", resolution="480", fps=10))
    tokenizer = instantiate(pipe_cfg.tokenizer)
    tokenizer.to(device="cuda", dtype=torch.bfloat16)

    latent_shape = (cache_len, 16, 16, 60, 80)
    est_gb = np.prod(latent_shape) * np.dtype(np.float16).itemsize / 1e9
    if is_rank0:
        print(f"dataset_len: {len(dataset)}")
        print(f"cache_len: {cache_len}")
        print(f"latent_shape: {latent_shape}")
        print(f"estimated_latent_size_gb: {est_gb:.2f}")
        print(f"world_size: {world_size}")
        np.memmap(
            args.output_dir / "video_latent.fp16.memmap",
            dtype=np.float16,
            mode="w+",
            shape=latent_shape,
        ).flush()
    barrier(world_size)

    latent_mmap = np.memmap(
        args.output_dir / "video_latent.fp16.memmap",
        dtype=np.float16,
        mode="r+",
        shape=latent_shape,
    )

    processed = 0
    t0 = time.perf_counter()
    num_latent_conditional_frames = int(tokenizer.get_latent_num_frames(5))

    pbar = tqdm(
        total=len(shard_indices),
        desc=f"precomputing video tokenizer latents rank {rank}/{world_size}",
        disable=world_size > 1 and not is_rank0,
    )
    with torch.inference_mode():
        for batch in loader:
            sample_idx, latent_np = encode_mimic_latent(tokenizer, batch, sigma_data=pipe_cfg.sigma_data)
            keep = sample_idx < cache_len
            if not np.any(keep):
                continue
            sample_idx = sample_idx[keep]
            latent_np = latent_np[keep]

            latent_mmap[sample_idx] = latent_np
            processed += len(sample_idx)
            pbar.update(len(sample_idx))
            if processed >= len(shard_indices):
                break
    pbar.close()
    latent_mmap.flush()
    barrier(world_size)

    total_processed = processed
    if world_size > 1:
        processed_tensor = torch.tensor([processed], device="cuda", dtype=torch.long)
        dist.all_reduce(processed_tensor, op=dist.ReduceOp.SUM)
        total_processed = int(processed_tensor.item())

    if is_rank0:
        metadata = {
            "format_version": 1,
            "repo": str(repo),
            "data_config": args.config,
            "data_dir": str(cfg.dataset.dataset.data_dir),
            "dataset_len": len(dataset),
            "cache_len": cache_len,
            "processed": total_processed,
            "world_size": world_size,
            "latent_file": "video_latent.fp16.memmap",
            "latent_dtype": "float16",
            "latent_shape": list(latent_shape),
            "sigma_data": float(pipe_cfg.sigma_data),
            "obs_pixel_frames": 5,
            "total_pixel_frames": 61,
            "num_latent_conditional_frames": num_latent_conditional_frames,
            "elapsed_sec": time.perf_counter() - t0,
        }
        with (args.output_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        print(f"processed: {total_processed}")
        print(f"elapsed_sec: {metadata['elapsed_sec']:.2f}")
        print(f"output_dir: {args.output_dir}")
    barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
