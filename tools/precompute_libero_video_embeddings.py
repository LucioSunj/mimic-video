#!/usr/bin/env python3
"""Precompute LIBERO world2action video conditioning embeddings.

This writes the frozen video2world hidden state used by the action decoder's
cross-attention to a memory-mapped fp16 cache. The cache is indexed by the
MimicDataset sample_idx, which is the same global chunk index consumed by the
training sampler.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm


def repo_paths() -> tuple[Path, Path]:
    this = Path(__file__).resolve()
    repo = this.parents[1]
    model_dir = repo / "model"
    return repo, model_dir


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


def draw_video_sigma(pipe, batch_size: int, *, is_video_batch: bool) -> torch.Tensor:
    sigma_B = pipe.scheduler.sample_sigma(batch_size)
    sigma_B_1 = rearrange(sigma_B, "b -> b 1")
    multiplier = math.sqrt(pipe.config.state_t) if pipe.config.adjust_video_noise and is_video_batch else 1.0
    sigma_B_1 = sigma_B_1 * multiplier
    if is_video_batch:
        log_200 = math.log(200)
        log_100000 = math.log(100000)
        mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < 0.05
        log_new_sigma = torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * (
            log_100000 - log_200
        ) + log_200
        sigma_B_1 = torch.where(mask, log_new_sigma.exp(), sigma_B_1)
    return sigma_B_1


@torch.inference_mode()
def compute_embeddings(pipe, batch: dict[str, torch.Tensor], xattn_layer_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_idx = batch["sample_idx"].detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    batch_cuda = move_to_cuda(batch)
    _, video_B_C_T_H_W, condition = pipe.get_mimic_data_and_condition(batch_cuda)
    video_sigma_B_1 = draw_video_sigma(pipe, video_B_C_T_H_W.shape[0], is_video_batch=condition.is_video)
    video_epsilon_B_C_T_H_W = torch.randn(video_B_C_T_H_W.size(), device="cuda", dtype=pipe.precision)
    world_pred = pipe.denoise(
        video_B_C_T_H_W + video_epsilon_B_C_T_H_W * rearrange(video_sigma_B_1, "b t -> b 1 t 1 1"),
        video_sigma_B_1,
        condition,
        use_cuda_graphs=False,
        return_only_hidden_states_up_to=xattn_layer_idx,
        return_decoded_video=False,
    )
    crossattn = world_pred.hidden_states[xattn_layer_idx]
    B, T, H, W, D = crossattn.shape
    crossattn = crossattn.reshape(B, T * H * W, D)
    return (
        sample_idx,
        crossattn.detach().to(dtype=torch.float16, device="cpu").numpy(),
        video_sigma_B_1.detach().to(dtype=torch.float32, device="cpu").numpy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="LIBERO dataloading config, e.g. libero_goal_full")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override dataset.dataset.data_dir")
    parser.add_argument("--video-dit-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xattn-layer-idx", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="Debug only: precompute the first N samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo, model_dir = repo_paths()
    sys.path.insert(0, str(model_dir))

    from cosmos_predict2.configs.config_video2world import get_cosmos_predict2_video2world_pipeline
    from cosmos_predict2.configs.defaults.data_action import get_data_config, get_dataset
    from cosmos_predict2.pipelines.video2world import Video2WorldPipeline

    if args.output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"output dir already exists; pass --overwrite to replace: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = get_data_config(args.config)
    if args.data_dir is not None:
        cfg.dataset.dataset.data_dir = str(args.data_dir)
    dataset = get_dataset(cfg, is_train=True)
    cache_len = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    if cache_len <= 0:
        raise SystemExit("empty dataset/cache length")

    loader_kwargs = dict(
        dataset=dataset,
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
    pipe_cfg.guardrail_config.enabled = False
    pipe = Video2WorldPipeline.from_config(
        pipe_cfg,
        dit_path=str(args.video_dit_path),
        use_text_encoder=False,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )
    pipe.requires_grad_(False)
    pipe.eval()

    crossattn_mmap = None
    sigma_mmap = None
    crossattn_shape = None
    processed = 0
    t0 = time.perf_counter()

    pbar = tqdm(total=cache_len, desc="precomputing video embeddings")
    for batch in loader:
        sample_idx, crossattn_np, sigma_np = compute_embeddings(pipe, batch, args.xattn_layer_idx)
        keep = sample_idx < cache_len
        if not np.any(keep):
            break
        sample_idx = sample_idx[keep]
        crossattn_np = crossattn_np[keep]
        sigma_np = sigma_np[keep]

        if crossattn_mmap is None:
            num_tokens, dim = crossattn_np.shape[1:]
            crossattn_shape = (cache_len, num_tokens, dim)
            est_gb = np.prod(crossattn_shape) * np.dtype(np.float16).itemsize / 1e9
            print(f"dataset_len: {len(dataset)}")
            print(f"cache_len: {cache_len}")
            print(f"crossattn_shape: {crossattn_shape}")
            print(f"estimated_crossattn_size_gb: {est_gb:.2f}")
            crossattn_mmap = np.memmap(
                args.output_dir / "crossattn_emb.fp16.memmap",
                dtype=np.float16,
                mode="w+",
                shape=crossattn_shape,
            )
            sigma_mmap = np.lib.format.open_memmap(
                args.output_dir / "video_sigma.npy",
                dtype=np.float32,
                mode="w+",
                shape=(cache_len, 1),
            )

        crossattn_mmap[sample_idx] = crossattn_np
        sigma_mmap[sample_idx] = sigma_np
        processed += len(sample_idx)
        pbar.update(len(sample_idx))
        if processed >= cache_len:
            break
    pbar.close()

    if crossattn_mmap is None or sigma_mmap is None or crossattn_shape is None:
        raise SystemExit("no embeddings were computed")
    crossattn_mmap.flush()
    sigma_mmap.flush()

    metadata = {
        "format_version": 1,
        "repo": str(repo),
        "data_config": args.config,
        "data_dir": str(cfg.dataset.dataset.data_dir),
        "dataset_len": len(dataset),
        "cache_len": cache_len,
        "processed": processed,
        "xattn_layer_idx": args.xattn_layer_idx,
        "video_dit_path": str(args.video_dit_path),
        "crossattn_file": "crossattn_emb.fp16.memmap",
        "crossattn_dtype": "float16",
        "crossattn_shape": list(crossattn_shape),
        "video_sigma_file": "video_sigma.npy",
        "seed": args.seed,
        "elapsed_sec": time.perf_counter() - t0,
    }
    with (args.output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
