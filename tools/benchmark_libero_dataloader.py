#!/usr/bin/env python3
"""Benchmark the mimic-video LIBERO action dataloader without running the model."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


def repo_paths() -> tuple[Path, Path]:
    this = Path(__file__).resolve()
    repo = this.parents[1]
    model_dir = repo / 'model'
    return repo, model_dir


def summarize_batch(batch: Any, prefix: str = '') -> list[str]:
    lines: list[str] = []
    if isinstance(batch, dict):
        for key in sorted(batch):
            lines.extend(summarize_batch(batch[key], f'{prefix}{key}'))
    elif torch.is_tensor(batch):
        lines.append(f'{prefix}: shape={tuple(batch.shape)} dtype={batch.dtype}')
    else:
        shape = getattr(batch, 'shape', None)
        dtype = getattr(batch, 'dtype', None)
        lines.append(f'{prefix}: shape={tuple(shape) if shape is not None else None} dtype={dtype}')
    return lines


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float('nan')
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((pct / 100.0) * (len(values) - 1))))
    return values[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Hydra dataloading config name, e.g. libero_goal_full')
    parser.add_argument('--data-dir', type=Path, default=None, help='Override dataset.dataset.data_dir')
    parser.add_argument('--batch-size', type=int, default=8, help='Local batch size for the benchmark')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--prefetch-factor', type=int, default=None)
    parser.add_argument('--persistent-workers', action='store_true')
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--num-warmup', type=int, default=10)
    parser.add_argument('--num-steps', type=int, default=100)
    args = parser.parse_args()

    _repo, model_dir = repo_paths()
    sys.path.insert(0, str(model_dir))

    from cosmos_predict2.configs.defaults.data_action import get_data_config, get_dataset
    from torch.utils.data import DataLoader

    cfg = get_data_config(args.config)
    if args.data_dir is not None:
        cfg.dataset.dataset.data_dir = str(args.data_dir)

    dataset = get_dataset(cfg, is_train=True)
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = args.persistent_workers
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
    loader = DataLoader(**loader_kwargs)

    times: list[float] = []
    total_samples = 0
    first_batch = None
    it = iter(loader)
    total_iters = args.num_warmup + args.num_steps
    for i in range(total_iters):
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        dt = time.perf_counter() - t0
        if i >= args.num_warmup:
            times.append(dt)
            if isinstance(batch, dict):
                first_tensor = next((v for v in batch.values() if torch.is_tensor(v)), None)
                if first_tensor is not None:
                    total_samples += int(first_tensor.shape[0])
            if first_batch is None:
                first_batch = batch

    total_time = sum(times)
    print(f'config: {args.config}')
    print(f'data_dir: {cfg.dataset.dataset.data_dir}')
    print(f'dataset_len: {len(dataset)}')
    print(f'batch_size: {args.batch_size}')
    print(f'num_workers: {args.num_workers}')
    print(f'prefetch_factor: {args.prefetch_factor}')
    print(f'persistent_workers: {args.persistent_workers}')
    print(f'pin_memory: {args.pin_memory}')
    print(f'avg_data_time: {statistics.mean(times):.6f}')
    print(f'p50_data_time: {percentile(times, 50):.6f}')
    print(f'p90_data_time: {percentile(times, 90):.6f}')
    print(f'p99_data_time: {percentile(times, 99):.6f}')
    print(f'max_data_time: {max(times):.6f}')
    print(f'samples_per_second: {total_samples / total_time if total_time > 0 else float("nan"):.3f}')
    if isinstance(first_batch, dict):
        print('keys_in_batch:')
        for key in sorted(first_batch):
            print(f'  {key}')
    print('tensor_shapes:')
    for line in summarize_batch(first_batch):
        print(f'  {line}')


if __name__ == '__main__':
    main()
