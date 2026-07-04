#!/usr/bin/env python3
"""Inspect LIBERO action samples stored as .safetensors or .zarr episodes.

The current mimic-video LIBERO action pipeline stores converted data as one
.zarr directory per episode. This script also supports .safetensors because some
precomputed embedding pipelines use that format.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable


def iter_samples(data_dir: Path) -> Iterable[Path]:
    yield from data_dir.rglob('*.safetensors')
    yield from data_dir.rglob('*.zarr')


def describe_value(name: str, value) -> None:
    shape = getattr(value, 'shape', None)
    dtype = getattr(value, 'dtype', None)
    print(f'  key: {name}')
    print(f'    shape: {tuple(shape) if shape is not None else None}')
    print(f'    dtype: {dtype}')


def inspect_safetensors(path: Path) -> None:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit('safetensors is not installed in this environment') from exc

    print(f'file: {path}')
    with safe_open(str(path), framework='pt', device='cpu') as f:
        for key in sorted(f.keys()):
            tensor = f.get_tensor(key)
            describe_value(key, tensor)


def inspect_zarr(path: Path) -> None:
    try:
        import zarr
    except ImportError as exc:
        raise SystemExit('zarr is not installed in this environment') from exc

    print(f'file: {path}')
    root = zarr.open(str(path), mode='r')

    def walk(group, prefix: str = '') -> None:
        for key in sorted(group.keys()):
            value = group[key]
            name = f'{prefix}/{key}' if prefix else key
            if hasattr(value, 'shape'):
                describe_value(name, value)
            else:
                walk(value, name)

    walk(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--num-files', type=int, default=3)
    args = parser.parse_args()

    paths = list(itertools.islice(iter_samples(args.data_dir), args.num_files))
    if not paths:
        raise SystemExit(f'No .safetensors or .zarr samples found under {args.data_dir}')

    for path in paths:
        if path.suffix == '.safetensors':
            inspect_safetensors(path)
        elif path.suffix == '.zarr':
            inspect_zarr(path)
        else:
            continue


if __name__ == '__main__':
    main()
