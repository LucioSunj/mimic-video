# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve VLSP experiment configs for Phase-0 eval/config parity audits.

Example:
    python -m scripts.vlsp_audit_config \
        --experiments vlsp_source_only_sample vlsp_source_condition_sample vlsp_baseline_gaussian \
        --markdown
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import types
from collections import OrderedDict
from dataclasses import asdict, is_dataclass
from typing import Any

import attrs
from omegaconf import DictConfig, ListConfig, OmegaConf


DEFAULT_EXPERIMENTS = [
    "vlsp_source_only_sample",
    "vlsp_source_condition_sample",
    "vlsp_baseline_gaussian",
]


def _install_config_import_stubs() -> None:
    """Keep config resolution CPU-only when optional media deps are absent."""
    if "imageio" not in sys.modules:
        imageio_stub = types.ModuleType("imageio")
        imageio_stub.__path__ = []
        sys.modules["imageio"] = imageio_stub
    if "imageio.v3" not in sys.modules:
        sys.modules["imageio.v3"] = types.ModuleType("imageio.v3")
    if "megatron" not in sys.modules:
        megatron_stub = types.ModuleType("megatron")
        megatron_stub.__path__ = []
        sys.modules["megatron"] = megatron_stub
    if "megatron.core" not in sys.modules:
        core_stub = types.ModuleType("megatron.core")
        core_stub.__path__ = []
        sys.modules["megatron.core"] = core_stub
    if "megatron.core.parallel_state" not in sys.modules:
        parallel_state_stub = types.ModuleType("megatron.core.parallel_state")
        parallel_state_stub.get_data_parallel_world_size = lambda: 1
        parallel_state_stub.get_data_parallel_rank = lambda: 0
        parallel_state_stub.is_initialized = lambda: False
        sys.modules["megatron.core.parallel_state"] = parallel_state_stub
        sys.modules["megatron.core"].parallel_state = parallel_state_stub
    if "decord" not in sys.modules:
        decord_stub = types.ModuleType("decord")

        class _VideoReader:  # pragma: no cover - import-only stub
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("decord is not available in this CPU-only config audit environment")

        decord_stub.VideoReader = _VideoReader
        decord_stub.cpu = lambda *args, **kwargs: None
        sys.modules["decord"] = decord_stub
    if "transformers" not in sys.modules:
        transformers_stub = types.ModuleType("transformers")
        transformers_stub.__path__ = []

        class _UnavailableTransformer:  # pragma: no cover - import-only stub
            @classmethod
            def from_pretrained(cls, *args: Any, **kwargs: Any) -> "_UnavailableTransformer":
                raise RuntimeError("transformers is not available in this CPU-only config audit environment")

        transformers_stub.T5EncoderModel = _UnavailableTransformer
        transformers_stub.T5TokenizerFast = _UnavailableTransformer
        sys.modules["transformers"] = transformers_stub
    if "transformers.utils" not in sys.modules:
        utils_stub = types.ModuleType("transformers.utils")
        utils_stub.__path__ = []
        sys.modules["transformers.utils"] = utils_stub
        sys.modules["transformers"].utils = utils_stub
    if "transformers.utils.logging" not in sys.modules:
        logging_stub = types.ModuleType("transformers.utils.logging")
        logging_stub.set_verbosity_error = lambda: None
        sys.modules["transformers.utils.logging"] = logging_stub
        sys.modules["transformers.utils"].logging = logging_stub
    if "apex" not in sys.modules:
        apex_stub = types.ModuleType("apex")
        apex_stub.__path__ = []
        sys.modules["apex"] = apex_stub
    if "apex.multi_tensor_apply" not in sys.modules:
        multi_tensor_apply_stub = types.ModuleType("apex.multi_tensor_apply")

        class _MultiTensorApplier:  # pragma: no cover - import-only stub
            available = False

            def __call__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("apex is not available in this CPU-only config audit environment")

        multi_tensor_apply_stub.multi_tensor_applier = _MultiTensorApplier()
        sys.modules["apex.multi_tensor_apply"] = multi_tensor_apply_stub
    if "transformer_engine" not in sys.modules:
        te_stub = types.ModuleType("transformer_engine")
        te_stub.__path__ = []
        sys.modules["transformer_engine"] = te_stub
    if "transformer_engine.pytorch" not in sys.modules:
        import torch

        te_pytorch_stub = types.ModuleType("transformer_engine.pytorch")
        te_pytorch_stub.__path__ = []
        te_pytorch_stub.RMSNorm = torch.nn.RMSNorm
        sys.modules["transformer_engine.pytorch"] = te_pytorch_stub
        sys.modules["transformer_engine"].pytorch = te_pytorch_stub
    if "transformer_engine.pytorch.attention" not in sys.modules:
        attention_stub = types.ModuleType("transformer_engine.pytorch.attention")

        class _DotProductAttention:  # pragma: no cover - import-only stub
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError(
                    "transformer_engine is not available in this CPU-only config audit environment"
                )

        def _apply_rotary_pos_emb(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - import-only stub
            raise RuntimeError("transformer_engine is not available in this CPU-only config audit environment")

        attention_stub.DotProductAttention = _DotProductAttention
        attention_stub.apply_rotary_pos_emb = _apply_rotary_pos_emb
        sys.modules["transformer_engine.pytorch.attention"] = attention_stub
    if "transformer_engine.pytorch.distributed" not in sys.modules:
        distributed_stub = types.ModuleType("transformer_engine.pytorch.distributed")
        distributed_stub.get_all_rng_states = lambda: []
        distributed_stub.graph_safe_rng_available = lambda: False
        sys.modules["transformer_engine.pytorch.distributed"] = distributed_stub
    if "transformer_engine.pytorch.module" not in sys.modules:
        te_module_stub = types.ModuleType("transformer_engine.pytorch.module")
        te_module_stub.__path__ = []
        sys.modules["transformer_engine.pytorch.module"] = te_module_stub
    if "transformer_engine.pytorch.module.base" not in sys.modules:
        import torch

        te_base_stub = types.ModuleType("transformer_engine.pytorch.module.base")
        te_base_stub.TransformerEngineBaseModule = torch.nn.Module
        sys.modules["transformer_engine.pytorch.module.base"] = te_base_stub
    if "torchvision" not in sys.modules:
        torchvision_stub = types.ModuleType("torchvision")
        torchvision_stub.__path__ = []
        sys.modules["torchvision"] = torchvision_stub
    if "torchvision.transforms" not in sys.modules:
        transforms_stub = types.ModuleType("torchvision.transforms")
        transforms_stub.__path__ = []

        class _InterpolationMode:  # pragma: no cover - import-only stub
            NEAREST = "nearest"

        class _Resize:  # pragma: no cover - import-only stub
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("torchvision is not available in this CPU-only config audit environment")

        class _GaussianBlur:  # pragma: no cover - import-only stub
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("torchvision is not available in this CPU-only config audit environment")

        transforms_stub.InterpolationMode = _InterpolationMode
        transforms_stub.Resize = _Resize
        transforms_stub.GaussianBlur = _GaussianBlur
        sys.modules["torchvision.transforms"] = transforms_stub
        sys.modules["torchvision"].transforms = transforms_stub
    if "torchvision.transforms.functional" not in sys.modules:
        functional_stub = types.ModuleType("torchvision.transforms.functional")

        def _unavailable(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - import-only stub
            raise RuntimeError("torchvision is not available in this CPU-only config audit environment")

        functional_stub.resize = _unavailable
        functional_stub.center_crop = _unavailable
        functional_stub.to_tensor = _unavailable
        sys.modules["torchvision.transforms.functional"] = functional_stub
        sys.modules["torchvision.transforms"].functional = functional_stub
    if "safetensors" not in sys.modules:
        safetensors_stub = types.ModuleType("safetensors")
        safetensors_stub.__path__ = []
        sys.modules["safetensors"] = safetensors_stub
    if "safetensors.torch" not in sys.modules:
        safetensors_torch_stub = types.ModuleType("safetensors.torch")

        def _safetensors_unavailable(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - import-only stub
            raise RuntimeError("safetensors is not available in this CPU-only config audit environment")

        safetensors_torch_stub.load = _safetensors_unavailable
        safetensors_torch_stub.load_file = _safetensors_unavailable
        safetensors_torch_stub.save_file = _safetensors_unavailable
        sys.modules["safetensors.torch"] = safetensors_torch_stub
    if "diffusers" not in sys.modules:
        diffusers_stub = types.ModuleType("diffusers")
        diffusers_stub.__path__ = []
        sys.modules["diffusers"] = diffusers_stub
    if "diffusers.configuration_utils" not in sys.modules:
        configuration_utils_stub = types.ModuleType("diffusers.configuration_utils")
        configuration_utils_stub.register_to_config = lambda fn: fn
        sys.modules["diffusers.configuration_utils"] = configuration_utils_stub
    if "diffusers.schedulers" not in sys.modules:
        schedulers_stub = types.ModuleType("diffusers.schedulers")

        class _KDPM2DiscreteScheduler:  # pragma: no cover - import-only stub
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.config = types.SimpleNamespace(**kwargs)

        schedulers_stub.KDPM2DiscreteScheduler = _KDPM2DiscreteScheduler
        sys.modules["diffusers.schedulers"] = schedulers_stub


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _jsonable(value: Any) -> Any:
    # NOTE: to_container / asdict can leave nested LazyDict callables (e.g. a
    # data_config ``_target_`` function) in the result, which json.dumps cannot
    # serialize. Recurse through the results and stringify callables so JSON
    # mode does not crash and markdown/JSON agree.
    if isinstance(value, (DictConfig, ListConfig)):
        return _jsonable(OmegaConf.to_container(value, resolve=True))
    if attrs.has(value.__class__):
        return _jsonable(attrs.asdict(value))
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if callable(value):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None) or repr(value)
        return f"<callable {module + '.' if module else ''}{qualname}>"
    if hasattr(value, "__dict__"):
        return {
            str(k): _jsonable(v)
            for k, v in vars(value).items()
            if not k.startswith("_") and isinstance(v, (str, int, float, bool, dict, list, tuple))
        }
    return str(value)


def _fields(prefix: str, obj: Any) -> OrderedDict[str, Any]:
    data = _jsonable(obj)
    fields: OrderedDict[str, Any] = OrderedDict()
    if isinstance(data, dict):
        for key in sorted(data):
            fields[f"{prefix}.{key}"] = data[key]
    else:
        fields[prefix] = data
    return fields


def _describe_config_ref(obj: Any) -> Any:
    """Full resolved mapping for a config reference (callables stringified by
    ``_jsonable``).

    A cherry-picked key subset used to hide the split-identifying kwargs (dataset
    dir/name/split) that live under whatever keys a given data_config actually
    uses, so the parity sheet could not tell goal_half from goal_tenth. Dumping
    the whole thing is strictly more informative and no longer crashes JSON mode.
    """
    if obj is None:
        return None
    return _jsonable(obj)


def resolve_experiment(experiment: str) -> OrderedDict[str, Any]:
    _install_config_import_stubs()
    config_log = io.StringIO()
    with contextlib.redirect_stdout(config_log):
        from cosmos_predict2.configs.config import make_config
        from imaginaire.utils.config_helper import override

        config = override(make_config(), ["--", f"experiment={experiment}"])
    if config_log.getvalue():
        print(config_log.getvalue(), file=sys.stderr, end="")
    model_config = config.model.config
    pipe_config = model_config.pipe_config
    net_config = pipe_config.net

    row: OrderedDict[str, Any] = OrderedDict()
    row["experiment"] = experiment
    row["job.group"] = _get(config.job, "group")
    row["job.name"] = _get(config.job, "name")
    row.update(_fields("action_source_prior", pipe_config.action_source_prior))
    row.update(_fields("action_conditioning", pipe_config.action_conditioning))
    row["ema.enabled"] = _get(pipe_config.ema, "enabled")
    row["ema.rate"] = _get(pipe_config.ema, "rate")
    row["scheduler.alpha"] = _get(pipe_config.scheduler, "alpha")
    row["scheduler.beta"] = _get(pipe_config.scheduler, "beta")
    row["scheduler.num_denoising_steps"] = _get(pipe_config.scheduler, "num_denoising_steps")
    row["net.max_horizon"] = _get(net_config, "max_horizon")
    row["net.out_channels"] = _get(net_config, "out_channels")
    row["net.crossattn_emb_channels"] = _get(net_config, "crossattn_emb_channels")
    row["pipe_config.xattn_layer_idx"] = _get(pipe_config, "xattn_layer_idx")
    row["model.config.offline_video_embedding_dir"] = _get(model_config, "offline_video_embedding_dir")
    row["model.config.offline_video_embedding_required"] = _get(model_config, "offline_video_embedding_required")
    row["model.config.offline_video_latent_dir"] = _get(model_config, "offline_video_latent_dir")
    row["model.config.offline_video_latent_required"] = _get(model_config, "offline_video_latent_required")
    row["model.config.sampled_mse_probe_interval"] = _get(model_config, "sampled_mse_probe_interval")
    row["data_config"] = _describe_config_ref(config.data_config)
    row["video_dataset_train"] = _describe_config_ref(config.video_dataset_train)
    row["video_dataset_val"] = _describe_config_ref(config.video_dataset_val)
    return row


def _filtered_rows(rows: OrderedDict[str, OrderedDict[str, Any]], diff_only: bool) -> list[str]:
    fields = list(next(iter(rows.values())).keys())
    if not diff_only or len(rows) <= 1:
        return fields
    first = next(iter(rows.values()))
    keep = []
    for field in fields:
        first_val = first[field]
        if any(row[field] != first_val for row in rows.values()):
            keep.append(field)
    return keep


def _format_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return "" if value is None else str(value)


def print_markdown(rows: OrderedDict[str, OrderedDict[str, Any]], diff_only: bool) -> None:
    fields = _filtered_rows(rows, diff_only)
    experiments = list(rows)
    print("| item | " + " | ".join(experiments) + " |")
    print("|---|" + "|".join("---" for _ in experiments) + "|")
    for field in fields:
        values = [_format_value(rows[experiment][field]).replace("|", "\\|") for experiment in experiments]
        print(f"| {field} | " + " | ".join(values) + " |")


def print_json(rows: OrderedDict[str, OrderedDict[str, Any]], diff_only: bool) -> None:
    fields = _filtered_rows(rows, diff_only)
    filtered = OrderedDict(
        (experiment, OrderedDict((field, row[field]) for field in fields)) for experiment, row in rows.items()
    )
    print(json.dumps(filtered, indent=2, sort_keys=False, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--diff", action="store_true", help="Only print fields that differ from the first experiment")
    parser.add_argument("--markdown", action="store_true", help="Emit a markdown table instead of JSON")
    args = parser.parse_args()

    rows = OrderedDict((experiment, resolve_experiment(experiment)) for experiment in args.experiments)
    if args.markdown:
        print_markdown(rows, args.diff)
    else:
        print_json(rows, args.diff)


if __name__ == "__main__":
    main()
