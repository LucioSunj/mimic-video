import copy
import itertools as it

import numpy as np
from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from omegaconf import MISSING

from cosmos_predict2.configs.defaults.data_action import DATA_CONFIGS
from cosmos_predict2.configs.defaults.world2action_model import VIDEO_MODEL_CKPT_NAMES
from cosmos_predict2.configs.defaults.world2action_pipe import ACTION_DECODER_NETS
from imaginaire.lazy_config import LazyCall as L

BASE: dict = dict(
    defaults=[
        {"override /model": MISSING},
        {"override /world2action_pipe": MISSING},
        {"override /data_config": MISSING},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mimic"},
        {"override /dataloader_train": "mimic"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=dict(
        config=dict(
            train_architecture="base",
            # video_sigma_mode="logitnormal",
            pipe_config=dict(xattn_layer_idx=MISSING),
            video_pipe_config=dict(guardrail_config=dict(enabled=False)),
        )
    ),
    optimizer=dict(
        lr=MISSING,
    ),
    scheduler=dict(
        f_max=[1],
        f_min=[0.2],
        warm_up_steps=[1_000],
        cycle_lengths=[500_000],
    ),
    job=dict(
        project="vam",
        group=MISSING,
        name=MISSING,
    ),
    model_parallel=dict(
        cpu_offloading_activations=False,
        cpu_offloading_weights=False,
    ),
    checkpoint=dict(save_iter=1_000),
    trainer=dict(
        distributed_parallelism="ddp",
        grad_accum_iter=1,
        max_iter=500_000,
        logging_iter=1_000,
        validation_iter=1_000,
        run_validation=True,
    ),
)

cs = ConfigStore.instance()
cs.store(name="config", node=BASE)

world2action_pipes = ACTION_DECODER_NETS.keys()
xattn_layer_idxs = [20]
lrs = np.logspace(-5, -3, 9)[[4]]
bszs = [1, 128, 256]


def get_local_batch_size(global_bsz: int) -> int:
    res = global_bsz / parallel_state.get_data_parallel_world_size()

    if not res.is_integer():
        msg = "That batch size doesn't work with the number of gpus you have."
        raise ValueError(msg)

    return int(res)


# A representative base experiment used as the foundation for the VLSP variants
# below. The VLSP variants simply layer source-prior / action-conditioning
# overrides on top of a concrete (video_ckpt, data_config, pipe, lr, bsz) base.
vlsp_base_cfg: dict | None = None

for video_ckpt, data_config, xattn_layer_idx, lr, bsz in it.product(
    VIDEO_MODEL_CKPT_NAMES, DATA_CONFIGS.keys(), xattn_layer_idxs, lrs, bszs
):
    pipes = [pipe for pipe in world2action_pipes if data_config.startswith(pipe)]
    if not pipes:
        continue
    if len(pipes) > 1:
        raise AssertionError("data_config to pipe should be n-to-1")
    pipe = pipes[0]

    exp_name = f"w2a_{data_config}_{video_ckpt}_lr{lr:.3e}_layer{xattn_layer_idx}_bsz{bsz}"

    cfg = copy.deepcopy(BASE)
    cfg["defaults"][0]["override /model"] = video_ckpt
    cfg["defaults"][1]["override /world2action_pipe"] = pipe
    cfg["defaults"][2]["override /data_config"] = data_config
    cfg["model"]["config"]["pipe_config"]["xattn_layer_idx"] = xattn_layer_idx
    cfg["optimizer"]["lr"] = lr.item()
    cfg["job"]["group"] = pipe
    cfg["job"]["name"] = exp_name
    cfg["dataloader_train"] = {"batch_size": L(get_local_batch_size)(global_bsz=bsz)}

    if "libero" in data_config:
        cfg["checkpoint"]["save_iter"] = 99999999
        cfg["trainer"]["run_validation"] = False

    cs.store(
        group="experiment",
        package="_global_",
        name=exp_name,
        node=cfg,
    )

    # Prefer a libero base for the VLSP variants (cheap to smoke-test); otherwise
    # fall back to the first valid combination.
    if vlsp_base_cfg is None or ("libero" in data_config and "libero" not in vlsp_base_cfg["job"]["name"]):
        vlsp_base_cfg = copy.deepcopy(cfg)


# --------------------------------------------------------------------------- #
#  VLSP experiment variants                                                    #
#                                                                              #
#  source prior input  <-- action_source_prior.*                              #
#  action DiT condition <-- action_conditioning.*  (independent axis)         #
#                                                                              #
#  Every variant below is registered on top of `vlsp_base_cfg`. To apply VLSP  #
#  to a *different* base experiment, just add the same                         #
#  `model.config.pipe_config.action_source_prior.*` /                          #
#  `model.config.pipe_config.action_conditioning.*` overrides on the CLI       #
#  (see VLSP.md).                                                              #
# --------------------------------------------------------------------------- #
VLSP_DIAG_MODEL_CONFIG: dict = {"sampled_mse_probe_interval": 500}
VLSP_DIAG_TRAINER_CONFIG: dict = {"logging_iter": 500}
VLSP_DIAG_CHECKPOINT_CONFIG: dict = {"save_iter": 5_000}


def _merge_dicts(base: dict, override: dict | None) -> dict:
    merged = copy.deepcopy(base)
    if override:
        merged.update(override)
    return merged


def _vlsp_variant(
    *,
    mode: str,
    conditioning: str = "normal",
    enabled: bool | None = None,
    model_config: dict | None = None,
    trainer_config: dict | None = None,
    checkpoint_config: dict | None = None,
    **prior_kwargs,
) -> dict:
    """Build the pipe_config overrides for one VLSP variant.

    ``model_config`` entries are merged into ``model.config`` (e.g. the
    ``sampled_mse_probe_interval`` training probe).
    """
    if enabled is None:
        # gaussian is the only mode that is meaningful with VLSP disabled.
        enabled = mode != "gaussian"
    action_source_prior: dict = {"enabled": enabled, "mode": mode}
    action_source_prior.update(prior_kwargs)
    return {
        "action_source_prior": action_source_prior,
        "action_conditioning": {"mode": conditioning},
        "model_config": _merge_dicts(VLSP_DIAG_MODEL_CONFIG, model_config),
        "trainer_config": _merge_dicts(VLSP_DIAG_TRAINER_CONFIG, trainer_config),
        "checkpoint_config": _merge_dicts(VLSP_DIAG_CHECKPOINT_CONFIG, checkpoint_config),
    }


VLSP_VARIANTS: dict[str, dict] = {
    # A. baseline (exact original behaviour)
    "vlsp_baseline_gaussian": _vlsp_variant(mode="gaussian", conditioning="normal", enabled=False),
    "baseline_gaussian": _vlsp_variant(mode="gaussian", conditioning="normal", enabled=False),
    # B/C. stochastic video-prior source
    "vlsp_source_only_sample": _vlsp_variant(mode="video_prior_sample", conditioning="zero_video"),
    "vlsp_source_condition_sample": _vlsp_variant(mode="video_prior_sample", conditioning="normal"),
    # D/E. deterministic video-prior source
    "vlsp_source_only_mean": _vlsp_variant(mode="video_prior_mean", conditioning="zero_video"),
    "vlsp_source_condition_mean": _vlsp_variant(mode="video_prior_mean", conditioning="normal"),
    # H. blend video source with Gaussian
    "vlsp_blend_alpha_025": _vlsp_variant(mode="video_prior_blend", blend_alpha=0.25),
    "vlsp_blend_alpha_050": _vlsp_variant(mode="video_prior_blend", blend_alpha=0.50),
    "vlsp_blend_alpha_075": _vlsp_variant(mode="video_prior_blend", blend_alpha=0.75),
    # F/G. negative controls
    "vlsp_shuffled_source": _vlsp_variant(mode="shuffled_video_prior", conditioning="normal"),
    "vlsp_shuffled_condition": _vlsp_variant(mode="video_prior_sample", conditioning="shuffled_video"),
    # J. source dropout / mixture with Gaussian
    "vlsp_dropout_020": _vlsp_variant(mode="video_prior_dropout", source_dropout_prob=0.20),
    # I. residual source
    "vlsp_residual": _vlsp_variant(mode="video_prior_residual", residual_scale=1.0),
    # debug-only smoke mode
    "vlsp_debug_gt_action_noisy": _vlsp_variant(mode="gt_action_noisy_debug", debug_noise_std=0.05),
    # ------------------------------------------------------------------ #
    # Run-2 anti-collapse variants (see VLSP_RUN1_ANALYSIS.md). Run 1
    # (kl=0, video_prior_sample) collapsed to a Dirac source
    # (logstd pinned at the -5 floor) => fixed task-independent actions.
    # All three keep conditioning=normal and enable the sampled-MSE probe.
    # ------------------------------------------------------------------ #
    # R1: primary fix — KL keeps sigma near 1 (source stays a "hint").
    "vlsp_r1_kl_1e3": _vlsp_variant(
        mode="video_prior_sample",
        conditioning="normal",
        kl_weight=1e-3,
        model_config={"sampled_mse_probe_interval": 500},
    ),
    # R2: structural fallback — blend guarantees a sqrt(1-a^2)=0.87-std
    #     Gaussian component that no optimizer pressure can remove; its
    #     worst-case degeneration is the baseline, not a broken model.
    "vlsp_r2_blend_050": _vlsp_variant(
        mode="video_prior_blend",
        conditioning="normal",
        blend_alpha=0.5,
        kl_weight=1e-3,
        model_config={"sampled_mse_probe_interval": 500},
    ),
    # R3: R1 + per-sample source dropout — the DiT keeps a noise-robust
    #     mode by seeing pure-Gaussian sources 20% of the time.
    "vlsp_r3_kl_dropout_020": _vlsp_variant(
        mode="video_prior_dropout",
        conditioning="normal",
        kl_weight=1e-3,
        source_dropout_prob=0.20,
        model_config={"sampled_mse_probe_interval": 500},
    ),
    # ------------------------------------------------------------------ #
    # Next diagnostic sweep: isolate whether raising the variance floor is
    # enough, and keep KL/blend controls in the same logging/checkpoint regime.
    # ------------------------------------------------------------------ #
    "vlsp_next_floor_m2": _vlsp_variant(
        mode="video_prior_sample",
        conditioning="normal",
        logstd_min=-2.0,
    ),
    "vlsp_next_kl_floor_m2": _vlsp_variant(
        mode="video_prior_sample",
        conditioning="normal",
        logstd_min=-2.0,
        kl_weight=1e-3,
    ),
    "vlsp_next_blend050_kl_floor_m2": _vlsp_variant(
        mode="video_prior_blend",
        conditioning="normal",
        blend_alpha=0.5,
        logstd_min=-2.0,
        kl_weight=1e-3,
    ),
    "vlsp_next_mean_floor_m2": _vlsp_variant(
        mode="video_prior_mean",
        conditioning="normal",
        logstd_min=-2.0,
        kl_weight=1e-3,
    ),
}

if vlsp_base_cfg is not None:
    for vlsp_name, overrides in VLSP_VARIANTS.items():
        vlsp_cfg = copy.deepcopy(vlsp_base_cfg)
        vlsp_cfg["job"]["group"] = "vlsp"
        vlsp_cfg["job"]["name"] = vlsp_name
        vlsp_cfg["model"]["config"]["pipe_config"]["action_source_prior"] = overrides["action_source_prior"]
        vlsp_cfg["model"]["config"]["pipe_config"]["action_conditioning"] = overrides["action_conditioning"]
        for extra_key, extra_val in overrides.get("model_config", {}).items():
            vlsp_cfg["model"]["config"][extra_key] = copy.deepcopy(extra_val)
        for extra_key, extra_val in overrides.get("trainer_config", {}).items():
            vlsp_cfg["trainer"][extra_key] = copy.deepcopy(extra_val)
        for extra_key, extra_val in overrides.get("checkpoint_config", {}).items():
            vlsp_cfg["checkpoint"][extra_key] = copy.deepcopy(extra_val)
        cs.store(
            group="experiment",
            package="_global_",
            name=vlsp_name,
            node=vlsp_cfg,
        )

# TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 CUDA_DEVICE_MAX_CONNECTIONS=1 NVTE_FUSED_ATTN=0 torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/config.py -- experiment=...
