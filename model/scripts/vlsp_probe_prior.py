# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Offline VLSP diagnostics on a trained checkpoint (CPU is fine, no sim needed).

Run-1 post-mortem tool (see VLSP_RUN1_ANALYSIS.md). Answers, in minutes:

  D1  Are `source_prior.*` weights present in the checkpoint at all?
      (If not, eval silently used a randomly initialized prior.)
  D2  Did the prior degenerate?
      - variance collapse: fraction of logstd at the clamp floor;
      - input independence: does `mu` change when the video latent changes?

Usage:
    # checkpoint key audit + synthetic-input probe (always works, CPU):
    python -m scripts.vlsp_probe_prior --ckpt /path/to/model/iter_000024000.pt

    # additionally probe with REAL video latents (recommended), where
    # latents.pt is a tensor [K, N, D] or a list of [N, D]/[B, N, D] tensors
    # from K different tasks (dump crossattn_emb in eval or a data loop):
    python -m scripts.vlsp_probe_prior --ckpt ... --latents latents.pt

Interpretation:
    logstd_floor_frac ~ 1.0            -> variance collapse CONFIRMED
    cross-input mu diff ~ 0            -> prior IGNORES its input (trained-time
                                          degeneration; eval OOD is moot)
    cross-input mu diff clearly > 0    -> prior is input-sensitive; suspect the
                                          eval-side inputs (generated video /
                                          stop-step sigma) or eval loading.
"""

from __future__ import annotations

import argparse
import re
from typing import Optional

import torch

SP_PREFIX = "source_prior."
SP_NET_PREFIX = "source_prior.net."


def load_flat_state_dict(path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # Model checkpoints are flat dicts (net./net_ema./source_prior.*); some
    # tooling wraps them under "model".
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise ValueError(f"Unrecognized checkpoint structure in {path}")
    return {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}


def audit_keys(sd: dict[str, torch.Tensor]) -> dict[str, int]:
    counts = {
        "net.*": sum(k.startswith("net.") for k in sd),
        "net_ema.*": sum(k.startswith("net_ema.") for k in sd),
        "source_prior.*": sum(k.startswith(SP_PREFIX) for k in sd),
        "source_prior_ema.*": sum(k.startswith("source_prior_ema.") for k in sd),
    }
    print("== D1: checkpoint key audit ==")
    for name, n in counts.items():
        print(f"  {name:<22} {n}")
    if counts["source_prior.*"] == 0:
        print(
            "  !! NO source_prior.* keys -> eval built a RANDOMLY INITIALIZED prior\n"
            "     (mu==0 exactly at init) -> constant near-mean actions. Fix the\n"
            "     checkpoint/save path before blaming the training run."
        )
    else:
        sample = [k for k in sd if k.startswith(SP_PREFIX)][:4]
        print(f"  sample keys: {sample}")
    return counts


def infer_arch(sp: dict[str, torch.Tensor]) -> dict:
    """Infer VideoLatentSourcePrior hyperparameters from checkpoint shapes."""
    arch: dict = {}
    arch["video_emb_dim"] = sp["ctx_norm.weight"].shape[0]
    arch["hidden_dim"] = sp["mu_head.weight"].shape[1]
    arch["action_dim"] = sp["mu_head.weight"].shape[0]
    arch["max_horizon"] = sp["horizon_queries"].shape[1]
    if "pool.latents" in sp:
        arch["pool_type"] = "perceiver"
        arch["num_perceiver_latents"] = sp["pool.latents"].shape[1]
    elif "pool.query" in sp:
        arch["pool_type"] = "attention"
    else:
        arch["pool_type"] = "mean"
    arch["use_state"] = any(k.startswith("state_proj.") for k in sp)
    if arch["use_state"]:
        arch["state_dim"] = sp["state_proj.weight"].shape[1]
    arch["use_context_timestep"] = any(k.startswith("time_proj.") for k in sp)
    arch["use_language"] = any(k.startswith("lang_proj.") for k in sp)
    trunk_linears = {
        int(m.group(1))
        for k, v in sp.items()
        if (m := re.match(r"trunk\.net\.(\d+)\.weight", k)) and v.dim() == 2  # Linear only (LayerNorm is 1-D)
    }
    arch["mlp_depth"] = max(1, len(trunk_linears))
    print("== inferred prior architecture ==")
    for k, v in sorted(arch.items()):
        print(f"  {k:<22} {v}")
    return arch


def build_prior_net(arch: dict, num_attention_heads: int):
    from types import SimpleNamespace

    from cosmos_predict2.models.action_source_prior import VideoLatentSourcePrior

    cfg = SimpleNamespace(
        enabled=True,
        mode="video_prior_sample",
        pool_type=arch["pool_type"],
        hidden_dim=arch["hidden_dim"],
        num_perceiver_latents=arch.get("num_perceiver_latents", 8),
        num_attention_heads=num_attention_heads,
        mlp_depth=max(1, arch["mlp_depth"]),
        logstd_min=-5.0,
        logstd_max=1.0,
        init_logstd=-1.0,
        residual_scale=1.0,
        blend_alpha=1.0,
        source_dropout_prob=0.0,
        dropout_granularity="sample",
        sampling_temperature=1.0,
        detach_video_latents=True,
        use_state=arch["use_state"],
        use_context_timestep=arch["use_context_timestep"],
        use_language=arch["use_language"],
        kl_weight=0.0,
        mean_l2_weight=0.0,
        std_reg_weight=0.0,
        debug_noise_std=0.05,
    )
    net = VideoLatentSourcePrior(
        cfg,
        action_dim=arch["action_dim"],
        video_emb_dim=arch["video_emb_dim"],
        state_dim=arch.get("state_dim", arch["action_dim"]),
        max_horizon=arch["max_horizon"],
    )
    return net


@torch.no_grad()
def probe(net, arch: dict, latents_groups: list[torch.Tensor], horizon: int, sigma: float) -> None:
    """Feed K latent groups through the prior; report input sensitivity + collapse."""
    net.eval()
    mus, logstds = [], []
    for group in latents_groups:
        x = group.float()
        if x.dim() == 2:  # [N, D] -> [1, N, D]
            x = x.unsqueeze(0)
        ctx = torch.full((x.shape[0], 1), float(sigma))
        mu, logstd = net(x, None, ctx, horizon, None)
        mus.append(mu.float())
        logstds.append(logstd.float())

    K = len(mus)
    mu_stack = torch.stack([m.mean(dim=0) for m in mus])  # [K, HA, A] per-group mean
    action_scale = mu_stack.abs().mean().clamp_min(1e-6)

    pair_diffs = []
    for i in range(K):
        for j in range(i + 1, K):
            pair_diffs.append((mu_stack[i] - mu_stack[j]).abs().mean())
    cross_input_diff = torch.stack(pair_diffs).mean().item() if pair_diffs else float("nan")

    logstd_all = torch.cat([ls.reshape(-1) for ls in logstds])
    floor_frac = (logstd_all <= -5.0 + 0.05).float().mean().item()

    print("== D2: prior degeneration probe ==")
    print(f"  groups (K)                    : {K}")
    print(f"  |mu| scale                    : {action_scale.item():.4f}")
    print(f"  cross-input mu |diff| (mean)  : {cross_input_diff:.6f}")
    print(f"  cross-input / |mu| ratio      : {cross_input_diff / action_scale.item():.4f}")
    print(f"  logstd mean                   : {logstd_all.mean().item():.4f}")
    print(f"  logstd floor fraction (<=-4.95): {floor_frac:.4f}")

    print("== verdicts ==")
    if floor_frac > 0.9:
        print("  [X] VARIANCE COLLAPSE confirmed (logstd pinned at the clamp floor).")
    elif floor_frac > 0.3:
        print("  [!] Partial variance collapse (a large share of logstd at the floor).")
    else:
        print("  [ok] No variance collapse in logstd.")
    ratio = cross_input_diff / action_scale.item()
    if ratio < 0.02:
        print(
            "  [X] INPUT-INDEPENDENT PRIOR: mu is (near-)identical across different\n"
            "      video latents -> the prior degenerated during TRAINING; eval-side\n"
            "      OOD is not the primary cause."
        )
    elif ratio < 0.15:
        print("  [!] Weak input sensitivity — inspect with real latents before concluding.")
    else:
        print(
            "  [ok] mu clearly varies with the input. If eval still produces a fixed\n"
            "      action, suspect the EVAL inputs (generated-video hidden states /\n"
            "      stop-step sigma) or eval-side loading, not the prior itself."
        )


def load_latent_groups(path: Optional[str], arch: dict, num_synthetic: int, tokens: int) -> list[torch.Tensor]:
    if path:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, torch.Tensor):
            groups = [obj[i] for i in range(obj.shape[0])]
        elif isinstance(obj, (list, tuple)):
            groups = [torch.as_tensor(t) for t in obj]
        else:
            raise ValueError("--latents must be a tensor [K,N,D] or a list of tensors")
        print(f"loaded {len(groups)} REAL latent groups from {path}")
        return groups
    g = torch.Generator().manual_seed(0)
    groups = [torch.randn(1, tokens, arch["video_emb_dim"], generator=g) for _ in range(num_synthetic)]
    print(
        f"using {num_synthetic} SYNTHETIC latent groups (no --latents given). "
        "Constant mu across these is conclusive for input-independence; variation is not\n"
        "conclusive for eval health — re-run with real latents for the full answer."
    )
    return groups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="model checkpoint (.pt with net./source_prior.* keys)")
    ap.add_argument("--latents", default=None, help="optional .pt with real crossattn latents [K,N,D]")
    ap.add_argument("--horizon", type=int, default=None, help="action horizon HA (default: max_horizon-1... uses 15 if unknown)")
    ap.add_argument("--sigma", type=float, default=8.0, help="context timestep (video sigma) fed to the prior")
    ap.add_argument("--num-synthetic", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=1225, help="synthetic token count (7x… matches T*H*W)")
    ap.add_argument("--heads", type=int, default=8, help="attention heads (attention/perceiver pools; not inferable)")
    args = ap.parse_args()

    sd = load_flat_state_dict(args.ckpt)
    counts = audit_keys(sd)
    if counts["source_prior.*"] == 0:
        return

    sp_net = {k[len(SP_NET_PREFIX) :]: v for k, v in sd.items() if k.startswith(SP_NET_PREFIX)}
    if not sp_net:
        print("!! source_prior.* keys exist but no source_prior.net.* — disabled/gaussian prior; nothing to probe.")
        return
    arch = infer_arch(sp_net)

    net = build_prior_net(arch, num_attention_heads=args.heads)
    missing, unexpected = net.load_state_dict(sp_net, strict=False)
    print(f"loaded prior net: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  missing (first 5): {missing[:5]}")

    horizon = args.horizon if args.horizon is not None else min(15, arch["max_horizon"])
    groups = load_latent_groups(args.latents, arch, args.num_synthetic, args.tokens)
    probe(net, arch, groups, horizon=horizon, sigma=args.sigma)


if __name__ == "__main__":
    main()
