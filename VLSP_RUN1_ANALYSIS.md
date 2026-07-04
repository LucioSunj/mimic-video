# VLSP Run 1 — Failure Analysis & Next Steps
# (第一次实验分析 + 下一步步骤)

**Status:** post-mortem of the first real training runs; defines the mandatory
diagnostics and the Run-2 plan (R1–R3). Referenced from `VLSP.md` §12.

- Runs analyzed: `vlsp_source_condition_goal_20260629_183112` (24k iters),
  `vlsp_source_only_goal_20260701_125625` (26k iters) — LIBERO-goal,
  `mode=video_prior_sample`, **no regularization** (`kl_weight=0`).
- Data: `source_cond_vs_source_only_metrics_20260704/` (train scalars, stdout,
  device monitor, plots.html) + rollout videos
  (e.g. `episode20_failure_open_the_middle_drawer_of_the_cabinet.mp4`).

## 1. Symptom

Every eval rollout executes the **same fixed action sequence, unrelated to the
task**, for both source-only (B) and source+cond (C).

## 2. Evidence (from training scalars)

| metric | source_cond @18.2k–24k | source_only @0.2k–26.3k | reading |
|---|---|---|---|
| `source/logstd_mean` | **-5.0 pinned** (== `logstd_min`) | -3.44 → **-5.0** | **variance collapse to the clamp floor**; σ = e⁻⁵ ≈ 0.0067 (`std_mean` = 0.00674 matches exactly) |
| `source/source_vs_x0_mse` | 0.79 → **0.91 (rising)** | 0.65 → 0.81 | mu did **NOT** collapse onto x0; it is *worse than predicting the dataset mean* (Var[x0] ≈ 0.55) |
| `loss/flow` | 0.006–0.03 | 5.48 → 0.016 | tiny — because the target `u = mu(video) − x0` became **deterministic/predictable**, not because the model learned |
| `source/mu_std` (global) | ≈ 0.61 | 0.17 → 0.63 | **inconclusive by design**: global std over `[B,HA,A]` cannot distinguish "mu varies with the video" from "one fixed trajectory for every video" (metric blind spot, fixed in Run 2) |

The earlier *"10× faster convergence in the first 1k iters"* was this collapse
in progress — the flow target got easier; nothing was learned faster. This is
exactly why §11 forbids comparing runs on the raw flow loss.

## 3. Root-cause chain (working hypothesis)

1. **Variance collapse (confirmed).** With `kl_weight=0`, source noise adds
   irreducible variance to the regression target `u = s − x0`; gradient descent
   therefore drives `logstd` to the clamp floor. The source becomes a per-input
   **Dirac**; flow matching degenerates into deterministic pair regression and
   loses the Gaussian source's "noise-dominated ⇒ always in-distribution at
   t≈1" robustness.
2. **Input-independence of the prior (suspected, now testable).** In
   source+cond there is **no loss pressure for mu to carry video information**
   — the DiT sees the video via cross-attention anyway; the loss only needs
   `u = mu − x0` to be *predictable*, and a **constant mu is maximally
   predictable**. `source_vs_x0_mse > Var[x0]` (an arbitrary offset trajectory,
   not even the mean) is consistent with this. A constant-mu Dirac source fully
   explains "fixed action, task-independent" for source-only.
3. **Eval-side second leg (to be separated by diagnostics).** For source+cond
   the DiT's condition pathway should still inject task info *if* the velocity
   field and eval inputs are healthy — so either the prior is constant (2.) AND
   the field off-manifold behaviour dominates, or the eval inputs
   (generated-video hidden states at the chosen `stop_video_denoising_step` /
   checkpoint-loading) are also degenerate. The diagnostics below separate
   these without retraining.

## 4. Three cheap diagnostics (run BEFORE any retraining)

**D1 — checkpoint / eval-loading audit (minutes, CPU).**
```bash
cd mimic-video/model
python -m scripts.vlsp_probe_prior --ckpt /path/to/checkpoints/.../model/iter_000024000.pt
```
The `D1` section confirms `source_prior.*` keys exist (if absent, eval ran a
randomly initialized prior — mu≡0 → constant near-mean actions — and the
training run is not even implicated). Also grep the eval log for
`Loaded source_prior from checkpoint` vs `randomly initialized`.

**D2 — prior degeneration probe (same command, no sim).**
The same script loads the trained prior net (architecture auto-inferred from
checkpoint shapes) and feeds K different video latents:
- `logstd floor fraction ≈ 1.0` → variance collapse confirmed (expected);
- `cross-input mu diff ≈ 0` → **prior ignores its input** → training-time
  degeneration confirmed; eval OOD is moot;
- mu clearly input-sensitive → suspect eval inputs / loading instead.
Synthetic latents are conclusive only for the "ignores input" verdict; for the
full answer re-run with real latents:
```bash
python -m scripts.vlsp_probe_prior --ckpt ... --latents real_latents.pt   # [K, N, D]
```
(dump `crossattn_emb` for a few different tasks from the eval pipeline or a
data-loader loop into `real_latents.pt`).

**D3 — eval-condition sweep with the EXISTING checkpoint (no retrain).**
```bash
# later stop step = cleaner video latent; does the action become task-dependent?
for STOP in 5 15 25 35; do
  python -m eval.libero.run --vam-stop-video-denoising-step $STOP ... 
done
```
If actions become task-dependent at late stop steps → the eval-time sigma /
generated-video gap is a major factor; if they stay fixed regardless → the
Dirac collapse alone is fatal. (Also record which stop step Run 1 actually
used.)

## 5. Run 2 plan: R1–R3 (all source+cond, same seed/data/eval as Run 1)

| Run | registered experiment | config delta | rationale |
|---|---|---|---|
| **R1** (primary) | `vlsp_r1_kl_1e3` | `kl_weight=1e-3` | KL pins q(s\|video) near N(0,1): kills the σ→0 gradient incentive; source stays a "hint" |
| **R2** (structural fallback) | `vlsp_r2_blend_050` | `video_prior_blend`, `blend_alpha=0.5`, `kl_weight=1e-3` | source = α·s_video + √(1−α²)·ε keeps an **un-collapsible 0.87-std Gaussian component**; worst-case degeneration = baseline, not a broken model — also covers the "mu carries no info" worst case |
| **R3** (robustness) | `vlsp_r3_kl_dropout_020` | R1 + `source_dropout_prob=0.2` | DiT sees pure-Gaussian sources 20% of the time → keeps a noise-robust mode |

All three enable the new **sampled-MSE training probe**
(`model.config.sampled_mse_probe_interval=500`), which logs
`probe/sampled_action_mse_gtvid` — a scale-invariant, source-independent metric
(LIBERO has `run_validation=False`, so Run 1 was blind between flow-loss and
sim eval).

**In-training gates (check at ~2k, 5k, 10k iters):**
- `source/logstd_mean` must stay > −2.5 (a `[VLSP]`-tagged warning fires
  otherwise, every 1k iters);
- `source/logstd_floor_frac` (new) must stay ≈ 0;
- `source/mu_batch_std` (new, cross-batch) must stay clearly > 0 — this is the
  metric that would have exposed Run 1's input-independent prior;
- `probe/sampled_action_mse_gtvid` must trend **down** and beat/track the
  baseline's value at the same iteration.

**Go/no-go:**
- R1 healthy → it becomes the main line; proceed to §11 Phase-3 negative
  controls (shuffled source) before any sweeps.
- R1 collapses again (logstd → floor despite KL) → raise `kl_weight` to `1e-2`
  or adopt R2 as the main line; if only R2 is healthy, the conclusion is "this
  backbone needs an explicit Gaussian component in the source".
- Everything unhealthy on the probe while D2 said the prior is fine → the
  problem is the eval/conditioning distribution, not the source: revisit D3
  (stop-step) before touching the prior again.
- `source_only` (B) and all sweeps stay **on hold** until a source+cond config
  passes these gates.

**Launch (same base as Run 1, e.g. goal suite):**
```bash
torchrun --nproc_per_node=4 -m scripts.train \
  --config=cosmos_predict2/configs/config.py -- experiment=vlsp_r1_kl_1e3
# likewise: experiment=vlsp_r2_blend_050 | experiment=vlsp_r3_kl_dropout_020
# equivalent ad-hoc overrides, if using suite-specific experiment names:
#   model.config.pipe_config.action_source_prior.kl_weight=1e-3 \
#   model.config.sampled_mse_probe_interval=500
```

## 6. Code added for this plan (all in this repo)

- `model/cosmos_predict2/models/action_source_prior.py` — new metrics
  `source/mu_batch_std` (cross-batch input-sensitivity; fixes the Run-1 blind
  spot) and `source/logstd_floor_frac` (collapse alarm).
- `model/cosmos_predict2/models/world2action_model.py` — `[VLSP]` collapse
  warning (logstd_mean < −2.5, throttled); `sampled_mse_probe_interval` config
  + `probe/sampled_action_mse_gtvid` training probe (full sampling on the train
  batch, unnormalized MSE, DDP/FSDP-safe collective).
- `model/cosmos_predict2/configs/experiment/world2action.py` — registered
  `vlsp_r1_kl_1e3`, `vlsp_r2_blend_050`, `vlsp_r3_kl_dropout_020` (probe on).
- `model/scripts/vlsp_probe_prior.py` — offline D1/D2 diagnostics (CPU).
