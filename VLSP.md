# VLSP: Video-Latent Source Prior for Action Flow

## 1. Method summary

mimic-video generates actions with a flow-matching action decoder. The partially
denoised video-model latent is used as a **cross-attention condition** for the
action DiT. The action flow is integrated from a Gaussian **source** endpoint:

```python
source   = torch.randn_like(action)        # N(0, I)
x_t      = (1 - t) * action + t * source
target_v = source - action
pred_v   = action_dit(x_t, t, cond=video_latents, obs, ...)
```

**VLSP** keeps the video latent as a condition *and additionally* uses it to
parameterize the **source** of the flow:

```python
video_latents      = partial_video_world_model(...)
source, metrics    = source_prior(video_latents, obs, context_timestep, ...)
x_t                = (1 - t) * action + t * source
target_v           = source - action
pred_v             = action_dit(x_t, t, cond=<normal | zero | shuffled>)
```

> **Existing mimic-video uses video latents as a *condition*.
> VLSP uses video latents to *initialize* the action generative process.**

The source-prior input and the action-decoder condition are **independently
configurable**, which is what enables the source-only / shuffled / blended
ablations below.

When `action_source_prior.enabled = False` the code path is **bit-identical** to
the original implementation (verified for both the training epsilon draw and the
inference `arch_invariant_rand(seed=...)` draw).

## 2. Equations

```
h_v        = W_video(o, l, sigma_v)                         # partially denoised video latent
q_phi(s|.) = N( mu_phi(h_v, o, sigma_v), diag(sigma_phi(h_v, ...)^2) )

x_t        = (1 - t) * a + t * s
u_t        = s - a
L_flow     = || v_theta(x_t, t, c) - u_t ||^2
L          = L_flow + lambda_KL * KL( q_phi || N(0, I) )

KL         = 0.5 * ( mu^2 + sigma^2 - 1 - log sigma^2 )     # diagonal Gaussian
```

The source `s` (and therefore `mu`, `sigma`) live in the **normalized action
space**, because the flow loss is computed on normalized actions. The final
prediction is unnormalized by the existing action unnormalizer.

## 3. Source modes

`action_source_prior.mode`:

| mode                    | source `s`                                                    | uses prior net | deterministic |
|-------------------------|--------------------------------------------------------------|:--------------:|:-------------:|
| `gaussian`              | `randn` (baseline; inference uses `arch_invariant_rand`)      | no             | given seed    |
| `video_prior_sample`    | `mu + T * exp(logstd) * eps`                                  | yes            | given seed    |
| `video_prior_mean`      | `mu`                                                          | yes            | yes           |
| `video_prior_residual`  | `eps + residual_scale * mu`                                   | yes            | given seed    |
| `video_prior_blend`     | `alpha * s_video + sqrt(1 - alpha^2) * eps`                   | yes            | given seed    |
| `video_prior_dropout`   | per-sample `mask * s_video + (1 - mask) * eps`               | yes            | given seed    |
| `shuffled_video_prior`  | like `video_prior_sample` with video latents batch-shuffled  | yes            | given seed    |
| `gt_action_noisy_debug` | `x0 + debug_noise_std * randn` (training/debug only)         | no             | no            |

`action_conditioning.mode` (independent of the source):

| mode             | video latent fed to the action DiT                  |
|------------------|-----------------------------------------------------|
| `normal`         | pass-through (baseline)                              |
| `zero_video`     | zeros (source-only experiments)                     |
| `shuffled_video` | shuffled across the batch (negative control)        |
| `dropout_video`  | per-sample random zeroing with prob `dropout_prob`  |

## 4. Config options

`cosmos_predict2/configs/config_world2action.py`:

```python
@attrs.define
class ActionSourcePriorConfig:
    enabled: bool = False               # master switch (False => exact baseline)
    mode: str = "gaussian"

    pool_type: str = "mean"             # mean | attention | perceiver
    hidden_dim: int = 1024
    num_perceiver_latents: int = 8
    num_attention_heads: int = 8
    mlp_depth: int = 2

    logstd_min: float = -5.0
    logstd_max: float = 1.0
    init_logstd: float = -1.0

    residual_scale: float = 1.0         # video_prior_residual
    blend_alpha: float = 1.0            # video_prior_blend
    source_dropout_prob: float = 0.0    # video_prior_dropout
    dropout_granularity: str = "sample" # sample | trajectory | element
    sampling_temperature: float = 1.0   # scales the stochastic term

    detach_video_latents: bool = True
    use_state: bool = True
    use_context_timestep: bool = True
    use_language: bool = False          # off by default (not plumbed to inference)

    kl_weight: float = 0.0
    mean_l2_weight: float = 0.0
    std_reg_weight: float = 0.0

    debug_noise_std: float = 0.05

@attrs.define
class ActionConditioningConfig:
    mode: str = "normal"                # normal | zero_video | shuffled_video | dropout_video
    dropout_prob: float = 0.0
```

Both are fields of `World2ActionPipelineConfig` and are overridable per
experiment / on the CLI at
`model.config.pipe_config.action_source_prior.*` and
`model.config.pipe_config.action_conditioning.*`.

### Pooling architectures (`pool_type`)

- `mean` — masked/ordinary mean over the video token dimension.
- `attention` — a single learned query attends to the video tokens.
- `perceiver` — a small learned latent array cross-attends to the video tokens, then is pooled.

The prior is **horizon-aware**: a learned per-step query is added to the pooled
global feature, so `mu`/`logstd` differ across the `HA` action steps rather than
broadcasting a single global vector.

## 5. Experiment matrix

Registered named experiments (`cosmos_predict2/configs/experiment/world2action.py`):

| experiment                       | source mode             | conditioning      |
|----------------------------------|-------------------------|-------------------|
| `vlsp_baseline_gaussian`         | gaussian (disabled)     | normal            |
| `baseline_gaussian`              | gaussian (disabled)     | normal            |
| `vlsp_source_only_sample`        | video_prior_sample      | zero_video        |
| `vlsp_source_condition_sample`   | video_prior_sample      | normal            |
| `vlsp_source_only_mean`          | video_prior_mean        | zero_video        |
| `vlsp_source_condition_mean`     | video_prior_mean        | normal            |
| `vlsp_blend_alpha_025/050/075`   | video_prior_blend       | normal            |
| `vlsp_shuffled_source`           | shuffled_video_prior    | normal            |
| `vlsp_shuffled_condition`        | video_prior_sample      | shuffled_video    |
| `vlsp_dropout_020`               | video_prior_dropout     | normal            |
| `vlsp_residual`                  | video_prior_residual    | normal            |
| `vlsp_debug_gt_action_noisy`     | gt_action_noisy_debug   | normal            |

These are built on top of a representative base experiment (a libero base when
available). Sweeps over `blend_alpha`, `residual_scale`, `source_dropout_prob`
and `sampling_temperature` can be done by overriding the corresponding field on
the CLI (below).

Full ablation table the code supports without further edits:

```
A. Baseline                source=gaussian               cond=normal
B. Source-only stochastic  source=video_prior_sample     cond=zero_video
C. Source + condition      source=video_prior_sample     cond=normal
D. Source-only determ.     source=video_prior_mean       cond=zero_video
E. Source + cond determ.   source=video_prior_mean       cond=normal
F. Shuffled source ctrl    source=shuffled_video_prior   cond=normal
G. Shuffled cond ctrl      source=video_prior_sample     cond=shuffled_video
H. Blend                   source=video_prior_blend      blend_alpha in {0.25,0.5,0.75,1.0}
I. Residual                source=video_prior_residual   residual_scale in {0.25,0.5,1.0,2.0}
J. Dropout                 source=video_prior_dropout    source_dropout_prob in {0.1,0.2,0.5}
K. Temperature             source=video_prior_sample     sampling_temperature in {0.0,0.5,1.0,1.5}
```

## 6. Example commands

Training uses `scripts.train`. Replace `<base>` with a concrete base experiment
name (e.g. `w2a_libero_goal_agentview_..._bsz1`) when overriding ad-hoc.

**Gaussian baseline (unchanged behaviour):**

```bash
torchrun --nproc_per_node=4 -m scripts.train \
  --config=cosmos_predict2/configs/config.py -- \
  experiment=vlsp_baseline_gaussian
```

**VLSP source-only (source uses the video latent, decoder gets zeroed video):**

```bash
torchrun --nproc_per_node=4 -m scripts.train \
  --config=cosmos_predict2/configs/config.py -- \
  experiment=vlsp_source_only_sample
```

**VLSP source + condition:**

```bash
torchrun --nproc_per_node=4 -m scripts.train \
  --config=cosmos_predict2/configs/config.py -- \
  experiment=vlsp_source_condition_sample
```

**Shuffled-source negative control:**

```bash
torchrun --nproc_per_node=4 -m scripts.train \
  --config=cosmos_predict2/configs/config.py -- \
  experiment=vlsp_shuffled_source
```

**Blend-alpha sweep (ad-hoc overrides on any base experiment):**

```bash
for ALPHA in 0.25 0.5 0.75 1.0; do
  torchrun --nproc_per_node=4 -m scripts.train \
    --config=cosmos_predict2/configs/config.py -- \
    experiment=<base> \
    model.config.pipe_config.action_source_prior.enabled=true \
    model.config.pipe_config.action_source_prior.mode=video_prior_blend \
    model.config.pipe_config.action_source_prior.blend_alpha=$ALPHA \
    model.config.pipe_config.action_conditioning.mode=normal
done
```

**Temperature sweep / KL regularizer (ad-hoc):**

```bash
torchrun --nproc_per_node=4 -m scripts.train \
  --config=cosmos_predict2/configs/config.py -- \
  experiment=vlsp_source_condition_sample \
  model.config.pipe_config.action_source_prior.sampling_temperature=0.5 \
  model.config.pipe_config.action_source_prior.kl_weight=1e-4 \
  model.config.pipe_config.action_source_prior.pool_type=perceiver
```

**Evaluation** selects the *same experiment name* so the pipeline is built with
the matching VLSP config; the trained source-prior weights are loaded from the
action-model checkpoint automatically (`World2ActionPipeline.from_config`):

```bash
python -m eval.libero.run \
  --vam-experiment-name vlsp_source_condition_sample \
  --vam-video-model-path <video.pt> \
  --vam-action-model-path <action_vlsp.pt> \
  ...
```

**Smoke test (no GPU / dataset required):**

```bash
cd model && python -m scripts.debug_vlsp
```

## 7. Logging

When the source prior is enabled the following are logged (per training step):

```
loss/flow, loss/source_prior, loss/source_kl, loss/source_mean_l2, loss/source_std_reg
source/mu_mean, source/mu_std, source/logstd_mean, source/std_mean
source/source_mean, source/source_std
source/source_vs_x0_mse, source/source_vs_gaussian_mse
source/dropout_rate_actual, source/source_mode_id, source/shuffle_enabled
condition/mode_id, condition/shuffle_enabled
```

## 8. Checkpoint compatibility

- The source prior is saved under `source_prior.*` (and `source_prior_ema.*`
  when EMA is enabled) in the model checkpoint, alongside the existing `net.*` /
  `net_ema.*` keys.
- Loading is **non-strict** for the source prior, so:
  - **Old checkpoints without source-prior keys load fine** — a freshly
    initialized source prior is kept (and for `mode="gaussian"` there are no
    source-prior parameters at all).
  - **A brand-new source prior can be seeded from an old action-decoder
    checkpoint** (the DiT loads strictly/non-strictly as before; the source prior
    just starts from its initialization).
- Both the training `Model.state_dict()/load_state_dict()` path and the
  inference `World2ActionPipeline.from_config()` path load source-prior weights,
  so eval picks up trained priors instead of leaving them random.

## 9. Optimizer / parallelism notes & limitations

- The source prior is **always fully trainable** (even under
  `train_architecture="lora"`, where the DiT only trains LoRA params). Its
  parameters are added to the optimizer and to `clip_grad_norm_`. In LoRA mode
  the source-prior params are upcast to fp32 to match the LoRA param group dtype.
- **DDP (default):** the whole model is DDP-wrapped, so source-prior gradients
  are synchronized. To keep `find_unused_parameters=False` valid, both prior
  heads (`mu`, `logstd`) are always kept in the autograd graph for every mode
  (the unused term is numerically zero), so deterministic modes (`mean`,
  `residual`) and all-dropout batches do not trip DDP.
- **EMA:** a simple param-wise EMA of the source prior is maintained for the
  standard non-FSDP path (`source_prior_ema`), updated with the DiT EMA beta.
- **FSDP limitation:** the source prior is left **replicated** (not sharded); it
  is small. Under an FSDP-only run (no DDP wrapper), gradients for replicated
  parameters are not automatically all-reduced across data-parallel ranks, and
  FSDP+EMA for the source prior is not specially handled. The default trainer
  (`distributed_parallelism="ddp"`, `fsdp_shard_size=0`) is unaffected. If you
  enable FSDP, add explicit gradient synchronization for the source prior or
  wrap it in its own FSDP unit.

## 10. Related work & design rationale

VLSP's core idea — replacing the Gaussian flow source with an informative,
video-conditioned source — is shared by two recent action flow-matching works.
Reading their code directly shaped several VLSP design decisions.

### A2A — Action-to-Action Flow Matching (RoboVerse)
- Flows from a **history encoding** (past states/actions → latent, the source)
  to **future-action latents** (target), with visual obs as a *separate*
  `global_cond`.
- Configurable-source interface (`flow_matchers.py`):
  `x0 = randn_like(target) if start is None else start` — the source is just an
  optional non-Gaussian `start`.
- *A2A-Noise* injects Gaussian noise into the history **states** before encoding,
  as a robustness regularizer.
- Anti-degeneracy: action-reconstruction + consistency + InfoNCE contrastive losses.

### VITA — Vision-to-Action Flow Matching Policy (ICLR 2026, arXiv:2507.13231)
- **"noise-free, conditioning-free"**: the flow `start` is the **vision latent
  itself** (`start = obs_encoder(vision)`, a single `nn.Linear` to `latent_dim`);
  the target is the **action-AE latent** (same dim); and the flow net
  `model(x_t, t)` takes **no condition**. The flowed latent is decoded back to
  actions at the end.
- Both endpoints live in an **aligned latent space** (vision ↔ action manifolds),
  so a short MLP flow suffices — *"conventional flow matching may struggle to
  transport from an unstructured Gaussian to a structured action space."*
- Anti-degeneracy: action reconstruction (encoder + flow), InfoNCE contrastive
  (vision↔action), consistency, optional action-VAE KL.

### How VLSP relates / differs

| | A2A | VITA | VLSP (this work) |
|---|---|---|---|
| flow space | action-AE latent | aligned vision/action latent | **raw normalized action space** (`x0` = GT action) |
| source | history encoding | vision latent (`Linear`) | **video latent → prior net** (mean/attention/perceiver pooling, optional diagonal Gaussian) |
| video as condition | yes (`global_cond`) | no (conditioning-free) | **configurable** (`action_conditioning`: normal / zero_video / …) |
| anti-collapse | recon + contrastive + consistency | recon + contrastive + consistency + KL | optional KL / mean-L2 / std (default 0) |
| action autoencoder | yes | yes | **no** (flow directly in action space, like mimic-video) |

**Design notes / consequences**

1. **A learned source map is unavoidable.** The video latent `[B, N, 2048]` and the
   action source `[B, HA, A]` differ in both shape and space, while
   `x_t = (1-t)·x0 + t·source` requires the source to match `x0`. VITA's
   `obs_encoder` (a `Linear`) and A2A's history encoder are the same unavoidable
   bridge — VLSP's `VideoLatentSourcePrior` is our analog. The minimal,
   VITA-like form is `source_mode=video_prior_mean` + `pool_type=mean`
   (deterministic projection, no sampling).
2. **VITA-style recipe in VLSP.** `source_mode=video_prior_mean` +
   `action_conditioning=zero_video` reproduces VITA's *source-only,
   conditioning-free* setup (experiment row D), here in raw action space.
3. **Source collapse.** Because our source is trainable and `x0` is the fixed GT
   action, the flow loss admits a degenerate optimum `source → x0 ⇒ u_t → 0` that
   bypasses the DiT. A2A/VITA prevent this with reconstruction / contrastive /
   consistency anchoring (and VITA deliberately *aligns* source≈target while
   keeping both decodable). VLSP currently offers KL / mean-L2 / std regularizers
   (default 0). **Recommendation**: for `video_prior_*` modes set a small
   `kl_weight` (e.g. `1e-3`) and watch `source/source_vs_x0_mse`; an optional
   VITA-style consistency anchor is a natural extension.
4. **Endpoint convention.** A2A/VITA (torchcfm) put the source at `t=0` and data
   at `t=1`; mimic-video puts data at `t=0` and the source at `t=1` (the sampler
   integrates `t: 1→0`). VLSP places the learned source at mimic-video's `t=1`
   endpoint — exactly where `torch.randn_like(action)` used to be — so the
   existing sampler/integration is unchanged.

### References
- **VITA: Vision-to-Action Flow Matching Policy**, ICLR 2026 — arXiv:[2507.13231](https://arxiv.org/abs/2507.13231) · code: https://github.com/ucd-dare/VITA (`flare/policies/vita/vita_policy.py`, `flare/flow/flow_matchers.py`)
- **A2A — Action-to-Action Flow Matching** (RoboVerse): https://github.com/JIAjindou/A2A_Flow_Matching (`roboverse_learn/il/policies/a2a/`)
- **MeanFlow** (1-step generative modeling) arXiv:[2505.13447](https://arxiv.org/abs/2505.13447); **Improved Mean Flows** arXiv:2512.02012 — flow-matcher options used by A2A/VITA.

## 11. Experiment protocol & order (go/no-go)

Run experiments **cheapest-to-validate / most-decisive first**, with a go/no-go
gate between phases. The letters (A–K) refer to the matrix in §5; the names are
the registered experiments from §6.

### Phase 0 — Plumbing / sanity (cheap, do first, don't skip)
1. `cd model && python -m scripts.debug_vlsp` (the CPU smoke test).
2. Short runs (a few hundred–1k steps, no convergence needed):
   - `vlsp_baseline_gaussian` — confirm the baseline trains **and matches the
     original mimic-video numbers** (regression check: VLSP disabled must be
     bit-identical to upstream).
   - `vlsp_source_condition_sample` — confirm the source-prior path trains: loss
     decreases, all `source/*` + `loss/*` metrics log, checkpoint save/load
     round-trips, no DDP unused-parameter error.
   - `vlsp_debug_gt_action_noisy` — near-oracle source; confirm the decoder can
     overfit (sanity that the pipeline itself is healthy).
- **Gate:** all three short runs healthy → Phase 1. Otherwise fix plumbing first.

### Phase 1 — Baseline (the anchor number)
3. Full train + eval `vlsp_baseline_gaussian` (LIBERO/Bridge). **Lock the
   seed / dataset / compute / eval protocol here**; every later run reuses it.

### Phase 2 — Core hypothesis: does the video-latent source help? (min. 2 runs)
4. `vlsp_source_condition_sample` (**C** = video source + normal condition).
   - **C vs A**: only the source differs (both keep the video condition) ⇒ the
     cleanest "does the video-latent source add value" comparison. **Primary result.**
5. `vlsp_source_only_sample` (**B** = video source + `zero_video`).
   - **B vs A**: can the source *replace* the cross-attention condition (stronger
     claim). **B vs C**: how much the condition still contributes given the source.
- **Gate:** if C shows no meaningful gain over A, do **not** start sweeps — go to
  Phase 4 (kl / determinism) to understand why (likely collapse or a too-weak prior).

### Phase 3 — Negative controls (prove it's signal, not capacity) — run as soon as C looks positive
6. `vlsp_shuffled_source` (**F** = source from batch-shuffled video).
   - If **F ≈ C**, the gain is just extra params / a smaller-variance source / a
     loss-scale artifact — **not** video information. **Stop and rethink.**
7. `vlsp_shuffled_condition` (**G**) — controls the conditioning pathway.
- **Gate:** controls **must** degrade vs C/A. If they don't, do not proceed to sweeps.

### Phase 4 — Mechanism ablations + the collapse dial
8. **Deterministic vs stochastic**: `vlsp_source_only_mean` / `vlsp_source_condition_mean`
   (**D/E**) vs B/C — does the reparameterized noise matter? (`mean` ≈ VITA-style.)
9. **kl dial**: sweep `kl_weight ∈ {0, 1e-3, 1e-2}` on the best config while watching
   `source/source_vs_x0_mse`. This characterizes whether the source is a *hint*
   (mse stays up; DiT does the work) or *the answer* (mse → 0; DiT bypassed).

### Phase 5 — Sweeps (lowest priority; only after Phases 2–3 are positive)
10. On the winning config: blend `α∈{.25,.5,.75}` (**H**), dropout `p∈{.1,.2,.5}`
    (**J**), residual scale (**I**), temperature (**K**), pooling
    (`mean|attention|perceiver`). These are tuning, not validation of the claim.

### Minimal decisive set (if compute is tight): **A · C · B · F**
`vlsp_baseline_gaussian` · `vlsp_source_condition_sample` · `vlsp_source_only_sample`
· `vlsp_shuffled_source`. These four answer the core question: *does the
video-latent source help, and is it real (video) information?* Everything else is
a bonus.

### Discipline (applies to every run)
- **Same seed / dataset / compute / eval protocol**; vary only the axis under test.
- **Do NOT compare runs on the raw flow loss** (`loss/flow`). The baseline regresses
  `noise − x0` while VLSP regresses `source − x0`, so their loss magnitudes are not
  comparable — a smaller VLSP loss can be pure scale, not better learning. Compare on
  **scale-invariant, source-independent metrics**: the unnormalized action prediction
  MSE after full sampling (validation `gtvid/full`, `genvid/full`) and/or eval success
  rate. (E.g. an early "10× faster" must be confirmed on these, not on the loss.)
- Always check **`genvid/*`** (generated-video conditioned), not only `gtvid/*`
  (GT-video) — generated video is what inference actually uses, so this is where the
  train/inference gap shows up.
- Standing monitors: `success_rate`, validation action MSE, `source/source_vs_x0_mse`,
  `source/source_vs_gaussian_mse`, `source/std_mean`, `loss/source_kl`.
- Honor the **go/no-go gates** — if a phase is unhealthy, go back; don't pile on runs.
