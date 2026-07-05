# VLSP Execution Plan — Tier-0 Diagnostics → Decision Node → Run-2
# (VLSP 执行计划：先审计，再诊断，后决策，最后重训)

**Status:** active plan as of 2026-07-05. This plan **supersedes the ordering**
of `VLSP_RUN1_ANALYSIS.md` §5 (R1–R3 must NOT be launched as-is) and pauses
`VLSP.md` §11 Phases 2–5 until the Phase-2 decision node below is resolved.
The Run-1 diagnostics (D1/D2/D3, `model/scripts/vlsp_probe_prior.py`) are
reused and extended here, not replaced.

**Why this ordering.** Run-1 produced two hard lessons: (a) the flow loss is
not a success signal (collapse makes the target easy), and (b) an eval
artifact already inverted one conclusion once. So every GPU-hour below is
ordered by information-per-hour toward five causal questions:

```
Q0  Are the eval numbers / configs trustworthy at all?            → Phase 0
Q1  Does the video latent actually contain action information?    → E0 (1.1)
Q2  Does Run-1's μ carry action information?                      → D2 (1.2)
Q3  Is C's low success due to a harmful source, a useless source,
    the video gap — or is the baseline equally low?               → A′/ablations (1.3–1.6)
Q4  With σ fixed, how do we make μ actually useful?               → R1′/R5′ (Phase 3)
```

**Frozen conclusions (do not re-litigate during execution):**

- Run-1's low flow loss is NOT evidence of learning.
- `logstd` pinned at the −5 floor is structural collapse, not noise.
- "C runs and is task-conditioned" does NOT show the VLSP source is useful —
  C also has the cross-attn condition.
- B's breakdown is important evidence but not sufficient on its own.
- Unmodified R1–R3 only test "can σ avoid collapse", not "does μ carry
  information".

**Evidence standard (headline metrics; everything else is supporting):**

| accepted as evidence | explicitly NOT evidence |
|---|---|
| batch-centered **R²(μ → x0)** on held-out data (def. in Appendix A) | raw flow loss |
| **true-source vs shuffled-source gap** (success and/or action MSE) | `source/mu_batch_std` > 0 (necessary, not sufficient) |
| **few-step sampling curve** (2/5/10/50 steps) vs baseline | "C produces task-conditioned actions" |
| final **success rate ≥ A′** under the audited, identical eval setup | any number from a non-audited eval dir |

Statistics: ≥ 200 episodes per arm (LIBERO-goal full eval = 10 tasks × 50
trials = 500). Arms are **paired by construction** if they use the same
episode indices: `run.py` uses `env.seed(0)` and
`set_init_state(initial_states[episode_idx])` (eval/libero/run.py:379), so
always run identical episode index sets across arms and compare per-episode.

---

## Phase 0 — Q0: eval / config parity audit  [blocking; CPU + log reading]

Nothing downstream is interpretable until this passes. Deliverable:
**`VLSP_EVAL_PARITY.md`** containing §0.1 verdicts + the §0.2 sheet.
**Detailed implementation plan: [VLSP_PHASE0_AUDIT_PLAN.md](VLSP_PHASE0_AUDIT_PLAN.md)**
(work items A–D, code sketches, per-row data sources, effort estimates,
exit checklist).

### 0.1 Repo-level hazards to check first (found by code inspection, 2026-07-05)

| # | Hazard | Where | Action |
|---|---|---|---|
| **P0-1** | **Rollout-dir collision + resume contamination.** `run_label = <ckpt stem>_stopafter<k>_execute<n>` contains **no experiment name**. Two runs whose checkpoints share a filename (e.g. every training run saves `iter_000024000.pt`) write into the SAME `./results/...` dir, and the resume scan **skips episodes found on disk and imports their success flags** into the current run's totals. | `eval/libero/run.py:319-322` (label), `:366-376` (resume scan) | Fix: include `vam_experiment_name` + seed in `run_label`. Audit ALL historical result dirs for cross-run reuse; any success number from a shared dir is void and must be re-run. This is the leading candidate for the earlier eval artifact. |
| **P0-2** | **Eval uses non-EMA (reg) weights.** `from_config` loads `net.*` into the DiT; `net_ema.*` keys are silently dropped (`strict=False`); the source prior loads `source_prior.*` (reg). | `model/cosmos_predict2/pipelines/world2action.py` (`from_config`) | Record `weights=reg` on every eval row. Confirm the in-training probe `probe/sampled_action_mse_gtvid` also uses reg weights so training and eval probes are comparable. |
| **P0-3** | **The eval condition mode is carried by `--vam_experiment_name`.** `prepare_action_condition` reads `action_conditioning.mode` from the registered experiment config; evaluating B's checkpoint under a non-B experiment name silently disables `zero_video` (and vice versa for C/A′). | `eval/libero/run.py:63`; `pipelines/world2action.py:251-278` | Recover the exact experiment name used in every historical eval. Required pairing: B ↔ `vlsp_source_only_sample`, C ↔ `vlsp_source_condition_sample`, A′ ↔ `vlsp_baseline_gaussian`. Verify train & eval both route through `apply_action_conditioning` (single code path — they do) and that the only `training=True/False` difference is dropout. |
| **P0-4** | **Fixed seed on every policy query.** `VAMInference._query_policy` calls the pipeline without `seed` → `seed=0` per chunk → the Gaussian/source stochastic draw is identical at every replan. | `run.py:173-181` → `video2world2action.py:35` → `world2action.py:338` | Not necessarily wrong, but must be identical across arms; record it. For the jitter/sample arms of the 1.4 grid, thread an incrementing seed (T4). |
| **P0-5** | **`x_t` rescale assumes a unit-variance source.** `denoise()` divides by `sqrt((1−t)²+t²)`; with a collapsed source (σ≈0) `x_t` at t≈1 is mis-scaled relative to training statistics. Concrete stiffness suspect for B's weird kinematics. | `pipelines/world2action.py:304` | Log-only in Phase 0; quantified by the solver ablation (1.5). Do not change yet. |
| **P0-6** | **D1 loading audit.** If `source_prior.*` keys are missing/ignored at eval, the whole behavioral comparison is void. | `from_config` logs; `model/scripts/vlsp_probe_prior.py` (D1) | Run D1 on B and C checkpoints; grep every eval log for `Loaded source_prior from checkpoint (missing=0 …)` vs the `randomly initialized` warning. |

### 0.2 Parity sheet (fill one column per arm; commit to `VLSP_EVAL_PARITY.md`)

| item | B train | B eval | C train | C eval | A′ eval |
|---|---|---|---|---|---|
| checkpoint iteration / file | | | | | |
| weights loaded (reg / EMA) | | | | | |
| normalizer stats json | | | | | |
| source mode (`action_source_prior.mode`) | | | | | |
| condition mode (`action_conditioning.mode`) | | | | | |
| zero_video implementation identical? (P0-3) | | | | | |
| `stop_video_denoising_step` | | | | | |
| video latent source (generated@stop / GT) | | | | | |
| video steps (`num_sampling_steps`, hardcoded 35 — run.py:123) | | | | | |
| action solver steps (`scheduler.num_denoising_steps` from experiment cfg — record value) | | | | | |
| `x_t` rescale present (P0-5) | | | | | |
| obs_dropout / condition dropout at eval (must be 0 / off) | | | | | |
| proprio path into DiT identical? | | | | | |
| seed policy (P0-4) | | | | | |
| task split / episode indices | | | | | |
| results dir (post P0-1 fix, collision-free?) | | | | | |

### 0.3 Exit rule

- **Any mismatch** → fix it → **re-run the B and C evals** → downgrade all
  prior behavioral conclusions to "unverified" until reproduced.
- **No mismatch** → B/C behavioral differences are admissible evidence for
  Phases 1–2.

---

## Phase 1 — no-retrain diagnostics (highest information per GPU-hour)

Execution priority: **E0 > A′ > D2 > source-ablation > solver / D3 >
taxonomy**. E0/D2 (T1–T3) are CPU/1-GPU and must not wait for the sim
machine; launch A′ on the sim GPUs in parallel.

### 1.0 Tooling (build once; keep eval-only code out of the training path)

| ID | Deliverable | Status |
|---|---|---|
| **T1** | `model/scripts/vlsp_dump_pairs.py` — walk the LIBERO train+val dataloaders through the frozen video pipeline **exactly as in training**; dump per sample `{video_tokens [N,2048], x0 [HA,A] normalized, state, context_timestep σ_v, task_id, split}` → `pairs_{split}.pt`. Optional `--from-eval`: same dump but tokens from the generation path at a given `stop_after_step` (reuse the `video2world2action.py:48-68` seam, which already returns `video_sigma`). Output shape doubles as the `--latents` input `vlsp_probe_prior.py` already accepts. | new |
| **T2** | `model/scripts/vlsp_probe_e0.py` — E0-a/b/c/d probes on T1 dumps (§1.1): data-mean baseline, ridge/linear full-token, attention-pool + linear, small perceiver + linear, and an exact replica of the current prior pooling; batch-centered R², normalized MSE, per-horizon, per-dim, per-σ_v-bin, train/val gap. Pure torch. | new |
| **T3** | Extend `model/scripts/vlsp_probe_prior.py` — on real pairs add: R²(μ→x0) and linear-probe(μ)→x0; `μ_vs_data_mean_mse`; per-horizon / per-dim R²; μ(true) vs μ(shuffled) vs μ(fixed video); task-id linear probe; `--ckpts` sweep (2k/5k/10k/final); run for **both** B and C. (σ-collapse + input-sensitivity already implemented.) | extend |
| **T4** | `eval/libero/run.py` additions (eval-only): (a) **run_label fix** (P0-1); (b) `--vam_source_override {learned, shuffled, batch_mean, fixed_mean, gaussian, jitter0.1, jitter0.3, blend0.5}` wrapping `sample_action_source` at `world2action.py:350`; (c) `--vam_action_denoise_steps N` + `--vam_action_solver {euler,heun}` (Heun added to `schedulers/beta_scheduler.py`; Euler unchanged default); (d) `--vam_log_sampler_stats` → per-query `‖v_t‖`, `‖x_t‖`, per-step action delta; (e) per-run `results.json` (per-task successes, episode indices, full config echo incl. overrides) that aggregates the 5 `eval_rank` shards. | new |
| **T5** | *(optional)* `model/scripts/vlsp_offline_action_mse.py` — offline `sampled_action_mse_gtvid` for any ckpt × source-override on held-out data; cheap pre-filter before sim time. | optional |

### 1.1 E0 — oracle probe: is there action information in the latent? (Q1)

Freeze everything; train only small probes `video latent → x0` on T1 dumps.

| variant | input | model | role |
|---|---|---|---|
| E0-a | none | predict `E[x0]` (train mean) | floor: R² ≡ 0 |
| E0-b | full tokens `[N, 2048]` | linear/ridge; attention-pool + linear; small perceiver + linear | ≈ information ceiling |
| E0-c | current prior pooling (replicate `pool_type` used in Run-1) | same head as E0-b | measures pooling loss |
| E0-d | E0-b/c split by training σ_v bins (low/mid/high, from dumped `context_timestep`) | same | which noise level carries signal |

Report: batch-centered **R²** (headline), normalized MSE = MSE/Var[x0],
per-horizon R², per-dim R² (gripper dim separately), train-val gap. Also run
E0-b on the `--from-eval` dump at Run-1's stop step → measures the
**train/eval latent gap** directly.

Reference bands (calibrate against E0-a and the baseline decoder, not as
hard thresholds): R² < 0.02 no usable info · 0.02–0.05 weak · 0.05–0.15
usable · > 0.15 strong.

| E0 outcome | conclusion | next |
|---|---|---|
| full-token R² ≈ 0 | latent carries ~no action info | **pause Run-2**; change the input (video layer via `xattn_layer_idx`, σ_v range, cleaner latent, VLSP-future, +language/state) |
| full-token > 0, pooled ≈ 0 | pooling bottleneck | Run-2 must switch pooling (perceiver / horizon-query cross-attn) |
| both > 0 | input usable | proceed 1.2 / Phase 3 |
| only low σ_v > 0 | training σ_v too noisy | bias σ_v low or curriculum in Run-2 |
| early-horizon ≫ late-horizon | source helps short horizon only | evaluate per-horizon from now on |
| train R² ≫ val R² | probe overfits; weak generalizable info | be conservative in Phase 2 |

### 1.2 D2 — does Run-1's μ carry action information? (Q2)

Run T3 on **B and C separately** (different training pressures), each at
iterations 2k / 5k / 10k / final, with real held-out pairs.

Report, per checkpoint: (1) σ collapse — `logstd_mean/min/max`,
`logstd_floor_frac` (>0.9 ⇒ effective Dirac); (2) μ input-sensitivity —
`mu_batch_std`, pairwise μ distance, μ(true) vs μ(shuffled/fixed);
(3) **μ action-relevance (core)** — R²(μ→x0), R²(linear(μ)→x0),
per-horizon/per-dim, `μ_vs_x0_mse` vs `μ_vs_data_mean_mse`; (4) task
separability — linear probe μ→task_id (supporting only).

| D2 outcome | conclusion | next |
|---|---|---|
| floor_frac≈1, mu_batch_std≈0, R²≈0 | textbook Dirac-constant source | C's behavior is condition-driven; B's collapse consistent |
| floor_frac≈1, mu_batch_std>0, R²≈0 | μ varies with nuisance, not actions | R5′ pressure needed; never cite std as success |
| C input-sensitive, B constant | C's flat direction landed on an arbitrary sensitive function | check source OOD for C; B does not represent C |
| R²(μ→x0) clearly > 0 | μ may carry signal | **immediately** run 1.4 to test whether the DiT uses it |
| B μ informative but B still broken | look at solver stiffness (1.5) / zero_video DiT / eval mismatch (Phase 0) | |

### 1.3 A′ — baseline eval under the audited, identical setup (Q3 anchor)

Answers: is C actually below baseline, at parity, or above?

- **Checkpoint sourcing (decide first):** identify Run-1's exact base
  experiment + data split from its training config. (i) If an existing
  baseline action decoder matches that split (see the `w2a_libero_*` entries
  in `eval/libero/eval.sh`), it may serve as a *provisional* anchor — record
  the iteration mismatch. (ii) The clean arm is training
  `experiment=vlsp_baseline_gaussian` (registered; bit-identical baseline,
  probe on) for the same iterations/budget as Run-1. Do (i) immediately,
  launch (ii) if any Phase-2 decision hinges on a ≤10 pp difference.
- **Identical everything** (post-P0 audited): suite/split, episode indices,
  `stop_video_denoising_step`, video model + 35 steps, action solver steps,
  normalizer stats, seeds, weights=reg. Only delta: experiment name / source.
- Command template (from `eval/libero/eval.sh`, post-T4):

```bash
python run.py \
  --vam_experiment_name vlsp_baseline_gaussian \
  --vam_video_model_path <same as Run-1> \
  --vam_action_model_path <A ckpt> \
  --vam_dataset_statistics_path <same stats json as Run-1> \
  --vam_img_horizon 5 --vam_lowdim_horizon 1 \
  --vam_stop_video_denoising_step <Run-1 value> \
  --vam_num_execute_actions 5 \
  --task_suite_name libero_goal --seed 0
```

- Metrics: success rate (total + per-task), failure categories (1.7),
  action norm/smoothness/gripper stats, rollout length before failure;
  paired per-episode comparison vs C.

| A′ vs C | conclusion | next |
|---|---|---|
| C ≈ A′ | VLSP currently harmless-but-useless | μ-pressure program (R5′) |
| C < A′ | source or training interface actively harmful | restore parity first: R1′ / R2 / gating |
| C > A′ | potential real gain | verify with 1.4 that the source is actually used |

### 1.4 Source-ablation grid on the C checkpoint (Q3: is the source used?)

Eval-only; same checkpoint, replace the source via `--vam_source_override`:

| arm | source fed to the sampler |
|---|---|
| 1 learned | μ(c_v) (+σ if not collapsed) |
| 2 shuffled | μ(c_v′), batch-shuffled videos |
| 3 batch_mean | μ̄ over the eval batch |
| 4 fixed_mean | one fixed μ̄ for all queries |
| 5 gaussian | N(0, I) baseline draw |
| 6 jitter0.1 | μ + 0.1·ε (incrementing seed, P0-4) |
| 7 jitter0.3 | μ + 0.3·ε |
| 8 blend0.5 | 0.5·μ + √0.75·ε |

Metrics: success rate, `sampled_action_mse_gtvid` (T5 offline variant as a
pre-filter), action norm/smoothness, per-task success, and
`source_use_gap = Success(true) − Success(shuffled)` (or
`MSE(shuffled) − MSE(true)`).

| outcome | conclusion | next |
|---|---|---|
| learned ≈ shuffled ≈ mean | source behaviorally **inert**; C rides the condition | R1/R2-style runs can only restore stability, not prove VLSP |
| learned > shuffled | μ informative AND used | focus on source OOD / stability |
| gaussian > learned | current VLSP source actively hurts | fixed σ / blend / gating first |
| jitter > learned | Dirac/too-narrow source harmful | fixing σ (or blending) is necessary |
| jitter < learned | C depends on a deterministic start | be careful with R2 blend |
| mean ≈ learned but ≠ shuffled | source provides a global bias, not per-video info | cross-check D2 R² and per-task clusters |

### 1.5 Solver ablation on B and C (numerical stiffness vs missing information)

Arms (via T4c): Euler 10 / 20 / 50, Heun 10 / 20. Log per step
(`--vam_log_sampler_stats`): `‖v_t‖`, `‖x_t‖`, action delta norm,
jerk/smoothness, early-spike detection — with P0-5 in mind (t≈1 rescale is
mis-calibrated for a collapsed source).

| outcome | conclusion | next |
|---|---|---|
| 50-step / Heun clearly smooths B | Dirac stiffness is a real factor | better solver at eval; high-t loss weighting in training |
| smoother but still task-wrong | numeric AND information problems coexist | μ pressure + solver |
| no effect on C | C limited by cond/video-gap/useless source | prioritize A′ / D3 / R5′ |
| `‖v_t‖` spikes at t≈1 | high-t region unstable | high-t loss profile / solver fix |

### 1.6 D3 — stop-step scan: how much is the video gap? (both C and A′)

`stop_video_denoising_step ∈ {5, 15, 25, 35}` for BOTH arms (the eval.sh
harness already sweeps stop steps — reuse it with collision-free labels).
Optional (if time): a teacher-forced GT-latent condition arm via T1-style GT
dumps. Metrics: success, `sampled_action_mse_gtvid` vs
`sampled_action_mse_generated`, per-task success, taxonomy labels.

| outcome | conclusion |
|---|---|
| C and A′ move together with stop step | video gap is a shared bottleneck, not VLSP-specific |
| C more sensitive than A′ | VLSP source interacts with the video gap |
| GT latent ≫ generated | world-model quality is the main bottleneck |
| insensitive to stop step | bottleneck is decoder / source / training budget |

### 1.7 Failure taxonomy (qualitative, cheap, do alongside 1.3–1.6)

Label ≥ 20 failure MP4s per key arm (rollout dirs already save every
episode) with: **A** kinematically weird motion · **B** normal motion, wrong
goal · **C** wrong direction from the start · **D** near-success, precision
miss · **E** gripper timing wrong · **F** collision/overshoot/jitter ·
**G** video prediction visibly wrong.

Mapping: A/F → source / solver / action distribution; B/C → condition /
video gap; D → training budget / precision; E → per-dim issue (check
per-dim R²); C-normal-but-B-weird → source-only info insufficient or
zero_video-specific pathology.

---

## Phase 2 — decision node (after 1.1–1.4; at the latest after 1.6)

Do not act on any single number; match the joint pattern:

| # | pattern | conclusion | action |
|---|---|---|---|
| 1 | E0 full-token ≈ 0 | source input has no action info | **pause Run-2 entirely**; change input: video layer (`xattn_layer_idx`), σ_v range/curriculum, cleaner latent, VLSP-future, +language/state, or action-AE target. Tuning KL/dropout is pointless here. |
| 2 | E0 full-token > 0, current pooling ≈ 0 | architecture drops the info | Run-2 with perceiver / horizon-query cross-attn pooling, then #3/#4 logic |
| 3 | E0 > 0, D2 μ≈0, A′ ≈ C | VLSP harmless-but-useless; C rides the condition | **R5′** (fixed σ + μ-aux + structured co-dropout + source-use metrics) |
| 4 | E0 > 0, D2 μ≈0, A′ > C | current VLSP actively harmful | first restore parity: **R1′**, R2 blend, source gating, Gaussian fallback; only then R5′ |
| 5 | D2 μ informative, but 1.4 learned ≈ shuffled | prior learned something; DiT ignores the source | add usage pressure: structured co-dropout, source-only branch, true-vs-shuffled ranking loss, high-t weighting (→ R5′ variants) |
| 6 | D2 μ informative AND 1.4 learned > shuffled | source already contributes behaviorally | shift to stability: fixed/calibrated σ, solver, source-OOD logging, D3 video gap, few-step curve |

---

## Phase 3 — Run-2 (revised; launch only what Phase 2 licenses)

### 3.1 R1′ — collapse-proof control (`vlsp_r1p_fixed_std05`)

**Question:** with σ fixed, is source+cond at least no worse than baseline?

**Config (zero new training code):** register on top of the existing variant
machinery (`configs/experiment/world2action.py`):

```
mode=video_prior_sample, conditioning=normal,
logstd_min=-0.6931, logstd_max=-0.6931,   # clamp ⇒ σ ≡ 0.5 exactly
kl_weight=0.0,                            # full KL's μ²/2 pulls μ→0 (action_source_prior.py:667-671) — don't
mean_l2_weight=0.0,                       # it is ‖μ‖² (pull to zero), NOT a μ-aux (config_world2action.py:81)
pool_type per E0 outcome
```

The clamp at `action_source_prior.py:300` makes this a hard fixed σ; the
logstd head just gets zero gradient. (A cleaner `fixed_std` knob is optional
polish, not a launch blocker.) Note the existing `vlsp_next_*_floor_m2`
variants use floor −2.0 ⇒ σ ≥ 0.135 — **weaker than required here**; don't
substitute them.

**Monitor:** `source/std_mean` (must sit at 0.5), variance by dim/horizon,
`mu_batch_std`, R²(μ→x0) via T3, flow loss by t-bin, `probe/sampled_action_mse_gtvid`,
success vs A′, and a 1.4-style learned-vs-shuffled eval at the end.

| gate | handling |
|---|---|
| success < A′ | check source scale / solver / blend / P0-5 rescale before touching μ |
| R²≈0 but success ≈ A′ | fine — that's baseline parity, R1′ passed |
| R² > 0 and learned > shuffled | unexpected upside → stability analysis (Phase-2 #6) |
| high-t loss spike | add high-t weighting or solver fix |

### 3.2 R5′ — the actual hypothesis test (`vlsp_r5p_mu_aux_codropout`)

**Question:** with information pressure on μ, does the source yield gains
beyond baseline?

**Code additions (all additive, VLSP-seam only):**

1. **μ auxiliary**: new `mu_aux_weight` field in `ActionSourcePriorConfig` +
   `L_mu = w·‖μ − x0‖²` in the prior regularizer (x0 is already passed to
   the prior during training — the `source/source_vs_x0_mse` metric proves
   the plumbing exists). Log `loss/source_mu_aux`. Do NOT implement via
   full KL (μ² term) or `mean_l2_weight` (wrong sign of pressure).
2. **Structured co-dropout**: config
   `action_conditioning.co_dropout = {normal: 0.6, source_only: 0.2, cond_only: 0.2}`;
   draw ONE per-sample mode mask in the training step and apply it
   consistently to both `sample_action_source` (cond_only ⇒ Gaussian
   source) and `prepare_action_condition` (source_only ⇒ zero video). Log
   per-branch flow loss (`loss/flow_normal|source_only|cond_only`). A DiT
   mode embedding is deferred — it touches original net code (violates the
   minimal-diff rule) until the gap metrics prove it necessary. If
   source-use gap stays 0, raise source_only; if success drops, lower it.
3. **Grad-ratio metric**: `source/grad_ratio_mu_aux_vs_flow` on prior params
   every K iters; tune `mu_aux_weight` so aux ≈ **10–30 %** of the flow
   gradient (below: μ won't learn; above: the prior becomes a standalone
   regressor fighting the flow objective).

**Config:** R1′ fixed σ=0.5 + items 1–3 + pooling per E0 + optional high-t
loss weighting if 1.5 flagged t≈1.

**Monitor (priority order):**
- *prior health*: σ fixed, `mu_batch_std`, μ horizon std, R²(μ→x0),
  per-step/per-dim R², `μ_vs_data_mean_mse`;
- *source usage*: true-vs-shuffled MSE gap (T5, every N k iters),
  true-vs-mean success gap, per-branch losses;
- *flow field*: loss by t-bin, high-t loss, `‖v_t‖` during sampling, x_t
  norm around the P0-5 rescale;
- *behavior*: `probe/sampled_action_mse_gtvid`, few-step success
  (2/5/10/50 via T4c), success vs A′, taxonomy.

**Early gates (~2k/5k):**

| metric | healthy | if not |
|---|---|---|
| R²(μ→x0) | clearly rising | with E0 positive: aux weight / architecture / grad-ratio problem |
| grad ratio | 10–30 % | retune `mu_aux_weight` |
| `mu_batch_std` | rising, moderate | rising std with flat R² = nuisance variation |
| source_only branch loss | falling | source carries no usable info yet |
| t-bin loss | no high-t blow-up | high-t weighting / solver |

**Mid gates:**

| pattern | handling |
|---|---|
| R² up, true ≈ shuffled | DiT ignores source → strengthen co-dropout / add ranking loss |
| true > shuffled, success flat | source useful but capped by video gap / solver / policy bottleneck (D3, 1.5) |
| success up, few-step flat | gain is regularization, not transport shortening — say so honestly |
| **few-step clearly beats baseline** | strongest positive VLSP signal — protect and reproduce it |
| μ train R² ≫ val R² | prior overfits → regularize / shrink / more data |

**Minimum conditions to declare R5′ healthy (ALL required):**

```
R²(μ→x0) > 0 and a reasonable fraction of the E0 ceiling
true-source > shuffled-source (behaviorally)
sampled_action_mse_gtvid better than R1′/A′
success rate ≥ A′
few-step curve not worse than baseline (better = headline)
```

### 3.3 R2 — blend fallback (`vlsp_r2_blend_050`, already registered)

Role: **parity restoration when A′ > C**, not proof of VLSP. Blend keeps an
un-collapsible √(1−α²)-std Gaussian component. If used, prefer a fixed-σ
variant (`vlsp_r2p_blend050_fixed_std05`, register like 3.1). Read:
R2 ≈ A′ ⇒ successful safety net only; R2 > A′ ⇒ possible source gain — must
pass the 1.4 grid; R2 < C ⇒ Gaussian component interferes or config issue.

### 3.4 Do-NOT list (binding)

1. **No long unmodified R1 (`kl_weight=1e-3`)** — λ=1e-3 may not prevent the
   floor; if run at all, add the hard floor/fixed σ first.
2. **No R4 (source-only) without fixed σ=0.5** — with σ≈0, a constant μ can
   be the *low-loss* solution in source-only; it would train the collapse in
   harder.
3. **`mu_batch_std` is a collapse alarm, never a success metric.** Headline
   = R²(μ→x0) + true-vs-shuffled gap.
4. **No full FastWAM training now.** Its current source is the clean
   first-frame latent — a present-conditioned prior, not VLSP's core claim.

---

## Phase 4 — gated restarts

| gate | conditions (ALL) |
|---|---|
| **restore B (source-only)** | σ not collapsed · R²(μ→x0) > 0 · true > shuffled · source_only branch loss falls normally in R5′ |
| **run R4** | E0 positive · R5′ still cannot make the DiT use the source · fixed σ=0.5 mandatory |
| **FastWAM full training** | mimic-video R5′ positive · FastWAM E0 (below) not negative |

**FastWAM until then (allowed):** E0 probe only — first-frame VAE tokens
`[B, 196, 48] → x0 [B, T, A]`, pooled / full-token / perceiver variants,
same R² bands as 1.1 — plus instrumentation, disabled-equivalence check,
tiny sanity run. If it ever trains: fixed σ=0.5, μ-aux, source-ablation
metrics, true/shuffled gap, few-step curve, blend on standby. Note the MoT
always sees video tokens, so the condition path masks μ even more easily
than in mimic-video — R5-style pressure is *more* load-bearing there.

---

## Appendix A — metric definitions

- **batch-centered R²**: on a held-out set, `R² = 1 − Σ‖ŷ−x0‖² / Σ‖x̄_val−x0‖²`
  where `x̄_val` is the *held-out* action mean. Report alongside
  normalized MSE = MSE/Var[x0]. Always beat E0-a before claiming signal.
- **source_use_gap**: `Success(true) − Success(shuffled)` on paired episodes,
  or `MSE(shuffled) − MSE(true)` offline. Positive = the DiT uses the source.
- **few-step curve**: success at 2/5/10/50 action denoise steps (T4c), same
  arms/episodes. VLSP's transport-shortening claim lives or dies here.
- **pairing**: identical episode index sets + `env.seed(0)` +
  `set_init_state(initial_states[episode_idx])` make arms paired by
  construction; report per-episode deltas, not just totals.

## Appendix B — command quick reference

```bash
# D1/D2 (existing + T3), from mimic-video/model:
python -m scripts.vlsp_probe_prior --ckpt <.../model/iter_0000XX000.pt> [--latents pairs_val.pt]

# T1 dumps → E0:
python -m scripts.vlsp_dump_pairs --split train,val [--from-eval --stop-step <k>]
python -m scripts.vlsp_probe_e0 --pairs pairs_train.pt pairs_val.pt

# A′ / grid / solver / D3 evals: eval/libero/run.py via eval.sh conventions
#   (PYTHONPATH=LIBERO etc.), with T4 flags; ALWAYS post-P0-1 run_label fix.

# training launches (Phase 3), from mimic-video/model:
torchrun --nproc_per_node=4 -m scripts.train \
  --config=cosmos_predict2/configs/config.py -- experiment=vlsp_r1p_fixed_std05
```

## Change log

- 2026-07-05 — initial version; grounded in the post-Run-1 causal analysis
  (Q0–Q4) and a code audit of `eval/libero/run.py`,
  `pipelines/world2action.py`, `models/action_source_prior.py`,
  `configs/experiment/world2action.py`. Referenced from `VLSP.md` §12.
