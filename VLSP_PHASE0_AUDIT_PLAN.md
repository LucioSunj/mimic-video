# VLSP Phase 0 — Eval/Config Parity Audit: Implementation Plan
# (第一步落地：eval / config parity 审计的详细实现计划)

**Parent:** [VLSP_EXECUTION_PLAN.md](VLSP_EXECUTION_PLAN.md) Phase 0 (question
Q0). Nothing in Phases 1–3 is interpretable until this passes.

**Deliverables:**
1. Fixed, auditable eval harness (work item A) — collision-free result dirs +
   per-run config echo.
2. Historical-results contamination verdicts (work item B).
3. `VLSP_EVAL_PARITY.md` — the filled parity sheet + per-arm verdicts (work
   item C/D).
4. If any fatal mismatch: re-run B/C evals under the fixed harness (work
   item D), then amend `VLSP.md` §12.

**Code-fact baseline.** The audit below is grounded in facts already verified
by code inspection (2026-07-05); the sheet records them so nobody re-derives
them per-run:

| # | verified fact | where |
|---|---|---|
| F1 | `run_label` has no experiment name; resume scan imports on-disk successes → cross-run contamination is possible whenever ckpt filenames collide (every run saves `iter_0000XX000.pt`) | `eval/libero/run.py:319-322`, `:366-376` |
| F2 | Eval loads **reg** weights: `net.*` → DiT, `net_ema.*` silently dropped; `source_prior.*` (reg) → prior | `pipelines/world2action.py` `from_config` |
| F3 | The in-training probe `probe/sampled_action_mse_gtvid` samples through `self.pipe` = **reg** weights, seed=0, GT-video condition → probe and eval use the same weight class (comparable) | `models/world2action_model.py:884-902` |
| F4 | `zero_video` is `torch.zeros_like` on ONE shared code path (`apply_action_conditioning`) for train and eval; `training=True/False` only changes RNG for `shuffled_video`/`dropout_video` | `models/action_source_prior.py:604-643`; call sites `world2action_model.py:835` (train), `pipelines/world2action.py:363-367` (eval) |
| F5 | Train `obs_dropout=0.2` (hardcoded), eval `obs_dropout=0.0` — **by design**, not a mismatch | `world2action_model.py:458`; `world2action.py:382` |
| F6 | The `x_t /= sqrt((1−t)²+t²)` rescale lives **inside** `denoise()` → applied identically in training loss and eval sampling. P0-5 is therefore a *distribution/stiffness* question (→ Plan 1.5), **not** a train/eval parity violation | `world2action.py:304`; train path `compute_loss → pipe.denoise` `world2action_model.py:452` |
| F7 | Eval condition mode is carried by `--vam_experiment_name` (config re-resolved at eval); wrong name silently flips `zero_video` on/off | `eval/libero/run.py:63`; `world2action.py:269-271` |
| F8 | Every policy query runs with `seed=0` (not threaded through `VAMInference._query_policy`) → identical stochastic source draw at every replan, plus `env.seed(0)` and per-episode `set_init_state(initial_states[episode_idx])` → arms are paired by construction | `run.py:173-181`, `:234`, `:379`; `video2world2action.py:35`; `world2action.py:338` |
| F9 | Training video latents come from ONE of three paths, config-selected: (a) `offline_video_embedding_dir` → precomputed GT-video hidden states with a **fixed per-sample σ_v** (`video_sigma.npy`), layer checked against `xattn_layer_idx` at load; (b) `offline_video_latent_dir` → offline tokenizer latents + online video DiT with σ_v drawn per step; (c) fully online. Which one Run-1 used is an audit item (C1) | `world2action_model.py:82-89, :164-180, :519-554, :661-722` |
| F10 | Training draws the source **before** the flow timestep to preserve the baseline RNG order; source uses global RNG at train (`training=True`, no generator) | `world2action_model.py:804-825` |
| F11 | Run-dir names in Run-1 (`vlsp_source_only_goal_20260701_125625`, `vlsp_source_condition_goal_20260629_183112`) do **not** literally match the registered variant names (`vlsp_source_only_sample`, `vlsp_source_condition_sample`) — the exact `experiment=` strings used at train AND eval must be recovered from logs, not assumed | `configs/experiment/world2action.py:179-262`; `VLSP_RUN1_ANALYSIS.md` |
| F12 | **A3 dry-run (2026-07-05, CPU) confirms:** the registered `vlsp_*` variants resolve to `ema.enabled=False` and **empty** `offline_video_embedding_dir` / `offline_video_latent_dir`. Since Run-1 almost certainly trained from an offline latent/embedding cache (LIBERO no-RGB flow), it must have set that path (and possibly `ema`, split, etc.) via **CLI overrides or a different registered name**. Recovering only the `experiment=` string is therefore insufficient — the **full launch command / override string** is a required Phase-0 input, and `video latent source (train)` stays `pending` until it is found | `scripts/vlsp_audit_config.py`; `VLSP_EVAL_PARITY.md` §Config Snapshot |

---

## 0. Inputs to collect first (on the training / eval machines)

Gather into one place (e.g. `audit_20260705/`) before touching code:

- [ ] B run dir (`vlsp_source_only_goal_20260701_125625`): checkpoints
  (`model/iter_0000*.pt`, note exact filenames), the startup config dump in
  train stdout, full train log.
- [ ] C run dir (`vlsp_source_condition_goal_20260629_183112`): same.
- [ ] **Every** historical eval invocation record for B/C (and any baseline):
  stdout logs, sbatch/tmux/shell history — we need the literal
  `--vam_experiment_name`, `--vam_action_model_path`,
  `--vam_stop_video_denoising_step`, `--vam_dataset_statistics_path`,
  `--task_suite_name`, `--seed`, `--eval_world_size` per invocation.
- [ ] The complete `results/` tree(s): `find results -maxdepth 2 -type d`.
- [ ] Training metrics dump `source_cond_vs_source_only_metrics_20260704/`.
- [ ] If F9(a): the offline embedding dir (`metadata.json`,
  `video_sigma.npy`) used by Run-1 training.

If an eval invocation's flags cannot be recovered from any log → its numbers
are classified **unknown** in B3 (same consequence as contaminated: void).

---

## Work item A — eval-harness fixes (eval-only code; no training-path changes)

### A1. Collision-free `run_label` + config echo + contamination guard

**File:** `eval/libero/run.py`. Blocks all future P0-1 recurrences.

1. **Label** (replaces `run.py:319-321`):

```python
def _short_exp(name: str, keep: int = 40) -> str:
    if len(name) <= keep:
        return name
    return f"{name[:keep]}~{hashlib.sha1(name.encode()).hexdigest()[:8]}"

run_label = (
    f"{_short_exp(vam_experiment_name)}_{vam_action_model_path.stem}"
    f"_stopafter{vam_stop_video_denoising_step}_execute{vam_num_execute_actions}"
    f"_seed{seed}"
)
```

   (`_short_exp` keeps the long baseline names from `eval.sh` under path
   limits; vlsp_* names pass through readable.)

2. **Config echo**: at startup write
   `rollout_dir / f"config_echo.rank{eval_rank}.json"` containing: all CLI
   args verbatim; resolved `action_source_prior` fields (enabled, mode,
   logstd_min/max, kl/mean_l2/std_reg, blend_alpha, source_dropout_prob,
   pool_type); `action_conditioning.mode`; `ema.enabled` + note
   `weights_loaded="reg"` (F2); `scheduler.num_denoising_steps`;
   `xattn_layer_idx`; ckpt path + file size + sha256; stats json path +
   sha256; git commit of the repo. All fields are readable off
   `policy.model.world2action_pipeline.config` after construction.

3. **Contamination guard**: before the resume scan, if any
   `config_echo.rank*.json` already exists in `rollout_dir`, compare the
   invariant fields (experiment name, ckpt sha256, stop step, suite, seed,
   stats sha256); on mismatch **raise** with both configs printed. Resume
   stays allowed only for a byte-identical setup.

4. **Machine-readable summary**: on completion write
   `rollout_dir / f"summary.rank{eval_rank}.json"`:
   `{episode_idx: {task_id, task, success}}` + totals. New tiny script
   `eval/libero/aggregate_results.py --dir results/<run_label>/<suite>`
   merges shard summaries → `summary.json` (total + per-task success, episode
   count, missing-episode check against the expected index set). Every
   Phase-1 arm reuses this.

**Acceptance:** one-task smoke run (`--num_trials_per_task 1`) produces the
new label, echo, summary; rerunning it resumes cleanly; rerunning with a
different experiment name against the same dir raises.

### A2. Startup audit log line

Same fields as the echo, one `log.info("[VLSP-EVAL-AUDIT] ...")` line — so
future audits can grep stdout even when the results dir is lost.

### A3. `model/scripts/vlsp_audit_config.py` (new, CPU-only)

Purpose: fill every *train-side / config-side* row of the parity sheet
mechanically, and prove what any experiment name resolves to (F7/F11).

```
python -m scripts.vlsp_audit_config \
    --experiments vlsp_source_only_sample vlsp_source_condition_sample vlsp_baseline_gaussian \
    [--diff]         # print only fields that differ from the first experiment
    [--markdown]     # emit parity-sheet rows directly
```

Implementation: `make_config()` + `override(config, ["--", f"experiment={name}"])`
(same code path as eval, `run.py:62-63`), then print:

- `job.name`, `job.group`
- `action_source_prior.*` (all fields of `ActionSourcePriorConfig`,
  `configs/config_world2action.py:34-85`)
- `action_conditioning.*` (`:88-100`)
- `ema.enabled`, `ema.rate`
- `scheduler.alpha/beta/num_denoising_steps`
- `net.max_horizon / out_channels / crossattn_emb_channels`, `xattn_layer_idx`
- `model.config.offline_video_embedding_dir / offline_video_latent_dir /
  *_required` (F9 selector), `sampled_mse_probe_interval`
- best effort: normalizer/data-config identifiers

No GPU, no checkpoint needed. **Also run it with the exact Run-1
`experiment=` strings recovered per F11** — if those names no longer resolve
(or resolve to different modes than assumed), that is itself a Phase-0
finding.

---

## Work item B — historical results audit (retroactive P0-1)

### B1. Inventory + collision detection

On each eval machine:

```bash
find results -mindepth 2 -maxdepth 2 -type d | sort > audit_20260705/result_dirs.txt
for d in results/*/*/; do printf "%s\t%d\t%d\n" "$d" \
  "$(ls "$d" | grep -c '^episode')" \
  "$(ls "$d" | grep -c '_success_')"; done > audit_20260705/dir_counts.tsv
```

Parse each `run_label` into `(ckpt_stem, stopafter, execute)`. Join against
the invocation records from §0: a label is **collision-suspect** if ≥ 2
distinct `(experiment_name, ckpt_path)` invocations map to it. Note the
concrete Run-1 hazard: B@`iter_000026000.pt` and C@`iter_000024000.pt` differ
in stem — but any *same-iteration* pair (B/C/baseline at 24k; re-evals of the
same ckpt at different code states) collides.

### B2. Per-directory contamination test

For each results dir × each invocation that wrote to it:

1. From that invocation's stdout, extract the episode set it actually
   executed: lines `Saved rollout MP4 at path .../episode{N}_...`.
2. Union over all invocations of the same `(experiment, ckpt)`.
3. Episodes present on disk but in **no** matching log → foreign or
   unlogged → dir is `contaminated` (foreign proven) or `unknown` (logs
   incomplete).
4. Flag any aggregate number that was computed by counting files in a dir
   (`success`-in-filename over total) — with 5-way sharding (per `eval.sh`)
   that was the only way to get a total, so contaminated dirs corrupt totals
   directly.

### B3. Verdict table (goes into `VLSP_EVAL_PARITY.md` §History)

| results dir | invocations (exp, ckpt, date) | episodes on disk / in logs | status | consequence |
|---|---|---|---|---|
| … | … | … | clean / contaminated / unknown | keep / **void → re-run** |

**Rule:** only `clean` numbers survive. Everything voided is re-run in D2
under the A1 harness. Explicitly re-examine the number behind the **"revised
status after eval setup audit"** paragraph in `VLSP.md` §12 — the earlier
artifact and this hazard must be reconciled (same root cause or two separate
issues; write down which).

---

## Work item C — fill the parity sheet

Sheet lives in `VLSP_EVAL_PARITY.md`; columns `B train | B eval | C train |
C eval | A′ eval(planned)`. How each row is obtained:

| row | source of truth |
|---|---|
| experiment string actually used | train stdout config dump; eval logs / shell history (F11) — **never assume** |
| ckpt iteration / file / sha256 | run dirs; echo (new runs) |
| weights loaded | constant: train=reg(+EMA tracked), eval=reg (F2); probe=reg (F3) → mark "consistent" |
| source mode / prior knobs | A3 on the recovered experiment strings |
| condition mode | A3 (must be: B `zero_video`, C `normal`) + F7 check that the *eval* name resolves the same |
| zero_video implementation parity | F4 → "single path, zeros, identical"; only remaining check = correct experiment name at eval |
| video-latent source at train | A3 `offline_*` fields + train log `Using offline video embeddings ...` line (F9); record which of (a)/(b)/(c) |
| σ_v seen at train | (a): `np.load(video_sigma.npy)` → histogram (min/median/max, save the plot path — feeds E0-d); (b)/(c): `draw_video_sigma` distribution from config |
| video latent at eval | generated video, `num_sampling_steps=35` hardcoded (`run.py:123`), stopped at `stop_after_step`; sigma at stop returned by `generate_video` and fed to prior + DiT (`video2world2action.py:48-68`) |
| stop_video_denoising_step | per-invocation flags; Run-1's value is REQUIRED (D3 anchors on it) |
| action solver | BetaScheduler Euler; `num_denoising_steps` from A3; x_t rescale consistent (F6) |
| obs_dropout / cond dropout | F5: train 0.2 / eval 0.0 by design; `action_conditioning.dropout_prob` from A3 (0 unless dropout mode) |
| proprio path | `state_B_HO_O` normalized then into DiT both sides (`world2action.py:345-346`, `world2action_model.py:789-796`); lowdim_horizon from eval flags (=1 in eval.sh) |
| seed policy | F8; per-invocation `--seed` |
| normalizer stats | train: data_config; eval: `--vam_dataset_statistics_path` (+sha256). Must match the split B/C trained on |
| task split / episodes | suite, `num_trials_per_task`, `eval_world_size`, per-shard episode index sets (episodes assigned by `total_episodes % world_size == rank` with pre-increment, `run.py:362-365`) |
| results dir status | B3 verdict |

Plus the **D1 checkpoint audit** (existing tool, minutes, CPU):

```bash
cd mimic-video/model
python -m scripts.vlsp_probe_prior --ckpt <B .../model/iter_000026000.pt>
python -m scripts.vlsp_probe_prior --ckpt <C .../model/iter_000024000.pt>
```

Pass = `source_prior.*` key count > 0 **and** every historical eval log shows
`Loaded source_prior from checkpoint (missing=0 …)` — a `randomly initialized`
warning in any eval log voids that eval on the spot.

---

## Work item D — verdicts, conditional re-runs, doc updates

### D1. Classify every sheet discrepancy

`fatal` (voids behavioral conclusions; triggers D2) — e.g. wrong eval
experiment name (F7), contaminated dir (B3), missing source-prior load, wrong
stats json, differing stop step between compared arms.
`by-design` (record, no action) — e.g. F5 dropout, train-σ_v vs eval-σ_v
distribution difference.
`unknown` (treat as fatal for trust purposes).

### D2. Re-run matrix (only what B3/D1 voided; A1 harness mandatory)

```bash
# from eval/libero, env per eval.sh (PYTHONPATH=LIBERO, VK_*, etc.)
python run.py \
  --vam_experiment_name vlsp_source_condition_sample \        # C — exact recovered name
  --vam_video_model_path <Run-1 video ckpt> \
  --vam_action_model_path <C .../iter_000024000.pt> \
  --vam_dataset_statistics_path <Run-1 stats json> \
  --vam_img_horizon 5 --vam_lowdim_horizon 1 \
  --vam_stop_video_denoising_step <Run-1 value> \
  --vam_num_execute_actions 5 \
  --task_suite_name libero_goal --seed 0 \
  --eval_rank {0..4} --eval_world_size 5          # one process per rank, per eval.sh
# B: same, with vlsp_source_only_sample + <B .../iter_000026000.pt>
python aggregate_results.py --dir results/<new run_label>/libero_goal
```

Full suite = 10 tasks × 50 trials = 500 episodes/arm (meets the ≥200 bar);
identical episode index sets across arms (F8 pairing).

### D3. Documentation + commit

- `VLSP_EVAL_PARITY.md`: sheet + B3 history table + D1 verdicts + (if run)
  corrected B/C numbers with per-task breakdown.
- `VLSP.md` §12: amend the Run-1 revised-status paragraph if corrected
  numbers move the story; link the parity doc.
- Commits (submodule first, then parent pointer, per repo workflow): one for
  A1/A2/A3 (eval tooling), one for the audit docs. Trailer:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## Execution order, dependencies, effort

| step | depends on | runs on | est. effort |
|---|---|---|---|
| §0 collect inputs | — | GPU/eval machine (file copy) | 0.5–1 h |
| A3 audit script + run on all names | — | laptop CPU | 1–1.5 h |
| A1 + A2 harness fix + smoke | — | laptop CPU + 1 short sim run | 1.5–2 h |
| B1–B3 history audit | §0 | laptop (logs local) | 1–2 h (log-quality dependent) |
| C sheet fill + D1 probe | §0, A3, B3 | laptop CPU | 1–1.5 h |
| D1 verdicts | C | — | 0.5 h |
| D2 re-runs (if triggered) | A1, D1 | eval GPUs | ~1 GPU-day wall-clock for B+C full suites (sharded ×5) |
| D3 docs + commits | all | laptop | 0.5 h |

Parallelization: A3/A1 can start immediately while §0 collection runs; B and
C audits are independent. Everything except D2 is CPU-only.

## Exit checklist (gate to Phase 1 of the execution plan)

- [ ] A1 merged: new labels include experiment+seed; config echo + summary
      json written; echo-mismatch guard raises; aggregator works.
- [ ] Exact Run-1 `experiment=` strings recovered and resolved via A3 (F11).
- [ ] Every historical results dir classified clean/contaminated/unknown;
      voided-numbers list written; the earlier "eval setup audit" correction
      reconciled with B3 findings.
- [ ] D1 pass on both B and C checkpoints (keys present + load line in logs).
- [ ] Parity sheet complete for B/C train+eval (A′ column planned); every
      discrepancy classified fatal / by-design / unknown.
- [ ] If any fatal: B/C re-evaluated under the fixed harness; corrected
      numbers + per-task breakdown recorded; `VLSP.md` §12 amended.
- [ ] Training σ_v distribution extracted and archived (unblocks E0-d).
- [ ] `VLSP_EVAL_PARITY.md` committed; parent pointer bumped.

**Verdict semantics** (unchanged from the execution plan): any fatal mismatch
⇒ prior B/C behavioral conclusions drop to "unverified" until reproduced under
the fixed harness; no mismatch ⇒ B/C behavior differences are admissible
evidence for Phases 1–2.
