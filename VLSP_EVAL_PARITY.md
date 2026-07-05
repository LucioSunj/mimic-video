# VLSP Eval/Config Parity Audit

Status: Phase-0 tooling implemented on 2026-07-05. The parity gate is **not
passed yet** because the historical eval logs, run dirs, checkpoints, stats
JSONs, and result trees are not present in this workspace.

## Tooling Implemented

- `eval/libero/run.py`
  - Future rollout dirs are collision-free:
    `<experiment>_<ckpt-stem>_stopafter<k>_execute<n>_seed<seed>/<suite>`.
  - Each rank writes `config_echo.rank<N>.json` with CLI args, resolved
    source-prior and conditioning config, scheduler steps, xattn layer,
    artifact file sizes and sha256 hashes, git commit, and the explicit
    `weights_loaded="reg"` note. `git commit` carries a `-dirty` suffix when the
    worktree has uncommitted changes.
  - **Two-stage collision guard.** A fail-fast CLI-only pre-check runs *before*
    the (minutes-long) model load and raises on a mismatched experiment name /
    checkpoint path / stop step / execute count / horizons / suite / trial count
    / world size / **`num_steps_wait`** / seed. After the model loads, the full
    guard additionally compares checkpoint **sha256**, stats sha256, and the
    behaviour-determining resolved config (`source_prior.enabled/mode`,
    `action_conditioning.mode`, `scheduler.num_denoising_steps`,
    `xattn_layer_idx`) — so a wrong `--vam_experiment_name` that silently flips
    `zero_video` (P0-3) is caught even when the CLI paths look identical.
  - **Soft code-drift check.** A git-commit difference (incl. `-dirty`) does not
    hard-raise but prints a loud `[VLSP-EVAL-AUDIT][WARN]` so mixing sampler
    versions in one dir is visible without blocking a legitimate resume.
  - `config_echo.rank<N>.json` and `summary.rank<N>.json` are written
    **atomically** (temp file + `os.replace`), so a peer rank's guard never
    reads a half-written echo.
  - Each rank writes `summary.rank<N>.json` with per-episode success records.
- `eval/libero/aggregate_results.py`
  - Merges rank summaries into `summary.json`, reports total and per-task
    success, and records missing episode IDs.
- `model/scripts/vlsp_audit_config.py`
  - Resolves experiment names through the same `make_config()` plus
    `override(... experiment=<name>)` path used by eval.
  - Emits JSON or markdown tables for source-prior, conditioning, scheduler,
    EMA, net dimensions, xattn layer, offline latent selectors, and data refs.
    JSON mode stringifies LazyDict `_target_` callables instead of crashing, and
    `data_config` / `video_dataset_*` are dumped in full so split-identifying
    kwargs are visible rather than hidden behind a cherry-picked key subset.

## 0.1 Hazard Verdicts

| hazard | verdict | consequence |
|---|---|---|
| P0-1 rollout-dir collision + resume contamination | Fixed for future runs by the new label, config echo, invariant guard, and summary JSON. Historical dirs are unaudited. | All historical success numbers remain unverified until their result dirs and invocation logs are classified clean/contaminated/unknown. |
| P0-2 eval loads non-EMA weights | Code-side verdict: eval loads `net.*` and `source_prior.*` reg weights. Echo records `weights_loaded="reg"`. | Not a mismatch if train/eval probes are compared as reg-weight probes. |
| P0-3 condition mode carried by `--vam_experiment_name` | Tooling records the literal eval experiment and makes resolved `action_conditioning.mode` / `source_prior.mode` **hard invariants** of the collision guard, so a wrong experiment name that flips `zero_video` now raises instead of contaminating. | Historical B/C evals are fatal/unknown until the literal eval experiment names are recovered. |
| P0-4 fixed seed on every policy query | Echo records the seed policy. `World2ActionPipeline.__call__` still defaults to `seed=0` per query. | By-design for Phase 0; use identical episode sets across arms. |
| P0-5 `x_t` rescale assumes unit-variance source | Logged as a known code fact; not changed in Phase 0. | Quantify later via solver/source ablations, not a parity blocker by itself. |
| P0-6 D1 loading audit | Tool command is documented, but B/C checkpoints are absent locally. | Pending. Any missing `source_prior.*` keys or random-init eval warning voids that eval. |

## Config Snapshot

**Fields that DIFFER across the three registered arms.** These are exactly the
rows the `--diff` flag keeps:

```bash
cd mimic-video/model
python -m scripts.vlsp_audit_config \
  --experiments vlsp_source_only_sample vlsp_source_condition_sample vlsp_baseline_gaussian \
  --diff --markdown
```

| item | vlsp_source_only_sample | vlsp_source_condition_sample | vlsp_baseline_gaussian |
|---|---|---|---|
| action_source_prior.enabled | True | True | False |
| action_source_prior.mode | video_prior_sample | video_prior_sample | gaussian |
| action_conditioning.mode | zero_video | normal | normal |

**Fields VERIFIED IDENTICAL across all three arms** (from the full,
non-`--diff` output; `--diff` drops them precisely because they match):
`scheduler.num_denoising_steps=10`, `pipe_config.xattn_layer_idx=20`,
`model.config.sampled_mse_probe_interval=500`, `ema.enabled=False`,
`ema.rate=0.1`, `action_source_prior.logstd_min=-5.0`,
`action_source_prior.logstd_max=1.0`, `action_source_prior.pool_type=mean`,
`action_source_prior.kl_weight=0.0`, `net.max_horizon=61`,
`net.out_channels=10`, `net.crossattn_emb_channels=2048`.

**Findings from this A3 dry-run that change the audit plan:**

1. **The registered variants have `ema.enabled=False`.** B/C therefore had **no
   EMA** — the P0-2 "reg vs EMA" question collapses to "reg throughout,
   consistent" *for the registered configs*. Still confirm the literal Run-1
   config had not overridden `ema.enabled` (see §Required External Inputs).
2. **Both `offline_video_embedding_dir` and `offline_video_latent_dir` resolve
   to empty** for every registered variant. But the repo's `LIBERO no-RGB latent
   training flow` implies Run-1 likely trained from an offline latent/embedding
   cache — which means Run-1 must have supplied that path (and possibly other
   knobs) via **CLI overrides or a different registered name**, not the bare
   variant. Consequence: recovering the *literal* Run-1 launch command (F11) is
   not optional — the experiment name alone does not reconstruct the training
   input pipeline, and `video latent source (train)` on the sheet stays
   `pending` until the full override string is found.
3. `data_config` resolves to `{"config_name": "libero_object_tenth", "_target_":
   "<callable ...get_data_config>"}` — i.e. the registered variant defaults to
   the **object_tenth** split, which does **not** match Run-1's **LIBERO-goal**
   suite. This is direct evidence (not just suspicion) that Run-1 overrode the
   data/base config on the CLI; the registered name is a template. Recover the
   full override string to learn Run-1's actual split (goal_half / goal_tenth /
   goal_one) and its offline-cache path.

## 0.2 Parity Sheet

Legend: `known` means code/config-side fact verified in this workspace.
`pending` means it requires external run logs, checkpoints, stats files, or
result dirs.

| item | B train | B eval | C train | C eval | A' eval |
|---|---|---|---|---|---|
| experiment string actually used | pending recover from train log; expected `vlsp_source_only_sample` | pending recover from eval invocation; must resolve to `vlsp_source_only_sample` | pending recover from train log; expected `vlsp_source_condition_sample` | pending recover from eval invocation; must resolve to `vlsp_source_condition_sample` | planned `vlsp_baseline_gaussian` |
| checkpoint iteration / file / sha256 | pending | pending; future echo records sha256 | pending | pending; future echo records sha256 | pending |
| weights loaded | reg (registered variant has `ema.enabled=False`; confirm Run-1 did not override) | known eval reg | reg (registered `ema.enabled=False`; confirm Run-1) | known eval reg | known eval reg |
| normalizer stats JSON | pending data/run config | pending eval flag and sha256 | pending data/run config | pending eval flag and sha256 | pending |
| source mode | expected `video_prior_sample` | expected `video_prior_sample` if eval name is correct | expected `video_prior_sample` | expected `video_prior_sample` if eval name is correct | `gaussian`, VLSP disabled |
| condition mode | expected `zero_video` | expected `zero_video` if eval name is correct | expected `normal` | expected `normal` if eval name is correct | `normal` |
| zero_video implementation identical? | known single `apply_action_conditioning` path | pending only on correct eval experiment name | n/a | n/a | n/a |
| `stop_video_denoising_step` | n/a for train | pending | n/a for train | pending | pending |
| video latent source | pending: registered variant has empty offline dirs, so Run-1 used a CLI override / other name — recover the full launch command | generated video latent at stop step | pending: same override recovery required | generated video latent at stop step | generated video latent at stop step |
| video steps | n/a | known hardcoded 35 in eval | n/a | known hardcoded 35 in eval | known hardcoded 35 in eval |
| action solver steps | known 10 (registered config) | known 10 (registered config) | known 10 (registered config) | known 10 (registered config) | known 10 (registered config) |
| `x_t` rescale present | known present | known present | known present | known present | known present |
| obs_dropout / condition dropout | obs_dropout 0.2 by design; conditioning dropout 0.0 for expected B | eval obs_dropout 0.0; conditioning dropout 0.0 | obs_dropout 0.2 by design; conditioning dropout 0.0 | eval obs_dropout 0.0; conditioning dropout 0.0 | eval obs_dropout 0.0 |
| proprio path into DiT identical? | known normalized lowdim path | known normalized lowdim path; eval lowdim horizon pending flag | known normalized lowdim path | known normalized lowdim path; eval lowdim horizon pending flag | known normalized lowdim path; eval lowdim horizon pending flag |
| seed policy | global training RNG; source drawn before flow timestep | known query source seed defaults to 0; env seed 0; paired init states | global training RNG; source drawn before flow timestep | known query source seed defaults to 0; env seed 0; paired init states | known query source seed defaults to 0; env seed 0; paired init states |
| task split / episode indices | pending | pending eval flags and summaries | pending | pending eval flags and summaries | pending |
| results dir status | n/a | historical pending; future collision-free guarded dir | n/a | historical pending; future collision-free guarded dir | planned guarded dir |

## History Audit

No local `results/` tree, checkpoint files, train stdout, eval stdout, or stats
JSONs were found in this workspace during this pass. Historical behavioral
numbers therefore remain **unknown/unverified**.

| results dir | invocations | episodes on disk / in logs | status | consequence |
|---|---|---|---|---|
| pending external collection | pending | pending | unknown | void for Phase-1 decisions until classified or re-run under the fixed harness |

## Required External Inputs

- B run dir `vlsp_source_only_goal_20260701_125625`: checkpoints, startup config
  dump, and full train log.
- C run dir `vlsp_source_condition_goal_20260629_183112`: checkpoints, startup
  config dump, and full train log.
- Exact historical eval invocations for B/C/A': stdout logs or shell history
  with `--vam_experiment_name`, checkpoint path, stats JSON, stop step, suite,
  seed, rank, and world size.
- Complete historical `results/` tree for contamination classification.
- Dataset statistics JSON used by each eval.
- If offline video embeddings were used: `metadata.json` and `video_sigma.npy`.

## Exit Verdict

Phase 0 is not complete. The code-side guardrails and audit tools are in place,
but B/C behavioral conclusions must remain **unverified** until the historical
audit is filled or B/C are re-run under the fixed harness.
