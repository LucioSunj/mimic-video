"""Run LIBERO evaluation with the Video Action Model (VAM) policy."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import re
import subprocess
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import attrs
import imageio
import numpy as np
import torch
import tqdm
import tyro
from einops import rearrange
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from omegaconf import DictConfig, ListConfig, OmegaConf
from scipy.spatial.transform import Rotation

from cosmos_predict2.configs.config import make_config
from cosmos_predict2.data.action.utils import extract_normalization_types
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import override

LIBERO_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
}

CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640

DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

EPISODE_RE = re.compile(r"^episode(?P<episode_id>\d+)_(?P<status>success|failure)_.*\.mp4$")


def _short_exp(name: str, keep: int = 40) -> str:
    if len(name) <= keep:
        return name
    return f"{name[:keep]}~{hashlib.sha1(name.encode()).hexdigest()[:8]}"


def _sha256_file(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_size(path: pathlib.Path) -> int | None:
    return path.stat().st_size if path.is_file() else None


def _git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    # A dirty worktree means the recorded commit does NOT fully describe the
    # code that produced these rollouts; flag it so the audit cannot silently
    # trust it (and so the soft resume-time check below can spot code drift).
    try:
        dirty = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        dirty = ""
    return f"{commit}-dirty" if dirty else commit


def _atomic_write_json(path: pathlib.Path, payload: Any) -> None:
    """Write JSON via a per-pid temp file + rename so a concurrent reader (e.g.
    another rank's guard) never observes a half-written file. The temp name does
    not match the ``*.json`` globs used to discover echoes/summaries."""
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return _jsonable(OmegaConf.to_container(value, resolve=True))
    if attrs.has(value.__class__):
        return _jsonable(attrs.asdict(value))
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, pathlib.Path):
        return str(value)
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
            if not k.startswith("_") and isinstance(v, (str, int, float, bool, pathlib.Path, dict, list, tuple))
        }
    return str(value)


def _make_run_label(
    *,
    vam_experiment_name: str,
    vam_action_model_path: pathlib.Path,
    vam_stop_video_denoising_step: int,
    vam_num_execute_actions: int,
    seed: int,
) -> str:
    return (
        f"{_short_exp(vam_experiment_name)}_{vam_action_model_path.stem}"
        f"_stopafter{vam_stop_video_denoising_step}_execute{vam_num_execute_actions}"
        f"_seed{seed}"
    )


def _index_existing_rollouts(rollout_dir: pathlib.Path) -> dict[int, tuple[pathlib.Path, bool]]:
    indexed: dict[int, tuple[pathlib.Path, bool]] = {}
    for path in rollout_dir.glob("episode*.mp4"):
        match = EPISODE_RE.match(path.name)
        if match is None:
            continue
        episode_id = int(match.group("episode_id"))
        success = match.group("status") == "success"
        if episode_id in indexed:
            prev_path, _ = indexed[episode_id]
            raise RuntimeError(f"Duplicate rollout files for episode {episode_id}: {prev_path} and {path}")
        indexed[episode_id] = (path, success)
    return indexed


# CLI-derived invariants that gate whether two runs may share a results dir.
# Kept separate so the fail-fast pre-check (before the slow model load) can be
# computed from CLI args alone; the full check adds checkpoint hashes + resolved
# config in _config_invariants once the model is loaded.
_CLI_INVARIANT_KEYS = (
    "vam_experiment_name",
    "vam_video_model_path",
    "vam_action_model_path",
    "vam_dataset_statistics_path",
    "vam_stop_video_denoising_step",
    "vam_num_execute_actions",
    "vam_img_horizon",
    "vam_lowdim_horizon",
    "task_suite_name",
    "num_trials_per_task",
    "eval_world_size",
    "num_steps_wait",
    "seed",
)


def _cli_invariants(cli_args: dict[str, Any]) -> dict[str, Any]:
    return {key: cli_args.get(key) for key in _CLI_INVARIANT_KEYS}


def _config_invariants(echo: dict[str, Any]) -> dict[str, Any]:
    """Hard invariants: any mismatch means the two runs are NOT comparable and
    must not share a results dir (raises). Code-version drift is checked
    separately and softly (see ``_soft_invariants``)."""
    cli = echo["cli_args"]
    artifacts = echo["artifacts"]
    resolved = echo.get("resolved_config") or {}
    source_prior = resolved.get("action_source_prior") or {}
    conditioning = resolved.get("action_conditioning") or {}
    return {
        "vam_experiment_name": cli["vam_experiment_name"],
        "vam_video_model_path": artifacts["video_model"]["path"],
        "vam_video_model_sha256": artifacts["video_model"]["sha256"],
        "vam_action_model_path": artifacts["action_model"]["path"],
        "vam_action_model_sha256": artifacts["action_model"]["sha256"],
        "vam_dataset_statistics_path": artifacts["dataset_statistics"]["path"],
        "vam_dataset_statistics_sha256": artifacts["dataset_statistics"]["sha256"],
        "vam_stop_video_denoising_step": cli["vam_stop_video_denoising_step"],
        "vam_num_execute_actions": cli["vam_num_execute_actions"],
        "vam_img_horizon": cli["vam_img_horizon"],
        "vam_lowdim_horizon": cli["vam_lowdim_horizon"],
        "task_suite_name": cli["task_suite_name"],
        "num_trials_per_task": cli["num_trials_per_task"],
        "eval_world_size": cli["eval_world_size"],
        "num_steps_wait": cli.get("num_steps_wait"),
        "seed": cli["seed"],
        # Resolved-config fields that change behaviour: a wrong --vam_experiment_name
        # (P0-3) flips these even when the CLI paths look identical.
        "source_prior_enabled": source_prior.get("enabled"),
        "source_prior_mode": source_prior.get("mode"),
        "action_conditioning_mode": conditioning.get("mode"),
        "scheduler_num_denoising_steps": resolved.get("scheduler_num_denoising_steps"),
        "xattn_layer_idx": resolved.get("xattn_layer_idx"),
    }


def _soft_invariants(echo: dict[str, Any]) -> dict[str, Any]:
    """Recorded and warned-on but non-fatal: code version can legitimately differ
    between resumes (an unrelated commit) yet still changes the sampler, so it is
    surfaced loudly rather than either ignored or made a hard blocker."""
    return {"git_commit": (echo.get("git") or {}).get("commit")}


def _load_existing_echoes(rollout_dir: pathlib.Path) -> list[tuple[pathlib.Path, dict[str, Any]]]:
    echoes: list[tuple[pathlib.Path, dict[str, Any]]] = []
    for existing_path in sorted(rollout_dir.glob("config_echo.rank*.json")):
        try:
            with existing_path.open("r", encoding="utf-8") as f:
                echoes.append((existing_path, json.load(f)))
        except (json.JSONDecodeError, OSError):
            # Concurrently-written or truncated echo: skip it. Atomic writes mean
            # a complete file appears under this name shortly, and the writing
            # rank runs its own full check.
            continue
    return echoes


def _precheck_cli_invariants(rollout_dir: pathlib.Path, cli_args: dict[str, Any]) -> None:
    """Fail-fast guard run BEFORE the (minutes-long) model load.

    Compares only CLI-derived invariants (no checkpoint hashing, no model) so a
    wrong ``--vam_experiment_name`` / stop step / seed pointed at a populated
    results dir raises immediately. The full guard (checkpoint sha256 + resolved
    config) still runs in ``_write_config_echo`` once the model is loaded.
    """
    current = _cli_invariants(cli_args)
    for existing_path, existing_echo in _load_existing_echoes(rollout_dir):
        existing_cli = existing_echo.get("cli_args") or {}
        existing = {key: existing_cli.get(key) for key in _CLI_INVARIANT_KEYS}
        if existing != current:
            raise RuntimeError(
                "Existing rollout directory was created with different CLI eval "
                "invariants (fail-fast pre-check).\n"
                f"Directory: {rollout_dir}\n"
                f"Existing echo: {existing_path}\n"
                f"Existing: {json.dumps(existing, indent=2, sort_keys=True)}\n"
                f"Current: {json.dumps(current, indent=2, sort_keys=True)}"
            )


def _write_config_echo(
    *,
    rollout_dir: pathlib.Path,
    eval_rank: int,
    cli_args: dict[str, Any],
    policy: "VAMInference",
) -> None:
    world2action_config = policy.model.world2action_pipeline.config
    action_model_path = pathlib.Path(cli_args["vam_action_model_path"])
    video_model_path = pathlib.Path(cli_args["vam_video_model_path"])
    stats_path = pathlib.Path(cli_args["vam_dataset_statistics_path"])
    echo = {
        "schema_version": 1,
        "cli_args": _jsonable(cli_args),
        "resolved_config": {
            "action_source_prior": _jsonable(world2action_config.action_source_prior),
            "action_conditioning": _jsonable(world2action_config.action_conditioning),
            "ema": {
                "enabled": bool(world2action_config.ema.enabled),
                "rate": _jsonable(getattr(world2action_config.ema, "rate", None)),
            },
            "weights_loaded": "reg",
            "scheduler": _jsonable(world2action_config.scheduler),
            "scheduler_num_denoising_steps": int(world2action_config.scheduler.num_denoising_steps),
            "xattn_layer_idx": int(world2action_config.xattn_layer_idx),
        },
        "artifacts": {
            "video_model": {
                "path": str(video_model_path),
                "size_bytes": _file_size(video_model_path),
                "sha256": _sha256_file(video_model_path),
            },
            "action_model": {
                "path": str(action_model_path),
                "size_bytes": _file_size(action_model_path),
                "sha256": _sha256_file(action_model_path),
            },
            "dataset_statistics": {
                "path": str(stats_path),
                "size_bytes": _file_size(stats_path),
                "sha256": _sha256_file(stats_path),
            },
        },
        "git": {"commit": _git_commit()},
        "audit_notes": {
            "weights_loaded": "reg",
            "action_obs_dropout_eval": 0.0,
            "video_num_sampling_steps": policy.num_sampling_steps,
            "seed_policy": "World2ActionPipeline is called without a seed override; default seed=0 per policy query.",
        },
    }
    current_invariants = _config_invariants(echo)
    current_soft = _soft_invariants(echo)
    for existing_path, existing_echo in _load_existing_echoes(rollout_dir):
        existing_invariants = _config_invariants(existing_echo)
        if existing_invariants != current_invariants:
            raise RuntimeError(
                "Existing rollout directory was created with different eval invariants.\n"
                f"Directory: {rollout_dir}\n"
                f"Existing echo: {existing_path}\n"
                f"Existing invariants: {json.dumps(existing_invariants, indent=2, sort_keys=True)}\n"
                f"Current invariants: {json.dumps(current_invariants, indent=2, sort_keys=True)}"
            )
        existing_soft = _soft_invariants(existing_echo)
        if existing_soft != current_soft:
            print(
                "[VLSP-EVAL-AUDIT][WARN] resuming into a results dir written by a "
                f"different code version: existing {existing_soft} vs current "
                f"{current_soft} ({existing_path}). Success numbers would mix "
                "sampler versions; start a fresh dir if that matters."
            )

    echo_path = rollout_dir / f"config_echo.rank{eval_rank}.json"
    _atomic_write_json(echo_path, echo)
    print(f"[VLSP-EVAL-AUDIT] {json.dumps(current_invariants, sort_keys=True)}")


def _write_rank_summary(
    *,
    rollout_dir: pathlib.Path,
    eval_rank: int,
    eval_world_size: int,
    task_suite_name: str,
    num_tasks: int,
    num_trials_per_task: int,
    episodes: dict[int, dict[str, Any]],
) -> None:
    per_task: dict[str, dict[str, Any]] = {}
    for episode in episodes.values():
        task_id = str(episode["task_id"])
        bucket = per_task.setdefault(
            task_id,
            {
                "task_id": episode["task_id"],
                "task": episode["task"],
                "episodes": 0,
                "successes": 0,
            },
        )
        bucket["episodes"] += 1
        bucket["successes"] += int(bool(episode["success"]))
    for bucket in per_task.values():
        bucket["success_rate"] = bucket["successes"] / max(bucket["episodes"], 1)

    successes = sum(int(bool(ep["success"])) for ep in episodes.values())
    summary = {
        "schema_version": 1,
        "rank": eval_rank,
        "world_size": eval_world_size,
        "task_suite_name": task_suite_name,
        "num_tasks": num_tasks,
        "num_trials_per_task": num_trials_per_task,
        "expected_total_episodes": num_tasks * num_trials_per_task,
        "episodes": {str(k): v for k, v in sorted(episodes.items())},
        "total_episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / max(len(episodes), 1),
        "per_task": per_task,
    }
    summary_path = rollout_dir / f"summary.rank{eval_rank}.json"
    _atomic_write_json(summary_path, summary)


def set_seed_everywhere(seed: int) -> None:
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_video2world2action_pipeline(
    experiment_name: str,
    video_model_path: str,
    action_model_path: str,
    dataset_statistics_path: pathlib.Path,
    dtype: torch.dtype = torch.bfloat16,
) -> Video2World2ActionPipeline:
    """Instantiate the video-to-world-to-action pipeline and load normalizer statistics."""
    config = make_config()
    config = override(config, ["--", f"experiment={experiment_name}"])

    # all libero task descriptions have been verified to be unproblematic
    config.model.config.video_pipe_config.guardrail_config.enabled = False

    video2world_pipe = Video2WorldPipeline.from_config(
        config=config.model.config.video_pipe_config,
        dit_path=video_model_path,
        device="cuda",
        torch_dtype=dtype,
        load_ema_to_reg=False,
    )

    world2action_pipe = World2ActionPipeline.from_config(
        config.model.config.pipe_config,
        dit_path=action_model_path,
        device="cuda",
        dtype=dtype,
    )

    data_config = instantiate(config.data_config)

    with dataset_statistics_path.open("rb") as stats_file:
        stats = json.load(stats_file)
    world2action_pipe.normalizer.build_from_stats(
        stats,
        normalization_types=extract_normalization_types(data_config.policy_io.policy_io),
        concat_groups=data_config.policy_io.concat_groups,
        device="cuda",
        dtype=dtype,
    )

    return Video2World2ActionPipeline(video2world_pipe, world2action_pipe).cuda()


class VAMInference:
    """Helper class that maintains temporal context and queries the VAM policy."""

    def __init__(
        self,
        experiment_name: str,
        video_model_path: str,
        action_model_path: str,
        dataset_statistics_path: pathlib.Path,
        img_horizon: int,
        lowdim_horizon: int,
        stop_video_denoising_step: int,
        num_execute_actions: int,
        rollout_dir: pathlib.Path,
    ):
        self.model = load_video2world2action_pipeline(
            experiment_name,
            video_model_path,
            action_model_path,
            dataset_statistics_path,
        )
        self._image_horizon = img_horizon
        self._lowdim_horizon = lowdim_horizon
        self.stop_video_denoising_step = stop_video_denoising_step
        self.num_execute_actions = num_execute_actions
        self.num_sampling_steps = 35
        self.rollout_dir = rollout_dir
        self.reset(task_description="")

    def reset(self, task_description: str) -> None:
        """Reset internal state for a new task/episode."""
        self.task_description = task_description
        self._image_history: deque[np.ndarray] = deque(maxlen=(self._image_horizon - 1) * 4 + 1)
        self._lowdim_history: deque[np.ndarray] = deque(maxlen=self._lowdim_horizon)
        self.action_buffer: np.ndarray | None = None
        self.action_buffer_idx = 0
        self._execute_horizon = 0

    def step(
        self,
        image: np.ndarray,
        task_description: str,
        obs: dict,
    ) -> np.ndarray:
        """Return the next action for the given observation."""
        if image.dtype != np.uint8:
            raise ValueError(f"Expected image dtype uint8, received {image.dtype}.")

        if task_description != self.task_description:
            self.reset(task_description)

        processed_image = self._process_image(image)
        self._add_image_to_history(processed_image)

        state_vec = self._state_from_observation(obs)
        self._add_lowdim_to_history(state_vec)

        if self.action_buffer is None:
            self._query_policy(task_description)

        current_action = self.action_buffer[self.action_buffer_idx]
        self.action_buffer_idx += 1
        if self.action_buffer_idx >= self._execute_horizon:
            self.action_buffer = None

        return self._convert_action(current_action)

    def _query_policy(self, task_description: str) -> None:
        """Query the model and cache the planned action sequence."""
        images = np.concatenate(list(self._image_history)[::4], axis=1)  # downsample from 20 fps to 5
        lowdims = np.stack(list(self._lowdim_history), axis=0)

        input_vid = torch.from_numpy(images[None]).cuda().to(dtype=torch.bfloat16)
        state_tensor = torch.from_numpy(lowdims[None]).cuda().to(dtype=torch.bfloat16)

        with torch.no_grad():
            pred_actions = self.model(
                input_vid=input_vid,
                state_B_HO_O=state_tensor,
                prompt=task_description,
                num_sampling_step=self.num_sampling_steps,
                stop_after_step=self.stop_video_denoising_step,
                use_cuda_graphs=True,
            )

        self.action_buffer = pred_actions[0].float().cpu().numpy()
        self._execute_horizon = self.num_execute_actions
        self.action_buffer_idx = 0

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        tensor = rearrange(image, "h w c -> c h w")[:, None, :, :]
        return 2.0 * (tensor.astype(np.float32) / 255.0 - 0.5)

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self._image_history.append(image)
        while len(self._image_history) < self._image_history.maxlen:
            self._image_history.append(image.copy())

    def _add_lowdim_to_history(self, lowdim: np.ndarray) -> None:
        self._lowdim_history.append(lowdim)
        while len(self._lowdim_history) < self._lowdim_horizon:
            self._lowdim_history.append(lowdim.copy())

    @staticmethod
    def _state_from_observation(obs: dict[str, np.ndarray]) -> np.ndarray:
        rot_6d = Rotation.from_quat(obs["robot0_eef_quat"]).as_matrix()[:2].reshape((6,))
        return np.concatenate((obs["robot0_eef_pos"], rot_6d, obs["robot0_gripper_qpos"][0][None]), axis=0)

    @staticmethod
    def _matrix_from_6d(orient6: np.ndarray) -> np.ndarray:
        r1 = orient6[:3]
        r2 = orient6[3:]
        r1_norm = r1 / (np.linalg.norm(r1) + 1e-9)
        r2_orth = r2 - np.dot(r2, r1_norm) * r1_norm
        r2_norm = r2_orth / (np.linalg.norm(r2_orth) + 1e-9)
        r3 = np.cross(r1_norm, r2_norm)
        return np.stack([r1_norm, r2_norm, r3], axis=0)

    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        delta_pos = action[:3]
        rot_matrix = self._matrix_from_6d(action[3:9])
        rot_vec = Rotation.from_matrix(rot_matrix).as_rotvec()
        gripper = np.sign(action[9][None])
        return np.concatenate([delta_pos, rot_vec, gripper], axis=0)


def get_libero_env(task) -> tuple[OffScreenRenderEnv, str]:
    """Initializes and returns the LIBERO environment alongside the task description."""
    task_description = task.language.replace("black bowl", "bowl")
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": CAMERA_HEIGHT,
        "camera_widths": CAMERA_WIDTH,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def get_libero_image(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Extract the agentview image and check that it matches the expected resolution."""
    image = obs["agentview_image"][::-1, ::-1]
    if image.shape != (CAMERA_HEIGHT, CAMERA_WIDTH, 3):
        raise ValueError(f"Unexpected agentview image shape {image.shape}.")
    return image


def save_rollout_video(
    rollout_images: Iterable[np.ndarray],
    idx: int,
    success: bool,
    task_description: str,
    rollout_dir: Path,
) -> Path:
    """Save an MP4 replay of the episode."""
    rollout_dir.mkdir(parents=True, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")
    mp4_path = rollout_dir / f"episode{idx}_{'success' if success else 'failure'}_{processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=20)
    try:
        for img in rollout_images:
            video_writer.append_data(img)
    finally:
        video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path


def run_episode(
    env: OffScreenRenderEnv,
    policy: VAMInference,
    task_description: str,
    initial_observation: dict[str, np.ndarray],
    max_steps: int,
    num_steps_wait: int,
) -> tuple[bool, list[np.ndarray]]:
    """Execute a single episode and return success flag along with captured frames."""
    obs = initial_observation
    replay_images: list[np.ndarray] = []
    success = False

    for step_idx in range(max_steps + num_steps_wait):
        if step_idx < num_steps_wait:
            obs, _, done, info = env.step(DUMMY_ACTION)
            if done:
                success = True
                break
            continue

        image = get_libero_image(obs)
        replay_images.append(image)

        action = policy.step(image, task_description, obs)

        obs, _, done, info = env.step(action.tolist())
        if done:
            success = True
            break

    return success, replay_images


def eval_vam_libero(
    vam_experiment_name: str,
    vam_video_model_path: str,
    vam_action_model_path: pathlib.Path,
    vam_dataset_statistics_path: pathlib.Path,
    vam_img_horizon: int,
    vam_lowdim_horizon: int,
    vam_stop_video_denoising_step: int,
    vam_num_execute_actions: int,
    task_suite_name: str,
    num_trials_per_task: int = 50,
    eval_rank: int = 0,
    eval_world_size: int = 1,
    num_steps_wait: int = 10,
    seed: int = 0,
) -> None:
    set_seed_everywhere(seed)

    run_label = _make_run_label(
        vam_experiment_name=vam_experiment_name,
        vam_action_model_path=vam_action_model_path,
        vam_stop_video_denoising_step=vam_stop_video_denoising_step,
        vam_num_execute_actions=vam_num_execute_actions,
        seed=seed,
    )
    rollout_dir = Path("./results") / run_label / task_suite_name
    rollout_dir.mkdir(parents=True, exist_ok=True)

    cli_args = {
        "vam_experiment_name": vam_experiment_name,
        "vam_video_model_path": vam_video_model_path,
        "vam_action_model_path": str(vam_action_model_path),
        "vam_dataset_statistics_path": str(vam_dataset_statistics_path),
        "vam_img_horizon": vam_img_horizon,
        "vam_lowdim_horizon": vam_lowdim_horizon,
        "vam_stop_video_denoising_step": vam_stop_video_denoising_step,
        "vam_num_execute_actions": vam_num_execute_actions,
        "task_suite_name": task_suite_name,
        "num_trials_per_task": num_trials_per_task,
        "eval_rank": eval_rank,
        "eval_world_size": eval_world_size,
        "num_steps_wait": num_steps_wait,
        "seed": seed,
    }
    # Fail-fast BEFORE the (minutes-long) model load: catch a wrong experiment /
    # ckpt / stop step / seed pointed at a populated results dir immediately.
    _precheck_cli_invariants(rollout_dir, cli_args)

    policy = VAMInference(
        vam_experiment_name,
        vam_video_model_path,
        str(vam_action_model_path),
        vam_dataset_statistics_path,
        vam_img_horizon,
        vam_lowdim_horizon,
        vam_stop_video_denoising_step,
        vam_num_execute_actions,
        rollout_dir,
    )
    # Full guard now that the resolved config is available (adds checkpoint
    # sha256 + source/condition mode + scheduler steps to the pre-check above).
    _write_config_echo(
        rollout_dir=rollout_dir,
        eval_rank=eval_rank,
        cli_args=cli_args,
        policy=policy,
    )
    existing_rollouts = _index_existing_rollouts(rollout_dir)

    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        raise ValueError(f"Task suite {task_suite_name} not available.")
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks

    max_steps = LIBERO_SUITE_MAX_STEPS[task_suite_name]

    global_episode_id = 0
    rank_episodes = 0
    rank_successes = 0
    rank_episode_results: dict[int, dict[str, Any]] = {}

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task)

        if len(initial_states) == 0:
            raise ValueError(f"No initial states provided for task {task_id}.")

        task_successes = 0
        task_episodes = 0

        try:
            for episode_idx in tqdm.tqdm(range(num_trials_per_task), desc="Episodes", leave=False):
                global_episode_id += 1

                if global_episode_id % eval_world_size != eval_rank:
                    continue
                task_episodes += 1
                rank_episodes += 1

                existing_rollout = existing_rollouts.get(global_episode_id)
                if existing_rollout is not None:
                    _, success = existing_rollout
                    task_successes += int(success)
                    rank_successes += int(success)
                    rank_episode_results[global_episode_id] = {
                        "episode_id": global_episode_id,
                        "task_id": task_id,
                        "task": task_description,
                        "trial_idx": episode_idx,
                        "success": bool(success),
                        "resumed_from_disk": True,
                    }
                    continue

                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

                policy.reset(task_description)

                success, replay_images = run_episode(
                    env,
                    policy,
                    task_description,
                    obs,
                    max_steps,
                    num_steps_wait,
                )

                if success:
                    task_successes += 1
                    rank_successes += 1

                mp4_path = save_rollout_video(
                    replay_images,
                    global_episode_id,
                    success,
                    task_description,
                    rollout_dir,
                )
                existing_rollouts[global_episode_id] = (mp4_path, success)
                rank_episode_results[global_episode_id] = {
                    "episode_id": global_episode_id,
                    "task_id": task_id,
                    "task": task_description,
                    "trial_idx": episode_idx,
                    "success": bool(success),
                    "resumed_from_disk": False,
                }

                success_rate = rank_successes / max(rank_episodes, 1)
                print(
                    f"Task {task_id} | Episode {episode_idx + 1} | Success: {success} "
                    f"| Rank Success Rate: {success_rate:.3f}\n"
                )
        finally:
            env.close()

        task_success_rate = task_successes / max(task_episodes, 1)
        print(f"Task {task_id} success rate: {task_success_rate:.3f}")

    _write_rank_summary(
        rollout_dir=rollout_dir,
        eval_rank=eval_rank,
        eval_world_size=eval_world_size,
        task_suite_name=task_suite_name,
        num_tasks=num_tasks,
        num_trials_per_task=num_trials_per_task,
        episodes=rank_episode_results,
    )

    overall_success_rate = rank_successes / max(rank_episodes, 1)
    print(
        f"Completed {rank_episodes} rank-assigned episodes | "
        f"Total successes: {rank_successes} | "
        f"Rank success rate: {overall_success_rate:.3f}\n"
    )


if __name__ == "__main__":
    tyro.cli(eval_vam_libero)
