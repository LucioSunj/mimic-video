"""Aggregate sharded LIBERO eval summaries written by run.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_rank_summaries(results_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(results_dir.glob("summary.rank*.json")):
        with path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["_path"] = str(path)
        summaries.append(summary)
    if not summaries:
        raise FileNotFoundError(f"No summary.rank*.json files found in {results_dir}")
    return summaries


def aggregate(results_dir: Path) -> dict[str, Any]:
    rank_summaries = _load_rank_summaries(results_dir)
    episodes: dict[str, dict[str, Any]] = {}
    duplicate_episode_ids: list[str] = []

    for rank_summary in rank_summaries:
        for episode_id, episode in rank_summary["episodes"].items():
            if episode_id in episodes:
                duplicate_episode_ids.append(episode_id)
                continue
            episodes[episode_id] = episode

    if duplicate_episode_ids:
        dupes = ", ".join(sorted(duplicate_episode_ids, key=int)[:20])
        raise RuntimeError(f"Duplicate episode ids across rank summaries: {dupes}")

    expected_total = max(int(s.get("expected_total_episodes", 0)) for s in rank_summaries)
    if expected_total <= 0:
        expected_total = max((int(k) for k in episodes), default=0)
    missing_episode_ids = [idx for idx in range(1, expected_total + 1) if str(idx) not in episodes]

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
    return {
        "schema_version": 1,
        "results_dir": str(results_dir),
        "rank_summary_files": [s["_path"] for s in rank_summaries],
        "ranks_present": sorted(int(s["rank"]) for s in rank_summaries),
        "world_size": max(int(s["world_size"]) for s in rank_summaries),
        "task_suite_name": rank_summaries[0]["task_suite_name"],
        "expected_total_episodes": expected_total,
        "total_episodes": len(episodes),
        "missing_episode_ids": missing_episode_ids,
        "successes": successes,
        "success_rate": successes / max(len(episodes), 1),
        "per_task": per_task,
        "episodes": dict(sorted(episodes.items(), key=lambda item: int(item[0]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path, help="Path to results/<run_label>/<suite>")
    args = parser.parse_args()

    summary = aggregate(args.dir)
    output_path = args.dir / "summary.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Wrote {output_path}")
    print(
        "success_rate={:.3f} successes={} episodes={} expected={} missing={}".format(
            summary["success_rate"],
            summary["successes"],
            summary["total_episodes"],
            summary["expected_total_episodes"],
            len(summary["missing_episode_ids"]),
        )
    )


if __name__ == "__main__":
    main()
