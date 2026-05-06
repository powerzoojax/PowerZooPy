"""Compute and store baseline scores for all registered PowerZoo tasks.

Run as a module::

    python -m powerzoo.benchmarks.compute [--tasks marl_opf marl_der_arbitrage]
                                          [--n_episodes 100]
                                          [--split test]
                                          [--seed 42]

Results are saved to ``powerzoo/benchmarks/baselines.json`` and printed to
stdout in a table format suitable for pasting into papers.

Why a separate script?
-----------------------
Baseline values depend on the *exact* eval protocol (n_episodes, seed, split,
solver version).  Shipping a script that researchers can re-run guarantees
reproducibility and keeps the env code free of hardcoded numbers.

The JSON file that ships with the library was generated with this script on
the reference hardware.  If you modify environments, re-run this script and
commit the updated baselines.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

# ---------------------------------------------------------------------------
# Avoid circular import: register envs before importing anything else
# ---------------------------------------------------------------------------
import powerzoo  # noqa: F401 — triggers gym registration


def _compute_single(
    task_id: str,
    n_episodes: int,
    split: str,
    seed: int,
    verbose: bool,
) -> dict:
    """Return {'random': float, 'oracle': float} for one task."""
    from powerzoo.tasks import make_task_env
    from powerzoo.benchmarks.evaluation import evaluate
    from powerzoo.benchmarks.policies.random_policy import RandomPolicy
    from powerzoo.benchmarks.policies.oracle import OraclePolicy
    from powerzoo.wrappers.gym_wrappers import GymnasiumWrapper

    print(f"\n  [{task_id}]  split={split}  n_episodes={n_episodes}")

    # ---- Random policy ----
    t0 = time.perf_counter()
    env_r = make_task_env(task_id, split=split)
    if not hasattr(env_r, "observation_space"):
        env_r = GymnasiumWrapper(env_r)
    rand_result = evaluate(
        RandomPolicy(env_r.action_space),
        env_r,
        n_episodes=n_episodes,
        seed_start=seed,
        verbose=verbose,
    )
    t_rand = time.perf_counter() - t0
    env_r.close()

    print(
        f"    RandomPolicy : mean={rand_result['mean_reward']:9.3f}  "
        f"std={rand_result['std_reward']:7.3f}  ({t_rand:.1f}s)"
    )

    # ---- Oracle policy ----
    t0 = time.perf_counter()
    env_o = make_task_env(task_id, split=split)
    if not hasattr(env_o, "observation_space"):
        env_o = GymnasiumWrapper(env_o)
    oracle_result = evaluate(
        OraclePolicy(env_o),
        env_o,
        n_episodes=n_episodes,
        seed_start=seed,
        verbose=verbose,
    )
    t_oracle = time.perf_counter() - t0
    env_o.close()

    print(
        f"    OraclePolicy : mean={oracle_result['mean_reward']:9.3f}  "
        f"std={oracle_result['std_reward']:7.3f}  ({t_oracle:.1f}s)"
    )

    return {
        "random": round(rand_result["mean_reward"], 4),
        "oracle": round(oracle_result["mean_reward"], 4),
    }


def main(
    tasks: Optional[List[str]] = None,
    n_episodes: int = 100,
    split: str = "test",
    seed: int = 42,
    verbose: bool = False,
) -> None:
    from powerzoo.tasks import list_tasks
    from powerzoo.benchmarks.scores import register_baseline, list_baselines

    if tasks is None:
        tasks = list_tasks()

    print("=" * 70)
    print("  PowerZoo Baseline Computation")
    print(f"  tasks={tasks}  n_episodes={n_episodes}  split={split}  seed={seed}")
    print("=" * 70)

    results = {}
    failed = []

    for task_id in tasks:
        try:
            bl = _compute_single(task_id, n_episodes, split, seed, verbose)
            results[task_id] = bl
            register_baseline(task_id, bl["random"], bl["oracle"], save=False)
        except Exception as exc:
            print(f"  ERROR computing '{task_id}': {exc}")
            failed.append(task_id)

    # Save once at the end
    from powerzoo.benchmarks.scores import _save_baselines, _get_baselines
    _save_baselines(_get_baselines())

    # Pretty-print summary table
    print("\n" + "=" * 70)
    print(f"  {'Task':<30} {'Random':>12} {'Oracle':>12} {'Gap':>10}")
    print("-" * 70)
    for task_id, bl in sorted(results.items()):
        gap = bl["oracle"] - bl["random"]
        print(
            f"  {task_id:<30} {bl['random']:>12.3f} {bl['oracle']:>12.3f} {gap:>10.3f}"
        )

    if failed:
        print(f"\n  FAILED tasks: {failed}")

    print("\n  Baselines saved to powerzoo/benchmarks/baselines.json")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute PowerZoo RandomPolicy / OraclePolicy baselines."
    )
    parser.add_argument(
        "--tasks", nargs="*", default=None,
        help="Task IDs to compute (default: all registered tasks)",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=100,
        help="Episodes per policy per task (default: 100)",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Data split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Starting random seed (default: 42)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-episode rewards",
    )
    args = parser.parse_args()

    main(
        tasks=args.tasks,
        n_episodes=args.n_episodes,
        split=args.split,
        seed=args.seed,
        verbose=args.verbose,
    )
