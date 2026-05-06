"""Example: Orchestration entry points (replaces old ScenarioManager demo)

The historical ``powerzoo.scenarios`` package (``make`` / ``register`` /
``list_scenarios``) has been removed. Use:

  - ``PowerEnv(config)`` for a custom grid + resources + reward dict, or
  - ``make_task_env(name)`` for registered benchmark tasks.

This script only **prints** task names and builds a minimal ``PowerEnv`` to
show that the orchestration façade imports cleanly.
"""

from __future__ import annotations

from powerzoo.envs.power_env import PowerEnv
from powerzoo.tasks import list_tasks, make_task_env

print("=" * 80)
print("PowerZoo — orchestration demo (no scenario registry)")
print("=" * 80)

print("\nRegistered benchmark tasks:")
for t in list_tasks():
    print(f"  - {t}")

minimal_config = {
    "name": "demo_transmission",
    "grid": {
        "type": "transmission",
        "case": "Case5",
        "start_date": "2024-01-01",
        "end_date": "2024-01-02",
        "delta_t_minutes": 30,
        "max_load_ratio": 0.9,
    },
    "resources": [],
    "reward": {"type": "economic_dispatch", "cost_weight": 1.0, "safety_weight": 0.5},
    "episode": {"max_steps": 8},
}

env = PowerEnv(minimal_config)
obs, info = env.reset(seed=0)
print(f"\nPowerEnv OK: obs keys = {list(obs.keys())}")

# One task env (sanity)
te = make_task_env("marl_opf", split="train")
print(f"make_task_env('marl_opf') -> {type(te).__name__}")

print("\nDone. See docs/en/getting-started.md and docs/en/api/orchestration.md.")
