"""Example: Multi-Agent OPF — minimal episode loop

Uses the default task adapter (no PettingZoo dependency). For
``framework='pettingzoo'``, install ``pettingzoo`` (see optional ``rl`` extra).

RLlib training: ``MARL02_opf_rllib.py`` or ``MARL04_opf_task_demo.py``.
"""

from __future__ import annotations

from powerzoo.tasks import make_task_env, list_tasks

print("=" * 80)
print("Multi-Agent OPF — minimal loop (__all__ termination)")
print("=" * 80)

print("\nRegistered tasks (sample):")
for name in list_tasks()[:12]:
    print(f"  - {name}")
if len(list_tasks()) > 12:
    print("  ...")

env = make_task_env("marl_opf", split="train")
print(f"\nEnv type: {type(env).__name__}")

obs, infos = env.reset(seed=0)
terminated = truncated = False
ep_reward = 0.0
steps = 0
while not (terminated or truncated):
    actions = {a: env.action_space[a].sample() for a in env.possible_agents}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)
    ep_reward += float(sum(rewards.values()))
    steps += 1
    terminated = bool(terminateds.get("__all__", False))
    truncated = bool(truncateds.get("__all__", False))

print(f"\nEpisode finished in {steps} steps")
print(f"  Sum of per-agent step rewards: {ep_reward:.4f}")

print("\nNext: MARL04_opf_task_demo.py (task-based RLlib) or docs Getting Started.")
