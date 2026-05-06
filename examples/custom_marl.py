"""custom_marl.py — multi-agent environment creation with make_env().

Shows three patterns for multi-agent tasks:

1. PettingZoo interface (plug into EPyMARL / MAPPO / custom loop)
2. Independent-learners training via Trainer.train_il()
3. Anonymous MARL config dict

Usage
-----
    python examples/custom_marl.py

Requirements: pip install powerzoo[rl,marl]
"""

from __future__ import annotations

from powerzoo.rl import make_env, info


# ── Pattern 1: standard PettingZoo env ───────────────────────────────────────
# Returns a PettingZoo ParallelEnv; plug into EPyMARL, MAPPO, or any
# PettingZoo-compatible framework.

print("=== Pattern 1: PettingZoo interface ===")
env = make_env('marl_opf', framework='pettingzoo', split='train')
obs_dict, _ = env.reset(seed=42)
agents = env.agents
print(f"Agents: {agents}")
print(f"Obs space (first agent): {env.observation_space(agents[0])}")

# Standard PettingZoo interaction loop
actions = {a: env.action_space(a).sample() for a in agents}
obs_dict, rewards, terms, truncs, infos = env.step(actions)
print(f"Reward sample: {dict(list(rewards.items())[:2])}")


# ── Pattern 2: inspect a task before creating an env ─────────────────────────

print("\n=== Pattern 2: task info ===")
d = info('marl_opf')
print(f"Task:       {d['task_id']}")
print(f"Agent mode: {d['agent_mode']}")
print(f"Difficulty: {d['difficulty']}")
if d.get('observation_space'):
    print(f"Obs shape:  {d['observation_space'].get('shape')}")


# ── Pattern 3: anonymous MARL config dict ────────────────────────────────────
# Construct a multi-agent env directly from a dict — no Task registration
# required.  The agent_type is inferred from the resource list.

print("\n=== Pattern 3: anonymous MARL config ===")
config = {
    'grid': {'type': 'transmission', 'case': 'case5'},
    'resources': [],
    'agents': {
        'agent_type': 'unit',
        'reward_type': 'shared',
    },
    'episode': {'max_steps': 24},
}
anon_env = make_env(config, framework='pettingzoo')
obs_dict, _ = anon_env.reset(seed=0)
print(f"Anonymous MARL agents: {anon_env.agents}")
