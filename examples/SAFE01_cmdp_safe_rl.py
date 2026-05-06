"""Safe RL / CMDP End-to-End Example for PowerZoo

This script demonstrates three CMDP integration paths with PowerZoo:

  A. Manual CMDP loop   — pure Python, no extra dependency.
                          Shows vector/selected cost signals and cumulative
                          episode cost tracking.

  B. OmniSafe           — drop-in integration via SafeRLWrapper (6-tuple API).
                          Requires: pip install omnisafe

  C. RLlib metric tracking — MARL reward-only baseline with CMDP cost metrics.
                             This is not a constrained optimizer.
                             Requires: pip install ray[rllib]

All three use the same task — marl_opf (IEEE 5-bus OPF) — so you can
compare reward-only behaviour and CMDP cost diagnostics on the same environment.

Constraint budget (cost_threshold)
-----------------------------------
  The benchmark defines cost_threshold = 10.0 (MW·step of thermal overload
  per episode).  Episodes where cumulative cost > 10.0 are "unsafe".
  Safe RL methods minimise reward subject to J_C ≤ cost_threshold.

Usage
-----
  python SAFE01_cmdp_safe_rl.py              # Part A only (no deps needed)
  python SAFE01_cmdp_safe_rl.py --omnisafe   # Part A + B (needs omnisafe)
  python SAFE01_cmdp_safe_rl.py --rllib      # Part A + C (needs ray[rllib])
  python SAFE01_cmdp_safe_rl.py --all        # All three parts
"""

import sys
import argparse
import numpy as np
from typing import Dict, List, Tuple


def _scalar_cost_from_info(info: Dict) -> float:
    """Project vector/named CMDP costs to a scalar for legacy metric tracking."""
    if 'selected_constraint_costs' in info:
        return float(np.sum(np.asarray(info['selected_constraint_costs'], dtype=float)))
    if 'constraint_costs' in info:
        return float(np.sum(np.asarray(info['constraint_costs'], dtype=float)))
    costs = info.get('costs')
    if isinstance(costs, dict):
        return float(sum(max(0.0, float(v)) for v in costs.values()))
    if costs is not None:
        return float(np.sum(np.asarray(costs, dtype=float)))
    if 'cost_sum' in info:
        return float(info['cost_sum'])
    return float(info.get('cost', 0.0))


# ============================================================================
# A.  Manual CMDP loop — no extra dependencies
# ============================================================================


def part_a_manual_cmdp_loop(n_episodes: int = 5, max_steps: int = 48):
    """Run a random-policy evaluation loop that tracks CMDP cost signals.

    This is the minimal boilerplate required to work with PowerZoo's CMDP
    interface.  Swap out the random action for your own policy.
    """
    from powerzoo.tasks import make_task_env

    print("=" * 60)
    print("Part A: Manual CMDP loop (random policy)")
    print("=" * 60)

    # Create the multi-agent OPF environment (standard split)
    env = make_task_env('marl_opf', split='test', max_steps=max_steps)

    # The per-episode cost budget from the task's eval_protocol
    # (access via: from powerzoo.tasks import make_task; t = make_task('marl_opf'))
    cost_threshold = 10.0

    ep_rewards: List[float] = []
    ep_costs:   List[float] = []
    n_safe = 0

    for ep in range(n_episodes):
        obs, _infos = env.reset(seed=ep * 7 + 42)
        ep_reward = 0.0
        ep_cost   = 0.0

        for _step in range(max_steps):
            # Random policy — replace with your own
            actions = {
                agent: env.action_space[agent].sample()
                for agent in env.get_agent_ids()
            }

            obs, rewards, terminateds, truncateds, infos = env.step(actions)

            # ── CMDP cost extraction ─────────────────────────────────────
            # Prefer selected_constraint_costs / constraint_costs.  For legacy
            # envs this helper falls back to named cost dicts or scalar aliases.
            step_cost = 0.0
            for agent, info in infos.items():
                if agent == '__all__':
                    continue
                step_cost += _scalar_cost_from_info(info)

            # You can also get per-component breakdown:
            #   info['costs'] = {'thermal_overload': x, 'voltage_violation': y}
            # e.g.:
            #   for agent, info in infos.items():
            #       comps = info.get('costs', {})
            #       thermal = comps.get('thermal_overload', 0.0)
            #       voltage = comps.get('voltage_violation', 0.0)

            ep_cost   += step_cost
            ep_reward += np.mean(list(rewards.values()))

            if terminateds.get('__all__') or truncateds.get('__all__'):
                break

        ep_rewards.append(ep_reward)
        ep_costs.append(ep_cost)
        safe = ep_cost <= cost_threshold
        n_safe += int(safe)

        print(f"  Episode {ep+1:2d}:  reward={ep_reward:7.3f}  "
              f"ep_cost={ep_cost:6.3f}  "
              f"{'SAFE' if safe else 'UNSAFE'}")

    print()
    print(f"  Mean reward :  {np.mean(ep_rewards):.4f}")
    print(f"  Mean ep_cost:  {np.mean(ep_costs):.4f}  (threshold={cost_threshold})")
    print(f"  Safe rate   :  {n_safe}/{n_episodes} = {n_safe/n_episodes:.0%}")
    print()

    return {
        'mean_reward':        float(np.mean(ep_rewards)),
        'mean_ep_cost':       float(np.mean(ep_costs)),
        'cost_violation_rate': 1.0 - n_safe / n_episodes,
    }


# ============================================================================
# B.  OmniSafe — single-agent wrapper + PPO-Lagrangian
# ============================================================================


def _make_single_agent_safe_env(cost_threshold: float = 10.0):
    """Wrap the single-agent OPF env for OmniSafe / Safety-Gymnasium.

    Returns a SafeRLWrapper that emits the 6-tuple
    (obs, reward, cost, terminated, truncated, info).
    """
    from powerzoo.tasks import make_task
    from powerzoo.tasks.adapters import create_task_env
    from powerzoo.wrappers.safe_rl_wrapper import SafeRLWrapper

    # Use the single-agent variant so OmniSafe sees a standard Gymnasium env
    task = make_task('marl_opf', split='train', max_steps=48)
    inner_env = task.create_single_agent_env()

    return SafeRLWrapper(inner_env, cost_threshold=cost_threshold)


def part_b_omnisafe(n_epochs: int = 10):
    """Train PPO-Lagrangian using OmniSafe on the single-agent OPF env.

    OmniSafe natively consumes the Safety-Gymnasium 6-tuple API, so the
    only change vs. a plain gymnasium env is the SafeRLWrapper.

    Install: pip install omnisafe
    """
    try:
        import omnisafe
    except ImportError:
        print("[Part B] OmniSafe not installed.  Run: pip install omnisafe")
        return None

    import gymnasium as gym

    print("=" * 60)
    print("Part B: OmniSafe PPO-Lagrangian")
    print("=" * 60)

    # OmniSafe needs to find the env via gymnasium.make or a custom env_id.
    # The cleanest way is to register a gym env and pass its id.
    env_id = 'PowerZoo-SafeOPF-v0'

    gym.register(
        id=env_id,
        entry_point=lambda: _make_single_agent_safe_env(cost_threshold=10.0),
    )

    cfgs = {
        'train_cfgs': {
            'total_steps': n_epochs * 2048,
            'device': 'cpu',
        },
        'algo_cfgs': {
            'steps_per_epoch': 2048,
            'update_iters': 10,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'use_tensorboard': False,
        },
        'model_cfgs': {
            'actor': {'hidden_sizes': [64, 64]},
            'critic': {'hidden_sizes': [64, 64]},
        },
    }

    agent = omnisafe.Agent('PPOLag', env_id, custom_cfgs=cfgs)
    agent.learn()

    # Evaluate and print cost-constrained metrics
    agent.plot(smooth=1)
    agent.render(num_episodes=2, render_mode='rgb_array', width=256, height=256)
    eval_result = agent.evaluate(num_episodes=10)
    print(f"\n  OmniSafe evaluation: {eval_result}")
    return eval_result


# ============================================================================
# C.  RLlib reward-only baseline with CMDP metrics (MARL, multi-agent)
# ============================================================================


def part_c_rllib_ppo_lag(n_iterations: int = 20, max_steps: int = 48):
    """Multi-agent PPO baseline with CMDP cost metrics.

    Install: pip install 'ray[rllib]'
    """
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.tune.registry import register_env
    except ImportError:
        print("[Part C] Ray RLlib not installed.  Run: pip install 'ray[rllib]'")
        return None

    from powerzoo.tasks import make_task_env

    print("=" * 60)
    print("Part C: RLlib PPO reward-only baseline + CMDP metrics (MARL)")
    print("=" * 60)

    cost_threshold = 10.0   # d  — CMDP budget

    register_env('safe_marl_opf', lambda cfg: make_task_env('marl_opf', **cfg))

    test_env = make_task_env('marl_opf', max_steps=max_steps)
    agents   = test_env.possible_agents
    obs_sp   = test_env.observation_space
    act_sp   = test_env.action_space

    ray.init(ignore_reinit_error=True, log_to_driver=False)

    try:
        config = (
            PPOConfig()
            .environment(
                env='safe_marl_opf',
                env_config={'max_steps': max_steps},
            )
            .framework('torch')
            .env_runners(num_env_runners=0)
            .training(
                lr=3e-4,
                gamma=0.99,
                train_batch_size=512,
                model={'fcnet_hiddens': [64, 64]},
                # PPO remains reward-only here; CMDP costs are logged below.
            )
            .multi_agent(
                policies={
                    a: (None, obs_sp[a], act_sp[a], {}) for a in agents
                },
                policy_mapping_fn=lambda aid, *_, **__: aid,
            )
        )

        algo = config.build()

        print(f"  Cost threshold: {cost_threshold}  |  optimizer: reward-only")
        print(f"  {'Iter':>4}  {'Mean reward':>12}  "
              f"{'Mean J_C':>10}  Status")
        print("  " + "-" * 44)

        for i in range(n_iterations):
            result = algo.train()

            # ── Extract episodic cost from training result ───────────────
            # RLlib stores custom_metrics in env_runners if the env places
            # them there.  As a fallback, run a short rollout manually.
            ep_cost = _estimate_episode_cost(algo, agents, max_steps)

            env_runners = result.get('env_runners', {})
            mean_reward  = env_runners.get('episode_return_mean',
                           env_runners.get('episode_reward_mean', 0.0))
            safe = ep_cost <= cost_threshold

            if (i + 1) % max(1, n_iterations // 5) == 0 or i == 0:
                print(f"  {i+1:4d}  {mean_reward:12.4f}  "
                      f"{ep_cost:10.4f}  "
                      f"{'SAFE' if safe else 'UNSAFE'}")

        algo.stop()

    finally:
        ray.shutdown()

    print()
    return {'training_contract': 'cmdp_env_plus_reward_only_baseline'}


def _estimate_episode_cost(
    algo,
    agents: List[str],
    max_steps: int,
) -> float:
    """Run one evaluation episode and return the summed CMDP cost."""
    from powerzoo.tasks import make_task_env
    import torch

    env = make_task_env('marl_opf', split='test', max_steps=max_steps)
    obs, _ = env.reset(seed=0)
    ep_cost = 0.0

    for _ in range(max_steps):
        actions = {}
        for agent in agents:
            if agent not in obs:
                continue
            try:
                policy = algo.get_policy(agent)
                action, _, _ = policy.compute_single_action(obs[agent])
                actions[agent] = action
            except Exception:
                actions[agent] = env.action_space[agent].sample()

        obs, _rewards, terminateds, truncateds, infos = env.step(actions)
        for agent, info in infos.items():
            if agent == '__all__':
                continue
            ep_cost += _scalar_cost_from_info(info)

        if terminateds.get('__all__') or truncateds.get('__all__'):
            break

    return ep_cost


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PowerZoo Safe RL Examples")
    parser.add_argument('--omnisafe', action='store_true',
                        help='Run Part B (OmniSafe PPO-Lagrangian)')
    parser.add_argument('--rllib',    action='store_true',
                        help='Run Part C (RLlib reward-only + CMDP metrics)')
    parser.add_argument('--all',      action='store_true',
                        help='Run all parts')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Episodes for Part A evaluation')
    args = parser.parse_args()

    # Part A is always run (no extra deps)
    result_a = part_a_manual_cmdp_loop(n_episodes=args.episodes)

    if args.omnisafe or args.all:
        part_b_omnisafe(n_epochs=10)

    if args.rllib or args.all:
        part_c_rllib_ppo_lag(n_iterations=20)

    print("=" * 60)
    print("Quick reference — key CMDP fields in info dict")
    print("=" * 60)
    print("""
  info[agent]['cost']            # scalar step cost (sum of violations)
  info[agent]['constraint_costs'] # fixed-order vector when available
  info[agent]['costs']           # dict {'thermal_overload': x, 'voltage_violation': y, ...}
  info[agent]['cost']            # scalar compatibility alias only when wrapped
  info[agent]['is_safe']         # bool: no violation this step
  info[agent]['cost_sum']        # aggregate diagnostic alias

  SafeRLWrapper.step() returns:
    obs, reward, cost, terminated, truncated, info   # OmniSafe 6-tuple

  GymnasiumSafeWrapper.step() returns:
    obs, reward, terminated, truncated, info         # standard 5-tuple
    # info['cost'] still present for Gymnasium-compatible safe RL libs

  Task metadata:
    from powerzoo.tasks import make_task
    task = make_task('marl_opf')
    print(task.eval_protocol['cost_threshold'])   # 10.0

  Constraint tightness:
    env = make_task_env('marl_opf', constraint_tightness='strict')
    # higher load ratio, tighter cost budget
""")


if __name__ == '__main__':
    main()
