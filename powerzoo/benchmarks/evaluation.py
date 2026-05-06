"""Standard evaluation harness for PowerZoo policies.

``evaluate(policy, env, n_episodes)`` runs the policy for a fixed number of
episodes and returns a standardised result dict suitable for benchmark tables.

``evaluate_task(policy, task_name)`` is a higher-level helper that reads the
task's ``eval_protocol`` (n_episodes, seed, split, cost_threshold) and
automatically computes the normalised score and cost violation rate, so that
paper results are directly comparable.

Result dict keys
----------------
``mean_reward``          Mean episode total reward over all episodes.
``std_reward``           Standard deviation.
``min_reward``           Minimum episode reward.
``max_reward``           Maximum episode reward.
``mean_ep_length``       Mean episode length (steps).
``episode_rewards``      Raw list of per-episode rewards.
``episode_metrics``      List of per-episode ``info['episode']['metrics']`` dicts
                         (if available).
``normalized_score``     Normalized score (0 = random baseline, 1 = oracle), or None when
                         baselines are not yet computed.
``steps_per_second``     Wall-clock throughput of this evaluation run.
``mean_episode_cost``    Mean cumulative cost per episode (CMDP J_C).
``std_episode_cost``     Standard deviation of cumulative cost.
``episode_costs``        Raw list of per-episode cumulative costs.
``cost_violation_rate``  Fraction of episodes where cumulative cost exceeds
                         ``cost_threshold`` (0–1).  None if threshold not given.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _scalar_step_reward(reward: Any) -> float:
    """Turn Gymnasium scalar or RLlib / PettingZoo parallel dict rewards into one float.

    Cooperative tasks broadcast the same reward to every agent; we take the mean so
    episode totals match the single shared reward per timestep (not ``n_agents``×).
    """
    if isinstance(reward, dict):
        vals = [float(v) for k, v in reward.items() if k != "__all__"]
        if not vals:
            return 0.0
        return float(np.mean(vals))
    return float(reward)


def _episode_done(terminated: Any, truncated: Any) -> bool:
    """Episode finished (Gymnasium bools or RLlib parallel dicts with ``__all__``)."""
    if isinstance(terminated, dict):
        tr = truncated if isinstance(truncated, dict) else {}
        return bool(terminated.get("__all__", False) or tr.get("__all__", False))
    return bool(terminated or truncated)


def _extract_step_cost(info: Any) -> float:
    """Extract per-step cost from an info dict.

    Handles two cases:
    - Single-agent: flat dict with ``info['cost']``.
    - Multi-agent: dict of per-agent dicts; averages agent costs.
    """
    if not isinstance(info, dict):
        return 0.0
    # Multi-agent: top-level values are dicts themselves
    first_val = next(iter(info.values()), None)
    if isinstance(first_val, dict):
        agent_costs = [
            float(v.get('cost', 0.0))
            for v in info.values()
            if isinstance(v, dict) and 'cost' in v
        ]
        return sum(agent_costs) / len(agent_costs) if agent_costs else 0.0
    # Single-agent
    return max(0.0, float(info.get('cost', 0.0)))


def _extract_step_cost_components(info: Any) -> Tuple[Tuple[str, ...], np.ndarray, float]:
    """Extract named vector costs plus the backward-compatible scalar alias."""
    if not isinstance(info, dict):
        return (), np.zeros((0,), dtype=np.float32), 0.0

    first_val = next(iter(info.values()), None)
    if isinstance(first_val, dict):
        names: Tuple[str, ...] = ()
        vectors: List[np.ndarray] = []
        scalars: List[float] = []
        for agent_info in info.values():
            if not isinstance(agent_info, dict):
                continue
            agent_names, agent_vec, agent_scalar = _extract_step_cost_components(agent_info)
            if agent_names and not names:
                names = agent_names
            if names:
                if agent_names and agent_names != names:
                    raise ValueError(
                        f"Inconsistent multi-agent constraint names: {agent_names} vs {names}"
                    )
                if agent_vec.shape == (len(names),):
                    vectors.append(agent_vec.astype(np.float32))
            scalars.append(float(agent_scalar))

        if names and vectors:
            return names, np.mean(np.stack(vectors, axis=0), axis=0), float(np.mean(scalars))
        if scalars:
            scalar = float(np.mean(scalars))
            return (), np.zeros((0,), dtype=np.float32), scalar
        return (), np.zeros((0,), dtype=np.float32), 0.0

    names = tuple(
        info.get('selected_constraint_names')
        or info.get('constraint_names')
        or ()
    )
    if 'selected_constraint_costs' in info:
        vector = np.asarray(info['selected_constraint_costs'], dtype=np.float32).reshape(-1)
    elif 'constraint_costs' in info:
        vector = np.asarray(info['constraint_costs'], dtype=np.float32).reshape(-1)
    elif isinstance(info.get('costs'), dict):
        if not names:
            names = tuple(info['costs'].keys())
        vector = np.asarray([info['costs'].get(name, 0.0) for name in names], dtype=np.float32)
    else:
        vector = np.zeros((0,), dtype=np.float32)

    scalar = float(
        info.get(
            'selected_cost_sum',
            info.get('cost_sum', info.get('cost', float(vector.sum()) if vector.size else 0.0)),
        )
    )
    return names, vector, scalar


def evaluate(
    policy,
    env,
    n_episodes: int = 10,
    seed_start: int = 0,
    verbose: bool = False,
    task_id: Optional[str] = None,
    cost_threshold: Optional[float] = None,
    cost_thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Run *policy* on *env* for *n_episodes* and return benchmark metrics.

    Args:
        policy:          Any object with an ``act(obs, info) -> action`` method.
        env:             A Gymnasium-compatible environment (must return
                         ``(obs, info)`` from ``reset()`` and
                         ``(obs, reward, terminated, truncated, info)`` from
                         ``step()``).  A ``SafeRLWrapper`` 6-tuple
                         ``(obs, reward, cost, terminated, truncated, info)``
                         is also supported.
                         RLlib-style multi-agent adapters (dict obs/act/reward and
                         ``__all__`` in terminated/truncated) are supported; per-step
                         reward is aggregated as the mean across agents when each
                         agent receives the same cooperative reward.
        n_episodes:      Number of episodes to run.
        seed_start:      The first episode uses ``seed=seed_start``, the second
                         ``seed=seed_start+1``, etc.  Pass ``None`` to use
                         random seeds.
        verbose:         Print per-episode summary if True.
        task_id:         When provided, the result includes a ``normalized_score``
                         field computed from pre-stored baselines.
        cost_threshold:  Maximum allowed cumulative episode cost (CMDP budget J_C).
                         Used to compute ``cost_violation_rate``.  Falls back to
                         ``env.cost_threshold`` if set on the wrapper.

    Returns:
        Dict with keys: ``mean_reward``, ``std_reward``, ``min_reward``,
        ``max_reward``, ``mean_ep_length``, ``episode_rewards``,
        ``episode_metrics``, ``normalized_score``, ``steps_per_second``,
        ``mean_episode_cost``, ``std_episode_cost``, ``episode_costs``,
        ``cost_violation_rate``.

    Example::

        from powerzoo.benchmarks.policies import RandomPolicy, evaluate
        from powerzoo.wrappers import GymnasiumWrapper
        from powerzoo.envs.grid.trans import TransGridEnv

        env = GymnasiumWrapper(TransGridEnv())
        result = evaluate(RandomPolicy(env.action_space), env, n_episodes=5)
        print(result['mean_reward'])
    """
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_metrics: List[Dict] = []
    episode_costs: List[float] = []
    constraint_names: Tuple[str, ...] = ()
    episode_cost_vectors: List[np.ndarray] = []

    # Resolve cost_threshold: explicit arg > env attribute > None
    if cost_threshold is None:
        cost_threshold = getattr(env, 'cost_threshold', None)
    if cost_thresholds is None:
        cost_thresholds = getattr(env, 'cost_thresholds', None)

    t_start = time.perf_counter()
    total_steps = 0

    for ep in range(n_episodes):
        seed = seed_start + ep if seed_start is not None else None
        obs, info = env.reset(seed=seed)

        if hasattr(policy, 'reset'):
            policy.reset()

        ep_reward = 0.0
        ep_cost   = 0.0
        ep_cost_vector = None
        ep_steps  = 0
        terminated: Any = False
        truncated: Any = False

        while not _episode_done(terminated, truncated):
            action = policy.act(obs, info)
            step_out = env.step(action)

            if len(step_out) == 6:
                # SafeRLWrapper 6-tuple: (obs, reward, cost, terminated, truncated, info)
                obs, reward, step_cost, terminated, truncated, info = step_out
                if not isinstance(step_cost, (int, float)):
                    step_cost = _extract_step_cost(info)
            else:
                obs, reward, terminated, truncated, info = step_out
                step_cost = _extract_step_cost(info)

            step_constraint_names, step_cost_vector, step_scalar = _extract_step_cost_components(info)
            if step_constraint_names:
                if not constraint_names:
                    constraint_names = step_constraint_names
                elif step_constraint_names != constraint_names:
                    raise ValueError(
                        f"Constraint names changed within evaluation run: {step_constraint_names} vs {constraint_names}"
                    )
                if ep_cost_vector is None:
                    ep_cost_vector = np.zeros((len(constraint_names),), dtype=np.float64)
                ep_cost_vector += step_cost_vector.astype(np.float64)
            if not isinstance(step_cost, (int, float)):
                step_cost = step_scalar

            ep_reward += _scalar_step_reward(reward)
            ep_cost   += float(step_cost)
            ep_steps  += 1

        episode_rewards.append(ep_reward)
        episode_costs.append(ep_cost)
        if constraint_names:
            if ep_cost_vector is None:
                ep_cost_vector = np.zeros((len(constraint_names),), dtype=np.float64)
            episode_cost_vectors.append(ep_cost_vector)
        episode_lengths.append(ep_steps)
        total_steps += ep_steps

        ep_metrics = (
            info.get('episode', {}).get('metrics', {})
            if isinstance(info, dict) and 'episode' in info
            else {}
        )
        episode_metrics.append(ep_metrics)

        if verbose:
            print(f"  Episode {ep+1:3d}/{n_episodes}: "
                  f"reward={ep_reward:9.3f}  cost={ep_cost:8.3f}  steps={ep_steps}")

    elapsed = time.perf_counter() - t_start
    rewards = np.array(episode_rewards, dtype=np.float64)
    costs   = np.array(episode_costs,   dtype=np.float64)
    mean_reward = float(rewards.mean())
    cost_vectors = (
        np.stack(episode_cost_vectors, axis=0)
        if episode_cost_vectors
        else np.zeros((len(episode_costs), 0), dtype=np.float64)
    )

    resolved_cost_thresholds: Optional[np.ndarray] = None
    if constraint_names:
        if cost_thresholds is not None:
            resolved_cost_thresholds = np.asarray(cost_thresholds, dtype=np.float64).reshape(-1)
        elif cost_threshold is not None:
            resolved_cost_thresholds = np.full(
                (len(constraint_names),),
                float(cost_threshold),
                dtype=np.float64,
            )
        if resolved_cost_thresholds is not None and resolved_cost_thresholds.shape != (len(constraint_names),):
            raise ValueError(
                f"cost_thresholds shape {resolved_cost_thresholds.shape} does not match "
                f"{len(constraint_names)} constraints."
            )

    # Compute normalized score if task_id provided
    ns: Optional[float] = None
    if task_id is not None:
        try:
            from powerzoo.benchmarks.scores import normalized_score
            ns = normalized_score(task_id, mean_reward)
        except Exception:
            pass

    # Cost violation rate: fraction of episodes exceeding cost_threshold
    cvr: Optional[float] = None
    if cost_threshold is not None:
        cvr = float(np.mean(costs > cost_threshold))

    result = {
        'mean_reward':         mean_reward,
        'std_reward':          float(rewards.std()),
        'min_reward':          float(rewards.min()),
        'max_reward':          float(rewards.max()),
        'mean_ep_length':      float(np.mean(episode_lengths)),
        'episode_rewards':     episode_rewards,
        'episode_metrics':     episode_metrics,
        'normalized_score':    ns,
        'steps_per_second':    total_steps / elapsed if elapsed > 0 else float('inf'),
        # --- CMDP cost metrics ---
        'mean_episode_cost':   float(costs.mean()),
        'std_episode_cost':    float(costs.std()),
        'episode_costs':       episode_costs,
        'cost_violation_rate': cvr,
        'constraint_names':    list(constraint_names) if constraint_names else None,
        'cost_thresholds': (
            resolved_cost_thresholds.tolist()
            if resolved_cost_thresholds is not None
            else None
        ),
    }
    if constraint_names:
        result['mean_episode_cost_by_constraint'] = {
            name: float(cost_vectors[:, idx].mean())
            for idx, name in enumerate(constraint_names)
        }
        result['std_episode_cost_by_constraint'] = {
            name: float(cost_vectors[:, idx].std())
            for idx, name in enumerate(constraint_names)
        }
        result['episode_costs_by_constraint'] = {
            name: cost_vectors[:, idx].tolist()
            for idx, name in enumerate(constraint_names)
        }
        if resolved_cost_thresholds is not None:
            result['cost_violation_rate_by_constraint'] = {
                name: float(np.mean(cost_vectors[:, idx] > resolved_cost_thresholds[idx]))
                for idx, name in enumerate(constraint_names)
            }

    if verbose:
        print(f"\n  Summary ({n_episodes} episodes):")
        print(f"    mean ± std  = {result['mean_reward']:.3f} ± {result['std_reward']:.3f}")
        print(f"    min / max   = {result['min_reward']:.3f} / {result['max_reward']:.3f}")
        if ns is not None:
            print(f"    norm score  = {ns:.4f}   (0=random, 1=oracle)")
        print(f"    mean cost   = {result['mean_episode_cost']:.3f} ± {result['std_episode_cost']:.3f}")
        if cvr is not None:
            print(f"    cost viol.  = {cvr:.1%}   (threshold={cost_threshold})")
        print(f"    throughput  = {result['steps_per_second']:.0f} steps/s")

    return result


def evaluate_task(
    policy,
    task_name: str,
    override_n_episodes: Optional[int] = None,
    override_seed: Optional[int] = None,
    override_split: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Evaluate *policy* using the task's standardized evaluation protocol.

    This is the recommended API for producing benchmark-comparable numbers.
    It reads the task's ``eval_protocol`` (n_episodes=100, seed=42, split='test',
    cost_threshold) and automatically computes the normalized score and cost
    violation rate.

    Args:
        policy:               Policy with ``act(obs, info) -> action`` method.
        task_name:            Registered task name (e.g. ``'marl_opf'``).
        override_n_episodes:  Override the protocol's episode count.
        override_seed:        Override the protocol's starting seed.
        override_split:       Override the protocol's data split.
        verbose:              Print per-episode progress.

    Returns:
        Same dict as ``evaluate()``, with ``normalized_score`` and
        ``cost_violation_rate`` populated from the task protocol.

    Example::

        from powerzoo.benchmarks.policies import OraclePolicy, evaluate_task
        from powerzoo.tasks import make_task_env

        env = make_task_env('marl_opf', split='test')
        result = evaluate_task(OraclePolicy(env), 'marl_opf')
        print(f"Oracle normalised score: {result['normalized_score']:.4f}")
        print(f"Cost violation rate:     {result['cost_violation_rate']:.1%}")
    """
    from powerzoo.tasks.registry import get_task_info, make_task
    from powerzoo.wrappers.gym_wrappers import GymnasiumWrapper

    info = get_task_info(task_name)
    protocol = info.get('eval_protocol') or {}

    n_episodes     = override_n_episodes or protocol.get('n_episodes', 100)
    seed           = override_seed        or protocol.get('seed_start', 42)
    split          = override_split       or protocol.get('split', 'test')
    cost_threshold = protocol.get('cost_threshold', None)
    task = make_task(task_name, split=split)
    spec = task.constraint_spec() if hasattr(task, 'constraint_spec') else None
    cost_thresholds = protocol.get('cost_thresholds', None)
    if cost_thresholds is None and spec is not None:
        cost_thresholds = list(spec.thresholds)

    env = task.create_env()
    if not hasattr(env, 'observation_space'):
        env = GymnasiumWrapper(env)

    return evaluate(
        policy, env,
        n_episodes=n_episodes,
        seed_start=seed,
        verbose=verbose,
        task_id=task_name,
        cost_threshold=cost_threshold,
        cost_thresholds=cost_thresholds,
    )
