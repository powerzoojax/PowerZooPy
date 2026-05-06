"""PowerZoo Benchmarks

Provides normalized scoring (linear between random and oracle returns) and baseline computation tools.

Usage
-----
    from powerzoo.benchmarks import normalized_score, get_baseline

    # Get normalised score  (0 = random, 1 = oracle)
    ns = normalized_score('marl_opf', policy_return=-450.0)

    # See raw baselines
    bl = get_baseline('marl_opf')
    print(bl)  # {'random': -1234.5, 'oracle': -312.7}

Evaluating a policy
--------------------
    from powerzoo.benchmarks import evaluate, evaluate_task

    result = evaluate(policy, env, n_episodes=100)
    print(result['mean_reward'], result['normalized_score'])

Regenerating baselines
-----------------------
    python -m powerzoo.benchmarks.compute

This runs RandomPolicy and OraclePolicy on every registered task and saves
the results to ``powerzoo/benchmarks/baselines.json``.
"""

from powerzoo.benchmarks.scores import (
    normalized_score,
    get_baseline,
    register_baseline,
    list_baselines,
)
from powerzoo.benchmarks.evaluation import evaluate, evaluate_task

# Sub-packages relocated here (v0.2.1)
from powerzoo.benchmarks import policies  # noqa: F401
from powerzoo.benchmarks import offline   # noqa: F401
from powerzoo.benchmarks import viz       # noqa: F401

__all__ = [
    "normalized_score",
    "get_baseline",
    "register_baseline",
    "list_baselines",
    "evaluate",
    "evaluate_task",
    "policies",
    "offline",
    "viz",
]
