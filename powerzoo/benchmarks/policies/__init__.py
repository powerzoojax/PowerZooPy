"""PowerZoo Baseline Policies

Provides reference policies for benchmarking RL agents.  All policies share
the same ``act(obs, info) -> action`` interface so they can be dropped into
the ``evaluate()`` harness without modification.

Available policies
------------------
``RandomPolicy``
    Samples uniformly from the action space.  Serves as the lower bound.

``RuleBasedPolicy``
    Hand-crafted heuristics:
    - Generators: proportional dispatch (cheapest first).
    - Batteries: charge when time_of_day < 0.4, discharge otherwise (price proxy).

``OraclePolicy``
    Runs the built-in OPF solver to find the optimal dispatch for each step.
    Requires a Gymnasium-wrapped ``TransGridEnv``.  Serves as the upper bound.

Evaluation
----------
Use ``evaluate(policy, env, n_episodes)`` to get a standardised result dict::

    from powerzoo.benchmarks.policies import RandomPolicy, evaluate
    from powerzoo.wrappers import GymnasiumWrapper
    from powerzoo.envs.grid.trans import TransGridEnv

    env = GymnasiumWrapper(TransGridEnv())
    result = evaluate(RandomPolicy(env.action_space), env, n_episodes=10)
    print(result)   # {'mean_reward': ..., 'std_reward': ..., ...}
"""

from powerzoo.benchmarks.policies.base import BasePolicy
from powerzoo.benchmarks.policies.random_policy import RandomPolicy
from powerzoo.benchmarks.policies.rule_based import RuleBasedPolicy
from powerzoo.benchmarks.policies.oracle import OraclePolicy
from powerzoo.benchmarks.evaluation import evaluate, evaluate_task

__all__ = [
    'BasePolicy',
    'RandomPolicy',
    'RuleBasedPolicy',
    'OraclePolicy',
    'evaluate',
    'evaluate_task',
]
