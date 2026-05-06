"""PowerZoo — Power System RL Benchmark Library  (v0.2)

A modular, Gymnasium-compatible RL environment suite for power-system
control research, targeting ML / CS benchmark use cases.

Quick start
-----------
Standard ``gym.make()`` interface::

    import gymnasium as gym
    env = gym.make("PowerZoo/TransGrid-medium-v0")
    obs, info = env.reset(seed=42)

Task-based API with train / val / test splits::

    from powerzoo.tasks import make_task_env

    train_env = make_task_env('marl_opf', split='train')
    test_env  = make_task_env('marl_opf', split='test')

Benchmarking with normalized scores (0 = random baseline, 1 = oracle)::

    from powerzoo.benchmarks.policies import RandomPolicy, evaluate
    from powerzoo.benchmarks import normalized_score

    result = evaluate(RandomPolicy(env.action_space), env, n_episodes=100)
    ns = normalized_score('marl_opf', result['mean_reward'])
    print(f"Normalised score: {ns:.3f}")   # 0 = random, 1 = oracle
"""

__version__ = "0.2.0"
__license__ = "MIT"

# --------------------------------------------------------------------------
# Register all PowerZoo environments with gymnasium (runs once on import)
# --------------------------------------------------------------------------
from powerzoo.registration import _register_all as _reg
_reg()
del _reg

# --------------------------------------------------------------------------
# Public API — flat imports for convenience
# --------------------------------------------------------------------------

# Grid environments
from powerzoo.envs.grid.trans import TransGridEnv
from powerzoo.envs.grid.dist  import DistGridEnv

# Resource environments
from powerzoo.envs.resource.battery   import BatteryEnv
from powerzoo.envs.resource.renewable import SolarEnv, WindEnv
from powerzoo.envs.resource.vehicle   import VehicleEnv

# Case loader
from powerzoo.case import load_case

# Wrappers
from powerzoo.wrappers.gym_wrappers import GymnasiumWrapper, NormalizationWrapper
from powerzoo.wrappers.marl_wrapper  import MARLWrapper, TaskPettingZooWrapper

# Policies & evaluation
from powerzoo.benchmarks.policies.random_policy import RandomPolicy
from powerzoo.benchmarks.policies.rule_based    import RuleBasedPolicy
from powerzoo.benchmarks.policies.oracle        import OraclePolicy
from powerzoo.benchmarks.evaluation   import evaluate

# Tasks (preferred high-level API)
from powerzoo.tasks.registry import make_task_env, make_task, list_tasks
from powerzoo.tasks.public import list_public_tasks

# RL module (unified make_env / Trainer / info / describe)
try:
    from powerzoo.rl import make_env, RLConfig, RewardWrapper, info, describe
    from powerzoo.rl import Trainer  # lazy inside __getattr__; re-exported here
    _HAS_RL = True
except Exception:
    _HAS_RL = False

# Benchmarks
from powerzoo.benchmarks import normalized_score

# Visualization
from powerzoo.benchmarks.viz import plot_episode, plot_dispatch, plot_eval_comparison

# Market  (cost-based LMP arbitrage + bid-based market)
from powerzoo.envs.market.cost_based_market import CostBasedMarketEnv
from powerzoo.envs.market.bid_based_market import BidBasedMarketEnv

__all__ = [
    "__version__",
    # envs
    "TransGridEnv", "DistGridEnv",
    "BatteryEnv", "SolarEnv", "WindEnv", "VehicleEnv",
    # case
    "load_case",
    # wrappers
    "GymnasiumWrapper", "NormalizationWrapper", "MARLWrapper", "TaskPettingZooWrapper",
    # policies
    "RandomPolicy", "RuleBasedPolicy", "OraclePolicy", "evaluate",
    # tasks
    "make_task_env", "make_task", "list_tasks", "list_public_tasks",
    # rl module
    "make_env", "RLConfig", "RewardWrapper", "info", "describe", "Trainer",
    # benchmarks
    "normalized_score",
    # visualization
    "plot_episode", "plot_dispatch", "plot_eval_comparison",
    # market
    "CostBasedMarketEnv", "BidBasedMarketEnv",
]
