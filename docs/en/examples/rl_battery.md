# RL — Battery Control with Stable-Baselines3

**Script:** `examples/RL05_battery_control_sb3.py`

This example is intentionally **low-level**: it builds a `DistGridEnv` + `BatteryEnv` by hand, wraps the pair in a custom `gymnasium.Env`, and trains a PPO agent with Stable-Baselines3. Read it when you want to see how the underlying physics objects connect to a vanilla SB3 loop.

> **For benchmark experiments**, prefer `make_task_env('battery_arbitrage')` (which already configures the same setup, plus a fixed `train`/`val`/`test` split, oracle baseline and `evaluate()` integration) or `powerzoo.rl.make_env(...)` + `Trainer`. See [Training · Trainers](../training/trainers.md).

## Prerequisites

```bash
uv sync --extra rl   # installs SB3, torch, gym, etc.
```

## Environment Wrapper

```python
import gymnasium as gym
import numpy as np
from powerzoo.envs.grid import DistGridEnv
from powerzoo.envs.resource import BatteryEnv


class BatteryGridEnv(gym.Env):
    """Single-battery control on a 33-bus distribution grid."""

    def __init__(self):
        super().__init__()
        self.grid = DistGridEnv()
        self.battery = BatteryEnv(
            capacity_mwh=2.0, power_mw=1.0,
            parent=self.grid, bus_id=18,
        )

        # Action: normalised power in [-1, 1]  →  [-power_mw, +power_mw]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation: [SOC, voltage_min, voltage_max, time_step/48]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid.reset()
        return self._obs(), {}

    def step(self, action):
        power_mw = float(action[0]) * self.battery.power_mw
        grid_action = {self.battery.resource_id: {"p_mw": power_mw}}
        state, reward, done, truncated, info = self.grid.step(grid_action)
        done = self.grid.time_step >= 48
        return self._obs(), reward, done, truncated, info

    def _obs(self):
        state = self.grid._get_state()
        v = state["nodes"]["voltage"].values
        return np.array([
            self.battery.soc,
            v.min(),
            v.max(),
            self.grid.time_step / 48.0,
        ], dtype=np.float32)
```

## Training

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

env = BatteryGridEnv()
check_env(env, warn=True)

model = PPO(
    "MlpPolicy", env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    verbose=1,
    tensorboard_log="./logs/battery_ppo",
)

model.learn(total_timesteps=200_000)
model.save("battery_ppo_agent")
```

## Evaluation

```python
model = PPO.load("battery_ppo_agent")
obs, _ = env.reset()
rewards = []

for _ in range(48):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    rewards.append(reward)
    if done:
        break

print(f"Episode return: {sum(rewards):.2f}")
```

## Multi-Agent (PettingZoo)

For multi-agent benchmarks, use `make_task_env(...)` (optionally `framework="pettingzoo"`) with a registered task — for example `marl_der_arbitrage` or `marl_opf`. See [Getting Started](../../getting-started.md) and `examples/MARL01_opf.py` for a runnable loop.

!!! note
    By default, `DistGridEnv` uses a loss-only reward
    (`-loss_penalty_weight * p_loss_MW`, with `loss_penalty_weight=0.1`).
    Voltage and thermal violations stay in the CMDP cost fields
    (`info['cost_voltage_violation']`, `info['cost_thermal_overload']`)
    unless you explicitly enable the optional soft-penalty weights.
    You can still override `_compute_reward` in a subclass to implement your
    own objective (e.g., peak shaving, energy arbitrage, or frequency regulation).
