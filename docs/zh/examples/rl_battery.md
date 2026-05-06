# RL — 用 Stable-Baselines3 控制电池

**脚本**：`examples/RL05_battery_control_sb3.py`

本示例有意保持在**低层**：手动构建 `DistGridEnv` + `BatteryEnv`，再用一个自定义 `gymnasium.Env` 把两者封装在一起，并用 Stable-Baselines3 训练一个 PPO agent。想了解底层物理对象如何接入一个朴素 SB3 循环时阅读它。

> **基准实验**请优先使用 `make_task_env('battery_arbitrage')`（已经准备好同样的接线，并提供固定的 `train`/`val`/`test` 切分、oracle baseline 与 `evaluate()` 集成）或 `powerzoo.rl.make_env(...)` + `Trainer`。见 [Training · Trainers](../training/trainers.md)。

## 前置依赖

```bash
uv sync --extra rl   # installs SB3, torch, gym, etc.
```

## 环境包装

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

## 训练

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

## 评估

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

## 多智能体（PettingZoo）

多智能体基准请使用 `make_task_env(...)`（可加 `framework="pettingzoo"`）配合一个已注册的任务——例如 `marl_der_arbitrage` 或 `marl_opf`。见 [快速开始](../../getting-started.md) 以及 `examples/MARL01_opf.py` 中的可运行循环。

!!! note
    默认情况下，`DistGridEnv` 使用仅网损形式的 reward（`-loss_penalty_weight * p_loss_MW`，`loss_penalty_weight=0.1`）。电压与热稳违反留在 CMDP cost 字段（`info['cost_voltage_violation']`、`info['cost_thermal_overload']`）中，除非你显式启用可选的 soft-penalty 权重。你也可以在子类中覆盖 `_compute_reward` 以实现自定义目标（如削峰、能量套利、调频等）。
