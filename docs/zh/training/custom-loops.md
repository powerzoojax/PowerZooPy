# 自定义循环

`Trainer` 是 SB3 的一层轻量封装。当你需要它未覆盖的功能——多目标 reward、自定义 MARL framework（EPyMARL、MAPPO、JaxMARL）、self-play 种群、向量化 rollout、自定义 policy——绕过 `Trainer`，在 `make_env(...)` 之上写自己的循环。

本页给出最常见的若干种模式。它们都从同一个构件起步：

```python
from powerzoo.rl import make_env

env = make_env("battery_arbitrage", split="train", normalize=True, seed=0)
```

MARL 任务再加上 `framework='pettingzoo'`。

## 模式 1 — 朴素 SB3 + 自定义 callback

```python
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

env = make_env("battery_arbitrage", normalize=True, seed=0)

model = SAC(
    "MlpPolicy", env,
    learning_rate=3e-4, buffer_size=200_000,
    verbose=1, tensorboard_log="./logs/",
)

cb = CheckpointCallback(save_freq=10_000, save_path="./ckpt/")
model.learn(total_timesteps=200_000, callback=cb)
```

这正是 `Trainer.train()` 内部执行的流程；当你需要精细控制 `algorithm` 参数或 callback 时，使用这种模式。

## 模式 2 — 使用 `info["reward_vector"]` 的多目标循环

对于提供向量 reward 的任务（目前是 `dc_microgrid` / `dc_microgrid_safe`）：

```python
import numpy as np
from powerzoo.rl import make_env

env = make_env("dc_microgrid", normalize=True, seed=0)
weights = np.array([1.0, 0.5, 0.3])   # energy, cost, carbon

obs, info = env.reset(seed=0)
total_vec = np.zeros(3)
done = False
while not done:
    action = env.action_space.sample()           # 替换为你的 policy
    obs, scalar_r, terminated, truncated, info = env.step(action)
    rv = np.asarray(info["reward_vector"])
    total_vec += rv
    weighted = float(weights @ rv)               # 你的标量化
    done = terminated or truncated
print("episode reward vector:", total_vec)
```

适用于：希望沿 Pareto 前沿的某个方向学习，或事后用多种标量化方式评估同一策略。

## 模式 3 — PettingZoo 上的独立学习者

```python
from powerzoo.rl import make_env

env = make_env("marl_opf", framework="pettingzoo", seed=0)

# 例：随机 rollout，按 agent 累计 reward
obs, info = env.reset(seed=0)
totals = {a: 0.0 for a in env.agents}
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, info = env.step(actions)
    for a, r in rewards.items():
        totals[a] += r
print(totals)
```

把它扩展为独立学习者训练：

1. 在 env 的单 agent 视图上为每个 agent 构建一个独立 SB3 模型（你需要一个 `SingleAgentView` wrapper，每次只暴露一个 agent，其余 agent 冻结）。
2. 轮流训练每个模型一段步数；其他 agent 用当前 policy 执行动作。

`Trainer.train_il(...)` 已在同构 space 的任务上实现了这套流程。对于异构任务（如 `marl_ders_benchmark`），需要自己实现按类型划分的 policy head。

## 模式 4 — JaxMARL / MAPPO / EPyMARL 接入

```python
from powerzoo.rl import make_env
import jax_marl_wrapper                # 你的桥接代码

env = make_env("marl_der_arbitrage", framework="pettingzoo", seed=0)
jax_env = jax_marl_wrapper.from_pettingzoo(env)

trainer = jax_marl_wrapper.MAPPO(jax_env, total_timesteps=5_000_000)
trainer.train()
```

PowerZoo 不自带 MAPPO / IPPO / QMIX；`framework='pettingzoo'` env 是与任意 MARL 库的集成点。任何能消费 PettingZoo Parallel API 的库都能直接接入——例如 JaxMARL（通过 PZ adapter）、EPyMARL（通过 Gymnasium MARL adapter）以及 RLlib。

## 模式 5 — 为 off-policy buffer 做向量化 rollout

PowerZoo env 是纯 CPU 实现；用 SB3 的 `make_vec_env`（CPU 并行可用 `SubprocVecEnv`）做向量化：

```python
from stable_baselines3.common.env_util import make_vec_env

def env_fn(seed=0):
    return make_env("battery_arbitrage", normalize=True, seed=seed)

vec_env = make_vec_env(env_fn, n_envs=8, seed=0, vec_env_cls="subproc")

from stable_baselines3 import PPO
model = PPO("MlpPolicy", vec_env, n_steps=512, batch_size=256, verbose=1)
model.learn(total_timesteps=2_000_000)
```

如需 GPU 加速 rollout 与 `lax.scan` 驱动的批量 env，请使用同级 [PowerZooJax](https://github.com/powerzoojax/PowerZooJax) 项目，它用纯 JAX 重现了同样的五大基准系列，并支持 `vmap` 在数千个 env 上并行。

## 模式 6 — 收集离线 RL 数据集

```python
from powerzoo.benchmarks.offline import DatasetGenerator
from powerzoo.benchmarks.policies import RandomPolicy

env = make_env("marl_opf", framework="auto")    # task adapter
gen = DatasetGenerator(env, info_keys=["cost_sum", "is_safe"])
dataset = gen.collect(
    policy=RandomPolicy(env.action_space),
    n_episodes=500,
    save_path="data/marl_opf_random.h5",
    seed=42,
)
```

HDF5 schema 与 `DatasetLoader` API 见 [API · Offline](../api/offline.md)。

## 实用提示

- 每次 run 开始时调用一次 `env.reset(seed=...)`，可保证 rollout 可复现。若要固定每个 episode 的随机性，传 `day_id=k` 给 `env.reset`（Grid env 支持该参数）。
- `info` dict 包含了除 reward 与 observation 之外的全部信息。想在训练而不仅是评估时跟踪约束满足情况，可以把 `info['cost_sum']`、`info['cost_*']` 每步写入 TensorBoard。
- 对于长 horizon 任务（`opf_118_7d`、`marl_ev_v2g`、`dc_microgrid`），建议把 `gamma` 提到 0.999——默认 0.99 在 168 / 288 / 336 步上会过度折扣末期 reward。
- 当 Safe-RL 方法需要向量形式的 cost 时，优先在 task env 中读取 `info['selected_constraint_costs']`，或在 core env 中读取 `info['constraint_costs']`。标量 `info['cost']` 只是兼容投影。

## 另见

- [Trainers](trainers.md) — 高层便利封装。
- [Wrappers](wrappers.md) — `make_env` 参数对应的 wrapper。
- [API · Offline](../api/offline.md) — `DatasetGenerator` / `DatasetLoader`。
- [Architecture · Training pipeline](../architecture/training-pipeline.md) — 端到端视图。
