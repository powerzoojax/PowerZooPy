# 预设配置

本页收集每个基准系列一份的现成 `RLConfig` YAML 模板。把模板放进你的实验文件夹，改少数几个关键字段，运行：

```python
from powerzoo.rl import RLConfig, Trainer

cfg = RLConfig.from_yaml("battery_sac.yaml")
Trainer(cfg).train().evaluate(split="test")
```

所有 YAML 共享同一 schema（`task` / `wrappers` / `reward` / `trainer` / `framework` / `seed`）；字段语义见 [Trainers](trainers.md)。

## 单 agent — `battery_arbitrage`

```yaml
task:
  name: battery_arbitrage
  split: train
wrappers:
  normalize: true
  forecast_horizon: 6
  safe_rl: false
trainer:
  algorithm: SAC
  total_timesteps: 200000
  hyperparams:
    learning_rate: 0.0003
    buffer_size: 200000
    batch_size: 256
    tau: 0.01
  save_path: ./results/battery_sac/
framework: auto
seed: 42
```

## 单 agent — `dc_scheduling`

```yaml
task:
  name: dc_scheduling
  split: train
wrappers:
  normalize: true
  safe_rl: true
  cost_threshold: 5.0
trainer:
  algorithm: PPO
  total_timesteps: 1000000
  hyperparams:
    learning_rate: 0.0003
    n_steps: 2048
    n_epochs: 10
  save_path: ./results/dc_scheduling_ppo/
framework: auto
seed: 0
```

## DC microgrid — `dc_microgrid_safe`（Safe RL）

```yaml
task:
  name: dc_microgrid_safe
  split: train
wrappers:
  normalize: true
  safe_rl: true
  cost_threshold: 0.5
trainer:
  algorithm: SAC
  total_timesteps: 2000000
  hyperparams:
    learning_rate: 0.0003
    gamma: 0.999
    buffer_size: 500000
  save_path: ./results/dc_microgrid_safe_sac/
framework: auto
seed: 0
```

## TSO — `marl_uc`（独立学习者）

```yaml
task:
  name: marl_uc
  split: train
trainer:
  algorithm: PPO
  total_timesteps: 3000000
  hyperparams:
    learning_rate: 0.0003
    n_steps: 1024
  save_path: ./results/marl_uc_ippo/
framework: pettingzoo
seed: 0
```

训练用：

```python
Trainer(cfg).train_il()
```

## TSO — `opf_118`（大规模 ED，IL）

```yaml
task:
  name: opf_118
  split: train
trainer:
  algorithm: PPO
  total_timesteps: 10000000
  hyperparams:
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 256
  save_path: ./results/opf_118_ippo/
framework: pettingzoo
seed: 0
```

## DERs — `marl_der_arbitrage`（同时 SAC）

```yaml
task:
  name: marl_der_arbitrage
  split: train
wrappers:
  safe_rl: true
  cost_threshold: 0.5
trainer:
  algorithm: SAC
  total_timesteps: 1500000
  hyperparams:
    learning_rate: 0.0003
    buffer_size: 500000
  save_path: ./results/marl_der_sac/
framework: pettingzoo
seed: 0
```

训练用：

```python
Trainer(cfg).train_marl_simultaneous()
```

## DERs — `marl_ev_v2g`（长 horizon）

```yaml
task:
  name: marl_ev_v2g
  split: train
trainer:
  algorithm: SAC
  total_timesteps: 3000000
  hyperparams:
    gamma: 0.999
    buffer_size: 500000
  save_path: ./results/marl_ev_sac/
framework: pettingzoo
seed: 0
```

## GenCos — `gencos_bidding`（独立 PPO）

```yaml
task:
  name: gencos_bidding
  split: train
trainer:
  algorithm: PPO
  total_timesteps: 5000000
  hyperparams:
    learning_rate: 0.0003
    n_steps: 1024
  save_path: ./results/gencos_ippo/
framework: pettingzoo
seed: 0
```

## DSO — `make_dso_env(...)`（工厂方式，无 YAML loader）

DSO 基准使用直接工厂方式，而不是 `RLConfig`。它的"preset" 就是一次 Python 函数调用：

```python
from powerzoo.tasks.dso_task import make_dso_env
from powerzoo.data import DataLoader
from stable_baselines3 import PPO

env = make_dso_env(split="train", data_loader=DataLoader())
model = PPO(
    "MlpPolicy", env,
    learning_rate=3e-4, n_steps=2048, batch_size=64, verbose=1,
)
model.learn(total_timesteps=3_000_000)
model.save("results/dso_ppo")
```

如需 Recurrent PPO baseline（对非平稳更鲁棒），使用 `sb3-contrib` 的 `RecurrentPPO` 自己包一个循环。

## 常见修改

- **训练更多 episode**：调 `total_timesteps`。单 agent 任务通常 200k–2M 步收敛；大规模 MARL 需要 5M–20M。
- **更换算法**：把 `trainer.algorithm` 设为 `SAC`、`PPO` 或 `TD3`。MARL 默认推荐 PPO；单 agent 连续控制通常 SAC 表现最好。
- **预测研究**：把 `wrappers.forecast_horizon` 设为 0、6 或 24，比较同一任务上无预测 / 短 horizon / 全天预测策略的差异。
- **Safe-RL 研究**：设 `wrappers.safe_rl: true`，并选一个与任务典型违反量级匹配的 `cost_threshold`（可以先跑一次随机 rollout 估算）。

## 另见

- [Trainers](trainers.md) — `Trainer` 与 `make_env` 参考。
- [Wrappers](wrappers.md) — 每个 wrapper 字段的含义。
- [Custom loops](custom-loops.md) — YAML preset 不够用时的写法。
- [Benchmarks](../benchmarks/overview.md) — 每个系列的超参建议。
