# Presets

This page collects ready-to-use `RLConfig` YAML templates, one per benchmark suite. Drop a template into your experiment folder, edit the few fields that matter, and run:

```python
from powerzoo.rl import RLConfig, Trainer

cfg = RLConfig.from_yaml("battery_sac.yaml")
Trainer(cfg).train().evaluate(split="test")
```

All YAMLs share the same schema (`task` / `wrappers` / `reward` / `trainer` / `framework` / `seed`); see [Trainers](trainers.md) for the field semantics.

## Single-agent — `battery_arbitrage`

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

## Single-agent — `dc_scheduling`

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

## DC microgrid — `dc_microgrid_safe` (Safe RL)

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

## TSO — `marl_uc` (independent learners)

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

Train with:

```python
Trainer(cfg).train_il()
```

## TSO — `opf_118` (large-scale ED, IL)

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

## DERs — `marl_der_arbitrage` (simultaneous SAC)

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

Train with:

```python
Trainer(cfg).train_marl_simultaneous()
```

## DERs — `marl_ev_v2g` (long horizon)

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

## GenCos — `gencos_bidding` (independent PPO)

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

## DSO — `make_dso_env(...)` (factory, no YAML loader)

The DSO benchmark uses a direct factory rather than `RLConfig`. The equivalent of a "preset" is a single Python function call:

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

For a Recurrent PPO baseline (handles non-stationarity better), wrap your own loop using the SB3 `RecurrentPPO` from `sb3-contrib`.

## Common modifications

- **More episodes per training run.** Tune `total_timesteps`. Single-agent tasks converge in 200k–2M steps; large MARL needs 5M–20M.
- **Different algorithm.** Set `trainer.algorithm` to `SAC`, `PPO` or `TD3`. PPO is the safe default for MARL; SAC is usually best for single-agent continuous control.
- **Forecast study.** Set `wrappers.forecast_horizon` to 0, 6 or 24 to compare no-forecast vs short-horizon vs full-day forecast policies on the same task.
- **Safe-RL study.** Set `wrappers.safe_rl: true` and a `cost_threshold` consistent with the task's typical violation magnitude (read it off from a random rollout).

## See also

- [Trainers](trainers.md) — `Trainer` and `make_env` reference.
- [Wrappers](wrappers.md) — what each wrapper field does.
- [Custom loops](custom-loops.md) — when a YAML preset is not enough.
- [Benchmarks](../benchmarks/overview.md) — per-suite hyperparameter recommendations.
