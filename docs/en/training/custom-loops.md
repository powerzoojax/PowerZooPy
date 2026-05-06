# Custom loops

`Trainer` is a thin SB3 convenience layer. When you need something it does not cover — multi-objective rewards, custom MARL frameworks (EPyMARL, MAPPO, JaxMARL), self-play populations, vectorised rollouts, custom policies — bypass `Trainer` and run your own loop on top of `make_env(...)`.

This page gives the four most common patterns. They all start from the same building block:

```python
from powerzoo.rl import make_env

env = make_env("battery_arbitrage", split="train", normalize=True, seed=0)
```

For MARL tasks add `framework='pettingzoo'`.

## Pattern 1 — vanilla SB3 with custom callback

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

This is what `Trainer.train()` does internally; use this pattern when you want fine control over `algorithm` arguments or callbacks.

## Pattern 2 — multi-objective loop using `info["reward_vector"]`

For tasks that expose a vector reward (currently `dc_microgrid` / `dc_microgrid_safe`):

```python
import numpy as np
from powerzoo.rl import make_env

env = make_env("dc_microgrid", normalize=True, seed=0)
weights = np.array([1.0, 0.5, 0.3])   # energy, cost, carbon

obs, info = env.reset(seed=0)
total_vec = np.zeros(3)
done = False
while not done:
    action = env.action_space.sample()           # replace with your policy
    obs, scalar_r, terminated, truncated, info = env.step(action)
    rv = np.asarray(info["reward_vector"])
    total_vec += rv
    weighted = float(weights @ rv)               # your scalarisation
    done = terminated or truncated
print("episode reward vector:", total_vec)
```

Use this when you want to learn along a chosen direction in the Pareto front, or to evaluate a policy along several scalarisations after the fact.

## Pattern 3 — independent learners on PettingZoo

```python
from powerzoo.rl import make_env

env = make_env("marl_opf", framework="pettingzoo", seed=0)

# example: random rollout with per-agent reward accounting
obs, info = env.reset(seed=0)
totals = {a: 0.0 for a in env.agents}
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, info = env.step(actions)
    for a, r in rewards.items():
        totals[a] += r
print(totals)
```

To turn this into independent SB3 training:

1. Build one SB3 model per agent on a single-agent view of the env (you need a `SingleAgentView` wrapper that exposes only one agent at a time and freezes the others).
2. Round-robin train each model for a chunk of steps; the others act with their current policy.

`Trainer.train_il(...)` already implements this for homogeneous-spaces tasks. For heterogeneous ones (e.g. `marl_ders_benchmark`), you have to build the per-type policy heads yourself.

## Pattern 4 — JaxMARL / MAPPO / EPyMARL wrapper

```python
from powerzoo.rl import make_env
import jax_marl_wrapper                # your bridge code

env = make_env("marl_der_arbitrage", framework="pettingzoo", seed=0)
jax_env = jax_marl_wrapper.from_pettingzoo(env)

trainer = jax_marl_wrapper.MAPPO(jax_env, total_timesteps=5_000_000)
trainer.train()
```

PowerZoo does not ship MAPPO / IPPO / QMIX directly; the `framework='pettingzoo'` env is the integration point for whichever MARL library you prefer. Any library that consumes the PettingZoo Parallel API will work — for example JaxMARL (via the PZ adapter), EPyMARL (via the Gymnasium MARL adapter), and RLlib.

## Pattern 5 — vectorised rollouts for off-policy buffers

PowerZoo envs are pure CPU; vectorise them with SB3's `make_vec_env` (`SubprocVecEnv` for CPU parallelism):

```python
from stable_baselines3.common.env_util import make_vec_env

def env_fn(seed=0):
    return make_env("battery_arbitrage", normalize=True, seed=seed)

vec_env = make_vec_env(env_fn, n_envs=8, seed=0, vec_env_cls="subproc")

from stable_baselines3 import PPO
model = PPO("MlpPolicy", vec_env, n_steps=512, batch_size=256, verbose=1)
model.learn(total_timesteps=2_000_000)
```

For GPU-accelerated rollouts and `lax.scan`-driven batched envs, use the sibling [PowerZooJax](https://github.com/powerzoojax/PowerZooJax) project, which reimplements the same five benchmark suites in pure JAX with `vmap` over thousands of envs.

## Pattern 6 — collect an offline RL dataset

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

The HDF5 schema and `DatasetLoader` API are documented in [API · Offline](../api/offline.md).

## Tips

- Always `env.reset(seed=...)` once at the start of a run to make the rollout reproducible. If you also want to pin per-episode randomness, pass `day_id=k` to `env.reset` (Grid envs accept it).
- The `info` dict carries everything except reward and observations. Log `info['cost_sum']`, `info['cost_*']` per step into TensorBoard if you want to track constraint satisfaction during training, not just at evaluation time.
- For long-horizon tasks (`opf_118_7d`, `marl_ev_v2g`, `dc_microgrid`), bump `gamma` to 0.999 — the default 0.99 discounts day-end rewards too aggressively for 168 / 288 / 336 steps.
- For Safe-RL methods that need vector costs, prefer `info['selected_constraint_costs']` on task envs or `info['constraint_costs']` on core envs. Scalar `info['cost']` is only a compatibility projection.

## See also

- [Trainers](trainers.md) — the high-level convenience layer.
- [Wrappers](wrappers.md) — what `make_env`'s arguments map to.
- [API · Offline](../api/offline.md) — `DatasetGenerator` / `DatasetLoader`.
- [Architecture · Training pipeline](../architecture/training-pipeline.md) — end-to-end view.
