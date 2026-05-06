# Wrappers

All wrappers are importable from `powerzoo.wrappers`.

Task-aware PettingZoo wrapping for benchmark tasks now lives under `powerzoo.tasks.interfaces.TaskPettingZooWrapper`. It remains re-exported from `powerzoo.wrappers` for backward compatibility.

```python
from powerzoo.wrappers import (
    GymnasiumWrapper,
    NormalizationWrapper,
    SafeRLWrapper,
    GymnasiumSafeWrapper,
    ForecastWrapper,
    MARLWrapper,
    FlattenWrapper,
)
```

---

::: powerzoo.wrappers.gym_wrappers.GymnasiumWrapper
    options:
      show_source: false
      members:
        - __init__
        - reset
        - step

Adapts any PowerZoo `GridEnv` to the standard Gymnasium 5-tuple API:

```python
from powerzoo.envs.grid.trans import TransGridEnv
from powerzoo.wrappers import GymnasiumWrapper

env = GymnasiumWrapper(TransGridEnv())
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

---

::: powerzoo.wrappers.gym_wrappers.NormalizationWrapper
    options:
      show_source: false
      members:
        - __init__
        - reset
        - step

Normalises observations (and optionally actions) to `[−1, 1]` using running statistics. Stacks on top of `GymnasiumWrapper`:

```python
from powerzoo.wrappers import GymnasiumWrapper, NormalizationWrapper

env = NormalizationWrapper(GymnasiumWrapper(TransGridEnv()))
```

---

::: powerzoo.wrappers.safe_rl_wrapper.SafeRLWrapper
    options:
      show_source: false
      members:
        - __init__
        - step

Returns a **6-tuple** `(obs, reward, cost, terminated, truncated, info)` compatible with [OmniSafe](https://github.com/PKU-Alignment/omnisafe) and Safety-Gymnasium.

Cost extraction priority:

1. `info['selected_constraint_costs']` — task-selected CMDP vector
2. `info['constraint_costs']` — core env full vector
3. `info['cost_sum']` or compatibility `info['cost']`
4. `0.0` — safe fallback

```python
from powerzoo.wrappers import GymnasiumWrapper, SafeRLWrapper

env = SafeRLWrapper(GymnasiumWrapper(TransGridEnv()), cost_threshold=25.0)
obs, info = env.reset(seed=0)
obs, reward, cost, terminated, truncated, info = env.step(env.action_space.sample())
```

| Parameter | Default | Description |
|---|---|---|
| `cost_threshold` | `25.0` | Scalar threshold exposed as `env.cost_threshold` for OmniSafe |

---

::: powerzoo.wrappers.safe_rl_wrapper.GymnasiumSafeWrapper
    options:
      show_source: false
      members:
        - __init__
        - step

Returns the standard Gymnasium **5-tuple** but injects `cost` into `info['cost']`. Use this when your algorithm reads cost from `info` rather than as a separate return value.

```python
from powerzoo.wrappers import GymnasiumWrapper, GymnasiumSafeWrapper

env = GymnasiumSafeWrapper(GymnasiumWrapper(TransGridEnv()))
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(info['cost'])  # non-negative scalar
```

---

::: powerzoo.wrappers.forecast_wrapper.ForecastWrapper
    options:
      show_source: false
      members:
        - __init__
        - reset
        - observation

Appends a `horizon`-length demand forecast to every observation. Extends `observation_space` automatically (base dim + `horizon`).

| Parameter | Default | Description |
|---|---|---|
| `horizon` | `6` | Number of future steps to append |
| `mode` | `'perfect'` | `'perfect'` (ground truth), `'noisy'` (Gaussian noise), or `'none'` (zeros) |
| `noise_std` | `0.02` | Fractional noise std for `mode='noisy'` (e.g. 0.02 = 2 %) |
| `normalize` | `True` | Divide forecast values by dataset maximum |

```python
from powerzoo.wrappers import GymnasiumWrapper, ForecastWrapper

env = ForecastWrapper(GymnasiumWrapper(TransGridEnv()), horizon=6, mode='noisy')
obs, info = env.reset(seed=0)
# obs[-6:] contains the next 6 half-hour demand values
```

---

::: powerzoo.wrappers.marl_wrapper.MARLWrapper
    options:
      show_source: false
      members:
        - __init__
        - reset
        - step

Converts a single-agent PowerZoo env to the **PettingZoo Parallel API**. Each resource registered in the underlying `GridEnv` becomes an independent agent.

```python
from powerzoo.wrappers import MARLWrapper

env = MARLWrapper(TransGridEnv(), agent_type='generators')
obs, info = env.reset(seed=0)
actions = {a: env.action_space(a).sample() for a in env.agents}
obs, rewards, terminations, truncations, info = env.step(actions)
```

---

::: powerzoo.wrappers.flatten.FlattenWrapper
    options:
      show_source: false
      members:
        - __init__
        - reset
        - step

Flattens dict/nested `observation_space` and `action_space` into 1-D `Box` spaces. Useful for algorithms that require flat vector inputs.

```python
from powerzoo.wrappers import GymnasiumWrapper, FlattenWrapper

env = FlattenWrapper(GymnasiumWrapper(TransGridEnv()))
```
