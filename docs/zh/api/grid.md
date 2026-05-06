# 电网环境

::: powerzoo.envs.base.BaseEnv
    options:
      show_source: false
      members:
        - reset
        - step
        - obs
        - reward

---

::: powerzoo.envs.grid.base.GridEnv
    options:
      show_source: false
      members:
        - __init__
        - reset
        - step
        - obs
        - register_resource
        - unregister_resource
        - cal_pf
        - safety_check

---

::: powerzoo.envs.grid.trans.TransGridEnv
    options:
      show_source: false
      members:
        - __init__
        - reset
        - obs
        - cal_pf
        - safety_check
        - render

---

::: powerzoo.envs.grid.dist.DistGridEnv
    options:
      show_source: false
      members:
        - __init__
        - reset
        - obs
        - cal_pf
        - safety_check
        - render

---

::: powerzoo.envs.grid.dist_3phase.DistGrid3PhaseEnv
    options:
      show_source: false
      members:
        - __init__
        - cal_pf
        - safety_check
