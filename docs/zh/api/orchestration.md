# 编排与数据

`PowerEnv` 是 grid 与挂载在其上的 resource 的编排入口。`DataLoader` 负责为电网与可再生加载时序数据。

::: powerzoo.envs.power_env.PowerEnv
    options:
      show_source: false
      members:
        - __init__
        - from_yaml
        - reset
        - step
        - render
        - close
        - get_resource_metadata
        - get_resource_status

---

::: powerzoo.data.data_loader.DataLoader
    options:
      show_source: false
      members:
        - __init__
        - load_signals
        - load_actual_series
        - load_forecast_panel
        - load_data
