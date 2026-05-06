# 资源

::: powerzoo.envs.resource.base.ResourceEnv
    options:
      show_source: false
      members:
        - __init__
        - attach
        - detach
        - reset
        - step
        - status
        - bus_id
        - grid_obs
        - grid_obs_names
        - grid_action_bounds
        - grid_action_from_normalized

---

::: powerzoo.envs.resource.battery.BatteryEnv
    options:
      show_source: false
      members:
        - __init__
        - reset
        - step
        - obs
        - status
        - get_soc_history
        - grid_obs
        - grid_obs_names

---

::: powerzoo.envs.resource.vehicle.VehicleEnv
    options:
      show_source: false
      members:
        - __init__
        - reset
        - step
        - obs
        - status
        - available_power
        - check_departure_ready

---

::: powerzoo.envs.resource.renewable.RenewableEnv
    options:
      show_source: false
      members:
        - __init__
        - reset
        - step
        - obs
        - status
        - attach
        - grid_obs
        - grid_obs_names
        - grid_action_from_normalized

---

::: powerzoo.envs.resource.renewable.SolarEnv
    options:
      show_source: false
      members:
        - __init__

---

::: powerzoo.envs.resource.renewable.WindEnv
    options:
      show_source: false
      members:
        - __init__

---

::: powerzoo.envs.resource.flexload.FlexLoad
    options:
      show_source: false
      members:
        - __init__
        - reset
        - step
        - obs
        - status

---

::: powerzoo.envs.resource.datacenter.DataCenterEnv
    options:
      show_source: false
      members:
        - __init__
        - reset
        - step
        - obs
        - status

## 自定义资源的生命周期与初始化

在 PowerZoo 中实现自定义物理资产环境（如储能、发电机）时，必须严格遵守以下生命周期与初始化约束：

1. **延迟挂载父对象**
   通过 `attach(parent_bus_id)` 进行节点挂载时，`delta_t_minutes` 参数会直接从底层 grid env 或父节点同步过来。`_dt_h` 的内部定义已弃用；应统一通过挂载后同步好的 `self.dt_hours` 访问，以保持系统级时间一致性。

2. **`_complete_resource_init()` 钩子**
   为防止子类在 `action_space` / `observation_space` 构建完成前进行 action space 同步而引发崩溃或死锁，所有继承自 `BaseEnv` 的自定义 Resource 类必须在 `__init__` 的最后一步显式调用 `self._complete_resource_init()`，以确保依赖装配与校验都能正确完成。
