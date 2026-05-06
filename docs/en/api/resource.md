# Resources

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

## Custom Resource Lifecycle and Initialization

When implementing custom physical asset environments (e.g., storage, generators) in PowerZoo, you must adhere strictly to the following lifecycle and initialization constraints:

1. **Deferred Parent Attachment**
   During node attachment via `attach(parent_bus_id)`, the `delta_t_minutes` parameter is synchronized directly from the underlying grid-level environment or parent node. Internal definitions of `_dt_h` are deprecated; unified access must use `self.dt_hours` synchronized post-attachment to maintain system-wide temporal consistency.

2. **`_complete_resource_init()` Hook**
   To prevent crashes or deadlocks triggered by action space synchronization before a subclass finishes building its `action_space` or `observation_space`, all custom Resource classes inheriting from `BaseEnv` must explicitly call `self._complete_resource_init()` as the final step in their `__init__` method. This guarantees final dependency assembly and validation correctly resolves.
