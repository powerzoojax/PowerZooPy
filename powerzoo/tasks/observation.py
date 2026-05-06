"""Canonical task-level observation mode configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


OBSERVATION_MODES: Tuple[str, ...] = (
    'global',
    'local',
    'local_plus_forecast',
    'local_plus_voltage',
    'ders_local',
)


@dataclass(frozen=True)
class ObservationConfig:
    """Normalized observation-mode specification for a task or agent group."""

    mode: str
    supported_modes: Tuple[str, ...]
    global_features: Tuple[str, ...] = ()
    local_features: Tuple[str, ...] = ()
    forecast_features: Tuple[str, ...] = ()
    forecast_horizon_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode,
            'supported_modes': list(self.supported_modes),
            'global_features': list(self.global_features),
            'local_features': list(self.local_features),
            'forecast_features': list(self.forecast_features),
            'forecast_horizon_steps': self.forecast_horizon_steps,
        }


def _as_tuple(values: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if values is None:
        return ()
    return tuple(values)


def make_observation_config(
    *,
    mode: str,
    supported_modes: Iterable[str],
    global_features: Iterable[str] = (),
    local_features: Iterable[str] = (),
    forecast_features: Iterable[str] = (),
    forecast_horizon_steps: int = 0,
) -> Dict[str, Any]:
    """Create and validate a canonical observation config dict."""
    return ObservationConfig(
        mode=mode,
        supported_modes=tuple(supported_modes),
        global_features=tuple(global_features),
        local_features=tuple(local_features),
        forecast_features=tuple(forecast_features),
        forecast_horizon_steps=int(forecast_horizon_steps),
    ).to_dict()


def normalize_observation_config(raw_config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize legacy/new observation config shapes to one canonical schema."""
    config = dict(raw_config or {})

    mode = config.get('mode', config.get('default_mode', 'global'))
    supported_modes = tuple(config.get('supported_modes', (mode,)))
    if not supported_modes:
        supported_modes = (mode,)

    invalid_modes = [value for value in supported_modes if value not in OBSERVATION_MODES]
    if invalid_modes:
        raise ValueError(
            f"Unsupported observation modes {invalid_modes}. "
            f"Expected subset of {OBSERVATION_MODES}."
        )
    if mode not in OBSERVATION_MODES:
        raise ValueError(
            f"Unsupported observation mode '{mode}'. "
            f"Expected one of {OBSERVATION_MODES}."
        )
    if mode not in supported_modes:
        raise ValueError(
            f"Observation mode '{mode}' must appear in supported_modes={supported_modes}."
        )

    global_features = _as_tuple(config.get('global_features', config.get('global')))
    local_features = _as_tuple(config.get('local_features', config.get('local')))
    forecast_features = _as_tuple(config.get('forecast_features', config.get('forecast')))
    forecast_horizon_steps = int(config.get('forecast_horizon_steps', 0) or 0)

    if 'local_plus_forecast' not in supported_modes:
        forecast_features = ()
        forecast_horizon_steps = 0
    elif forecast_horizon_steps <= 0:
        raise ValueError(
            "Tasks supporting 'local_plus_forecast' must set forecast_horizon_steps > 0."
        )

    return ObservationConfig(
        mode=mode,
        supported_modes=supported_modes,
        global_features=global_features,
        local_features=local_features,
        forecast_features=forecast_features,
        forecast_horizon_steps=forecast_horizon_steps,
    ).to_dict()


def normalize_agents_observation_configs(agents_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize observation configs on top-level agents_config or nested agent groups."""
    normalized = dict(agents_config)

    if 'agent_groups' in normalized:
        groups = []
        for group in normalized['agent_groups']:
            group_cfg = dict(group)
            if 'observation' in group_cfg:
                group_cfg['observation'] = normalize_observation_config(group_cfg['observation'])
            groups.append(group_cfg)
        normalized['agent_groups'] = groups
        return normalized

    if 'observation' in normalized:
        normalized['observation'] = normalize_observation_config(normalized['observation'])

    return normalized
