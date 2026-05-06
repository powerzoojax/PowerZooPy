"""Python for-loop throughput baseline for PowerZoo.

Measures steps-per-second (sps) for the 5 standard PowerZoo task
environments using a plain Python for-loop, so the result can be merged
as the third column (``powerzoo_python``) into PowerZooJax's
``throughput_table.json``.

Tasks measured (kept in lock-step with PowerZooJax):

    tso          -> CentralizedComparisonTSOEnv  (Case118, 48 steps, Box(108))
    dso          -> make_dso_env / TaskCMDPWrapper(FlattenWrapper(PowerEnv))
    ders         -> TaskResourceMultiAgentEnv    (Case118zh, 12 agents)
    dc_microgrid -> DCMicrogridEnv               (288 steps, Box(5))
    gencos       -> GenCosMARLEnv (via _GenCosAdapter; 5 agents)

Usage::

    python -m benchmarks.throughput_python                       # all 5 tasks
    python -m benchmarks.throughput_python --task tso            # one task
    python -m benchmarks.throughput_python --n-steps 20000       # more steps
    python -m benchmarks.throughput_python --warmup-steps 100    # warmup
    python -m benchmarks.throughput_python --output PATH         # custom path

Output: ``benchmarks/results/throughput_python_results.json`` plus a
timestamped archival copy.  Schema mirrors the requirements doc:

    {
      "measured_at": "<ISO>",
      "machine": "see HARDWARE.md",
      "powerzoo_commit": "<git rev>",
      "results": [
        {"task": "tso", "env_class": "...", "n_envs": 1,
         "total_steps": 10000, "sps": 1234.5, "walltime_s": 8.1,
         "compile_time_s": null, "python_version": "3.x.y",
         "notes": ""},
        ...
      ]
    }

Design notes:
  - Pure Python only (no JAX, no vmap, no batching).
  - One ``env.step(...)`` call counts as 1 step regardless of agent count
    (matches JAX side where 1 vmap step = 1 step in the table).
  - Warmup steps are excluded from the timing window.
  - Episode boundaries trigger a Python-side ``env.reset()`` inside the
    timed loop --- this is the realistic Gymnasium usage pattern.
  - DSO / DERs fall back to synthetic profiles when real Ausgrid data
    is unavailable; the fallback is recorded in ``notes``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Trigger gym registration as a side effect; ignored if it fails (the
# task-based envs we use go through powerzoo.tasks.make_task_env, not
# gym.make, so registration failure is non-fatal for throughput.)
try:
    import powerzoo  # noqa: F401
except Exception:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASKS: Tuple[str, ...] = ("tso", "dso", "ders", "dc_microgrid", "gencos")

DEFAULT_N_STEPS = 10_000
DEFAULT_WARMUP_STEPS = 100

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Core timing helpers
# ---------------------------------------------------------------------------

def _is_marl_env(env: Any) -> bool:
    """Detect whether the env exposes a multi-agent (dict) interface."""
    if hasattr(env, "action_spaces") and isinstance(env.action_spaces, dict):
        return True
    # spaces.Dict is a Gymnasium Space whose .spaces is a dict
    act_space = getattr(env, "action_space", None)
    if act_space is not None and hasattr(act_space, "spaces") and isinstance(
        act_space.spaces, dict
    ):
        return True
    return False


def _marl_action_spaces(env: Any) -> Dict[str, Any]:
    """Return the per-agent action spaces dict regardless of attr name."""
    if hasattr(env, "action_spaces") and isinstance(env.action_spaces, dict):
        return env.action_spaces
    return env.action_space.spaces  # spaces.Dict


def _marl_active_agents(env: Any) -> List[str]:
    """Return currently-active agents; fall back to possible_agents."""
    agents = getattr(env, "agents", None)
    if agents:
        return list(agents)
    possible = getattr(env, "possible_agents", None)
    if possible:
        return list(possible)
    return list(_marl_action_spaces(env).keys())


def measure_sps_single_agent(
    env: Any,
    n_steps: int,
    seed: int = 42,
    warmup_steps: int = 0,
) -> Tuple[int, float]:
    """Measure (steps_done, walltime_s) for a Gymnasium single-agent env."""
    obs, _ = env.reset(seed=seed)

    if warmup_steps > 0:
        for i in range(warmup_steps):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=seed + i + 1)

    t0 = time.perf_counter()
    steps_done = 0
    reset_counter = 0
    while steps_done < n_steps:
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        steps_done += 1
        if terminated or truncated:
            reset_counter += 1
            obs, _ = env.reset(seed=seed + 10_000 + reset_counter)
    elapsed = time.perf_counter() - t0
    return steps_done, elapsed


def measure_sps_marl(
    env: Any,
    n_steps: int,
    seed: int = 42,
    warmup_steps: int = 0,
) -> Tuple[int, float]:
    """Measure (steps_done, walltime_s) for a MARL dict-API env."""
    action_spaces = _marl_action_spaces(env)
    obs, _ = env.reset(seed=seed)

    def _sample_actions() -> Dict[str, Any]:
        agents = _marl_active_agents(env)
        return {ag: action_spaces[ag].sample() for ag in agents}

    def _episode_done(terms: Dict[str, bool], truncs: Dict[str, bool]) -> bool:
        # RLlib-style __all__ key when present; otherwise require all-agent done.
        if "__all__" in terms or "__all__" in truncs:
            return bool(terms.get("__all__", False)) or bool(
                truncs.get("__all__", False)
            )
        if not terms and not truncs:
            return False
        return all(terms.values()) or all(truncs.values())

    if warmup_steps > 0:
        for i in range(warmup_steps):
            actions = _sample_actions()
            obs, _, terms, truncs, _ = env.step(actions)
            if _episode_done(terms, truncs):
                obs, _ = env.reset(seed=seed + i + 1)

    t0 = time.perf_counter()
    steps_done = 0
    reset_counter = 0
    while steps_done < n_steps:
        actions = _sample_actions()
        obs, _, terms, truncs, _ = env.step(actions)
        steps_done += 1
        if _episode_done(terms, truncs):
            reset_counter += 1
            obs, _ = env.reset(seed=seed + 10_000 + reset_counter)
    elapsed = time.perf_counter() - t0
    return steps_done, elapsed


def measure_sps_auto(
    env: Any,
    n_steps: int,
    seed: int = 42,
    warmup_steps: int = 0,
) -> Tuple[int, float]:
    """Dispatch to MARL or single-agent timing based on env interface."""
    if _is_marl_env(env):
        return measure_sps_marl(env, n_steps, seed=seed, warmup_steps=warmup_steps)
    return measure_sps_single_agent(env, n_steps, seed=seed, warmup_steps=warmup_steps)


# ---------------------------------------------------------------------------
# Task env factories (one per task; each returns (env, env_class_name, notes))
# ---------------------------------------------------------------------------

def _make_tso_env() -> Tuple[Any, str, str]:
    # Bypass make_task_env(): the parent class is multi-agent, so the
    # generic adapter would return TaskUCMultiAgentEnv (54 agents) instead
    # of the intended centralized Box(108,) wrapper.  Calling
    # task.create_env() honours the CentralizedComparisonTSOTask override.
    from powerzoo.tasks.middle.comparison_tso import (
        CentralizedComparisonTSOTask,
    )

    task = CentralizedComparisonTSOTask()
    env = task.create_env()
    return env, type(env).__name__, ""


def _make_dso_env() -> Tuple[Any, str, str]:
    from powerzoo.tasks.dso_task import (
        DSO_FEEDER_BUS_MAP,
        make_dso_env as _factory,
    )

    notes = ""
    try:
        env = _factory(split="train")
    except Exception as exc:
        # Real Ausgrid data unavailable: fall back to synthetic feeder shapes.
        max_steps = 48
        synthetic_shapes = {
            feeder: np.ones(max_steps, dtype=np.float32)
            for feeder in DSO_FEEDER_BUS_MAP
        }
        env = _factory(
            split="train",
            feeder_shapes=synthetic_shapes,
            max_steps=max_steps,
        )
        notes = (
            f"feeder_shapes=synthetic_constant (Ausgrid load failed: "
            f"{type(exc).__name__})"
        )
    return env, type(env).__name__, notes


def _make_ders_env() -> Tuple[Any, str, str]:
    from powerzoo.tasks import make_task_env

    env = make_task_env("marl_ders_benchmark", split="train")
    return env, type(env).__name__, ""


def _make_dc_microgrid_env() -> Tuple[Any, str, str]:
    from powerzoo.tasks import make_task_env

    env = make_task_env("dc_microgrid", workload_source="synthetic")
    return env, type(env).__name__, "workload_source=synthetic"


def _make_gencos_env() -> Tuple[Any, str, str]:
    from powerzoo.tasks import make_task_env

    env = make_task_env("gencos_bidding")
    # Adapter delegates to the underlying GenCosMARLEnv.  Report the
    # delegated class for transparency when available.
    inner = getattr(env, "_env", None)
    cls_name = type(inner).__name__ if inner is not None else type(env).__name__
    return env, cls_name, ""


_FACTORIES: Dict[str, Callable[[], Tuple[Any, str, str]]] = {
    "tso": _make_tso_env,
    "dso": _make_dso_env,
    "ders": _make_ders_env,
    "dc_microgrid": _make_dc_microgrid_env,
    "gencos": _make_gencos_env,
}


# ---------------------------------------------------------------------------
# Per-task driver
# ---------------------------------------------------------------------------

def measure_task(
    task: str,
    n_steps: int = DEFAULT_N_STEPS,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run timing for one task; never raises -- errors go into the result dict."""
    if task not in _FACTORIES:
        raise ValueError(f"Unknown task: {task!r}. Known tasks: {TASKS}")

    py_ver = sys.version.split()[0]
    base: Dict[str, Any] = {
        "task": task,
        "env_class": None,
        "n_envs": 1,
        "total_steps": 0,
        "sps": None,
        "walltime_s": None,
        "compile_time_s": None,
        "python_version": py_ver,
        "notes": "",
    }

    if verbose:
        print(f"[throughput_python] task={task} n_steps={n_steps} "
              f"warmup={warmup_steps}", flush=True)

    try:
        env, env_class, notes = _FACTORIES[task]()
        base["env_class"] = env_class
        base["notes"] = notes
    except Exception as exc:
        base["error"] = f"factory_failed: {type(exc).__name__}: {exc}"
        base["traceback"] = traceback.format_exc(limit=3)
        if verbose:
            print(f"  FAILED to construct: {base['error']}", flush=True)
        return base

    try:
        steps_done, walltime = measure_sps_auto(
            env, n_steps=n_steps, seed=seed, warmup_steps=warmup_steps,
        )
    except Exception as exc:
        base["error"] = f"rollout_failed: {type(exc).__name__}: {exc}"
        base["traceback"] = traceback.format_exc(limit=3)
        if verbose:
            print(f"  FAILED during rollout: {base['error']}", flush=True)
        return base

    base["total_steps"] = int(steps_done)
    base["walltime_s"] = float(walltime)
    base["sps"] = float(steps_done / walltime) if walltime > 0 else None

    if verbose:
        sps_str = f"{base['sps']:,.1f}" if base["sps"] is not None else "n/a"
        print(f"  -> {steps_done} steps in {walltime:.2f}s = {sps_str} sps",
              flush=True)
    return base


def measure_all(
    n_steps: int = DEFAULT_N_STEPS,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    tasks: Optional[List[str]] = None,
    seed: int = 42,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    tasks = list(tasks) if tasks else list(TASKS)
    return [
        measure_task(
            t,
            n_steps=n_steps,
            warmup_steps=warmup_steps,
            seed=seed,
            verbose=verbose,
        )
        for t in tasks
    ]


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def build_report(
    results: List[Dict[str, Any]],
    machine: str = "see HARDWARE.md",
) -> Dict[str, Any]:
    return {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": machine,
        "powerzoo_commit": _git_commit(),
        "results": results,
    }


def write_report(
    report: Dict[str, Any],
    output: Optional[Path] = None,
    archive: bool = True,
) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if output is None:
        output = _RESULTS_DIR / "throughput_python_results.json"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if archive:
        ts = time.strftime("%Y%m%d_%H%M%S")
        archive_path = _RESULTS_DIR / f"throughput_python_results_{ts}.json"
        archive_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def print_summary(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("PowerZoo Python for-loop throughput")
    print(f"  measured_at: {report['measured_at']}")
    print(f"  commit:      {report.get('powerzoo_commit')}")
    print("-" * 72)
    print(f"{'task':<14} {'env_class':<32} {'steps':>7} {'sps':>10} {'time(s)':>8}")
    print("-" * 72)
    for r in report["results"]:
        if "error" in r:
            print(f"{r['task']:<14} ERROR: {r['error']}")
            continue
        sps = r.get("sps")
        sps_str = f"{sps:,.1f}" if sps is not None else "n/a"
        wt = r.get("walltime_s")
        wt_str = f"{wt:.2f}" if wt is not None else "n/a"
        print(
            f"{r['task']:<14} {str(r.get('env_class','')):<32} "
            f"{r.get('total_steps',0):>7d} {sps_str:>10} {wt_str:>8}"
        )
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PowerZoo Python for-loop throughput baseline."
    )
    p.add_argument(
        "--task",
        choices=list(TASKS) + ["all"],
        default="all",
        help="Task to measure; 'all' (default) runs all 5.",
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N_STEPS,
        help=f"Steps per measurement (default: {DEFAULT_N_STEPS}).",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help=f"Warmup steps before timing (default: {DEFAULT_WARMUP_STEPS}).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for env.reset() and action sampling.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path for the JSON report; defaults to "
            "benchmarks/results/throughput_python_results.json."
        ),
    )
    p.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip writing the timestamped archive copy.",
    )
    p.add_argument(
        "--machine",
        default="see HARDWARE.md",
        help="Machine identifier stored in the report.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-task progress prints.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    tasks = list(TASKS) if args.task == "all" else [args.task]

    results = measure_all(
        n_steps=args.n_steps,
        warmup_steps=args.warmup_steps,
        tasks=tasks,
        seed=args.seed,
        verbose=not args.quiet,
    )
    report = build_report(results, machine=args.machine)
    out_path = write_report(report, output=args.output, archive=not args.no_archive)
    print(f"[throughput_python] wrote {out_path}", flush=True)
    print_summary(report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
