"""Pre-computed baseline scores for PowerZoo benchmark tasks.

Design rationale (answers the generality question)
---------------------------------------------------
Hardcoding baselines into the *environment* itself would reduce generality.
Instead, baselines live here — a separate, optional module — following common
RL benchmark practice (environment code stays generic; baselines ship for the
default evaluation protocol):

  * The environment is generic and reusable by anyone.
  * The *benchmark* module ships task-specific baselines for the *default*
    protocol (fixed seed, fixed n_episodes, fixed data split).
  * Researchers can add their own entries or override via the JSON file.

How baselines are computed
--------------------------
Run ``python -m powerzoo.benchmarks.compute`` to regenerate.
Results are stored in ``powerzoo/benchmarks/baselines.json``.

The JSON file ships with pre-filled values from the reference runs.
If the file is absent or a key is missing, ``normalized_score`` returns
``None`` with a warning rather than crashing.

Score formula  (linear between random and oracle returns)
----------------------------------------------------------
    normalized_score = (policy_return - random_return)
                       / (oracle_return - random_return)

  * 0.0 corresponds to a completely random policy.
  * 1.0 corresponds to the oracle (optimal given perfect foresight).
  * Negative values mean *worse* than random.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Location of the JSON file that stores pre-computed baselines
# ---------------------------------------------------------------------------
_BASELINE_FILE = Path(__file__).parent / "baselines.json"

# ---------------------------------------------------------------------------
# In-memory fallback: populated during ``python -m powerzoo.benchmarks.compute``
# and shipped in baselines.json.  Values of None mean "not yet computed".
# ---------------------------------------------------------------------------
_DEFAULT_BASELINES: Dict[str, Dict[str, Optional[float]]] = {
    # task_id → {"random": float | None, "oracle": float | None}
    "marl_opf":              {"random": None, "oracle": None},
    "marl_opf_7d":           {"random": None, "oracle": None},
    "marl_opf_118":          {"random": None, "oracle": None},
    "marl_der_arbitrage":    {"random": None, "oracle": None},
    "marl_der_arbitrage_7d": {"random": None, "oracle": None},
    "marl_ev_v2g":           {"random": None, "oracle": None},
    "marl_ev_v2g_1d":        {"random": None, "oracle": None},
}


def _load_baselines() -> Dict[str, Dict[str, Optional[float]]]:
    """Load baselines from JSON file, fall back to in-memory defaults."""
    if _BASELINE_FILE.exists():
        try:
            with open(_BASELINE_FILE, "r") as f:
                data = json.load(f)
            # Merge with defaults (so new tasks added to defaults are present)
            merged = dict(_DEFAULT_BASELINES)
            merged.update(data)
            return merged
        except Exception as exc:
            logger.warning("Could not load baselines.json: %s — using defaults.", exc)
    return dict(_DEFAULT_BASELINES)


def _save_baselines(baselines: Dict[str, Dict[str, Optional[float]]]) -> None:
    """Persist baselines to JSON file."""
    _BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_BASELINE_FILE, "w") as f:
        json.dump(baselines, f, indent=2)
    logger.info("Baselines saved to %s", _BASELINE_FILE)


# Lazily loaded at first call
_BASELINES: Optional[Dict] = None


def _get_baselines() -> Dict[str, Dict[str, Optional[float]]]:
    global _BASELINES
    if _BASELINES is None:
        _BASELINES = _load_baselines()
    return _BASELINES


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_baseline(task_id: str) -> Dict[str, Optional[float]]:
    """Return the random / oracle baseline scores for *task_id*.

    Args:
        task_id: Task name as registered in ``powerzoo.tasks`` (e.g.
                 ``'marl_opf'``).

    Returns:
        ``{"random": float | None, "oracle": float | None}``
    """
    baselines = _get_baselines()
    if task_id not in baselines:
        raise KeyError(
            f"No baseline registered for task '{task_id}'. "
            f"Available: {sorted(baselines.keys())}"
        )
    return dict(baselines[task_id])


def normalized_score(task_id: str, policy_return: float) -> Optional[float]:
    """Compute normalized score from random and oracle baselines.

    normalized_score = (policy_return − random_return)
                       / (oracle_return − random_return)

    Returns ``None`` (with a warning) when baselines are not yet computed
    for *task_id*.  Run ``python -m powerzoo.benchmarks.compute`` to
    generate them.

    Args:
        task_id:       Task name (e.g. ``'marl_opf'``).
        policy_return: Mean episode return of the policy being evaluated.

    Returns:
        Normalised score in (-∞, 1], or ``None`` if baselines are missing.

    Example::

        from powerzoo.benchmarks import normalized_score
        ns = normalized_score('marl_opf', -450.0)
        print(f"Score: {ns:.3f}")
    """
    try:
        bl = get_baseline(task_id)
    except KeyError:
        warnings.warn(
            f"normalized_score: unknown task '{task_id}'. Returning None.",
            UserWarning, stacklevel=2,
        )
        return None

    rand, oracle = bl["random"], bl["oracle"]
    if rand is None or oracle is None:
        warnings.warn(
            f"Baselines for '{task_id}' have not been computed yet.  "
            "Run:  python -m powerzoo.benchmarks.compute  to generate them.  "
            "Returning None.",
            UserWarning, stacklevel=2,
        )
        return None

    denom = oracle - rand
    if abs(denom) < 1e-8:
        warnings.warn(
            f"Oracle and random baselines for '{task_id}' are identical "
            f"({rand:.4f}). Returning None.",
            UserWarning, stacklevel=2,
        )
        return None

    return (policy_return - rand) / denom


def register_baseline(
    task_id: str,
    random_return: float,
    oracle_return: float,
    save: bool = True,
) -> None:
    """Register (or update) baseline scores for a task.

    Useful for custom tasks or when re-running the compute script.

    Args:
        task_id:       Task name.
        random_return: Mean episode return of a uniform random policy.
        oracle_return: Mean episode return of the oracle (optimal) policy.
        save:          Persist to ``baselines.json`` immediately (default True).
    """
    baselines = _get_baselines()
    baselines[task_id] = {"random": random_return, "oracle": oracle_return}
    if save:
        _save_baselines(baselines)
    logger.info(
        "Registered baseline for '%s': random=%.4f  oracle=%.4f",
        task_id, random_return, oracle_return,
    )


def list_baselines() -> Dict[str, Dict[str, Optional[float]]]:
    """Return a copy of all registered baselines."""
    return dict(_get_baselines())
