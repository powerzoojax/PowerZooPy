"""P1-β regression test: SBX algorithms get registered next to SB3.

If sbx is installed, ``Trainer.ALGORITHMS`` must include ``SBX_PPO`` /
``SBX_SAC`` / ``SBX_TD3`` after a Trainer instance is created (the
algorithms are populated in ``Trainer.__init__``).

If sbx is NOT installed, ``Trainer`` must still work with SB3 only —
the SBX import block is silently skipped.
"""

from __future__ import annotations

import importlib.util

import pytest


SBX_AVAILABLE = importlib.util.find_spec("sbx") is not None


@pytest.mark.skipif(not SBX_AVAILABLE, reason="sbx not installed; nothing to register")
def test_sbx_registered_alongside_sb3():
    from powerzoo.rl import Trainer

    # Building a Trainer triggers the lazy ALGORITHMS populate.
    t = Trainer("battery_arbitrage", total_timesteps=10)
    assert "PPO" in Trainer.ALGORITHMS
    assert "SBX_PPO" in Trainer.ALGORITHMS, (
        "sbx is installed but Trainer.ALGORITHMS does not contain SBX_PPO"
    )
    assert "SBX_SAC" in Trainer.ALGORITHMS
    assert "SBX_TD3" in Trainer.ALGORITHMS


def test_sbx_register_does_not_break_sb3():
    """Even if sbx isn't installed, SB3 algorithms must still be available."""
    from powerzoo.rl import Trainer

    t = Trainer("battery_arbitrage", total_timesteps=10)
    for k in ("PPO", "SAC", "TD3"):
        assert k in Trainer.ALGORITHMS
