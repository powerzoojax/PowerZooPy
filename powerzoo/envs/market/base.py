from abc import ABC, abstractmethod
from typing import Any, Dict


class MarketEnv(ABC):
    """Market environment base class independent from physical grid.

    Defines interfaces for market clearing, settlement and revenue calculation.

    Standard call sequence per time step::

        1. market.step(bidding)   — receive bids (overwrites ``last_bids``)
        2. market.clear()         — run clearing
        3. market.settle()        — settle trades
        4. market.revenue()       — compute revenues
    """

    def __init__(self):
        self.last_bids: Dict[str, Any] = {}

    @abstractmethod
    def step(self, bidding: Dict[str, Any]) -> None:
        """Accept bids for current time step."""
        ...

    @abstractmethod
    def clear(self) -> Dict[str, Any]:
        """Run market clearing."""
        ...

    @abstractmethod
    def settle(self) -> Dict[str, Any]:
        """Settle transactions."""
        ...

    @abstractmethod
    def revenue(self) -> Dict[str, Any]:
        """Compute revenues per participant."""
        ...
