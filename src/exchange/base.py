import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ExchangeClient(ABC):
    """Abstract base class for exchange clients."""

    @abstractmethod
    def get_balance(self, currency: str = "USDT") -> float:
        ...

    @abstractmethod
    def get_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        """Return DataFrame with columns: timestamp, open, high, low, close, volume."""
        ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "Market",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> dict:
        ...

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> dict:
        ...

    @abstractmethod
    def get_positions(self, symbol: Optional[str] = None) -> list[dict]:
        """Return list of dicts with: symbol, side, size, entry_price, unrealised_pnl."""
        ...

    @abstractmethod
    def get_last_price(self, symbol: str) -> float:
        ...

    @abstractmethod
    def get_instrument_info(self, symbol: str) -> dict:
        """Return dict with: min_qty, max_qty, qty_step, tick_size, lot_size (optional)."""
        ...

    @abstractmethod
    def get_closed_pnl(self, symbol: str, limit: int = 10) -> list[dict]:
        ...

    # Optional methods with default implementations
    def set_leverage(self, symbol: str, leverage: int):
        pass

    def set_trailing_stop(
        self, symbol: str, trailing_stop: float, active_price: float
    ):
        pass

    def get_funding_rate(self, symbol: str) -> float:
        return 0.0

    def get_orderbook(self, symbol: str, limit: int = 50) -> dict:
        """Return {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}."""
        return {"bids": [], "asks": []}

    def get_used_margin(self, symbols: list[str] | None = None) -> float:
        """Sum of initial margin for given symbols. Returns 0 by default (disabled)."""
        return 0.0
