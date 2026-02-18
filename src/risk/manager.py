import logging
import math
from datetime import date

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config: dict):
        risk = config["risk"]
        self.risk_per_trade = risk["risk_per_trade"] / 100
        self.stop_loss_pct = risk["stop_loss"] / 100
        self.take_profit_pct = risk["take_profit"] / 100
        self.max_open_positions = risk["max_open_positions"]
        self.max_daily_loss_pct = risk["max_daily_loss"] / 100

        self._daily_pnl: float = 0.0
        self._daily_date: date = date.today()
        self._halted: bool = False

    # ── Daily loss tracking ──────────────────────────────────

    def _check_day_reset(self):
        today = date.today()
        if today != self._daily_date:
            self._daily_pnl = 0.0
            self._daily_date = today
            self._halted = False
            logger.info("Daily PnL reset for %s", today)

    def record_pnl(self, pnl: float, balance: float):
        self._check_day_reset()
        self._daily_pnl += pnl
        if balance > 0 and abs(self._daily_pnl) / balance >= self.max_daily_loss_pct and self._daily_pnl < 0:
            self._halted = True
            logger.warning(
                "Daily loss limit reached: %.2f (%.1f%% of %.2f)",
                self._daily_pnl, self._daily_pnl / balance * 100, balance,
            )

    @property
    def is_halted(self) -> bool:
        self._check_day_reset()
        return self._halted

    @property
    def daily_pnl(self) -> float:
        self._check_day_reset()
        return self._daily_pnl

    # ── Pre-trade checks ─────────────────────────────────────

    def can_open_position(self, open_positions_count: int) -> bool:
        if self.is_halted:
            logger.info("Trading halted — daily loss limit reached")
            return False
        if open_positions_count >= self.max_open_positions:
            logger.info(
                "Max open positions reached (%d/%d)",
                open_positions_count, self.max_open_positions,
            )
            return False
        return True

    # ── Position sizing ──────────────────────────────────────

    def calculate_position_size(
        self,
        balance: float,
        price: float,
        qty_step: float,
        min_qty: float,
    ) -> float:
        risk_amount = balance * self.risk_per_trade
        position_value = risk_amount / self.stop_loss_pct
        qty = position_value / price
        # Round down to qty_step
        qty = math.floor(qty / qty_step) * qty_step
        if qty < min_qty:
            return 0.0
        return round(qty, 8)

    # ── SL / TP ──────────────────────────────────────────────

    def calculate_sl_tp(
        self, price: float, side: str
    ) -> tuple[float, float]:
        if side == "Buy":
            sl = price * (1 - self.stop_loss_pct)
            tp = price * (1 + self.take_profit_pct)
        else:
            sl = price * (1 + self.stop_loss_pct)
            tp = price * (1 - self.take_profit_pct)
        return round(sl, 6), round(tp, 6)
