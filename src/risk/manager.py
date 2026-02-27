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

        # ATR clamp ranges (configurable for swing vs scalp)
        self.atr_sl_min = risk.get("atr_sl_min", 0.3) / 100
        self.atr_sl_max = risk.get("atr_sl_max", 3.0) / 100
        self.atr_tp_min = risk.get("atr_tp_min", 0.5) / 100
        self.atr_tp_max = risk.get("atr_tp_max", 5.0) / 100
        self.atr_trail_min = risk.get("atr_trail_min", 0.2) / 100
        self.atr_trail_max = risk.get("atr_trail_max", 2.0) / 100
        self.max_position_margin_pct = risk.get("max_position_margin_pct", 30) / 100
        self.max_risk_per_trade = risk.get("max_risk_per_trade", 0)  # absolute $ cap per trade (0 = off)
        self.daily_drawdown_from_profit = risk.get("daily_drawdown_from_profit", 0) / 100  # e.g. 50 = halt if lost 50% of peak daily profit

        # Commission rate (0 for Bybit, e.g. 0.04 for T-Invest = 0.04%)
        self.commission_rate = risk.get("commission_rate", 0.0) / 100

        self._daily_pnl: float = 0.0
        self._daily_peak_pnl: float = 0.0
        self._daily_date: date = date.today()
        self._halted: bool = False

    # ── Daily loss tracking ──────────────────────────────────

    def _check_day_reset(self):
        today = date.today()
        if today != self._daily_date:
            self._daily_pnl = 0.0
            self._daily_peak_pnl = 0.0
            self._daily_date = today
            self._halted = False
            logger.info("Daily PnL reset for %s", today)

    def record_pnl(self, pnl: float, balance: float):
        self._check_day_reset()
        self._daily_pnl += pnl
        # Classic daily loss limit (% of balance); 0 = disabled
        if self.max_daily_loss_pct > 0 and balance > 0 and abs(self._daily_pnl) / balance >= self.max_daily_loss_pct and self._daily_pnl < 0:
            self._halted = True
            logger.warning(
                "Daily loss limit reached: %.2f (%.1f%% of %.2f)",
                self._daily_pnl, self._daily_pnl / balance * 100, balance,
            )

    def check_global_drawdown(self, global_daily_pnl: float):
        """Check drawdown from peak across ALL bots. Called by engine after each trade."""
        if self.daily_drawdown_from_profit <= 0:
            return
        if global_daily_pnl > self._daily_peak_pnl:
            self._daily_peak_pnl = global_daily_pnl
        if self._daily_peak_pnl > 0 and global_daily_pnl <= self._daily_peak_pnl * (1 - self.daily_drawdown_from_profit):
            self._halted = True
            logger.warning(
                "Global drawdown from profit: peak=%.2f now=%.2f (lost %.0f%% of all-bots profit)",
                self._daily_peak_pnl, global_daily_pnl,
                (1 - global_daily_pnl / self._daily_peak_pnl) * 100,
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
        sl_pct: float | None = None,
        leverage: int = 1,
    ) -> float:
        risk_amount = balance * self.risk_per_trade
        if self.max_risk_per_trade > 0:
            risk_amount = min(risk_amount, self.max_risk_per_trade)
        effective_sl = sl_pct if sl_pct is not None else self.stop_loss_pct
        if effective_sl <= 0:
            effective_sl = self.stop_loss_pct
        position_value = risk_amount / effective_sl
        # Cap notional so margin doesn't exceed max_position_margin_pct of balance
        max_notional = balance * leverage * self.max_position_margin_pct
        if position_value > max_notional:
            logger.info(
                "Position capped: %.0f -> %.0f USDT (margin limit %.0f%%)",
                position_value, max_notional, self.max_position_margin_pct * 100,
            )
            position_value = max_notional
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

    # ── ATR-based SL / TP ─────────────────────────────────────

    def calculate_sl_tp_atr(
        self, price: float, side: str, atr: float,
        sl_mult: float = 1.5, tp_mult: float = 3.0,
    ) -> tuple[float, float, float, float]:
        """ATR-based SL/TP with clamping. Returns (sl, tp, sl_pct, tp_pct)."""
        sl_dist = atr * sl_mult
        tp_dist = atr * tp_mult

        sl_pct = sl_dist / price
        tp_pct = tp_dist / price

        sl_pct = max(self.atr_sl_min, min(self.atr_sl_max, sl_pct))
        tp_pct = max(self.atr_tp_min, min(self.atr_tp_max, tp_pct))

        if side == "Buy":
            sl = price * (1 - sl_pct)
            tp = price * (1 + tp_pct)
        else:
            sl = price * (1 + sl_pct)
            tp = price * (1 - tp_pct)

        return round(sl, 6), round(tp, 6), sl_pct, tp_pct

    def min_profit_target_pct(self) -> float:
        """Minimum TP% to cover round-trip commission (entry + exit)."""
        return self.commission_rate * 2

    def net_pnl(self, gross_pnl: float, position_value: float) -> float:
        """PnL after deducting estimated commission."""
        commission = position_value * self.commission_rate * 2
        return gross_pnl - commission

    def calculate_trailing_distance_atr(
        self, atr: float, price: float, mult: float = 1.0,
    ) -> float:
        """ATR-based trailing distance, clamped to configurable range."""
        trail_pct = (atr * mult) / price
        trail_pct = max(self.atr_trail_min, min(self.atr_trail_max, trail_pct))
        return price * trail_pct
