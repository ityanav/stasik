import logging

import pandas as pd

logger = logging.getLogger(__name__)


class TurtleMixin:
    """Mixin providing Turtle strategy exit and trailing SL methods."""

    async def _check_turtle_exit(self, symbol: str, df, state: dict, category: str):
        """Check if price broke exit channel — close all units."""
        price = df["close"].iloc[-1]
        side = state["side"]
        system = state.get("system", 1)

        # Use appropriate exit channel based on entry system
        from src.strategy.indicators import calculate_donchian
        if system == 1:
            exit_period = self.signal_gen.exit_period_s1
        else:
            exit_period = self.signal_gen.exit_period_s2

        exit_upper, exit_lower = calculate_donchian(df, exit_period)
        # Use previous bar's exit channel (exclude current forming bar)
        exit_high = exit_upper.iloc[-2] if pd.notna(exit_upper.iloc[-2]) else None
        exit_low = exit_lower.iloc[-2] if pd.notna(exit_lower.iloc[-2]) else None

        should_exit = False
        if side == "Buy" and exit_low is not None and price < exit_low:
            should_exit = True
            logger.info("Turtle EXIT: %s price %.2f < exit low %.2f (S%d, %d-period)",
                        symbol, price, exit_low, system, exit_period)
        elif side == "Sell" and exit_high is not None and price > exit_high:
            should_exit = True
            logger.info("Turtle EXIT: %s price %.2f > exit high %.2f (S%d, %d-period)",
                        symbol, price, exit_high, system, exit_period)

        if should_exit:
            await self._close_all_turtle_units(symbol, category, price)

    async def _close_all_turtle_units(self, symbol: str, category: str, exit_price: float = 0):
        """Close all trades for symbol and record PnL. Update System 1 filter."""
        open_trades = await self.db.get_open_trades()
        symbol_trades = [t for t in open_trades if t["symbol"] == symbol]
        if not symbol_trades:
            self._turtle_state.pop(symbol, None)
            return

        state = self._turtle_state.get(symbol, {})
        side = state.get("side", symbol_trades[0]["side"])
        close_side = "Sell" if side == "Buy" else "Buy"

        total_pnl = 0
        total_qty = 0
        for trade in symbol_trades:
            qty = trade["qty"]
            entry = trade["entry_price"]
            total_qty += qty

        # Close entire position at once
        if total_qty > 0:
            try:
                self.client.place_order(
                    symbol=symbol, side=close_side, qty=total_qty,
                    category=category, reduce_only=True,
                )
            except Exception:
                logger.exception("Turtle: failed to close all units for %s", symbol)

        # Get actual exit price
        if exit_price <= 0:
            try:
                exit_price = self.client.get_last_price(symbol, category=category)
            except Exception:
                exit_price = 0

        # Close each trade in DB
        balance = self.client.get_balance()
        for trade in symbol_trades:
            qty = trade["qty"]
            entry = trade["entry_price"]
            pnl = self._calc_net_pnl(side, entry, exit_price, qty)
            total_pnl += pnl
            try:
                await self.db.close_trade(trade["id"], exit_price, pnl)
                await self.db.update_daily_pnl(pnl)
                await self._record_pnl(pnl, balance)
            except Exception:
                logger.exception("Turtle: DB error closing trade #%d", trade["id"])

        # System 1 filter: remember if this breakout was profitable
        self._turtle_last_breakout[symbol] = total_pnl > 0

        # Clean up state
        units = state.get("units", len(symbol_trades))
        self._turtle_state.pop(symbol, None)

        emoji = "+" if total_pnl >= 0 else ""
        direction = "LONG" if side == "Buy" else "SHORT"
        msg = (
            f"🐢 Turtle CLOSED {direction} {symbol}\n"
            f"Exit: {exit_price:.2f} | Units: {units}\n"
            f"PnL: {emoji}{total_pnl:.2f} USDT\n"
            f"Balance: {balance:,.0f} USDT"
        )
        logger.info(msg)
        await self._notify(msg)

    async def _turtle_update_trailing_sl(self, symbol: str, new_sl: float):
        """Update SL on all open trades for a symbol (when pyramiding)."""
        open_trades = await self.db.get_open_trades()
        for trade in open_trades:
            if trade["symbol"] == symbol:
                try:
                    await self.db.update_trade(trade["id"], stop_loss=new_sl)
                except Exception:
                    logger.warning("Turtle: failed to update SL for trade #%d", trade["id"])
