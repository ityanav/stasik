import logging
import time
from datetime import datetime, timezone, timedelta

from src.strategy.signals import Signal

logger = logging.getLogger(__name__)


class PositionCloseMixin:
    """Mixin providing position close / exit monitoring methods."""

    async def _check_closed_trades(self):
        open_trades = await self.db.get_open_trades()
        if not open_trades:
            return

        for trade in open_trades:
            if trade["category"] not in ("linear", "tbank"):
                continue
            try:
                await self._check_db_take_profit(trade)
                await self._check_breakeven(trade)
                await self._check_smart_exit(trade)
                await self._check_profit_target(trade)
                await self._check_loss_target(trade)
                await self._check_close_at_profit(trade)
                await self._check_trade_closed(trade)
            except Exception:
                logger.exception("Error checking trade %s", trade["id"])

    async def _check_trade_closed(self, trade: dict):
        # Skip if already closed by TP/SL check earlier in the loop
        current = await self.db.get_trade(trade["id"])
        if not current or current.get("status") == "closed":
            return

        symbol = trade["symbol"]
        category = trade["category"]
        if self.exchange_type == "tbank":
            positions = self.client.get_positions(symbol=symbol)
        else:
            positions = self.client.get_positions(symbol=symbol, category="linear")

        # Check if we still have a position for this symbol+side
        still_open = False
        for p in positions:
            if p["symbol"] == symbol and p["size"] > 0:
                still_open = True
                break

        if still_open:
            return

        # Position closed — get real PnL from exchange
        entry_price = trade["entry_price"]
        qty = trade["qty"]
        side = trade["side"]

        exit_price = None
        pnl = None

        try:
            closed_records = self.client.get_closed_pnl(symbol=symbol, limit=20)
            # Find matching record by entry price and side
            for rec in closed_records:
                if rec["side"] == side and abs(rec["entry_price"] - entry_price) < entry_price * 0.001:
                    exit_price = rec["exit_price"]
                    # Exchange closedPnl is GROSS (no commission) — always calculate net
                    pnl = self._calc_net_pnl(side, entry_price, exit_price, qty)
                    logger.info("Got exit from exchange for %s: net_pnl=%.2f (exit=%.4f)", symbol, pnl, exit_price)
                    break
        except Exception:
            logger.warning("Failed to get closed PnL from exchange for %s", symbol)

        # Fallback: calculate manually
        if exit_price is None:
            if self.exchange_type == "tbank":
                exit_price = self.client.get_last_price(symbol)
            else:
                exit_price = self.client.get_last_price(symbol, category="linear")
        if pnl is None:
            pnl = self._calc_net_pnl(side, entry_price, exit_price, qty)
            logger.info("Using calculated net PnL for %s: %.2f (fallback)", symbol, pnl)

        # Update DB
        balance = self.client.get_balance()
        await self.db.close_trade(trade["id"], exit_price, pnl)
        await self.db.update_daily_pnl(pnl)
        await self._record_pnl(pnl, balance)

        # Cooldown after loss
        if pnl < 0 and self._cooldown_seconds > 0:
            self._cooldowns[symbol] = time.time() + self._cooldown_seconds
            cooldown_min = self._cooldown_seconds // 60
            logger.info("Кулдаун %s: %d мин после убытка", symbol, cooldown_min)

        # Clean up breakeven tracking
        self._breakeven_done.discard(trade["id"])

        # Send notification
        icon = "🟢" if pnl >= 0 else "🔴"
        currency = "RUB" if self.exchange_type == "tbank" else "USDT"
        msk = datetime.now(timezone(timedelta(hours=3)))
        msk_time = msk.strftime("%H:%M")
        dur_str = ""
        opened_at = trade.get("opened_at")
        if opened_at:
            try:
                opened_dt = datetime.fromisoformat(opened_at)
                dur_sec = int((datetime.now(timezone.utc) - opened_dt.replace(tzinfo=timezone(timedelta(hours=3)))).total_seconds())
                if dur_sec >= 86400:
                    dur_str = f"{dur_sec // 86400}d {(dur_sec % 86400) // 3600}h"
                elif dur_sec >= 3600:
                    dur_str = f"{dur_sec // 3600}h {(dur_sec % 3600) // 60}m"
                elif dur_sec >= 60:
                    dur_str = f"{dur_sec // 60}m"
                else:
                    dur_str = f"{dur_sec}s"
            except Exception:
                pass
        direction = "LONG" if side == "Buy" else "SHORT"
        inst = self.instance_name or "BOT"
        dur_part = f" | {dur_str}" if dur_str else ""
        msg = f"{icon} {inst} | {symbol} {direction} закрыт\n   {pnl:+,.2f} {currency} (net){dur_part} | {msk_time}"
        logger.info(msg)
        await self._notify(msg)

    async def _check_close_at_profit(self, trade: dict):
        """Close position as soon as PnL > 0 (triggered by Telegram button)."""
        symbol = trade["symbol"]
        if symbol not in self._close_at_profit:
            return

        side = trade["side"]
        entry = trade["entry_price"]
        qty = trade["qty"]

        try:
            if self.exchange_type == "tbank":
                cur_price = self.client.get_last_price(symbol)
            else:
                cur_price = self.client.get_last_price(symbol, category="linear")
        except Exception:
            return

        net_pnl = self._calc_net_pnl(side, entry, cur_price, qty)

        if net_pnl <= 0:
            return

        # In net profit — close it
        close_side = "Sell" if side == "Buy" else "Buy"
        try:
            if self.exchange_type == "tbank":
                self.client.place_order(symbol=symbol, side=close_side, qty=qty)
            else:
                self.client.place_order(symbol=symbol, side=close_side, qty=qty, category="linear", reduce_only=True)
        except Exception:
            logger.exception("Close-at-profit: failed to close %s", symbol)
            return

        self._close_at_profit.discard(symbol)
        currency = "RUB" if self.exchange_type == "tbank" else "USDT"
        msk = datetime.now(timezone(timedelta(hours=3)))
        msk_time = msk.strftime("%H:%M")
        dur_str = ""
        opened_at = trade.get("opened_at")
        if opened_at:
            try:
                opened_dt = datetime.fromisoformat(opened_at)
                dur_sec = int((datetime.now(timezone.utc) - opened_dt.replace(tzinfo=timezone(timedelta(hours=3)))).total_seconds())
                if dur_sec >= 86400:
                    dur_str = f"{dur_sec // 86400}d {(dur_sec % 86400) // 3600}h"
                elif dur_sec >= 3600:
                    dur_str = f"{dur_sec // 3600}h {(dur_sec % 3600) // 60}m"
                elif dur_sec >= 60:
                    dur_str = f"{dur_sec // 60}m"
                else:
                    dur_str = f"{dur_sec}s"
            except Exception:
                pass
        direction = "LONG" if side == "Buy" else "SHORT"
        inst = self.instance_name or "BOT"
        dur_part = f" | {dur_str}" if dur_str else ""
        msg = f"🟢 {inst} | {symbol} {direction} закрыт\n   +{net_pnl:,.2f} {currency} (net){dur_part} | {msk_time}"
        logger.info(msg)
        await self._notify(msg)

    def add_close_at_profit(self, symbol: str) -> str:
        """Add symbol to close-at-profit watchlist. Returns status message."""
        self._close_at_profit.add(symbol)
        return f"⏳ {symbol}: закрою как только PnL > 0"

    def remove_close_at_profit(self, symbol: str):
        self._close_at_profit.discard(symbol)

    async def _check_profit_target(self, trade: dict):
        """Close trade when unrealized PnL reaches profit_target_usd.

        Config (risk section):
          profit_target_usd: 50   # close when uPnL >= $50
        """
        risk_cfg = self.config.get("risk", {})
        target = risk_cfg.get("profit_target_usd", 0)
        if target <= 0:
            return  # feature disabled

        symbol = trade["symbol"]
        side = trade["side"]
        entry = trade["entry_price"]
        qty = trade["qty"]

        # Get current price
        try:
            if self.exchange_type == "tbank":
                cur_price = self.client.get_last_price(symbol)
            else:
                cur_price = self.client.get_last_price(symbol, category="linear")
        except Exception:
            return

        # Calculate unrealized net PnL (after commission)
        net_pnl = self._calc_net_pnl(side, entry, cur_price, qty)

        if net_pnl < target:
            return

        # Close the position
        close_side = "Sell" if side == "Buy" else "Buy"
        try:
            if self.exchange_type == "tbank":
                self.client.place_order(symbol=symbol, side=close_side, qty=qty)
            else:
                self.client.place_order(symbol=symbol, side=close_side, qty=qty, category="linear", reduce_only=True)
        except Exception:
            logger.exception("Profit target: failed to close %s", symbol)
            return

        # Update DB
        try:
            balance = self.client.get_balance()
            await self.db.close_trade(trade["id"], cur_price, net_pnl)
            await self.db.update_daily_pnl(net_pnl)
            await self._record_pnl(net_pnl, balance)
        except Exception:
            logger.exception("Profit target: DB update failed for %s", symbol)

        direction = "ЛОНГ" if side == "Buy" else "ШОРТ"
        logger.info(
            "💰 Profit target: закрыл %s %s при net PnL +%.0f (цель: %d)",
            direction, symbol, net_pnl, target,
        )
        await self._notify(
            f"💰 Profit target: закрыл {direction} {symbol}\n"
            f"Net PnL: +{net_pnl:,.0f} USDT (цель: {target}$)"
        )

    async def _check_loss_target(self, trade: dict):
        """Close trade when unrealized loss reaches loss_target_usd.

        Config (risk section):
          loss_target_usd: 50   # close when uPnL <= -$50
        """
        risk_cfg = self.config.get("risk", {})
        target = risk_cfg.get("loss_target_usd", 0)
        if target <= 0:
            return  # feature disabled

        symbol = trade["symbol"]
        side = trade["side"]
        entry = trade["entry_price"]
        qty = trade["qty"]

        try:
            if self.exchange_type == "tbank":
                cur_price = self.client.get_last_price(symbol)
            else:
                cur_price = self.client.get_last_price(symbol, category="linear")
        except Exception:
            return

        # Net PnL (loss + commission)
        net_pnl = self._calc_net_pnl(side, entry, cur_price, qty)

        if net_pnl > -target:
            return  # loss not big enough yet

        # Close the position
        close_side = "Sell" if side == "Buy" else "Buy"
        try:
            if self.exchange_type == "tbank":
                self.client.place_order(symbol=symbol, side=close_side, qty=qty)
            else:
                self.client.place_order(symbol=symbol, side=close_side, qty=qty, category="linear", reduce_only=True)
        except Exception:
            logger.exception("Loss target: failed to close %s", symbol)
            return

        try:
            balance = self.client.get_balance()
            await self.db.close_trade(trade["id"], cur_price, net_pnl)
            await self.db.update_daily_pnl(net_pnl)
            await self._record_pnl(net_pnl, balance)
        except Exception:
            logger.exception("Loss target: DB update failed for %s", symbol)

        direction = "ЛОНГ" if side == "Buy" else "ШОРТ"
        logger.info(
            "🛑 Loss target: закрыл %s %s при net PnL %.0f (лимит: -%d)",
            direction, symbol, net_pnl, target,
        )
        await self._notify(
            f"🛑 Loss target: закрыл {direction} {symbol}\n"
            f"Net PnL: {net_pnl:,.0f} USDT (лимит: -{target}$)"
        )


    async def _check_db_take_profit(self, trade: dict):
        """Close position when price hits take_profit stored in DB (fetches price via REST)."""
        tp = trade.get("take_profit")
        if not tp or tp <= 0:
            return
        try:
            if self.exchange_type == "tbank":
                cur_price = self.client.get_last_price(trade["symbol"])
            else:
                cur_price = self.client.get_last_price(trade["symbol"], category="linear")
        except Exception:
            return
        await self._check_db_take_profit_with_price(trade, cur_price)

    async def _check_db_take_profit_with_price(self, trade: dict, cur_price: float):
        """Close position when price hits take_profit stored in DB."""
        tp = trade.get("take_profit")
        if not tp or tp <= 0 or not cur_price:
            return

        symbol = trade["symbol"]
        side = trade["side"]
        entry = trade["entry_price"]
        qty = trade["qty"]

        # Buy: close when price >= TP (price rose)
        if side == "Buy" and cur_price < tp:
            return
        # Sell: close when price <= TP (price fell)
        if side == "Sell" and cur_price > tp:
            return

        # TP hit — close position (net PnL after commission)
        net_pnl = self._calc_net_pnl(side, entry, cur_price, qty)

        # Try to close on exchange (may fail if position is phantom)
        close_side = "Sell" if side == "Buy" else "Buy"
        try:
            if self.exchange_type == "tbank":
                self.client.place_order(symbol=symbol, side=close_side, qty=qty)
            else:
                self.client.place_order(symbol=symbol, side=close_side, qty=qty, category="linear", reduce_only=True)
        except Exception:
            logger.warning("DB take-profit: no position on exchange for %s, closing in DB only", symbol)

        # Always close in DB
        try:
            balance = self.client.get_balance()
            await self.db.close_trade(trade["id"], cur_price, net_pnl)
            await self.db.update_daily_pnl(net_pnl)
            await self._record_pnl(net_pnl, balance)
        except Exception:
            logger.exception("DB take-profit: DB update failed for %s", symbol)

        direction = "ЛОНГ" if side == "Buy" else "ШОРТ"
        fee = self._calc_fee(entry, cur_price, qty)
        logger.info("🎯 TP сработал: %s %s @ %.6f (TP=%.6f, net PnL=%.2f, fee=%.2f)", direction, symbol, cur_price, tp, net_pnl, fee)
        await self._notify(
            f"🎯 Тейк-профит: {direction} {symbol}\n"
            f"Цена: {cur_price:,.6f} (TP: {tp:,.6f})\n"
            f"Net PnL: {net_pnl:+,.2f} (fee: {fee:.2f})"
        )

    async def _check_smart_exit(self, trade: dict):
        """Close trade early with small profit if signal reversed.

        Config (risk section):
          smart_exit_min: 100   # min unrealized PnL (USDT/RUB) to consider
          smart_exit_max: 500   # close if uPnL in [min, max] and signal reversed
        """
        risk_cfg = self.config.get("risk", {})
        exit_min = risk_cfg.get("smart_exit_min", 0)
        if exit_min <= 0:
            return  # feature disabled

        exit_max = risk_cfg.get("smart_exit_max", 500)

        symbol = trade["symbol"]
        side = trade["side"]
        entry = trade["entry_price"]
        qty = trade["qty"]
        category = trade["category"]

        # Get current price
        try:
            if self.exchange_type == "tbank":
                cur_price = self.client.get_last_price(symbol)
            else:
                cur_price = self.client.get_last_price(symbol, category="linear")
        except Exception:
            return

        # Calculate unrealized net PnL (after commission)
        net_pnl = self._calc_net_pnl(side, entry, cur_price, qty)

        if net_pnl < exit_min or net_pnl > exit_max:
            return

        # Check if current signal is opposite to our position
        try:
            df = self.client.get_klines(symbol, self.timeframe, limit=100, category=category)
            if df is None or df.empty:
                return
            result = self.signal_gen.generate(df, symbol)
        except Exception:
            return

        # Signal must oppose our direction
        opposite = False
        if side == "Buy" and result.signal == Signal.SELL and result.score <= -2:
            opposite = True
        elif side == "Sell" and result.signal == Signal.BUY and result.score >= 2:
            opposite = True

        if not opposite:
            return

        # Close the position
        close_side = "Sell" if side == "Buy" else "Buy"
        try:
            if self.exchange_type == "tbank":
                self.client.place_order(symbol=symbol, side=close_side, qty=qty)
            else:
                self.client.place_order(symbol=symbol, side=close_side, qty=qty, category="linear", reduce_only=True)
        except Exception:
            logger.exception("Smart exit: failed to close %s", symbol)
            return

        # Update DB
        try:
            balance = self.client.get_balance()
            await self.db.close_trade(trade["id"], cur_price, net_pnl)
            await self.db.update_daily_pnl(net_pnl)
            await self._record_pnl(net_pnl, balance)
        except Exception:
            logger.exception("Smart exit: DB update failed for %s", symbol)

        direction = "ЛОНГ" if side == "Buy" else "ШОРТ"
        logger.info(
            "🧠 Smart exit: закрыл %s %s при net PnL +%.0f (сигнал развернулся: score=%d)",
            direction, symbol, net_pnl, result.score,
        )
        await self._notify(
            f"🧠 Smart exit: закрыл {direction} {symbol}\n"
            f"Net PnL: +{net_pnl:,.0f} (сигнал развернулся, score={result.score})"
        )

    async def _check_breakeven(self, trade: dict):
        """Move SL to entry when profit reaches breakeven_activation threshold."""
        trade_id = trade["id"]
        if trade_id in self._breakeven_done:
            return

        if self._breakeven_activation <= 0:
            return

        symbol = trade["symbol"]
        entry = trade["entry_price"]
        side = trade["side"]

        if self.exchange_type == "tbank":
            current_price = self.client.get_last_price(symbol)
        else:
            current_price = self.client.get_last_price(symbol, category="linear")

        # Net profit % (after commission)
        qty = trade["qty"]
        net_pnl = self._calc_net_pnl(side, entry, current_price, qty)
        position_value = entry * qty if entry > 0 else 1
        net_profit_pct = net_pnl / position_value

        if net_profit_pct < self._breakeven_activation:
            return

        # Move SL to entry (breakeven)
        # If multiple positions per symbol, update DB SL (polling will handle it)
        open_for_symbol = [t for t in await self.db.get_open_trades() if t["symbol"] == symbol]
        if len(open_for_symbol) > 1:
            await self.db.update_stop_loss(trade_id, entry)
        else:
            try:
                if self.exchange_type == "bybit":
                    self.client.session.set_trading_stop(
                        category="linear", symbol=symbol,
                        stopLoss=str(round(entry, 6)), positionIdx=self.client._pos_idx(side),
                    )
                else:
                    logger.info("Breakeven SL for %s: exchange does not support SL move, skipping", symbol)
                    return
            except Exception:
                logger.warning("Failed to set breakeven SL for %s", symbol)
                return

        self._breakeven_done.add(trade_id)
        msg = (
            f"🛡️ Безубыток {symbol}\n"
            f"Net прибыль: {net_profit_pct*100:.2f}% → SL перенесён на вход ({entry})"
        )
        logger.info(msg)
        await self._notify(msg)
