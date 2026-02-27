import asyncio
import logging
from datetime import datetime, timedelta, timezone

from src.strategy.signals import Trend

logger = logging.getLogger(__name__)

MSK = timezone(timedelta(hours=3))


class MonitoringMixin:
    """Mixin providing SL/TP monitoring and position reconciliation methods."""

    async def _run_sl_tp_monitor(self):
        """Real-time SL/TP monitor using WebSocket ticker stream.

        Subscribes to live price updates and checks SL/TP on every tick
        (~100ms latency). Falls back to 10s polling if WS unavailable.
        """
        if self.exchange_type == "tbank":
            await self._run_sl_tp_poll()
            return

        # Demo accounts don't support WebSocket — use fast polling
        demo = self.config.get("bybit", {}).get("demo", False)
        if demo:
            logger.info("SL/TP monitor: demo mode, using 5s polling")
            await self._run_sl_tp_poll()
            return

        # Track subscribed symbols to re-subscribe when positions change
        self._ws_prices: dict[str, float] = {}
        self._ws_subscribed: set[str] = set()
        self._ws = None

        try:
            from pybit.unified_trading import WebSocket
            testnet = self.config.get("bybit", {}).get("testnet", False)
            self._ws = WebSocket(testnet=testnet, channel_type="linear")
            logger.info("SL/TP monitor: WebSocket connected")
        except Exception:
            logger.warning("SL/TP monitor: WebSocket failed, falling back to polling")
            await self._run_sl_tp_poll()
            return

        # Refresh subscriptions & check SL/TP every 2s
        self._ws_reconcile_counter = 0
        while self._running:
            try:
                # Get current open trades with SL/TP
                open_trades = await self.db.get_open_trades()
                needed_symbols = set()
                for t in open_trades:
                    sl = t.get("stop_loss") or 0
                    tp = t.get("take_profit") or 0
                    if sl > 0 or tp > 0:
                        needed_symbols.add(t["symbol"])

                # Subscribe to new symbols
                new_syms = needed_symbols - self._ws_subscribed
                old_syms = self._ws_subscribed - needed_symbols
                for sym in new_syms:
                    try:
                        self._ws.ticker_stream(symbol=sym,
                                               callback=self._ws_on_ticker)
                        self._ws_subscribed.add(sym)
                        logger.info("SL/TP WS: subscribed %s", sym)
                    except Exception:
                        logger.warning("SL/TP WS: subscribe failed %s", sym)
                # Unsubscribe closed symbols
                for sym in old_syms:
                    try:
                        self._ws.unsubscribe(args=[f"tickers.{sym}"],
                                             channel_type="linear")
                    except Exception:
                        pass
                    self._ws_subscribed.discard(sym)
                    self._ws_prices.pop(sym, None)

                # Check SL/TP using WS prices
                for trade in open_trades:
                    if not self._running:
                        break
                    sym = trade["symbol"]
                    cur_price = self._ws_prices.get(sym)
                    if not cur_price:
                        continue
                    tp = trade.get("take_profit") or 0
                    if tp > 0:
                        try:
                            await self._check_db_take_profit_with_price(trade, cur_price)
                        except Exception:
                            logger.exception("TP monitor error %s", sym)

                # Trend reversal exit (every 15th iteration = ~30s)
                self._ws_reconcile_counter += 1
                if open_trades and self._ws_reconcile_counter % 15 == 0:
                    for trade in open_trades:
                        cur_price = self._ws_prices.get(trade["symbol"])
                        if cur_price:
                            await self._check_trend_exit(trade, cur_price)

                # Detect manually closed positions (every 15th iteration = ~30s)
                if open_trades and self._ws_reconcile_counter >= 15:
                    self._ws_reconcile_counter = 0
                    await self._detect_closed_positions(open_trades)
            except Exception:
                logger.exception("SL/TP WS monitor error")
            await asyncio.sleep(2)

    def _ws_on_ticker(self, message: dict):
        """Callback for WebSocket ticker updates."""
        try:
            data = message.get("data", {})
            symbol = data.get("symbol", "")
            last_price = data.get("lastPrice")
            if symbol and last_price:
                self._ws_prices[symbol] = float(last_price)
        except Exception:
            pass

    async def _run_sl_tp_poll(self):
        """Fast polling SL/TP check every 5 seconds."""
        self._reconcile_poll_counter = 0
        while self._running:
            try:
                open_trades = await self.db.get_open_trades()
                for trade in open_trades:
                    if not self._running:
                        break
                    tp = trade.get("take_profit") or 0
                    if tp > 0:
                        try:
                            await self._check_db_take_profit(trade)
                        except Exception:
                            logger.exception("TP poll error %s", trade.get("symbol"))
                # Fast trade timeout: close if not in profit after N minutes
                stale_timeout = getattr(self, '_stale_timeout_min', 0)
                if stale_timeout > 0:
                    for trade in open_trades:
                        await self._check_stale_trade(trade, stale_timeout)

                # Trend reversal exit (every 6th poll = ~30s)
                self._reconcile_poll_counter += 1
                if open_trades and self._reconcile_poll_counter % 6 == 0:
                    for trade in open_trades:
                        await self._check_trend_exit(trade)

                # Detect manually closed positions (every 6th poll = ~30s)
                if open_trades and self._reconcile_poll_counter >= 6:
                    self._reconcile_poll_counter = 0
                    await self._detect_closed_positions(open_trades)
            except Exception:
                logger.exception("SL/TP poll error")
            await asyncio.sleep(5)

    async def _detect_closed_positions(self, open_trades: list[dict]):
        """Detect positions closed externally (manually or by exchange SL/TP).

        Runs every ~30s inside the poll loop. Fetches exchange positions once
        and compares with DB open trades.
        """
        try:
            if self.exchange_type == "tbank":
                exchange_positions = self.client.get_positions()
            else:
                exchange_positions = self.client.get_positions(category="linear")

            exch_sizes = {}  # (symbol, side) -> total exchange size
            for p in exchange_positions:
                if p["size"] > 0:
                    key = (p["symbol"], p["side"])
                    exch_sizes[key] = p["size"]

            # Sum DB qty per (symbol, side)
            db_qty = {}
            for trade in open_trades:
                key = (trade["symbol"], trade["side"])
                db_qty[key] = db_qty.get(key, 0) + trade["qty"]

            for trade in open_trades:
                key = (trade["symbol"], trade["side"])
                if key not in exch_sizes:
                    # Fully closed on exchange
                    logger.info("Detected externally closed position: %s %s (trade #%d)",
                                trade["side"], trade["symbol"], trade["id"])
                    try:
                        await self._check_trade_closed(trade)
                    except Exception:
                        logger.exception("Failed to process externally closed trade #%d", trade["id"])
                elif db_qty.get(key, 0) > 0 and exch_sizes[key] < db_qty[key] * 0.5:
                    logger.warning("Exchange size %.4f < DB total %.4f for %s %s — partial external close?",
                                   exch_sizes[key], db_qty[key], trade["side"], trade["symbol"])
        except Exception:
            logger.exception("_detect_closed_positions error")

    async def _check_trend_exit(self, trade: dict, cur_price: float = None):
        """Close position when HTF trend reverses against it.

        When htf_filter is on, positions only open with the trend. If the
        trend later flips against the position, we close immediately —
        no point holding a counter-trend position.
        """
        if not self.config.get("strategy", {}).get("htf_filter", True):
            return

        symbol = trade["symbol"]
        side = trade["side"]
        entry = trade["entry_price"]

        # Current price
        if cur_price is None:
            try:
                if self.exchange_type == "tbank":
                    cur_price = self.client.get_last_price(symbol)
                else:
                    cur_price = (getattr(self, '_ws_prices', {}).get(symbol)
                                 or self.client.get_last_price(symbol, category="linear"))
            except Exception:
                return
        if not cur_price or cur_price <= 0:
            return

        # HTF trend
        category = "tbank" if self.exchange_type == "tbank" else "linear"
        htf_trend, _, _ = self._get_htf_data(symbol, category)

        # Is trend against position?
        if side == "Buy" and htf_trend != Trend.BEARISH:
            return
        if side == "Sell" and htf_trend != Trend.BULLISH:
            return

        # PnL estimate
        if side == "Buy":
            pnl_pct = (cur_price - entry) / entry * 100
        else:
            pnl_pct = (entry - cur_price) / entry * 100

        inst = getattr(self, 'instance_name', 'BOT')
        logger.info("%s Trend exit: %s %s (trend=%s, pnl=%.2f%%, price=%.4f) — closing",
                    inst, side, symbol, htf_trend.value, pnl_pct, cur_price)

        close_side = "Sell" if side == "Buy" else "Buy"
        qty = trade["qty"]
        try:
            if self.exchange_type == "tbank":
                self.client.place_order(symbol=symbol, side=close_side, qty=qty)
            else:
                self.client.place_order(symbol=symbol, side=close_side, qty=qty,
                                        category="linear", reduce_only=True)
            msg = (f"🔄 {inst} | Trend exit {side} {symbol}\n"
                   f"Тренд развернулся ({htf_trend.value}) — закрыт\n"
                   f"PnL: {pnl_pct:+.2f}%")
            await self._notify(msg)
        except Exception:
            logger.exception("Failed to close trend-exit trade %s %s", side, symbol)

    async def _check_stale_trade(self, trade: dict, timeout_min: int):
        """Close position if not in profit after timeout_min minutes."""
        opened_at = trade.get("opened_at")
        if not opened_at:
            return
        try:
            opened_dt = datetime.fromisoformat(opened_at)
            if opened_dt.tzinfo is None:
                opened_dt = opened_dt.replace(tzinfo=MSK)
            age_sec = (datetime.now(timezone.utc) - opened_dt).total_seconds()
            if age_sec < timeout_min * 60:
                return

            symbol = trade["symbol"]
            side = trade["side"]
            entry = trade["entry_price"]

            # Get current price
            if self.exchange_type == "tbank":
                cur_price = self.client.get_last_price(symbol)
            else:
                cur_price = self._ws_prices.get(symbol) or self.client.get_last_price(symbol, category="linear")

            if not cur_price or cur_price <= 0:
                return

            # Check if in profit (gross, before commission)
            if side == "Buy":
                in_profit = cur_price > entry
            else:
                in_profit = cur_price < entry

            if in_profit:
                return

            # Not in profit after timeout — close
            age_min = int(age_sec // 60)
            inst = getattr(self, 'instance_name', 'BOT')
            logger.info("%s Stale timeout: %s %s (age=%dm, entry=%.4f, cur=%.4f) — closing",
                        inst, side, symbol, age_min, entry, cur_price)

            close_side = "Sell" if side == "Buy" else "Buy"
            qty = trade["qty"]
            try:
                if self.exchange_type == "tbank":
                    self.client.place_order(symbol=symbol, side=close_side, qty=qty)
                else:
                    self.client.place_order(symbol=symbol, side=close_side, qty=qty,
                                            category="linear", reduce_only=True)
            except Exception:
                logger.exception("Failed to close stale trade %s %s", side, symbol)
        except Exception:
            logger.exception("_check_stale_trade error for trade #%d", trade.get("id", 0))
