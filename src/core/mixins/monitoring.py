import asyncio
import logging

logger = logging.getLogger(__name__)


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

                # Detect manually closed positions (every 15th iteration = ~30s)
                self._ws_reconcile_counter += 1
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
                # Detect manually closed positions (every 6th poll = ~30s)
                self._reconcile_poll_counter += 1
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
