import asyncio
import logging
from datetime import datetime

from src.exchange.client import BybitClient
from src.risk.manager import RiskManager
from src.storage.database import Database
from src.strategy.signals import Signal, SignalGenerator

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, config: dict, notifier=None):
        self.config = config
        self.client = BybitClient(config)
        self.signal_gen = SignalGenerator(config)
        self.risk = RiskManager(config)
        self.db = Database()
        self.notifier = notifier  # async callable(text)

        self.pairs: list[str] = config["trading"]["pairs"]
        self.timeframe: str = str(config["trading"]["timeframe"])
        self.market_type: str = config["trading"]["market_type"]
        self.leverage: int = config["trading"].get("leverage", 1)

        self._running = False
        self._instrument_cache: dict[str, dict] = {}

    async def start(self):
        await self.db.connect()
        self._running = True

        # Set leverage for futures pairs
        if self.market_type in ("futures", "both"):
            for pair in self.pairs:
                self.client.set_leverage(pair, self.leverage, category="linear")

        await self._notify("Bot started. Pairs: " + ", ".join(self.pairs))
        logger.info("Trading engine started")

        try:
            await self._run_loop()
        except asyncio.CancelledError:
            logger.info("Trading engine cancelled")
        finally:
            await self.db.close()
            await self._notify("Bot stopped.")
            logger.info("Trading engine stopped")

    async def stop(self):
        self._running = False

    async def _run_loop(self):
        interval_sec = int(self.timeframe) * 60
        while self._running:
            try:
                await self._tick()
            except Exception:
                logger.exception("Error in trading tick")
            await asyncio.sleep(interval_sec)

    async def _tick(self):
        if self.risk.is_halted:
            logger.info("Trading halted — daily loss limit")
            return

        categories = self._get_categories()

        for pair in self.pairs:
            for category in categories:
                try:
                    await self._process_pair(pair, category)
                except Exception:
                    logger.exception("Error processing %s (%s)", pair, category)

    def _get_categories(self) -> list[str]:
        mt = self.market_type
        if mt == "spot":
            return ["spot"]
        elif mt == "futures":
            return ["linear"]
        else:
            return ["spot", "linear"]

    async def _process_pair(self, symbol: str, category: str):
        df = self.client.get_klines(
            symbol=symbol, interval=self.timeframe, limit=200, category=category
        )
        if len(df) < 50:
            logger.warning("Not enough data for %s (%s): %d candles", symbol, category, len(df))
            return

        result = self.signal_gen.generate(df)

        if result.signal == Signal.HOLD:
            return

        # Check existing positions
        open_positions = self.client.get_positions(category=category) if category == "linear" else []
        open_trades = await self.db.get_open_trades()
        symbol_open = [t for t in open_trades if t["symbol"] == symbol and t["category"] == category]

        if symbol_open:
            logger.debug("Already have open trade for %s (%s), skipping", symbol, category)
            return

        open_count = len(open_trades)
        if not self.risk.can_open_position(open_count):
            return

        side = "Buy" if result.signal == Signal.BUY else "Sell"

        # Spot: only Buy (no short selling)
        if category == "spot" and side == "Sell":
            return

        await self._open_trade(symbol, side, category, result.score, result.details)

    async def _open_trade(self, symbol: str, side: str, category: str, score: int, details: dict):
        price = self.client.get_last_price(symbol, category=category)
        balance = self.client.get_balance()

        info = self._get_instrument_info(symbol, category)
        qty = self.risk.calculate_position_size(
            balance=balance,
            price=price,
            qty_step=info["qty_step"],
            min_qty=info["min_qty"],
        )
        if qty <= 0:
            logger.info("Position size too small for %s, skipping", symbol)
            return

        sl, tp = self.risk.calculate_sl_tp(price, side)

        order = self.client.place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            category=category,
            stop_loss=sl if category == "linear" else None,
            take_profit=tp if category == "linear" else None,
        )

        order_id = order.get("orderId", "")
        trade_id = await self.db.insert_trade(
            symbol=symbol,
            side=side,
            category=category,
            qty=qty,
            entry_price=price,
            stop_loss=sl,
            take_profit=tp,
            order_id=order_id,
        )

        msg = (
            f"{'🟢' if side == 'Buy' else '🔴'} {side} {symbol} ({category})\n"
            f"Price: {price}\n"
            f"Qty: {qty}\n"
            f"SL: {sl} | TP: {tp}\n"
            f"Score: {score} | {details}"
        )
        logger.info(msg)
        await self._notify(msg)

    def _get_instrument_info(self, symbol: str, category: str) -> dict:
        key = f"{symbol}_{category}"
        if key not in self._instrument_cache:
            self._instrument_cache[key] = self.client.get_instrument_info(symbol, category)
        return self._instrument_cache[key]

    async def _notify(self, text: str):
        if self.notifier:
            try:
                await self.notifier(text)
            except Exception:
                logger.exception("Failed to send notification")

    # ── Status info ──────────────────────────────────────────

    async def get_status(self) -> str:
        balance = self.client.get_balance()
        open_trades = await self.db.get_open_trades()
        daily = self.risk.daily_pnl
        total = await self.db.get_total_pnl()

        lines = [
            f"Status: {'RUNNING' if self._running else 'STOPPED'}",
            f"{'HALTED — daily loss limit' if self.risk.is_halted else ''}",
            f"Balance: {balance:.2f} USDT",
            f"Open trades: {len(open_trades)}",
            f"Daily PnL: {daily:+.2f} USDT",
            f"Total PnL: {total:+.2f} USDT",
            f"Pairs: {', '.join(self.pairs)}",
            f"Timeframe: {self.timeframe}m",
        ]
        return "\n".join(line for line in lines if line)

    async def get_positions_text(self) -> str:
        categories = self._get_categories()
        lines = []
        for cat in categories:
            if cat == "linear":
                positions = self.client.get_positions(category=cat)
                for p in positions:
                    lines.append(
                        f"{p['side']} {p['symbol']} | size={p['size']} "
                        f"entry={p['entry_price']} uPnL={p['unrealised_pnl']:+.2f}"
                    )
        open_trades = await self.db.get_open_trades()
        if open_trades:
            lines.append("\n-- DB open trades --")
            for t in open_trades:
                lines.append(
                    f"{t['side']} {t['symbol']} ({t['category']}) "
                    f"qty={t['qty']} entry={t['entry_price']}"
                )
        return "\n".join(lines) if lines else "No open positions."

    async def get_pnl_text(self) -> str:
        daily = await self.db.get_daily_pnl()
        total = await self.db.get_total_pnl()
        recent = await self.db.get_recent_trades(5)
        lines = [
            f"Daily PnL: {daily:+.2f} USDT",
            f"Total PnL: {total:+.2f} USDT",
            "",
            "Recent trades:",
        ]
        for t in recent:
            pnl = t.get("pnl") or 0
            lines.append(
                f"  {t['side']} {t['symbol']} | pnl={pnl:+.2f} | {t['status']}"
            )
        return "\n".join(lines)
