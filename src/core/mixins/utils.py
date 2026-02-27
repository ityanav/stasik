import logging
import time
from datetime import datetime, timedelta, timezone

from src.exchange.base import ExchangeClient
from src.strategy.signals import MomentumGenerator, SMCGenerator, SignalGenerator, TurtleGenerator

logger = logging.getLogger(__name__)


class UtilsMixin:
    """Utility / helper methods extracted from TradingEngine."""

    @staticmethod
    def _create_client(config: dict) -> ExchangeClient:
        exchange = config.get("exchange", "bybit")
        if exchange == "tbank":
            from src.exchange.tbank_client import TBankClient
            return TBankClient(config)
        else:
            from src.exchange.client import BybitClient
            return BybitClient(config)

    @staticmethod
    def _create_signal_gen(config: dict):
        mode = config.get("strategy", {}).get("strategy_mode", "trend")
        if mode == "momentum":
            logger.info("Strategy mode: Momentum (breakout, long-only)")
            return MomentumGenerator(config)
        if mode == "turtle":
            logger.info("Strategy mode: Turtle Trading (Donchian breakout)")
            return TurtleGenerator(config)
        if mode == "smc":
            logger.info("Strategy mode: FIBA (Fibonacci + Liquidity Sweep)")
            return SMCGenerator(config)
        return SignalGenerator(config)

    def _calc_net_pnl(self, side: str, entry: float, exit_price: float, qty: float) -> float:
        """Calculate net PnL after round-trip commission (entry + exit sides)."""
        if side == "Buy":
            gross = (exit_price - entry) * qty
        else:
            gross = (entry - exit_price) * qty
        fee = (entry * qty + exit_price * qty) * self.risk.commission_rate
        return gross - fee

    def _calc_fee(self, entry: float, exit_price: float, qty: float) -> float:
        """Calculate round-trip commission fee."""
        return (entry * qty + exit_price * qty) * self.risk.commission_rate

    @staticmethod
    def _timeframe_to_seconds(tf: str) -> int:
        """Convert Bybit timeframe string to seconds."""
        tf_map = {"D": 86400, "W": 604800, "M": 2592000}
        if tf in tf_map:
            return tf_map[tf]
        return int(tf) * 60

    async def _restore_cooldowns(self):
        """Restore cooldowns from trades closed today with negative PnL."""
        if self._cooldown_seconds <= 0:
            return
        try:
            _msk = timezone(timedelta(hours=3))
            today_start = datetime.now(_msk).replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%S.%f")
            cursor = await self.db._db.execute(
                "SELECT symbol, closed_at FROM trades "
                "WHERE status = 'closed' AND pnl < 0 AND closed_at >= ?",
                (today_start,),
            )
            rows = await cursor.fetchall()
            now = time.time()
            restored = 0
            for row in rows:
                symbol = row["symbol"]
                closed_at = datetime.fromisoformat(row["closed_at"]).replace(tzinfo=timezone(timedelta(hours=3)))
                cooldown_until = closed_at.timestamp() + self._cooldown_seconds
                if cooldown_until > now:
                    self._cooldowns[symbol] = cooldown_until
                    remaining = int(cooldown_until - now)
                    logger.info("Restored cooldown %s: %ds remaining", symbol, remaining)
                    restored += 1
            if restored:
                logger.info("Restored %d cooldowns", restored)
        except Exception:
            logger.warning("Failed to restore cooldowns", exc_info=True)

    async def _reconcile_positions(self):
        """Reconcile exchange positions with DB after restart."""
        try:
            # Get all positions from exchange
            if self.exchange_type == "tbank":
                exchange_positions = self.client.get_positions()
            else:
                exchange_positions = self.client.get_positions(category="linear")

            # Get all open trades from DB
            db_open = await self.db.get_open_trades()

            # Build lookup: symbol -> exchange position
            exch_by_symbol = {}
            for p in exchange_positions:
                if p["size"] > 0:
                    key = (p["symbol"], p["side"])
                    exch_by_symbol[key] = p

            closed_count = 0
            recovered_count = 0
            synced_count = 0

            # DB→Exchange: DB says open, exchange doesn't have it
            for trade in db_open:
                key = (trade["symbol"], trade["side"])
                if key in exch_by_symbol:
                    synced_count += 1
                else:
                    logger.info(
                        "Reconciliation: closing orphan DB trade #%d %s %s (not on exchange)",
                        trade["id"], trade["side"], trade["symbol"],
                    )
                    try:
                        await self._check_trade_closed(trade)
                        closed_count += 1
                    except Exception:
                        logger.exception("Failed to close orphan trade #%d", trade["id"])

            # Exchange→DB: exchange has position, DB doesn't
            # Only warn — don't auto-insert, because multiple instances
            # share the same exchange account and the position may belong
            # to another instance.
            untracked_count = 0
            db_symbols = {(t["symbol"], t["side"]) for t in db_open}
            for key, pos in exch_by_symbol.items():
                symbol, side = key
                if symbol not in self.pairs:
                    continue
                if key not in db_symbols:
                    untracked_count += 1
                    logger.warning(
                        "Reconciliation: untracked exchange position %s %s "
                        "(size=%.4f, entry=%.4f) — may belong to another instance",
                        side, symbol, pos["size"], pos["entry_price"],
                    )

            logger.info(
                "Reconciliation complete: %d synced, %d closed, %d untracked",
                synced_count, closed_count, untracked_count,
            )
        except Exception:
            logger.exception("Position reconciliation failed")

    async def _get_all_daily_pnl(self) -> dict[str, float]:
        """Get daily PnL for this instance and all other instances.
        Returns dict like {"SCALP": -1748.0, "SWING": 0.0}."""
        from datetime import date
        from pathlib import Path

        result = {}

        # Own instance
        name = self.instance_name or "BOT"
        result[name] = await self.db.get_daily_pnl()

        # Other instances
        for inst in self.config.get("other_instances", []):
            inst_name = inst.get("name", "???")
            db_path = inst.get("db_path", "")
            if db_path and Path(db_path).exists():
                try:
                    import aiosqlite
                    async with aiosqlite.connect(db_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute(
                            "SELECT pnl FROM daily_pnl WHERE trade_date = ?",
                            (date.today().isoformat(),),
                        )
                        row = await cur.fetchone()
                        result[inst_name] = float(row["pnl"]) if row else 0.0
                except Exception:
                    logger.warning("Failed to read daily PnL for %s", inst_name)
                    result[inst_name] = 0.0
            else:
                result[inst_name] = 0.0

        return result

    async def _get_daily_total_pnl(self) -> float:
        """Get total daily PnL across all instances (for short notifications)."""
        daily_map = await self._get_all_daily_pnl()
        return sum(daily_map.values())

    async def _record_pnl(self, pnl: float, balance: float):
        """Record PnL and check global drawdown across all bots."""
        self.risk.record_pnl(pnl, balance)
        if self.risk.daily_drawdown_from_profit > 0:
            try:
                daily_map = await self._get_all_daily_pnl()
                global_pnl = sum(daily_map.values())
                self.risk.check_global_drawdown(global_pnl)
            except Exception:
                logger.warning("Failed to check global drawdown")
