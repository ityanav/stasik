import asyncio
import logging
import math
import time
from datetime import date, datetime

import httpx
import pandas as pd

from src.exchange.base import ExchangeClient
from src.risk.manager import RiskManager
from src.storage.database import Database
from src.strategy.ai_analyst import AIAnalyst, extract_indicator_values, format_risk_text, summarize_candles
from src.strategy.indicators import calculate_atr, calculate_sma_deviation, detect_swing_points, detect_swing_points_zigzag
from src.strategy.signals import GrinderGenerator, KotegawaGenerator, MomentumGenerator, SMCGenerator, Signal, SignalGenerator, SignalResult, Trend, TurtleGenerator

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, config: dict, notifier=None, db_path: str | None = None):
        self.config = config
        self.instance_name = config.get("instance_name", "")
        self.exchange_type = config.get("exchange", "bybit")
        self.client = self._create_client(config)
        self.signal_gen = self._create_signal_gen(config)
        self.risk = RiskManager(config)
        if db_path:
            from pathlib import Path
            self.db = Database(db_path=Path(db_path), instance_name=self.instance_name)
        else:
            self.db = Database(instance_name=self.instance_name)
        self.notifier = notifier  # async callable(text)
        self.ai_analyst = AIAnalyst.from_config(config)

        self.pairs: list[str] = config["trading"]["pairs"]
        self.timeframe: str = str(config["trading"]["timeframe"])
        self.market_type: str = config["trading"]["market_type"]
        self.leverage: int = config["trading"].get("leverage", 1)

        self._running = False
        self._instrument_cache: dict[str, dict] = {}
        self._tick_count: int = 0
        self.htf_timeframe: str = str(config["trading"].get("htf_timeframe", "15"))
        self._htf_cache: dict[str, tuple[Trend, float, float]] = {}  # symbol -> (trend, adx, timestamp)
        self._adx_period: int = config["strategy"].get("adx_period", 14)
        self._adx_min: float = config["strategy"].get("adx_min", 20)

        # Fear & Greed Index
        self._fng_extreme_greed: int = config["strategy"].get("fng_extreme_greed", 80)
        self._fng_extreme_fear: int = config["strategy"].get("fng_extreme_fear", 20)
        self._fng_cache: tuple[int, float] | None = None  # (value, timestamp)

        # Funding rate
        self._funding_rate_max: float = config["strategy"].get("funding_rate_max", 0.0003)
        self._funding_cache: dict[str, tuple[float, float]] = {}  # symbol -> (rate, timestamp)

        # Multi-TF cache for AI
        self._mtf_cache: dict[str, tuple[pd.DataFrame, float]] = {}  # "symbol_tf" -> (df, timestamp)
        self._extra_timeframes: list[str] = config.get("ai", {}).get("extra_timeframes", ["5", "15", "60"])

        # Correlation groups: symbol -> group name
        self._corr_groups: dict[str, str] = {}
        self._max_per_group: int = config["risk"].get("max_per_group", 1)
        for group_name, symbols in config["risk"].get("correlation_groups", {}).items():
            for s in symbols:
                self._corr_groups[s] = group_name

        # ATR config
        self._atr_period: int = config["risk"].get("atr_period", 14)
        self._atr_sl_mult: float = config["risk"].get("atr_sl_multiplier", 1.5)
        self._atr_tp_mult: float = config["risk"].get("atr_tp_multiplier", 3.0)
        self._atr_trail_mult: float = config["risk"].get("atr_trailing_multiplier", 1.0)

        # Cooldown after loss
        self._cooldowns: dict[str, float] = {}  # symbol -> cooldown_until timestamp
        self._cooldown_seconds: int = config["risk"].get("cooldown_after_loss", 5) * 60

        # Breakeven
        self._breakeven_done: set[int] = set()  # trade IDs that already moved to breakeven
        # Close-at-profit watchlist: symbols to close as soon as PnL > 0
        self._close_at_profit: set[str] = set()
        self._breakeven_activation: float = config["risk"].get("breakeven_activation", 0.5) / 100
        self._halt_closed: bool = False  # flag: already closed positions on halt

        # Kotegawa scaling in: track symbols where we already added to position
        self._scaled_in: set[str] = set()

        # Reversal confirmation: pending signals waiting for confirming candle
        self._pending_signals: dict[str, dict] = {}  # symbol -> pending signal data
        strat_cfg = config.get("strategy", {})
        self._reversal_confirmation: bool = strat_cfg.get("reversal_confirmation", False)
        self._reversal_candles: int = strat_cfg.get("reversal_candles", 1)
        self._reversal_max_wait: int = strat_cfg.get("reversal_max_wait", 3)

        # Trading hours filter
        trading_hours = config["trading"].get("trading_hours")
        if trading_hours and len(trading_hours) == 2:
            self._trading_hour_start: int = trading_hours[0]
            self._trading_hour_end: int = trading_hours[1]
        else:
            self._trading_hour_start = 0
            self._trading_hour_end = 24

        # Swing mode: candle dedup (only process signals on new candle)
        candle_sec = self._timeframe_to_seconds(str(config["trading"]["timeframe"]))
        self._is_swing = candle_sec >= 3600  # 1h+ = swing-like
        self._last_candle_ts: dict[str, float] = {}  # symbol -> last candle open timestamp

        # Market bias (auto-detected from BTC daily trend)
        self._market_bias: str = "neutral"  # "bullish", "bearish", "neutral"
        self._market_bias_ts: float = 0  # last update timestamp
        self._market_bias_interval: int = 3600  # recheck every hour
        self._market_bias_symbol: str = config["strategy"].get("bias_symbol", "BTCUSDT")
        self._market_bias_enabled: bool = config["strategy"].get("market_bias", True)
        self._bias_score_bonus: int = config["strategy"].get("bias_score_bonus", 1)
        self._bias_block_bearish: bool = config["strategy"].get("bias_block_bearish", False)

        # Liquidity filter
        strat = config.get("strategy", {})
        self._min_turnover_24h: float = strat.get("min_24h_turnover", 0)
        self._max_spread_pct: float = strat.get("max_spread_pct", 0)
        self._tickers_cache: dict = {}
        self._tickers_cache_ts: float = 0
        self._liquidity_log_ts: dict[str, float] = {}  # symbol -> last log timestamp

        # AI lessons from strategy reviews
        self._ai_lessons: list[str] = []

        # Grinder grid mode
        grinder_cfg = config.get("grinder", {})
        self._grinder_mode: bool = config.get("strategy", {}).get("strategy_mode") == "grinder" and bool(grinder_cfg)
        self._grinder_position_size_usdt: float = grinder_cfg.get("position_size_usdt", 5000)
        self._grinder_tp_usdt: float = grinder_cfg.get("tp_usdt", 10)
        self._grinder_sl_usdt: float = grinder_cfg.get("sl_usdt", 10)
        self._grinder_auto_reentry: bool = grinder_cfg.get("auto_reentry", True)
        self._grinder_tp_parts: int = grinder_cfg.get("tp_parts", 3)
        self._grinder_hedge_enabled: bool = grinder_cfg.get("hedge_enabled", True)
        self._grinder_hedge_after_usdt: float = grinder_cfg.get("hedge_after_usdt", 1.0)  # hedge when loss > N USDT
        self._grinder_hedged: set[str] = set()  # symbols already hedged
        self._grinder_last_direction: dict[str, str] = {}  # symbol -> last trade side
        self._grinder_ls_period: str = grinder_cfg.get("ls_period", "5min")

        # Turtle Trading mode
        turtle_cfg = config.get("turtle", {})
        self._turtle_mode: bool = config.get("strategy", {}).get("strategy_mode") == "turtle" and bool(turtle_cfg)
        self._turtle_state: dict[str, dict] = {}  # symbol -> {units, entries, side, system, last_add_price, entry_n}
        self._turtle_last_breakout: dict[str, bool] = {}  # symbol -> was_last_breakout_profitable (System 1 filter)
        self._turtle_risk_pct: float = turtle_cfg.get("risk_pct", 1.0)
        self._turtle_max_units: int = turtle_cfg.get("max_units", 4)
        self._turtle_pyramid_n_mult: float = turtle_cfg.get("pyramid_n_mult", 0.5)
        self._turtle_sl_n_mult: float = turtle_cfg.get("sl_n_mult", 2.0)

        # SMC mode
        smc_cfg = config.get("smc", {})
        self._smc_mode: bool = config.get("strategy", {}).get("strategy_mode") == "smc" and bool(smc_cfg)
        self._smc_htf_structure_cache: dict[str, tuple[dict, float]] = {}  # symbol -> (swings, timestamp)
        self._smc_structure_interval: int = 300  # refresh HTF structure every 5 min

        # SMC ZigZag swing detection
        self._smc_zigzag_enabled: bool = smc_cfg.get("zigzag_enabled", False)
        self._smc_zigzag_atr_period: int = smc_cfg.get("zigzag_atr_period", 14)
        self._smc_zigzag_atr_mult: float = smc_cfg.get("zigzag_atr_mult", 2.0)

        # SMC Fibonacci Pivot Points
        self._smc_fib_pivots: bool = smc_cfg.get("fib_pivots", False)
        self._smc_fib_pivot_proximity_pct: float = smc_cfg.get("fib_pivot_proximity_pct", 0.3)
        self._smc_daily_klines_cache: dict[str, tuple[pd.DataFrame, float]] = {}  # symbol -> (daily_df, ts)
        self._smc_daily_klines_ttl: int = 3600  # 1 hour

        # SMC Graduated TP
        self._smc_graduated_tp_enabled: bool = smc_cfg.get("graduated_tp", False)
        self._smc_tp_stages: list[float] = smc_cfg.get("tp_stages", [1.0, 1.272, 1.618])
        self._smc_tp_sizes: list[float] = smc_cfg.get("tp_sizes", [0.30, 0.30, 0.40])

        # SMC PnL momentum partial close
        self._smc_partial_enabled: bool = smc_cfg.get("partial_close_enabled", False)
        self._smc_partial_threshold: float = smc_cfg.get("partial_close_threshold", 100)  # min net PnL $ to start tracking
        self._smc_partial_pullback: float = smc_cfg.get("partial_close_pullback", 0.3)  # close when PnL drops 30% from peak
        self._smc_partial_pct: float = smc_cfg.get("partial_close_pct", 50)  # close 50% of position
        self._smc_pnl_tracker: dict[str, dict] = {}  # symbol -> {peak_pnl, prev_pnl, samples}
        self._smc_graduated_tracker: dict[str, dict] = {}  # symbol -> {levels: [price], sizes: [float], stage: int}

        # Max margin per instance
        self._max_margin_usdt: float = config["risk"].get("max_margin_usdt", 0)
        self._margin_cache: tuple[float, float] = (0.0, 0.0)  # (used_margin, timestamp)
        self._margin_cache_ttl: int = 30  # seconds

        # Weekly report
        self._weekly_report_day: int = config.get("weekly_report_day", 0)  # 0=Monday
        self._last_weekly_report: str = ""  # ISO date of last report

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
        if mode == "kotegawa":
            logger.info("Strategy mode: Kotegawa (mean reversion)")
            return KotegawaGenerator(config)
        if mode == "momentum":
            logger.info("Strategy mode: Momentum (breakout, long-only)")
            return MomentumGenerator(config)
        if mode == "grinder":
            logger.info("Strategy mode: Grinder (ping-pong)")
            return GrinderGenerator(config)
        if mode == "turtle":
            logger.info("Strategy mode: Turtle Trading (Donchian breakout)")
            return TurtleGenerator(config)
        if mode == "smc":
            logger.info("Strategy mode: SMC/ICT (Fibonacci + Liquidity Sweep)")
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

    def _update_market_bias(self):
        """Detect market regime from BTC daily chart (EMA20 vs EMA50 + price position)."""
        import time as _time
        now = _time.time()
        if now - self._market_bias_ts < self._market_bias_interval:
            return  # cached
        self._market_bias_ts = now

        if not self._market_bias_enabled or self.exchange_type != "bybit":
            self._market_bias = "neutral"
            return

        try:
            # Get daily candles for bias symbol (BTC)
            df = self.client.get_klines(self._market_bias_symbol, "D", limit=60, category="linear")
            if df is None or len(df) < 50:
                return

            close = df["close"]
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()

            price = close.iloc[-1]
            e20 = ema20.iloc[-1]
            e50 = ema50.iloc[-1]

            old_bias = self._market_bias

            # Strong bearish: price below both EMAs AND EMA20 < EMA50
            if price < e20 and price < e50 and e20 < e50:
                self._market_bias = "bearish"
            # Strong bullish: price above both EMAs AND EMA20 > EMA50
            elif price > e20 and price > e50 and e20 > e50:
                self._market_bias = "bullish"
            # Weak bearish: price below EMA50 OR EMA20 crossing below EMA50
            elif price < e50 or e20 < e50:
                self._market_bias = "bearish"
            # Weak bullish
            elif price > e50 or e20 > e50:
                self._market_bias = "bullish"
            else:
                self._market_bias = "neutral"

            if self._market_bias != old_bias:
                logger.info(
                    "Market bias changed: %s → %s (BTC price=%.0f, EMA20=%.0f, EMA50=%.0f)",
                    old_bias, self._market_bias, price, e20, e50,
                )
        except Exception:
            logger.warning("Failed to update market bias", exc_info=True)

    async def start(self):
        await self.db.connect()
        self._running = True

        # Set leverage for futures pairs (Bybit only)
        if self.market_type in ("futures", "both") and self.exchange_type == "bybit":
            for pair in self.pairs:
                self.client.set_leverage(pair, self.leverage, category="linear")

        # Restore daily PnL from DB so loss limit isn't bypassed after restart
        today_pnl = await self.db.get_daily_pnl()
        self.risk._daily_pnl = today_pnl
        if today_pnl != 0:
            logger.info("Restored daily PnL: %.2f", today_pnl)

        # Restore global peak PnL for drawdown-from-profit check
        if self.risk.daily_drawdown_from_profit > 0:
            try:
                daily_map = await self._get_all_daily_pnl()
                global_pnl = sum(daily_map.values())
                self.risk._daily_peak_pnl = max(global_pnl, 0)
                if global_pnl > 0:
                    logger.info("Restored global peak PnL: %.2f", global_pnl)
            except Exception:
                pass

        # Restore cooldowns from recent losing trades closed today
        await self._restore_cooldowns()

        # Reconcile exchange positions with DB
        await self._reconcile_positions()

        if not self._grinder_mode:
            await self._notify("🚀 Бот запущен\nПары: " + ", ".join(self.pairs))
        logger.info("Trading engine started")

        asyncio.create_task(self._run_sl_tp_monitor())
        await self._run_loop()

    async def _restore_cooldowns(self):
        """Restore cooldowns from trades closed today with negative PnL."""
        if self._cooldown_seconds <= 0:
            return
        try:
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0).isoformat()
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
                closed_at = datetime.fromisoformat(row["closed_at"])
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

    async def stop(self):
        """Pause trading loop. DB and AI stay open for resume()."""
        self._running = False
        logger.info("Trading engine paused")
        await self._notify("⏸ Торговля приостановлена")

    async def resume(self):
        """Resume trading loop after stop()."""
        if self._running:
            return
        self._running = True
        logger.info("Trading engine resumed")
        await self._notify("▶️ Торговля возобновлена\nПары: " + ", ".join(self.pairs))
        asyncio.create_task(self._run_loop())
        asyncio.create_task(self._run_sl_tp_monitor())

    async def shutdown(self):
        """Full shutdown — close all connections. Called on process exit."""
        self._running = False
        await self.ai_analyst.close()
        await self.db.close()
        await self._notify("🛑 Бот остановлен")
        logger.info("Trading engine stopped")

    @staticmethod
    def _timeframe_to_seconds(tf: str) -> int:
        """Convert Bybit timeframe string to seconds."""
        tf_map = {"D": 86400, "W": 604800, "M": 2592000}
        if tf in tf_map:
            return tf_map[tf]
        return int(tf) * 60

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
                    sl = trade.get("stop_loss") or 0
                    tp = trade.get("take_profit") or 0
                    if sl > 0:
                        try:
                            await self._check_db_stop_loss_with_price(trade, cur_price)
                        except Exception:
                            logger.exception("SL monitor error %s", sym)
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
                    sl = trade.get("stop_loss") or 0
                    tp = trade.get("take_profit") or 0
                    # Dynamic SL: отключён — предсказуемый SL (Котегава/ATR)
                    if sl > 0:
                        try:
                            await self._check_db_stop_loss(trade)
                        except Exception:
                            logger.exception("SL poll error %s", trade.get("symbol"))
                    if tp > 0:
                        try:
                            await self._check_db_take_profit(trade)
                        except Exception:
                            logger.exception("TP poll error %s", trade.get("symbol"))
                # Grinder hedge: open opposite when position goes negative
                if self._grinder_mode and self._grinder_hedge_enabled:
                    await self._grinder_check_hedge(open_trades)
                # SMC graduated TP (replaces simple partial close when enabled)
                if self._smc_mode and self._smc_graduated_tp_enabled:
                    await self._smc_check_graduated_tp(open_trades)
                elif self._smc_mode and self._smc_partial_enabled:
                    await self._smc_check_partial_close(open_trades)
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

            exch_open = set()
            for p in exchange_positions:
                if p["size"] > 0:
                    exch_open.add((p["symbol"], p["side"]))

            for trade in open_trades:
                key = (trade["symbol"], trade["side"])
                if key not in exch_open:
                    logger.info("Detected externally closed position: %s %s (trade #%d)",
                                trade["side"], trade["symbol"], trade["id"])
                    # Clean up graduated TP tracker
                    self._smc_graduated_tracker.pop(trade["symbol"], None)
                    self._smc_pnl_tracker.pop(trade["symbol"], None)
                    try:
                        await self._check_trade_closed(trade)
                    except Exception:
                        logger.exception("Failed to process externally closed trade #%d", trade["id"])
        except Exception:
            logger.exception("_detect_closed_positions error")

    async def _smc_check_partial_close(self, open_trades: list[dict]):
        """Close part of SMC position when PnL momentum fades.

        Logic:
        - Track unrealized net PnL for each open position
        - When net PnL > threshold ($100): start tracking peak PnL
        - When PnL drops by pullback% from peak: close partial_pct% of position
        - Only fires once per position (uses partial_closed flag)
        """
        commission_rate = self.risk.commission_rate

        for trade in open_trades:
            symbol = trade["symbol"]
            side = trade["side"]
            entry = trade["entry_price"]
            qty = trade["qty"]
            category = trade["category"]

            # Skip if already partially closed
            if (trade.get("partial_closed") or 0) > 0:
                self._smc_pnl_tracker.pop(symbol, None)
                continue

            # Get current price
            try:
                cur_price = self.client.get_last_price(symbol, category=category)
            except Exception:
                continue

            # Calculate unrealized net PnL
            if side == "Buy":
                gross_pnl = (cur_price - entry) * qty
            else:
                gross_pnl = (entry - cur_price) * qty
            round_trip_fee = entry * qty * commission_rate * 2
            net_pnl = gross_pnl - round_trip_fee

            tracker = self._smc_pnl_tracker.get(symbol)

            if net_pnl < self._smc_partial_threshold:
                # Below threshold — reset tracker
                if tracker:
                    self._smc_pnl_tracker.pop(symbol, None)
                continue

            # Above threshold — start/update tracking
            if not tracker:
                self._smc_pnl_tracker[symbol] = {
                    "peak_pnl": net_pnl,
                    "prev_pnl": net_pnl,
                    "samples": 1,
                }
                logger.info("SMC partial: %s tracking started, net PnL=$%.2f", symbol, net_pnl)
                continue

            tracker["samples"] += 1

            # Update peak
            if net_pnl > tracker["peak_pnl"]:
                tracker["peak_pnl"] = net_pnl
                tracker["prev_pnl"] = net_pnl
                return  # still growing, wait

            # Check pullback from peak
            peak = tracker["peak_pnl"]
            pullback_ratio = (peak - net_pnl) / peak if peak > 0 else 0

            if pullback_ratio < self._smc_partial_pullback:
                tracker["prev_pnl"] = net_pnl
                return  # pullback not deep enough yet

            # ── Pullback triggered: close partial ──
            logger.info(
                "SMC partial: %s peak=$%.2f, now=$%.2f, pullback=%.0f%% → closing %.0f%%",
                symbol, peak, net_pnl, pullback_ratio * 100, self._smc_partial_pct,
            )

            close_pct = self._smc_partial_pct / 100
            close_qty = qty * close_pct

            # Round to exchange step
            info = self._get_instrument_info(symbol, category)
            close_qty = math.floor(close_qty / info["qty_step"]) * info["qty_step"]
            close_qty = round(close_qty, 8)

            if close_qty < info["min_qty"]:
                logger.info("SMC partial: %s close_qty too small, skipping", symbol)
                self._smc_pnl_tracker.pop(symbol, None)
                continue

            remaining_qty = round(qty - close_qty, 8)
            if remaining_qty < info["min_qty"]:
                # Remainder too small — close everything
                close_qty = qty
                remaining_qty = 0

            close_side = "Sell" if side == "Buy" else "Buy"
            try:
                self.client.place_order(
                    symbol=symbol, side=close_side, qty=close_qty,
                    category=category, reduce_only=True,
                )
            except Exception:
                logger.exception("SMC partial: failed to close %s", symbol)
                self._smc_pnl_tracker.pop(symbol, None)
                continue

            # Calculate net PnL for closed part (after commission)
            partial_pnl = self._calc_net_pnl(side, entry, cur_price, close_qty)

            opened_at = trade.get("opened_at", datetime.utcnow().isoformat())

            if remaining_qty <= 0:
                # Closed everything
                try:
                    balance = self.client.get_balance()
                    await self.db.close_trade(trade["id"], cur_price, partial_pnl)
                    await self.db.update_daily_pnl(partial_pnl)
                    await self._record_pnl(partial_pnl, balance)
                except Exception:
                    logger.exception("SMC partial: DB update failed %s", symbol)
            else:
                # Partial close — record and update remaining qty
                try:
                    balance = self.client.get_balance()
                    await self.db.insert_partial_close(
                        symbol=symbol, side=side, category=category,
                        qty=close_qty, entry_price=entry, exit_price=cur_price,
                        pnl=partial_pnl, stage=1, opened_at=opened_at,
                    )
                    await self.db.mark_scale_out(trade["id"], 1, remaining_qty)
                    await self.db.update_daily_pnl(partial_pnl)
                    await self._record_pnl(partial_pnl, balance)
                except Exception:
                    logger.exception("SMC partial: DB update failed %s", symbol)

            # Notify
            direction = "LONG" if side == "Buy" else "SHORT"
            pct_str = f"{self._smc_partial_pct:.0f}%"
            msg = (
                f"📊 SMC Partial Close: {symbol}\n"
                f"{direction} | Закрыто {pct_str} ({close_qty})\n"
                f"Peak PnL: ${peak:.2f} → Now: ${net_pnl:.2f}\n"
                f"Partial PnL: ${partial_pnl:.2f}\n"
                f"Остаток: {remaining_qty}"
            )
            await self._notify(msg)

            self._smc_pnl_tracker.pop(symbol, None)

    async def _smc_check_graduated_tp(self, open_trades: list[dict]):
        """Graduated TP: close portions at Fib extension levels.

        3 stages by default:
          Stage 0 → close 30% at Fib 1.0, then move SL to breakeven
          Stage 1 → close 30% at Fib 1.272
          Stage 2 → close 40% at Fib 1.618 (or trail)
        """
        for trade in open_trades:
            symbol = trade["symbol"]
            tracker = self._smc_graduated_tracker.get(symbol)
            if not tracker:
                continue

            stage = tracker["stage"]
            levels = tracker["levels"]
            sizes = tracker["sizes"]
            side = tracker["side"]
            category = trade["category"]
            entry = trade["entry_price"]
            qty = trade["qty"]

            if stage >= len(levels) or stage >= len(sizes):
                self._smc_graduated_tracker.pop(symbol, None)
                continue

            # Get current price
            try:
                cur_price = self.client.get_last_price(symbol, category=category)
            except Exception:
                continue

            target = levels[stage]
            hit = False
            if side == "Buy" and cur_price >= target:
                hit = True
            elif side == "Sell" and cur_price <= target:
                hit = True

            if not hit:
                continue

            # Calculate close qty for this stage
            close_pct = sizes[stage]
            close_qty = qty * close_pct

            info = self._get_instrument_info(symbol, category)
            close_qty = math.floor(close_qty / info["qty_step"]) * info["qty_step"]
            close_qty = round(close_qty, 8)

            if close_qty < info["min_qty"]:
                # Too small — skip this stage
                tracker["stage"] += 1
                continue

            remaining_qty = round(qty - close_qty, 8)
            is_final = (stage == len(levels) - 1) or remaining_qty < info["min_qty"]
            if is_final:
                close_qty = qty
                remaining_qty = 0

            close_side = "Sell" if side == "Buy" else "Buy"
            try:
                self.client.place_order(
                    symbol=symbol, side=close_side, qty=close_qty,
                    category=category, reduce_only=True,
                )
            except Exception:
                logger.exception("SMC graduated TP: failed to close %s stage %d", symbol, stage)
                continue

            # Net PnL for closed portion (after commission)
            partial_pnl = self._calc_net_pnl(side, entry, cur_price, close_qty)

            opened_at = trade.get("opened_at", datetime.utcnow().isoformat())
            stage_num = stage + 1

            if is_final:
                try:
                    balance = self.client.get_balance()
                    await self.db.close_trade(trade["id"], cur_price, partial_pnl)
                    await self.db.update_daily_pnl(partial_pnl)
                    await self._record_pnl(partial_pnl, balance)
                except Exception:
                    logger.exception("SMC graduated TP: DB update failed %s", symbol)
                self._smc_graduated_tracker.pop(symbol, None)
            else:
                try:
                    balance = self.client.get_balance()
                    await self.db.insert_partial_close(
                        symbol=symbol, side=side, category=category,
                        qty=close_qty, entry_price=entry, exit_price=cur_price,
                        pnl=partial_pnl, stage=stage_num, opened_at=opened_at,
                    )
                    await self.db.mark_scale_out(trade["id"], stage_num, remaining_qty)
                    await self.db.update_daily_pnl(partial_pnl)
                    await self._record_pnl(partial_pnl, balance)
                except Exception:
                    logger.exception("SMC graduated TP: DB update failed %s", symbol)

                tracker["stage"] += 1

                # After stage 1: move SL to breakeven on EXCHANGE + DB
                if stage == 0:
                    try:
                        if self.exchange_type == "bybit":
                            self.client.session.set_trading_stop(
                                category=category, symbol=symbol,
                                stopLoss=str(round(entry, 6)), positionIdx=0,
                            )
                        await self.db.update_stop_loss(trade["id"], entry)
                        logger.info("SMC graduated TP: %s SL → breakeven (%.6f) ON EXCHANGE", symbol, entry)
                    except Exception:
                        logger.exception("SMC graduated TP: failed to move SL to breakeven %s", symbol)

            direction = "LONG" if side == "Buy" else "SHORT"
            ext_name = levels[stage] if stage < len(levels) else "final"
            msg = (
                f"📊 SMC Graduated TP Stage {stage_num}: {symbol}\n"
                f"{direction} | Closed {close_pct*100:.0f}% ({close_qty}) at {ext_name}\n"
                f"PnL: ${partial_pnl:.2f}\n"
                f"Remaining: {remaining_qty}"
            )
            await self._notify(msg)

    async def _run_loop(self):
        interval_sec = self._timeframe_to_seconds(self.timeframe)
        # For D/W/M: check every hour for position monitoring, not once per candle
        check_interval = min(interval_sec, self.config["trading"].get("check_interval", 3600))
        if interval_sec > 3600:
            logger.info("Swing mode: TF=%s, candle=%ds, check every %ds",
                        self.timeframe, interval_sec, check_interval)
            interval_sec = check_interval
        # Custom tick interval (faster than candle close)
        tick_override = self.config["trading"].get("tick_interval", 0)
        if tick_override > 0 and tick_override < interval_sec:
            logger.info("Fast tick: %ds (candle %ds)", tick_override, interval_sec)
            interval_sec = tick_override
        review_every = self.ai_analyst.review_interval  # in ticks (minutes)
        while self._running:
            try:
                await self._check_closed_trades()
                await self._tick()
                self._tick_count += 1
                if self.ai_analyst.enabled and review_every > 0 and self._tick_count % review_every == 0:
                    await self._ai_review_strategy()
                await self._maybe_weekly_report()
            except Exception:
                logger.exception("Error in trading tick")
            await asyncio.sleep(interval_sec)

    async def _tick(self):
        if self.risk.is_halted:
            if not self._halt_closed:
                self._halt_closed = True
                logger.warning("Daily loss limit — closing all positions")
                await self._close_all_on_halt()
            return
        else:
            self._halt_closed = False

        # Update market bias (cached, ~1 call/hour)
        self._update_market_bias()

        # Weekend filter for stock exchanges (MOEX closed Sat/Sun)
        now_utc = datetime.utcnow()
        if self.config.get("exchange") == "tbank" and now_utc.weekday() >= 5:
            logger.info("Выходной день (MOEX закрыта). Мониторинг позиций продолжается.")
            return

        # Trading hours filter (UTC)
        current_hour = now_utc.hour
        if self._trading_hour_start < self._trading_hour_end:
            in_session = self._trading_hour_start <= current_hour < self._trading_hour_end
        else:
            in_session = current_hour >= self._trading_hour_start or current_hour < self._trading_hour_end
        if not in_session:
            logger.info("Вне торговой сессии (UTC %d:00, окно %d-%d). Мониторинг позиций продолжается.",
                        current_hour, self._trading_hour_start, self._trading_hour_end)
            return

        categories = self._get_categories()

        for pair in self.pairs:
            for category in categories:
                try:
                    await self._process_pair(pair, category)
                except Exception:
                    logger.exception("Error processing %s (%s)", pair, category)

    def _get_categories(self) -> list[str]:
        if self.exchange_type == "tbank":
            return ["tbank"]
        mt = self.market_type
        if mt == "spot":
            return ["spot"]
        elif mt == "futures":
            return ["linear"]
        else:
            return ["spot", "linear"]

    def _get_tickers_cached(self) -> dict:
        """Return all tickers, cached for 5 minutes."""
        now = time.time()
        if now - self._tickers_cache_ts > 300:
            try:
                self._tickers_cache = self.client.get_all_tickers()
                self._tickers_cache_ts = now
            except Exception:
                logger.warning("Failed to fetch tickers for liquidity filter", exc_info=True)
        return self._tickers_cache

    def _check_liquidity(self, symbol: str) -> bool:
        """Pre-trade liquidity filter. Returns True if symbol passes."""
        if self._min_turnover_24h == 0 and self._max_spread_pct == 0:
            return True

        tickers = self._get_tickers_cached()
        ticker = tickers.get(symbol)
        if not ticker:
            return True  # no data — allow trade (conservative)

        turnover = ticker["turnover24h"]
        last_price = ticker["last_price"]
        bid = ticker["bid1"]
        ask = ticker["ask1"]

        passed = True
        reasons = []

        if self._min_turnover_24h > 0 and turnover < self._min_turnover_24h:
            passed = False
            reasons.append(f"turnover=${turnover/1e6:.1f}M<${self._min_turnover_24h/1e6:.0f}M")

        if self._max_spread_pct > 0 and last_price > 0 and bid > 0 and ask > 0:
            spread_pct = (ask - bid) / last_price * 100
            if spread_pct > self._max_spread_pct:
                passed = False
                reasons.append(f"spread={spread_pct:.3f}%>{self._max_spread_pct}%")

        if not passed:
            now = time.time()
            last_log = self._liquidity_log_ts.get(symbol, 0)
            if now - last_log > 1800:  # log once per 30 min per symbol
                self._liquidity_log_ts[symbol] = now
                logger.info("Liquidity filter: skip %s (%s)", symbol, ", ".join(reasons))

        return passed

    async def _process_pair(self, symbol: str, category: str):
        # ── Grinder contrarian-path: L/S ratio + funding ──
        if self._grinder_mode:
            try:
                ls_ratio = self.client.get_long_short_ratio(symbol, period=self._grinder_ls_period, category=category)
            except Exception:
                ls_ratio = {"buy_ratio": 0.5, "sell_ratio": 0.5}
            funding = self.client.get_funding_rate(symbol, category=category)
            oi_data = self.client.get_open_interest_history(symbol, category=category)
            obi = self.client.get_orderbook_imbalance(symbol, category=category)
            self.signal_gen.update_market_data(symbol, ls_ratio, funding, oi_data=oi_data, obi=obi)

            df = self.client.get_klines(
                symbol=symbol, interval=self.timeframe, limit=30, category=category
            )
            if len(df) < 2:
                return

            result = self.signal_gen.generate(df, symbol)
            if result.signal == Signal.HOLD:
                return

            side = "Buy" if result.signal == Signal.BUY else "Sell"
            open_trades = await self.db.get_open_trades()
            symbol_open = [t for t in open_trades if t["symbol"] == symbol]

            if symbol_open:
                # Hedge active (both directions) — don't interfere, let TP/SL resolve
                if len(symbol_open) >= 2:
                    return
                existing = symbol_open[0]
                if existing["side"] == side:
                    return  # same direction — hold
                # opposite signal → close & reopen
                await self._close_grinder_trade(existing, category)
                if self.risk.can_open_position(len(open_trades) - 1):
                    await self._open_grinder_trade(symbol, side, category, result.details)
                return

            if not self.risk.can_open_position(len(open_trades)):
                return
            if not self._check_margin_limit():
                return
            await self._open_grinder_trade(symbol, side, category, result.details)
            return

        # ── SMC/ICT path: Fibonacci + Liquidity Sweep ──
        if self._smc_mode:
            now = time.time()

            # 1. Fetch/cache HTF structure (1H klines → swing points)
            cached = self._smc_htf_structure_cache.get(symbol)
            if cached and (now - cached[1]) < self._smc_structure_interval:
                swings = cached[0]
            else:
                htf_df = self.client.get_klines(
                    symbol=symbol, interval=self.htf_timeframe, limit=200, category=category
                )
                if len(htf_df) < 30:
                    return
                smc_cfg = self.config.get("smc", {})
                if self._smc_zigzag_enabled:
                    swings = detect_swing_points_zigzag(
                        htf_df,
                        atr_period=self._smc_zigzag_atr_period,
                        atr_mult=self._smc_zigzag_atr_mult,
                    )
                else:
                    swings = detect_swing_points(
                        htf_df,
                        lookback=smc_cfg.get("swing_lookback", 5),
                        min_distance=smc_cfg.get("swing_min_distance", 10),
                    )
                self._smc_htf_structure_cache[symbol] = (swings, now)

                if not swings.get("last_swing_high") or not swings.get("last_swing_low"):
                    logger.debug("SMC %s: no swing structure found on HTF", symbol)
                    return

                # 2. Update signal generator with HTF structure
                self.signal_gen.update_structure(symbol, swings, htf_df)

            if not swings.get("last_swing_high") or not swings.get("last_swing_low"):
                return

            # 2b. Fetch/cache daily klines for Fibonacci Pivot Points
            if self._smc_fib_pivots:
                daily_cached = self._smc_daily_klines_cache.get(symbol)
                if not daily_cached or (now - daily_cached[1]) >= self._smc_daily_klines_ttl:
                    try:
                        daily_df = self.client.get_klines(
                            symbol=symbol, interval="D", limit=5, category=category
                        )
                        if len(daily_df) >= 2:
                            self._smc_daily_klines_cache[symbol] = (daily_df, now)
                            self.signal_gen.update_pivots(symbol, daily_df)
                    except Exception:
                        logger.debug("SMC %s: failed to fetch daily klines for pivots", symbol)

            # 3. Fetch entry TF klines (15m)
            df = self.client.get_klines(
                symbol=symbol, interval=self.timeframe, limit=200, category=category
            )
            if len(df) < 50:
                return

            # 4. Generate signal
            result = self.signal_gen.generate(df, symbol)
            if result.signal == Signal.HOLD:
                return

            side = "Buy" if result.signal == Signal.BUY else "Sell"

            # 5. Check existing positions
            open_trades = await self.db.get_open_trades()
            symbol_open = [t for t in open_trades if t["symbol"] == symbol]
            if symbol_open:
                return

            if not self.risk.can_open_position(len(open_trades)):
                return

            if not self._check_margin_limit():
                return

            # Correlation group limit
            group = self._corr_groups.get(symbol)
            if group:
                group_open = sum(
                    1 for t in open_trades
                    if self._corr_groups.get(t["symbol"]) == group
                )
                if group_open >= self._max_per_group:
                    logger.info("SMC корреляция: отклонён %s — %d/%d в группе '%s'",
                                symbol, group_open, self._max_per_group, group)
                    return

            # 6. Open trade with SMC SL/TP
            atr = calculate_atr(df, self._atr_period)
            price = df["close"].iloc[-1]

            # Skip if ATR too small — commission will eat profits
            # Need at least 0.3% ATR to cover round-trip fees
            if atr > 0 and price > 0:
                atr_pct = atr / price * 100
                if atr_pct < 0.3:
                    logger.info("SMC %s: skip — ATR=%.4f (%.3f%%) too small, commission would eat profits",
                                symbol, atr, atr_pct)
                    return

            await self._open_trade(
                symbol, side, category, result.score, result.details,
                atr=atr, df=df,
            )

            # Store graduated TP levels for this position (ATR-based, not raw Fib)
            if self._smc_graduated_tp_enabled and atr and atr > 0:
                price = self.client.get_last_price(symbol, category=category)
                # Use ATR multipliers for stages: 1.0 ATR, 1.5 ATR, 2.0 ATR
                stage_mults = self._smc_tp_stages  # [1.0, 1.272, 1.618] used as ATR multipliers
                if side == "Buy":
                    levels = [round(price + atr * m, 6) for m in stage_mults]
                else:
                    levels = [round(price - atr * m, 6) for m in stage_mults]
                self._smc_graduated_tracker[symbol] = {
                    "levels": levels,
                    "sizes": list(self._smc_tp_sizes),
                    "stage": 0,
                    "side": side,
                }
                stage_pcts = [abs(l - price) / price * 100 for l in levels]
                logger.info("SMC graduated TP: %s stages=%s (%.2f%%, %.2f%%, %.2f%%) sizes=%s",
                            symbol, levels, *stage_pcts, self._smc_tp_sizes)
            return

        # ── Turtle Trading path: Donchian breakout + pyramiding ──
        if self._turtle_mode:
            df = self.client.get_klines(
                symbol=symbol, interval=self.timeframe, limit=200, category=category
            )
            if len(df) < 60:
                return

            result = self.signal_gen.generate(df, symbol)
            n_value = result.details.get("n_value", 0)
            system = result.details.get("system")

            open_trades = await self.db.get_open_trades()
            symbol_open = [t for t in open_trades if t["symbol"] == symbol]
            turtle_state = self._turtle_state.get(symbol)

            if symbol_open and turtle_state:
                # Check exit channel + pyramid
                await self._check_turtle_exit(symbol, df, turtle_state, category)
                if self._turtle_state.get(symbol):  # still open after exit check
                    await self._check_turtle_pyramid(symbol, df, turtle_state, n_value, category)
            elif not symbol_open and result.signal != Signal.HOLD:
                side = "Buy" if result.signal == Signal.BUY else "Sell"
                # System 1 filter: skip if last breakout was profitable
                if system == 1 and self._turtle_last_breakout.get(symbol, False):
                    logger.info("Turtle S1 filter: skip %s %s (last breakout profitable)", side, symbol)
                    # But System 2 breakout overrides the filter
                    if result.details.get("s2_breakout"):
                        system = 2
                        logger.info("Turtle S2 override: %s %s (55-period breakout)", side, symbol)
                    else:
                        return
                if not self.risk.can_open_position(len(open_trades)):
                    return
                if not self._check_margin_limit():
                    return
                await self._open_turtle_trade(symbol, side, category, n_value, system, df)
            elif not symbol_open and turtle_state:
                # State cleanup: position closed externally
                del self._turtle_state[symbol]
            return

        # Cooldown check (before API calls to save quota)
        now = time.time()
        cooldown_until = self._cooldowns.get(symbol, 0)
        if now < cooldown_until:
            remaining = int(cooldown_until - now)
            logger.debug("Кулдаун %s: ещё %d сек", symbol, remaining)
            return

        # Liquidity filter (before candle fetch to save API calls)
        if not self._check_liquidity(symbol):
            return

        # Hard block: skip all new entries in bearish market (momentum long-only)
        if self._bias_block_bearish and self._market_bias == "bearish":
            return

        df = self.client.get_klines(
            symbol=symbol, interval=self.timeframe, limit=200, category=category
        )
        if len(df) < 50:
            logger.warning("Not enough data for %s (%s): %d candles", symbol, category, len(df))
            return

        # Swing mode: skip if no new candle since last check
        if self._is_swing:
            last_candle_time = df["timestamp"].iloc[-1]
            last_ts = last_candle_time.timestamp() if hasattr(last_candle_time, 'timestamp') else float(last_candle_time)
            prev_ts = self._last_candle_ts.get(symbol, 0)
            if last_ts <= prev_ts:
                logger.debug("Swing %s: нет новой свечи, пропуск", symbol)
                return
            self._last_candle_ts[symbol] = last_ts

        # Calculate ATR for dynamic SL/TP
        atr = calculate_atr(df, self._atr_period)

        # Fetch order book (if enabled)
        orderbook = None
        if self.config.get("strategy", {}).get("orderbook_enabled", False):
            try:
                orderbook = self.client.get_orderbook(symbol, limit=50, category=category)
            except Exception:
                logger.debug("Failed to fetch orderbook for %s", symbol)

        result = self.signal_gen.generate(df, symbol, orderbook=orderbook)

        # Apply market bias: adjust score based on BTC daily trend
        original_score = result.score
        if self._market_bias == "bearish" and self.exchange_type == "bybit":
            # Bearish market: boost sell signals, penalize buy signals
            if result.score < 0:  # sell signal
                result = SignalResult(signal=result.signal, score=result.score - self._bias_score_bonus, details=result.details)
            elif result.score > 0:  # buy signal
                result = SignalResult(signal=result.signal, score=result.score - self._bias_score_bonus, details=result.details)
                # Re-check if signal flipped
                if result.score <= -self.signal_gen.min_score:
                    result = SignalResult(signal=Signal.SELL, score=result.score, details=result.details)
                elif abs(result.score) < self.signal_gen.min_score:
                    result = SignalResult(signal=Signal.HOLD, score=result.score, details=result.details)
        elif self._market_bias == "bullish" and self.exchange_type == "bybit":
            # Bullish market: boost buy signals, penalize sell signals
            if result.score > 0:  # buy signal
                result = SignalResult(signal=result.signal, score=result.score + self._bias_score_bonus, details=result.details)
            elif result.score < 0:  # sell signal
                result = SignalResult(signal=result.signal, score=result.score + self._bias_score_bonus, details=result.details)
                if result.score >= self.signal_gen.min_score:
                    result = SignalResult(signal=Signal.BUY, score=result.score, details=result.details)
                elif abs(result.score) < self.signal_gen.min_score:
                    result = SignalResult(signal=Signal.HOLD, score=result.score, details=result.details)

        if original_score != result.score:
            logger.info("Market bias [%s]: %s score %d → %d", self._market_bias, symbol, original_score, result.score)

        if result.signal == Signal.HOLD:
            # Clear pending reversal if signal is gone
            if symbol in self._pending_signals:
                logger.info("⏳ Reversal reset %s: сигнал стал HOLD", symbol)
                del self._pending_signals[symbol]
            return

        # HTF trend + ADX + SMA deviation filter
        htf_trend, adx, htf_sma_dev = self._get_htf_data(symbol, category)

        # ADX filter: skip if market is ranging (no clear trend)
        if adx < self._adx_min:
            logger.info("ADX фильтр: отклонён %s %s (ADX=%.1f < %d — боковик)",
                        result.signal.value, symbol, adx, self._adx_min)
            return

        # ADX max filter: skip if trend is too strong (mean-reversion dangerous)
        adx_max = self.config.get("strategy", {}).get("adx_max", 0)
        if adx_max > 0 and adx > adx_max:
            logger.info("ADX max фильтр: отклонён %s %s (ADX=%.1f > %d — сильный тренд)",
                        result.signal.value, symbol, adx, adx_max)
            return

        # HTF SMA deviation filter: отклонение должно быть видно и на старшем TF
        htf_min_dev = self.config.get("strategy", {}).get("htf_min_sma_dev", 0)
        if htf_min_dev > 0:
            # BUY: HTF должен показывать отклонение вниз (dev < 0)
            if result.signal == Signal.BUY and htf_sma_dev > -htf_min_dev:
                logger.info("HTF SMA фильтр: отклонён BUY %s (HTF dev=%.2f%%, нужно ≤ -%.1f%%)",
                            symbol, htf_sma_dev, htf_min_dev)
                return
            # SELL: HTF должен показывать отклонение вверх (dev > 0)
            if result.signal == Signal.SELL and htf_sma_dev < htf_min_dev:
                logger.info("HTF SMA фильтр: отклонён SELL %s (HTF dev=%.2f%%, нужно ≥ +%.1f%%)",
                            symbol, htf_sma_dev, htf_min_dev)
                return

        # HTF trend filter: block counter-trend trades unless signal is very strong
        # Котегава: контр-тренд требует полного набора подтверждений
        htf_score_min = self.config.get("strategy", {}).get("htf_score_min", 3)
        if self.config.get("strategy", {}).get("htf_filter", True):
            if htf_trend == Trend.BEARISH and result.signal == Signal.BUY and abs(result.score) < htf_score_min:
                logger.info("HTF фильтр: отклонён BUY %s (тренд HTF медвежий, score=%d < %d)", symbol, result.score, htf_score_min)
                return
            if htf_trend == Trend.BULLISH and result.signal == Signal.SELL and abs(result.score) < htf_score_min:
                logger.info("HTF фильтр: отклонён SELL %s (тренд HTF бычий, score=%d < %d)", symbol, result.score, htf_score_min)
                return

        # Fear & Greed Index filter (crypto only)
        fng_value = await self._get_fear_greed() if self.exchange_type == "bybit" else None
        if fng_value is not None:
            if fng_value > self._fng_extreme_greed and result.signal == Signal.BUY:
                logger.info("FnG фильтр: отклонён BUY %s (FnG=%d > %d — Extreme Greed)",
                            symbol, fng_value, self._fng_extreme_greed)
                return
            if fng_value < self._fng_extreme_fear and result.signal == Signal.SELL:
                logger.info("FnG фильтр: отклонён SELL %s (FnG=%d < %d — Extreme Fear)",
                            symbol, fng_value, self._fng_extreme_fear)
                return

        # Funding rate filter (crypto only)
        funding_rate = self._get_funding_rate_cached(symbol, category) if self.exchange_type == "bybit" else 0.0
        if abs(funding_rate) > self._funding_rate_max:
            if funding_rate > 0 and result.signal == Signal.BUY:
                logger.info("Funding фильтр: отклонён BUY %s (funding=%.4f%% > 0 — перегрев лонгов)",
                            symbol, funding_rate * 100)
                return
            if funding_rate < 0 and result.signal == Signal.SELL:
                logger.info("Funding фильтр: отклонён SELL %s (funding=%.4f%% < 0 — перегрев шортов)",
                            symbol, funding_rate * 100)
                return

        # Check existing positions
        open_trades = await self.db.get_open_trades()
        symbol_open = [t for t in open_trades if t["symbol"] == symbol and t["category"] == category]
        side = "Buy" if result.signal == Signal.BUY else "Sell"

        # Kotegawa scaling in: allow adding to position on extreme deviation
        is_scale_in = False
        if symbol_open and isinstance(self.signal_gen, KotegawaGenerator):
            sma_dev_score = abs(result.details.get("sma_dev", 0))
            existing_side = symbol_open[0]["side"]
            # Scale in only if: extreme deviation, same direction, not yet scaled
            if sma_dev_score >= 2 and side == existing_side and symbol not in self._scaled_in:
                is_scale_in = True
                logger.info(
                    "Котегава scale-in: %s %s — extreme deviation, добавляем к позиции",
                    side, symbol,
                )
            else:
                logger.debug("Already have open trade for %s (%s), skipping", symbol, category)
                return
        elif symbol_open:
            logger.debug("Already have open trade for %s (%s), skipping", symbol, category)
            return

        open_count = len(open_trades)
        if not is_scale_in and not self.risk.can_open_position(open_count):
            return

        if not self._check_margin_limit():
            return

        # Correlation group limit
        group = self._corr_groups.get(symbol)
        if group:
            group_open = sum(
                1 for t in open_trades
                if self._corr_groups.get(t["symbol"]) == group
            )
            if group_open >= self._max_per_group:
                logger.info(
                    "Корреляция: отклонён %s — уже %d/%d из группы '%s'",
                    symbol, group_open, self._max_per_group, group,
                )
                return

        # Spot: only Buy (no short selling)
        if category == "spot" and side == "Sell":
            return

        # Reversal confirmation: wait for a confirming candle before entry
        if (
            self._reversal_confirmation
            and isinstance(self.signal_gen, KotegawaGenerator)
            and not is_scale_in
        ):
            pending = self._pending_signals.get(symbol)
            if pending:
                # Signal direction changed — reset pending
                if pending["side"] != side:
                    logger.info("⏳ Reversal reset %s: сигнал сменился %s → %s", symbol, pending["side"], side)
                    del self._pending_signals[symbol]
                    # Fall through to create new pending below

            pending = self._pending_signals.get(symbol)
            if pending is None:
                # New signal — store as pending, don't enter yet
                self._pending_signals[symbol] = {
                    "side": side,
                    "score": result.score,
                    "details": result.details,
                    "candles_waited": 0,
                }
                logger.info(
                    "⏳ Pending reversal %s %s (score=%d, dev=%.1f%%) — ждём подтверждающую свечу",
                    side, symbol, result.score, result.details.get("sma_dev", 0),
                )
                return
            else:
                # Check last closed candle (iloc[-2], current candle is still forming)
                prev_candle = df.iloc[-2]
                confirmed = False
                if side == "Buy" and prev_candle["close"] > prev_candle["open"]:
                    confirmed = True  # green candle confirms buy
                elif side == "Sell" and prev_candle["close"] < prev_candle["open"]:
                    confirmed = True  # red candle confirms sell

                if confirmed:
                    logger.info(
                        "✅ Reversal confirmed %s %s (waited %d candles)",
                        side, symbol, pending["candles_waited"],
                    )
                    del self._pending_signals[symbol]
                    # Continue to AI / _open_trade below
                else:
                    pending["candles_waited"] += 1
                    # Update score/details with fresh values
                    pending["score"] = result.score
                    pending["details"] = result.details
                    if pending["candles_waited"] >= self._reversal_max_wait:
                        logger.info(
                            "❌ Reversal timeout %s %s (waited %d candles, max=%d)",
                            side, symbol, pending["candles_waited"], self._reversal_max_wait,
                        )
                        del self._pending_signals[symbol]
                    else:
                        logger.info(
                            "⏳ Waiting reversal %s %s (candle %d/%d)",
                            side, symbol, pending["candles_waited"], self._reversal_max_wait,
                        )
                    return

        # AI analyst filter
        ai_reasoning = ""
        ai_sl = None
        ai_tp = None
        ai_size_mult = None
        if self.ai_analyst.enabled:
            indicator_text = extract_indicator_values(df, self.config)
            candles_text = summarize_candles(df)
            risk_text = format_risk_text(self.config)

            # Orderbook context for AI
            if orderbook and (orderbook.get("bids") or orderbook.get("asks")):
                from src.strategy.indicators import analyze_orderbook
                ob = analyze_orderbook(orderbook)
                ob_text = (
                    f"Bid vol: {ob['bid_vol']:,.0f} | Ask vol: {ob['ask_vol']:,.0f} | "
                    f"Imbalance: {ob['imbalance']:+.1%} ({'покупатели' if ob['imbalance'] > 0 else 'продавцы'})"
                )
                if ob["walls"]:
                    ob_text += f"\nСтенки: {ob['walls']}"
                indicator_text += f"\n\nСтакан (orderbook):\n{ob_text}"

            # Multi-TF context
            mtf_data = self._get_mtf_data(symbol, category)

            # Recent losses for AI context
            recent_losses = []
            try:
                recent = await self.db.get_recent_trades(10)
                recent_losses = [t for t in recent if t.get("status") == "closed" and (t.get("pnl") or 0) < 0]
            except Exception:
                pass

            verdict = await self.ai_analyst.analyze(
                signal=result.signal.value,
                score=result.score,
                details=result.details,
                indicator_text=indicator_text,
                candles_text=candles_text,
                risk_text=risk_text,
                mtf_data=mtf_data,
                config=self.config,
                recent_losses=recent_losses,
                lessons=self._ai_lessons,
                market_bias=self._market_bias,
            )

            if verdict.error:
                # Fallback: AI unavailable — trade on technical signals
                logger.warning("AI unavailable (%s), using technical signals for %s", verdict.error, symbol)
            elif not verdict.confirmed:
                direction = "BUY" if side == "Buy" else "SELL"
                msg = (
                    f"🤖 AI отклонил {direction} {symbol}\n"
                    f"Уверенность: {verdict.confidence}/10\n"
                    f"Причина: {verdict.reasoning}"
                )
                logger.info(msg)
                # Notify only if not notify_only (SCALP sends, DEGEN/SWING skip reject spam)
                if not self.config.get("telegram", {}).get("notify_only", False):
                    await self._notify(msg)
                return
            else:
                ai_reasoning = f"🤖 AI ({verdict.confidence}/10): {verdict.reasoning}"
                ai_sl = verdict.stop_loss
                ai_tp = verdict.take_profit
                ai_size_mult = verdict.position_size
                # Validate AI SL/TP: ensure minimum RR of 1.5:1
                if ai_sl is not None and ai_tp is not None and ai_sl > 0:
                    rr = ai_tp / ai_sl
                    if rr < 1.5:
                        logger.info("AI RR too low (%.2f:1, SL=%.1f%% TP=%.1f%%) — using ATR values",
                                    rr, ai_sl, ai_tp)
                        ai_sl = None
                        ai_tp = None
                logger.info("AI confirmed %s %s — confidence %d/10", side, symbol, verdict.confidence)

        await self._open_trade(
            symbol, side, category, result.score, result.details,
            ai_reasoning=ai_reasoning,
            ai_sl_pct=ai_sl,
            ai_tp_pct=ai_tp,
            ai_size_mult=ai_size_mult,
            atr=atr,
            is_scale_in=is_scale_in,
            df=df,
        )

    def _check_margin_limit(self) -> bool:
        """Check if instance margin limit is exceeded. Returns True if OK to trade."""
        if self._max_margin_usdt <= 0:
            return True  # disabled
        now = time.time()
        cached_margin, cached_ts = self._margin_cache
        if now - cached_ts < self._margin_cache_ttl:
            used = cached_margin
        else:
            try:
                used = self.client.get_used_margin(self.pairs)
                self._margin_cache = (used, now)
            except Exception as e:
                logger.warning("Margin check failed: %s — allowing trade", e)
                return True
        if used >= self._max_margin_usdt:
            logger.info(
                "Margin limit: %s used $%.0f / $%.0f — blocking new entry",
                self.instance_name, used, self._max_margin_usdt,
            )
            return False
        return True

    async def _open_trade(
        self, symbol: str, side: str, category: str, score: int, details: dict,
        ai_reasoning: str = "",
        ai_sl_pct: float | None = None,
        ai_tp_pct: float | None = None,
        ai_size_mult: float | None = None,
        atr: float | None = None,
        is_scale_in: bool = False,
        df=None,
    ):
        price = self.client.get_last_price(symbol, category=category)
        balance = self.client.get_balance()

        # ATR-based SL/TP (default), AI overrides take priority
        atr_sl_pct = None
        atr_tp_pct = None
        if atr and atr > 0:
            _, _, atr_sl_pct, atr_tp_pct = self.risk.calculate_sl_tp_atr(
                price, side, atr, self._atr_sl_mult, self._atr_tp_mult
            )

        info = self._get_instrument_info(symbol, category)
        # Use Kotegawa SL% or ATR SL% for adaptive position sizing
        if isinstance(self.signal_gen, KotegawaGenerator) and df is not None and len(df) >= 10:
            lookback = self.config.get("strategy", {}).get("kotegawa_sl_lookback", 10)
            sl_buffer = self.config.get("strategy", {}).get("kotegawa_sl_buffer", 0.2)
            recent = df.tail(lookback)
            if side == "Buy":
                swing_low = recent["low"].min()
                pre_sl = swing_low * (1 - sl_buffer / 100)
            else:
                swing_high = recent["high"].max()
                pre_sl = swing_high * (1 + sl_buffer / 100)
            sizing_sl = abs(price - pre_sl) / price
        else:
            sizing_sl = atr_sl_pct if atr_sl_pct else None
        qty = self.risk.calculate_position_size(
            balance=balance,
            price=price,
            qty_step=info["qty_step"],
            min_qty=info["min_qty"],
            sl_pct=sizing_sl,
            leverage=self.leverage,
        )

        # Kotegawa scaling: scale-in adds 50% of normal size
        if is_scale_in:
            qty = math.floor((qty * 0.5) / info["qty_step"]) * info["qty_step"]
            qty = round(qty, 8)

        # AI position size multiplier
        if ai_size_mult is not None and 0.1 <= ai_size_mult <= 2.0:
            qty = math.floor((qty * ai_size_mult) / info["qty_step"]) * info["qty_step"]
            qty = round(qty, 8)

        # Cap qty by instrument max
        max_qty = info.get("max_qty", 0)
        if max_qty > 0 and qty > max_qty:
            logger.info("Qty capped: %.2f -> %.2f (max_qty %s)", qty, max_qty, symbol)
            qty = math.floor(max_qty / info["qty_step"]) * info["qty_step"]
            qty = round(qty, 8)

        if qty <= 0:
            logger.info("Position size too small for %s, skipping", symbol)
            return

        # SL/TP priority: SMC (swept_level+ATR) > Kotegawa (recent low/high) > AI > ATR > fixed
        sl_source = "fixed"
        tp_source = "fixed"

        # SMC SL/TP: behind swept_level + 0.5 ATR buffer; TP from Fib extension
        smc_sl = None
        smc_tp = None
        if isinstance(self.signal_gen, SMCGenerator) and details:
            sweep_level = details.get("sweep_level", 0)
            tp1_level = details.get("tp1_level", 0)
            atr_buf = atr * 0.5 if atr and atr > 0 else 0
            if sweep_level and sweep_level > 0:
                if side == "Buy":
                    smc_sl = round(sweep_level - atr_buf, 6)
                else:
                    smc_sl = round(sweep_level + atr_buf, 6)
                # Enforce minimum SL distance (1 ATR)
                sl_dist = abs(price - smc_sl)
                min_sl_dist = atr if atr and atr > 0 else price * 0.005
                if sl_dist < min_sl_dist:
                    if side == "Buy":
                        smc_sl = round(price - min_sl_dist, 6)
                    else:
                        smc_sl = round(price + min_sl_dist, 6)
                    logger.info("SMC SL widened: %s %.6f → %.6f (min 1 ATR=%.6f)",
                                symbol, sweep_level - atr_buf if side == "Buy" else sweep_level + atr_buf,
                                smc_sl, min_sl_dist)
            if tp1_level and tp1_level > 0:
                smc_tp = round(tp1_level, 6)
                # Cap SMC TP by ATR multiplier so it doesn't overshoot
                if atr and atr > 0:
                    max_tp_dist = atr * self._atr_tp_mult
                    if side == "Buy":
                        atr_cap = round(price + max_tp_dist, 6)
                        if smc_tp > atr_cap:
                            logger.info("SMC TP capped: %s %.6f → %.6f (ATR cap %.2f%%)",
                                        symbol, smc_tp, atr_cap, max_tp_dist / price * 100)
                            smc_tp = atr_cap
                    else:
                        atr_cap = round(price - max_tp_dist, 6)
                        if smc_tp < atr_cap:
                            logger.info("SMC TP capped: %s %.6f → %.6f (ATR cap %.2f%%)",
                                        symbol, smc_tp, atr_cap, max_tp_dist / price * 100)
                            smc_tp = atr_cap

        # Kotegawa SL: behind recent low (Buy) / recent high (Sell)
        kotegawa_sl = None
        if isinstance(self.signal_gen, KotegawaGenerator) and df is not None and len(df) >= 10:
            lookback = self.config.get("strategy", {}).get("kotegawa_sl_lookback", 10)
            sl_buffer = self.config.get("strategy", {}).get("kotegawa_sl_buffer", 0.2)  # 0.2% запас
            recent = df.tail(lookback)
            if side == "Buy":
                swing_low = recent["low"].min()
                kotegawa_sl = swing_low * (1 - sl_buffer / 100)
            else:
                swing_high = recent["high"].max()
                kotegawa_sl = swing_high * (1 + sl_buffer / 100)
            kotegawa_sl = round(kotegawa_sl, 6)
            # Clamp: SL не дальше atr_sl_max от цены
            sl_max_pct = self.config.get("risk", {}).get("atr_sl_max", 10.0) / 100
            if side == "Buy":
                min_sl = price * (1 - sl_max_pct)
                if kotegawa_sl < min_sl:
                    logger.info("Котегава SL clamped: %.6f → %.6f (max %.1f%%)", kotegawa_sl, min_sl, sl_max_pct * 100)
                    kotegawa_sl = round(min_sl, 6)
            else:
                max_sl = price * (1 + sl_max_pct)
                if kotegawa_sl > max_sl:
                    logger.info("Котегава SL clamped: %.6f → %.6f (max %.1f%%)", kotegawa_sl, max_sl, sl_max_pct * 100)
                    kotegawa_sl = round(max_sl, 6)

        if smc_sl is not None:
            sl = smc_sl
            sl_dist_pct = abs(price - sl) / price * 100
            sl_source = f"SMC:sweep({sl_dist_pct:.2f}%)"
        elif kotegawa_sl is not None:
            sl = kotegawa_sl
            sl_dist_pct = abs(price - sl) / price * 100
            sl_source = f"Котегава:low/high({sl_dist_pct:.2f}%)"
        elif ai_sl_pct is not None and 0.3 <= ai_sl_pct <= 5.0:
            sl = price * (1 - ai_sl_pct / 100) if side == "Buy" else price * (1 + ai_sl_pct / 100)
            sl = round(sl, 6)
            sl_source = f"AI:{ai_sl_pct:.1f}%"
        elif atr_sl_pct:
            sl = self.risk.calculate_sl_tp_atr(price, side, atr, self._atr_sl_mult, self._atr_tp_mult)[0]
            sl_source = f"ATR:{atr_sl_pct*100:.2f}%"
        else:
            sl = self.risk.calculate_sl_tp(price, side)[0]

        # Clamp SL by max dollar loss (max_sl_usd)
        max_sl_usd = self.config.get("risk", {}).get("max_sl_usd", 0)
        if max_sl_usd > 0 and qty > 0:
            sl_loss = abs(sl - price) * qty
            if sl_loss > max_sl_usd:
                max_sl_dist = max_sl_usd / qty
                if side == "Buy":
                    sl = round(price - max_sl_dist, 6)
                else:
                    sl = round(price + max_sl_dist, 6)
                sl_source += f"→cap${max_sl_usd}"
                logger.info("SL capped: %s $%.0f → $%.0f (max_sl_usd=%d)", symbol, sl_loss, max_sl_usd, max_sl_usd)

        if smc_tp is not None and not self._smc_graduated_tp_enabled:
            tp = smc_tp
            tp_dist_pct = abs(tp - price) / price * 100
            tp_source = f"SMC:fib_ext({tp_dist_pct:.2f}%)"
        elif smc_tp is not None and self._smc_graduated_tp_enabled:
            # Graduated TP manages exits — no DB TP needed
            tp = 0
            tp_source = "graduated_tp(no_db_tp)"
        elif ai_tp_pct is not None and 0.5 <= ai_tp_pct <= 10.0:
            tp = price * (1 + ai_tp_pct / 100) if side == "Buy" else price * (1 - ai_tp_pct / 100)
            tp = round(tp, 6)
            tp_source = f"AI:{ai_tp_pct:.1f}%"
        elif atr_tp_pct:
            tp = self.risk.calculate_sl_tp_atr(price, side, atr, self._atr_sl_mult, self._atr_tp_mult)[1]
            tp_source = f"ATR:{atr_tp_pct*100:.2f}%"
        else:
            tp = self.risk.calculate_sl_tp(price, side)[1]

        # TP с учётом комиссии: чистая прибыль >= min_tp_net после round-trip fee
        min_tp_net = self.config.get("risk", {}).get("min_tp_net", 0)
        if min_tp_net > 0 and qty > 0:
            position_value = price * qty
            round_trip_fee = position_value * self.risk.commission_rate * 2
            min_gross_profit = min_tp_net + round_trip_fee
            current_tp_profit = abs(tp - price) * qty
            if current_tp_profit < min_gross_profit:
                min_tp_dist = min_gross_profit / qty
                if side == "Buy":
                    tp = round(price + min_tp_dist, 6)
                else:
                    tp = round(price - min_tp_dist, 6)
                tp_source += f"→min${min_tp_net}"
                logger.info("TP adjusted for commission: %s $%.0f → $%.0f (net=$%.0f + fee=$%.2f)",
                            symbol, current_tp_profit, min_gross_profit, min_tp_net, round_trip_fee)

        # R:R filter for SMC: reject if SL distance > TP distance (only when DB TP is set)
        if isinstance(self.signal_gen, SMCGenerator) and tp > 0 and sl > 0:
            sl_dist = abs(price - sl)
            tp_dist = abs(tp - price)
            if tp_dist > 0 and sl_dist / tp_dist > 1.0:
                logger.info("SMC R:R reject: %s SL=%.2f%% TP=%.2f%% R:R=%.2f (need >= 1.0)",
                            symbol, sl_dist / price * 100, tp_dist / price * 100, tp_dist / sl_dist)
                return

        # Place order with retry on qty rejection
        order = None
        for attempt in range(3):
            try:
                if self.exchange_type == "tbank":
                    order = self.client.place_order(
                        symbol=symbol, side=side, qty=qty,
                        stop_loss=sl, take_profit=tp,
                    )
                else:
                    order = self.client.place_order(
                        symbol=symbol, side=side, qty=qty, category=category,
                        stop_loss=sl if category == "linear" else None,
                        take_profit=tp if category == "linear" else None,
                    )
                break
            except Exception as e:
                err_msg = str(e).lower()
                if "max_qty" in err_msg or "exceeds maximum" in err_msg or "too large" in err_msg:
                    qty = math.floor(qty * 0.5 / info["qty_step"]) * info["qty_step"]
                    qty = round(qty, 8)
                    if qty <= 0:
                        logger.warning("Order rejected for %s: qty too large, cannot reduce further", symbol)
                        return
                    logger.info("Order qty rejected for %s, retrying with qty=%.2f (attempt %d)", symbol, qty, attempt + 2)
                else:
                    raise
        if order is None:
            logger.warning("Failed to place order for %s after retries", symbol)
            return

        order_id = order.get("orderId", "")
        await self.db.insert_trade(
            symbol=symbol,
            side=side,
            category=category,
            qty=qty,
            entry_price=price,
            stop_loss=sl,
            take_profit=tp,
            order_id="scale_in" if is_scale_in else order_id,
        )

        # Track Kotegawa scale-in
        if is_scale_in:
            self._scaled_in.add(symbol)

        # Set trailing stop for futures / tbank (ATR-based or fixed)
        trailing_msg = ""
        if category in ("linear", "tbank"):
            if atr and atr > 0:
                trailing_distance = self.risk.calculate_trailing_distance_atr(
                    atr, price, self._atr_trail_mult
                )
                trail_pct = trailing_distance / price * 100
                trail_activation_pct = self.config["risk"].get("trailing_activation", 0)
                if side == "Buy":
                    active_price = price * (1 + trail_activation_pct / 100)
                else:
                    active_price = price * (1 - trail_activation_pct / 100)
                self.client.set_trailing_stop(
                    symbol=symbol,
                    trailing_stop=trailing_distance,
                    active_price=active_price,
                )
                trailing_msg = f"\n📐 Trailing SL: {trail_pct:.2f}% ATR (активация при {trail_activation_pct}% прибыли)"
            else:
                trail_pct = self.config["risk"].get("trailing_stop", 0)
                trail_activation_pct = self.config["risk"].get("trailing_activation", 0)
                if trail_pct > 0:
                    trailing_distance = price * trail_pct / 100
                    if side == "Buy":
                        active_price = price * (1 + trail_activation_pct / 100)
                    else:
                        active_price = price * (1 - trail_activation_pct / 100)
                    self.client.set_trailing_stop(
                        symbol=symbol,
                        trailing_stop=trailing_distance,
                        active_price=active_price,
                    )
                    trailing_msg = f"\n📐 Trailing SL: {trail_pct}% (активация при {trail_activation_pct}% прибыли)"

        direction = "ЛОНГ 📈" if side == "Buy" else "ШОРТ 📉"
        pos_value = qty * price
        size_note = f" (AI: x{ai_size_mult})" if ai_size_mult else ""
        atr_note = f"\n📊 ATR: {atr:.4f}" if atr and atr > 0 else ""
        daily_total = await self._get_daily_total_pnl()
        day_arrow = "▲" if daily_total >= 0 else "▼"
        msg = (
            f"{'🟢' if side == 'Buy' else '🔴'} Открыл {direction} {symbol}\n"
            f"Цена: {price:.2f}\n"
            f"SL: {sl:.2f} ({sl_source}) | TP: {tp:.2f} ({tp_source}){trailing_msg}{atr_note}\n"
            f"Баланс: {balance:,.0f} USDT ({day_arrow} сегодня: {daily_total:+,.0f})\n"
            f"Объём: {qty}{size_note} (~{pos_value:,.0f} USDT)"
        )
        if ai_reasoning:
            msg += f"\n{ai_reasoning}"
        logger.info(msg)
        await self._notify(msg)

    # ── Grinder ping-pong trade ─────────────────────────────

    async def _close_grinder_trade(self, trade: dict, category: str):
        """Close a grinder position immediately (direction flip)."""
        symbol = trade["symbol"]
        side = trade["side"]
        qty = trade["qty"]
        entry = trade["entry_price"]
        close_side = "Sell" if side == "Buy" else "Buy"
        try:
            self.client.place_order(
                symbol=symbol, side=close_side, qty=qty,
                category=category, reduce_only=True,
            )
        except Exception:
            logger.exception("Grinder: failed to close %s", symbol)
            return
        # Get exit price and record
        try:
            exit_price = self.client.get_last_price(symbol, category=category)
            if side == "Buy":
                pnl = (exit_price - entry) * qty
            else:
                pnl = (entry - exit_price) * qty
            balance = self.client.get_balance()
            await self.db.close_trade(trade["id"], exit_price, pnl)
            await self.db.update_daily_pnl(pnl)
            await self._record_pnl(pnl, balance)
            direction = "LONG" if side == "Buy" else "SHORT"
            emoji = "+" if pnl >= 0 else ""
            logger.info(
                "Grinder flip: closed %s %s @ %.2f (PnL: %s%.2f)",
                direction, symbol, exit_price, emoji, pnl,
            )
        except Exception:
            logger.exception("Grinder: DB update failed for %s", symbol)

    async def _open_grinder_trade(self, symbol: str, side: str, category: str, details: dict | None = None, size_override: float | None = None):
        """Open a grinder trade with fixed USDT size, fixed USDT SL/TP."""
        price = self.client.get_last_price(symbol, category=category)
        if price <= 0:
            return

        info = self._get_instrument_info(symbol, category)
        position_usdt = size_override if size_override else self._grinder_position_size_usdt
        qty = position_usdt / price
        qty = math.floor(qty / info["qty_step"]) * info["qty_step"]
        qty = round(qty, 8)
        if qty <= 0:
            logger.info("Grinder: position size too small for %s", symbol)
            return

        # Fixed USDT SL/TP → convert to price offset
        tp_offset = self._grinder_tp_usdt / qty if qty > 0 else 0
        sl_offset = self._grinder_sl_usdt / qty if qty > 0 else 0

        if side == "Buy":
            tp = round(price + tp_offset, 6)
            sl = round(price - sl_offset, 6)
        else:
            tp = round(price - tp_offset, 6)
            sl = round(price + sl_offset, 6)

        # Place order — if partial TP enabled, don't set TP on exchange (managed by polling)
        exchange_tp = None if self._grinder_tp_parts > 1 else tp
        try:
            order = self.client.place_order(
                symbol=symbol, side=side, qty=qty, category=category,
                stop_loss=sl, take_profit=exchange_tp,
            )
        except Exception:
            logger.exception("Grinder: failed to place order for %s", symbol)
            return

        order_id = order.get("orderId", "")
        await self.db.insert_trade(
            symbol=symbol, side=side, category=category,
            qty=qty, entry_price=price, stop_loss=sl, take_profit=tp,
            order_id=order_id,
        )

        self._grinder_last_direction[symbol] = side

        balance = self.client.get_balance()
        direction = "LONG" if side == "Buy" else "SHORT"

        msg = (
            f"{'🟢' if side == 'Buy' else '🔴'} Grinder {direction} {symbol}\n"
            f"Price: {price:.2f} | Size: ~${position_usdt:,.0f}\n"
            f"TP: +${self._grinder_tp_usdt} | SL: -${self._grinder_sl_usdt}\n"
            f"Balance: {balance:,.0f} USDT"
        )
        logger.info(msg)

    async def _grinder_partial_tp(self, trade: dict, cur_price: float):
        """Grinder partial TP: close position in equal parts at each TP level.

        Stage 0 → close 1/N at TP1, move TP to TP2
        Stage 1 → close 1/N at TP2, move TP to TP3
        ...
        Last stage → close remainder, trigger re-entry.
        Each TP level is spaced equally: TP_offset = tp_usdt / qty per part.
        """
        symbol = trade["symbol"]
        side = trade["side"]
        entry = trade["entry_price"]
        qty = trade["qty"]
        category = trade["category"]
        n_parts = self._grinder_tp_parts
        current_stage = trade.get("partial_closed", 0) or 0
        is_last = current_stage >= n_parts - 1

        # Qty to close this stage
        if is_last:
            close_qty = qty
        else:
            remaining_parts = n_parts - current_stage
            close_qty = qty / remaining_parts

        # Round to exchange step
        info = self._get_instrument_info(symbol, "linear")
        close_qty = math.floor(close_qty / info["qty_step"]) * info["qty_step"]
        close_qty = round(close_qty, 8)

        if close_qty < info["min_qty"]:
            close_qty = qty
            is_last = True

        remaining_qty = round(qty - close_qty, 8)
        if remaining_qty < info["min_qty"]:
            close_qty = qty
            remaining_qty = 0
            is_last = True

        # Net PnL for this partial (after commission)
        partial_pnl = self._calc_net_pnl(side, entry, cur_price, close_qty)

        # Close partial on exchange
        close_side = "Sell" if side == "Buy" else "Buy"
        try:
            self.client.place_order(
                symbol=symbol, side=close_side, qty=close_qty,
                category="linear", reduce_only=True,
            )
        except Exception:
            logger.warning("Grinder partial TP: exchange close failed %s", symbol)

        new_stage = current_stage + 1
        opened_at = trade.get("opened_at", datetime.utcnow().isoformat())

        if is_last:
            # Final part — fully close trade
            try:
                balance = self.client.get_balance()
                await self.db.mark_scale_out(trade["id"], new_stage, 0)
                await self.db.close_trade(trade["id"], cur_price, partial_pnl)
                await self.db.update_daily_pnl(partial_pnl)
                await self._record_pnl(partial_pnl, balance)
            except Exception:
                logger.exception("Grinder partial TP final: DB error %s", symbol)
            logger.info(
                "Grinder TP %d/%d (final) %s: closed %.4f @ %.6f, PnL=%.2f",
                new_stage, n_parts, symbol, close_qty, cur_price, partial_pnl,
            )
        else:
            # Intermediate part — record partial, move TP further
            try:
                balance = self.client.get_balance()
                await self.db.insert_partial_close(
                    symbol=symbol, side=side, category=category,
                    qty=close_qty, entry_price=entry, exit_price=cur_price,
                    pnl=partial_pnl, stage=new_stage, opened_at=opened_at,
                )
                # Move TP to next level (same offset from current TP)
                old_tp = trade["take_profit"]
                tp_offset = self._grinder_tp_usdt / (self._grinder_position_size_usdt / entry) if entry > 0 else 0
                if side == "Buy":
                    new_tp = round(old_tp + tp_offset, 6)
                else:
                    new_tp = round(old_tp - tp_offset, 6)
                await self.db.mark_scale_out(trade["id"], new_stage, remaining_qty)
                await self.db.update_trade(trade["id"], take_profit=new_tp)
                await self.db.update_daily_pnl(partial_pnl)
                await self._record_pnl(partial_pnl, balance)
            except Exception:
                logger.exception("Grinder partial TP %d: DB error %s", new_stage, symbol)
            logger.info(
                "Grinder TP %d/%d %s: closed %.4f @ %.6f, PnL=%.2f, remaining=%.4f, new_TP=%.6f",
                new_stage, n_parts, symbol, close_qty, cur_price, partial_pnl,
                remaining_qty, new_tp if not is_last else 0,
            )

    async def _grinder_check_hedge(self, open_trades: list[dict]):
        """Open opposite position when existing grinder trade goes into loss.

        If LONG is losing > hedge_after_usdt → open SHORT (same size, same TP/SL).
        If SHORT is losing → open LONG.
        Only one hedge per symbol. Hedge cleared when original trade closes.
        """
        # Clean up hedged set: remove symbols with no open trades
        open_symbols = {t["symbol"] for t in open_trades}
        self._grinder_hedged -= (self._grinder_hedged - open_symbols)

        for trade in open_trades:
            symbol = trade["symbol"]
            if symbol in self._grinder_hedged:
                continue  # already hedged

            side = trade["side"]
            entry = trade["entry_price"]
            qty = trade["qty"]
            category = trade.get("category", "linear")

            try:
                cur_price = self.client.get_last_price(symbol, category=category)
            except Exception:
                continue

            if cur_price <= 0:
                continue

            # Calculate unrealized net PnL (after commission)
            upnl = self._calc_net_pnl(side, entry, cur_price, qty)

            # Only hedge if losing more than threshold (net)
            if upnl >= -self._grinder_hedge_after_usdt:
                continue

            # Check we have room for another position
            if not self.risk.can_open_position(len(open_trades)):
                continue

            # Open opposite direction at half size
            hedge_side = "Sell" if side == "Buy" else "Buy"
            self._grinder_hedged.add(symbol)
            hedge_size = self._grinder_position_size_usdt / 2
            logger.info(
                "Grinder HEDGE %s %s: original %s PnL=%.2f, opening opposite $%.0f",
                hedge_side, symbol, side, upnl, hedge_size,
            )
            await self._open_grinder_trade(symbol, hedge_side, category, {"hedge": True}, size_override=hedge_size)

    # ── Turtle Trading methods ─────────────────────────────

    async def _open_turtle_trade(self, symbol: str, side: str, category: str,
                                  n_value: float, system: int | None, df):
        """Open first Turtle unit with N-based sizing, SL=2N, no TP."""
        price = self.client.get_last_price(symbol, category=category)
        balance = self.client.get_balance()
        if price <= 0 or n_value <= 0:
            return

        # Unit size: (risk_pct% of equity) / N per unit
        dollar_risk = balance * (self._turtle_risk_pct / 100)
        qty_raw = dollar_risk / n_value
        info = self._get_instrument_info(symbol, category)
        qty = math.floor(qty_raw / info["qty_step"]) * info["qty_step"]
        qty = round(qty, 8)
        if qty <= 0:
            logger.info("Turtle: position size too small for %s (N=%.4f)", symbol, n_value)
            return

        # SL = 2N from entry
        sl_dist = self._turtle_sl_n_mult * n_value
        if side == "Buy":
            sl = round(price - sl_dist, 6)
        else:
            sl = round(price + sl_dist, 6)

        # No TP — exit only via Donchian exit channel
        try:
            order = self.client.place_order(
                symbol=symbol, side=side, qty=qty, category=category,
                stop_loss=sl, take_profit=None,
            )
        except Exception:
            logger.exception("Turtle: failed to place order for %s", symbol)
            return

        order_id = order.get("orderId", "")
        await self.db.insert_trade(
            symbol=symbol, side=side, category=category,
            qty=qty, entry_price=price, stop_loss=sl, take_profit=0,
            order_id=order_id,
        )

        # Track turtle state
        self._turtle_state[symbol] = {
            "units": 1,
            "entries": [{"price": price, "qty": qty}],
            "side": side,
            "system": system or 1,
            "last_add_price": price,
            "entry_n": n_value,
        }

        direction = "LONG" if side == "Buy" else "SHORT"
        pos_value = qty * price
        daily_total = await self._get_daily_total_pnl()
        day_arrow = "▲" if daily_total >= 0 else "▼"
        msg = (
            f"🐢 Turtle {direction} {symbol} (S{system or '?'})\n"
            f"Price: {price:.2f} | N: {n_value:.4f}\n"
            f"SL: {sl:.2f} ({self._turtle_sl_n_mult}N) | TP: exit channel\n"
            f"Unit 1/{self._turtle_max_units} | Size: {qty} (~${pos_value:,.0f})\n"
            f"Balance: {balance:,.0f} USDT ({day_arrow} {daily_total:+,.0f})"
        )
        logger.info(msg)
        await self._notify(msg)

    async def _check_turtle_pyramid(self, symbol: str, df, state: dict,
                                     n_value: float, category: str):
        """Add unit every 0.5N price movement in profit direction (max 4 units)."""
        if state["units"] >= self._turtle_max_units:
            return
        if n_value <= 0:
            return

        price = df["close"].iloc[-1]
        last_add = state["last_add_price"]
        side = state["side"]
        threshold = self._turtle_pyramid_n_mult * n_value

        # Check if price moved enough since last unit addition
        if side == "Buy" and price < last_add + threshold:
            return
        if side == "Sell" and price > last_add - threshold:
            return

        if not self._check_margin_limit():
            return

        balance = self.client.get_balance()
        dollar_risk = balance * (self._turtle_risk_pct / 100)
        qty_raw = dollar_risk / n_value
        info = self._get_instrument_info(symbol, category)
        qty = math.floor(qty_raw / info["qty_step"]) * info["qty_step"]
        qty = round(qty, 8)
        if qty <= 0:
            return

        # New SL for ALL units: 2N from this new entry
        sl_dist = self._turtle_sl_n_mult * n_value
        if side == "Buy":
            new_sl = round(price - sl_dist, 6)
        else:
            new_sl = round(price + sl_dist, 6)

        try:
            self.client.place_order(
                symbol=symbol, side=side, qty=qty, category=category,
                stop_loss=new_sl, take_profit=None,
            )
        except Exception:
            logger.exception("Turtle pyramid: failed for %s", symbol)
            return

        await self.db.insert_trade(
            symbol=symbol, side=side, category=category,
            qty=qty, entry_price=price, stop_loss=new_sl, take_profit=0,
            order_id=f"pyramid_{state['units'] + 1}",
        )

        # Update SL on all existing trades for this symbol
        await self._turtle_update_trailing_sl(symbol, new_sl)

        state["units"] += 1
        state["entries"].append({"price": price, "qty": qty})
        state["last_add_price"] = price
        state["entry_n"] = n_value

        unit_num = state["units"]
        logger.info(
            "🐢 Turtle PYRAMID %s %s: unit %d/%d @ %.2f (SL moved to %.2f)",
            side, symbol, unit_num, self._turtle_max_units, price, new_sl,
        )
        await self._notify(
            f"🐢 Pyramid {side} {symbol}: unit {unit_num}/{self._turtle_max_units} @ {price:.2f}\n"
            f"SL all units → {new_sl:.2f}"
        )

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
            if side == "Buy":
                pnl = (exit_price - entry) * qty
            else:
                pnl = (entry - exit_price) * qty
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

    # ── AI strategy review ──────────────────────────────────

    async def _ai_review_strategy(self):
        recent = await self.db.get_recent_trades(20)
        closed = [t for t in recent if t.get("status") == "closed"]

        if len(closed) < 3:
            logger.info("AI review: too few closed trades (%d), skipping", len(closed))
            return

        update = await self.ai_analyst.review_strategy(
            strategy_config=self.config["strategy"],
            risk_config=self.config["risk"],
            recent_trades=closed,
            market_bias=self._market_bias,
            lessons=self._ai_lessons,
        )

        if update.error:
            logger.warning("AI review failed: %s", update.error)
            return

        # Save lessons from AI (replace old with new, max 5)
        if update.lessons:
            self._ai_lessons = update.lessons[:5]
            logger.info("AI lessons updated: %s", self._ai_lessons)

        if not update.changes:
            logger.info("AI review: no changes needed. Lessons: %d", len(self._ai_lessons))
            return

        # Apply changes
        strategy_keys = {"rsi_oversold", "rsi_overbought", "ema_fast", "ema_slow",
                         "bb_period", "bb_std", "vol_threshold", "min_score"}
        risk_keys = {"stop_loss", "take_profit", "risk_per_trade"}

        changes_text = []
        for key, value in update.changes.items():
            if key in strategy_keys:
                old = self.config["strategy"].get(key)
                self.config["strategy"][key] = value
                changes_text.append(f"  {key}: {old} → {value}")
            elif key in risk_keys:
                old = self.config["risk"].get(key)
                self.config["risk"][key] = value
                changes_text.append(f"  {key}: {old} → {value}")

        if not changes_text:
            return

        # Rebuild signal generator with new parameters
        self.signal_gen = self._create_signal_gen(self.config)
        # Update risk manager SL/TP if changed
        if "stop_loss" in update.changes:
            self.risk.stop_loss_pct = update.changes["stop_loss"] / 100
        if "take_profit" in update.changes:
            self.risk.take_profit_pct = update.changes["take_profit"] / 100
        if "risk_per_trade" in update.changes:
            self.risk.risk_per_trade = update.changes["risk_per_trade"] / 100

        msg = (
            f"🧠 AI пересмотрел стратегию\n"
            f"Изменения:\n" + "\n".join(changes_text) + "\n"
            f"Причина: {update.reasoning}"
        )
        if self._ai_lessons:
            msg += "\n\nУроки:\n" + "\n".join(f"  • {l}" for l in self._ai_lessons)
        logger.info(msg)
        await self._notify(msg)

    # ── Monitor closed trades ─────────────────────────────────

    async def _check_closed_trades(self):
        open_trades = await self.db.get_open_trades()
        if not open_trades:
            return

        for trade in open_trades:
            if trade["category"] not in ("linear", "tbank"):
                continue
            try:
                await self._check_db_stop_loss(trade)
                await self._check_db_take_profit(trade)
                await self._check_breakeven(trade)
                await self._check_smart_exit(trade)
                await self._check_kotegawa_scale_out(trade)
                await self._check_profit_target(trade)
                await self._check_loss_target(trade)
                await self._check_close_at_profit(trade)
                await self._check_trade_closed(trade)
            except Exception:
                logger.exception("Error checking trade %s", trade["id"])

    async def _check_trade_closed(self, trade: dict):
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
                    pnl = rec["pnl"]
                    exit_price = rec["exit_price"]
                    logger.info("Got real PnL from exchange for %s: %.2f (exit=%.4f)", symbol, pnl, exit_price)
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
        self._scaled_in.discard(symbol)

        # Send notification
        if pnl >= 0:
            emoji = "💰"
            result_text = f"заработал: +{pnl:,.2f} USDT"
        else:
            emoji = "💸"
            result_text = f"просрал: {pnl:,.2f} USDT"

        direction = "ЛОНГ" if side == "Buy" else "ШОРТ"
        daily_pnl = await self.db.get_daily_pnl()
        total_pnl = await self.db.get_total_pnl()

        cooldown_note = ""
        if pnl < 0 and self._cooldown_seconds > 0:
            cooldown_note = f"\n⏳ Кулдаун {symbol}: {self._cooldown_seconds // 60} мин"

        daily_total = await self._get_daily_total_pnl()
        day_arrow = "▲" if daily_total >= 0 else "▼"

        msg = (
            f"{emoji} Закрыл {direction} {symbol}\n"
            f"Вход: {entry_price} → Выход: {exit_price}\n"
            f"Результат: {result_text}{cooldown_note}\n"
            f"─────────────\n"
            f"За день: {daily_pnl:+,.2f} USDT\n"
            f"Всего: {total_pnl:+,.2f} USDT\n"
            f"Баланс: {balance:,.0f} USDT ({day_arrow} сегодня: {daily_total:+,.0f})"
        )
        logger.info(msg)
        if not self._grinder_mode:
            await self._notify(msg)

        # ── Grinder auto re-entry ─────────────────────────────
        if self._grinder_mode and self._grinder_auto_reentry:
            # Skip cooldown for grinder
            self._cooldowns.pop(symbol, None)
            # Determine if TP or SL hit
            tp_price = trade.get("take_profit", 0)
            sl_price = trade.get("stop_loss", 0)
            hit_tp = False
            if tp_price and sl_price and exit_price:
                tp_dist = abs(exit_price - tp_price)
                sl_dist = abs(exit_price - sl_price)
                hit_tp = tp_dist < sl_dist
            if hit_tp:
                # TP hit → re-enter same direction
                new_side = side
                logger.info("Grinder re-entry: TP hit, same direction %s %s", new_side, symbol)
            else:
                # SL hit → reverse
                new_side = "Sell" if side == "Buy" else "Buy"
                logger.info("Grinder re-entry: SL hit, reversing to %s %s", new_side, symbol)
            # Get current levels for SL/TP
            details = {}
            if hasattr(self.signal_gen, '_support'):
                details["support"] = self.signal_gen._support.get(symbol, 0)
                details["resistance"] = self.signal_gen._resistance.get(symbol, 0)
            await self._open_grinder_trade(symbol, new_side, category, details)

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
        direction = "ЛОНГ" if side == "Buy" else "ШОРТ"
        logger.info("✅ Close-at-profit: закрыл %s %s при net PnL +%.0f %s", direction, symbol, net_pnl, currency)
        await self._notify(
            f"✅ Закрыл {direction} {symbol} в плюсе\n"
            f"Net PnL: +{net_pnl:,.0f} {currency}"
        )

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

    async def _check_db_stop_loss(self, trade: dict):
        """Close position when price hits stop_loss stored in DB (fetches price via REST)."""
        sl = trade.get("stop_loss")
        if not sl or sl <= 0:
            return
        try:
            if self.exchange_type == "tbank":
                cur_price = self.client.get_last_price(trade["symbol"])
            else:
                cur_price = self.client.get_last_price(trade["symbol"], category="linear")
        except Exception:
            return
        await self._check_db_stop_loss_with_price(trade, cur_price)

    async def _dynamic_sl_tighten(self, trade: dict):
        """Tighten SL to max 60% of TP (in $). Triggers at 80% of limit."""
        sl = trade.get("stop_loss") or 0
        tp = trade.get("take_profit") or 0
        entry = trade.get("entry_price") or 0
        qty = trade.get("qty") or 0
        side = trade.get("side", "")
        symbol = trade.get("symbol", "")
        if not all([sl, tp, entry, qty, side]):
            return

        tp_usd = abs(tp - entry) * qty
        max_sl_usd = tp_usd * 0.6
        sl_usd = abs(sl - entry) * qty
        if tp_usd <= 0 or sl_usd <= max_sl_usd:
            return  # SL already ≤ 60% of TP

        # Get current price
        try:
            if self.exchange_type == "tbank":
                cur_price = self.client.get_last_price(symbol)
            else:
                cur_price = self.client.get_last_price(symbol, category="linear")
        except Exception:
            return
        if not cur_price:
            return

        # Calculate unrealized net loss (after commission)
        unrealized_pnl = self._calc_net_pnl(side, entry, cur_price, qty)

        # When net loss reaches 80% of limit → tighten SL to 60% of TP
        if unrealized_pnl < 0 and abs(unrealized_pnl) >= max_sl_usd * 0.8:
            new_sl_dist = abs(tp - entry) * 0.6
            if side == "Buy":
                new_sl = round(entry - new_sl_dist, 6)
            else:
                new_sl = round(entry + new_sl_dist, 6)
            logger.info("Dynamic SL tighten: %s loss $%.0f → SL $%.0f→$%.0f (60%% of TP $%.0f)",
                        symbol, abs(unrealized_pnl), sl_usd, max_sl_usd, tp_usd)
            await self.db.update_trade(trade["id"], stop_loss=new_sl)

    async def _check_db_stop_loss_with_price(self, trade: dict, cur_price: float):
        """Close position when price hits stop_loss stored in DB."""
        sl = trade.get("stop_loss")
        if not sl or sl <= 0 or not cur_price:
            return

        symbol = trade["symbol"]
        side = trade["side"]
        entry = trade["entry_price"]
        qty = trade["qty"]

        # Buy: close when price <= SL (price fell)
        if side == "Buy" and cur_price > sl:
            return
        # Sell: close when price >= SL (price rose)
        if side == "Sell" and cur_price < sl:
            return

        # SL hit — close position (net PnL after commission)
        net_pnl = self._calc_net_pnl(side, entry, cur_price, qty)

        # Try to close on exchange (may fail if position is phantom)
        close_side = "Sell" if side == "Buy" else "Buy"
        try:
            if self.exchange_type == "tbank":
                self.client.place_order(symbol=symbol, side=close_side, qty=qty)
            else:
                self.client.place_order(symbol=symbol, side=close_side, qty=qty, category="linear", reduce_only=True)
        except Exception:
            logger.warning("DB stop-loss: no position on exchange for %s, closing in DB only", symbol)

        # Always close in DB
        try:
            balance = self.client.get_balance()
            await self.db.close_trade(trade["id"], cur_price, net_pnl)
            await self.db.update_daily_pnl(net_pnl)
            await self._record_pnl(net_pnl, balance)
        except Exception:
            logger.exception("DB stop-loss: DB update failed for %s", symbol)

        # Clean up graduated TP tracker
        self._smc_graduated_tracker.pop(symbol, None)
        self._smc_pnl_tracker.pop(symbol, None)

        direction = "ЛОНГ" if side == "Buy" else "ШОРТ"
        fee = self._calc_fee(entry, cur_price, qty)
        logger.info("🛑 SL сработал: %s %s @ %.6f (SL=%.6f, net PnL=%.2f, fee=%.2f)", direction, symbol, cur_price, sl, net_pnl, fee)
        await self._notify(
            f"🛑 Стоп-лосс: {direction} {symbol}\n"
            f"Цена: {cur_price:,.6f} (SL: {sl:,.6f})\n"
            f"Net PnL: {net_pnl:+,.2f} (fee: {fee:.2f})"
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

        # ── Grinder partial TP ──────────────────────────────
        if self._grinder_mode and self._grinder_tp_parts > 1:
            await self._grinder_partial_tp(trade, cur_price)
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

    async def _check_kotegawa_scale_out(self, trade: dict):
        """Kotegawa scale-out: close position in stages as price reverts toward SMA.

        3-stage progressive close (default 35/35/30%) at increasing distances toward SMA.
        After each stage, SL is moved to lock in profit.

        Config (strategy section):
          scale_out_enabled: true
          scale_out_stages: [0.30, 0.60, 0.90]   # % distance entry→SMA for each stage
          scale_out_sizes: [0.35, 0.35, 0.30]     # % of original position per stage
          kotegawa_exit_ratio: 0.8                 # fallback single-exit ratio
        """
        if not isinstance(self.signal_gen, KotegawaGenerator):
            return

        strat = self.config.get("strategy", {})

        if not strat.get("scale_out_enabled", False):
            # Fallback: old single-exit behavior
            return await self._check_kotegawa_exit_single(trade)

        symbol = trade["symbol"]
        side = trade["side"]
        entry = trade["entry_price"]
        qty = trade["qty"]
        category = trade["category"]

        stages = strat.get("scale_out_stages", [0.30, 0.60, 0.90])
        sizes = strat.get("scale_out_sizes", [0.35, 0.35, 0.30])
        current_stage = trade.get("partial_closed", 0) or 0

        if current_stage >= len(stages):
            return  # all stages done

        try:
            df = self.client.get_klines(symbol, self.timeframe, limit=100, category=category)
            if df is None or df.empty:
                return
        except Exception:
            return

        from src.strategy.indicators import calculate_sma
        sma = calculate_sma(df, self.signal_gen.sma_period)
        sma_val = sma.iloc[-1]
        if not sma_val or sma_val <= 0:
            return

        # Target = entry + stages[current_stage] * (sma - entry)
        target_price = entry + stages[current_stage] * (sma_val - entry)

        try:
            if self.exchange_type == "tbank":
                cur_price = self.client.get_last_price(symbol)
            else:
                cur_price = self.client.get_last_price(symbol, category="linear")
        except Exception:
            return

        # Buy: close when price >= target (price rose toward SMA)
        if side == "Buy" and cur_price < target_price:
            return
        # Sell: close when price <= target (price fell toward SMA)
        if side == "Sell" and cur_price > target_price:
            return

        is_last_stage = current_stage == len(stages) - 1

        if is_last_stage:
            # Last stage: close everything remaining
            close_qty = qty
        else:
            # Calculate fraction of CURRENT remaining qty
            remaining_sizes = sum(sizes[current_stage:])
            if remaining_sizes <= 0:
                close_qty = qty
            else:
                close_qty = qty * sizes[current_stage] / remaining_sizes

        # Round to exchange step, check min_qty
        cat = "tbank" if self.exchange_type == "tbank" else "linear"
        info = self._get_instrument_info(symbol, cat)
        close_qty = math.floor(close_qty / info["qty_step"]) * info["qty_step"]
        close_qty = round(close_qty, 8)

        if close_qty < info["min_qty"]:
            # Can't close partial — close everything
            close_qty = qty
            is_last_stage = True

        remaining_qty = round(qty - close_qty, 8)
        if remaining_qty < info["min_qty"]:
            # Remainder too small — close everything
            close_qty = qty
            remaining_qty = 0
            is_last_stage = True

        # Place close order
        close_side = "Sell" if side == "Buy" else "Buy"
        try:
            if self.exchange_type == "tbank":
                self.client.place_order(symbol=symbol, side=close_side, qty=close_qty)
            else:
                self.client.place_order(symbol=symbol, side=close_side, qty=close_qty, category=category, reduce_only=True)
        except Exception:
            logger.exception("Scale-out stage %d: failed to close %s", current_stage + 1, symbol)
            return

        # Calculate net PnL for this partial (after commission)
        partial_pnl = self._calc_net_pnl(side, entry, cur_price, close_qty)

        new_stage = current_stage + 1
        opened_at = trade.get("opened_at", datetime.utcnow().isoformat())

        if is_last_stage or remaining_qty <= 0:
            # Final stage — close the original trade with only the remaining portion's PnL
            try:
                balance = self.client.get_balance()
                await self.db.mark_scale_out(trade["id"], new_stage, 0)
                await self.db.close_trade(trade["id"], cur_price, partial_pnl)
                await self.db.update_daily_pnl(partial_pnl)
                await self._record_pnl(partial_pnl, balance)
            except Exception:
                logger.exception("Scale-out final: DB update failed for %s", symbol)

            if partial_pnl < 0 and self._cooldown_seconds > 0:
                self._cooldowns[symbol] = time.time() + self._cooldown_seconds
            self._scaled_in.discard(symbol)
        else:
            # Intermediate stage — insert separate closed trade for this partial
            try:
                balance = self.client.get_balance()
                await self.db.insert_partial_close(
                    symbol=symbol, side=side, category=category,
                    qty=close_qty, entry_price=entry, exit_price=cur_price,
                    pnl=partial_pnl, stage=new_stage, opened_at=opened_at,
                )
                await self.db.mark_scale_out(trade["id"], new_stage, remaining_qty)
                await self.db.update_daily_pnl(partial_pnl)
                await self._record_pnl(partial_pnl, balance)
            except Exception:
                logger.exception("Scale-out stage %d: DB update failed for %s", new_stage, symbol)

            # Move SL after each stage
            await self._move_sl_after_scale_out(symbol, side, entry, stages, current_stage, sma_val, trade["id"])

        # Notify
        dev = calculate_sma_deviation(df, self.signal_gen.sma_period)
        direction = "LONG" if side == "Buy" else "SHORT"
        stage_pct = int(stages[current_stage] * 100)

        if is_last_stage or remaining_qty <= 0:
            emoji = "🎯" if partial_pnl >= 0 else "📉"
            logger.info(
                "%s Scale-out FINAL (%d/%d): %s %s — %d%% to SMA (dev=%.1f%%, PnL=%.2f)",
                emoji, new_stage, len(stages), direction, symbol, stage_pct, dev, partial_pnl,
            )
            await self._notify(
                f"{emoji} Scale-out закрытие: {direction} {symbol}\n"
                f"Стадия {new_stage}/{len(stages)} (финал)\n"
                f"PnL: {partial_pnl:+,.2f}"
            )
        else:
            logger.info(
                "✂️ Scale-out %d/%d: %s %s — закрыто %.4f @ %d%% к SMA (dev=%.1f%%, PnL=+%.2f), осталось %.4f",
                new_stage, len(stages), direction, symbol, close_qty, stage_pct, dev, partial_pnl, remaining_qty,
            )
            await self._notify(
                f"✂️ Scale-out {new_stage}/{len(stages)}: {direction} {symbol}\n"
                f"Закрыто: {close_qty} ({int(sizes[current_stage]*100)}%)\n"
                f"PnL: {partial_pnl:+,.2f}\n"
                f"Остаток: {remaining_qty}"
            )

    async def _move_sl_after_scale_out(self, symbol: str, side: str, entry: float,
                                       stages: list, completed_stage: int, sma_val: float,
                                       trade_id: int):
        """Move SL after scale-out stage to lock in profit.

        After stage 0 (first close): SL → entry (breakeven)
        After stage 1 (second close): SL → stage 0 target price (lock profit)
        """
        if completed_stage == 0:
            # After first partial: move SL to breakeven (entry)
            new_sl = entry
        elif completed_stage >= 1:
            # After second partial: move SL to stage 0 target (lock profit from first stage)
            new_sl = entry + stages[0] * (sma_val - entry)
        else:
            return

        try:
            if self.exchange_type == "bybit":
                self.client.session.set_trading_stop(
                    category="linear", symbol=symbol,
                    stopLoss=str(round(new_sl, 6)), positionIdx=0,
                )
                logger.info("Scale-out SL moved for %s: → %.6f (stage %d done)", symbol, new_sl, completed_stage + 1)
            else:
                logger.info("Scale-out SL for %s: tbank does not support SL move, tracking in DB", symbol)
        except Exception:
            logger.warning("Failed to move SL after scale-out for %s", symbol)

        # Update SL in DB for dashboard monitoring
        try:
            await self.db.update_trade(trade_id, stop_loss=new_sl)
        except Exception:
            logger.warning("Failed to update SL in DB after scale-out for %s", symbol)

    async def _check_kotegawa_exit_single(self, trade: dict):
        """Fallback: single Kotegawa exit (original behavior when scale_out_enabled=false)."""
        symbol = trade["symbol"]
        side = trade["side"]
        entry = trade["entry_price"]
        qty = trade["qty"]
        category = trade["category"]

        exit_ratio = self.config.get("strategy", {}).get("kotegawa_exit_ratio", 0.8)

        try:
            df = self.client.get_klines(symbol, self.timeframe, limit=100, category=category)
            if df is None or df.empty:
                return
        except Exception:
            return

        from src.strategy.indicators import calculate_sma
        sma = calculate_sma(df, self.signal_gen.sma_period)
        sma_val = sma.iloc[-1]
        if not sma_val or sma_val <= 0:
            return

        target_price = entry + exit_ratio * (sma_val - entry)

        try:
            if self.exchange_type == "tbank":
                cur_price = self.client.get_last_price(symbol)
            else:
                cur_price = self.client.get_last_price(symbol, category="linear")
        except Exception:
            return

        if side == "Buy" and cur_price < target_price:
            return
        if side == "Sell" and cur_price > target_price:
            return

        # Net PnL (after commission)
        upnl = self._calc_net_pnl(side, entry, cur_price, qty)

        close_side = "Sell" if side == "Buy" else "Buy"
        try:
            if self.exchange_type == "tbank":
                self.client.place_order(symbol=symbol, side=close_side, qty=qty)
            else:
                self.client.place_order(symbol=symbol, side=close_side, qty=qty, category=category, reduce_only=True)
        except Exception:
            logger.exception("Kotegawa exit: failed to close %s", symbol)
            return

        try:
            balance = self.client.get_balance()
            await self.db.close_trade(trade["id"], cur_price, upnl)
            await self.db.update_daily_pnl(upnl)
            await self._record_pnl(upnl, balance)
        except Exception:
            logger.exception("Kotegawa exit: DB update failed for %s", symbol)

        if upnl < 0 and self._cooldown_seconds > 0:
            self._cooldowns[symbol] = time.time() + self._cooldown_seconds

        self._scaled_in.discard(symbol)

        dev = calculate_sma_deviation(df, self.signal_gen.sma_period)
        direction = "LONG" if side == "Buy" else "SHORT"
        emoji = "🎯" if upnl >= 0 else "📉"
        ratio_pct = int(exit_ratio * 100)
        logger.info(
            "%s Котегава выход: %s %s — цель %d%% к SMA (dev=%.1f%%, PnL=%.2f)",
            emoji, direction, symbol, ratio_pct, dev, upnl,
        )
        await self._notify(
            f"{emoji} Котегава выход: {direction} {symbol}\n"
            f"Цель достигнута (откл: {dev:+.1f}%)\n"
            f"PnL: {upnl:+,.2f}"
        )

    async def _check_grinder_bb_exit(self, trade: dict):
        """Grinder BB exit: DEPRECATED (v1 logic). No longer called — grinder v2 uses SL/TP only."""
        return  # v2: all exits via exchange SL/TP + auto re-entry
        if not isinstance(self.signal_gen, GrinderGenerator):
            return

        symbol = trade["symbol"]
        side = trade["side"]
        entry = trade["entry_price"]
        qty = trade["qty"]
        category = trade["category"]

        try:
            df = self.client.get_klines(symbol, self.timeframe, limit=100, category=category)
            if df is None or df.empty:
                return
        except Exception:
            return

        from src.strategy.indicators import calculate_bollinger
        upper, middle, lower = calculate_bollinger(df, self.signal_gen.bb_period, self.signal_gen.bb_std)
        mid_val = middle.iloc[-1]

        try:
            if self.exchange_type == "tbank":
                cur_price = self.client.get_last_price(symbol)
            else:
                cur_price = self.client.get_last_price(symbol, category="linear")
        except Exception:
            return

        # Update TP in DB to middle band (for dashboard display)
        try:
            await self.db.update_trade(trade["id"], take_profit=mid_val)
        except Exception:
            pass

        # Check max hold time (force close)
        max_hold = self.signal_gen.max_hold_candles
        opened_at = trade.get("opened_at")
        force_close = False
        if opened_at and max_hold > 0:
            from datetime import datetime, timezone
            if isinstance(opened_at, str):
                opened_dt = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
            else:
                opened_dt = opened_at
            if opened_dt.tzinfo is None:
                opened_dt = opened_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            # 1m candles → each candle = 60 seconds
            tf_seconds = int(self.timeframe) * 60 if self.timeframe.isdigit() else 60
            elapsed_candles = (now - opened_dt).total_seconds() / tf_seconds
            if elapsed_candles >= max_hold:
                force_close = True

        # Check if price reached middle band
        reached_middle = False
        if side == "Buy" and cur_price >= mid_val:
            reached_middle = True
        elif side == "Sell" and cur_price <= mid_val:
            reached_middle = True

        if not reached_middle and not force_close:
            return

        # Close position (net PnL after commission)
        upnl = self._calc_net_pnl(side, entry, cur_price, qty)

        close_side = "Sell" if side == "Buy" else "Buy"
        try:
            if self.exchange_type == "tbank":
                self.client.place_order(symbol=symbol, side=close_side, qty=qty)
            else:
                self.client.place_order(symbol=symbol, side=close_side, qty=qty, category=category, reduce_only=True)
        except Exception:
            logger.exception("Grinder BB exit: failed to close %s", symbol)
            return

        try:
            balance = self.client.get_balance()
            await self.db.close_trade(trade["id"], cur_price, upnl)
            await self.db.update_daily_pnl(upnl)
            await self._record_pnl(upnl, balance)
        except Exception:
            logger.exception("Grinder BB exit: DB update failed for %s", symbol)

        if upnl < 0 and self._cooldown_seconds > 0:
            self._cooldowns[symbol] = time.time() + self._cooldown_seconds

        direction = "LONG" if side == "Buy" else "SHORT"
        reason = "timeout" if force_close and not reached_middle else "middle band"
        emoji = "🎯" if upnl >= 0 else "📉"
        logger.info(
            "%s Grinder выход (%s): %s %s @ %.6f (mid=%.6f, PnL=%.2f)",
            emoji, reason, direction, symbol, cur_price, mid_val, upnl,
        )
        await self._notify(
            f"{emoji} Grinder выход ({reason}): {direction} {symbol}\n"
            f"Цена: {cur_price:,.6f} (BB mid: {mid_val:,.6f})\n"
            f"PnL: {upnl:+,.2f}"
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
        try:
            if self.exchange_type == "bybit":
                self.client.session.set_trading_stop(
                    category="linear", symbol=symbol,
                    stopLoss=str(round(entry, 6)), positionIdx=0,
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

    def _get_mtf_data(self, symbol: str, category: str) -> dict[str, pd.DataFrame]:
        """Fetch multi-timeframe klines for AI context, with per-TF caching."""
        mtf_data: dict[str, pd.DataFrame] = {}
        now = time.time()
        for tf in self._extra_timeframes:
            cache_key = f"{symbol}_{tf}"
            cached = self._mtf_cache.get(cache_key)
            cache_ttl = self._timeframe_to_seconds(tf)  # cache for one candle period
            if cached and now - cached[1] < cache_ttl:
                mtf_data[tf] = cached[0]
                continue
            try:
                tf_df = self.client.get_klines(
                    symbol=symbol, interval=tf, limit=100, category=category
                )
                tf_df.attrs["symbol"] = symbol
                mtf_data[tf] = tf_df
                self._mtf_cache[cache_key] = (tf_df, now)
            except Exception:
                logger.warning("Failed to fetch %sm klines for %s", tf, symbol)
        return mtf_data

    async def _get_fear_greed(self) -> int | None:
        """Fetch Fear & Greed Index from alternative.me, cached for 1 hour."""
        now = time.time()
        if self._fng_cache and now - self._fng_cache[1] < 3600:
            return self._fng_cache[0]

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get("https://api.alternative.me/fng/?limit=1")
                data = resp.json()
                value = int(data["data"][0]["value"])
                self._fng_cache = (value, now)
                logger.info("Fear & Greed Index: %d (%s)", value, data["data"][0].get("value_classification", ""))
                return value
        except Exception as e:
            logger.warning("Failed to fetch Fear & Greed Index: %s", e)
            return None

    def _get_funding_rate_cached(self, symbol: str, category: str) -> float:
        """Get funding rate with 30 min cache."""
        now = time.time()
        cached = self._funding_cache.get(symbol)
        if cached and now - cached[1] < 1800:
            return cached[0]

        rate = self.client.get_funding_rate(symbol, category)
        self._funding_cache[symbol] = (rate, now)
        return rate

    def _get_htf_data(self, symbol: str, category: str) -> tuple[Trend, float, float]:
        """Get higher timeframe trend + ADX + SMA deviation, cached for 5 minutes."""
        from src.strategy.indicators import calculate_adx, calculate_sma_deviation
        now = time.time()
        cached = self._htf_cache.get(symbol)
        if cached and now - cached[3] < 300:  # 5 min cache
            return cached[0], cached[1], cached[2]

        try:
            htf_df = self.client.get_klines(
                symbol=symbol, interval=self.htf_timeframe, limit=100, category=category
            )
            htf_df.attrs["symbol"] = symbol
            trend = self.signal_gen.get_htf_trend(htf_df)
            adx = calculate_adx(htf_df, self._adx_period)
            htf_sma_dev = calculate_sma_deviation(htf_df, 25)
        except Exception:
            logger.warning("Failed to get HTF data for %s, allowing trade", symbol)
            trend = Trend.NEUTRAL
            adx = 25.0  # default: allow trading
            htf_sma_dev = 0.0

        self._htf_cache[symbol] = (trend, adx, htf_sma_dev, now)
        return trend, adx, htf_sma_dev

    def _get_instrument_info(self, symbol: str, category: str) -> dict:
        key = f"{symbol}_{category}"
        if key not in self._instrument_cache:
            self._instrument_cache[key] = self.client.get_instrument_info(symbol, category)
        return self._instrument_cache[key]

    async def _notify(self, text: str):
        if self._grinder_mode:
            return
        if self.notifier:
            try:
                if self.instance_name:
                    text = f"[{self.instance_name}] {text}"
                await self.notifier(text)
            except Exception:
                logger.exception("Failed to send notification")

    # ── Weekly report ─────────────────────────────────────────

    async def _maybe_weekly_report(self):
        """Send weekly report on configured day (default Monday) at first tick after 8:00 UTC."""
        from datetime import date
        today = date.today()
        if today.weekday() != self._weekly_report_day:
            return
        if datetime.utcnow().hour < 8:
            return
        today_str = today.isoformat()
        if self._last_weekly_report == today_str:
            return

        self._last_weekly_report = today_str
        await self._send_weekly_report()

    async def _send_weekly_report(self):
        """Generate and send weekly performance report."""
        import aiosqlite
        from pathlib import Path

        name = self.instance_name or "BOT"
        stats = await self.db.get_weekly_stats()

        if stats["total"] == 0:
            lines = [
                f"📊 Недельный отчёт [{name}]",
                "─────────────",
                "Сделок за неделю: 0",
                "Бот не торговал.",
            ]
        else:
            wr = stats["win_rate"]
            lines = [
                f"📊 Недельный отчёт [{name}]",
                "─────────────",
                f"PnL за неделю: {stats['weekly_pnl']:+,.2f} USDT",
                f"Сделок: {stats['total']} ({stats['wins']}W / {stats['losses']}L)",
                f"Win Rate: {wr:.1f}%",
                f"Средний PnL: {stats['avg_pnl']:+,.2f} USDT",
            ]

            if stats["best_trade"]:
                bt = stats["best_trade"]
                d = "ЛОНГ" if bt["side"] == "Buy" else "ШОРТ"
                lines.append(f"🏆 Лучшая: {d} {bt['symbol']} → +{stats['best']:,.2f} USDT")

            if stats["worst_trade"]:
                wt = stats["worst_trade"]
                d = "ЛОНГ" if wt["side"] == "Buy" else "ШОРТ"
                lines.append(f"💀 Худшая: {d} {wt['symbol']} → {stats['worst']:,.2f} USDT")

        # Other instances
        for inst in self.config.get("other_instances", []):
            inst_name = inst.get("name", "???")
            db_path = inst.get("db_path", "")
            if not db_path or not Path(db_path).exists():
                continue
            try:
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    from datetime import date as dt_date, timedelta
                    week_ago = (dt_date.today() - timedelta(days=7)).isoformat()

                    cur = await db.execute(
                        "SELECT COALESCE(SUM(pnl), 0) as pnl FROM daily_pnl WHERE trade_date >= ?",
                        (week_ago,),
                    )
                    row = await cur.fetchone()
                    inst_pnl = float(row["pnl"])

                    cur = await db.execute(
                        "SELECT COUNT(*) as total, "
                        "COALESCE(SUM(CASE WHEN pnl >= 0 THEN 1 ELSE 0 END), 0) as wins, "
                        "COALESCE(SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END), 0) as losses "
                        "FROM trades WHERE status = 'closed' AND closed_at >= ?",
                        (week_ago,),
                    )
                    row = await cur.fetchone()
                    i_total = int(row["total"])
                    i_wins = int(row["wins"])
                    i_losses = int(row["losses"])

                    lines.append(f"\n📊 [{inst_name}] за неделю")
                    lines.append(f"PnL: {inst_pnl:+,.2f} USDT")
                    if i_total > 0:
                        i_wr = i_wins / i_total * 100
                        lines.append(f"Сделок: {i_total} ({i_wins}W / {i_losses}L) WR: {i_wr:.1f}%")
                    else:
                        lines.append("Сделок: 0")
            except Exception:
                logger.warning("Failed to read weekly stats for %s", inst_name)

        msg = "\n".join(lines)
        logger.info(msg)
        await self._notify(msg)

    # ── Balance helpers ──────────────────────────────────────

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

    async def _format_balance_block(self) -> str:
        """Format full balance block with accounts, yesterday, today, current."""
        balance = self.client.get_balance()
        daily_map = await self._get_all_daily_pnl()
        daily_total = sum(daily_map.values())

        yesterday = balance - daily_total
        arrow = "📈" if daily_total >= 0 else "📉"

        # Breakdown by bot: [SCALP]: -1,748 | [SWING]: +0
        parts = []
        for name, pnl in daily_map.items():
            parts.append(f"[{name}]: {pnl:+,.0f}")
        breakdown = " | ".join(parts)

        lines = [
            f"💰 Счёт Bybit: {balance:,.0f} USDT",
        ]

        # T-Bank balance
        tbank_balance = self._get_tbank_balance()
        if tbank_balance is not None:
            lines.append(f"🏦 Счёт TBank: {tbank_balance:,.0f} RUB")

        lines.extend([
            f"📊 Вчера: {yesterday:,.0f} USDT",
            f"{arrow} Сегодня: {daily_total:+,.0f} USDT ({breakdown})",
        ])
        return "\n".join(lines)

    def _get_tbank_balance(self) -> float | None:
        """Get T-Bank balance from tbank config (if available)."""
        import yaml
        from pathlib import Path

        # Find a tbank config from other_instances
        for inst in self.config.get("other_instances", []):
            if "TBANK" not in inst.get("name", "").upper():
                continue
            # Try to find config path for this instance
            service = inst.get("service", "")
            if not service:
                continue
            # Read config from systemd service ExecStart
            for cfg_name in ("config/tbank_scalp.yaml", "config/tbank_swing.yaml"):
                cfg_path = Path("/root/stasik") / cfg_name
                if not cfg_path.exists():
                    continue
                try:
                    with open(cfg_path) as f:
                        tbank_cfg = yaml.safe_load(f)
                    token = tbank_cfg.get("tbank", {}).get("token", "")
                    if not token or token == "YOUR_TOKEN_HERE":
                        continue
                    sandbox = tbank_cfg.get("tbank", {}).get("sandbox", True)
                    from t_tech.invest import Client
                    from t_tech.invest.constants import INVEST_GRPC_API, INVEST_GRPC_API_SANDBOX
                    target = INVEST_GRPC_API_SANDBOX if sandbox else INVEST_GRPC_API
                    with Client(token, target=target) as client:
                        accounts = client.users.get_accounts()
                        if not accounts.accounts:
                            continue
                        acc_id = accounts.accounts[0].id
                        if sandbox:
                            portfolio = client.sandbox.get_sandbox_portfolio(account_id=acc_id)
                        else:
                            portfolio = client.operations.get_portfolio(account_id=acc_id)
                        total = portfolio.total_amount_portfolio
                        return float(total.units) + float(total.nano) / 1e9
                except Exception:
                    logger.debug("Failed to get T-Bank balance", exc_info=True)
            break  # Only try once
        return None

    def _get_other_instance_prices(self, symbols: list[str], is_tbank: bool) -> dict[str, float]:
        """Get live prices for symbols from another instance's exchange."""
        prices = {}
        if not symbols:
            return prices
        try:
            if is_tbank:
                prices = self._get_tbank_prices(symbols)
            else:
                # Same exchange (Bybit) — use our client
                for sym in symbols:
                    try:
                        prices[sym] = self.client.get_last_price(sym)
                    except Exception:
                        pass
        except Exception:
            logger.debug("Failed to get prices for other instance", exc_info=True)
        return prices

    def _get_tbank_prices(self, symbols: list[str]) -> dict[str, float]:
        """Get live prices from T-Bank API for given tickers."""
        import yaml
        from pathlib import Path
        for cfg_name in ("config/tbank_scalp.yaml", "config/tbank_swing.yaml"):
            cfg_path = Path("/root/stasik") / cfg_name
            if not cfg_path.exists():
                continue
            try:
                with open(cfg_path) as f:
                    tbank_cfg = yaml.safe_load(f)
                token = tbank_cfg.get("tbank", {}).get("token", "")
                if not token or token == "YOUR_TOKEN_HERE":
                    continue
                sandbox = tbank_cfg.get("tbank", {}).get("sandbox", True)
                from t_tech.invest import Client
                from t_tech.invest.constants import INVEST_GRPC_API, INVEST_GRPC_API_SANDBOX
                target = INVEST_GRPC_API_SANDBOX if sandbox else INVEST_GRPC_API
                with Client(token, target=target) as client:
                    # Load instruments to get FIGI mapping
                    all_shares = client.instruments.shares(instrument_status=1).instruments
                    all_futures = client.instruments.futures(instrument_status=1).instruments
                    figi_map = {}
                    for inst in list(all_shares) + list(all_futures):
                        if inst.ticker in symbols:
                            figi_map[inst.ticker] = inst.figi
                    if not figi_map:
                        continue
                    figi_list = list(figi_map.values())
                    resp = client.market_data.get_last_prices(instrument_id=figi_list)
                    prices = {}
                    figi_to_ticker = {v: k for k, v in figi_map.items()}
                    for lp in resp.last_prices:
                        ticker = figi_to_ticker.get(lp.figi)
                        if ticker:
                            price = float(lp.price.units) + float(lp.price.nano) / 1e9
                            prices[ticker] = price
                    return prices
            except Exception:
                logger.debug("Failed to get T-Bank prices from %s", cfg_name, exc_info=True)
        return {}

    def _get_tbank_klines(self, symbol: str, interval: str, limit: int = 100):
        """Get klines for a T-Bank symbol by borrowing token from tbank config."""
        import yaml
        from pathlib import Path
        for cfg_name in ("config/tbank_scalp.yaml", "config/tbank_swing.yaml"):
            cfg_path = Path("/root/stasik") / cfg_name
            if not cfg_path.exists():
                continue
            try:
                with open(cfg_path) as f:
                    tbank_cfg = yaml.safe_load(f)
                token = tbank_cfg.get("tbank", {}).get("token", "")
                if not token or token == "YOUR_TOKEN_HERE":
                    continue
                sandbox = tbank_cfg.get("tbank", {}).get("sandbox", True)
                from src.exchange.tbank_client import TBankClient
                # Minimal config just for klines
                mini_cfg = {
                    "tbank": {"token": token, "sandbox": sandbox, "account_id": "", "commission_rate": 0.0004},
                    "trading": {"pairs": [symbol], "instrument_type": "share"},
                }
                tc = TBankClient(mini_cfg)
                return tc.get_klines(symbol, interval, limit=limit)
            except Exception:
                logger.debug("Failed to get T-Bank klines for %s from %s", symbol, cfg_name, exc_info=True)
        return None

    async def _get_daily_total_pnl(self) -> float:
        """Get total daily PnL across all instances (for short notifications)."""
        daily_map = await self._get_all_daily_pnl()
        return sum(daily_map.values())

    # ── Status info ──────────────────────────────────────────

    async def get_status(self) -> str:
        balance_block = await self._format_balance_block()
        open_trades = await self.db.get_open_trades()
        daily = await self.db.get_daily_pnl()
        total = await self.db.get_total_pnl()

        name = self.instance_name or "BOT"
        currency = "RUB" if self.exchange_type == "tbank" else "USDT"
        tf_label = self.timeframe if self._is_swing else f"{self.timeframe}м"

        lines = [
            balance_block,
            "",
            f"━━━ [{name}] ━━━",
            f"{'🟢 РАБОТАЕТ' if self._running else '🔴 ОСТАНОВЛЕН'}",
            f"{'⛔ СТОП — дневной лимит потерь' if self.risk.is_halted else ''}",
            f"Открытых сделок: {len(open_trades)}",
            f"За день: {daily:+,.2f} {currency}",
            f"Всего: {total:+,.2f} {currency}",
            f"Пары: {', '.join(self.pairs)}",
            f"Таймфрейм: {tf_label} | Плечо: {self.leverage}x",
        ]
        if self._market_bias_enabled and self.exchange_type == "bybit":
            bias_emoji = {"bearish": "🐻", "bullish": "🐂", "neutral": "➖"}.get(self._market_bias, "➖")
            lines.append(f"Рынок: {bias_emoji} {self._market_bias.upper()} (bias ±{self._bias_score_bonus})")

        # Show other instances
        for inst in self.config.get("other_instances", []):
            inst_lines = await self._get_other_instance_status(inst)
            if inst_lines:
                lines.append("")
                lines.extend(inst_lines)

        return "\n".join(line for line in lines if line)

    async def _get_other_instance_status(self, inst: dict) -> list[str]:
        """Read another instance's DB and show its status."""
        import subprocess
        from pathlib import Path

        name = inst.get("name", "???")
        db_path = inst.get("db_path", "")
        service = inst.get("service", "")
        tf = inst.get("timeframe", "?")
        leverage = inst.get("leverage", "?")
        pairs = inst.get("pairs", [])

        # Check if service is running
        running = False
        if service:
            try:
                result = subprocess.run(
                    ["systemctl", "is-active", service],
                    capture_output=True, text=True, timeout=3,
                )
                running = result.stdout.strip() == "active"
            except Exception:
                pass

        status_emoji = "🟢 РАБОТАЕТ" if running else "🔴 ОСТАНОВЛЕН"

        # Read PnL from DB
        daily = 0.0
        total = 0.0
        open_count = 0
        if db_path and Path(db_path).exists():
            try:
                import aiosqlite
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    # Daily PnL
                    from datetime import date
                    cur = await db.execute(
                        "SELECT pnl FROM daily_pnl WHERE trade_date = ?",
                        (date.today().isoformat(),),
                    )
                    row = await cur.fetchone()
                    if row:
                        daily = float(row["pnl"])
                    # Total PnL
                    cur = await db.execute("SELECT COALESCE(SUM(pnl), 0) as total FROM daily_pnl")
                    row = await cur.fetchone()
                    if row:
                        total = float(row["total"])
                    # Open trades
                    cur = await db.execute("SELECT COUNT(*) as cnt FROM trades WHERE status = 'open'")
                    row = await cur.fetchone()
                    if row:
                        open_count = int(row["cnt"])
            except Exception:
                logger.warning("Failed to read DB for instance %s", name)

        currency = "RUB" if "TBANK" in name.upper() else "USDT"
        lev_str = str(leverage)
        if not lev_str.endswith("x"):
            lev_str += "x"
        lines = [
            f"━━━ [{name}] ━━━",
            status_emoji,
            f"Открытых сделок: {open_count}",
            f"За день: {daily:+,.2f} {currency}",
            f"Всего: {total:+,.2f} {currency}",
            f"Пары: {', '.join(pairs)}",
            f"Таймфрейм: {tf} | Плечо: {lev_str}",
        ]
        return lines

    async def get_open_positions_list(self) -> list[dict]:
        """Return list of open positions with unrealized PnL for Telegram buttons."""
        import aiosqlite
        from pathlib import Path

        categories = self._get_categories()
        result = []

        # Current engine positions — from own DB, enriched with live price
        own_trades = await self.db.get_open_trades()
        for t in own_trades:
            sym = t["symbol"]
            entry = t["entry_price"]
            qty_val = t["qty"]
            side = t["side"]
            upnl = 0.0
            try:
                if self.exchange_type == "tbank":
                    cur_price = self.client.get_last_price(sym)
                else:
                    cur_price = self.client.get_last_price(sym, category="linear")
                upnl = self._calc_net_pnl(side, entry, cur_price, qty_val)
            except Exception:
                pass
            result.append({
                "symbol": sym,
                "side": side,
                "size": qty_val,
                "entry_price": entry,
                "upnl": upnl,
                "category": t["category"],
                "instance": self.instance_name or "BOT",
            })

        # Other instances — open trades from DB
        for inst in self.config.get("other_instances", []):
            inst_name = inst.get("name", "???")
            db_path = inst.get("db_path", "")
            is_tbank = "TBANK" in inst_name.upper()
            if db_path and Path(db_path).exists():
                try:
                    async with aiosqlite.connect(db_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute(
                            "SELECT symbol, side, entry_price, qty FROM trades WHERE status = 'open'"
                        )
                        rows = await cur.fetchall()
                        if rows:
                            symbols = [r["symbol"] for r in rows]
                            prices = self._get_other_instance_prices(symbols, is_tbank)
                            for r in rows:
                                sym = r["symbol"]
                                entry = r["entry_price"]
                                qty_val = r["qty"]
                                cur_price = prices.get(sym)
                                upnl = 0.0
                                if cur_price and entry > 0 and qty_val > 0:
                                    upnl = self._calc_net_pnl(r["side"], entry, cur_price, qty_val)
                                result.append({
                                    "symbol": sym,
                                    "side": r["side"],
                                    "size": qty_val,
                                    "entry_price": entry,
                                    "upnl": upnl,
                                    "category": "tbank" if is_tbank else "linear",
                                    "instance": inst_name,
                                })
                except Exception:
                    logger.warning("Failed to read positions from %s", inst_name, exc_info=True)

        return result

    def _find_instance_for_symbol(self, symbol: str) -> dict | None:
        """Find which other_instance owns a symbol by checking its DB."""
        import sqlite3
        from pathlib import Path
        for inst in self.config.get("other_instances", []):
            db_path = inst.get("db_path", "")
            if db_path and Path(db_path).exists():
                try:
                    conn = sqlite3.connect(db_path)
                    conn.row_factory = sqlite3.Row
                    row = conn.execute(
                        "SELECT id FROM trades WHERE symbol = ? AND status = 'open' LIMIT 1",
                        (symbol,),
                    ).fetchone()
                    conn.close()
                    if row:
                        return inst
                except Exception:
                    pass
        return None

    def _get_tbank_client_for_instance(self, inst: dict):
        """Create a temporary TBankClient for an other_instance."""
        from pathlib import Path
        config_map = {
            "stasik-tbank-scalp": "config/tbank_scalp.yaml",
            "stasik-tbank-swing": "config/tbank_swing.yaml",
        }
        service = inst.get("service", "")
        config_path = config_map.get(service)
        if not config_path or not Path(config_path).exists():
            return None
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        from src.exchange.tbank_client import TBankClient
        return TBankClient(cfg)

    async def close_position(self, symbol: str) -> str:
        """Close a single position by symbol (works across all instances)."""
        import aiosqlite

        # Determine if this symbol belongs to current engine or another instance
        is_other = False
        other_inst = None
        other_client = None
        other_db_path = None

        # Try current engine first
        closed = False
        pnl = 0.0
        close_side = "Buy"
        is_tbank = self.exchange_type == "tbank"

        try:
            categories = self._get_categories()
            for cat in categories:
                if cat not in ("linear", "tbank"):
                    continue
                if is_tbank:
                    positions = self.client.get_positions(symbol=symbol)
                else:
                    positions = self.client.get_positions(symbol=symbol, category=cat)
                for p in positions:
                    if p["symbol"] != symbol or p["size"] <= 0:
                        continue
                    close_side = "Sell" if p["side"] == "Buy" else "Buy"
                    if is_tbank:
                        self.client.place_order(symbol=symbol, side=close_side, qty=p["size"])
                    else:
                        self.client.place_order(symbol=symbol, side=close_side, qty=p["size"], category=cat, reduce_only=True)
                    closed = True
        except Exception as e:
            err_str = str(e)
            if "30079" in err_str or "not available for trading" in err_str.lower():
                return f"⏸ Биржа MOEX закрыта — {symbol} нельзя закрыть сейчас. Попробуй в торговую сессию (10:00-18:50 МСК)."
            # Symbol not on this exchange — check other instances
            pass

        if not closed:
            # Check other instances
            other_inst = self._find_instance_for_symbol(symbol)
            if other_inst and "TBANK" in other_inst.get("name", "").upper():
                try:
                    other_client = self._get_tbank_client_for_instance(other_inst)
                    if other_client:
                        positions = other_client.get_positions(symbol=symbol)
                        for p in positions:
                            if p["symbol"] != symbol or p["size"] <= 0:
                                continue
                            close_side = "Sell" if p["side"] == "Buy" else "Buy"
                            other_client.place_order(symbol=symbol, side=close_side, qty=p["size"])
                            closed = True
                            is_other = True
                            is_tbank = True
                except Exception as e:
                    err_str = str(e)
                    if "30079" in err_str or "not available for trading" in err_str.lower():
                        return f"⏸ Биржа MOEX закрыта — {symbol} нельзя закрыть сейчас. Попробуй в торговую сессию (10:00-18:50 МСК)."
                    logger.exception("Failed to close %s via other instance", symbol)
                    return f"❌ Ошибка при закрытии {symbol}"

        if not closed:
            return f"Позиция {symbol} не найдена на бирже."

        # Update DB — current engine or other instance
        db_path = other_inst["db_path"] if is_other and other_inst else None
        if is_other and db_path:
            try:
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    cur = await db.execute(
                        "SELECT id, side, entry_price, qty FROM trades WHERE symbol = ? AND status = 'open'",
                        (symbol,),
                    )
                    rows = await cur.fetchall()
                    for t in rows:
                        exit_price = other_client.get_last_price(symbol) if other_client else 0
                        if t["side"] == "Buy":
                            pnl = (exit_price - t["entry_price"]) * t["qty"]
                        else:
                            pnl = (t["entry_price"] - exit_price) * t["qty"]
                        await db.execute(
                            "UPDATE trades SET status='closed', exit_price=?, pnl=?, closed_at=datetime('now') WHERE id=?",
                            (exit_price, pnl, t["id"]),
                        )
                        # Update daily_pnl in the other instance's DB
                        today = date.today().isoformat()
                        await db.execute(
                            """INSERT INTO daily_pnl (trade_date, pnl, trades_count)
                               VALUES (?, ?, 1)
                               ON CONFLICT(trade_date)
                               DO UPDATE SET pnl = pnl + ?, trades_count = trades_count + 1""",
                            (today, pnl, pnl),
                        )
                    await db.commit()
            except Exception:
                logger.exception("Failed to update other instance DB for %s", symbol)
        else:
            open_trades = await self.db.get_open_trades()
            for t in open_trades:
                if t["symbol"] != symbol:
                    continue
                try:
                    if is_tbank:
                        exit_price = self.client.get_last_price(symbol)
                    else:
                        exit_price = self.client.get_last_price(symbol, category=t["category"])
                    if t["side"] == "Buy":
                        pnl = (exit_price - t["entry_price"]) * t["qty"]
                    else:
                        pnl = (t["entry_price"] - exit_price) * t["qty"]
                    await self.db.close_trade(t["id"], exit_price, pnl)
                    await self.db.update_daily_pnl(pnl)
                    balance = self.client.get_balance()
                    await self._record_pnl(pnl, balance)
                except Exception:
                    logger.exception("Failed to update DB for %s", symbol)

        direction = "ЛОНГ" if close_side == "Sell" else "ШОРТ"
        currency = "RUB" if is_tbank else "USDT"
        inst_tag = f"[{other_inst['name']}] " if is_other and other_inst else ""
        msg = f"❌ {inst_tag}Закрыл {direction} {symbol}"
        try:
            msg += f"\nPnL: {pnl:+,.2f} {currency}"
        except Exception:
            pass
        logger.info(msg)
        await self._notify(msg)
        return msg

    async def _close_all_on_halt(self):
        """Auto-close all positions when daily loss limit reached."""
        try:
            open_trades = await self.db.get_open_trades()
            if not open_trades:
                await self._notify(
                    f"🛑 Дневной лимит убытков достигнут ({self.risk.daily_pnl:,.0f})\n"
                    f"Новые сделки заблокированы до завтра."
                )
                return

            # Close all positions on exchange
            categories = self._get_categories()
            closed_count = 0
            for cat in categories:
                if cat not in ("linear", "tbank"):
                    continue
                if self.exchange_type == "tbank":
                    positions = self.client.get_positions()
                else:
                    positions = self.client.get_positions(category=cat)
                for p in positions:
                    try:
                        close_side = "Sell" if p["side"] == "Buy" else "Buy"
                        if self.exchange_type == "tbank":
                            self.client.place_order(symbol=p["symbol"], side=close_side, qty=p["size"])
                        else:
                            self.client.place_order(symbol=p["symbol"], side=close_side, qty=p["size"], category=cat, reduce_only=True)
                        closed_count += 1
                    except Exception:
                        logger.exception("Halt close failed: %s", p["symbol"])

            # Update DB
            for t in open_trades:
                try:
                    if self.exchange_type == "tbank":
                        exit_price = self.client.get_last_price(t["symbol"])
                    else:
                        exit_price = self.client.get_last_price(t["symbol"], category="linear")
                    if t["side"] == "Buy":
                        pnl = (exit_price - t["entry_price"]) * t["qty"]
                    else:
                        pnl = (t["entry_price"] - exit_price) * t["qty"]
                    await self.db.close_trade(t["id"], exit_price, pnl)
                    await self.db.update_daily_pnl(pnl)
                except Exception:
                    logger.exception("Halt DB update failed: %s", t["symbol"])

            await self._notify(
                f"🛑 СТОП! Дневной лимит убытков ({self.risk.daily_pnl:,.0f})\n"
                f"Закрыто {closed_count} позиций. Торговля остановлена до завтра."
            )
        except Exception:
            logger.exception("_close_all_on_halt error")

    async def close_all_positions(self) -> str:
        import aiosqlite
        from pathlib import Path

        categories = self._get_categories()
        closed = []

        # 1. Close current engine positions
        for cat in categories:
            if cat not in ("linear", "tbank"):
                continue
            if self.exchange_type == "tbank":
                positions = self.client.get_positions()
            else:
                positions = self.client.get_positions(category=cat)
            for p in positions:
                try:
                    close_side = "Sell" if p["side"] == "Buy" else "Buy"
                    if self.exchange_type == "tbank":
                        self.client.place_order(
                            symbol=p["symbol"],
                            side=close_side,
                            qty=p["size"],
                        )
                    else:
                        self.client.place_order(
                            symbol=p["symbol"],
                            side=close_side,
                            qty=p["size"],
                            category=cat,
                        )
                    closed.append(f"{p['symbol']} ({p['side']})")
                except Exception:
                    logger.exception("Failed to close position %s", p["symbol"])

        # Mark current engine DB trades as closed
        open_trades = await self.db.get_open_trades()
        for t in open_trades:
            try:
                if self.exchange_type == "tbank":
                    exit_price = self.client.get_last_price(t["symbol"])
                else:
                    exit_price = self.client.get_last_price(t["symbol"], category=t["category"])
                if t["side"] == "Buy":
                    pnl = (exit_price - t["entry_price"]) * t["qty"]
                else:
                    pnl = (t["entry_price"] - exit_price) * t["qty"]
                await self.db.close_trade(t["id"], exit_price, pnl)
                await self.db.update_daily_pnl(pnl)
                balance = self.client.get_balance()
                await self._record_pnl(pnl, balance)
            except Exception:
                logger.exception("Failed to update DB for %s", t["symbol"])

        # 2. Close other instances positions (cross-instance)
        for inst in self.config.get("other_instances", []):
            db_path = inst.get("db_path", "")
            inst_name = inst.get("name", "???")
            if not db_path or not Path(db_path).exists():
                continue
            try:
                import sqlite3 as sqlite3_sync
                conn = sqlite3_sync.connect(db_path)
                conn.row_factory = sqlite3_sync.Row
                rows = conn.execute(
                    "SELECT id, symbol, side FROM trades WHERE status = 'open'"
                ).fetchall()
                conn.close()
                if not rows:
                    continue
            except Exception:
                continue

            is_tbank = "TBANK" in inst_name.upper()
            other_client = None
            if is_tbank:
                try:
                    other_client = self._get_tbank_client_for_instance(inst)
                except Exception:
                    logger.warning("Cannot create client for %s", inst_name)
                    continue
            else:
                # Bybit instances share the same account — use own client
                other_client = self.client

            if not other_client:
                continue

            for row in rows:
                symbol = row["symbol"]
                try:
                    if is_tbank:
                        positions = other_client.get_positions(symbol=symbol)
                    else:
                        positions = other_client.get_positions(symbol=symbol, category="linear")
                    for p in positions:
                        if p["symbol"] != symbol or p["size"] <= 0:
                            continue
                        close_side = "Sell" if p["side"] == "Buy" else "Buy"
                        if is_tbank:
                            other_client.place_order(symbol=symbol, side=close_side, qty=p["size"])
                        else:
                            other_client.place_order(symbol=symbol, side=close_side, qty=p["size"], category="linear", reduce_only=True)
                        closed.append(f"{symbol} ({p['side']}) [{inst_name}]")
                except Exception as e:
                    err_str = str(e)
                    if "30079" in err_str or "not available for trading" in err_str.lower():
                        closed.append(f"{symbol} ⏸ MOEX закрыта [{inst_name}]")
                    else:
                        logger.exception("Failed to close %s on %s", symbol, inst_name)

            # Update other instance DB
            try:
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    cur = await db.execute(
                        "SELECT id, symbol, side, entry_price, qty FROM trades WHERE status = 'open'"
                    )
                    open_rows = await cur.fetchall()
                    for t in open_rows:
                        try:
                            exit_price = other_client.get_last_price(t["symbol"]) if is_tbank else other_client.get_last_price(t["symbol"], category="linear")
                            if t["side"] == "Buy":
                                pnl = (exit_price - t["entry_price"]) * t["qty"]
                            else:
                                pnl = (t["entry_price"] - exit_price) * t["qty"]
                            await db.execute(
                                "UPDATE trades SET status='closed', exit_price=?, pnl=?, closed_at=datetime('now') WHERE id=?",
                                (exit_price, pnl, t["id"]),
                            )
                            today = date.today().isoformat()
                            await db.execute(
                                """INSERT INTO daily_pnl (trade_date, pnl, trades_count)
                                   VALUES (?, ?, 1)
                                   ON CONFLICT(trade_date)
                                   DO UPDATE SET pnl = pnl + ?, trades_count = trades_count + 1""",
                                (today, pnl, pnl),
                            )
                        except Exception:
                            logger.exception("Failed to update DB for %s [%s]", t["symbol"], inst_name)
                    await db.commit()
            except Exception:
                logger.exception("Failed to update %s DB", inst_name)

        if closed:
            msg = f"❌ Закрыто {len(closed)} позиций:\n" + "\n".join(f"  • {c}" for c in closed)
        else:
            msg = "Нет позиций для закрытия."
        logger.info(msg)
        return msg

    async def get_pairs_text(self) -> str:
        """Show pairs with total closed PnL and open position status."""
        import aiosqlite
        from pathlib import Path

        currency = "RUB" if self.exchange_type == "tbank" else "USDT"
        name = self.instance_name or "BOT"

        # Get closed PnL per pair from main DB
        pair_pnl: dict[str, float] = {}
        pair_trades: dict[str, int] = {}
        try:
            db_path = self.db.db_path if hasattr(self.db, 'db_path') else self.config.get("database", {}).get("path", "")
            if db_path:
                async with aiosqlite.connect(str(db_path)) as db:
                    db.row_factory = aiosqlite.Row
                    cur = await db.execute(
                        "SELECT symbol, COALESCE(SUM(pnl), 0) as total, COUNT(*) as cnt "
                        "FROM trades WHERE status = 'closed' GROUP BY symbol"
                    )
                    for r in await cur.fetchall():
                        pair_pnl[r["symbol"]] = float(r["total"])
                        pair_trades[r["symbol"]] = int(r["cnt"])
        except Exception:
            pass

        # Open positions
        open_symbols = set()
        try:
            open_trades = await self.db.get_open_trades()
            for t in open_trades:
                open_symbols.add(t["symbol"])
        except Exception:
            pass

        lines = [f"🪙 Пары [{name}]:"]
        for pair in self.pairs:
            pnl = pair_pnl.get(pair, 0)
            cnt = pair_trades.get(pair, 0)
            emoji = "🟢" if pnl >= 0 else "🔴"
            open_mark = " 📍" if pair in open_symbols else ""
            if cnt > 0:
                lines.append(f"  {emoji} {pair}: {pnl:+,.0f} {currency} ({cnt} сделок){open_mark}")
            else:
                lines.append(f"  ⚪ {pair}: нет сделок{open_mark}")

        # Other instances
        for inst in self.config.get("other_instances", []):
            inst_name = inst.get("name", "???")
            db_path = inst.get("db_path", "")
            is_tbank = "TBANK" in inst_name.upper()
            inst_currency = "RUB" if is_tbank else "USDT"
            inst_pairs = inst.get("pairs", [])
            if not db_path or not Path(db_path).exists() or not inst_pairs:
                continue
            try:
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    cur = await db.execute(
                        "SELECT symbol, COALESCE(SUM(pnl), 0) as total, COUNT(*) as cnt "
                        "FROM trades WHERE status = 'closed' GROUP BY symbol"
                    )
                    inst_pnl = {r["symbol"]: (float(r["total"]), int(r["cnt"])) for r in await cur.fetchall()}
                    cur2 = await db.execute("SELECT DISTINCT symbol FROM trades WHERE status = 'open'")
                    inst_open = {r["symbol"] for r in await cur2.fetchall()}
            except Exception:
                inst_pnl = {}
                inst_open = set()

            lines.append(f"\n🪙 Пары [{inst_name}]:")
            for pair in inst_pairs:
                pnl_val, cnt = inst_pnl.get(pair, (0, 0))
                emoji = "🟢" if pnl_val >= 0 else "🔴"
                open_mark = " 📍" if pair in inst_open else ""
                if cnt > 0:
                    lines.append(f"  {emoji} {pair}: {pnl_val:+,.0f} {inst_currency} ({cnt} сделок){open_mark}")
                else:
                    lines.append(f"  ⚪ {pair}: нет сделок{open_mark}")

        return "\n".join(lines)

    async def get_positions_text(self) -> str:
        import aiosqlite
        from pathlib import Path

        name = self.instance_name or "BOT"
        currency = "RUB" if self.exchange_type == "tbank" else "USDT"
        categories = self._get_categories()
        lines = []
        total_pnl = 0.0
        has_positions = False

        # Current engine positions — from own DB, enriched with live price
        own_trades = await self.db.get_open_trades()
        if own_trades:
            has_positions = True
            lines.append(f"━━━ [{name}] ━━━")
        for t in own_trades:
            sym = t["symbol"]
            entry = t["entry_price"]
            size = t["qty"]
            side = t["side"]
            cat = t["category"]
            direction = "ЛОНГ" if side == "Buy" else "ШОРТ"
            upnl = 0.0
            mark = 0.0
            try:
                if self.exchange_type == "tbank":
                    mark = self.client.get_last_price(sym)
                else:
                    mark = self.client.get_last_price(sym, category="linear")
                if side == "Buy":
                    upnl = (mark - entry) * size
                else:
                    upnl = (entry - mark) * size
            except Exception:
                pass
            total_pnl += upnl
            if entry > 0 and size > 0:
                pnl_pct = (upnl / (entry * size)) * 100
            else:
                pnl_pct = 0.0

            # Target exit price: SMA for Kotegawa, TP for others
            target_line = ""
            if isinstance(self.signal_gen, KotegawaGenerator):
                try:
                    from src.strategy.indicators import calculate_sma
                    df = self.client.get_klines(sym, self.timeframe, limit=100, category=cat)
                    if df is not None and not df.empty:
                        sma = calculate_sma(df, self.signal_gen.sma_period)
                        sma_val = sma.iloc[-1]
                        if sma_val and sma_val > 0 and mark > 0:
                            exit_ratio = self.config.get("strategy", {}).get("kotegawa_exit_ratio", 0.8)
                            target = entry + exit_ratio * (sma_val - entry)
                            if side == "Buy":
                                target_pnl = (target - entry) * size
                                target_pct = (target / mark - 1) * 100
                            else:
                                target_pnl = (entry - target) * size
                                target_pct = (1 - target / mark) * 100
                            target_line = f"\n   Цель: {target:,.2f} ({target_pct:+.1f}%, {target_pnl:+,.0f} {currency})"
                except Exception:
                    pass
            elif t.get("take_profit") and t["take_profit"] > 0:
                tp = t["take_profit"]
                if side == "Buy":
                    target_pnl = (tp - entry) * size
                else:
                    target_pnl = (entry - tp) * size
                target_line = f"\n   Цель: {tp:,.2f} ({target_pnl:+,.0f} {currency})"

            # Stop-loss line
            sl_line = ""
            sl = t.get("stop_loss")
            if sl and sl > 0 and mark > 0:
                if side == "Buy":
                    sl_pct = (sl / mark - 1) * 100
                    sl_pnl = (sl - entry) * size
                else:
                    sl_pct = (1 - sl / mark) * 100
                    sl_pnl = (entry - sl) * size
                sl_line = f"\n   Стоп-лосс: {sl:,.2f} ({sl_pct:+.1f}%, {sl_pnl:+,.0f} {currency})"

            pos_value = entry * size
            emoji = "🟢" if upnl >= 0 else "🔴"
            lines.append(
                f"{emoji} {direction} {sym}\n"
                f"   Сейчас: {mark:.2f}\n"
                f"   PnL: {pnl_pct:+.2f}%, {upnl:+,.2f} {currency}{target_line}{sl_line}\n"
                f"   Сумма входа: {pos_value:,.0f} {currency}\n"
            )

        # Other instances — open trades from DB with live prices
        tbank_pnl = 0.0
        bybit_other_pnl = 0.0
        for inst in self.config.get("other_instances", []):
            inst_name = inst.get("name", "???")
            db_path = inst.get("db_path", "")
            is_tbank = "TBANK" in inst_name.upper()
            inst_currency = "RUB" if is_tbank else "USDT"
            if db_path and Path(db_path).exists():
                try:
                    async with aiosqlite.connect(db_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute(
                            "SELECT symbol, side, entry_price, qty, stop_loss FROM trades WHERE status = 'open'"
                        )
                        rows = await cur.fetchall()
                        if rows:
                            # Get live prices
                            symbols = [r["symbol"] for r in rows]
                            prices = self._get_other_instance_prices(symbols, is_tbank)
                            has_positions = True
                            lines.append(f"\n━━━ [{inst_name}] ━━━")
                            for r in rows:
                                sym = r["symbol"]
                                direction = "ЛОНГ" if r["side"] == "Buy" else "ШОРТ"
                                entry = r["entry_price"]
                                qty_val = r["qty"]
                                cur_price = prices.get(sym)
                                if cur_price and entry > 0 and qty_val > 0:
                                    upnl = self._calc_net_pnl(r["side"], entry, cur_price, qty_val)
                                    pnl_pct = (cur_price / entry - 1) * 100 if r["side"] == "Buy" else (1 - cur_price / entry) * 100
                                    if is_tbank:
                                        tbank_pnl += upnl
                                    else:
                                        bybit_other_pnl += upnl
                                    # SMA target for other Kotegawa instances
                                    target_line = ""
                                    try:
                                        from src.strategy.indicators import calculate_sma
                                        inst_tf = inst.get("timeframe", self.timeframe)
                                        # Normalize timeframe: "1м" -> "1", "15м" -> "15"
                                        inst_tf = inst_tf.replace("м", "").replace("m", "")
                                        if is_tbank:
                                            df = self._get_tbank_klines(sym, inst_tf, limit=100)
                                        else:
                                            df = self.client.get_klines(sym, inst_tf, limit=100, category="linear")
                                        if df is not None and not df.empty:
                                            sma = calculate_sma(df, 25)
                                            sma_val = sma.iloc[-1]
                                            if sma_val and sma_val > 0 and cur_price and cur_price > 0:
                                                exit_ratio = self.config.get("strategy", {}).get("kotegawa_exit_ratio", 0.8)
                                                target = entry + exit_ratio * (sma_val - entry)
                                                if r["side"] == "Buy":
                                                    t_pnl = (target - entry) * qty_val
                                                    t_pct = (target / cur_price - 1) * 100
                                                else:
                                                    t_pnl = (entry - target) * qty_val
                                                    t_pct = (1 - target / cur_price) * 100
                                                target_line = f"\n   Цель: {target:,.2f} ({t_pct:+.1f}%, {t_pnl:+,.0f} {inst_currency})"
                                    except Exception:
                                        pass
                                    # Stop-loss line
                                    sl_line = ""
                                    sl = r["stop_loss"] if r["stop_loss"] else 0
                                    if sl and sl > 0 and cur_price > 0:
                                        if r["side"] == "Buy":
                                            sl_pct = (sl / cur_price - 1) * 100
                                            sl_pnl = (sl - entry) * qty_val
                                        else:
                                            sl_pct = (1 - sl / cur_price) * 100
                                            sl_pnl = (entry - sl) * qty_val
                                        sl_line = f"\n   Стоп-лосс: {sl:,.2f} ({sl_pct:+.1f}%, {sl_pnl:+,.0f} {inst_currency})"

                                    inst_pos_value = entry * qty_val
                                    emoji = "🟢" if upnl >= 0 else "🔴"
                                    lines.append(
                                        f"{emoji} {direction} {sym}\n"
                                        f"   Сейчас: {cur_price:.2f}\n"
                                        f"   PnL: {pnl_pct:+.2f}%, {upnl:+,.2f} {inst_currency}{target_line}{sl_line}\n"
                                        f"   Сумма входа: {inst_pos_value:,.0f} {inst_currency}\n"
                                    )
                                else:
                                    inst_pos_value = entry * qty_val
                                    lines.append(
                                        f"📊 {direction} {sym}\n"
                                        f"   Сейчас: ?\n"
                                        f"   Сумма входа: {inst_pos_value:,.0f} {inst_currency}\n"
                                    )
                except Exception:
                    logger.warning("Failed to read positions from %s", inst_name, exc_info=True)

        if has_positions:
            summary = []
            bybit_total = total_pnl + bybit_other_pnl
            if bybit_total != 0 or currency == "USDT":
                e = "⬆️" if bybit_total >= 0 else "⬇️"
                summary.append(f"{e} PnL Bybit: {bybit_total:+,.2f} USDT")
            if tbank_pnl != 0:
                e = "⬆️" if tbank_pnl >= 0 else "⬇️"
                summary.append(f"{e} PnL TBank: {tbank_pnl:+,.2f} RUB")
            if summary:
                lines.append("\n" + "\n".join(summary))
        return "\n".join(lines) if has_positions else "Нет открытых позиций."

    async def get_pnl_text(self) -> str:
        import aiosqlite
        from pathlib import Path

        name = self.instance_name or "BOT"
        currency = "RUB" if self.exchange_type == "tbank" else "USDT"

        # Main instance
        daily = await self.db.get_daily_pnl()
        total = await self.db.get_total_pnl()

        lines = [
            f"━━━ [{name}] ━━━",
            f"За день: {daily:+,.2f} {currency}",
            f"Всего: {total:+,.2f} {currency}",
        ]

        # Other instances
        other_daily_sum = 0.0
        other_total_sum = 0.0
        for inst in self.config.get("other_instances", []):
            inst_name = inst.get("name", "???")
            db_path = inst.get("db_path", "")
            inst_daily = 0.0
            inst_total = 0.0
            if db_path and Path(db_path).exists():
                try:
                    async with aiosqlite.connect(db_path) as db:
                        db.row_factory = aiosqlite.Row
                        from datetime import date
                        cur = await db.execute(
                            "SELECT pnl FROM daily_pnl WHERE trade_date = ?",
                            (date.today().isoformat(),),
                        )
                        row = await cur.fetchone()
                        if row:
                            inst_daily = float(row["pnl"])
                        cur = await db.execute("SELECT COALESCE(SUM(pnl), 0) as total FROM daily_pnl")
                        row = await cur.fetchone()
                        if row:
                            inst_total = float(row["total"])
                except Exception:
                    pass
            other_daily_sum += inst_daily
            other_total_sum += inst_total
            inst_currency = "RUB" if "TBANK" in inst_name.upper() else "USDT"
            lines.append(f"\n━━━ [{inst_name}] ━━━")
            lines.append(f"За день: {inst_daily:+,.2f} {inst_currency}")
            lines.append(f"Всего: {inst_total:+,.2f} {inst_currency}")

        # Collect recent trades from all instances
        all_trades = []
        main_recent = await self.db.get_recent_trades(10)
        for t in main_recent:
            t["instance"] = t.get("instance") or name
            all_trades.append(t)

        for inst in self.config.get("other_instances", []):
            inst_name = inst.get("name", "???")
            db_path = inst.get("db_path", "")
            if db_path and Path(db_path).exists():
                try:
                    async with aiosqlite.connect(db_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute(
                            "SELECT * FROM trades ORDER BY id DESC LIMIT 10"
                        )
                        rows = await cur.fetchall()
                        for r in rows:
                            t = dict(r)
                            t["instance"] = t.get("instance") or inst_name
                            all_trades.append(t)
                except Exception:
                    pass

        all_trades.sort(key=lambda t: t.get("opened_at", ""), reverse=True)

        lines.append("")
        lines.append("Последние сделки:")
        for t in all_trades[:7]:
            pnl = t.get("pnl") or 0
            direction = "ЛОНГ" if t["side"] == "Buy" else "ШОРТ"
            result = f"+{pnl:,.2f}" if pnl >= 0 else f"{pnl:,.2f}"
            tag = t.get("instance", "")
            t_currency = "RUB" if "TBANK" in tag.upper() else "USDT"
            lines.append(
                f"  [{tag}] {direction} {t['symbol']} | {result} {t_currency} | {t['status']}"
            )
        return "\n".join(lines)
