import asyncio
import logging
from datetime import datetime

import pandas as pd

from src.risk.manager import RiskManager
from src.storage.database import Database
from src.strategy.ai_analyst import AIAnalyst
from src.strategy.signals import Trend

from src.core.mixins import (
    DataFetcherMixin,
    MarketBiasMixin,
    MonitoringMixin,
    PositionCloseMixin,
    PositionCrossMixin,
    PositionOpenMixin,
    ReportingMixin,
    TurtleMixin,
    UtilsMixin,
)

logger = logging.getLogger(__name__)


class TradingEngine(
    UtilsMixin,
    DataFetcherMixin,
    MarketBiasMixin,
    PositionOpenMixin,
    PositionCloseMixin,
    PositionCrossMixin,
    TurtleMixin,
    MonitoringMixin,
    ReportingMixin,
):
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
        self._htf_cache: dict[str, tuple[Trend, float, float]] = {}
        self._adx_period: int = config["strategy"].get("adx_period", 14)
        self._adx_min: float = config["strategy"].get("adx_min", 20)

        # Fear & Greed Index
        self._fng_extreme_greed: int = config["strategy"].get("fng_extreme_greed", 80)
        self._fng_extreme_fear: int = config["strategy"].get("fng_extreme_fear", 20)
        self._fng_cache: tuple[int, float] | None = None

        # Funding rate
        self._funding_rate_max: float = config["strategy"].get("funding_rate_max", 0.0003)
        self._funding_cache: dict[str, tuple[float, float]] = {}

        # Multi-TF cache for AI
        self._mtf_cache: dict[str, tuple[pd.DataFrame, float]] = {}
        self._extra_timeframes: list[str] = config.get("ai", {}).get("extra_timeframes", ["5", "15", "60"])

        # Correlation groups: symbol -> group name
        self._corr_groups: dict[str, str] = {}
        self._max_per_group: int = config["risk"].get("max_per_group", 1)
        self._max_per_symbol: int = config["risk"].get("max_per_symbol", 1)
        for group_name, symbols in config["risk"].get("correlation_groups", {}).items():
            for s in symbols:
                self._corr_groups[s] = group_name

        # ATR config
        self._atr_period: int = config["risk"].get("atr_period", 14)
        self._atr_sl_mult: float = config["risk"].get("atr_sl_multiplier", 1.5)
        self._atr_tp_mult: float = config["risk"].get("atr_tp_multiplier", 3.0)
        self._atr_trail_mult: float = config["risk"].get("atr_trailing_multiplier", 1.0)

        # Cooldown after loss
        self._cooldowns: dict[str, float] = {}
        self._cooldown_seconds: int = config["risk"].get("cooldown_after_loss", 5) * 60

        # Breakeven
        self._breakeven_done: set[int] = set()
        self._close_at_profit: set[str] = set()
        self._breakeven_activation: float = config["risk"].get("breakeven_activation", 0.5) / 100
        self._halt_closed: bool = False

        # Trading hours filter
        trading_hours = config["trading"].get("trading_hours")
        if trading_hours and len(trading_hours) == 2:
            self._trading_hour_start: int = trading_hours[0]
            self._trading_hour_end: int = trading_hours[1]
        else:
            self._trading_hour_start = 0
            self._trading_hour_end = 24

        # Kill zone: avoid Asian session (21:00-08:00 UTC = 00:00-11:00 MSK)
        self._avoid_asia: bool = config.get("strategy", {}).get("avoid_asia", False)

        # Fast trade timeout: close if not in profit after N minutes (0 = disabled)
        self._stale_timeout_min: int = config.get("risk", {}).get("stale_timeout_min", 0)

        # Swing mode: candle dedup
        candle_sec = self._timeframe_to_seconds(str(config["trading"]["timeframe"]))
        self._is_swing = candle_sec >= 3600
        self._last_candle_ts: dict[str, float] = {}

        # Market bias
        self._market_bias: str = "neutral"
        self._market_bias_ts: float = 0
        self._market_bias_interval: int = 3600
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
        self._liquidity_log_ts: dict[str, float] = {}

        # AI lessons from strategy reviews
        self._ai_lessons: list[str] = []

        # Turtle Trading mode
        turtle_cfg = config.get("turtle", {})
        self._turtle_mode: bool = config.get("strategy", {}).get("strategy_mode") == "turtle" and bool(turtle_cfg)
        self._turtle_state: dict[str, dict] = {}
        self._turtle_last_breakout: dict[str, bool] = {}
        self._turtle_risk_pct: float = turtle_cfg.get("risk_pct", 1.0)
        self._turtle_max_units: int = turtle_cfg.get("max_units", 4)
        self._turtle_pyramid_n_mult: float = turtle_cfg.get("pyramid_n_mult", 0.5)
        self._turtle_sl_n_mult: float = turtle_cfg.get("sl_n_mult", 2.0)

        # SMC mode
        smc_cfg = config.get("smc", {})
        self._smc_mode: bool = config.get("strategy", {}).get("strategy_mode") == "smc" and bool(smc_cfg)
        self._smc_htf_structure_cache: dict[str, tuple[dict, float]] = {}
        self._smc_structure_interval: int = 300

        # SMC ZigZag swing detection
        self._smc_zigzag_enabled: bool = smc_cfg.get("zigzag_enabled", False)
        self._smc_zigzag_atr_period: int = smc_cfg.get("zigzag_atr_period", 14)
        self._smc_zigzag_atr_mult: float = smc_cfg.get("zigzag_atr_mult", 2.0)

        # SMC Fibonacci Pivot Points
        self._smc_fib_pivots: bool = smc_cfg.get("fib_pivots", False)
        self._smc_fib_pivot_proximity_pct: float = smc_cfg.get("fib_pivot_proximity_pct", 0.3)
        self._smc_daily_klines_cache: dict[str, tuple[pd.DataFrame, float]] = {}
        self._smc_daily_klines_ttl: int = 3600

        # Max margin per instance
        self._max_margin_usdt: float = config["risk"].get("max_margin_usdt", 0)
        self._margin_cache: tuple[float, float] = (0.0, 0.0)
        self._margin_cache_ttl: int = 30

        # Weekly report
        self._weekly_report_day: int = config.get("weekly_report_day", 0)
        self._last_weekly_report: str = ""

    # ── Lifecycle ────────────────────────────────────────────────────

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
            # Re-check halt condition with current balance
            if today_pnl < 0:
                balance = self.client.get_balance()
                if balance > 0 and self.risk.max_daily_loss_pct > 0:
                    if abs(today_pnl) / balance >= self.risk.max_daily_loss_pct:
                        self.risk._halted = True
                        logger.warning("Daily loss limit still active after restart: %.2f (%.1f%% of %.2f)", today_pnl, today_pnl / balance * 100, balance)

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

        await self._notify("🚀 Бот запущен\nПары: " + ", ".join(self.pairs))
        logger.info("Trading engine started")

        asyncio.create_task(self._run_sl_tp_monitor())
        await self._run_loop()

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

    # ── Main loop ────────────────────────────────────────────────────

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

        # Kill zone: avoid Asian session (21:00-08:00 UTC)
        if self._avoid_asia:
            if current_hour >= 21 or current_hour < 8:
                logger.info("Kill Zone: Азия (UTC %d:00). Пропуск сигналов.", current_hour)
                return

        categories = self._get_categories()

        for pair in self.pairs:
            for category in categories:
                try:
                    await self._process_pair(pair, category)
                except Exception:
                    logger.exception("Error processing %s (%s)", pair, category)
