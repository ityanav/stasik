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
from src.strategy.indicators import calculate_atr, calculate_sma_deviation
from src.strategy.signals import KotegawaGenerator, Signal, SignalGenerator, SignalResult, Trend

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

        # AI lessons from strategy reviews
        self._ai_lessons: list[str] = []

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
        return SignalGenerator(config)

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

        # Restore cooldowns from recent losing trades closed today
        await self._restore_cooldowns()

        # Reconcile exchange positions with DB
        await self._reconcile_positions()

        await self._notify("🚀 Бот запущен\nПары: " + ", ".join(self.pairs))
        logger.info("Trading engine started")

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

        # Trading hours filter (UTC)
        current_hour = datetime.utcnow().hour
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

    async def _process_pair(self, symbol: str, category: str):
        # Cooldown check (before API calls to save quota)
        now = time.time()
        cooldown_until = self._cooldowns.get(symbol, 0)
        if now < cooldown_until:
            remaining = int(cooldown_until - now)
            logger.debug("Кулдаун %s: ещё %d сек", symbol, remaining)
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
            return

        # HTF trend + ADX filter
        htf_trend, adx = self._get_htf_data(symbol, category)

        # ADX filter: skip if market is ranging (no clear trend)
        if adx < self._adx_min:
            logger.info("ADX фильтр: отклонён %s %s (ADX=%.1f < %d — боковик)",
                        result.signal.value, symbol, adx, self._adx_min)
            return

        # HTF trend filter: block counter-trend trades unless signal is strong (score >= 2)
        # Skip HTF filter if disabled in config (for aggressive meme trading)
        if self.config.get("strategy", {}).get("htf_filter", True):
            if htf_trend == Trend.BEARISH and result.signal == Signal.BUY and abs(result.score) < 2:
                logger.info("HTF фильтр: отклонён BUY %s (тренд HTF медвежий, score=%d)", symbol, result.score)
                return
            if htf_trend == Trend.BULLISH and result.signal == Signal.SELL and abs(result.score) < 2:
                logger.info("HTF фильтр: отклонён SELL %s (тренд HTF бычий, score=%d)", symbol, result.score)
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

        # SL/TP priority: Kotegawa (recent low/high) > AI > ATR > fixed
        sl_source = "fixed"
        tp_source = "fixed"

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

        if kotegawa_sl is not None:
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

        if ai_tp_pct is not None and 0.5 <= ai_tp_pct <= 10.0:
            tp = price * (1 + ai_tp_pct / 100) if side == "Buy" else price * (1 - ai_tp_pct / 100)
            tp = round(tp, 6)
            tp_source = f"AI:{ai_tp_pct:.1f}%"
        elif atr_tp_pct:
            tp = self.risk.calculate_sl_tp_atr(price, side, atr, self._atr_sl_mult, self._atr_tp_mult)[1]
            tp_source = f"ATR:{atr_tp_pct*100:.2f}%"
        else:
            tp = self.risk.calculate_sl_tp(price, side)[1]

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
            f"Цена: {price}\n"
            f"Объём: {qty}{size_note} (~{pos_value:,.0f} USDT)\n"
            f"SL: {sl} ({sl_source}) | TP: {tp} ({tp_source}){trailing_msg}{atr_note}\n"
            f"Баланс: {balance:,.0f} USDT ({day_arrow} сегодня: {daily_total:+,.0f})"
        )
        if ai_reasoning:
            msg += f"\n{ai_reasoning}"
        logger.info(msg)
        await self._notify(msg)

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
                await self._check_breakeven(trade)
                if not trade.get("partial_closed"):
                    await self._check_partial_close(trade)
                await self._check_smart_exit(trade)
                await self._check_kotegawa_exit(trade)
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
            if side == "Buy":
                pnl = (exit_price - entry_price) * qty
            else:
                pnl = (entry_price - exit_price) * qty
            logger.info("Using calculated PnL for %s: %.2f (fallback)", symbol, pnl)

        # Update DB
        balance = self.client.get_balance()
        await self.db.close_trade(trade["id"], exit_price, pnl)
        await self.db.update_daily_pnl(pnl)
        self.risk.record_pnl(pnl, balance)

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

        if side == "Buy":
            upnl = (cur_price - entry) * qty
        else:
            upnl = (entry - cur_price) * qty

        if upnl <= 0:
            return

        # In profit — close it
        close_side = "Sell" if side == "Buy" else "Buy"
        try:
            if self.exchange_type == "tbank":
                self.client.place_order(symbol=symbol, side=close_side, qty=qty)
            else:
                self.client.place_order(symbol=symbol, side=close_side, qty=qty, category="linear")
        except Exception:
            logger.exception("Close-at-profit: failed to close %s", symbol)
            return

        self._close_at_profit.discard(symbol)
        currency = "RUB" if self.exchange_type == "tbank" else "USDT"
        direction = "ЛОНГ" if side == "Buy" else "ШОРТ"
        logger.info("✅ Close-at-profit: закрыл %s %s при PnL +%.0f %s", direction, symbol, upnl, currency)
        await self._notify(
            f"✅ Закрыл {direction} {symbol} в плюсе\n"
            f"PnL: +{upnl:,.0f} {currency}"
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

        # Calculate unrealized PnL
        if side == "Buy":
            upnl = (cur_price - entry) * qty
        else:
            upnl = (entry - cur_price) * qty

        if upnl < target:
            return

        # Close the position
        close_side = "Sell" if side == "Buy" else "Buy"
        try:
            if self.exchange_type == "tbank":
                self.client.place_order(symbol=symbol, side=close_side, qty=qty)
            else:
                self.client.place_order(symbol=symbol, side=close_side, qty=qty, category="linear")
        except Exception:
            logger.exception("Profit target: failed to close %s", symbol)
            return

        # Update DB
        exit_price = cur_price
        actual_pnl = upnl
        try:
            balance = self.client.get_balance()
            await self.db.close_trade(trade["id"], exit_price, actual_pnl)
            await self.db.update_daily_pnl(actual_pnl)
            self.risk.record_pnl(actual_pnl, balance)
        except Exception:
            logger.exception("Profit target: DB update failed for %s", symbol)

        direction = "ЛОНГ" if side == "Buy" else "ШОРТ"
        logger.info(
            "💰 Profit target: закрыл %s %s при PnL +%.0f (цель: %d)",
            direction, symbol, upnl, target,
        )
        await self._notify(
            f"💰 Profit target: закрыл {direction} {symbol}\n"
            f"PnL: +{upnl:,.0f} USDT (цель: {target}$)"
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

        if side == "Buy":
            upnl = (cur_price - entry) * qty
        else:
            upnl = (entry - cur_price) * qty

        if upnl > -target:
            return  # loss not big enough yet

        # Close the position
        close_side = "Sell" if side == "Buy" else "Buy"
        try:
            if self.exchange_type == "tbank":
                self.client.place_order(symbol=symbol, side=close_side, qty=qty)
            else:
                self.client.place_order(symbol=symbol, side=close_side, qty=qty, category="linear")
        except Exception:
            logger.exception("Loss target: failed to close %s", symbol)
            return

        exit_price = cur_price
        actual_pnl = upnl
        try:
            balance = self.client.get_balance()
            await self.db.close_trade(trade["id"], exit_price, actual_pnl)
            await self.db.update_daily_pnl(actual_pnl)
            self.risk.record_pnl(actual_pnl, balance)
        except Exception:
            logger.exception("Loss target: DB update failed for %s", symbol)

        direction = "ЛОНГ" if side == "Buy" else "ШОРТ"
        logger.info(
            "🛑 Loss target: закрыл %s %s при PnL %.0f (лимит: -%d)",
            direction, symbol, upnl, target,
        )
        await self._notify(
            f"🛑 Loss target: закрыл {direction} {symbol}\n"
            f"PnL: {upnl:,.0f} USDT (лимит: -{target}$)"
        )

    async def _check_kotegawa_exit(self, trade: dict):
        """Close position when price reverts toward SMA (Kotegawa mean-reversion exit).

        Closes at kotegawa_exit_ratio (default 0.8 = 80%) of the way from entry to SMA.
        Smaller target = higher probability of hitting it.

        Config (strategy section):
          kotegawa_exit_ratio: 0.8  # close at 80% of distance from entry to SMA
        """
        if not isinstance(self.signal_gen, KotegawaGenerator):
            return

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

        # Target = entry + ratio * (sma - entry)
        target_price = entry + exit_ratio * (sma_val - entry)

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

        # Price reached target — close position
        try:
            if self.exchange_type == "tbank":
                cur_price = self.client.get_last_price(symbol)
            else:
                cur_price = self.client.get_last_price(symbol, category="linear")
        except Exception:
            return

        if side == "Buy":
            upnl = (cur_price - entry) * qty
        else:
            upnl = (entry - cur_price) * qty

        close_side = "Sell" if side == "Buy" else "Buy"
        try:
            if self.exchange_type == "tbank":
                self.client.place_order(symbol=symbol, side=close_side, qty=qty)
            else:
                self.client.place_order(symbol=symbol, side=close_side, qty=qty, category=category)
        except Exception:
            logger.exception("Kotegawa exit: failed to close %s", symbol)
            return

        try:
            balance = self.client.get_balance()
            await self.db.close_trade(trade["id"], cur_price, upnl)
            await self.db.update_daily_pnl(upnl)
            self.risk.record_pnl(upnl, balance)
        except Exception:
            logger.exception("Kotegawa exit: DB update failed for %s", symbol)

        if upnl < 0 and self._cooldown_seconds > 0:
            self._cooldowns[symbol] = time.time() + self._cooldown_seconds

        self._scaled_in.discard(symbol)

        dev = calculate_sma_deviation(df, self.signal_gen.sma_period)
        direction = "ЛОНГ" if side == "Buy" else "ШОРТ"
        emoji = "🎯" if upnl >= 0 else "📉"
        ratio_pct = int(exit_ratio * 100)
        logger.info(
            "%s Котегава выход: %s %s — цель %d%% к SMA (dev=%.1f%%, PnL=%.2f)",
            emoji, direction, symbol, ratio_pct, dev, upnl,
        )
        await self._notify(
            f"{emoji} Котегава выход: {direction} {symbol}\n"
            f"Цель {ratio_pct}% к SMA достигнута (откл: {dev:+.1f}%)\n"
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

        # Calculate unrealized PnL
        if side == "Buy":
            upnl = (cur_price - entry) * qty
        else:
            upnl = (entry - cur_price) * qty

        if upnl < exit_min or upnl > exit_max:
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
                self.client.place_order(symbol=symbol, side=close_side, qty=qty, category="linear")
        except Exception:
            logger.exception("Smart exit: failed to close %s", symbol)
            return

        # Update DB
        try:
            balance = self.client.get_balance()
            await self.db.close_trade(trade["id"], cur_price, upnl)
            await self.db.update_daily_pnl(upnl)
            self.risk.record_pnl(upnl, balance)
        except Exception:
            logger.exception("Smart exit: DB update failed for %s", symbol)

        direction = "ЛОНГ" if side == "Buy" else "ШОРТ"
        logger.info(
            "🧠 Smart exit: закрыл %s %s при PnL +%.0f (сигнал развернулся: score=%d)",
            direction, symbol, upnl, result.score,
        )
        await self._notify(
            f"🧠 Smart exit: закрыл {direction} {symbol}\n"
            f"PnL: +{upnl:,.0f} (сигнал развернулся, score={result.score})"
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

        if side == "Buy":
            profit_pct = (current_price - entry) / entry
        else:
            profit_pct = (entry - current_price) / entry

        if profit_pct < self._breakeven_activation:
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
            f"Прибыль: {profit_pct*100:.2f}% → SL перенесён на вход ({entry})"
        )
        logger.info(msg)
        await self._notify(msg)

    async def _check_partial_close(self, trade: dict):
        """Close 50% of position when price reaches 50% of TP distance."""
        if not self.config["risk"].get("partial_close_enabled", True):
            return

        partial_pct = self.config["risk"].get("partial_close_pct", 50) / 100
        partial_trigger = self.config["risk"].get("partial_close_trigger", 50) / 100

        symbol = trade["symbol"]
        entry = trade["entry_price"]
        tp = trade["take_profit"]
        side = trade["side"]
        qty = trade["qty"]

        if self.exchange_type == "tbank":
            current_price = self.client.get_last_price(symbol)
        else:
            current_price = self.client.get_last_price(symbol, category="linear")

        if side == "Buy":
            tp_distance = tp - entry
            current_progress = current_price - entry
        else:
            tp_distance = entry - tp
            current_progress = entry - current_price

        if tp_distance <= 0:
            return

        progress_ratio = current_progress / tp_distance
        if progress_ratio < partial_trigger:
            return

        # Close partial position
        close_qty = qty * partial_pct
        category = "tbank" if self.exchange_type == "tbank" else "linear"
        info = self._get_instrument_info(symbol, category)
        close_qty = math.floor(close_qty / info["qty_step"]) * info["qty_step"]
        close_qty = round(close_qty, 8)

        if close_qty < info["min_qty"]:
            return

        remaining_qty = round(qty - close_qty, 8)
        close_side = "Sell" if side == "Buy" else "Buy"

        try:
            if self.exchange_type == "tbank":
                self.client.place_order(symbol=symbol, side=close_side, qty=close_qty)
            else:
                self.client.place_order(symbol=symbol, side=close_side, qty=close_qty, category="linear")
        except Exception:
            logger.exception("Failed to partial close %s", symbol)
            return

        # Move SL to breakeven
        try:
            if self.exchange_type == "bybit":
                self.client.session.set_trading_stop(
                    category="linear", symbol=symbol,
                    stopLoss=str(round(entry, 6)), positionIdx=0,
                )
        except Exception:
            logger.warning("Failed to move SL to breakeven for %s", symbol)

        await self.db.mark_partial_close(trade["id"], remaining_qty)

        if side == "Buy":
            partial_pnl = (current_price - entry) * close_qty
        else:
            partial_pnl = (entry - current_price) * close_qty

        msg = (
            f"✂️ Частичное закрытие {symbol}\n"
            f"Закрыто: {close_qty} из {qty} ({partial_pct*100:.0f}%)\n"
            f"Прибыль: +{partial_pnl:,.2f} USDT\n"
            f"SL → безубыток ({entry})\n"
            f"Остаток: {remaining_qty} — бежит к TP"
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

    def _get_htf_data(self, symbol: str, category: str) -> tuple[Trend, float]:
        """Get higher timeframe trend + ADX, cached for 5 minutes."""
        from src.strategy.indicators import calculate_adx
        now = time.time()
        cached = self._htf_cache.get(symbol)
        if cached and now - cached[2] < 300:  # 5 min cache
            return cached[0], cached[1]

        try:
            htf_df = self.client.get_klines(
                symbol=symbol, interval=self.htf_timeframe, limit=100, category=category
            )
            htf_df.attrs["symbol"] = symbol
            trend = self.signal_gen.get_htf_trend(htf_df)
            adx = calculate_adx(htf_df, self._adx_period)
        except Exception:
            logger.warning("Failed to get HTF data for %s, allowing trade", symbol)
            trend = Trend.NEUTRAL
            adx = 25.0  # default: allow trading

        self._htf_cache[symbol] = (trend, adx, now)
        return trend, adx

    def _get_instrument_info(self, symbol: str, category: str) -> dict:
        key = f"{symbol}_{category}"
        if key not in self._instrument_cache:
            self._instrument_cache[key] = self.client.get_instrument_info(symbol, category)
        return self._instrument_cache[key]

    async def _notify(self, text: str):
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
                if side == "Buy":
                    upnl = (cur_price - entry) * qty_val
                else:
                    upnl = (entry - cur_price) * qty_val
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
                                    if r["side"] == "Buy":
                                        upnl = (cur_price - entry) * qty_val
                                    else:
                                        upnl = (entry - cur_price) * qty_val
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
                        self.client.place_order(symbol=symbol, side=close_side, qty=p["size"], category=cat)
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
                    self.risk.record_pnl(pnl, balance)
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
                            self.client.place_order(symbol=p["symbol"], side=close_side, qty=p["size"], category=cat)
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
                self.risk.record_pnl(pnl, balance)
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
                            other_client.place_order(symbol=symbol, side=close_side, qty=p["size"], category="linear")
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
                            ratio_pct = int(exit_ratio * 100)
                            target_line = f"\n   Цель ({ratio_pct}% SMA): {target:,.4f} ({target_pct:+.1f}%, {target_pnl:+,.0f} {currency})"
                except Exception:
                    pass
            elif t.get("take_profit") and t["take_profit"] > 0:
                tp = t["take_profit"]
                if side == "Buy":
                    target_pnl = (tp - entry) * size
                else:
                    target_pnl = (entry - tp) * size
                target_line = f"\n   Цель (TP): {tp:,.4f} ({target_pnl:+,.0f} {currency})"

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
                sl_line = f"\n   Стоп-лосс: {sl:,.4f} ({sl_pct:+.1f}%, {sl_pnl:+,.0f} {currency})"

            emoji = "🟢" if upnl >= 0 else "🔴"
            lines.append(
                f"{emoji} {direction} {sym}\n"
                f"   Вход: {entry}  →  Сейчас: {mark}\n"
                f"   Объём: {size} | PnL: ({pnl_pct:+.2f}%, {upnl:+,.2f} {currency}){target_line}{sl_line}\n"
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
                                    if r["side"] == "Buy":
                                        upnl = (cur_price - entry) * qty_val
                                    else:
                                        upnl = (entry - cur_price) * qty_val
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
                                                ratio_pct = int(exit_ratio * 100)
                                                target_line = f"\n   Цель ({ratio_pct}% SMA): {target:,.4f} ({t_pct:+.1f}%, {t_pnl:+,.0f} {inst_currency})"
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
                                        sl_line = f"\n   Стоп-лосс: {sl:,.4f} ({sl_pct:+.1f}%, {sl_pnl:+,.0f} {inst_currency})"

                                    emoji = "🟢" if upnl >= 0 else "🔴"
                                    lines.append(
                                        f"{emoji} {direction} {sym}\n"
                                        f"   Вход: {entry}  →  Сейчас: {cur_price}\n"
                                        f"   Объём: {qty_val} | PnL: ({pnl_pct:+.2f}%, {upnl:+,.2f} {inst_currency}){target_line}{sl_line}\n"
                                    )
                                else:
                                    lines.append(
                                        f"📊 {direction} {sym}\n"
                                        f"   Вход: {entry}  →  Сейчас: ?\n"
                                        f"   Объём: {qty_val}\n"
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
