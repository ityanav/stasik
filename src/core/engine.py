import asyncio
import logging
import time
from datetime import datetime

import httpx
import pandas as pd

from src.exchange.client import BybitClient
from src.risk.manager import RiskManager
from src.storage.database import Database
from src.strategy.ai_analyst import AIAnalyst, extract_indicator_values, format_risk_text, summarize_candles
from src.strategy.indicators import calculate_atr
from src.strategy.signals import Signal, SignalGenerator, Trend

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, config: dict, notifier=None, db_path: str | None = None):
        self.config = config
        self.instance_name = config.get("instance_name", "")
        self.client = BybitClient(config)
        self.signal_gen = SignalGenerator(config)
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
        self._breakeven_activation: float = config["risk"].get("breakeven_activation", 0.5) / 100

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

        # Weekly report
        self._weekly_report_day: int = config.get("weekly_report_day", 0)  # 0=Monday
        self._last_weekly_report: str = ""  # ISO date of last report

    async def start(self):
        await self.db.connect()
        self._running = True

        # Set leverage for futures pairs
        if self.market_type in ("futures", "both"):
            for pair in self.pairs:
                self.client.set_leverage(pair, self.leverage, category="linear")

        await self._notify("🚀 Бот запущен\nПары: " + ", ".join(self.pairs))
        logger.info("Trading engine started")

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
            logger.info("Trading halted — daily loss limit")
            return

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

        result = self.signal_gen.generate(df)

        if result.signal == Signal.HOLD:
            return

        # HTF trend + ADX filter
        htf_trend, adx = self._get_htf_data(symbol, category)

        # ADX filter: skip if market is ranging (no clear trend)
        if adx < self._adx_min:
            logger.info("ADX фильтр: отклонён %s %s (ADX=%.1f < %d — боковик)",
                        result.signal.value, symbol, adx, self._adx_min)
            return

        # HTF trend filter: only trade in direction of higher timeframe trend
        if htf_trend == Trend.BEARISH and result.signal == Signal.BUY:
            logger.info("HTF фильтр: отклонён BUY %s (тренд 15м медвежий)", symbol)
            return
        if htf_trend == Trend.BULLISH and result.signal == Signal.SELL:
            logger.info("HTF фильтр: отклонён SELL %s (тренд 15м бычий)", symbol)
            return

        # Fear & Greed Index filter
        fng_value = await self._get_fear_greed()
        if fng_value is not None:
            if fng_value > self._fng_extreme_greed and result.signal == Signal.BUY:
                logger.info("FnG фильтр: отклонён BUY %s (FnG=%d > %d — Extreme Greed)",
                            symbol, fng_value, self._fng_extreme_greed)
                return
            if fng_value < self._fng_extreme_fear and result.signal == Signal.SELL:
                logger.info("FnG фильтр: отклонён SELL %s (FnG=%d < %d — Extreme Fear)",
                            symbol, fng_value, self._fng_extreme_fear)
                return

        # Funding rate filter
        funding_rate = self._get_funding_rate_cached(symbol, category)
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

        if symbol_open:
            logger.debug("Already have open trade for %s (%s), skipping", symbol, category)
            return

        open_count = len(open_trades)
        if not self.risk.can_open_position(open_count):
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

        side = "Buy" if result.signal == Signal.BUY else "Sell"

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

            # Multi-TF context
            mtf_data = self._get_mtf_data(symbol, category)

            verdict = await self.ai_analyst.analyze(
                signal=result.signal.value,
                score=result.score,
                details=result.details,
                indicator_text=indicator_text,
                candles_text=candles_text,
                risk_text=risk_text,
                mtf_data=mtf_data,
                config=self.config,
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
                await self._notify(msg)
                return
            else:
                ai_reasoning = f"🤖 AI ({verdict.confidence}/10): {verdict.reasoning}"
                ai_sl = verdict.stop_loss
                ai_tp = verdict.take_profit
                ai_size_mult = verdict.position_size
                logger.info("AI confirmed %s %s — confidence %d/10", side, symbol, verdict.confidence)

        await self._open_trade(
            symbol, side, category, result.score, result.details,
            ai_reasoning=ai_reasoning,
            ai_sl_pct=ai_sl,
            ai_tp_pct=ai_tp,
            ai_size_mult=ai_size_mult,
            atr=atr,
        )

    async def _open_trade(
        self, symbol: str, side: str, category: str, score: int, details: dict,
        ai_reasoning: str = "",
        ai_sl_pct: float | None = None,
        ai_tp_pct: float | None = None,
        ai_size_mult: float | None = None,
        atr: float | None = None,
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
        # Use ATR SL% for adaptive position sizing
        sizing_sl = atr_sl_pct if atr_sl_pct else None
        qty = self.risk.calculate_position_size(
            balance=balance,
            price=price,
            qty_step=info["qty_step"],
            min_qty=info["min_qty"],
            sl_pct=sizing_sl,
        )

        # AI position size multiplier
        if ai_size_mult is not None and 0.1 <= ai_size_mult <= 2.0:
            import math
            qty = math.floor((qty * ai_size_mult) / info["qty_step"]) * info["qty_step"]
            qty = round(qty, 8)

        if qty <= 0:
            logger.info("Position size too small for %s, skipping", symbol)
            return

        # SL/TP priority: AI > ATR > fixed
        sl_source = "fixed"
        tp_source = "fixed"

        if ai_sl_pct is not None and 0.3 <= ai_sl_pct <= 5.0:
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

        order = self.client.place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            category=category,
            stop_loss=sl if category == "linear" else None,
            take_profit=tp if category == "linear" else None,
        )

        order_id = order.get("orderId", "")
        await self.db.insert_trade(
            symbol=symbol,
            side=side,
            category=category,
            qty=qty,
            entry_price=price,
            stop_loss=sl,
            take_profit=tp,
            order_id=order_id,
        )

        # Set trailing stop for futures (ATR-based or fixed)
        trailing_msg = ""
        if category == "linear":
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
        )

        if update.error:
            logger.warning("AI review failed: %s", update.error)
            return

        if not update.changes:
            logger.info("AI review: no changes needed")
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
        self.signal_gen = SignalGenerator(self.config)
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
        logger.info(msg)
        await self._notify(msg)

    # ── Monitor closed trades ─────────────────────────────────

    async def _check_closed_trades(self):
        open_trades = await self.db.get_open_trades()
        if not open_trades:
            return

        for trade in open_trades:
            if trade["category"] != "linear":
                continue
            try:
                await self._check_breakeven(trade)
                if not trade.get("partial_closed"):
                    await self._check_partial_close(trade)
                await self._check_trade_closed(trade)
            except Exception:
                logger.exception("Error checking trade %s", trade["id"])

    async def _check_trade_closed(self, trade: dict):
        symbol = trade["symbol"]
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

        current_price = self.client.get_last_price(symbol, category="linear")

        if side == "Buy":
            profit_pct = (current_price - entry) / entry
        else:
            profit_pct = (entry - current_price) / entry

        if profit_pct < self._breakeven_activation:
            return

        # Move SL to entry (breakeven)
        try:
            self.client.session.set_trading_stop(
                category="linear", symbol=symbol,
                stopLoss=str(round(entry, 6)), positionIdx=0,
            )
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

        import math

        partial_pct = self.config["risk"].get("partial_close_pct", 50) / 100
        partial_trigger = self.config["risk"].get("partial_close_trigger", 50) / 100

        symbol = trade["symbol"]
        entry = trade["entry_price"]
        tp = trade["take_profit"]
        side = trade["side"]
        qty = trade["qty"]

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
        info = self._get_instrument_info(symbol, "linear")
        close_qty = math.floor(close_qty / info["qty_step"]) * info["qty_step"]
        close_qty = round(close_qty, 8)

        if close_qty < info["min_qty"]:
            return

        remaining_qty = round(qty - close_qty, 8)
        close_side = "Sell" if side == "Buy" else "Buy"

        try:
            self.client.place_order(
                symbol=symbol, side=close_side, qty=close_qty, category="linear",
            )
        except Exception:
            logger.exception("Failed to partial close %s", symbol)
            return

        # Move SL to breakeven
        try:
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
        """Format full balance block with initial, yesterday, today, current."""
        initial = self.config.get("initial_balance", 0)
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
            f"💰 Инвестировал: {initial:,.0f} USDT",
            f"📊 Вчера: {yesterday:,.0f} USDT",
            f"{arrow} Сегодня: {daily_total:+,.0f} USDT ({breakdown})",
            f"💵 Сейчас: {balance:,.0f} USDT",
        ]
        return "\n".join(lines)

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
        tf_label = self.timeframe if self._is_swing else f"{self.timeframe}м"

        lines = [
            balance_block,
            "",
            f"━━━ [{name}] ━━━",
            f"{'🟢 РАБОТАЕТ' if self._running else '🔴 ОСТАНОВЛЕН'}",
            f"{'⛔ СТОП — дневной лимит потерь' if self.risk.is_halted else ''}",
            f"Открытых сделок: {len(open_trades)}",
            f"За день: {daily:+,.2f} USDT",
            f"Всего: {total:+,.2f} USDT",
            f"Пары: {', '.join(self.pairs)}",
            f"Таймфрейм: {tf_label} | Плечо: {self.leverage}x",
        ]

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

        lines = [
            f"━━━ [{name}] ━━━",
            status_emoji,
            f"Открытых сделок: {open_count}",
            f"За день: {daily:+,.2f} USDT",
            f"Всего: {total:+,.2f} USDT",
            f"Пары: {', '.join(pairs)}",
            f"Таймфрейм: {tf} | Плечо: {leverage}x",
        ]
        return lines

    async def close_all_positions(self) -> str:
        categories = self._get_categories()
        closed = []
        for cat in categories:
            if cat != "linear":
                continue
            positions = self.client.get_positions(category=cat)
            for p in positions:
                try:
                    close_side = "Sell" if p["side"] == "Buy" else "Buy"
                    self.client.place_order(
                        symbol=p["symbol"],
                        side=close_side,
                        qty=p["size"],
                        category=cat,
                    )
                    closed.append(f"{p['symbol']} ({p['side']})")
                except Exception:
                    logger.exception("Failed to close position %s", p["symbol"])

        # Mark DB trades as closed
        open_trades = await self.db.get_open_trades()
        for t in open_trades:
            try:
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

        if closed:
            msg = f"❌ Закрыто {len(closed)} позиций:\n" + "\n".join(f"  • {c}" for c in closed)
        else:
            msg = "Нет позиций для закрытия."
        logger.info(msg)
        await self._notify(msg)
        return msg

    async def get_positions_text(self) -> str:
        categories = self._get_categories()
        lines = []
        total_pnl = 0.0
        for cat in categories:
            if cat == "linear":
                positions = self.client.get_positions(category=cat)
                for p in positions:
                    direction = "ЛОНГ" if p["side"] == "Buy" else "ШОРТ"
                    upnl = p["unrealised_pnl"]
                    total_pnl += upnl
                    entry = p["entry_price"]
                    # PnL в процентах от входа
                    if entry > 0 and p["size"] > 0:
                        pnl_pct = (upnl / (entry * p["size"])) * 100
                    else:
                        pnl_pct = 0.0
                    emoji = "🟢" if upnl >= 0 else "🔴"
                    lines.append(
                        f"{emoji} {direction} {p['symbol']}\n"
                        f"   Вход: {entry} | Объём: {p['size']}\n"
                        f"   PnL: {upnl:+,.2f} USDT ({pnl_pct:+.2f}%)"
                    )
        if lines:
            total_emoji = "🟢" if total_pnl >= 0 else "🔴"
            lines.append(f"\n{total_emoji} Итого PnL: {total_pnl:+,.2f} USDT")
        return "\n".join(lines) if lines else "Нет открытых позиций."

    async def get_pnl_text(self) -> str:
        import aiosqlite
        from pathlib import Path

        name = self.instance_name or "BOT"

        # Main instance
        daily = await self.db.get_daily_pnl()
        total = await self.db.get_total_pnl()

        lines = [
            f"━━━ [{name}] ━━━",
            f"За день: {daily:+,.2f} USDT",
            f"Всего: {total:+,.2f} USDT",
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
            lines.append(f"\n━━━ [{inst_name}] ━━━")
            lines.append(f"За день: {inst_daily:+,.2f} USDT")
            lines.append(f"Всего: {inst_total:+,.2f} USDT")

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
            lines.append(
                f"  [{tag}] {direction} {t['symbol']} | {result} USDT | {t['status']}"
            )
        return "\n".join(lines)
