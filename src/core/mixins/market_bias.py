import logging
import time

from src.strategy.ai_analyst import extract_indicator_values, format_risk_text, summarize_candles
from src.strategy.indicators import calculate_atr, detect_swing_points, detect_swing_points_zigzag
from src.strategy.signals import Signal, SignalResult, Trend

logger = logging.getLogger(__name__)


class MarketBiasMixin:
    """Mixin for market bias detection, category helpers, and pair processing."""

    # ── Market Bias ──────────────────────────────────────────────────────

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

    # ── Categories ───────────────────────────────────────────────────────

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

    # ── Process Pair ─────────────────────────────────────────────────────

    async def _process_pair(self, symbol: str, category: str):
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
                    logger.debug("FIBA %s: no swing structure found on HTF", symbol)
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
                        logger.debug("FIBA %s: failed to fetch daily klines for pivots", symbol)

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

            # Combo filter: only allow winning indicator combinations
            combo_filter = self.config.get("strategy", {}).get("combo_filter")
            if combo_filter:
                _KEY_MAP = {
                    "fib": "fib_zone", "sweep": "liq_sweep", "fvg": "fvg",
                    "ob": "order_block", "cluster": "cluster_bonus",
                }
                _structure = set(_KEY_MAP.values())
                active = frozenset(k for k in _structure if result.details.get(k, 0) != 0)
                allowed = [frozenset(_KEY_MAP.get(x, x) for x in combo) for combo in combo_filter]
                if active not in allowed:
                    _short = {v: k for k, v in _KEY_MAP.items()}
                    active_str = "+".join(sorted(_short.get(k, k) for k in active))
                    logger.info("Combo фильтр: отклонён %s %s (комбо [%s] не в разрешённых)",
                                result.signal.value, symbol, active_str)
                    return

            side = "Buy" if result.signal == Signal.BUY else "Sell"

            # 5. Check existing positions
            open_trades = await self.db.get_open_trades()
            symbol_open = [t for t in open_trades if t["symbol"] == symbol]
            if len(symbol_open) >= self._max_per_symbol:
                return

            # One-way mode: only same direction allowed
            if symbol_open and symbol_open[0]["side"] != side:
                return

            if not self.risk.can_open_position(len(open_trades)):
                return

            if not self._check_margin_limit():
                return

            # Correlation group limit (count unique symbols, not trades)
            group = self._corr_groups.get(symbol)
            if group:
                group_symbols = set(
                    t["symbol"] for t in open_trades
                    if self._corr_groups.get(t["symbol"]) == group
                )
                if len(group_symbols) >= self._max_per_group:
                    # Allow extra positions in own symbol (already in group)
                    if symbol not in group_symbols:
                        logger.info("FIBA корреляция: отклонён %s — %d/%d символов в группе '%s'",
                                    symbol, len(group_symbols), self._max_per_group, group)
                        return

            # 6. Open trade with SMC SL/TP
            atr = calculate_atr(df, self._atr_period)
            price = df["close"].iloc[-1]

            # Skip if ATR too small — commission will eat profits
            # Need at least 0.3% ATR to cover round-trip fees
            if atr > 0 and price > 0:
                atr_pct = atr / price * 100
                if atr_pct < 0.3:
                    logger.info("FIBA %s: skip — ATR=%.4f (%.3f%%) too small, commission would eat profits",
                                symbol, atr, atr_pct)
                    return

            await self._open_trade(
                symbol, side, category, result.score, result.details,
                atr=atr, df=df,
            )
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

        is_scale_in = False
        if symbol_open:
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
