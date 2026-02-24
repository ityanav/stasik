import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from src.strategy.indicators import (
    analyze_orderbook,
    calculate_adx,
    calculate_atr,
    calculate_bollinger,
    calculate_bollinger_bandwidth,
    calculate_bollinger_pband,
    calculate_donchian,
    calculate_ema,
    calculate_fib_pivots,
    calculate_fibonacci_levels,
    calculate_macd,
    calculate_rsi,
    calculate_sma_deviation,
    calculate_volume_signal,
    detect_candlestick_patterns,
    detect_displacement,
    detect_fair_value_gap,
    detect_liquidity_sweep,
    detect_order_blocks,
    detect_rsi_divergence,
    detect_swing_points,
    detect_swing_points_zigzag,
    find_fibonacci_clusters,
)

logger = logging.getLogger(__name__)


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class SignalResult:
    signal: Signal
    score: int
    details: dict[str, int]


class Trend(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SignalGenerator:
    def __init__(self, config: dict):
        strat = config["strategy"]
        self.rsi_period = strat["rsi_period"]
        self.rsi_overbought = strat["rsi_overbought"]
        self.rsi_oversold = strat["rsi_oversold"]
        self.ema_fast = strat["ema_fast"]
        self.ema_slow = strat["ema_slow"]
        self.macd_fast = strat["macd_fast"]
        self.macd_slow = strat["macd_slow"]
        self.macd_signal = strat["macd_signal"]
        self.bb_period = strat.get("bb_period", 20)
        self.bb_std = strat.get("bb_std", 2.0)
        self.vol_period = strat.get("vol_period", 20)
        self.vol_threshold = strat.get("vol_threshold", 1.5)
        self.min_score = strat["min_score"]
        self.patterns_enabled = strat.get("patterns_enabled", True)

    def generate(self, df: pd.DataFrame, symbol: str = "?", orderbook: dict | None = None) -> SignalResult:
        df.attrs["symbol"] = symbol
        scores: dict[str, int] = {}

        # ── RSI ───────────────────────────────────────────
        rsi = calculate_rsi(df, self.rsi_period)
        rsi_val = rsi.iloc[-1]
        if rsi_val < self.rsi_oversold:
            scores["rsi"] = 1
        elif rsi_val > self.rsi_overbought:
            scores["rsi"] = -1
        else:
            scores["rsi"] = 0

        # ── EMA crossover ─────────────────────────────────
        ema_fast, ema_slow = calculate_ema(df, self.ema_fast, self.ema_slow)
        if len(ema_fast) >= 2 and len(ema_slow) >= 2:
            prev_fast = ema_fast.iloc[-2]
            prev_slow = ema_slow.iloc[-2]
            curr_fast = ema_fast.iloc[-1]
            curr_slow = ema_slow.iloc[-1]
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                scores["ema"] = 1
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                scores["ema"] = -1
            else:
                scores["ema"] = 0
        else:
            scores["ema"] = 0

        # ── MACD crossover ────────────────────────────────
        macd_line, signal_line, _ = calculate_macd(
            df, self.macd_fast, self.macd_slow, self.macd_signal
        )
        if len(macd_line) >= 2 and len(signal_line) >= 2:
            prev_macd = macd_line.iloc[-2]
            prev_sig = signal_line.iloc[-2]
            curr_macd = macd_line.iloc[-1]
            curr_sig = signal_line.iloc[-1]
            if prev_macd <= prev_sig and curr_macd > curr_sig:
                scores["macd"] = 1
            elif prev_macd >= prev_sig and curr_macd < curr_sig:
                scores["macd"] = -1
            else:
                scores["macd"] = 0
        else:
            scores["macd"] = 0

        # ── Bollinger Bands ───────────────────────────────
        upper, middle, lower = calculate_bollinger(df, self.bb_period, self.bb_std)
        close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]

        if close <= lower.iloc[-1] and prev_close > lower.iloc[-2]:
            # Цена пробила нижнюю полосу — перепроданность → покупка
            scores["bb"] = 1
        elif close >= upper.iloc[-1] and prev_close < upper.iloc[-2]:
            # Цена пробила верхнюю полосу — перекупленность → продажа
            scores["bb"] = -1
        elif close < lower.iloc[-1]:
            # Цена ниже нижней полосы — сильная перепроданность
            scores["bb"] = 1
        elif close > upper.iloc[-1]:
            # Цена выше верхней полосы — сильная перекупленность
            scores["bb"] = -1
        else:
            scores["bb"] = 0

        # ── Volume analysis ───────────────────────────────
        _, vol_ratio = calculate_volume_signal(df, self.vol_period)

        # Объём усиливает сигнал: если объём выше порога,
        # добавляем +1/-1 в направлении движения свечи
        if vol_ratio >= self.vol_threshold:
            candle_change = close - df["open"].iloc[-1]
            if candle_change > 0:
                scores["vol"] = 1   # бычья свеча на высоком объёме
            elif candle_change < 0:
                scores["vol"] = -1  # медвежья свеча на высоком объёме
            else:
                scores["vol"] = 0
        else:
            scores["vol"] = 0

        # ── Candlestick patterns ─────────────────────────────
        if self.patterns_enabled:
            pat_result = detect_candlestick_patterns(df)
            scores["pattern"] = pat_result["score"]
        else:
            scores["pattern"] = 0

        # ── Order blocks (ICT) ──────────────────────────────
        if self.patterns_enabled:
            ob_blocks = detect_order_blocks(df)
            scores["order_block"] = ob_blocks["score"]
        else:
            scores["order_block"] = 0

        # ── Order book imbalance ─────────────────────────────
        if orderbook and (orderbook.get("bids") or orderbook.get("asks")):
            ob_result = analyze_orderbook(orderbook)
            scores["book"] = ob_result["score"]
        else:
            scores["book"] = 0

        total = sum(scores.values())

        if total >= self.min_score:
            signal = Signal.BUY
        elif total <= -self.min_score:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        logger.info(
            "Сигнал %s: %s (score=%d, rsi=%.1f, vol=%.1fx, details=%s)",
            df.attrs.get("symbol", "?"), signal.value, total, rsi_val, vol_ratio, scores,
        )
        return SignalResult(signal=signal, score=total, details=scores)

    def get_htf_trend(self, df: pd.DataFrame) -> Trend:
        """Determine trend from higher timeframe using EMA + price position."""
        if len(df) < self.ema_slow + 2:
            return Trend.NEUTRAL

        ema_fast, ema_slow = calculate_ema(df, self.ema_fast, self.ema_slow)
        fast_val = ema_fast.iloc[-1]
        slow_val = ema_slow.iloc[-1]
        price = df["close"].iloc[-1]

        # EMA alignment + price confirmation
        if fast_val > slow_val and price > fast_val:
            trend = Trend.BULLISH
        elif fast_val < slow_val and price < fast_val:
            trend = Trend.BEARISH
        else:
            trend = Trend.NEUTRAL

        symbol = df.attrs.get("symbol", "?")
        logger.info(
            "HTF тренд %s: %s (EMA9=%.2f, EMA21=%.2f, price=%.2f)",
            symbol, trend.value, fast_val, slow_val, price,
        )
        return trend


class MomentumGenerator:
    """Momentum/breakout strategy for meme coins (long-only).

    Core idea: catch the beginning of a pump via breakout + volume + momentum,
    ride it with trailing stop. Never short.

    Signals (each 0 or +1/+2, never negative):
      breakout      — price > max of last N candles (+1, +2 if on high volume)
      rsi_momentum  — RSI in 55-75 zone (rising momentum, not overheated)
      volume        — volume spike on green candle
      ema_trend     — EMA fast > EMA slow (short-term uptrend)
      bb_breakout   — price > upper BB (volatility breakout)
      pattern       — bullish candlestick pattern (positive score only)
    """

    def __init__(self, config: dict):
        strat = config["strategy"]
        self.breakout_period = strat.get("breakout_period", 20)
        self.rsi_period = strat.get("rsi_period", 14)
        self.rsi_momentum_min = strat.get("rsi_momentum_min", 55)
        self.rsi_momentum_max = strat.get("rsi_momentum_max", 75)
        self.ema_fast = strat.get("ema_fast", 9)
        self.ema_slow = strat.get("ema_slow", 21)
        self.bb_period = strat.get("bb_period", 20)
        self.bb_std = strat.get("bb_std", 2.0)
        self.vol_period = strat.get("vol_period", 20)
        self.vol_threshold = strat.get("vol_threshold", 1.5)
        self.min_score = strat.get("min_score", 3)
        self.patterns_enabled = strat.get("patterns_enabled", True)

        # Anti-pump filter
        self.pump_filter = strat.get("pump_filter", False)
        self.pump_candle_pct = strat.get("pump_candle_pct", 5.0)
        self.pump_vol_spike = strat.get("pump_vol_spike", 3.0)
        self.pump_min_steps = strat.get("pump_min_steps", 2)

    def generate(self, df: pd.DataFrame, symbol: str = "?", orderbook: dict | None = None) -> SignalResult:
        df.attrs["symbol"] = symbol
        scores: dict[str, int] = {}

        close = df["close"].iloc[-1]

        # ── Breakout: price > max of last N candles ────────────
        lookback = df["high"].iloc[-(self.breakout_period + 1):-1]
        prev_max = lookback.max()
        if close > prev_max:
            _, vol_ratio = calculate_volume_signal(df, self.vol_period)
            scores["breakout"] = 2 if vol_ratio >= self.vol_threshold else 1
        else:
            scores["breakout"] = 0

        # ── RSI momentum (55-75 zone) ─────────────────────────
        rsi = calculate_rsi(df, self.rsi_period)
        rsi_val = rsi.iloc[-1]
        if self.rsi_momentum_min <= rsi_val <= self.rsi_momentum_max:
            scores["rsi_momentum"] = 1
        else:
            scores["rsi_momentum"] = 0

        # ── Volume spike on green candle ──────────────────────
        _, vol_ratio = calculate_volume_signal(df, self.vol_period)
        candle_green = close > df["open"].iloc[-1]
        if vol_ratio >= self.vol_threshold and candle_green:
            scores["volume"] = 1
        else:
            scores["volume"] = 0

        # ── EMA trend: fast > slow ────────────────────────────
        ema_fast, ema_slow = calculate_ema(df, self.ema_fast, self.ema_slow)
        if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
            scores["ema_trend"] = 1
        else:
            scores["ema_trend"] = 0

        # ── BB breakout: price > upper band ───────────────────
        upper, _, _ = calculate_bollinger(df, self.bb_period, self.bb_std)
        if close > upper.iloc[-1]:
            scores["bb_breakout"] = 1
        else:
            scores["bb_breakout"] = 0

        # ── Bullish candlestick pattern ───────────────────────
        if self.patterns_enabled:
            pat_result = detect_candlestick_patterns(df)
            scores["pattern"] = max(pat_result["score"], 0)  # only positive
        else:
            scores["pattern"] = 0

        total = sum(scores.values())

        if total >= self.min_score:
            # Anti-pump filter: block isolated spikes without prior momentum
            is_pump, pump_reason = self._is_pump(df)
            if is_pump:
                signal = Signal.HOLD
                logger.info(
                    "Anti-pump filter: blocked %s BUY (score=%d, reason=%s)",
                    symbol, total, pump_reason,
                )
            else:
                signal = Signal.BUY
        else:
            signal = Signal.HOLD  # never SELL

        logger.info(
            "Momentum %s: %s (score=%d, rsi=%.1f, vol=%.1fx, details=%s)",
            symbol, signal.value, total, rsi_val, vol_ratio, scores,
        )
        return SignalResult(signal=signal, score=total, details=scores)

    def _is_pump(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Detect pump-and-dump: isolated spike without prior step-like momentum.

        Returns (True, reason) if pump detected, (False, "") otherwise.

        Checks:
        1. Single-candle spike > pump_candle_pct% without step structure before it
        2. Volume concentrated in 1 candle (>pump_vol_spike * avg) while prior
           candles had below-average volume (no gradual buildup)
        """
        if not self.pump_filter or len(df) < 10:
            return False, ""

        close = df["close"].iloc[-1]
        open_ = df["open"].iloc[-1]
        candle_pct = (close - open_) / open_ * 100 if open_ > 0 else 0

        # ── Check 1: single-candle spike without prior steps ────────
        if candle_pct > self.pump_candle_pct:
            # Count green "step" candles in the 5 candles before current
            steps = 0
            for i in range(-6, -1):
                if len(df) + i >= 0:
                    c = df["close"].iloc[i]
                    o = df["open"].iloc[i]
                    if c > o:
                        steps += 1

            if steps < self.pump_min_steps:
                return True, f"spike {candle_pct:.1f}% (steps={steps}<{self.pump_min_steps})"

        # ── Check 2: volume spike without buildup ───────────────────
        vol = df["volume"]
        avg_vol = vol.iloc[-self.vol_period - 1:-1].mean()
        curr_vol = vol.iloc[-1]

        if avg_vol > 0 and curr_vol > self.pump_vol_spike * avg_vol:
            # Check if prior 3 candles all had at-or-below-average volume (no buildup)
            prior_low = all(
                vol.iloc[i] <= avg_vol for i in range(-4, -1)
            )
            # And current candle is a big green move (>3%)
            if prior_low and candle_pct > 3.0:
                return True, f"vol spike {curr_vol / avg_vol:.1f}x (no buildup, +{candle_pct:.1f}%)"

        return False, ""

    def get_htf_trend(self, df: pd.DataFrame) -> Trend:
        """HTF trend for Momentum — same EMA logic."""
        if len(df) < self.ema_slow + 2:
            return Trend.NEUTRAL

        ema_fast, ema_slow = calculate_ema(df, self.ema_fast, self.ema_slow)
        fast_val = ema_fast.iloc[-1]
        slow_val = ema_slow.iloc[-1]
        price = df["close"].iloc[-1]

        if fast_val > slow_val and price > fast_val:
            trend = Trend.BULLISH
        elif fast_val < slow_val and price < fast_val:
            trend = Trend.BEARISH
        else:
            trend = Trend.NEUTRAL

        symbol = df.attrs.get("symbol", "?")
        logger.info(
            "HTF тренд %s: %s (EMA%d=%.4f, EMA%d=%.4f, price=%.4f)",
            symbol, trend.value, self.ema_fast, fast_val, self.ema_slow, slow_val, price,
        )
        return trend



class TurtleGenerator:
    """Turtle Trading: Donchian Channel breakout (trend-following).

    Richard Dennis' system: trade WITH the trend on channel breakouts.
    System 1: 20-period entry, 10-period exit. Filter: skip if last breakout profitable.
    System 2: 55-period entry, 20-period exit. No filter.
    """

    def __init__(self, config: dict):
        turtle = config.get("turtle", {})
        strat = config.get("strategy", {})

        # Donchian channel periods
        self.entry_period_s1 = turtle.get("entry_period_s1", 20)
        self.exit_period_s1 = turtle.get("exit_period_s1", 10)
        self.entry_period_s2 = turtle.get("entry_period_s2", 55)
        self.exit_period_s2 = turtle.get("exit_period_s2", 20)
        self.n_period = turtle.get("n_period", 20)

        self.system1_enabled = turtle.get("system1_enabled", True)
        self.system2_enabled = turtle.get("system2_enabled", True)
        self.min_score = strat.get("min_score", 1)

        # EMA for HTF trend
        self.ema_fast = strat.get("ema_fast", 9)
        self.ema_slow = strat.get("ema_slow", 21)

    def generate(self, df: pd.DataFrame, symbol: str = "?", orderbook: dict | None = None) -> SignalResult:
        df.attrs["symbol"] = symbol
        scores: dict[str, int] = {}
        details: dict = {}

        min_bars = max(self.entry_period_s2, self.n_period) + 2
        if len(df) < min_bars:
            return SignalResult(signal=Signal.HOLD, score=0, details={"insufficient_data": True})

        close = df["close"].iloc[-1]

        # ATR(N) — the "N" in Turtle parlance
        atr_series = df["high"].combine(df["close"].shift(1), max) - df["low"].combine(df["close"].shift(1), min)
        n_value = atr_series.rolling(window=self.n_period).mean().iloc[-1]
        if pd.isna(n_value) or n_value <= 0:
            n_value = calculate_atr(df, self.n_period)
        details["n_value"] = round(float(n_value), 6)

        # Donchian Channels (exclude current bar: use iloc[-2] as prev)
        # System 1 entry channel (20-period)
        if self.system1_enabled:
            s1_upper, s1_lower = calculate_donchian(df, self.entry_period_s1)
            prev_s1_high = s1_upper.iloc[-2]
            prev_s1_low = s1_lower.iloc[-2]
            details["s1_high"] = round(float(prev_s1_high), 6) if pd.notna(prev_s1_high) else 0
            details["s1_low"] = round(float(prev_s1_low), 6) if pd.notna(prev_s1_low) else 0

            if pd.notna(prev_s1_high) and close > prev_s1_high:
                scores["s1"] = 1
                details["s1_breakout"] = "long"
            elif pd.notna(prev_s1_low) and close < prev_s1_low:
                scores["s1"] = -1
                details["s1_breakout"] = "short"
            else:
                scores["s1"] = 0

        # System 2 entry channel (55-period)
        if self.system2_enabled:
            s2_upper, s2_lower = calculate_donchian(df, self.entry_period_s2)
            prev_s2_high = s2_upper.iloc[-2]
            prev_s2_low = s2_lower.iloc[-2]
            details["s2_high"] = round(float(prev_s2_high), 6) if pd.notna(prev_s2_high) else 0
            details["s2_low"] = round(float(prev_s2_low), 6) if pd.notna(prev_s2_low) else 0

            if pd.notna(prev_s2_high) and close > prev_s2_high:
                scores["s2"] = 1
                details["s2_breakout"] = "long"
            elif pd.notna(prev_s2_low) and close < prev_s2_low:
                scores["s2"] = -1
                details["s2_breakout"] = "short"
            else:
                scores["s2"] = 0

        # Exit channels (for position management in engine)
        exit_s1_upper, exit_s1_lower = calculate_donchian(df, self.exit_period_s1)
        exit_s2_upper, exit_s2_lower = calculate_donchian(df, self.exit_period_s2)
        details["exit_s1_high"] = round(float(exit_s1_upper.iloc[-1]), 6) if pd.notna(exit_s1_upper.iloc[-1]) else 0
        details["exit_s1_low"] = round(float(exit_s1_lower.iloc[-1]), 6) if pd.notna(exit_s1_lower.iloc[-1]) else 0
        details["exit_s2_high"] = round(float(exit_s2_upper.iloc[-1]), 6) if pd.notna(exit_s2_upper.iloc[-1]) else 0
        details["exit_s2_low"] = round(float(exit_s2_lower.iloc[-1]), 6) if pd.notna(exit_s2_lower.iloc[-1]) else 0

        # Determine which system triggered
        system = None
        if scores.get("s1", 0) != 0:
            system = 1
        if scores.get("s2", 0) != 0:
            system = 2 if system is None else system  # prefer S1 if both
        details["system"] = system

        total = sum(scores.values())

        if total >= self.min_score:
            signal = Signal.BUY
        elif total <= -self.min_score:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        logger.info(
            "Turtle %s: %s (score=%d, system=%s, n=%.4f, close=%.2f, s1=%d, s2=%d)",
            symbol, signal.value, total, system, n_value, close,
            scores.get("s1", 0), scores.get("s2", 0),
        )
        return SignalResult(signal=signal, score=total, details=details)

    def get_htf_trend(self, df: pd.DataFrame) -> Trend:
        """HTF trend for Turtle — EMA-based."""
        if len(df) < self.ema_slow + 2:
            return Trend.NEUTRAL

        ema_fast, ema_slow = calculate_ema(df, self.ema_fast, self.ema_slow)
        fast_val = ema_fast.iloc[-1]
        slow_val = ema_slow.iloc[-1]
        price = df["close"].iloc[-1]

        if fast_val > slow_val and price > fast_val:
            trend = Trend.BULLISH
        elif fast_val < slow_val and price < fast_val:
            trend = Trend.BEARISH
        else:
            trend = Trend.NEUTRAL

        symbol = df.attrs.get("symbol", "?")
        logger.info(
            "HTF тренд %s: %s (EMA%d=%.4f, EMA%d=%.4f, price=%.4f)",
            symbol, trend.value, self.ema_fast, fast_val, self.ema_slow, slow_val, price,
        )
        return trend


class SMCGenerator:
    """SMC/ICT: Fibonacci + Liquidity Sweep + Order Blocks.

    Smart Money Concepts strategy — precision entries at Fibonacci
    retracement zones after liquidity sweeps (stop hunts).

    7 indicators (scoring):
      1. fib_zone    — price in Fib retracement zone (±2 premium, ±1 standard)
      2. liq_sweep   — liquidity sweep detected (±2 fresh, ±1 old)
      3. fvg         — Fair Value Gap nearby (±1)
      4. order_block — ICT Order Block (±1)
      5. displacement — impulsive candle (±1)
      6. volume      — above-average directional volume (±1)
      7. rsi_div     — RSI divergence bonus (±1)

    min_score: 3 (need confluence). Max score ±9.
    """

    def __init__(self, config: dict):
        smc = config.get("smc", {})
        strat = config.get("strategy", {})

        # Swing detection
        self.swing_lookback = smc.get("swing_lookback", 5)
        self.swing_min_distance = smc.get("swing_min_distance", 10)

        # Fibonacci zones
        self.fib_premium_min = smc.get("fib_premium_min", 0.618)
        self.fib_premium_max = smc.get("fib_premium_max", 0.786)
        self.fib_standard_min = smc.get("fib_standard_min", 0.382)
        self.fib_standard_max = smc.get("fib_standard_max", 0.618)

        # Liquidity sweep
        self.sweep_lookback = smc.get("sweep_lookback", 30)
        self.sweep_fresh_candles = smc.get("sweep_fresh_candles", 5)

        # FVG
        self.fvg_lookback = smc.get("fvg_lookback", 20)
        self.fvg_proximity_pct = smc.get("fvg_proximity_pct", 0.5)

        # Displacement
        self.displacement_body_pct = smc.get("displacement_body_pct", 0.3)
        self.displacement_vol_mult = smc.get("displacement_vol_mult", 1.5)

        # Order Blocks (reuse existing detect_order_blocks)
        self.ob_lookback = smc.get("ob_lookback", 50)
        self.ob_proximity_pct = smc.get("ob_proximity_pct", 1.0)
        self.ob_displacement_pct = smc.get("ob_displacement_pct", 1.5)

        # TP extensions
        self.tp_extension_1 = smc.get("tp_extension_1", 1.272)
        self.tp_extension_2 = smc.get("tp_extension_2", 1.618)

        # OTE 0.705 bonus
        self.ote_bonus = smc.get("ote_bonus", False)
        self.ote_proximity_pct = smc.get("ote_proximity_pct", 0.3)

        # Fibonacci clusters
        self.cluster_enabled = smc.get("cluster_enabled", False)
        self.cluster_threshold_pct = smc.get("cluster_threshold_pct", 1.0)
        self.cluster_min_levels = smc.get("cluster_min_levels", 3)
        self.cluster_proximity_pct = smc.get("cluster_proximity_pct", 0.5)

        # Fibonacci Pivot Points
        self.fib_pivots_enabled = smc.get("fib_pivots", False)
        self.fib_pivot_proximity_pct = smc.get("fib_pivot_proximity_pct", 0.3)

        # ADX max (optional regime filter)
        self.adx_max = smc.get("adx_max", 0)

        # Volume
        self.vol_period = strat.get("vol_period", 20)
        self.vol_threshold = strat.get("vol_threshold", 1.5)

        # RSI
        self.rsi_period = strat.get("rsi_period", 14)

        # Scoring
        self.min_score = strat.get("min_score", 3)

        # EMA for HTF trend
        self.ema_fast = strat.get("ema_fast", 9)
        self.ema_slow = strat.get("ema_slow", 21)

        # Per-symbol caches (updated by engine via update_structure)
        self._htf_swings: dict[str, dict] = {}      # symbol -> swing_points
        self._fib_levels: dict[str, dict] = {}       # symbol -> fib levels
        self._fib_direction: dict[str, str] = {}     # symbol -> "bullish"/"bearish"
        self._fib_clusters: dict[str, list] = {}     # symbol -> cluster zones
        self._fib_pivots: dict[str, dict] = {}       # symbol -> pivot levels

    def update_structure(self, symbol: str, swings: dict, htf_df: pd.DataFrame):
        """Cache HTF swing structure and compute Fibonacci levels.

        Called by engine before generate() with HTF data.
        """
        self._htf_swings[symbol] = swings

        last_high = swings.get("last_swing_high")
        last_low = swings.get("last_swing_low")
        if not last_high or not last_low:
            self._fib_levels[symbol] = {}
            self._fib_direction[symbol] = ""
            return

        high_idx, high_price = last_high
        low_idx, low_price = last_low

        # Direction: if most recent swing is a high → bearish retracement
        #            if most recent swing is a low → bullish retracement
        if high_idx > low_idx:
            direction = "bearish"
        else:
            direction = "bullish"

        self._fib_direction[symbol] = direction
        self._fib_levels[symbol] = calculate_fibonacci_levels(
            high_price, low_price, direction
        )

        # Compute Fibonacci clusters from multiple swing pairs
        if self.cluster_enabled:
            self._fib_clusters[symbol] = find_fibonacci_clusters(
                swings.get("swing_highs", []),
                swings.get("swing_lows", []),
                threshold_pct=self.cluster_threshold_pct,
            )

    def update_pivots(self, symbol: str, daily_df: pd.DataFrame):
        """Cache Fibonacci Pivot Points from daily data. Called by engine."""
        if self.fib_pivots_enabled and len(daily_df) >= 2:
            self._fib_pivots[symbol] = calculate_fib_pivots(daily_df)

    def generate(self, df: pd.DataFrame, symbol: str = "?", orderbook: dict | None = None) -> SignalResult:
        df.attrs["symbol"] = symbol
        scores: dict[str, int] = {}
        details: dict = {}

        fib = self._fib_levels.get(symbol, {})
        direction = self._fib_direction.get(symbol, "")
        swings = self._htf_swings.get(symbol, {})

        if not fib or not direction:
            return SignalResult(signal=Signal.HOLD, score=0, details={"no_structure": True})

        close = df["close"].iloc[-1]
        retracement = fib.get("retracement", {})
        extensions = fib.get("extension", {})

        # ── ADX regime filter (optional) ──
        if self.adx_max > 0 and len(df) >= 30:
            adx = calculate_adx(df, period=14)
            if adx > self.adx_max:
                logger.info("SMC %s: HOLD (ADX=%.1f > %d)", symbol, adx, self.adx_max)
                return SignalResult(signal=Signal.HOLD, score=0,
                                    details={"regime_filter": True, "adx": round(adx, 1)})

        # ── 1. Fibonacci zone scoring ──
        fib_score = 0
        fib_zone_name = ""
        if retracement:
            fib_0382 = retracement.get(0.382, 0)
            fib_0618 = retracement.get(0.618, 0)
            fib_0786 = retracement.get(0.786, 0)

            if direction == "bullish":
                # Buy zone: price retraced DOWN
                premium_low = min(fib_0618, fib_0786)
                premium_high = max(fib_0618, fib_0786)
                standard_low = min(fib_0382, fib_0618)
                standard_high = max(fib_0382, fib_0618)

                if premium_low <= close <= premium_high:
                    fib_score = 2
                    fib_zone_name = "premium_buy"
                elif standard_low <= close <= standard_high:
                    fib_score = 1
                    fib_zone_name = "standard_buy"
            else:
                # Sell zone: price retraced UP
                premium_low = min(fib_0618, fib_0786)
                premium_high = max(fib_0618, fib_0786)
                standard_low = min(fib_0382, fib_0618)
                standard_high = max(fib_0382, fib_0618)

                if premium_low <= close <= premium_high:
                    fib_score = -2
                    fib_zone_name = "premium_sell"
                elif standard_low <= close <= standard_high:
                    fib_score = -1
                    fib_zone_name = "standard_sell"

        scores["fib_zone"] = fib_score
        details["fib_zone_name"] = fib_zone_name

        # ── OTE 0.705 bonus ──
        ote_bonus_val = 0
        if self.ote_bonus and retracement and fib_score != 0:
            ote_level = retracement.get(0.705, 0)
            if ote_level > 0:
                prox = close * self.ote_proximity_pct / 100
                if abs(close - ote_level) <= prox:
                    ote_bonus_val = 1 if fib_score > 0 else -1
                    details["ote_hit"] = True
        scores["ote_bonus"] = ote_bonus_val

        # ── 2. Liquidity sweep scoring ──
        sweep = detect_liquidity_sweep(
            df, swings,
            lookback=self.sweep_lookback,
            fresh_candles=self.sweep_fresh_candles,
        )
        scores["liq_sweep"] = sweep["score"]
        details["sweep_type"] = sweep["type"]
        details["swept_level"] = sweep["swept_level"]
        details["sweep_fresh"] = sweep["is_fresh"]

        # ── 3. Fair Value Gap scoring ──
        fvg = detect_fair_value_gap(
            df, lookback=self.fvg_lookback, proximity_pct=self.fvg_proximity_pct,
        )
        scores["fvg"] = fvg["score"]

        # ── 4. Order Block scoring (reuse existing) ──
        ob = detect_order_blocks(
            df,
            lookback=self.ob_lookback,
            proximity_pct=self.ob_proximity_pct,
            displacement_pct=self.ob_displacement_pct,
        )
        scores["order_block"] = ob["score"]

        # ── 5. Displacement scoring ──
        disp = detect_displacement(
            df,
            body_pct=self.displacement_body_pct,
            vol_mult=self.displacement_vol_mult,
        )
        scores["displacement"] = disp["score"]

        # ── 6. Volume scoring ──
        _, vol_ratio = calculate_volume_signal(df, self.vol_period)
        if vol_ratio >= self.vol_threshold:
            candle_change = close - df["open"].iloc[-1]
            if candle_change > 0:
                scores["volume"] = 1
            elif candle_change < 0:
                scores["volume"] = -1
            else:
                scores["volume"] = 0
        else:
            scores["volume"] = 0
        details["vol_ratio"] = round(vol_ratio, 2)

        # ── 7. RSI divergence bonus ──
        rsi_div = detect_rsi_divergence(df, self.rsi_period)
        scores["rsi_div"] = rsi_div

        # ── 8. Fibonacci Cluster bonus ──
        cluster_bonus = 0
        if self.cluster_enabled:
            clusters = self._fib_clusters.get(symbol, [])
            for cl in clusters:
                if cl["count"] >= self.cluster_min_levels:
                    prox = close * self.cluster_proximity_pct / 100
                    if abs(close - cl["price"]) <= prox:
                        cluster_bonus = 1 if fib_score > 0 else (-1 if fib_score < 0 else 0)
                        details["cluster_price"] = round(cl["price"], 6)
                        details["cluster_count"] = cl["count"]
                        break
        scores["cluster_bonus"] = cluster_bonus

        # ── 9. Fibonacci Pivot Points bonus ──
        pivot_bonus = 0
        if self.fib_pivots_enabled:
            pivots = self._fib_pivots.get(symbol, {})
            if pivots and pivots.get("pivot", 0) > 0:
                prox = close * self.fib_pivot_proximity_pct / 100
                # Check support levels (bullish)
                for key in ("s1", "s2", "s3"):
                    level = pivots.get(key, 0)
                    if level > 0 and abs(close - level) <= prox:
                        pivot_bonus = 1
                        details["pivot_level"] = key
                        break
                # Check resistance levels (bearish)
                if pivot_bonus == 0:
                    for key in ("r1", "r2", "r3"):
                        level = pivots.get(key, 0)
                        if level > 0 and abs(close - level) <= prox:
                            pivot_bonus = -1
                            details["pivot_level"] = key
                            break
        scores["pivot_bonus"] = pivot_bonus

        total = sum(scores.values())

        if total >= self.min_score:
            signal = Signal.BUY
        elif total <= -self.min_score:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        # TP levels from Fibonacci extensions
        if extensions:
            details["tp1_level"] = round(extensions.get(self.tp_extension_1, 0), 6)
            details["tp2_level"] = round(extensions.get(self.tp_extension_2, 0), 6)

        # Store sweep level for SL calculation in engine
        details["sweep_level"] = sweep["swept_level"]
        details["fib_direction"] = direction

        logger.info(
            "SMC %s: %s (score=%d, fib=%d[%s], ote=%d, sweep=%d[%s], fvg=%d, ob=%d, disp=%d, vol=%d, rsi_div=%d, cluster=%d, pivot=%d)",
            symbol, signal.value, total,
            scores["fib_zone"], fib_zone_name,
            scores.get("ote_bonus", 0),
            scores["liq_sweep"], sweep["type"],
            scores["fvg"], scores["order_block"],
            scores["displacement"], scores["volume"], scores["rsi_div"],
            scores.get("cluster_bonus", 0), scores.get("pivot_bonus", 0),
        )
        return SignalResult(signal=signal, score=total, details={**details, **scores})

    def get_htf_trend(self, df: pd.DataFrame) -> Trend:
        """HTF trend for SMC — EMA-based."""
        if len(df) < self.ema_slow + 2:
            return Trend.NEUTRAL

        ema_fast, ema_slow = calculate_ema(df, self.ema_fast, self.ema_slow)
        fast_val = ema_fast.iloc[-1]
        slow_val = ema_slow.iloc[-1]
        price = df["close"].iloc[-1]

        if fast_val > slow_val and price > fast_val:
            trend = Trend.BULLISH
        elif fast_val < slow_val and price < fast_val:
            trend = Trend.BEARISH
        else:
            trend = Trend.NEUTRAL

        symbol = df.attrs.get("symbol", "?")
        logger.info(
            "SMC HTF %s: %s (EMA%d=%.4f, EMA%d=%.4f, price=%.4f)",
            symbol, trend.value, self.ema_fast, fast_val, self.ema_slow, slow_val, price,
        )
        return trend
