import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from src.strategy.indicators import (
    analyze_orderbook,
    calculate_bollinger,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma_deviation,
    calculate_volume_signal,
    detect_candlestick_patterns,
    detect_order_blocks,
    detect_rsi_divergence,
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


class KotegawaGenerator:
    """Mean-reversion strategy inspired by Takashi Kotegawa (BNF).

    Core idea: buy panic dips, sell euphoria pumps — price reverts to SMA.

    Signals (each -1/0/+1, some give +2/-2 for extreme):
      sma_dev  — % deviation from 25-SMA (main signal, can give +-2)
      rsi      — extreme RSI (can give +-2)
      rsi_div  — RSI divergence (price vs RSI disagreement, +-1)
      bb       — price vs Bollinger Bands
      vol      — volume spike confirms panic/euphoria
      pattern  — candlestick reversal patterns (hammer, engulfing)
      book     — orderbook imbalance (contrarian: sellers dominate → bounce)
    """

    def __init__(self, config: dict):
        strat = config["strategy"]
        # SMA deviation thresholds (%)
        self.sma_period = strat.get("sma_period", 25)
        self.sma_mild = strat.get("sma_deviation_mild", 3.0)    # mild deviation → score +-1
        self.sma_extreme = strat.get("sma_deviation_extreme", 8.0)  # extreme → score +-2
        # RSI
        self.rsi_period = strat.get("rsi_period", 14)
        self.rsi_oversold = strat.get("rsi_oversold", 25)       # strong oversold
        self.rsi_overbought = strat.get("rsi_overbought", 75)
        self.rsi_extreme_low = strat.get("rsi_extreme_low", 15)  # extreme → +2
        self.rsi_extreme_high = strat.get("rsi_extreme_high", 85)
        # BB
        self.bb_period = strat.get("bb_period", 20)
        self.bb_std = strat.get("bb_std", 2.0)
        # Volume
        self.vol_period = strat.get("vol_period", 20)
        self.vol_threshold = strat.get("vol_threshold", 1.5)     # vol spike = panic
        # Other
        self.min_score = strat.get("min_score", 3)
        self.patterns_enabled = strat.get("patterns_enabled", True)
        # EMA for HTF (reuse params)
        self.ema_fast = strat.get("ema_fast", 9)
        self.ema_slow = strat.get("ema_slow", 21)

    def generate(self, df: pd.DataFrame, symbol: str = "?", orderbook: dict | None = None) -> SignalResult:
        df.attrs["symbol"] = symbol
        scores: dict[str, int] = {}

        # ── SMA deviation (MAIN signal) ─────────────────────
        dev = calculate_sma_deviation(df, self.sma_period)
        if dev <= -self.sma_extreme:
            scores["sma_dev"] = 2    # crashed far below SMA → strong buy
        elif dev <= -self.sma_mild:
            scores["sma_dev"] = 1    # dipped below SMA → buy
        elif dev >= self.sma_extreme:
            scores["sma_dev"] = -2   # pumped far above SMA → strong sell
        elif dev >= self.sma_mild:
            scores["sma_dev"] = -1   # above SMA → sell
        else:
            scores["sma_dev"] = 0    # near mean → no signal

        # ── RSI (extreme = double score) ─────────────────────
        rsi = calculate_rsi(df, self.rsi_period)
        rsi_val = rsi.iloc[-1]
        if rsi_val <= self.rsi_extreme_low:
            scores["rsi"] = 2        # extreme panic
        elif rsi_val < self.rsi_oversold:
            scores["rsi"] = 1        # oversold
        elif rsi_val >= self.rsi_extreme_high:
            scores["rsi"] = -2       # extreme euphoria
        elif rsi_val > self.rsi_overbought:
            scores["rsi"] = -1       # overbought
        else:
            scores["rsi"] = 0

        # ── RSI divergence ──────────────────────────────────
        rsi_div = detect_rsi_divergence(df, self.rsi_period)
        scores["rsi_div"] = rsi_div  # +1 bullish, -1 bearish, 0 none

        # ── Bollinger Bands ──────────────────────────────────
        upper, middle, lower = calculate_bollinger(df, self.bb_period, self.bb_std)
        close = df["close"].iloc[-1]

        if close < lower.iloc[-1]:
            scores["bb"] = 1         # below lower band → buy the dip
        elif close > upper.iloc[-1]:
            scores["bb"] = -1        # above upper band → sell the pump
        else:
            scores["bb"] = 0

        # ── Volume spike (contrarian: panic volume = entry) ──
        _, vol_ratio = calculate_volume_signal(df, self.vol_period)
        if vol_ratio >= self.vol_threshold:
            # High volume + red candle = panic selling → buy signal
            candle_change = close - df["open"].iloc[-1]
            if candle_change < 0:
                scores["vol"] = 1    # panic dump on volume → buy
            elif candle_change > 0:
                scores["vol"] = -1   # FOMO pump on volume → sell
            else:
                scores["vol"] = 0
        else:
            scores["vol"] = 0

        # ── Candlestick reversal patterns ────────────────────
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

        # ── Order book (contrarian) ──────────────────────────
        if orderbook and (orderbook.get("bids") or orderbook.get("asks")):
            ob_result = analyze_orderbook(orderbook)
            # Contrarian: heavy selling pressure = bounce incoming
            scores["book"] = -ob_result["score"]
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
            "Котегава %s: %s (score=%d, dev=%.1f%%, rsi=%.1f, rsi_div=%d, vol=%.1fx, details=%s)",
            symbol, signal.value, total, dev, rsi_val, rsi_div, vol_ratio, scores,
        )
        return SignalResult(signal=signal, score=total, details=scores)

    def get_htf_trend(self, df: pd.DataFrame) -> Trend:
        """HTF trend for Kotegawa — same EMA logic."""
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
