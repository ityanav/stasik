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
            "Котегава %s: %s (score=%d, dev=%.1f%%, rsi=%.1f, vol=%.1fx, details=%s)",
            symbol, signal.value, total, dev, rsi_val, vol_ratio, scores,
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
