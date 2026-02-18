import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from src.strategy.indicators import calculate_ema, calculate_macd, calculate_rsi

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
        self.min_score = strat["min_score"]

    def generate(self, df: pd.DataFrame) -> SignalResult:
        scores: dict[str, int] = {}

        # RSI
        rsi = calculate_rsi(df, self.rsi_period)
        rsi_val = rsi.iloc[-1]
        if rsi_val < self.rsi_oversold:
            scores["rsi"] = 1
        elif rsi_val > self.rsi_overbought:
            scores["rsi"] = -1
        else:
            scores["rsi"] = 0

        # EMA crossover
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

        # MACD crossover
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

        total = sum(scores.values())

        if total >= self.min_score:
            signal = Signal.BUY
        elif total <= -self.min_score:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        logger.debug(
            "Signal: %s (score=%d, rsi=%.1f, details=%s)",
            signal.value, total, rsi_val, scores,
        )
        return SignalResult(signal=signal, score=total, details=scores)
