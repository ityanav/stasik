import pandas as pd
import ta


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return ta.momentum.RSIIndicator(close=df["close"], window=period).rsi()


def calculate_ema(
    df: pd.DataFrame, fast: int = 9, slow: int = 21
) -> tuple[pd.Series, pd.Series]:
    ema_fast = ta.trend.EMAIndicator(close=df["close"], window=fast).ema_indicator()
    ema_slow = ta.trend.EMAIndicator(close=df["close"], window=slow).ema_indicator()
    return ema_fast, ema_slow


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_ind = ta.trend.MACD(
        close=df["close"],
        window_fast=fast,
        window_slow=slow,
        window_sign=signal,
    )
    return macd_ind.macd(), macd_ind.macd_signal(), macd_ind.macd_diff()
