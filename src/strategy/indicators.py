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


def calculate_bollinger(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    bb = ta.volatility.BollingerBands(
        close=df["close"], window=period, window_dev=std_dev
    )
    return bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Returns current ADX value. >25 = trending, <20 = ranging."""
    adx = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=period
    )
    return adx.adx().iloc[-1]


def calculate_volume_signal(df: pd.DataFrame, period: int = 20) -> tuple[pd.Series, float]:
    """Returns volume SMA and current volume ratio (current / average)."""
    vol_sma = df["volume"].rolling(window=period).mean()
    current_vol = df["volume"].iloc[-1]
    avg_vol = vol_sma.iloc[-1]
    ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
    return vol_sma, ratio
