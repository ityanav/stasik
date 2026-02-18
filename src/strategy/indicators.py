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


def detect_candlestick_patterns(df: pd.DataFrame) -> dict:
    """Detect candlestick patterns on last 3 candles.
    Returns {"score": clamped[-1,+1], "patterns": {name: score}}."""
    if len(df) < 3:
        return {"score": 0, "patterns": {}}

    c = df.iloc[-1]   # current
    p = df.iloc[-2]   # previous
    pp = df.iloc[-3]  # two bars ago

    patterns: dict[str, int] = {}

    body = abs(c["close"] - c["open"])
    full_range = c["high"] - c["low"]
    if full_range == 0:
        return {"score": 0, "patterns": {}}

    upper_shadow = c["high"] - max(c["close"], c["open"])
    lower_shadow = min(c["close"], c["open"]) - c["low"]
    is_green = c["close"] > c["open"]

    # Doji
    if body / full_range < 0.1:
        patterns["doji"] = 0

    # Hammer (bullish reversal)
    if body > 0 and lower_shadow >= 2 * body and upper_shadow < body:
        patterns["hammer"] = 1

    # Shooting star (bearish reversal)
    if body > 0 and upper_shadow >= 2 * body and lower_shadow < body:
        patterns["shooting_star"] = -1

    # Engulfing patterns
    p_body_lo = min(p["close"], p["open"])
    p_body_hi = max(p["close"], p["open"])
    c_body_lo = min(c["close"], c["open"])
    c_body_hi = max(c["close"], c["open"])

    if p["close"] < p["open"] and is_green and c_body_lo <= p_body_lo and c_body_hi >= p_body_hi:
        patterns["bullish_engulfing"] = 1

    if p["close"] > p["open"] and not is_green and c_body_lo <= p_body_lo and c_body_hi >= p_body_hi:
        patterns["bearish_engulfing"] = -1

    # Morning star (3-candle bullish)
    pp_body = abs(pp["close"] - pp["open"])
    p_body_size = abs(p["close"] - p["open"])
    if (pp["close"] < pp["open"]
            and pp_body > 0 and p_body_size < pp_body * 0.3
            and is_green
            and c["close"] > (pp["open"] + pp["close"]) / 2):
        patterns["morning_star"] = 1

    # Evening star (3-candle bearish)
    if (pp["close"] > pp["open"]
            and pp_body > 0 and p_body_size < pp_body * 0.3
            and not is_green
            and c["close"] < (pp["open"] + pp["close"]) / 2):
        patterns["evening_star"] = -1

    total = sum(patterns.values())
    score = max(-1, min(1, total))
    return {"score": score, "patterns": patterns}
