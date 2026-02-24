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


def calculate_bollinger_pband(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> float:
    """Returns Bollinger %B (0.0 = lower band, 1.0 = upper, can exceed bounds)."""
    bb = ta.volatility.BollingerBands(close=df["close"], window=period, window_dev=std_dev)
    val = bb.bollinger_pband().iloc[-1]
    return val if pd.notna(val) else 0.5


def calculate_bollinger_bandwidth(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> float:
    """Returns Bollinger Bandwidth (upper-lower)/middle — squeeze detection."""
    bb = ta.volatility.BollingerBands(close=df["close"], window=period, window_dev=std_dev)
    val = bb.bollinger_wband().iloc[-1]
    return val if pd.notna(val) else 0.0


def calculate_sma(df: pd.DataFrame, period: int = 25) -> pd.Series:
    """Returns Simple Moving Average."""
    return df["close"].rolling(window=period).mean()


def calculate_sma_deviation(df: pd.DataFrame, period: int = 25) -> float:
    """Returns % deviation of current price from SMA.
    Negative = price below SMA (oversold), Positive = above (overbought)."""
    sma = calculate_sma(df, period)
    sma_val = sma.iloc[-1]
    if pd.isna(sma_val) or sma_val == 0:
        return 0.0
    price = df["close"].iloc[-1]
    return ((price - sma_val) / sma_val) * 100


def calculate_donchian(df: pd.DataFrame, period: int = 20) -> tuple[pd.Series, pd.Series]:
    """Donchian Channel: rolling max(high) / min(low) over N periods."""
    upper = df["high"].rolling(window=period).max()
    lower = df["low"].rolling(window=period).min()
    return upper, lower


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Returns current ATR value (Average True Range)."""
    atr = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=period
    )
    value = atr.average_true_range().iloc[-1]
    return value if pd.notna(value) else 0.0


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


def detect_rsi_divergence(df: pd.DataFrame, rsi_period: int = 14, lookback: int = 20) -> int:
    """Detect RSI divergence (price vs RSI disagreement).

    Returns:
        +1 = bullish divergence (price lower low, RSI higher low → reversal up)
        -1 = bearish divergence (price higher high, RSI lower high → reversal down)
         0 = no divergence
    """
    if len(df) < lookback + rsi_period:
        return 0

    rsi = calculate_rsi(df, rsi_period)
    recent = df.tail(lookback)
    rsi_recent = rsi.tail(lookback)

    if rsi_recent.isna().any():
        return 0

    prices = recent["close"].values
    rsi_vals = rsi_recent.values

    # Split into two halves: first half vs second half
    mid = lookback // 2
    first_prices = prices[:mid]
    second_prices = prices[mid:]
    first_rsi = rsi_vals[:mid]
    second_rsi = rsi_vals[mid:]

    price_low1 = first_prices.min()
    price_low2 = second_prices.min()
    rsi_low1 = first_rsi.min()
    rsi_low2 = second_rsi.min()

    price_high1 = first_prices.max()
    price_high2 = second_prices.max()
    rsi_high1 = first_rsi.max()
    rsi_high2 = second_rsi.max()

    # Bullish divergence: price makes lower low, RSI makes higher low
    if price_low2 < price_low1 and rsi_low2 > rsi_low1 + 2:
        return 1

    # Bearish divergence: price makes higher high, RSI makes lower high
    if price_high2 > price_high1 and rsi_high2 < rsi_high1 - 2:
        return -1

    return 0


def analyze_orderbook(orderbook: dict, depth_pct: float = 0.5) -> dict:
    """Analyze order book imbalance and walls.

    Args:
        orderbook: {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}
        depth_pct: how deep to look (% from mid price)

    Returns:
        {"score": -1/0/+1, "bid_vol": float, "ask_vol": float,
         "imbalance": float (-1..+1), "walls": str}
    """
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    if not bids or not asks:
        return {"score": 0, "bid_vol": 0, "ask_vol": 0, "imbalance": 0, "walls": ""}

    mid_price = (bids[0][0] + asks[0][0]) / 2
    depth_range = mid_price * depth_pct / 100

    # Sum volume within depth range
    bid_vol = sum(q for p, q in bids if p >= mid_price - depth_range)
    ask_vol = sum(q for p, q in asks if p <= mid_price + depth_range)

    total = bid_vol + ask_vol
    if total == 0:
        return {"score": 0, "bid_vol": 0, "ask_vol": 0, "imbalance": 0, "walls": ""}

    # Imbalance: +1 = all bids (bullish), -1 = all asks (bearish)
    imbalance = (bid_vol - ask_vol) / total

    # Detect walls (single level with > 20% of total volume in range)
    walls = []
    wall_threshold = total * 0.20
    for p, q in bids[:10]:
        if q >= wall_threshold:
            walls.append(f"BID wall {p:.4g} ({q:.0f})")
    for p, q in asks[:10]:
        if q >= wall_threshold:
            walls.append(f"ASK wall {p:.4g} ({q:.0f})")

    # Score: strong imbalance → signal
    if imbalance > 0.3:
        score = 1   # buyers dominate
    elif imbalance < -0.3:
        score = -1  # sellers dominate
    else:
        score = 0

    return {
        "score": score,
        "bid_vol": round(bid_vol, 2),
        "ask_vol": round(ask_vol, 2),
        "imbalance": round(imbalance, 3),
        "walls": "; ".join(walls) if walls else "",
    }


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


def detect_order_blocks(
    df: pd.DataFrame,
    lookback: int = 50,
    proximity_pct: float = 1.0,
    displacement_pct: float = 1.5,
) -> dict:
    """Detect ICT Order Blocks — institutional supply/demand zones.

    Bullish OB: last bearish candle before impulsive move up (displacement).
    Bearish OB: last bullish candle before impulsive move down.
    Mitigated blocks (price already passed through zone) are skipped.

    Returns {"score": -1/0/+1, "blocks": [...], "nearest": dict|None}
    """
    empty = {"score": 0, "blocks": [], "nearest": None}
    if len(df) < lookback + 3:
        return empty

    close = df["close"].iloc[-1]
    recent = df.iloc[-(lookback + 3):]
    blocks: list[dict] = []

    for i in range(len(recent) - 2):
        candle = recent.iloc[i]
        c_open, c_close = candle["open"], candle["close"]
        c_high, c_low = candle["high"], candle["low"]
        body = abs(c_close - c_open)
        if body == 0:
            continue

        is_bearish = c_close < c_open
        is_bullish = c_close > c_open

        # Check displacement over next 1-2 candles
        next1 = recent.iloc[i + 1]
        next2 = recent.iloc[i + 2] if i + 2 < len(recent) else None

        if is_bearish:
            # Bullish OB: bearish candle → impulsive move UP
            max_close = next1["close"]
            if next2 is not None:
                max_close = max(max_close, next2["close"])
            displacement = (max_close - c_high) / c_high * 100
            if displacement >= displacement_pct and max_close > c_high:
                zone_low = min(c_open, c_close)
                zone_high = max(c_open, c_close)
                # Check mitigation: if price later closed below zone_low, it's mitigated
                subsequent = recent.iloc[i + 2:]
                mitigated = (subsequent["close"] < zone_low).any()
                if not mitigated:
                    blocks.append({
                        "type": "bullish",
                        "zone_low": zone_low,
                        "zone_high": zone_high,
                        "age": len(recent) - i - 1,
                    })

        elif is_bullish:
            # Bearish OB: bullish candle → impulsive move DOWN
            min_close = next1["close"]
            if next2 is not None:
                min_close = min(min_close, next2["close"])
            displacement = (c_low - min_close) / c_low * 100
            if displacement >= displacement_pct and min_close < c_low:
                zone_low = min(c_open, c_close)
                zone_high = max(c_open, c_close)
                # Mitigated if price later closed above zone_high
                subsequent = recent.iloc[i + 2:]
                mitigated = (subsequent["close"] > zone_high).any()
                if not mitigated:
                    blocks.append({
                        "type": "bearish",
                        "zone_low": zone_low,
                        "zone_high": zone_high,
                        "age": len(recent) - i - 1,
                    })

    if not blocks:
        return empty

    # Find nearest block to current price
    def distance(b: dict) -> float:
        mid = (b["zone_low"] + b["zone_high"]) / 2
        return abs(close - mid)

    nearest = min(blocks, key=distance)
    prox = close * proximity_pct / 100

    # Score: is current price near a block?
    score = 0
    if nearest["type"] == "bullish" and nearest["zone_low"] - prox <= close <= nearest["zone_high"] + prox:
        score = 1  # price at bullish support → confirms BUY
    elif nearest["type"] == "bearish" and nearest["zone_low"] - prox <= close <= nearest["zone_high"] + prox:
        score = -1  # price at bearish resistance → confirms SELL

    return {"score": score, "blocks": blocks, "nearest": nearest}
