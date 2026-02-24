import numpy as np
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


# ── SMC (Smart Money Concepts) indicators ──────────────────────────


def detect_swing_points(
    df: pd.DataFrame, lookback: int = 5, min_distance: int = 10
) -> dict:
    """Detect fractal swing highs/lows.

    A swing high is a bar whose high is >= the highs of `lookback` bars
    on each side. Same logic (inverted) for swing lows.

    Args:
        lookback: number of bars on each side to compare.
        min_distance: minimum bars between swings of the same type.

    Returns:
        {"swing_highs": [(idx, price), ...], "swing_lows": [...],
         "last_swing_high": tuple|None, "last_swing_low": tuple|None}
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []

    for i in range(lookback, n - lookback):
        # Swing high: high[i] >= all highs in [i-lookback, i+lookback]
        window_highs = highs[i - lookback : i + lookback + 1]
        if highs[i] >= window_highs.max():
            if not swing_highs or (i - swing_highs[-1][0]) >= min_distance:
                swing_highs.append((i, float(highs[i])))

        # Swing low: low[i] <= all lows in [i-lookback, i+lookback]
        window_lows = lows[i - lookback : i + lookback + 1]
        if lows[i] <= window_lows.min():
            if not swing_lows or (i - swing_lows[-1][0]) >= min_distance:
                swing_lows.append((i, float(lows[i])))

    return {
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
        "last_swing_high": swing_highs[-1] if swing_highs else None,
        "last_swing_low": swing_lows[-1] if swing_lows else None,
    }


def calculate_fibonacci_levels(
    swing_high: float, swing_low: float, direction: str
) -> dict:
    """Calculate Fibonacci retracement and extension levels.

    Args:
        swing_high: the swing high price.
        swing_low: the swing low price.
        direction: "bullish" (retracement down from high to low, buy zone)
                   or "bearish" (retracement up from low to high, sell zone).

    Returns:
        {"retracement": {0.236: price, ...}, "extension": {1.272: price, ...},
         "range": float}
    """
    price_range = swing_high - swing_low
    if price_range <= 0:
        return {"retracement": {}, "extension": {}, "range": 0.0}

    ret_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    ext_levels = [1.272, 1.618, 2.0]

    retracement: dict[float, float] = {}
    extension: dict[float, float] = {}

    if direction == "bullish":
        # Retracement down from high: buy zone is below
        for lvl in ret_levels:
            retracement[lvl] = swing_high - price_range * lvl
        # Extension above high (TP targets)
        for lvl in ext_levels:
            extension[lvl] = swing_low + price_range * lvl
    else:
        # Bearish: retracement up from low, sell zone is above
        for lvl in ret_levels:
            retracement[lvl] = swing_low + price_range * lvl
        # Extension below low (TP targets)
        for lvl in ext_levels:
            extension[lvl] = swing_high - price_range * lvl

    return {
        "retracement": retracement,
        "extension": extension,
        "range": price_range,
    }


def detect_liquidity_sweep(
    df: pd.DataFrame,
    swing_points: dict,
    lookback: int = 30,
    fresh_candles: int = 5,
) -> dict:
    """Detect liquidity sweep (stop hunt) around swing levels.

    Bullish sweep: wick BELOW a swing low, but close ABOVE it
    (swept longs' stops, then reversed up → institutional buy).
    Bearish sweep: wick ABOVE a swing high, close BELOW it.

    Args:
        swing_points: output from detect_swing_points().
        lookback: how many recent candles to check for a sweep.
        fresh_candles: a sweep within this many candles is "fresh" (±2).

    Returns:
        {"score": int, "type": str, "swept_level": float, "is_fresh": bool}
    """
    empty = {"score": 0, "type": "", "swept_level": 0.0, "is_fresh": False}
    if len(df) < 3:
        return empty

    n = len(df)
    check_start = max(0, n - lookback)

    # Check bullish sweeps (wick below swing lows)
    for idx, level in reversed(swing_points.get("swing_lows", [])):
        for i in range(max(check_start, idx + 1), n):
            row = df.iloc[i]
            if row["low"] < level and row["close"] > level:
                age = n - 1 - i
                is_fresh = age <= fresh_candles
                score = 2 if is_fresh else 1
                return {
                    "score": score,
                    "type": "bullish_sweep",
                    "swept_level": level,
                    "is_fresh": is_fresh,
                }

    # Check bearish sweeps (wick above swing highs)
    for idx, level in reversed(swing_points.get("swing_highs", [])):
        for i in range(max(check_start, idx + 1), n):
            row = df.iloc[i]
            if row["high"] > level and row["close"] < level:
                age = n - 1 - i
                is_fresh = age <= fresh_candles
                score = -2 if is_fresh else -1
                return {
                    "score": score,
                    "type": "bearish_sweep",
                    "swept_level": level,
                    "is_fresh": is_fresh,
                }

    return empty


def detect_fair_value_gap(
    df: pd.DataFrame, lookback: int = 20, proximity_pct: float = 0.5
) -> dict:
    """Detect Fair Value Gaps (FVG) — 3-candle imbalance zones.

    Bullish FVG: candle[i-2].high < candle[i].low (gap up).
    Bearish FVG: candle[i-2].low > candle[i].high (gap down).
    An FVG is "mitigated" if price later retraces into the gap.

    Args:
        lookback: how many bars back to search for FVGs.
        proximity_pct: % distance to consider "near" an FVG.

    Returns:
        {"score": int, "fvg_zones": list, "nearest": dict|None}
    """
    empty = {"score": 0, "fvg_zones": [], "nearest": None}
    if len(df) < lookback + 3:
        return empty

    close = df["close"].iloc[-1]
    recent = df.iloc[-(lookback + 3) :]
    fvg_zones: list[dict] = []

    for i in range(2, len(recent)):
        prev2 = recent.iloc[i - 2]
        curr = recent.iloc[i]

        # Bullish FVG: gap between candle[i-2] high and candle[i] low
        if prev2["high"] < curr["low"]:
            zone_low = prev2["high"]
            zone_high = curr["low"]
            # Check if mitigated: any subsequent bar's low entered the zone
            subsequent = recent.iloc[i + 1 :] if i + 1 < len(recent) else pd.DataFrame()
            mitigated = (subsequent["low"] <= zone_high).any() if len(subsequent) > 0 else False
            if not mitigated:
                fvg_zones.append({
                    "type": "bullish",
                    "zone_low": float(zone_low),
                    "zone_high": float(zone_high),
                    "age": len(recent) - i - 1,
                })

        # Bearish FVG: gap between candle[i] high and candle[i-2] low
        if prev2["low"] > curr["high"]:
            zone_low = curr["high"]
            zone_high = prev2["low"]
            subsequent = recent.iloc[i + 1 :] if i + 1 < len(recent) else pd.DataFrame()
            mitigated = (subsequent["high"] >= zone_low).any() if len(subsequent) > 0 else False
            if not mitigated:
                fvg_zones.append({
                    "type": "bearish",
                    "zone_low": float(zone_low),
                    "zone_high": float(zone_high),
                    "age": len(recent) - i - 1,
                })

    if not fvg_zones:
        return empty

    # Find nearest FVG to current price
    def distance(fvg: dict) -> float:
        mid = (fvg["zone_low"] + fvg["zone_high"]) / 2
        return abs(close - mid)

    nearest = min(fvg_zones, key=distance)
    prox = close * proximity_pct / 100

    score = 0
    if nearest["type"] == "bullish" and nearest["zone_low"] - prox <= close <= nearest["zone_high"] + prox:
        score = 1  # price at bullish FVG → buy
    elif nearest["type"] == "bearish" and nearest["zone_low"] - prox <= close <= nearest["zone_high"] + prox:
        score = -1  # price at bearish FVG → sell

    return {"score": score, "fvg_zones": fvg_zones, "nearest": nearest}


def detect_displacement(
    df: pd.DataFrame, body_pct: float = 0.3, vol_mult: float = 1.5
) -> dict:
    """Detect displacement — impulsive candle with large body + volume.

    A displacement candle has:
    - Body > body_pct % of price
    - Volume > vol_mult * 20-period average
    - Small wicks relative to body (body > 60% of full range)

    Returns:
        {"score": int, "body_pct": float, "vol_ratio": float}
    """
    empty = {"score": 0, "body_pct": 0.0, "vol_ratio": 0.0}
    if len(df) < 21:
        return empty

    c = df.iloc[-1]
    body = abs(c["close"] - c["open"])
    full_range = c["high"] - c["low"]
    price = c["close"]

    if price <= 0 or full_range <= 0:
        return empty

    body_pct_val = (body / price) * 100
    body_ratio = body / full_range

    avg_vol = df["volume"].iloc[-21:-1].mean()
    vol_ratio = c["volume"] / avg_vol if avg_vol > 0 else 0.0

    if body_pct_val >= body_pct and vol_ratio >= vol_mult and body_ratio >= 0.6:
        is_green = c["close"] > c["open"]
        score = 1 if is_green else -1
    else:
        score = 0

    return {
        "score": score,
        "body_pct": round(body_pct_val, 3),
        "vol_ratio": round(vol_ratio, 2),
    }
