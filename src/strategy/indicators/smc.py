import numpy as np
import pandas as pd
import ta


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


def detect_swing_points_zigzag(
    df: pd.DataFrame, atr_period: int = 14, atr_mult: float = 2.0
) -> dict:
    """ATR-based ZigZag swing detection — adaptive to volatility.

    A swing is confirmed when price reverses by atr_mult * ATR from the
    current extreme. More robust than fractal-based detection in trending markets.

    Returns same format as detect_swing_points() for drop-in replacement.
    """
    n = len(df)
    if n < atr_period + 5:
        return {"swing_highs": [], "swing_lows": [], "last_swing_high": None, "last_swing_low": None}

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    # Calculate ATR series
    atr_ind = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=atr_period
    )
    atr_series = atr_ind.average_true_range().values

    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []

    # State: 1 = looking for high (trending up), -1 = looking for low (trending down)
    state = 0
    extreme_idx = 0
    extreme_val = closes[atr_period]

    for i in range(atr_period, n):
        atr_val = atr_series[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue
        threshold = atr_val * atr_mult

        if state == 0:
            # Initialize direction
            if highs[i] > extreme_val:
                state = 1
                extreme_idx = i
                extreme_val = highs[i]
            elif lows[i] < extreme_val:
                state = -1
                extreme_idx = i
                extreme_val = lows[i]
            continue

        if state == 1:
            # Trending up — track highest high
            if highs[i] > extreme_val:
                extreme_idx = i
                extreme_val = highs[i]
            elif extreme_val - lows[i] >= threshold:
                # Reversal down — confirm swing high
                swing_highs.append((extreme_idx, float(extreme_val)))
                state = -1
                extreme_idx = i
                extreme_val = lows[i]
        else:
            # Trending down — track lowest low
            if lows[i] < extreme_val:
                extreme_idx = i
                extreme_val = lows[i]
            elif highs[i] - extreme_val >= threshold:
                # Reversal up — confirm swing low
                swing_lows.append((extreme_idx, float(extreme_val)))
                state = 1
                extreme_idx = i
                extreme_val = highs[i]

    return {
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
        "last_swing_high": swing_highs[-1] if swing_highs else None,
        "last_swing_low": swing_lows[-1] if swing_lows else None,
    }


def find_fibonacci_clusters(
    swing_highs: list[tuple[int, float]],
    swing_lows: list[tuple[int, float]],
    threshold_pct: float = 1.0,
) -> list[dict]:
    """Find Fibonacci confluence zones from multiple swing pairs.

    Calculates Fib retracement levels from up to 4 most recent swing
    high/low pairs. Groups levels that are within threshold_pct of each
    other. Zones with 3+ overlapping levels are strong confluence.

    Returns:
        List of cluster dicts: [{"price": mid_price, "count": N, "levels": [...]}]
        sorted by count descending.
    """
    if len(swing_highs) < 1 or len(swing_lows) < 1:
        return []

    # Take up to 4 most recent swing highs and lows
    recent_highs = [p for _, p in swing_highs[-4:]]
    recent_lows = [p for _, p in swing_lows[-4:]]

    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.705, 0.786]
    all_levels: list[float] = []

    # Generate fib levels from all pairs
    for sh in recent_highs:
        for sl in recent_lows:
            if sh <= sl:
                continue
            price_range = sh - sl
            for ratio in fib_ratios:
                # Bullish retracement (from high down)
                all_levels.append(sh - price_range * ratio)
                # Bearish retracement (from low up)
                all_levels.append(sl + price_range * ratio)

    if not all_levels:
        return []

    # Sort and cluster levels within threshold_pct
    all_levels.sort()
    clusters: list[dict] = []
    used = [False] * len(all_levels)

    for i in range(len(all_levels)):
        if used[i]:
            continue
        cluster_levels = [all_levels[i]]
        used[i] = True
        for j in range(i + 1, len(all_levels)):
            if used[j]:
                continue
            mid = (all_levels[i] + all_levels[j]) / 2
            if mid > 0 and abs(all_levels[j] - all_levels[i]) / mid * 100 <= threshold_pct:
                cluster_levels.append(all_levels[j])
                used[j] = True

        if len(cluster_levels) >= 2:
            clusters.append({
                "price": sum(cluster_levels) / len(cluster_levels),
                "count": len(cluster_levels),
                "levels": cluster_levels,
            })

    clusters.sort(key=lambda c: c["count"], reverse=True)
    return clusters


def calculate_fib_pivots(daily_df: pd.DataFrame) -> dict:
    """Calculate Fibonacci Pivot Points from daily OHLC data.

    P = (H + L + C) / 3
    R1 = P + 0.382 * (H - L),  R2 = P + 0.618 * (H - L),  R3 = P + 1.0 * (H - L)
    S1 = P - 0.382 * (H - L),  S2 = P - 0.618 * (H - L),  S3 = P - 1.0 * (H - L)

    Uses the last completed daily candle (iloc[-2] if available, else iloc[-1]).

    Returns:
        {"pivot": P, "r1": R1, "r2": R2, "r3": R3, "s1": S1, "s2": S2, "s3": S3}
    """
    empty = {"pivot": 0, "r1": 0, "r2": 0, "r3": 0, "s1": 0, "s2": 0, "s3": 0}
    if len(daily_df) < 2:
        return empty

    # Use last completed bar (not current partial)
    bar = daily_df.iloc[-2]
    h = bar["high"]
    l = bar["low"]
    c = bar["close"]

    if h <= l or h == 0:
        return empty

    p = (h + l + c) / 3
    r = h - l  # daily range

    return {
        "pivot": round(p, 6),
        "r1": round(p + 0.382 * r, 6),
        "r2": round(p + 0.618 * r, 6),
        "r3": round(p + 1.0 * r, 6),
        "s1": round(p - 0.382 * r, 6),
        "s2": round(p - 0.618 * r, 6),
        "s3": round(p - 1.0 * r, 6),
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

    ret_levels = [0.236, 0.382, 0.5, 0.618, 0.705, 0.786]
    ext_levels = [1.0, 1.272, 1.618, 2.0]

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
