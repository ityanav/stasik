import math

import numpy as np
import pandas as pd

from .classic import calculate_rsi


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


# ── Volume Profile ──────────────────────────────────────────────────────
def calculate_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 50,
    value_area_pct: float = 70.0,
    proximity_pct: float = 0.3,
) -> dict:
    """Volume Profile: POC, VAH, VAL from price-volume distribution.

    Distributes each candle's volume across price bins proportional to its
    high-low range, then finds the Point of Control (max-volume bin) and
    Value Area (70% of total volume centered on POC).

    Score: +1 near VAL (support), -1 near VAH (resistance), ±1 near POC
    (direction based on last candle colour).
    """
    empty: dict = {"score": 0, "poc": 0.0, "vah": 0.0, "val": 0.0}
    if len(df) < 20:
        return empty

    high_max = df["high"].max()
    low_min = df["low"].min()
    if high_max <= low_min:
        return empty

    bin_size = (high_max - low_min) / num_bins
    if bin_size <= 0:
        return empty

    bins = [0.0] * num_bins
    for _, row in df.iterrows():
        h, l, v = row["high"], row["low"], row["volume"]
        if h <= l or v <= 0:
            continue
        lo_bin = max(0, int((l - low_min) / bin_size))
        hi_bin = min(num_bins - 1, int((h - low_min) / bin_size))
        span = hi_bin - lo_bin + 1
        vol_per_bin = v / span
        for b in range(lo_bin, hi_bin + 1):
            bins[b] += vol_per_bin

    poc_idx = max(range(num_bins), key=lambda i: bins[i])
    poc_price = low_min + (poc_idx + 0.5) * bin_size

    total_vol = sum(bins)
    if total_vol <= 0:
        return empty

    target = total_vol * value_area_pct / 100.0
    area_vol = bins[poc_idx]
    lo_i, hi_i = poc_idx, poc_idx
    while area_vol < target and (lo_i > 0 or hi_i < num_bins - 1):
        add_lo = bins[lo_i - 1] if lo_i > 0 else 0.0
        add_hi = bins[hi_i + 1] if hi_i < num_bins - 1 else 0.0
        if add_lo >= add_hi and lo_i > 0:
            lo_i -= 1
            area_vol += add_lo
        elif hi_i < num_bins - 1:
            hi_i += 1
            area_vol += add_hi
        else:
            lo_i -= 1
            area_vol += add_lo

    val_price = low_min + (lo_i + 0.5) * bin_size
    vah_price = low_min + (hi_i + 0.5) * bin_size

    close = df["close"].iloc[-1]
    prox = close * proximity_pct / 100.0
    score = 0
    if abs(close - val_price) <= prox:
        score = 1  # near support
    elif abs(close - vah_price) <= prox:
        score = -1  # near resistance
    elif abs(close - poc_price) <= prox:
        score = 1 if df["close"].iloc[-1] > df["open"].iloc[-1] else -1

    return {
        "score": score,
        "poc": round(poc_price, 6),
        "vah": round(vah_price, 6),
        "val": round(val_price, 6),
    }


# ── Cumulative Delta ────────────────────────────────────────────────────
def calculate_cumulative_delta(
    df: pd.DataFrame,
    lookback: int = 50,
    divergence_lookback: int = 20,
) -> dict:
    """Cumulative Delta: buy/sell pressure from OHLCV candles.

    Without tick data, approximates buy volume as vol*(close-low)/(high-low).
    Delta = buy_vol - sell_vol. Cumulative sum over lookback bars.
    Divergence over divergence_lookback: price LL + delta HL = bullish (+1),
    price HH + delta LH = bearish (-1).
    """
    empty: dict = {"score": 0, "current_delta": 0.0, "divergence_type": "none"}
    if len(df) < lookback:
        return empty

    window = df.iloc[-lookback:].copy()
    ranges = window["high"] - window["low"]
    ranges = ranges.replace(0, float("nan"))
    buy_pct = (window["close"] - window["low"]) / ranges
    buy_pct = buy_pct.fillna(0.5)
    buy_vol = window["volume"] * buy_pct
    sell_vol = window["volume"] * (1 - buy_pct)
    delta = buy_vol - sell_vol
    cum_delta = delta.cumsum()
    current_delta = float(cum_delta.iloc[-1])

    score = 0
    div_type = "none"
    n = min(divergence_lookback, len(cum_delta))
    if n >= 5:
        recent_close = window["close"].iloc[-n:]
        recent_cd = cum_delta.iloc[-n:]
        price_min = recent_close.min()
        price_max = recent_close.max()
        cd_min = recent_cd.min()
        cd_max = recent_cd.max()
        last_price = recent_close.iloc[-1]
        last_cd = recent_cd.iloc[-1]

        # Bullish divergence: price makes lower low, delta makes higher low
        if last_price <= price_min * 1.001 and last_cd > cd_min * 1.05:
            score = 1
            div_type = "bullish"
        # Bearish divergence: price makes higher high, delta makes lower high
        elif last_price >= price_max * 0.999 and last_cd < cd_max * 0.95:
            score = -1
            div_type = "bearish"

    return {
        "score": score,
        "current_delta": round(current_delta, 2),
        "divergence_type": div_type,
    }


# ── Murray Math Lines ───────────────────────────────────────────────────
def calculate_murray_math_lines(
    df: pd.DataFrame,
    period: int = 64,
    proximity_pct: float = 0.3,
) -> dict:
    """Murray Math Lines: harmonic price levels based on Gann theory.

    HH/LL over period bars → octave = nearest power of 2 that covers the range,
    base = snapped to grid. 9 levels (0/8 .. 8/8), step = octave/8.
    Score: +1 near 0/8 or 2/8 (support), -1 near 8/8 or 6/8 (resistance),
    ±1 near 4/8 (pivot — direction from last candle).
    """
    empty: dict = {"score": 0, "levels": {}, "nearest_level": "", "nearest_price": 0.0}
    if len(df) < period:
        return empty

    window = df.iloc[-period:]
    hh = float(window["high"].max())
    ll = float(window["low"].min())
    price_range = hh - ll
    if price_range <= 0:
        return empty

    # Find nearest power of 2 >= price_range
    octave = 2 ** math.ceil(math.log2(price_range)) if price_range > 0 else 1.0

    # Snap base to grid
    step = octave / 8.0
    base = math.floor(ll / step) * step

    levels = {}
    level_names = ["0/8", "1/8", "2/8", "3/8", "4/8", "5/8", "6/8", "7/8", "8/8"]
    for i, name in enumerate(level_names):
        levels[name] = round(base + i * step, 6)

    close = float(df["close"].iloc[-1])
    prox = close * proximity_pct / 100.0

    # Find nearest level
    nearest_name = ""
    nearest_price = 0.0
    min_dist = float("inf")
    for name, lvl in levels.items():
        dist = abs(close - lvl)
        if dist < min_dist:
            min_dist = dist
            nearest_name = name
            nearest_price = lvl

    score = 0
    support_levels = {"0/8", "2/8"}
    resistance_levels = {"8/8", "6/8"}
    pivot_level = "4/8"

    if min_dist <= prox:
        if nearest_name in support_levels:
            score = 1
        elif nearest_name in resistance_levels:
            score = -1
        elif nearest_name == pivot_level:
            score = 1 if close > df["open"].iloc[-1] else -1

    return {
        "score": score,
        "levels": levels,
        "nearest_level": nearest_name,
        "nearest_price": nearest_price,
    }
