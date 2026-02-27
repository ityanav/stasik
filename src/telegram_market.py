"""Market indicator for Telegram bot — EMA trend + Fib zone proximity."""

import logging
from datetime import datetime, timezone, timedelta

import pandas as pd
from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)

# Bybit demo credentials (read-only market data)
API_KEY = "Ct24OvPFimR3vLSlcW"
API_SECRET = "CNzK2w9eQ41j4A4dszBLvvOkAH61Z5BQG9oq"

FIBA_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "APTUSDT"]
BUBA_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT", "APTUSDT"]


def _get_session() -> HTTP:
    return HTTP(api_key=API_KEY, api_secret=API_SECRET, demo=True)


def _fetch_klines(session: HTTP, symbol: str, interval: str, limit: int = 50) -> pd.DataFrame:
    """Fetch klines and return DataFrame with close prices."""
    resp = session.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)
    rows = resp["result"]["list"]
    if not rows:
        return pd.DataFrame()
    # rows are newest-first; reverse for chronological order
    df = pd.DataFrame(reversed(rows), columns=["time", "open", "high", "low", "close", "volume", "turnover"])
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    return df


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _find_fib_zone(df: pd.DataFrame, lookback: int = 30) -> dict:
    """Find the nearest swing high/low and calculate Fib retracement zones."""
    if len(df) < lookback:
        return {"in_zone": False, "dist_pct": None}

    recent = df.tail(lookback)
    swing_high = recent["high"].max()
    swing_low = recent["low"].min()
    price = df["close"].iloc[-1]

    rng = swing_high - swing_low
    if rng <= 0:
        return {"in_zone": False, "dist_pct": None}

    # Fibonacci levels from swing
    fib_618 = swing_high - rng * 0.618
    fib_786 = swing_high - rng * 0.786

    # Premium zone for BUY (price near swing low, retraced to 0.618-0.786)
    buy_zone_top = fib_618
    buy_zone_bot = fib_786

    # Premium zone for SELL (price near swing high, retraced to 0.618-0.786)
    sell_zone_bot = swing_low + rng * 0.618
    sell_zone_top = swing_low + rng * 0.786

    # Check if price is in any premium zone
    in_buy = buy_zone_bot <= price <= buy_zone_top
    in_sell = sell_zone_bot <= price <= sell_zone_top

    if in_buy or in_sell:
        return {"in_zone": True, "dist_pct": 0.0}

    # Distance to nearest zone edge
    dist_buy = min(abs(price - buy_zone_top), abs(price - buy_zone_bot))
    dist_sell = min(abs(price - sell_zone_top), abs(price - sell_zone_bot))
    nearest_dist = min(dist_buy, dist_sell)
    dist_pct = nearest_dist / price * 100

    return {"in_zone": False, "dist_pct": round(dist_pct, 1)}


def _analyze_pair(session: HTTP, symbol: str, interval: str) -> dict:
    """Analyze a single pair: EMA trend + Fib zone."""
    try:
        df = _fetch_klines(session, symbol, interval, limit=50)
        if len(df) < 21:
            return {"symbol": symbol, "error": True}

        price = df["close"].iloc[-1]
        ema9 = _ema(df["close"], 9).iloc[-1]
        ema21 = _ema(df["close"], 21).iloc[-1]

        # Trend direction and strength
        dev_pct = (ema9 - ema21) / ema21 * 100
        bullish = ema9 > ema21

        # Fib zone proximity
        fib = _find_fib_zone(df)

        return {
            "symbol": symbol,
            "price": price,
            "bullish": bullish,
            "dev_pct": round(dev_pct, 2),
            "in_fib": fib["in_zone"],
            "fib_dist": fib["dist_pct"],
            "error": False,
        }
    except Exception as e:
        logger.warning("Failed to analyze %s %s: %s", symbol, interval, e)
        return {"symbol": symbol, "error": True}


def _format_pair(r: dict) -> str:
    """Format single pair result: arrow + symbol + dev% + fib status."""
    if r["error"]:
        return f"  ❓ {r['symbol'][:3]}"

    label = r["symbol"].replace("USDT", "")
    arrow = "↗️" if r["bullish"] else "↘️"
    sign = "+" if r["dev_pct"] >= 0 else ""
    dev = f"{sign}{r['dev_pct']}%"

    if r["in_fib"]:
        fib = "🎯 FIB"
    elif r["fib_dist"] is not None:
        fib = f"~{r['fib_dist']}%"
    else:
        fib = ""

    return f"  {arrow} {label:<5} {dev:>7}  {fib}"


def get_market_overview() -> str:
    """Generate market overview text for Telegram."""
    session = _get_session()
    now_msk = datetime.now(timezone(timedelta(hours=3)))

    lines = [f"📊 РЫНОК  {now_msk:%H:%M} МСК", ""]

    # FIBA: 15m entry, 1H structure
    lines.append("── FIBA (15м) ──")
    for sym in FIBA_PAIRS:
        r = _analyze_pair(session, sym, "15")
        lines.append(_format_pair(r))

    lines.append("")

    # BUBA: 1H entry, 4H structure
    lines.append("── BUBA (1H) ──")
    for sym in BUBA_PAIRS:
        r = _analyze_pair(session, sym, "60")
        lines.append(_format_pair(r))

    lines.append("")
    lines.append("↗️/↘️ = EMA9/21 тренд")
    lines.append("% = сила тренда")
    lines.append("🎯 FIB = цена в Fib зоне")
    lines.append("~X% = расст. до Fib зоны")

    return "\n".join(lines)
