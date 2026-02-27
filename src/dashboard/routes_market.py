"""Dashboard mixin: HTF trend API per bot."""

import logging

from aiohttp import web
from pybit.unified_trading import HTTP
import pandas as pd

logger = logging.getLogger(__name__)

# Bybit demo credentials (read-only market data)
_API_KEY = "Ct24OvPFimR3vLSlcW"
_API_SECRET = "CNzK2w9eQ41j4A4dszBLvvOkAH61Z5BQG9oq"

# Bot definitions: name, htf_interval, ema_fast, ema_slow, pairs
BOTS = [
    {
        "name": "FIBA",
        "htf": "60",
        "htf_label": "1H",
        "ema_fast": 9,
        "ema_slow": 21,
        "pairs": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "APTUSDT"],
    },
    {
        "name": "BUBA",
        "htf": "240",
        "htf_label": "4H",
        "ema_fast": 9,
        "ema_slow": 21,
        "pairs": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT", "APTUSDT"],
    },
]


def _get_trend(session: HTTP, symbol: str, interval: str, ema_fast: int, ema_slow: int) -> dict:
    """Fetch HTF klines and compute EMA trend."""
    try:
        resp = session.get_kline(category="linear", symbol=symbol, interval=interval, limit=60)
        rows = resp["result"]["list"]
        if not rows or len(rows) < ema_slow + 2:
            return {"symbol": symbol.replace("USDT", ""), "trend": "NEUTRAL", "error": True}

        df = pd.DataFrame(reversed(rows), columns=["time", "open", "high", "low", "close", "volume", "turnover"])
        close = df["close"].astype(float)
        price = close.iloc[-1]

        ef = close.ewm(span=ema_fast, adjust=False).mean().iloc[-1]
        es = close.ewm(span=ema_slow, adjust=False).mean().iloc[-1]

        if ef > es and price > ef:
            trend = "BULLISH"
        elif ef < es and price < ef:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        dev = (ef - es) / es * 100

        return {
            "symbol": symbol.replace("USDT", ""),
            "trend": trend,
            "dev": round(dev, 2),
            "price": round(price, 2),
            "error": False,
        }
    except Exception as e:
        logger.warning("Trend fetch failed %s %s: %s", symbol, interval, e)
        return {"symbol": symbol.replace("USDT", ""), "trend": "NEUTRAL", "error": True}


class RouteMarketMixin:
    """Provides /api/market endpoint with HTF trend data per bot."""

    async def _api_market(self, request: web.Request) -> web.Response:
        cached = self._cache_get("market_data")
        if cached is not None:
            return web.json_response(cached)

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, self._fetch_market_data)
            self._cache_set("market_data", data, 30.0)
            return web.json_response(data)
        except Exception:
            logger.exception("Failed to fetch market data")
            return web.json_response({"bots": []})

    def _fetch_market_data(self) -> dict:
        session = HTTP(api_key=_API_KEY, api_secret=_API_SECRET, demo=True)

        bots = []
        for bot in BOTS:
            pairs = []
            for sym in bot["pairs"]:
                t = _get_trend(session, sym, bot["htf"], bot["ema_fast"], bot["ema_slow"])
                pairs.append(t)
            bots.append({
                "name": bot["name"],
                "htf": bot["htf_label"],
                "ema": f"{bot['ema_fast']}/{bot['ema_slow']}",
                "pairs": pairs,
            })

        return {"bots": bots}
