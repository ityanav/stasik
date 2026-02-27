"""Dashboard mixin: market indicator API (EMA trend + Fib zones)."""

import logging

from aiohttp import web

from src.telegram_market import get_market_overview, _get_session, _analyze_pair, FIBA_PAIRS, BUBA_PAIRS

logger = logging.getLogger(__name__)


class RouteMarketMixin:
    """Provides /api/market endpoint with EMA trend + Fib zone data."""

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
            return web.json_response({"fiba": [], "buba": []})

    def _fetch_market_data(self) -> dict:
        session = _get_session()

        fiba = []
        for sym in FIBA_PAIRS:
            r = _analyze_pair(session, sym, "15")
            if not r["error"]:
                fiba.append({
                    "symbol": sym.replace("USDT", ""),
                    "price": float(r["price"]),
                    "bullish": bool(r["bullish"]),
                    "dev": float(r["dev_pct"]),
                    "in_fib": bool(r["in_fib"]),
                    "fib_dist": float(r["fib_dist"]) if r["fib_dist"] is not None else None,
                })

        buba = []
        for sym in BUBA_PAIRS:
            r = _analyze_pair(session, sym, "60")
            if not r["error"]:
                buba.append({
                    "symbol": sym.replace("USDT", ""),
                    "price": float(r["price"]),
                    "bullish": bool(r["bullish"]),
                    "dev": float(r["dev_pct"]),
                    "in_fib": bool(r["in_fib"]),
                    "fib_dist": float(r["fib_dist"]) if r["fib_dist"] is not None else None,
                })

        return {"fiba": fiba, "buba": buba}
