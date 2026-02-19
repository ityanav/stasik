import logging
from typing import Optional

import pandas as pd
from pybit.unified_trading import HTTP

from src.exchange.base import ExchangeClient

logger = logging.getLogger(__name__)


class BybitClient(ExchangeClient):
    def __init__(self, config: dict):
        self.config = config
        bybit_cfg = config["bybit"]
        self.testnet = bybit_cfg.get("testnet", False)
        self.demo = bybit_cfg.get("demo", False)

        http_kwargs: dict = {
            "api_key": bybit_cfg["api_key"],
            "api_secret": bybit_cfg["api_secret"],
        }
        if self.demo:
            http_kwargs["demo"] = True
        else:
            http_kwargs["testnet"] = self.testnet

        self.session = HTTP(**http_kwargs)
        self.leverage = config["trading"].get("leverage", 1)
        mode = "demo" if self.demo else f"testnet={self.testnet}"
        logger.info("BybitClient initialized (%s)", mode)

    # ── Balance ──────────────────────────────────────────────

    def get_balance(self, coin: str = "USDT", account_type: str = "UNIFIED") -> float:
        resp = self.session.get_wallet_balance(
            accountType=account_type, coin=coin
        )
        for acct in resp["result"]["list"]:
            for c in acct["coin"]:
                if c["coin"] == coin:
                    return float(c["walletBalance"])
        return 0.0

    # ── Klines ───────────────────────────────────────────────

    def get_klines(
        self,
        symbol: str,
        interval: str = "5",
        limit: int = 200,
        category: str = "linear",
    ) -> pd.DataFrame:
        resp = self.session.get_kline(
            category=category,
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
        rows = resp["result"]["list"]
        df = pd.DataFrame(
            rows,
            columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
        )
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    # ── Orders ───────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        category: str = "linear",
        order_type: str = "Market",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> dict:
        params: dict = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
        }
        if stop_loss is not None:
            params["stopLoss"] = str(round(stop_loss, 6))
        if take_profit is not None:
            params["takeProfit"] = str(round(take_profit, 6))

        resp = self.session.place_order(**params)
        logger.info(
            "Order placed: %s %s %s qty=%s sl=%s tp=%s -> %s",
            category, side, symbol, qty, stop_loss, take_profit,
            resp["result"].get("orderId"),
        )
        return resp["result"]

    def cancel_order(
        self, symbol: str, order_id: str, category: str = "linear"
    ) -> dict:
        resp = self.session.cancel_order(
            category=category, symbol=symbol, orderId=order_id
        )
        logger.info("Order cancelled: %s %s", symbol, order_id)
        return resp["result"]

    # ── Positions ────────────────────────────────────────────

    def get_positions(
        self, symbol: Optional[str] = None, category: str = "linear"
    ) -> list[dict]:
        params: dict = {"category": category, "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = symbol
        resp = self.session.get_positions(**params)
        positions = []
        for p in resp["result"]["list"]:
            size = float(p["size"])
            if size > 0:
                positions.append(
                    {
                        "symbol": p["symbol"],
                        "side": p["side"],
                        "size": size,
                        "entry_price": float(p["avgPrice"]),
                        "unrealised_pnl": float(p["unrealisedPnl"]),
                        "leverage": p["leverage"],
                    }
                )
        return positions

    def set_leverage(self, symbol: str, leverage: int, category: str = "linear"):
        try:
            self.session.set_leverage(
                category=category,
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            logger.info("Leverage set: %s -> %sx", symbol, leverage)
        except Exception as e:
            if "leverage not modified" in str(e).lower():
                pass
            else:
                logger.warning("Failed to set leverage for %s: %s", symbol, e)

    # ── Trailing stop ─────────────────────────────────────────

    def set_trailing_stop(
        self,
        symbol: str,
        trailing_stop: float,
        active_price: float,
        category: str = "linear",
    ):
        try:
            self.session.set_trading_stop(
                category=category,
                symbol=symbol,
                trailingStop=str(round(trailing_stop, 6)),
                activePrice=str(round(active_price, 6)),
                positionIdx=0,
            )
            logger.info(
                "Trailing stop set: %s trail=%s activePrice=%s",
                symbol, trailing_stop, active_price,
            )
        except Exception as e:
            logger.warning("Failed to set trailing stop for %s: %s", symbol, e)

    # ── Closed PnL ──────────────────────────────────────────

    def get_closed_pnl(self, symbol: str, category: str = "linear", limit: int = 10) -> list[dict]:
        """Get recent closed PnL records from exchange."""
        resp = self.session.get_closed_pnl(
            category=category, symbol=symbol, limit=limit
        )
        results = []
        for r in resp["result"]["list"]:
            results.append({
                "order_id": r.get("orderId", ""),
                "symbol": r["symbol"],
                "side": r["side"],
                "qty": float(r["qty"]),
                "entry_price": float(r["avgEntryPrice"]),
                "exit_price": float(r["avgExitPrice"]),
                "pnl": float(r["closedPnl"]),
                "closed_at": r.get("updatedTime", ""),
            })
        return results

    # ── Ticker (last price) ──────────────────────────────────

    def get_last_price(self, symbol: str, category: str = "linear") -> float:
        resp = self.session.get_tickers(category=category, symbol=symbol)
        return float(resp["result"]["list"][0]["lastPrice"])

    # ── Funding rate ──────────────────────────────────────────

    def get_funding_rate(self, symbol: str, category: str = "linear") -> float:
        """Get current funding rate from tickers API. Returns 0.0 on error."""
        try:
            resp = self.session.get_tickers(category=category, symbol=symbol)
            rate_str = resp["result"]["list"][0].get("fundingRate", "0")
            return float(rate_str)
        except Exception as e:
            logger.warning("Failed to get funding rate for %s: %s", symbol, e)
            return 0.0

    # ── Instrument info (min qty, tick size) ─────────────────

    def get_instrument_info(self, symbol: str, category: str = "linear") -> dict:
        resp = self.session.get_instruments_info(category=category, symbol=symbol)
        info = resp["result"]["list"][0]
        lot_filter = info["lotSizeFilter"]
        price_filter = info["priceFilter"]
        # Spot uses "basePrecision" instead of "qtyStep"
        qty_step = lot_filter.get("qtyStep") or lot_filter.get("basePrecision", "1")
        return {
            "min_qty": float(lot_filter["minOrderQty"]),
            "max_qty": float(lot_filter["maxOrderQty"]),
            "qty_step": float(qty_step),
            "tick_size": float(price_filter["tickSize"]),
        }
