import logging
from typing import Optional

import pandas as pd
from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)


class BybitClient:
    def __init__(self, config: dict):
        self.config = config
        bybit_cfg = config["bybit"]
        self.testnet = bybit_cfg["testnet"]
        self.session = HTTP(
            testnet=self.testnet,
            api_key=bybit_cfg["api_key"],
            api_secret=bybit_cfg["api_secret"],
        )
        self.leverage = config["trading"].get("leverage", 1)
        logger.info(
            "BybitClient initialized (testnet=%s)", self.testnet
        )

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

    # ── Ticker (last price) ──────────────────────────────────

    def get_last_price(self, symbol: str, category: str = "linear") -> float:
        resp = self.session.get_tickers(category=category, symbol=symbol)
        return float(resp["result"]["list"][0]["lastPrice"])

    # ── Instrument info (min qty, tick size) ─────────────────

    def get_instrument_info(self, symbol: str, category: str = "linear") -> dict:
        resp = self.session.get_instruments_info(category=category, symbol=symbol)
        info = resp["result"]["list"][0]
        lot_filter = info["lotSizeFilter"]
        price_filter = info["priceFilter"]
        return {
            "min_qty": float(lot_filter["minOrderQty"]),
            "max_qty": float(lot_filter["maxOrderQty"]),
            "qty_step": float(lot_filter["qtyStep"]),
            "tick_size": float(price_filter["tickSize"]),
        }
