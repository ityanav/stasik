import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from t_tech.invest import Client
from t_tech.invest.constants import INVEST_GRPC_API, INVEST_GRPC_API_SANDBOX
from t_tech.invest.services import (
    CandleInterval,
    InstrumentIdType,
    InstrumentStatus,
    MoneyValue,
    OrderDirection,
    OrderType,
    Quotation,
    StopOrderDirection,
    StopOrderExpirationType,
    StopOrderType,
)
from t_tech.invest.utils import money_to_decimal, quotation_to_decimal

from src.exchange.base import ExchangeClient

logger = logging.getLogger(__name__)


@dataclass
class InstrumentInfo:
    figi: str
    uid: str
    ticker: str
    lot: int
    min_price_increment: float
    currency: str
    instrument_type: str  # share, future, currency


# Map Stasik timeframe strings to T-Invest CandleInterval
_TF_MAP = {
    "1": CandleInterval.CANDLE_INTERVAL_1_MIN,
    "2": CandleInterval.CANDLE_INTERVAL_2_MIN,
    "3": CandleInterval.CANDLE_INTERVAL_3_MIN,
    "5": CandleInterval.CANDLE_INTERVAL_5_MIN,
    "10": CandleInterval.CANDLE_INTERVAL_10_MIN,
    "15": CandleInterval.CANDLE_INTERVAL_15_MIN,
    "30": CandleInterval.CANDLE_INTERVAL_30_MIN,
    "60": CandleInterval.CANDLE_INTERVAL_HOUR,
    "120": CandleInterval.CANDLE_INTERVAL_2_HOUR,
    "240": CandleInterval.CANDLE_INTERVAL_4_HOUR,
    "D": CandleInterval.CANDLE_INTERVAL_DAY,
    "W": CandleInterval.CANDLE_INTERVAL_WEEK,
    "M": CandleInterval.CANDLE_INTERVAL_MONTH,
}

# Max candle request range per interval
_MAX_RANGE_DAYS = {
    "1": 1, "2": 1, "3": 1, "5": 1, "10": 1,
    "15": 1, "30": 2, "60": 7, "120": 7, "240": 7,
    "D": 365, "W": 730, "M": 3650,
}


def _quotation_to_float(q: Quotation | MoneyValue | None) -> float:
    if q is None:
        return 0.0
    return float(quotation_to_decimal(q)) if isinstance(q, Quotation) else float(money_to_decimal(q))


def _float_to_quotation(value: float) -> Quotation:
    units = int(value)
    nano = int(round((value - units) * 1_000_000_000))
    return Quotation(units=units, nano=nano)


class TBankClient(ExchangeClient):
    def __init__(self, config: dict):
        self.config = config
        tbank_cfg = config["tbank"]
        self.token = tbank_cfg["token"]
        self.sandbox = tbank_cfg.get("sandbox", True)
        self.account_id = tbank_cfg.get("account_id", "")
        self.commission_rate = tbank_cfg.get("commission_rate", 0.0004)
        self.instrument_type = config["trading"].get("instrument_type", "share")
        self._target = INVEST_GRPC_API_SANDBOX if self.sandbox else INVEST_GRPC_API
        self._instrument_cache: dict[str, InstrumentInfo] = {}

        self._init_and_load(config["trading"]["pairs"])

        mode = "sandbox" if self.sandbox else "production"
        logger.info("TBankClient initialized (%s), account=%s", mode, self.account_id)

    def _new_client(self) -> Client:
        """Create a fresh gRPC client (each `with` block opens/closes channel)."""
        return Client(self.token, target=self._target)

    def _init_and_load(self, tickers: list[str]):
        """Initialize account and load instruments in a single connection."""
        with self._new_client() as client:
            # ── Init account ──
            if self.sandbox:
                if not self.account_id:
                    # Try to reuse existing sandbox account
                    try:
                        accounts = client.sandbox.get_sandbox_accounts()
                        for acc in accounts.accounts:
                            if acc.status.name == "ACCOUNT_STATUS_OPEN":
                                self.account_id = acc.id
                                logger.info("Reusing sandbox account: %s", self.account_id)
                                break
                    except Exception:
                        pass
                if not self.account_id:
                    resp = client.sandbox.open_sandbox_account()
                    self.account_id = resp.account_id
                    client.sandbox.sandbox_pay_in(
                        account_id=self.account_id,
                        amount=MoneyValue(currency="rub", units=1_000_000, nano=0),
                    )
                    logger.info("Created sandbox account: %s (1M RUB)", self.account_id)
            else:
                if not self.account_id:
                    accounts = client.users.get_accounts()
                    for acc in accounts.accounts:
                        if acc.status.name == "ACCOUNT_STATUS_OPEN":
                            self.account_id = acc.id
                            break
                    if not self.account_id:
                        raise RuntimeError("No open T-Invest account found")
                    logger.info("Using account: %s", self.account_id)

            # ── Load instruments ──
            all_instruments = []
            try:
                if self.instrument_type == "share":
                    resp = client.instruments.shares(instrument_status=1)
                    all_instruments = resp.instruments
                elif self.instrument_type == "future":
                    resp = client.instruments.futures(instrument_status=1)
                    all_instruments = resp.instruments
                elif self.instrument_type == "currency":
                    resp = client.instruments.currencies(instrument_status=1)
                    all_instruments = resp.instruments
                else:
                    resp = client.instruments.shares(instrument_status=1)
                    all_instruments = resp.instruments
            except Exception as e:
                logger.error("Failed to load instrument list: %s", e)
                return

            ticker_map = {}
            for inst in all_instruments:
                ticker_map[inst.ticker] = inst

            for ticker in tickers:
                inst = ticker_map.get(ticker)
                if inst:
                    self._instrument_cache[ticker] = InstrumentInfo(
                        figi=inst.figi,
                        uid=inst.uid,
                        ticker=inst.ticker,
                        lot=inst.lot,
                        min_price_increment=_quotation_to_float(inst.min_price_increment),
                        currency=inst.currency,
                        instrument_type=self.instrument_type,
                    )
                    logger.info(
                        "Mapped %s -> FIGI=%s, lot=%d, min_price=%.4f",
                        ticker, inst.figi, inst.lot,
                        self._instrument_cache[ticker].min_price_increment,
                    )
                else:
                    logger.warning("Instrument not found: %s (type=%s)", ticker, self.instrument_type)

    def _get_figi(self, symbol: str) -> str:
        info = self._instrument_cache.get(symbol)
        if not info:
            raise ValueError(f"Unknown instrument: {symbol}")
        return info.figi

    def _get_uid(self, symbol: str) -> str:
        info = self._instrument_cache.get(symbol)
        if not info:
            raise ValueError(f"Unknown instrument: {symbol}")
        return info.uid

    # ── Balance ──────────────────────────────────────────────

    def get_balance(self, currency: str = "RUB") -> float:
        with self._new_client() as client:
            if self.sandbox:
                portfolio = client.sandbox.get_sandbox_portfolio(account_id=self.account_id)
            else:
                portfolio = client.operations.get_portfolio(account_id=self.account_id)

            total = _quotation_to_float(portfolio.total_amount_portfolio)
            return total

    # ── Klines ───────────────────────────────────────────────

    def get_klines(
        self, symbol: str, interval: str = "5", limit: int = 200, **kwargs
    ) -> pd.DataFrame:
        figi = self._get_figi(symbol)
        candle_interval = _TF_MAP.get(interval)
        if candle_interval is None:
            raise ValueError(f"Unsupported interval: {interval}")

        max_days = _MAX_RANGE_DAYS.get(interval, 1)
        now = datetime.now(timezone.utc)
        from_dt = now - timedelta(days=max_days)

        with self._new_client() as client:
            resp = client.market_data.get_candles(
                instrument_id=figi,
                from_=from_dt,
                to=now,
                interval=candle_interval,
            )

        rows = []
        for c in resp.candles:
            if not c.is_complete and interval not in ("D", "W", "M"):
                continue
            rows.append({
                "timestamp": c.time,
                "open": _quotation_to_float(c.open),
                "high": _quotation_to_float(c.high),
                "low": _quotation_to_float(c.low),
                "close": _quotation_to_float(c.close),
                "volume": float(c.volume),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
        return df

    # ── Orders ───────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "Market",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        **kwargs,
    ) -> dict:
        figi = self._get_figi(symbol)
        inst = self._instrument_cache[symbol]

        # Convert qty to lots
        lots = max(1, int(qty / inst.lot)) if inst.lot > 0 else int(qty)

        direction = (
            OrderDirection.ORDER_DIRECTION_BUY
            if side == "Buy"
            else OrderDirection.ORDER_DIRECTION_SELL
        )
        otype = OrderType.ORDER_TYPE_MARKET

        with self._new_client() as client:
            if self.sandbox:
                resp = client.sandbox.post_sandbox_order(
                    figi=figi,
                    quantity=lots,
                    direction=direction,
                    account_id=self.account_id,
                    order_type=otype,
                )
            else:
                resp = client.orders.post_order(
                    figi=figi,
                    quantity=lots,
                    direction=direction,
                    account_id=self.account_id,
                    order_type=otype,
                )

            order_id = resp.order_id
            logger.info(
                "Order placed: %s %s %s lots=%d -> %s",
                side, symbol, order_type, lots, order_id,
            )

            # Set stop-loss as stop order
            if stop_loss is not None:
                try:
                    sl_direction = (
                        StopOrderDirection.STOP_ORDER_DIRECTION_SELL
                        if side == "Buy"
                        else StopOrderDirection.STOP_ORDER_DIRECTION_BUY
                    )
                    if self.sandbox:
                        # Sandbox doesn't support stop orders, skip
                        logger.info("Sandbox: skipping stop-loss for %s", symbol)
                    else:
                        client.stop_orders.post_stop_order(
                            figi=figi,
                            quantity=lots,
                            stop_price=_float_to_quotation(stop_loss),
                            direction=sl_direction,
                            account_id=self.account_id,
                            stop_order_type=StopOrderType.STOP_ORDER_TYPE_STOP_LOSS,
                            expiration_type=StopOrderExpirationType.STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL,
                        )
                except Exception as e:
                    logger.warning("Failed to set SL for %s: %s", symbol, e)

            # Set take-profit as stop order
            if take_profit is not None:
                try:
                    tp_direction = (
                        StopOrderDirection.STOP_ORDER_DIRECTION_SELL
                        if side == "Buy"
                        else StopOrderDirection.STOP_ORDER_DIRECTION_BUY
                    )
                    if self.sandbox:
                        logger.info("Sandbox: skipping take-profit for %s", symbol)
                    else:
                        client.stop_orders.post_stop_order(
                            figi=figi,
                            quantity=lots,
                            stop_price=_float_to_quotation(take_profit),
                            direction=tp_direction,
                            account_id=self.account_id,
                            stop_order_type=StopOrderType.STOP_ORDER_TYPE_TAKE_PROFIT,
                            expiration_type=StopOrderExpirationType.STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL,
                        )
                except Exception as e:
                    logger.warning("Failed to set TP for %s: %s", symbol, e)

        return {"orderId": order_id}

    def cancel_order(self, symbol: str, order_id: str) -> dict:
        with self._new_client() as client:
            if self.sandbox:
                client.sandbox.cancel_sandbox_order(
                    account_id=self.account_id, order_id=order_id
                )
            else:
                client.orders.cancel_order(
                    account_id=self.account_id, order_id=order_id
                )
            logger.info("Order cancelled: %s %s", symbol, order_id)
        return {"orderId": order_id}

    # ── Positions ────────────────────────────────────────────

    def get_positions(self, symbol: Optional[str] = None, **kwargs) -> list[dict]:
        with self._new_client() as client:
            if self.sandbox:
                portfolio = client.sandbox.get_sandbox_portfolio(account_id=self.account_id)
            else:
                portfolio = client.operations.get_portfolio(account_id=self.account_id)

        positions = []
        for pos in portfolio.positions:
            qty = _quotation_to_float(pos.quantity)
            if qty == 0:
                continue

            # Find ticker by FIGI
            ticker = pos.figi
            for t, info in self._instrument_cache.items():
                if info.figi == pos.figi:
                    ticker = t
                    break

            if symbol and ticker != symbol:
                continue

            avg_price = _quotation_to_float(pos.average_position_price)
            current_price = _quotation_to_float(pos.current_price)
            expected_yield = _quotation_to_float(pos.expected_yield)

            side = "Buy" if qty > 0 else "Sell"
            positions.append({
                "symbol": ticker,
                "side": side,
                "size": abs(qty),
                "entry_price": avg_price,
                "unrealised_pnl": float(expected_yield) if expected_yield else (current_price - avg_price) * qty,
                "leverage": "1",
            })

        return positions

    # ── Ticker (last price) ──────────────────────────────────

    def get_last_price(self, symbol: str, **kwargs) -> float:
        figi = self._get_figi(symbol)
        with self._new_client() as client:
            resp = client.market_data.get_last_prices(instrument_id=[figi])
            for lp in resp.last_prices:
                if lp.figi == figi or lp.instrument_uid == self._get_uid(symbol):
                    return _quotation_to_float(lp.price)
        raise ValueError(f"No price for {symbol}")

    # ── Instrument info ─────────────────────────────────────

    def get_instrument_info(self, symbol: str, *args, **kwargs) -> dict:
        info = self._instrument_cache.get(symbol)
        if not info:
            raise ValueError(f"Unknown instrument: {symbol}")
        return {
            "min_qty": float(info.lot),
            "max_qty": 1_000_000.0,
            "qty_step": float(info.lot),
            "tick_size": info.min_price_increment,
            "lot_size": info.lot,
        }

    # ── Closed PnL ──────────────────────────────────────────

    def get_closed_pnl(self, symbol: str, limit: int = 10, **kwargs) -> list[dict]:
        figi = self._get_figi(symbol)
        now = datetime.now(timezone.utc)
        from_dt = now - timedelta(days=7)

        with self._new_client() as client:
            if self.sandbox:
                resp = client.sandbox.get_sandbox_operations(
                    account_id=self.account_id,
                    from_=from_dt,
                    to=now,
                    figi=figi,
                )
            else:
                resp = client.operations.get_operations(
                    account_id=self.account_id,
                    from_=from_dt,
                    to=now,
                    figi=figi,
                )

        results = []
        for op in resp.operations:
            if op.state.name != "OPERATION_STATE_EXECUTED":
                continue
            payment = _quotation_to_float(op.payment)
            if payment == 0:
                continue
            results.append({
                "order_id": op.id,
                "symbol": symbol,
                "side": "Buy" if op.operation_type.name.endswith("BUY") else "Sell",
                "qty": float(op.quantity),
                "entry_price": _quotation_to_float(op.price),
                "exit_price": _quotation_to_float(op.price),
                "pnl": payment,
                "closed_at": op.date.isoformat() if op.date else "",
            })

        return results[:limit]

    # ── Trailing stop ─────────────────────────────────────────

    def set_trailing_stop(
        self, symbol: str, trailing_stop: float, active_price: float
    ):
        if self.sandbox:
            logger.info("Sandbox: skipping trailing stop for %s", symbol)
            return

        figi = self._get_figi(symbol)
        positions = self.get_positions(symbol=symbol)
        if not positions:
            return

        pos = positions[0]
        lots = max(1, int(pos["size"]))

        direction = (
            StopOrderDirection.STOP_ORDER_DIRECTION_SELL
            if pos["side"] == "Buy"
            else StopOrderDirection.STOP_ORDER_DIRECTION_BUY
        )

        with self._new_client() as client:
            try:
                client.stop_orders.post_stop_order(
                    figi=figi,
                    quantity=lots,
                    stop_price=_float_to_quotation(active_price),
                    direction=direction,
                    account_id=self.account_id,
                    stop_order_type=StopOrderType.STOP_ORDER_TYPE_STOP_LOSS,
                    expiration_type=StopOrderExpirationType.STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL,
                )
                logger.info("Trailing stop set: %s trail=%s active=%s", symbol, trailing_stop, active_price)
            except Exception as e:
                logger.warning("Failed to set trailing stop for %s: %s", symbol, e)
