"""Telegram bot actions — systemctl, position closing, DB updates."""

import logging
import sqlite3
import subprocess
from datetime import datetime, timezone, timedelta

import yaml

logger = logging.getLogger(__name__)


def systemctl_action(action: str, service: str) -> bool:
    try:
        r = subprocess.run(["systemctl", action, service], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except Exception:
        logger.exception("systemctl %s %s failed", action, service)
        return False


def close_bybit_position(config_path: str, symbol: str) -> str:
    """Close a Bybit position via API."""
    try:
        from pybit.unified_trading import HTTP

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        bybit_cfg = cfg["bybit"]
        http_kwargs = {
            "api_key": bybit_cfg["api_key"],
            "api_secret": bybit_cfg["api_secret"],
        }
        if bybit_cfg.get("demo"):
            http_kwargs["demo"] = True
        else:
            http_kwargs["testnet"] = bybit_cfg.get("testnet", False)

        session = HTTP(**http_kwargs)
        resp = session.get_positions(category="linear", symbol=symbol)
        for p in resp["result"]["list"]:
            size = float(p["size"])
            if size > 0:
                close_side = "Sell" if p["side"] == "Buy" else "Buy"
                session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=close_side,
                    orderType="Market",
                    qty=str(size),
                    reduceOnly=True,
                )
                return f"✅ {symbol} закрыт (market {close_side} {size})"
        return f"⚠️ {symbol} — позиция не найдена на бирже"
    except Exception as e:
        return f"❌ Ошибка закрытия {symbol}: {e}"


def close_tbank_position(config_path: str, symbol: str) -> str:
    """Close a TBank position via API."""
    try:
        from src.exchange.tbank_client import TBankClient

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        client = TBankClient(cfg)
        positions = client.get_positions(symbol=symbol)
        for p in positions:
            if p["symbol"] == symbol and p["size"] > 0:
                close_side = "Sell" if p["side"] == "Buy" else "Buy"
                client.place_order(symbol=symbol, side=close_side, qty=p["size"])
                return f"✅ {symbol} закрыт (market {close_side} {p['size']})"
        return f"⚠️ {symbol} — позиция не найдена на бирже"
    except Exception as e:
        err_str = str(e)
        if "30079" in err_str or "not available for trading" in err_str.lower():
            return f"⏸ Биржа MOEX закрыта — {symbol} нельзя закрыть сейчас"
        return f"❌ Ошибка закрытия {symbol}: {e}"


def update_db_closed(db_path: str, trade_id: int):
    """Mark trade as closed in DB."""
    try:
        conn = sqlite3.connect(db_path)
        now = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%dT%H:%M:%S.%f")
        conn.execute("UPDATE trades SET status='closed', closed_at=? WHERE id=?", (now, trade_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("DB update failed for trade %s: %s", trade_id, e)
