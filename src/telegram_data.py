"""Telegram bot data layer — instance config, DB queries, exchange APIs."""

import logging
import sqlite3
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Instance definitions: name, db_path, service, currency, config_path, exchange_type
INSTANCES = [
    {"name": "FIBA", "db": "/root/stasik/data/fiba.db", "service": "stasik-fiba", "currency": "USDT", "config": "/root/stasik/config/fiba.yaml", "exchange": "bybit"},
    {"name": "BUBA", "db": "/root/stasik/data/buba.db", "service": "stasik-buba", "currency": "USDT", "config": "/root/stasik/config/buba.yaml", "exchange": "bybit"},
    {"name": "TBANK-SCALP", "db": "/root/stasik/data/tbank_scalp.db", "service": "stasik-tbank-scalp", "currency": "RUB", "config": "/root/stasik/config/tbank_scalp.yaml", "exchange": "tbank"},
    {"name": "TBANK-SWING", "db": "/root/stasik/data/tbank_swing.db", "service": "stasik-tbank-swing", "currency": "RUB", "config": "/root/stasik/config/tbank_swing.yaml", "exchange": "tbank"},
    {"name": "MIDAS", "db": "/root/stasik/data/midas.db", "service": "stasik-midas", "currency": "RUB", "config": "/root/stasik/config/midas.yaml", "exchange": "tbank"},
]

INSTANCE_ICONS = {
    "FIBA": "🧠",
    "BUBA": "🦬",
    "TBANK-SCALP": "🏦",
    "TBANK-SWING": "📅",
    "MIDAS": "👑",
}


def query_db(db_path: str, sql: str, params: tuple = ()) -> list:
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("DB query failed (%s): %s", db_path, e)
        return []


def is_service_active(service: str) -> bool:
    try:
        r = subprocess.run(["systemctl", "is-active", service], capture_output=True, text=True, timeout=3)
        return r.stdout.strip() == "active"
    except Exception:
        return False


def find_instance(name: str) -> Optional[dict]:
    for inst in INSTANCES:
        if inst["name"] == name:
            return inst
    return None


def get_bybit_balance() -> float:
    """Get Bybit USDT wallet balance."""
    try:
        from pybit.unified_trading import HTTP
        with open("/root/stasik/config/fiba.yaml") as f:
            cfg = yaml.safe_load(f)
        bybit_cfg = cfg["bybit"]
        http_kwargs = {"api_key": bybit_cfg["api_key"], "api_secret": bybit_cfg["api_secret"]}
        if bybit_cfg.get("demo"):
            http_kwargs["demo"] = True
        else:
            http_kwargs["testnet"] = bybit_cfg.get("testnet", False)
        session = HTTP(**http_kwargs)
        resp = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        for acct in resp["result"]["list"]:
            for c in acct["coin"]:
                if c["coin"] == "USDT":
                    return float(c["walletBalance"])
    except Exception as e:
        logger.warning("Bybit balance error: %s", e)
    return 0.0


def get_tbank_balance() -> float:
    """Get TBank RUB balance."""
    try:
        from src.exchange.tbank_client import TBankClient
        with open("/root/stasik/config/tbank_scalp.yaml") as f:
            cfg = yaml.safe_load(f)
        tc = TBankClient(cfg)
        return tc.get_balance("RUB")
    except Exception as e:
        logger.warning("TBank balance error: %s", e)
    return 0.0


def get_instance_daily_stats(inst: dict, today: str) -> dict:
    """Get daily PnL stats for an instance. Returns dict with day_pnl, wins, losses, open_cnt."""
    result = {"day_pnl": 0.0, "wins": 0, "losses": 0, "open_cnt": 0}

    if not inst["db"] or not Path(inst["db"]).exists():
        return result

    # Open positions count
    open_rows = query_db(inst["db"], "SELECT COUNT(*) as cnt FROM trades WHERE status='open'")
    result["open_cnt"] = open_rows[0]["cnt"] if open_rows else 0

    if not is_service_active(inst["service"]):
        return result

    # Daily net PnL
    rows = query_db(
        inst["db"],
        "SELECT SUM(pnl) as total, "
        "SUM(CASE WHEN pnl >= 0 THEN 1 ELSE 0 END) as wins, "
        "SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses "
        "FROM trades WHERE status='closed' AND date(closed_at)=?",
        (today,),
    )
    if rows and rows[0]["total"] is not None:
        gross = float(rows[0]["total"])
        result["wins"] = int(rows[0]["wins"] or 0)
        result["losses"] = int(rows[0]["losses"] or 0)

        # Calc fees for net PnL
        fee_rows = query_db(
            inst["db"],
            "SELECT entry_price, exit_price, qty FROM trades "
            "WHERE status='closed' AND date(closed_at)=?",
            (today,),
        )
        fee_rate = 0.0004 if inst["currency"] == "RUB" else 0.00055
        total_fee = sum(
            (float(r["entry_price"] or 0) * float(r["qty"] or 0) +
             float(r["exit_price"] or 0) * float(r["qty"] or 0)) * fee_rate
            for r in fee_rows
        )
        result["day_pnl"] = gross - total_fee

    return result


def enrich_pnl(positions: list[dict]):
    """Add live unrealised net PnL to positions from exchanges."""
    # Bybit
    bybit_marks = {}
    bybit_session = None
    bybit_positions = [p for p in positions if p["exchange"] == "bybit"]
    if bybit_positions:
        try:
            from pybit.unified_trading import HTTP
            with open("/root/stasik/config/fiba.yaml") as f:
                cfg = yaml.safe_load(f)
            bybit_cfg = cfg["bybit"]
            http_kwargs = {"api_key": bybit_cfg["api_key"], "api_secret": bybit_cfg["api_secret"]}
            if bybit_cfg.get("demo"):
                http_kwargs["demo"] = True
            else:
                http_kwargs["testnet"] = bybit_cfg.get("testnet", False)
            bybit_session = HTTP(**http_kwargs)
            resp = bybit_session.get_positions(category="linear", limit=200)
            for p in resp["result"]["list"]:
                if float(p["size"]) > 0:
                    mark = float(p.get("markPrice") or 0)
                    if mark > 0:
                        bybit_marks[p["symbol"]] = mark
        except Exception as e:
            logger.warning("Bybit positions fetch error: %s", e)

    # TBank
    tbank_pnl = {}
    tbank_positions = [p for p in positions if p["exchange"] == "tbank"]
    if tbank_positions:
        try:
            from src.exchange.tbank_client import TBankClient
            with open("/root/stasik/config/tbank_scalp.yaml") as f:
                cfg = yaml.safe_load(f)
            tc = TBankClient(cfg)
            raw = tc.get_positions()
            for p in raw:
                tbank_pnl[p["symbol"]] = float(p.get("unrealised_pnl", 0))
        except Exception as e:
            logger.warning("TBank positions fetch error: %s", e)

    for pos in positions:
        entry = float(pos["entry_price"] or 0)
        qty = float(pos["qty"] or 0)
        fee_rate = 0.0004 if pos["currency"] == "RUB" else 0.00055

        if pos["exchange"] == "bybit":
            mark = bybit_marks.get(pos["symbol"], 0)
            # Fallback: get last price via tickers if mark not found
            if mark <= 0 and bybit_session:
                try:
                    resp = bybit_session.get_tickers(category="linear", symbol=pos["symbol"])
                    mark = float(resp["result"]["list"][0]["lastPrice"])
                except Exception:
                    pass
            if mark > 0 and entry > 0 and qty > 0:
                direction = 1 if pos["side"] == "Buy" else -1
                gross = (mark - entry) * qty * direction
                fee = (entry * qty + mark * qty) * fee_rate
                pos["net_pnl"] = round(gross - fee, 2)
            else:
                pos["net_pnl"] = 0.0
        elif pos["exchange"] == "tbank":
            gross = tbank_pnl.get(pos["symbol"], 0)
            fee = entry * qty * fee_rate * 2  # approx round-trip
            pos["net_pnl"] = round(gross - fee, 2)
        else:
            pos["net_pnl"] = 0.0
