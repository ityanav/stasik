"""Standalone trade notifier — polls DBs for new opens/closes, sends Telegram."""

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import aiohttp
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trade_notifier")

MSK = timezone(timedelta(hours=3))

# All trade databases to monitor
DATABASES = {
    "FIBA": "/root/stasik/data/fiba.db",
    "BUBA": "/root/stasik/data/buba.db",
    "TBANK-SCALP": "/root/stasik/data/tbank_scalp.db",
    "TBANK-SWING": "/root/stasik/data/tbank_swing.db",
    "MIDAS": "/root/stasik/data/midas.db",
}

POLL_INTERVAL = 5  # seconds

# Load telegram config
_cfg_path = Path("/root/stasik/config/telegram.yaml")
if _cfg_path.exists():
    with open(_cfg_path) as f:
        _tg = yaml.safe_load(f).get("telegram", {})
else:
    _tg = {}

TG_TOKEN = _tg.get("token", "")
TG_CHAT_ID = str(_tg.get("chat_id", ""))


def _is_tbank(instance: str) -> bool:
    upper = instance.upper()
    return "TBANK" in upper or "MIDAS" in upper


def _currency(instance: str) -> str:
    return "RUB" if _is_tbank(instance) else "USDT"


def _now_msk() -> str:
    return datetime.now(MSK).strftime("%d.%m %H:%M")


def _get_max_id(db_path: str) -> int:
    """Get the max trade ID in a DB."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT MAX(id) FROM trades")
        row = cur.fetchone()
        conn.close()
        return row[0] or 0
    except Exception:
        return 0


def _get_new_trades(db_path: str, after_id: int) -> list[dict]:
    """Get all trades with id > after_id."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT id, symbol, side, qty, entry_price, exit_price, pnl, "
            "status, opened_at, closed_at, instance FROM trades "
            "WHERE id > ? ORDER BY id",
            (after_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.warning("Failed to read %s: %s", db_path, e)
        return []


def _get_newly_closed(db_path: str, open_ids: set[int]) -> list[dict]:
    """Check if any previously open trades are now closed."""
    if not open_ids:
        return []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("?" for _ in open_ids)
        cur = conn.execute(
            f"SELECT id, symbol, side, qty, entry_price, exit_price, pnl, "
            f"status, opened_at, closed_at, instance FROM trades "
            f"WHERE id IN ({placeholders}) AND status = 'closed'",
            list(open_ids),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.warning("Failed to check closed in %s: %s", db_path, e)
        return []


def _get_open_ids(db_path: str) -> set[int]:
    """Get IDs of all currently open trades."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT id FROM trades WHERE status = 'open'")
        ids = {row[0] for row in cur.fetchall()}
        conn.close()
        return ids
    except Exception:
        return set()


def _format_open(instance: str, trade: dict) -> str:
    symbol = trade["symbol"]
    side = "LONG" if trade["side"] == "Buy" else "SHORT"
    entry = float(trade["entry_price"] or 0)
    qty = float(trade["qty"] or 0)
    amount = round(entry * qty, 2)
    cur = _currency(instance)
    inst_label = trade.get("instance") or instance

    return (
        f"\U0001f535 {inst_label} | {side} {symbol}\n"
        f"{amount:.0f} {cur} | {_now_msk()}"
    )


def _format_close(instance: str, trade: dict) -> str:
    symbol = trade["symbol"]
    side = trade["side"]
    pnl = float(trade["pnl"] or 0)
    entry = float(trade["entry_price"] or 0)
    exit_p = float(trade["exit_price"] or 0)
    qty = float(trade["qty"] or 0)
    cur = _currency(instance)

    fee_rate = 0.0004 if _is_tbank(instance) else 0.00055
    fee = (entry * qty + exit_p * qty) * fee_rate
    net_pnl = pnl - fee

    sign = "+" if net_pnl >= 0 else ""
    emoji = "\u2705" if net_pnl >= 0 else "\u274c"
    inst_label = trade.get("instance") or instance

    return (
        f"{emoji} {inst_label} | {symbol} | {side}\n"
        f"Entry: {entry:.4g} \u2192 Exit: {exit_p:.4g}\n"
        f"Net PnL: {sign}{net_pnl:.2f} {cur} | {_now_msk()}"
    )


async def _send_telegram(text: str):
    if not TG_TOKEN or not TG_CHAT_ID:
        logger.warning("Telegram not configured, skipping notification")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(url, json={"chat_id": TG_CHAT_ID, "text": text})
    except Exception as e:
        logger.warning("Telegram send failed: %s", e)


async def main():
    logger.info("Trade notifier started, monitoring %d databases", len(DATABASES))

    # Initialize watermarks and open trade sets
    watermarks: dict[str, int] = {}
    open_trades: dict[str, set[int]] = {}

    for name, db_path in DATABASES.items():
        if Path(db_path).exists():
            watermarks[name] = _get_max_id(db_path)
            open_trades[name] = _get_open_ids(db_path)
            logger.info("  %s: watermark=%d, open=%d", name, watermarks[name], len(open_trades[name]))
        else:
            watermarks[name] = 0
            open_trades[name] = set()
            logger.info("  %s: DB not found, will check later", name)

    while True:
        await asyncio.sleep(POLL_INTERVAL)

        for name, db_path in DATABASES.items():
            if not Path(db_path).exists():
                continue

            # 1. Check for new trades (opens)
            last_id = watermarks.get(name, 0)
            new_trades = _get_new_trades(db_path, last_id)

            for trade in new_trades:
                tid = trade["id"]
                if trade["status"] == "open":
                    msg = _format_open(name, trade)
                    logger.info("New open: %s/%s id=%d", name, trade["symbol"], tid)
                    await _send_telegram(msg)
                    open_trades.setdefault(name, set()).add(tid)
                elif trade["status"] == "closed":
                    # Opened and already closed (e.g. instant close)
                    msg = _format_close(name, trade)
                    logger.info("New closed: %s/%s id=%d", name, trade["symbol"], tid)
                    await _send_telegram(msg)
                    open_trades.get(name, set()).discard(tid)
                watermarks[name] = max(watermarks.get(name, 0), tid)

            # 2. Check if previously open trades got closed
            tracked = open_trades.get(name, set())
            if tracked:
                closed = _get_newly_closed(db_path, tracked)
                for trade in closed:
                    msg = _format_close(name, trade)
                    logger.info("Closed: %s/%s id=%d", name, trade["symbol"], trade["id"])
                    await _send_telegram(msg)
                    tracked.discard(trade["id"])


if __name__ == "__main__":
    asyncio.run(main())
