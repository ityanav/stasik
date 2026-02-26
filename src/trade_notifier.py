"""Standalone trade close notifier — polls DBs and sends Telegram messages."""

import asyncio
import logging
import sqlite3
import time
from pathlib import Path

import aiohttp
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trade_notifier")

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


def _get_last_closed_id(db_path: str) -> int:
    """Get the max ID of closed trades in a DB."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.execute(
            "SELECT MAX(id) FROM trades WHERE status = 'closed'"
        )
        row = cur.fetchone()
        conn.close()
        return row[0] or 0
    except Exception:
        return 0


def _get_new_closed_trades(db_path: str, after_id: int) -> list[dict]:
    """Get closed trades with id > after_id."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT id, symbol, side, qty, entry_price, exit_price, pnl, "
            "opened_at, closed_at, instance FROM trades "
            "WHERE status = 'closed' AND id > ? ORDER BY id",
            (after_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.warning("Failed to read %s: %s", db_path, e)
        return []


def _format_message(instance: str, trade: dict) -> str:
    symbol = trade["symbol"]
    side = trade["side"]
    pnl = float(trade["pnl"] or 0)
    entry = float(trade["entry_price"] or 0)
    exit_p = float(trade["exit_price"] or 0)
    qty = float(trade["qty"] or 0)
    cur = _currency(instance)

    # Net PnL (after commission)
    fee_rate = 0.0004 if _is_tbank(instance) else 0.00055
    fee = (entry * qty + exit_p * qty) * fee_rate
    net_pnl = pnl - fee

    sign = "+" if net_pnl >= 0 else ""
    emoji = "\u2705" if net_pnl >= 0 else "\u274c"

    inst_label = trade.get("instance") or instance

    return (
        f"{emoji} {inst_label} | {symbol} | {side}\n"
        f"Entry: {entry:.4g} \u2192 Exit: {exit_p:.4g}\n"
        f"Net PnL: {sign}{net_pnl:.2f} {cur}"
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

    # Initialize watermarks — start from current max IDs (don't notify old trades)
    watermarks: dict[str, int] = {}
    for name, db_path in DATABASES.items():
        if Path(db_path).exists():
            watermarks[name] = _get_last_closed_id(db_path)
            logger.info("  %s: watermark=%d", name, watermarks[name])
        else:
            watermarks[name] = 0
            logger.info("  %s: DB not found, will check later", name)

    while True:
        await asyncio.sleep(POLL_INTERVAL)

        for name, db_path in DATABASES.items():
            if not Path(db_path).exists():
                continue

            last_id = watermarks.get(name, 0)
            new_trades = _get_new_closed_trades(db_path, last_id)

            for trade in new_trades:
                msg = _format_message(name, trade)
                logger.info("New closed trade: %s/%s id=%d", name, trade["symbol"], trade["id"])
                await _send_telegram(msg)
                watermarks[name] = max(watermarks[name], trade["id"])


if __name__ == "__main__":
    asyncio.run(main())
