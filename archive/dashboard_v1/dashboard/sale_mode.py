"""SALE Mode mixin for Dashboard — auto-close Bybit positions when gross PnL >= target."""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp
import aiosqlite

logger = logging.getLogger(__name__)


class SaleModeMixin:
    """Mixin providing SALE-mode auto-close functionality (Bybit only, GROSS PnL)."""

    def _sale_read_settings(self) -> dict:
        try:
            if self._sale_settings_path.exists():
                return json.loads(self._sale_settings_path.read_text())
        except Exception:
            pass
        return {"enabled": False, "target": 0}

    async def _sale_close_position(self, instance: str, symbol: str, side: str):
        """Close a Bybit position (SALE mode — Bybit only)."""
        exchange_closed = False
        client = self._get_client()
        if client:
            try:
                positions = client.get_positions(symbol=symbol, category="linear")
                for p in positions:
                    if p["symbol"] == symbol and p["size"] > 0:
                        close_side = "Sell" if p["side"] == "Buy" else "Buy"
                        client.place_order(
                            symbol=symbol, side=close_side, qty=p["size"],
                            category="linear", reduce_only=True,
                        )
                        exchange_closed = True
            except Exception:
                logger.warning("[SALE] Failed to close %s on Bybit", symbol)

        db_path = self._resolve_instance_db(instance)
        if db_path and Path(db_path).exists():
            try:
                mark = 0
                if client:
                    try:
                        mark = client.get_last_price(symbol, category="linear")
                    except Exception:
                        pass
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    cur = await db.execute(
                        "SELECT id, side, entry_price, qty FROM trades WHERE symbol=? AND status='open'",
                        (symbol,),
                    )
                    rows = await cur.fetchall()
                    for r in rows:
                        entry = float(r["entry_price"])
                        qty = float(r["qty"])
                        pnl = 0.0
                        if mark > 0:
                            direction = 1 if r["side"] == "Buy" else -1
                            pnl = round((mark - entry) * qty * direction, 2)
                        await db.execute(
                            "UPDATE trades SET exit_price=?, pnl=?, status='closed', closed_at=? WHERE id=?",
                            (mark if mark > 0 else entry, pnl, datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%dT%H:%M:%S.%f"), r["id"]),
                        )
                    await db.commit()
            except Exception:
                logger.exception("[SALE] Failed to close %s in DB", symbol)

        logger.info("[SALE] Closed %s/%s (exchange=%s)", instance, symbol, exchange_closed)

    async def _sale_notify(self, instance: str, symbol: str, side: str, gross_pnl: float, target: int):
        tg = self.config.get("telegram", {})
        token = tg.get("token")
        chat_id = tg.get("chat_id")
        if not token or not chat_id:
            return
        net_pnl = gross_pnl  # gross_pnl passed from sale_loop already
        icon = "\U0001f7e2" if net_pnl >= 0 else "\U0001f534"
        direction = "LONG" if side == "Buy" else "SHORT"
        msk = datetime.now(timezone(timedelta(hours=3)))
        msk_time = msk.strftime("%H:%M")
        text = f"{icon} {instance} | {symbol} {direction} закрыт\n   {net_pnl:+,.2f} USDT (net) | {msk_time} MSK"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(url, json={"chat_id": chat_id, "text": text})
        except Exception as e:
            logger.warning("[SALE] Telegram notify failed: %s", e)

    async def _sale_loop(self):
        """Background loop: close Bybit positions when GROSS PnL >= target."""
        logger.info("[SALE] Background loop started")
        closing = set()
        while True:
            try:
                await asyncio.sleep(10)
                settings = self._sale_read_settings()
                if not settings.get("enabled") or not settings.get("target"):
                    continue

                target = settings["target"]
                positions = await self._night_get_positions_with_pnl()

                for pos in positions:
                    inst_upper = (pos.get("instance") or "").upper()
                    # SALE only for Bybit
                    if "TBANK" in inst_upper or "MIDAS" in inst_upper:
                        continue

                    gross_pnl = pos.get("unrealised_pnl", 0)
                    if gross_pnl >= target:
                        key = f"{pos['instance']}_{pos['symbol']}"
                        if key in closing:
                            continue
                        closing.add(key)
                        logger.info(
                            "[SALE] Target hit: %s gross=%.2f target=%d — closing",
                            key, gross_pnl, target,
                        )
                        try:
                            await self._sale_close_position(
                                pos["instance"], pos["symbol"], pos["side"],
                            )
                            await self._sale_notify(
                                pos["instance"], pos["symbol"], pos["side"],
                                gross_pnl, target,
                            )
                        except Exception:
                            logger.exception("[SALE] Error closing %s", key)
                        finally:
                            closing.discard(key)
            except asyncio.CancelledError:
                logger.info("[SALE] Background loop stopped")
                return
            except Exception:
                logger.exception("[SALE] Loop error")
