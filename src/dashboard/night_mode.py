"""Night Mode mixin for Dashboard — auto-close positions when net PnL >= target."""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp
import aiosqlite

from src.dashboard.services import _other_instances

logger = logging.getLogger(__name__)


class NightModeMixin:
    """Mixin providing night-mode auto-close functionality."""

    def _night_read_settings(self) -> dict:
        """Read night mode settings from JSON file."""
        try:
            if self._night_settings_path.exists():
                return json.loads(self._night_settings_path.read_text())
        except Exception:
            pass
        return {"enabled": False, "target": 0}

    async def _night_get_positions_with_pnl(self) -> list[dict]:
        """Get all open positions with unrealised PnL (reuses existing logic)."""
        positions = []
        instance_name = self.config.get("instance_name", "SCALP")

        # Main instance
        main_open = await self.db.get_open_trades()
        for t in main_open:
            positions.append({
                "symbol": t["symbol"],
                "side": t["side"],
                "size": float(t["qty"]),
                "entry_price": float(t["entry_price"]),
                "unrealised_pnl": 0.0,
                "instance": instance_name,
            })

        # Other instances
        for inst in _other_instances(self.config):
            inst_name = inst.get("name", "???")
            db_path = inst.get("db_path", "")
            if db_path and Path(db_path).exists():
                try:
                    async with aiosqlite.connect(db_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute(
                            "SELECT symbol, side, entry_price, qty FROM trades WHERE status = 'open'"
                        )
                        rows = await cur.fetchall()
                        for r in rows:
                            positions.append({
                                "symbol": r["symbol"],
                                "side": r["side"],
                                "size": float(r["qty"]),
                                "entry_price": float(r["entry_price"]),
                                "unrealised_pnl": 0.0,
                                "instance": inst_name,
                            })
                except Exception:
                    pass

        # Enrich with live prices — Bybit
        client = self._get_client()
        if client:
            try:
                raw = self._get_bybit_positions_cached()
                live_mark = {p["symbol"]: float(p.get("mark_price") or 0) for p in raw}
                for pos in positions:
                    inst_upper = (pos["instance"] or "").upper()
                    if "TBANK" in inst_upper or "MIDAS" in inst_upper:
                        continue
                    mark = live_mark.get(pos["symbol"], 0)
                    if not mark:
                        try:
                            mark = client.get_last_price(pos["symbol"], category="linear")
                        except Exception:
                            pass
                    if mark > 0:
                        direction = 1 if pos["side"] == "Buy" else -1
                        pos["unrealised_pnl"] = round(
                            (mark - float(pos["entry_price"])) * float(pos["size"]) * direction, 2
                        )
            except Exception:
                pass

        # Enrich — TBank/Midas
        tc = self._get_tbank_client()
        if tc:
            try:
                tbank_raw = self._get_tbank_positions_cached()
                tbank_mark = {p["symbol"]: p for p in tbank_raw}
                for pos in positions:
                    inst_upper = (pos["instance"] or "").upper()
                    if "TBANK" not in inst_upper and "MIDAS" not in inst_upper:
                        continue
                    live = tbank_mark.get(pos["symbol"])
                    if live:
                        pos["unrealised_pnl"] = round(float(live.get("unrealised_pnl", 0)), 2)
                    else:
                        try:
                            mark = tc.get_last_price(pos["symbol"])
                            if mark > 0:
                                direction = 1 if pos["side"] == "Buy" else -1
                                pos["unrealised_pnl"] = round(
                                    (mark - float(pos["entry_price"])) * float(pos["size"]) * direction, 2
                                )
                        except Exception:
                            pass
            except Exception:
                pass

        return positions

    async def _night_close_position(self, instance: str, symbol: str, side: str):
        """Close a position via exchange + DB (reuses close-position logic)."""
        is_tbank = any(k in (instance or "").upper() for k in ("TBANK", "MIDAS"))
        exchange_closed = False

        if is_tbank:
            try:
                from src.exchange.tbank_client import TBankClient
                for inst in _other_instances(self.config):
                    if inst.get("name", "").upper() == (instance or "").upper():
                        cfg_path = inst.get("config_path", "")
                        if cfg_path and Path(cfg_path).exists():
                            import yaml
                            with open(cfg_path) as f:
                                tcfg = yaml.safe_load(f)
                            tc = TBankClient(tcfg)
                            positions = tc.get_positions(symbol=symbol)
                            for p in positions:
                                if p["symbol"] == symbol and p["size"] > 0:
                                    close_side = "Sell" if p["side"] == "Buy" else "Buy"
                                    tc.place_order(symbol=symbol, side=close_side, qty=p["size"])
                                    exchange_closed = True
                            break
            except Exception as e:
                logger.warning("[NIGHT] Failed to close %s on TBank: %s", symbol, e)
        else:
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
                    logger.warning("[NIGHT] Failed to close %s on Bybit", symbol)

        # Close in DB
        db_path = self._resolve_instance_db(instance)
        if db_path and Path(db_path).exists():
            try:
                mark = 0
                if not is_tbank:
                    client = self._get_client()
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
                logger.exception("[NIGHT] Failed to close %s in DB", symbol)

        logger.info("[NIGHT] Closed %s/%s (exchange=%s)", instance, symbol, exchange_closed)

    async def _night_notify(self, instance: str, symbol: str, side: str, net_pnl: float, target: int):
        """Send Telegram notification about night-mode close."""
        tg = self.config.get("telegram", {})
        token = tg.get("token")
        chat_id = tg.get("chat_id")
        if not token or not chat_id:
            return
        currency = "RUB" if any(k in (instance or "").upper() for k in ("TBANK", "MIDAS")) else "USDT"
        icon = "\U0001f7e2" if net_pnl >= 0 else "\U0001f534"
        msk = datetime.now(timezone(timedelta(hours=3)))
        msk_time = msk.strftime("%H:%M")
        text = f"{icon} {net_pnl:+,.2f} {currency} [{instance}] [{msk_time} MSK]"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(url, json={"chat_id": chat_id, "text": text})
        except Exception as e:
            logger.warning("[NIGHT] Telegram notify failed: %s", e)

    async def _night_loop(self):
        """Background loop: check positions and auto-close when net PnL >= target."""
        logger.info("[NIGHT] Background loop started")
        closing = set()  # track in-flight closes
        while True:
            try:
                await asyncio.sleep(10)
                settings = self._night_read_settings()
                if not settings.get("enabled") or not settings.get("target"):
                    continue

                target = settings["target"]
                positions = await self._night_get_positions_with_pnl()

                for pos in positions:
                    gross_pnl = pos.get("unrealised_pnl", 0)
                    inst_upper = (pos.get("instance") or "").upper()
                    fee_rate = 0.0004 if ("TBANK" in inst_upper or "MIDAS" in inst_upper) else 0.00055
                    entry_amount = float(pos["entry_price"]) * float(pos["size"])
                    fee = entry_amount * fee_rate * 2
                    net_pnl = gross_pnl - fee

                    if net_pnl >= target:
                        key = f"{pos['instance']}_{pos['symbol']}"
                        if key in closing:
                            continue
                        closing.add(key)
                        logger.info(
                            "[NIGHT] Target hit: %s net=%.2f target=%d — closing",
                            key, net_pnl, target,
                        )
                        try:
                            await self._night_close_position(
                                pos["instance"], pos["symbol"], pos["side"],
                            )
                            await self._night_notify(
                                pos["instance"], pos["symbol"], pos["side"],
                                net_pnl, target,
                            )
                        except Exception:
                            logger.exception("[NIGHT] Error closing %s", key)
                        finally:
                            closing.discard(key)
            except asyncio.CancelledError:
                logger.info("[NIGHT] Background loop stopped")
                return
            except Exception:
                logger.exception("[NIGHT] Loop error")
