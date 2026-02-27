"""Cross-instance position closing mixin (close_position, close_all_positions, _close_all_on_halt)."""

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


class PositionCrossMixin:
    """Methods for closing positions across engine instances."""

    async def close_position(self, symbol: str) -> str:
        """Close a single position by symbol (works across all instances)."""
        import aiosqlite

        # Determine if this symbol belongs to current engine or another instance
        is_other = False
        other_inst = None
        other_client = None
        other_db_path = None

        # Try current engine first
        closed = False
        pnl = 0.0
        close_side = "Buy"
        is_tbank = self.exchange_type == "tbank"

        try:
            categories = self._get_categories()
            for cat in categories:
                if cat not in ("linear", "tbank"):
                    continue
                if is_tbank:
                    positions = self.client.get_positions(symbol=symbol)
                else:
                    positions = self.client.get_positions(symbol=symbol, category=cat)
                for p in positions:
                    if p["symbol"] != symbol or p["size"] <= 0:
                        continue
                    close_side = "Sell" if p["side"] == "Buy" else "Buy"
                    if is_tbank:
                        self.client.place_order(symbol=symbol, side=close_side, qty=p["size"])
                    else:
                        self.client.place_order(symbol=symbol, side=close_side, qty=p["size"], category=cat, reduce_only=True)
                    closed = True
        except Exception as e:
            err_str = str(e)
            if "30079" in err_str or "not available for trading" in err_str.lower():
                return f"⏸ Биржа MOEX закрыта — {symbol} нельзя закрыть сейчас. Попробуй в торговую сессию (10:00-18:50 МСК)."
            # Symbol not on this exchange — check other instances
            pass

        if not closed:
            # Check other instances
            other_inst = self._find_instance_for_symbol(symbol)
            if other_inst and "TBANK" in other_inst.get("name", "").upper():
                try:
                    other_client = self._get_tbank_client_for_instance(other_inst)
                    if other_client:
                        positions = other_client.get_positions(symbol=symbol)
                        for p in positions:
                            if p["symbol"] != symbol or p["size"] <= 0:
                                continue
                            close_side = "Sell" if p["side"] == "Buy" else "Buy"
                            other_client.place_order(symbol=symbol, side=close_side, qty=p["size"])
                            closed = True
                            is_other = True
                            is_tbank = True
                except Exception as e:
                    err_str = str(e)
                    if "30079" in err_str or "not available for trading" in err_str.lower():
                        return f"⏸ Биржа MOEX закрыта — {symbol} нельзя закрыть сейчас. Попробуй в торговую сессию (10:00-18:50 МСК)."
                    logger.exception("Failed to close %s via other instance", symbol)
                    return f"❌ Ошибка при закрытии {symbol}"

        if not closed:
            return f"Позиция {symbol} не найдена на бирже."

        # Update DB — current engine or other instance
        db_path = other_inst["db_path"] if is_other and other_inst else None
        if is_other and db_path:
            try:
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    cur = await db.execute(
                        "SELECT id, side, entry_price, qty FROM trades WHERE symbol = ? AND status = 'open'",
                        (symbol,),
                    )
                    rows = await cur.fetchall()
                    for t in rows:
                        exit_price = other_client.get_last_price(symbol) if other_client else 0
                        pnl = self._calc_net_pnl(t["side"], t["entry_price"], exit_price, t["qty"])
                        await db.execute(
                            "UPDATE trades SET status='closed', exit_price=?, pnl=?, closed_at=? WHERE id=?",
                            (exit_price, pnl, datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%dT%H:%M:%S.%f"), t["id"]),
                        )
                        # Update daily_pnl in the other instance's DB
                        today = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d")
                        await db.execute(
                            """INSERT INTO daily_pnl (trade_date, pnl, trades_count)
                               VALUES (?, ?, 1)
                               ON CONFLICT(trade_date)
                               DO UPDATE SET pnl = pnl + ?, trades_count = trades_count + 1""",
                            (today, pnl, pnl),
                        )
                    await db.commit()
            except Exception:
                logger.exception("Failed to update other instance DB for %s", symbol)
        else:
            open_trades = await self.db.get_open_trades()
            for t in open_trades:
                if t["symbol"] != symbol:
                    continue
                try:
                    if is_tbank:
                        exit_price = self.client.get_last_price(symbol)
                    else:
                        exit_price = self.client.get_last_price(symbol, category=t["category"])
                    pnl = self._calc_net_pnl(t["side"], t["entry_price"], exit_price, t["qty"])
                    await self.db.close_trade(t["id"], exit_price, pnl)
                    await self.db.update_daily_pnl(pnl)
                    balance = self.client.get_balance()
                    await self._record_pnl(pnl, balance)
                except Exception:
                    logger.exception("Failed to update DB for %s", symbol)

        currency = "RUB" if is_tbank else "USDT"
        from datetime import datetime, timezone, timedelta
        msk = datetime.now(timezone(timedelta(hours=3)))
        msk_time = msk.strftime("%H:%M")
        try:
            icon = "🟢" if pnl >= 0 else "🔴"
            dur_str = ""
            try:
                opened_at = t.get("opened_at") if t else None
                if opened_at:
                    opened_dt = datetime.fromisoformat(opened_at)
                    dur_sec = int((datetime.now(timezone.utc) - opened_dt.replace(tzinfo=timezone(timedelta(hours=3)))).total_seconds())
                    if dur_sec >= 86400:
                        dur_str = f"{dur_sec // 86400}d {(dur_sec % 86400) // 3600}h"
                    elif dur_sec >= 3600:
                        dur_str = f"{dur_sec // 3600}h {(dur_sec % 3600) // 60}m"
                    elif dur_sec >= 60:
                        dur_str = f"{dur_sec // 60}m"
                    else:
                        dur_str = f"{dur_sec}s"
            except Exception:
                pass
            msg = f"{icon} {pnl:+,.2f} {currency} [CLOSED] [{dur_str}] [{msk_time} MSK]"
        except Exception:
            msg = f"🔴 {symbol} [CLOSED] [{msk_time} MSK]"
        logger.info(msg)
        await self._notify(msg)
        return msg

    async def _close_all_on_halt(self):
        """Auto-close all positions when daily loss limit reached."""
        try:
            open_trades = await self.db.get_open_trades()
            if not open_trades:
                await self._notify(
                    f"🛑 Дневной лимит убытков достигнут ({self.risk.daily_pnl:,.0f})\n"
                    f"Новые сделки заблокированы до завтра."
                )
                return

            # Close all positions on exchange
            categories = self._get_categories()
            closed_count = 0
            for cat in categories:
                if cat not in ("linear", "tbank"):
                    continue
                if self.exchange_type == "tbank":
                    positions = self.client.get_positions()
                else:
                    positions = self.client.get_positions(category=cat)
                for p in positions:
                    try:
                        close_side = "Sell" if p["side"] == "Buy" else "Buy"
                        if self.exchange_type == "tbank":
                            self.client.place_order(symbol=p["symbol"], side=close_side, qty=p["size"])
                        else:
                            self.client.place_order(symbol=p["symbol"], side=close_side, qty=p["size"], category=cat, reduce_only=True)
                        closed_count += 1
                    except Exception:
                        logger.exception("Halt close failed: %s", p["symbol"])

            # Update DB
            for t in open_trades:
                try:
                    if self.exchange_type == "tbank":
                        exit_price = self.client.get_last_price(t["symbol"])
                    else:
                        exit_price = self.client.get_last_price(t["symbol"], category="linear")
                    pnl = self._calc_net_pnl(t["side"], t["entry_price"], exit_price, t["qty"])
                    await self.db.close_trade(t["id"], exit_price, pnl)
                    await self.db.update_daily_pnl(pnl)
                except Exception:
                    logger.exception("Halt DB update failed: %s", t["symbol"])

            await self._notify(
                f"🛑 СТОП! Дневной лимит убытков ({self.risk.daily_pnl:,.0f})\n"
                f"Закрыто {closed_count} позиций. Торговля остановлена до завтра."
            )
        except Exception:
            logger.exception("_close_all_on_halt error")

    async def close_all_positions(self) -> str:
        import aiosqlite
        from pathlib import Path

        categories = self._get_categories()
        closed = []

        # 1. Close current engine positions
        for cat in categories:
            if cat not in ("linear", "tbank"):
                continue
            if self.exchange_type == "tbank":
                positions = self.client.get_positions()
            else:
                positions = self.client.get_positions(category=cat)
            for p in positions:
                try:
                    close_side = "Sell" if p["side"] == "Buy" else "Buy"
                    if self.exchange_type == "tbank":
                        self.client.place_order(
                            symbol=p["symbol"],
                            side=close_side,
                            qty=p["size"],
                        )
                    else:
                        self.client.place_order(
                            symbol=p["symbol"],
                            side=close_side,
                            qty=p["size"],
                            category=cat,
                        )
                    closed.append(f"{p['symbol']} ({p['side']})")
                except Exception:
                    logger.exception("Failed to close position %s", p["symbol"])

        # Mark current engine DB trades as closed
        open_trades = await self.db.get_open_trades()
        for t in open_trades:
            try:
                if self.exchange_type == "tbank":
                    exit_price = self.client.get_last_price(t["symbol"])
                else:
                    exit_price = self.client.get_last_price(t["symbol"], category=t["category"])
                pnl = self._calc_net_pnl(t["side"], t["entry_price"], exit_price, t["qty"])
                await self.db.close_trade(t["id"], exit_price, pnl)
                await self.db.update_daily_pnl(pnl)
                balance = self.client.get_balance()
                await self._record_pnl(pnl, balance)
            except Exception:
                logger.exception("Failed to update DB for %s", t["symbol"])

        # 2. Close other instances positions (cross-instance)
        for inst in self.config.get("other_instances", []):
            db_path = inst.get("db_path", "")
            inst_name = inst.get("name", "???")
            if not db_path or not Path(db_path).exists():
                continue
            try:
                import sqlite3 as sqlite3_sync
                conn = sqlite3_sync.connect(db_path)
                conn.row_factory = sqlite3_sync.Row
                rows = conn.execute(
                    "SELECT id, symbol, side FROM trades WHERE status = 'open'"
                ).fetchall()
                conn.close()
                if not rows:
                    continue
            except Exception:
                continue

            is_tbank = "TBANK" in inst_name.upper()
            other_client = None
            if is_tbank:
                try:
                    other_client = self._get_tbank_client_for_instance(inst)
                except Exception:
                    logger.warning("Cannot create client for %s", inst_name)
                    continue
            else:
                # Bybit instances share the same account — use own client
                other_client = self.client

            if not other_client:
                continue

            for row in rows:
                symbol = row["symbol"]
                try:
                    if is_tbank:
                        positions = other_client.get_positions(symbol=symbol)
                    else:
                        positions = other_client.get_positions(symbol=symbol, category="linear")
                    for p in positions:
                        if p["symbol"] != symbol or p["size"] <= 0:
                            continue
                        close_side = "Sell" if p["side"] == "Buy" else "Buy"
                        if is_tbank:
                            other_client.place_order(symbol=symbol, side=close_side, qty=p["size"])
                        else:
                            other_client.place_order(symbol=symbol, side=close_side, qty=p["size"], category="linear", reduce_only=True)
                        closed.append(f"{symbol} ({p['side']}) [{inst_name}]")
                except Exception as e:
                    err_str = str(e)
                    if "30079" in err_str or "not available for trading" in err_str.lower():
                        closed.append(f"{symbol} ⏸ MOEX закрыта [{inst_name}]")
                    else:
                        logger.exception("Failed to close %s on %s", symbol, inst_name)

            # Update other instance DB
            try:
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    cur = await db.execute(
                        "SELECT id, symbol, side, entry_price, qty FROM trades WHERE status = 'open'"
                    )
                    open_rows = await cur.fetchall()
                    for t in open_rows:
                        try:
                            exit_price = other_client.get_last_price(t["symbol"]) if is_tbank else other_client.get_last_price(t["symbol"], category="linear")
                            # Net PnL: use engine's commission rate (same exchange)
                            pnl = self._calc_net_pnl(t["side"], t["entry_price"], exit_price, t["qty"])
                            await db.execute(
                                "UPDATE trades SET status='closed', exit_price=?, pnl=?, closed_at=? WHERE id=?",
                                (exit_price, pnl, datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%dT%H:%M:%S.%f"), t["id"]),
                            )
                            today = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d")
                            await db.execute(
                                """INSERT INTO daily_pnl (trade_date, pnl, trades_count)
                                   VALUES (?, ?, 1)
                                   ON CONFLICT(trade_date)
                                   DO UPDATE SET pnl = pnl + ?, trades_count = trades_count + 1""",
                                (today, pnl, pnl),
                            )
                        except Exception:
                            logger.exception("Failed to update DB for %s [%s]", t["symbol"], inst_name)
                    await db.commit()
            except Exception:
                logger.exception("Failed to update %s DB", inst_name)

        if closed:
            msg = f"❌ Закрыто {len(closed)} позиций:\n" + "\n".join(f"  • {c}" for c in closed)
        else:
            msg = "Нет позиций для закрытия."
        logger.info(msg)
        return msg
