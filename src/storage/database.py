import logging
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

MSK = timezone(timedelta(hours=3))


def now_msk() -> datetime:
    return datetime.now(MSK)


def now_msk_iso() -> str:
    return now_msk().strftime("%Y-%m-%dT%H:%M:%S.%f")

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "trades.db"


class Database:
    def __init__(self, db_path: Path = DB_PATH, instance_name: str = ""):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: aiosqlite.Connection | None = None
        self.instance_name = instance_name

    async def connect(self):
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._create_tables()
        logger.info("Database connected: %s", self.db_path)

    async def close(self):
        if self._db:
            await self._db.close()

    async def _create_tables(self):
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                category TEXT NOT NULL,
                qty REAL NOT NULL,
                entry_price REAL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                pnl REAL,
                order_id TEXT,
                status TEXT DEFAULT 'open',
                opened_at TEXT NOT NULL,
                closed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_pnl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date TEXT NOT NULL UNIQUE,
                pnl REAL NOT NULL DEFAULT 0,
                trades_count INTEGER NOT NULL DEFAULT 0
            );
        """)
        await self._db.commit()

        # Migration: add partial_closed column if not exists
        try:
            await self._db.execute(
                "ALTER TABLE trades ADD COLUMN partial_closed INTEGER DEFAULT 0"
            )
            await self._db.commit()
            logger.info("Migration: added partial_closed column")
        except Exception:
            pass  # column already exists

        # Migration: add instance column
        try:
            await self._db.execute(
                "ALTER TABLE trades ADD COLUMN instance TEXT DEFAULT ''"
            )
            await self._db.commit()
            logger.info("Migration: added instance column")
        except Exception:
            pass  # column already exists

    # ── Trades ───────────────────────────────────────────────

    async def insert_trade(
        self,
        symbol: str,
        side: str,
        category: str,
        qty: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        order_id: str,
    ) -> int:
        cursor = await self._db.execute(
            """INSERT INTO trades
               (symbol, side, category, qty, entry_price, stop_loss, take_profit, order_id, opened_at, instance)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol, side, category, qty, entry_price, stop_loss, take_profit, order_id,
             now_msk_iso(), self.instance_name),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def close_trade(self, trade_id: int, exit_price: float, pnl: float):
        await self._db.execute(
            """UPDATE trades
               SET exit_price = ?, pnl = ?, status = 'closed', closed_at = ?
               WHERE id = ?""",
            (exit_price, pnl, now_msk_iso(), trade_id),
        )
        await self._db.commit()

    async def get_open_trades(self) -> list[dict]:
        await self._db.commit()  # flush WAL — ensure fresh read
        cursor = await self._db.execute(
            "SELECT * FROM trades WHERE status = 'open' ORDER BY opened_at DESC"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def mark_partial_close(self, trade_id: int, new_qty: float):
        """Mark trade as partially closed, update remaining quantity."""
        await self._db.execute(
            "UPDATE trades SET partial_closed = 1, qty = ? WHERE id = ?",
            (new_qty, trade_id),
        )
        await self._db.commit()

    async def mark_scale_out(self, trade_id: int, stage: int, new_qty: float):
        """Update scale-out stage and remaining quantity after partial close."""
        await self._db.execute(
            "UPDATE trades SET partial_closed = ?, qty = ? WHERE id = ?",
            (stage, new_qty, trade_id),
        )
        await self._db.commit()

    async def update_stop_loss(self, trade_id: int, new_sl: float):
        """Update stop loss for an open trade (e.g. move to breakeven)."""
        await self._db.execute(
            "UPDATE trades SET stop_loss = ? WHERE id = ?",
            (new_sl, trade_id),
        )
        await self._db.commit()

    async def insert_partial_close(
        self,
        symbol: str,
        side: str,
        category: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        stage: int,
        opened_at: str,
    ) -> int:
        """Insert a closed trade record for a scale-out partial close."""
        cursor = await self._db.execute(
            """INSERT INTO trades
               (symbol, side, category, qty, entry_price, exit_price, pnl,
                stop_loss, take_profit, order_id, status, opened_at, closed_at,
                partial_closed, instance)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, '', 'closed', ?, ?, ?, ?)""",
            (symbol, side, category, qty, entry_price, exit_price, pnl,
             opened_at, now_msk_iso(), stage, self.instance_name),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def update_trade(self, trade_id: int, **kwargs):
        """Update fields of an open trade (e.g., stop_loss, take_profit)."""
        if not kwargs:
            return
        fields = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [trade_id]
        await self._db.execute(
            f"UPDATE trades SET {fields} WHERE id = ?", values
        )
        await self._db.commit()

    async def get_recent_trades(self, limit: int = 10, offset: int = 0) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Daily PnL ────────────────────────────────────────────

    async def update_daily_pnl(self, pnl: float):
        today = now_msk().date().isoformat()
        await self._db.execute(
            """INSERT INTO daily_pnl (trade_date, pnl, trades_count)
               VALUES (?, ?, 1)
               ON CONFLICT(trade_date)
               DO UPDATE SET pnl = pnl + ?, trades_count = trades_count + 1""",
            (today, pnl, pnl),
        )
        await self._db.commit()

    async def get_daily_pnl(self, day: date | None = None) -> float:
        day = day or now_msk().date()
        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(pnl), 0) as total FROM trades "
            "WHERE status = 'closed' AND date(closed_at) = ?",
            (day.isoformat(),),
        )
        row = await cursor.fetchone()
        return float(row["total"]) if row else 0.0

    async def get_total_pnl(self) -> float:
        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(pnl), 0) as total FROM trades WHERE status = 'closed'"
        )
        row = await cursor.fetchone()
        return float(row["total"])

    async def get_daily_pnl_history(self, days: int = 30) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT trade_date, pnl, trades_count FROM daily_pnl ORDER BY trade_date DESC LIMIT ?",
            (days,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in reversed(rows)]

    async def get_weekly_stats(self) -> dict:
        """Get trade stats for the last 7 days."""
        from datetime import timedelta
        week_ago = (now_msk().date() - timedelta(days=7)).isoformat()
        today = now_msk().date().isoformat()

        # PnL for the week
        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(pnl), 0) as pnl, COALESCE(SUM(trades_count), 0) as cnt "
            "FROM daily_pnl WHERE trade_date >= ?",
            (week_ago,),
        )
        row = await cursor.fetchone()
        weekly_pnl = float(row["pnl"])
        weekly_trades_count = int(row["cnt"])

        # Wins / losses for the week
        cursor = await self._db.execute(
            "SELECT "
            "COUNT(*) as total, "
            "COALESCE(SUM(CASE WHEN pnl >= 0 THEN 1 ELSE 0 END), 0) as wins, "
            "COALESCE(SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END), 0) as losses, "
            "COALESCE(MAX(pnl), 0) as best, "
            "COALESCE(MIN(pnl), 0) as worst, "
            "COALESCE(AVG(pnl), 0) as avg_pnl "
            "FROM trades WHERE status = 'closed' AND closed_at >= ?",
            (week_ago,),
        )
        row = await cursor.fetchone()

        # Best and worst trade details
        best_trade = None
        worst_trade = None
        if row["total"] > 0:
            cursor = await self._db.execute(
                "SELECT symbol, side, pnl, instance FROM trades "
                "WHERE status = 'closed' AND closed_at >= ? ORDER BY pnl DESC LIMIT 1",
                (week_ago,),
            )
            best_row = await cursor.fetchone()
            if best_row:
                best_trade = dict(best_row)

            cursor = await self._db.execute(
                "SELECT symbol, side, pnl, instance FROM trades "
                "WHERE status = 'closed' AND closed_at >= ? ORDER BY pnl ASC LIMIT 1",
                (week_ago,),
            )
            worst_row = await cursor.fetchone()
            if worst_row:
                worst_trade = dict(worst_row)

        return {
            "weekly_pnl": weekly_pnl,
            "total": int(row["total"]),
            "wins": int(row["wins"]),
            "losses": int(row["losses"]),
            "win_rate": (int(row["wins"]) / int(row["total"]) * 100) if int(row["total"]) > 0 else 0,
            "best": float(row["best"]),
            "worst": float(row["worst"]),
            "avg_pnl": float(row["avg_pnl"]),
            "best_trade": best_trade,
            "worst_trade": worst_trade,
        }

    async def get_trade_stats(self) -> dict:
        """Aggregate trade statistics."""
        cursor = await self._db.execute(
            "SELECT COUNT(*) as total, "
            "COALESCE(SUM(CASE WHEN pnl >= 0 THEN 1 ELSE 0 END), 0) as wins, "
            "COALESCE(SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END), 0) as losses, "
            "COALESCE(SUM(pnl), 0) as total_pnl "
            "FROM trades WHERE status = 'closed'"
        )
        row = await cursor.fetchone()
        return dict(row)
