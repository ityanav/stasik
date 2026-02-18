import logging
from datetime import date, datetime
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "trades.db"


class Database:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: aiosqlite.Connection | None = None

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
               (symbol, side, category, qty, entry_price, stop_loss, take_profit, order_id, opened_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol, side, category, qty, entry_price, stop_loss, take_profit, order_id,
             datetime.utcnow().isoformat()),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def close_trade(self, trade_id: int, exit_price: float, pnl: float):
        await self._db.execute(
            """UPDATE trades
               SET exit_price = ?, pnl = ?, status = 'closed', closed_at = ?
               WHERE id = ?""",
            (exit_price, pnl, datetime.utcnow().isoformat(), trade_id),
        )
        await self._db.commit()

    async def get_open_trades(self) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM trades WHERE status = 'open'"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_recent_trades(self, limit: int = 10) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Daily PnL ────────────────────────────────────────────

    async def update_daily_pnl(self, pnl: float):
        today = date.today().isoformat()
        await self._db.execute(
            """INSERT INTO daily_pnl (trade_date, pnl, trades_count)
               VALUES (?, ?, 1)
               ON CONFLICT(trade_date)
               DO UPDATE SET pnl = pnl + ?, trades_count = trades_count + 1""",
            (today, pnl, pnl),
        )
        await self._db.commit()

    async def get_daily_pnl(self, day: date | None = None) -> float:
        day = day or date.today()
        cursor = await self._db.execute(
            "SELECT pnl FROM daily_pnl WHERE trade_date = ?",
            (day.isoformat(),),
        )
        row = await cursor.fetchone()
        return float(row["pnl"]) if row else 0.0

    async def get_total_pnl(self) -> float:
        cursor = await self._db.execute("SELECT COALESCE(SUM(pnl), 0) as total FROM daily_pnl")
        row = await cursor.fetchone()
        return float(row["total"])
