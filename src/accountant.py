"""Бухгалтер — AI profit-taking watchdog.

Monitors open Bybit positions (SCALP + SMC) every 30s and uses DeepSeek AI
to decide whether to take profit early instead of waiting for TP.
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import httpx
import pandas as pd
import yaml

from src.exchange.client import BybitClient
from src.storage.database import Database
from src.strategy.indicators import calculate_ema, calculate_rsi

logger = logging.getLogger(__name__)

# ── AI system prompt ───────────────────────────────────────────────

ACCOUNTANT_PROMPT = """\
Ты — бухгалтер трейдинг-бота. Решаешь: фиксировать прибыль, держать, или двигать SL/TP.

Данные:
- Позиция: символ, LONG/SHORT, вход, текущая цена, PnL в % и $
- Безубыток (вход + комиссия round-trip)
- Время в сделке
- Расстояние до TP и SL в %
- RSI (14), EMA 9 vs 21 (тренд), объём
- Последние 10 свечей 15m

Решения:
- **HOLD** — ничего не делать
- **CLOSE** — закрыть позицию целиком
- **MOVE_SL** — передвинуть стоп-лосс. Указать `new_sl` (цена). Варианты:
  - Безубыток: SL = цена безубытка (дана в данных)
  - Трейлинг: SL = текущая - X% (лонг) / текущая + X% (шорт)
  - Двигай SL ТОЛЬКО В СТОРОНУ ПРИБЫЛИ (ближе к цене, никогда дальше от неё)
- **MOVE_TP** — передвинуть тейк-профит. Указать `new_tp` (цена).
  - Расширить TP если тренд сильный и momentum растёт
  - Сузить TP если momentum слабеет и цена замедляется

Правила:
- HOLD если тренд сильный и PnL растёт (EMA 9 > 21 для лонга)
- CLOSE если: RSI перекуплен (>70 для лонга, <30 для шорта) + momentum слабеет
- CLOSE если: цена была ближе к TP, теперь отдаляется (разворот)
- CLOSE если: объём падает и цена стоит на месте
- HOLD если PnL слишком мал (< 0.3%) — не стоит фиксировать
- MOVE_SL в безубыток если PnL > 0.5% и тренд неуверенный
- MOVE_SL трейлинг если PnL > 1% и тренд сильный — подтяни SL за ценой
- Лучше зафиксировать +0.5% чем получить -1.5% по SL
- Учитывай уроки из прошлых ошибок. Не повторяй те же ошибки.

JSON (без markdown):
CLOSE/HOLD: {"decision":"CLOSE","confidence":1-10,"reasoning":"кратко"}
MOVE_SL: {"decision":"MOVE_SL","confidence":1-10,"reasoning":"кратко","new_sl":12345.0}
MOVE_TP: {"decision":"MOVE_TP","confidence":1-10,"reasoning":"кратко","new_tp":12345.0}\
"""

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


# ── Helpers ────────────────────────────────────────────────────────

def _summarize_candles(df: pd.DataFrame, n: int = 10) -> str:
    tail = df.tail(n)
    lines = []
    for _, row in tail.iterrows():
        ts = row["timestamp"]
        ts_str = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)
        lines.append(
            f"{ts_str} O={row['open']:.2f} H={row['high']:.2f} "
            f"L={row['low']:.2f} C={row['close']:.2f} V={row['volume']:.0f}"
        )
    return "\n".join(lines)


def _calc_net_pnl(side: str, entry: float, exit_price: float,
                   qty: float, commission_rate: float) -> float:
    gross = (exit_price - entry) * qty if side == "Buy" else (entry - exit_price) * qty
    fee = (entry * qty + exit_price * qty) * commission_rate
    return gross - fee


# ── Accountant ─────────────────────────────────────────────────────

class Accountant:
    def __init__(self, config: dict):
        self.config = config
        self.client = BybitClient(config)

        acct = config["accountant"]
        self.check_interval: int = acct.get("check_interval", 30)
        self.min_profit_pct: float = acct.get("min_profit_pct", 0.15)
        self.min_confidence: int = acct.get("min_confidence", 6)
        self.timeframe: str = str(acct.get("timeframe", "15"))

        ai = config["ai"]
        self.ai_key: str = ai["api_key"]
        self.ai_model: str = ai.get("model", "deepseek-chat")
        self.ai_timeout: int = ai.get("timeout", 15)

        self.databases: list[dict] = config["databases"]

        tg = config.get("telegram", {})
        self.tg_token: str = tg.get("token", "")
        self.tg_chat: str = str(tg.get("chat_id", ""))

        self._http: httpx.AsyncClient | None = None
        self._dbs: dict[str, Database] = {}  # path -> Database
        self._own_db: aiosqlite.Connection | None = None
        self._last_hold_log: dict[tuple, float] = {}  # (trade_id, instance) -> timestamp
        self._evaluated_trades: set[tuple] = set()  # (trade_id, instance)

    # ── lifecycle ──────────────────────────────────────────────

    async def start(self):
        self._http = httpx.AsyncClient(timeout=self.ai_timeout)
        for db_cfg in self.databases:
            path = db_cfg["path"]
            db = Database(Path(path), instance_name=db_cfg["instance"])
            await db.connect()
            self._dbs[path] = db
        await self._init_own_db()
        instances = [d["instance"] for d in self.databases]
        logger.info("Accountant started: monitoring %s", instances)

    async def stop(self):
        for db in self._dbs.values():
            await db.close()
        if self._own_db:
            await self._own_db.close()
        if self._http:
            await self._http.aclose()
        logger.info("Accountant stopped")

    async def run(self):
        await self.start()
        try:
            while True:
                try:
                    await self._tick()
                except Exception:
                    logger.exception("Accountant tick error")
                await asyncio.sleep(self.check_interval)
        finally:
            await self.stop()

    # ── main tick ──────────────────────────────────────────────

    async def _tick(self):
        await self._evaluate_closed_trades()

        positions = await self._collect_open_trades()
        if not positions:
            return

        logger.info("Accountant: checking %d positions", len(positions))

        for pos in positions:
            net_pnl_pct = pos["net_pnl_pct"]
            if net_pnl_pct < self.min_profit_pct:
                continue

            verdict = await self._ask_ai(pos)
            if verdict is None:
                continue

            decision = verdict.get("decision", "HOLD")
            confidence = verdict.get("confidence", 0)
            reasoning = verdict.get("reasoning", "")

            logger.info(
                "AI verdict %s: %s +%.2f%% (confidence=%d) — %s",
                decision, pos["symbol"], net_pnl_pct, confidence, reasoning,
            )

            await self._log_decision(pos, decision, confidence, reasoning)

            if decision == "CLOSE" and confidence >= self.min_confidence:
                await self._close_position(pos, reasoning)
            elif decision == "MOVE_SL" and confidence >= self.min_confidence:
                new_sl = verdict.get("new_sl")
                if new_sl:
                    await self._move_sl(pos, float(new_sl), reasoning)
            elif decision == "MOVE_TP" and confidence >= self.min_confidence:
                new_tp = verdict.get("new_tp")
                if new_tp:
                    await self._move_tp(pos, float(new_tp), reasoning)

    # ── collect trades ─────────────────────────────────────────

    async def _collect_open_trades(self) -> list[dict]:
        result = []
        for db_cfg in self.databases:
            path = db_cfg["path"]
            db = self._dbs.get(path)
            if not db:
                continue

            trades = await db.get_open_trades()
            for t in trades:
                t = dict(t)
                t["instance"] = db_cfg["instance"]
                t["db_path"] = path
                t["commission_rate"] = db_cfg["commission_rate"]

                symbol = t["symbol"]
                try:
                    cur_price = self.client.get_last_price(symbol)
                except Exception:
                    logger.warning("Cannot get price for %s", symbol)
                    continue

                t["cur_price"] = cur_price
                side = t["side"]
                entry = t["entry_price"]
                qty = t["qty"]
                comm = db_cfg["commission_rate"]

                net_pnl = _calc_net_pnl(side, entry, cur_price, qty, comm)
                # PnL as percent of position notional
                notional = entry * qty
                net_pnl_pct = (net_pnl / notional * 100) if notional else 0.0

                t["net_pnl"] = net_pnl
                t["net_pnl_pct"] = net_pnl_pct
                result.append(t)
        return result

    # ── AI decision ────────────────────────────────────────────

    async def _ask_ai(self, pos: dict) -> dict | None:
        symbol = pos["symbol"]
        try:
            df = self.client.get_klines(symbol, interval=self.timeframe, limit=50)
        except Exception:
            logger.warning("Cannot get klines for %s", symbol)
            return None

        rsi_series = calculate_rsi(df, period=14)
        rsi = float(rsi_series.iloc[-1]) if len(rsi_series) else 0.0
        ema_fast, ema_slow = calculate_ema(df, fast=9, slow=21)
        ema9 = float(ema_fast.iloc[-1]) if len(ema_fast) else 0.0
        ema21 = float(ema_slow.iloc[-1]) if len(ema_slow) else 0.0

        last_vol = float(df["volume"].iloc[-1]) if len(df) else 0
        avg_vol = float(df["volume"].tail(20).mean()) if len(df) >= 20 else last_vol

        candles = _summarize_candles(df, n=10)

        side = pos["side"]
        direction = "LONG" if side == "Buy" else "SHORT"
        entry = pos["entry_price"]
        cur = pos["cur_price"]
        tp = pos.get("take_profit") or 0.0
        sl = pos.get("stop_loss") or 0.0

        tp_dist_pct = abs((tp - cur) / cur * 100) if tp else 0.0
        sl_dist_pct = abs((sl - cur) / cur * 100) if sl else 0.0

        opened_at = pos.get("opened_at", "")
        time_in_trade = ""
        if opened_at:
            try:
                opened_dt = datetime.fromisoformat(opened_at)
                delta = datetime.utcnow() - opened_dt
                hours = int(delta.total_seconds() // 3600)
                mins = int((delta.total_seconds() % 3600) // 60)
                time_in_trade = f"{hours}h {mins}m"
            except Exception:
                time_in_trade = "?"

        comm = pos["commission_rate"]
        if side == "Buy":
            breakeven = entry * (1 + 2 * comm)
        else:
            breakeven = entry * (1 - 2 * comm)

        user_msg = (
            f"Символ: {symbol}\n"
            f"Направление: {direction}\n"
            f"Вход: {entry:.4f}, Текущая: {cur:.4f}\n"
            f"Безубыток (вход+комиссия): {breakeven:.4f}\n"
            f"PnL: {pos['net_pnl_pct']:.2f}% ({pos['net_pnl']:.2f}$)\n"
            f"Время в сделке: {time_in_trade}\n"
            f"TP: {tp:.4f} ({tp_dist_pct:.2f}% от текущей)\n"
            f"SL: {sl:.4f} ({sl_dist_pct:.2f}% от текущей)\n"
            f"RSI(14): {rsi:.1f}\n"
            f"EMA9: {ema9:.4f}, EMA21: {ema21:.4f} "
            f"({'бычий' if ema9 > ema21 else 'медвежий'} тренд)\n"
            f"Объём: {last_vol:.0f} (avg20: {avg_vol:.0f}, "
            f"ratio: {last_vol / avg_vol:.2f}x)\n"
            f"Инстанс: {pos['instance']}\n\n"
            f"Свечи {self.timeframe}m (последние 10):\n{candles}"
        )

        lessons = await self._get_recent_lessons(limit=10)
        if lessons:
            user_msg += "\n\nУроки из прошлых ошибок:\n" + "\n".join(
                f"- {l}" for l in lessons
            )

        return await self._call_deepseek(user_msg)

    async def _call_deepseek(self, user_msg: str) -> dict | None:
        try:
            resp = await self._http.post(
                DEEPSEEK_URL,
                headers={
                    "Authorization": f"Bearer {self.ai_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.ai_model,
                    "messages": [
                        {"role": "system", "content": ACCOUNTANT_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 300,
                },
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
            logger.warning("DeepSeek error: %s", e)
            return None

    # ── self-learning ─────────────────────────────────────────

    async def _init_own_db(self):
        db_path = Path(__file__).resolve().parent.parent / "data" / "accountant.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._own_db = await aiosqlite.connect(str(db_path))
        self._own_db.row_factory = aiosqlite.Row
        await self._own_db.executescript("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER NOT NULL,
                instance TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                decision TEXT NOT NULL,
                confidence INTEGER,
                reasoning TEXT,
                price_at_decision REAL,
                net_pnl_pct REAL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER NOT NULL,
                instance TEXT NOT NULL,
                symbol TEXT NOT NULL,
                lesson TEXT NOT NULL,
                lesson_type TEXT NOT NULL,
                pnl_at_decision REAL,
                pnl_at_close REAL,
                created_at TEXT NOT NULL
            );
        """)
        await self._own_db.commit()
        logger.info("Accountant own DB initialized: %s", db_path)

    async def _log_decision(self, pos: dict, decision: str,
                            confidence: int, reasoning: str):
        trade_id = pos["id"]
        instance = pos["instance"]
        key = (trade_id, instance)

        # Deduplicate HOLD: max once per 60s per (trade_id, instance)
        if decision == "HOLD":
            now = time.monotonic()
            last = self._last_hold_log.get(key, 0)
            if now - last < 60:
                return
            self._last_hold_log[key] = now

        await self._own_db.execute(
            """INSERT INTO decisions
               (trade_id, instance, symbol, side, decision, confidence,
                reasoning, price_at_decision, net_pnl_pct, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trade_id, instance, pos["symbol"], pos["side"],
             decision, confidence, reasoning,
             pos["cur_price"], pos["net_pnl_pct"],
             datetime.utcnow().isoformat()),
        )
        await self._own_db.commit()

    async def _evaluate_closed_trades(self):
        """Check trades that were tracked in decisions and are now closed."""
        for db_cfg in self.databases:
            path = db_cfg["path"]
            db = self._dbs.get(path)
            instance = db_cfg["instance"]
            if not db:
                continue

            # Get distinct trade_ids we have decisions for in this instance
            cursor = await self._own_db.execute(
                "SELECT DISTINCT trade_id FROM decisions WHERE instance = ?",
                (instance,),
            )
            tracked_ids = [row[0] for row in await cursor.fetchall()]

            for trade_id in tracked_ids:
                key = (trade_id, instance)
                if key in self._evaluated_trades:
                    continue

                # Check if trade is closed in trades DB
                try:
                    t_cursor = await db._db.execute(
                        "SELECT * FROM trades WHERE id = ? AND status = 'closed'",
                        (trade_id,),
                    )
                    row = await t_cursor.fetchone()
                except Exception:
                    continue

                if not row:
                    continue  # still open

                trade = dict(row)
                self._evaluated_trades.add(key)

                # Get the last decision for this trade
                d_cursor = await self._own_db.execute(
                    """SELECT * FROM decisions
                       WHERE trade_id = ? AND instance = ?
                       ORDER BY created_at DESC LIMIT 1""",
                    (trade_id, instance),
                )
                last_dec = await d_cursor.fetchone()
                if not last_dec:
                    continue

                last_dec = dict(last_dec)
                pnl_at_decision = last_dec["net_pnl_pct"] or 0.0
                decision = last_dec["decision"]
                symbol = last_dec["symbol"]

                # Calculate final PnL % from trade
                entry = trade.get("entry_price", 0)
                exit_p = trade.get("exit_price", 0)
                qty = trade.get("qty", 0)
                comm = db_cfg["commission_rate"]
                if entry and exit_p and qty:
                    final_net = _calc_net_pnl(trade["side"], entry, exit_p, qty, comm)
                    notional = entry * qty
                    pnl_at_close = (final_net / notional * 100) if notional else 0.0
                else:
                    pnl_at_close = 0.0

                lesson, lesson_type = self._derive_lesson(
                    decision, pnl_at_decision, pnl_at_close, symbol,
                )

                await self._own_db.execute(
                    """INSERT INTO lessons
                       (trade_id, instance, symbol, lesson, lesson_type,
                        pnl_at_decision, pnl_at_close, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (trade_id, instance, symbol, lesson, lesson_type,
                     pnl_at_decision, pnl_at_close,
                     datetime.utcnow().isoformat()),
                )
                await self._own_db.commit()
                logger.info(
                    "Lesson [%s] %s: decision_pnl=%.2f%% close_pnl=%.2f%% — %s",
                    lesson_type, symbol, pnl_at_decision, pnl_at_close, lesson,
                )

    @staticmethod
    def _derive_lesson(decision: str, pnl_at_dec: float,
                       pnl_at_close: float, symbol: str) -> tuple[str, str]:
        diff = pnl_at_close - pnl_at_dec

        if decision == "CLOSE":
            if diff > 0.3:
                # Closed too early — price went further in profit
                return (
                    f"{symbol}: закрыл при {pnl_at_dec:.2f}%, но цена дошла до "
                    f"{pnl_at_close:.2f}% — потерял {diff:.2f}% прибыли. "
                    f"Не спеши закрывать при сильном тренде.",
                    "early_close",
                )
            else:
                # Good close — price reversed or stayed
                return (
                    f"{symbol}: закрыл при {pnl_at_dec:.2f}%, финал "
                    f"{pnl_at_close:.2f}% — правильное решение.",
                    "good_close",
                )
        else:  # HOLD
            if pnl_at_close < 0 and pnl_at_dec > 0:
                # Held but went negative — missed exit
                return (
                    f"{symbol}: держал при +{pnl_at_dec:.2f}%, финал "
                    f"{pnl_at_close:.2f}% — упущена прибыль, нужно было "
                    f"закрывать. Фиксируй при развороте.",
                    "missed_close",
                )
            elif pnl_at_close < pnl_at_dec - 0.3:
                # Held but lost significant profit
                return (
                    f"{symbol}: держал при +{pnl_at_dec:.2f}%, финал "
                    f"{pnl_at_close:.2f}% — потерял {abs(diff):.2f}% прибыли. "
                    f"Нужно было фиксировать раньше.",
                    "missed_close",
                )
            else:
                # Good hold — price went further or stayed positive
                return (
                    f"{symbol}: держал при {pnl_at_dec:.2f}%, финал "
                    f"{pnl_at_close:.2f}% — правильно держал.",
                    "good_hold",
                )

    async def _get_recent_lessons(self, limit: int = 10) -> list[str]:
        cursor = await self._own_db.execute(
            "SELECT lesson FROM lessons ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    # ── move SL / TP ─────────────────────────────────────────────

    async def _move_sl(self, pos: dict, new_sl: float, reasoning: str):
        symbol = pos["symbol"]
        side = pos["side"]
        old_sl = pos.get("stop_loss") or 0.0
        cur = pos["cur_price"]

        # Validate: SL moves only toward profit
        if side == "Buy":
            if old_sl and new_sl <= old_sl:
                logger.warning("MOVE_SL rejected %s: new_sl %.4f <= old %.4f", symbol, new_sl, old_sl)
                return
            if new_sl >= cur:
                logger.warning("MOVE_SL rejected %s: new_sl %.4f >= cur %.4f", symbol, new_sl, cur)
                return
        else:  # Sell
            if old_sl and new_sl >= old_sl:
                logger.warning("MOVE_SL rejected %s: new_sl %.4f >= old %.4f", symbol, new_sl, old_sl)
                return
            if new_sl <= cur:
                logger.warning("MOVE_SL rejected %s: new_sl %.4f <= cur %.4f", symbol, new_sl, cur)
                return

        db = self._dbs.get(pos["db_path"])
        if db:
            try:
                await db.update_trade(pos["id"], stop_loss=new_sl)
            except Exception:
                logger.exception("DB update_trade failed for %s SL", symbol)
                return

        logger.info(
            "Accountant moved SL %s %s: %.4f → %.4f (%s)",
            symbol, pos["instance"], old_sl, new_sl, reasoning,
        )
        await self._notify(
            f"\U0001f6e1 Бухгалтер двинул SL: {symbol} ({pos['instance']})\n"
            f"SL: {old_sl:.4f} → {new_sl:.4f}\n"
            f"Причина: {reasoning}"
        )

    async def _move_tp(self, pos: dict, new_tp: float, reasoning: str):
        symbol = pos["symbol"]
        side = pos["side"]
        cur = pos["cur_price"]

        # Validate: TP must be on the correct side of current price
        if side == "Buy" and new_tp <= cur:
            logger.warning("MOVE_TP rejected %s: new_tp %.4f <= cur %.4f", symbol, new_tp, cur)
            return
        if side == "Sell" and new_tp >= cur:
            logger.warning("MOVE_TP rejected %s: new_tp %.4f >= cur %.4f", symbol, new_tp, cur)
            return

        db = self._dbs.get(pos["db_path"])
        if db:
            try:
                await db.update_trade(pos["id"], take_profit=new_tp)
            except Exception:
                logger.exception("DB update_trade failed for %s TP", symbol)
                return

        old_tp = pos.get("take_profit") or 0.0
        logger.info(
            "Accountant moved TP %s %s: %.4f → %.4f (%s)",
            symbol, pos["instance"], old_tp, new_tp, reasoning,
        )
        await self._notify(
            f"\U0001f3af Бухгалтер двинул TP: {symbol} ({pos['instance']})\n"
            f"TP: {old_tp:.4f} → {new_tp:.4f}\n"
            f"Причина: {reasoning}"
        )

    # ── close position ─────────────────────────────────────────

    async def _close_position(self, pos: dict, reasoning: str):
        symbol = pos["symbol"]
        side = pos["side"]
        qty = pos["qty"]
        close_side = "Sell" if side == "Buy" else "Buy"

        try:
            self.client.place_order(
                symbol, close_side, qty,
                category="linear", reduce_only=True,
            )
        except Exception:
            logger.exception("Failed to close %s", symbol)
            return

        cur_price = pos["cur_price"]
        net_pnl = _calc_net_pnl(
            side, pos["entry_price"], cur_price, qty, pos["commission_rate"],
        )

        db = self._dbs.get(pos["db_path"])
        if db:
            try:
                await db.close_trade(pos["id"], cur_price, net_pnl)
                await db.update_daily_pnl(net_pnl)
            except Exception:
                logger.exception("DB update failed for %s", symbol)

        logger.info(
            "Accountant closed %s %s: PnL=%.2f$ (%s)",
            symbol, pos["instance"], net_pnl, reasoning,
        )

        await self._notify(
            f"\U0001f4b0 Бухгалтер зафиксировал: {symbol} ({pos['instance']})\n"
            f"PnL: {'+' if net_pnl >= 0 else ''}{net_pnl:.2f}$ "
            f"({pos['net_pnl_pct']:.2f}%)\n"
            f"Причина: {reasoning}"
        )

    # ── telegram ───────────────────────────────────────────────

    async def _notify(self, text: str):
        if not self.tg_token or not self.tg_chat:
            return
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        try:
            await self._http.post(
                url,
                json={"chat_id": self.tg_chat, "text": text},
                timeout=10,
            )
        except Exception:
            logger.warning("Telegram notification failed")


# ── Entry point ────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def amain(config_path: str):
    config = load_config(config_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                Path(__file__).resolve().parent.parent / "accountant.log"
            ),
        ],
    )

    acct = Accountant(config)

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _sig():
        logger.info("Shutdown signal received")
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _sig)

    task = asyncio.create_task(acct.run())
    await stop.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await acct.stop()
    logger.info("Accountant shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accountant — AI Profit-Taking Bot")
    parser.add_argument("--config", "-c", default="config/accountant.yaml")
    args = parser.parse_args()
    asyncio.run(amain(args.config))
