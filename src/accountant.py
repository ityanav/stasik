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
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
import yaml

from src.exchange.client import BybitClient
from src.storage.database import Database
from src.strategy.indicators import calculate_ema, calculate_rsi

logger = logging.getLogger(__name__)

# ── AI system prompt ───────────────────────────────────────────────

ACCOUNTANT_PROMPT = """\
Ты — бухгалтер трейдинг-бота. Решаешь: фиксировать прибыль СЕЙЧАС или держать.

Данные:
- Позиция: символ, LONG/SHORT, вход, текущая цена, PnL в % и $
- Время в сделке
- Расстояние до TP и SL в %
- RSI (14), EMA 9 vs 21 (тренд), объём
- Последние 10 свечей 15m

Правила:
- HOLD если тренд сильный и PnL растёт (EMA 9 > 21 для лонга)
- CLOSE если: RSI перекуплен (>70 для лонга, <30 для шорта) + momentum слабеет
- CLOSE если: цена была ближе к TP, теперь отдаляется (разворот)
- CLOSE если: объём падает и цена стоит на месте
- HOLD если PnL слишком мал (< 0.3%) — не стоит фиксировать
- Лучше зафиксировать +0.5% чем получить -1.5% по SL

JSON (без markdown): {"decision":"CLOSE"/"HOLD","confidence":1-10,"reasoning":"кратко"}\
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

    # ── lifecycle ──────────────────────────────────────────────

    async def start(self):
        self._http = httpx.AsyncClient(timeout=self.ai_timeout)
        for db_cfg in self.databases:
            path = db_cfg["path"]
            db = Database(Path(path), instance_name=db_cfg["instance"])
            await db.connect()
            self._dbs[path] = db
        instances = [d["instance"] for d in self.databases]
        logger.info("Accountant started: monitoring %s", instances)

    async def stop(self):
        for db in self._dbs.values():
            await db.close()
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

            if decision == "CLOSE" and confidence >= self.min_confidence:
                await self._close_position(pos, reasoning)

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

        user_msg = (
            f"Символ: {symbol}\n"
            f"Направление: {direction}\n"
            f"Вход: {entry:.4f}, Текущая: {cur:.4f}\n"
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
