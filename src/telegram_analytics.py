"""Telegram Аналитик — AI analysis of trade signal scores."""

import json
import logging
import sqlite3
from pathlib import Path

import httpx
import yaml

from src.telegram_data import INSTANCES

logger = logging.getLogger(__name__)


def get_all_trades_with_scores() -> list[dict]:
    """Collect all closed trades with signal scores from all instance DBs."""
    trades = []
    for inst in INSTANCES:
        db_path = inst["db"]
        if not db_path or not Path(db_path).exists():
            continue
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    t.id, t.symbol, t.side, t.entry_price, t.exit_price,
                    t.qty, t.pnl, t.opened_at, t.closed_at,
                    s.score, s.details, s.net_pnl as score_pnl
                FROM trades t
                LEFT JOIN signal_scores s ON s.trade_id = t.id
                WHERE t.status = 'closed'
                ORDER BY t.closed_at DESC
            """).fetchall()
            conn.close()
            for r in rows:
                row = dict(r)
                row["instance"] = inst["name"]
                row["currency"] = inst["currency"]
                # Parse details JSON
                if row.get("details"):
                    try:
                        row["details"] = json.loads(row["details"])
                    except Exception:
                        row["details"] = None
                trades.append(row)
        except Exception as e:
            logger.warning("Analytics DB query failed (%s): %s", db_path, e)
    return trades


def _get_deepseek_key() -> str:
    """Get DeepSeek API key from fiba config."""
    cfg_path = "/root/stasik/config/fiba.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["ai"]["api_key"]


def analyze_trades(trades: list[dict]) -> str:
    """Send trades to DeepSeek for analysis, return text report."""
    if not trades:
        return "Нет закрытых сделок для анализа."

    # Prepare compact data for AI
    data = []
    for t in trades:
        entry = {
            "inst": t["instance"],
            "symbol": t["symbol"],
            "side": t["side"],
            "pnl": t.get("score_pnl") or t.get("pnl") or 0,
            "cur": t["currency"],
        }
        if t.get("score") is not None:
            entry["score"] = t["score"]
        if t.get("details"):
            # Only non-zero indicators
            active = {k: v for k, v in t["details"].items()
                      if isinstance(v, (int, float)) and v != 0}
            if active:
                entry["indicators"] = active
        if t.get("opened_at"):
            entry["opened"] = t["opened_at"][:16]
        data.append(entry)

    total = len(data)
    with_scores = sum(1 for d in data if "score" in d)

    system_prompt = (
        "Ты — аналитик торговых стратегий. Тебе дана база сделок с индикаторами входа и результатом (pnl).\n"
        "Проанализируй данные и ответь:\n"
        "1. Какие комбинации индикаторов чаще дают прибыль?\n"
        "2. Какие комбинации убыточны?\n"
        "3. Какой минимальный score оптимален?\n"
        "4. Какие индикаторы бесполезны (не влияют на результат)?\n"
        "5. Конкретные рекомендации по улучшению стратегии.\n\n"
        "Отвечай кратко, по делу, с цифрами. На русском. Без markdown заголовков."
    )

    user_prompt = (
        f"Всего сделок: {total}, из них с разбивкой по индикаторам: {with_scores}\n\n"
        f"Данные (JSON):\n{json.dumps(data, ensure_ascii=False, indent=None)}"
    )

    try:
        api_key = _get_deepseek_key()
        resp = httpx.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 2000,
            },
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        text = result["choices"][0]["message"]["content"]
        return f"🔬 Аналитик ({total} сделок)\n\n{text}"
    except httpx.TimeoutException:
        return "❌ DeepSeek не ответил (таймаут 30с)"
    except Exception as e:
        logger.exception("Analytics AI error")
        return f"❌ Ошибка анализа: {e}"
