#!/usr/bin/env python3
"""Backfill signal_scores from journalctl logs for all instances."""

import json
import re
import sqlite3
import subprocess
from datetime import datetime, timedelta

INSTANCES = {
    "FIBA": {"service": "stasik-fiba", "db": "/root/stasik/data/fiba.db"},
    "BUBA": {"service": "stasik-buba", "db": "/root/stasik/data/buba.db"},
    "TBANK-SCALP": {"service": "stasik-tbank-scalp", "db": "/root/stasik/data/tbank_scalp.db"},
    "TBANK-SWING": {"service": "stasik-tbank-swing", "db": "/root/stasik/data/tbank_swing.db"},
    "MIDAS": {"service": "stasik-midas", "db": "/root/stasik/data/midas.db"},
}

# SMC format: "FIBA BTCUSDT: BUY (score=3, mom=1, fib=0[], ...)"
SMC_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[INFO\] src\.strategy\.signals: FIBA (\w+): (BUY|SELL) \((.+)\)"
)

# Trend format: "Сигнал GAZP: BUY (score=2, ...details={'rsi': 0, ...})"
# or "Котегава SLVRUB_TOM: BUY (score=3, ...details={'sma_dev': 1, ...})"
TREND_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[INFO\] src\.strategy\.signals: (?:Сигнал|Котегава|Momentum) (\S+): (BUY|SELL) \((.+)\)"
)

SCORE_RE = re.compile(r"(\w+)=(-?\d+)")
DETAILS_DICT_RE = re.compile(r"details=(\{[^}]+\})")


def parse_smc_line(line: str) -> dict | None:
    m = SMC_RE.search(line)
    if not m:
        return None
    ts_str, symbol, direction, details_str = m.groups()
    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    scores = {}
    for km in SCORE_RE.finditer(details_str):
        key, val = km.group(1), int(km.group(2))
        scores[key] = val
    total_score = scores.pop("score", 0)
    return {"timestamp": ts, "symbol": symbol, "direction": direction, "score": total_score, "details": scores}


def parse_trend_line(line: str) -> dict | None:
    m = TREND_RE.search(line)
    if not m:
        return None
    ts_str, symbol, direction, details_str = m.groups()
    if symbol == "?":
        return None  # no symbol info
    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")

    # Extract score
    score_m = re.search(r"score=(-?\d+)", details_str)
    total_score = int(score_m.group(1)) if score_m else 0

    # Try to parse details dict
    scores = {}
    dict_m = DETAILS_DICT_RE.search(details_str)
    if dict_m:
        try:
            d = eval(dict_m.group(1))  # safe: only from our own logs
            scores = {k: v for k, v in d.items() if isinstance(v, (int, float))}
        except Exception:
            pass

    return {"timestamp": ts, "symbol": symbol, "direction": direction, "score": total_score, "details": scores}


def get_signals_from_logs(service: str, instance: str) -> list[dict]:
    result = subprocess.run(
        ["journalctl", "-u", service, "--no-pager", "-o", "cat"],
        capture_output=True, text=True, timeout=30,
    )
    signals = []
    is_smc = instance in ("FIBA", "BUBA")
    for line in result.stdout.splitlines():
        if ": BUY (" not in line and ": SELL (" not in line:
            continue
        parsed = parse_smc_line(line) if is_smc else parse_trend_line(line)
        if parsed:
            signals.append(parsed)
    return signals


def match_trade_to_signal(trade: dict, signals: list[dict]) -> dict | None:
    trade_time = datetime.strptime(trade["opened_at"][:19], "%Y-%m-%dT%H:%M:%S")
    trade_symbol = trade["symbol"]
    trade_side = "BUY" if trade["side"] == "Buy" else "SELL"

    best = None
    best_diff = timedelta(seconds=120)

    for sig in signals:
        if sig["symbol"] != trade_symbol or sig["direction"] != trade_side:
            continue
        diff = trade_time - sig["timestamp"]
        if timedelta(seconds=-5) <= diff <= best_diff:
            if best is None or diff < (trade_time - best["timestamp"]):
                best = sig
                best_diff = diff
    return best


def backfill_instance(name: str, config: dict):
    db_path = config["db"]
    service = config["service"]

    print(f"\n{'='*40}")
    print(f"Instance: {name} ({service})")

    signals = get_signals_from_logs(service, name)
    print(f"  Signals in logs: {len(signals)}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    trades = [dict(r) for r in conn.execute(
        "SELECT id, symbol, side, entry_price, exit_price, pnl, opened_at, closed_at "
        "FROM trades WHERE status='closed' ORDER BY id"
    ).fetchall()]

    existing = {r[0] for r in conn.execute("SELECT trade_id FROM signal_scores").fetchall()}

    print(f"  Closed trades: {len(trades)}")
    print(f"  Already have scores: {len(existing)}")

    matched = 0
    skipped = 0
    for trade in trades:
        if trade["id"] in existing:
            skipped += 1
            continue
        signal = match_trade_to_signal(trade, signals)
        if signal:
            details_json = json.dumps(signal["details"], ensure_ascii=False)
            conn.execute(
                "INSERT INTO signal_scores (trade_id, score, details, net_pnl, created_at) VALUES (?, ?, ?, ?, ?)",
                (trade["id"], signal["score"], details_json, trade["pnl"], trade["opened_at"]),
            )
            matched += 1

    conn.commit()
    conn.close()

    print(f"  Matched & inserted: {matched}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  No match found: {len(trades) - matched - skipped}")


def main():
    print("Backfilling signal_scores from journalctl logs...")
    for name, config in INSTANCES.items():
        try:
            backfill_instance(name, config)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'='*40}")
    print("SUMMARY:")
    for name, config in INSTANCES.items():
        conn = sqlite3.connect(config["db"])
        total = conn.execute("SELECT COUNT(*) FROM signal_scores").fetchone()[0]
        conn.close()
        print(f"  {name}: {total} signal_scores records")
    print("Done!")


if __name__ == "__main__":
    main()
