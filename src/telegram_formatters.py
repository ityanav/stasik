"""Telegram bot formatters — dashboard and positions text output."""

from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.telegram_data import (
    INSTANCES,
    INSTANCE_ICONS,
    query_db,
    is_service_active,
    enrich_pnl,
    get_bybit_balance,
    get_tbank_balance,
    get_instance_daily_stats,
)


def format_dashboard() -> str:
    lines = ["━━━ 📊 DASHBOARD ━━━\n"]

    # ── Balances ──
    bybit_balance = get_bybit_balance()
    tbank_balance = get_tbank_balance()

    # ── Daily PnL per instance ──
    today = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d")
    total_usdt = 0.0
    total_rub = 0.0
    total_wins = 0
    total_losses = 0
    total_open = 0
    inst_lines = []

    for inst in INSTANCES:
        active = is_service_active(inst["service"])
        icon = "🟢" if active else "🔴"
        cur = inst["currency"]
        short = inst["name"].replace("TBANK-", "TB-")

        if not inst["db"] or not Path(inst["db"]).exists():
            inst_lines.append(f"{icon} {short:9s} —")
            continue

        stats = get_instance_daily_stats(inst, today)
        total_open += stats["open_cnt"]

        if not active:
            inst_lines.append(f"{icon} {short:9s} стоп")
            continue

        day_pnl = stats["day_pnl"]
        wins = stats["wins"]
        losses = stats["losses"]
        total_wins += wins
        total_losses += losses

        if cur == "USDT":
            total_usdt += day_pnl
        else:
            total_rub += day_pnl

        cur_sym = "$" if cur == "USDT" else "₽"
        sign = "+" if day_pnl >= 0 else ""
        pos_str = f"{stats['open_cnt']} поз" if stats["open_cnt"] > 0 else "0 поз"
        pnl_str = f"{sign}{day_pnl:,.2f}{cur_sym}" if cur == "USDT" else f"{sign}{day_pnl:,.0f}{cur_sym}"
        inst_lines.append(f"{icon} {short:9s} {pos_str} │ {pnl_str}")

    # Bybit daily PnL
    bybit_sign = "+" if total_usdt >= 0 else ""
    lines.append(f"💰 Bybit: {bybit_balance:,.0f} USDT ({bybit_sign}{total_usdt:,.0f}$)")
    lines.append(f"💰 TBank: {tbank_balance:,.0f} RUB ({'+' if total_rub >= 0 else ''}{total_rub:,.0f}₽)")

    lines.append("\n── Боты ──")
    lines.extend(inst_lines)

    # ── Totals ──
    total_trades = total_wins + total_losses
    lines.append("\n── Итого сегодня ──")
    parts = []
    if total_usdt != 0 or any(i["currency"] == "USDT" for i in INSTANCES):
        parts.append(f"USDT: {'+' if total_usdt >= 0 else ''}{total_usdt:,.2f}")
    if total_rub != 0 or any(i["currency"] == "RUB" for i in INSTANCES):
        parts.append(f"RUB: {'+' if total_rub >= 0 else ''}{total_rub:,.0f}")
    lines.append("  │  ".join(parts))

    if total_trades > 0:
        wr = total_wins / total_trades * 100
        lines.append(f"WR: {wr:.1f}% ({total_wins}W/{total_losses}L/{total_open}O)")
    elif total_open > 0:
        lines.append(f"Открыто: {total_open}")

    msk = datetime.now(timezone(timedelta(hours=3)))
    lines.append(f"\n🕐 {msk.strftime('%H:%M')} MSK")
    lines.append("━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


def format_positions() -> tuple[str, list[dict]]:
    all_positions = []
    for inst in INSTANCES:
        if not inst["db"] or not Path(inst["db"]).exists():
            continue
        rows = query_db(
            inst["db"],
            "SELECT id, symbol, side, entry_price, qty, pnl, opened_at FROM trades WHERE status='open' ORDER BY opened_at DESC",
        )
        for r in rows:
            r["instance"] = inst["name"]
            r["currency"] = inst["currency"]
            r["exchange"] = inst["exchange"]
            all_positions.append(r)

    if not all_positions:
        return "📈 Нет открытых позиций.", []

    enrich_pnl(all_positions)

    lines = [f"━━ 📈 Позиции ({len(all_positions)}) ━━\n"]

    # Group by instance
    grouped = OrderedDict()
    for p in all_positions:
        grouped.setdefault(p["instance"], []).append(p)

    net_usdt = 0.0
    net_rub = 0.0
    for inst_name, positions in grouped.items():
        icon = INSTANCE_ICONS.get(inst_name, "📊")
        short = inst_name.replace("TBANK-", "TB-")
        lines.append(f"{icon} {short}")
        for p in positions:
            arrow = "↑" if p["side"] == "Buy" else "↓"
            direction = "LONG" if p["side"] == "Buy" else "SHORT"
            pnl = p.get("net_pnl", 0)
            sign = "+" if pnl >= 0 else ""
            cur_sym = "$" if p["currency"] == "USDT" else "₽"
            pnl_str = f"{sign}{pnl:,.2f}{cur_sym}" if p["currency"] == "USDT" else f"{sign}{pnl:,.0f}{cur_sym}"
            lines.append(f"  {arrow} {direction} {p['symbol']}  {pnl_str}")
            if p["currency"] == "USDT":
                net_usdt += pnl
            else:
                net_rub += pnl
        lines.append("")

    # Net totals
    parts = []
    if net_usdt != 0:
        parts.append(f"{'+' if net_usdt >= 0 else ''}{net_usdt:,.2f}$")
    if net_rub != 0:
        parts.append(f"{'+' if net_rub >= 0 else ''}{net_rub:,.0f}₽")
    if parts:
        lines.append(f"Net: {' | '.join(parts)}")

    return "\n".join(lines), all_positions
