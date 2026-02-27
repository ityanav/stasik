import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportingMixin:
    """Mixin providing reporting, notifications, and status methods."""

    async def _ai_review_strategy(self):
        recent = await self.db.get_recent_trades(20)
        closed = [t for t in recent if t.get("status") == "closed"]

        if len(closed) < 3:
            logger.info("AI review: too few closed trades (%d), skipping", len(closed))
            return

        update = await self.ai_analyst.review_strategy(
            strategy_config=self.config["strategy"],
            risk_config=self.config["risk"],
            recent_trades=closed,
            market_bias=self._market_bias,
            lessons=self._ai_lessons,
        )

        if update.error:
            logger.warning("AI review failed: %s", update.error)
            return

        # Save lessons from AI (replace old with new, max 5)
        if update.lessons:
            self._ai_lessons = update.lessons[:5]
            logger.info("AI lessons updated: %s", self._ai_lessons)

        if not update.changes:
            logger.info("AI review: no changes needed. Lessons: %d", len(self._ai_lessons))
            return

        # Apply changes
        strategy_keys = {"rsi_oversold", "rsi_overbought", "ema_fast", "ema_slow",
                         "bb_period", "bb_std", "vol_threshold", "min_score"}
        risk_keys = {"stop_loss", "take_profit", "risk_per_trade"}

        changes_text = []
        for key, value in update.changes.items():
            if key in strategy_keys:
                old = self.config["strategy"].get(key)
                self.config["strategy"][key] = value
                changes_text.append(f"  {key}: {old} → {value}")
            elif key in risk_keys:
                old = self.config["risk"].get(key)
                self.config["risk"][key] = value
                changes_text.append(f"  {key}: {old} → {value}")

        if not changes_text:
            return

        # Rebuild signal generator with new parameters
        self.signal_gen = self._create_signal_gen(self.config)
        # Update risk manager SL/TP if changed
        if "stop_loss" in update.changes:
            self.risk.stop_loss_pct = update.changes["stop_loss"] / 100
        if "take_profit" in update.changes:
            self.risk.take_profit_pct = update.changes["take_profit"] / 100
        if "risk_per_trade" in update.changes:
            self.risk.risk_per_trade = update.changes["risk_per_trade"] / 100

        msg = (
            f"🧠 AI пересмотрел стратегию\n"
            f"Изменения:\n" + "\n".join(changes_text) + "\n"
            f"Причина: {update.reasoning}"
        )
        if self._ai_lessons:
            msg += "\n\nУроки:\n" + "\n".join(f"  • {l}" for l in self._ai_lessons)
        logger.info(msg)
        await self._notify(msg)

    async def _notify(self, text: str):
        if self.notifier:
            try:
                if self.instance_name:
                    text = f"[{self.instance_name}] {text}"
                await self.notifier(text)
            except Exception:
                logger.exception("Failed to send notification")

    # ── Weekly report ─────────────────────────────────────────

    async def _maybe_weekly_report(self):
        """Send weekly report on configured day (default Monday) at first tick after 8:00 UTC."""
        from datetime import date
        today = date.today()
        if today.weekday() != self._weekly_report_day:
            return
        if datetime.utcnow().hour < 8:
            return
        today_str = today.isoformat()
        if self._last_weekly_report == today_str:
            return

        self._last_weekly_report = today_str
        await self._send_weekly_report()

    async def _send_weekly_report(self):
        """Generate and send weekly performance report."""
        import aiosqlite
        from pathlib import Path

        name = self.instance_name or "BOT"
        stats = await self.db.get_weekly_stats()

        if stats["total"] == 0:
            lines = [
                f"📊 Недельный отчёт [{name}]",
                "─────────────",
                "Сделок за неделю: 0",
                "Бот не торговал.",
            ]
        else:
            wr = stats["win_rate"]
            lines = [
                f"📊 Недельный отчёт [{name}]",
                "─────────────",
                f"PnL за неделю: {stats['weekly_pnl']:+,.2f} USDT",
                f"Сделок: {stats['total']} ({stats['wins']}W / {stats['losses']}L)",
                f"Win Rate: {wr:.1f}%",
                f"Средний PnL: {stats['avg_pnl']:+,.2f} USDT",
            ]

            if stats["best_trade"]:
                bt = stats["best_trade"]
                d = "ЛОНГ" if bt["side"] == "Buy" else "ШОРТ"
                lines.append(f"🏆 Лучшая: {d} {bt['symbol']} → +{stats['best']:,.2f} USDT")

            if stats["worst_trade"]:
                wt = stats["worst_trade"]
                d = "ЛОНГ" if wt["side"] == "Buy" else "ШОРТ"
                lines.append(f"💀 Худшая: {d} {wt['symbol']} → {stats['worst']:,.2f} USDT")

        # Other instances
        for inst in self.config.get("other_instances", []):
            inst_name = inst.get("name", "???")
            db_path = inst.get("db_path", "")
            if not db_path or not Path(db_path).exists():
                continue
            try:
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    from datetime import date as dt_date, timedelta
                    week_ago = (dt_date.today() - timedelta(days=7)).isoformat()

                    cur = await db.execute(
                        "SELECT COALESCE(SUM(pnl), 0) as pnl FROM daily_pnl WHERE trade_date >= ?",
                        (week_ago,),
                    )
                    row = await cur.fetchone()
                    inst_pnl = float(row["pnl"])

                    cur = await db.execute(
                        "SELECT COUNT(*) as total, "
                        "COALESCE(SUM(CASE WHEN pnl >= 0 THEN 1 ELSE 0 END), 0) as wins, "
                        "COALESCE(SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END), 0) as losses "
                        "FROM trades WHERE status = 'closed' AND closed_at >= ?",
                        (week_ago,),
                    )
                    row = await cur.fetchone()
                    i_total = int(row["total"])
                    i_wins = int(row["wins"])
                    i_losses = int(row["losses"])

                    lines.append(f"\n📊 [{inst_name}] за неделю")
                    lines.append(f"PnL: {inst_pnl:+,.2f} USDT")
                    if i_total > 0:
                        i_wr = i_wins / i_total * 100
                        lines.append(f"Сделок: {i_total} ({i_wins}W / {i_losses}L) WR: {i_wr:.1f}%")
                    else:
                        lines.append("Сделок: 0")
            except Exception:
                logger.warning("Failed to read weekly stats for %s", inst_name)

        msg = "\n".join(lines)
        logger.info(msg)
        await self._notify(msg)

    async def _format_balance_block(self) -> str:
        """Format full balance block with accounts, yesterday, today, current."""
        balance = self.client.get_balance()
        daily_map = await self._get_all_daily_pnl()
        daily_total = sum(daily_map.values())

        yesterday = balance - daily_total
        arrow = "📈" if daily_total >= 0 else "📉"

        # Breakdown by bot: [SCALP]: -1,748 | [SWING]: +0
        parts = []
        for name, pnl in daily_map.items():
            parts.append(f"[{name}]: {pnl:+,.0f}")
        breakdown = " | ".join(parts)

        lines = [
            f"💰 Счёт Bybit: {balance:,.0f} USDT",
        ]

        # T-Bank balance
        tbank_balance = self._get_tbank_balance()
        if tbank_balance is not None:
            lines.append(f"🏦 Счёт TBank: {tbank_balance:,.0f} RUB")

        lines.extend([
            f"📊 Вчера: {yesterday:,.0f} USDT",
            f"{arrow} Сегодня: {daily_total:+,.0f} USDT ({breakdown})",
        ])
        return "\n".join(lines)

    async def get_status(self) -> str:
        balance_block = await self._format_balance_block()
        open_trades = await self.db.get_open_trades()
        daily = await self.db.get_daily_pnl()
        total = await self.db.get_total_pnl()

        name = self.instance_name or "BOT"
        currency = "RUB" if self.exchange_type == "tbank" else "USDT"
        tf_label = self.timeframe if self._is_swing else f"{self.timeframe}м"

        lines = [
            balance_block,
            "",
            f"━━━ [{name}] ━━━",
            f"{'🟢 РАБОТАЕТ' if self._running else '🔴 ОСТАНОВЛЕН'}",
            f"{'⛔ СТОП — дневной лимит потерь' if self.risk.is_halted else ''}",
            f"Открытых сделок: {len(open_trades)}",
            f"За день: {daily:+,.2f} {currency}",
            f"Всего: {total:+,.2f} {currency}",
            f"Пары: {', '.join(self.pairs)}",
            f"Таймфрейм: {tf_label} | Плечо: {self.leverage}x",
        ]
        if self._market_bias_enabled and self.exchange_type == "bybit":
            bias_emoji = {"bearish": "🐻", "bullish": "🐂", "neutral": "➖"}.get(self._market_bias, "➖")
            lines.append(f"Рынок: {bias_emoji} {self._market_bias.upper()} (bias ±{self._bias_score_bonus})")

        # Show other instances
        for inst in self.config.get("other_instances", []):
            inst_lines = await self._get_other_instance_status(inst)
            if inst_lines:
                lines.append("")
                lines.extend(inst_lines)

        return "\n".join(line for line in lines if line)

    async def _get_other_instance_status(self, inst: dict) -> list[str]:
        """Read another instance's DB and show its status."""
        import subprocess
        from pathlib import Path

        name = inst.get("name", "???")
        db_path = inst.get("db_path", "")
        service = inst.get("service", "")
        tf = inst.get("timeframe", "?")
        leverage = inst.get("leverage", "?")
        pairs = inst.get("pairs", [])

        # Check if service is running
        running = False
        if service:
            try:
                result = subprocess.run(
                    ["systemctl", "is-active", service],
                    capture_output=True, text=True, timeout=3,
                )
                running = result.stdout.strip() == "active"
            except Exception:
                pass

        status_emoji = "🟢 РАБОТАЕТ" if running else "🔴 ОСТАНОВЛЕН"

        # Read PnL from DB
        daily = 0.0
        total = 0.0
        open_count = 0
        if db_path and Path(db_path).exists():
            try:
                import aiosqlite
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    # Daily PnL
                    from datetime import date
                    cur = await db.execute(
                        "SELECT pnl FROM daily_pnl WHERE trade_date = ?",
                        (date.today().isoformat(),),
                    )
                    row = await cur.fetchone()
                    if row:
                        daily = float(row["pnl"])
                    # Total PnL
                    cur = await db.execute("SELECT COALESCE(SUM(pnl), 0) as total FROM daily_pnl")
                    row = await cur.fetchone()
                    if row:
                        total = float(row["total"])
                    # Open trades
                    cur = await db.execute("SELECT COUNT(*) as cnt FROM trades WHERE status = 'open'")
                    row = await cur.fetchone()
                    if row:
                        open_count = int(row["cnt"])
            except Exception:
                logger.warning("Failed to read DB for instance %s", name)

        currency = "RUB" if "TBANK" in name.upper() else "USDT"
        lev_str = str(leverage)
        if not lev_str.endswith("x"):
            lev_str += "x"
        lines = [
            f"━━━ [{name}] ━━━",
            status_emoji,
            f"Открытых сделок: {open_count}",
            f"За день: {daily:+,.2f} {currency}",
            f"Всего: {total:+,.2f} {currency}",
            f"Пары: {', '.join(pairs)}",
            f"Таймфрейм: {tf} | Плечо: {lev_str}",
        ]
        return lines

    async def get_open_positions_list(self) -> list[dict]:
        """Return list of open positions with unrealized PnL for Telegram buttons."""
        import aiosqlite
        from pathlib import Path

        categories = self._get_categories()
        result = []

        # Current engine positions — from own DB, enriched with live price
        own_trades = await self.db.get_open_trades()
        for t in own_trades:
            sym = t["symbol"]
            entry = t["entry_price"]
            qty_val = t["qty"]
            side = t["side"]
            upnl = 0.0
            try:
                if self.exchange_type == "tbank":
                    cur_price = self.client.get_last_price(sym)
                else:
                    cur_price = self.client.get_last_price(sym, category="linear")
                upnl = self._calc_net_pnl(side, entry, cur_price, qty_val)
            except Exception:
                pass
            result.append({
                "symbol": sym,
                "side": side,
                "size": qty_val,
                "entry_price": entry,
                "upnl": upnl,
                "category": t["category"],
                "instance": self.instance_name or "BOT",
            })

        # Other instances — open trades from DB
        for inst in self.config.get("other_instances", []):
            inst_name = inst.get("name", "???")
            db_path = inst.get("db_path", "")
            is_tbank = "TBANK" in inst_name.upper()
            if db_path and Path(db_path).exists():
                try:
                    async with aiosqlite.connect(db_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute(
                            "SELECT symbol, side, entry_price, qty FROM trades WHERE status = 'open'"
                        )
                        rows = await cur.fetchall()
                        if rows:
                            symbols = [r["symbol"] for r in rows]
                            prices = self._get_other_instance_prices(symbols, is_tbank)
                            for r in rows:
                                sym = r["symbol"]
                                entry = r["entry_price"]
                                qty_val = r["qty"]
                                cur_price = prices.get(sym)
                                upnl = 0.0
                                if cur_price and entry > 0 and qty_val > 0:
                                    upnl = self._calc_net_pnl(r["side"], entry, cur_price, qty_val)
                                result.append({
                                    "symbol": sym,
                                    "side": r["side"],
                                    "size": qty_val,
                                    "entry_price": entry,
                                    "upnl": upnl,
                                    "category": "tbank" if is_tbank else "linear",
                                    "instance": inst_name,
                                })
                except Exception:
                    logger.warning("Failed to read positions from %s", inst_name, exc_info=True)

        return result

    async def get_pairs_text(self) -> str:
        """Show pairs with total closed PnL and open position status."""
        import aiosqlite
        from pathlib import Path

        currency = "RUB" if self.exchange_type == "tbank" else "USDT"
        name = self.instance_name or "BOT"

        # Get closed PnL per pair from main DB
        pair_pnl: dict[str, float] = {}
        pair_trades: dict[str, int] = {}
        try:
            db_path = self.db.db_path if hasattr(self.db, 'db_path') else self.config.get("database", {}).get("path", "")
            if db_path:
                async with aiosqlite.connect(str(db_path)) as db:
                    db.row_factory = aiosqlite.Row
                    cur = await db.execute(
                        "SELECT symbol, COALESCE(SUM(pnl), 0) as total, COUNT(*) as cnt "
                        "FROM trades WHERE status = 'closed' GROUP BY symbol"
                    )
                    for r in await cur.fetchall():
                        pair_pnl[r["symbol"]] = float(r["total"])
                        pair_trades[r["symbol"]] = int(r["cnt"])
        except Exception:
            pass

        # Open positions
        open_symbols = set()
        try:
            open_trades = await self.db.get_open_trades()
            for t in open_trades:
                open_symbols.add(t["symbol"])
        except Exception:
            pass

        lines = [f"🪙 Пары [{name}]:"]
        for pair in self.pairs:
            pnl = pair_pnl.get(pair, 0)
            cnt = pair_trades.get(pair, 0)
            emoji = "🟢" if pnl >= 0 else "🔴"
            open_mark = " 📍" if pair in open_symbols else ""
            if cnt > 0:
                lines.append(f"  {emoji} {pair}: {pnl:+,.0f} {currency} ({cnt} сделок){open_mark}")
            else:
                lines.append(f"  ⚪ {pair}: нет сделок{open_mark}")

        # Other instances
        for inst in self.config.get("other_instances", []):
            inst_name = inst.get("name", "???")
            db_path = inst.get("db_path", "")
            is_tbank = "TBANK" in inst_name.upper()
            inst_currency = "RUB" if is_tbank else "USDT"
            inst_pairs = inst.get("pairs", [])
            if not db_path or not Path(db_path).exists() or not inst_pairs:
                continue
            try:
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    cur = await db.execute(
                        "SELECT symbol, COALESCE(SUM(pnl), 0) as total, COUNT(*) as cnt "
                        "FROM trades WHERE status = 'closed' GROUP BY symbol"
                    )
                    inst_pnl = {r["symbol"]: (float(r["total"]), int(r["cnt"])) for r in await cur.fetchall()}
                    cur2 = await db.execute("SELECT DISTINCT symbol FROM trades WHERE status = 'open'")
                    inst_open = {r["symbol"] for r in await cur2.fetchall()}
            except Exception:
                inst_pnl = {}
                inst_open = set()

            lines.append(f"\n🪙 Пары [{inst_name}]:")
            for pair in inst_pairs:
                pnl_val, cnt = inst_pnl.get(pair, (0, 0))
                emoji = "🟢" if pnl_val >= 0 else "🔴"
                open_mark = " 📍" if pair in inst_open else ""
                if cnt > 0:
                    lines.append(f"  {emoji} {pair}: {pnl_val:+,.0f} {inst_currency} ({cnt} сделок){open_mark}")
                else:
                    lines.append(f"  ⚪ {pair}: нет сделок{open_mark}")

        return "\n".join(lines)
