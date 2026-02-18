import asyncio
import logging
from datetime import datetime

from src.exchange.client import BybitClient
from src.risk.manager import RiskManager
from src.storage.database import Database
from src.strategy.ai_analyst import AIAnalyst, extract_indicator_values, format_risk_text, summarize_candles
from src.strategy.signals import Signal, SignalGenerator

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, config: dict, notifier=None):
        self.config = config
        self.client = BybitClient(config)
        self.signal_gen = SignalGenerator(config)
        self.risk = RiskManager(config)
        self.db = Database()
        self.notifier = notifier  # async callable(text)
        self.ai_analyst = AIAnalyst.from_config(config)

        self.pairs: list[str] = config["trading"]["pairs"]
        self.timeframe: str = str(config["trading"]["timeframe"])
        self.market_type: str = config["trading"]["market_type"]
        self.leverage: int = config["trading"].get("leverage", 1)

        self._running = False
        self._instrument_cache: dict[str, dict] = {}
        self._tick_count: int = 0

    async def start(self):
        await self.db.connect()
        self._running = True

        # Set leverage for futures pairs
        if self.market_type in ("futures", "both"):
            for pair in self.pairs:
                self.client.set_leverage(pair, self.leverage, category="linear")

        await self._notify("🚀 Бот запущен\nПары: " + ", ".join(self.pairs))
        logger.info("Trading engine started")

        try:
            await self._run_loop()
        except asyncio.CancelledError:
            logger.info("Trading engine cancelled")
        finally:
            await self.ai_analyst.close()
            await self.db.close()
            await self._notify("🛑 Бот остановлен")
            logger.info("Trading engine stopped")

    async def stop(self):
        self._running = False

    async def _run_loop(self):
        interval_sec = int(self.timeframe) * 60
        review_every = self.ai_analyst.review_interval  # in ticks (minutes)
        while self._running:
            try:
                await self._check_closed_trades()
                await self._tick()
                self._tick_count += 1
                if self.ai_analyst.enabled and review_every > 0 and self._tick_count % review_every == 0:
                    await self._ai_review_strategy()
            except Exception:
                logger.exception("Error in trading tick")
            await asyncio.sleep(interval_sec)

    async def _tick(self):
        if self.risk.is_halted:
            logger.info("Trading halted — daily loss limit")
            return

        categories = self._get_categories()

        for pair in self.pairs:
            for category in categories:
                try:
                    await self._process_pair(pair, category)
                except Exception:
                    logger.exception("Error processing %s (%s)", pair, category)

    def _get_categories(self) -> list[str]:
        mt = self.market_type
        if mt == "spot":
            return ["spot"]
        elif mt == "futures":
            return ["linear"]
        else:
            return ["spot", "linear"]

    async def _process_pair(self, symbol: str, category: str):
        df = self.client.get_klines(
            symbol=symbol, interval=self.timeframe, limit=200, category=category
        )
        if len(df) < 50:
            logger.warning("Not enough data for %s (%s): %d candles", symbol, category, len(df))
            return

        result = self.signal_gen.generate(df)

        if result.signal == Signal.HOLD:
            return

        # Check existing positions
        open_trades = await self.db.get_open_trades()
        symbol_open = [t for t in open_trades if t["symbol"] == symbol and t["category"] == category]

        if symbol_open:
            logger.debug("Already have open trade for %s (%s), skipping", symbol, category)
            return

        open_count = len(open_trades)
        if not self.risk.can_open_position(open_count):
            return

        side = "Buy" if result.signal == Signal.BUY else "Sell"

        # Spot: only Buy (no short selling)
        if category == "spot" and side == "Sell":
            return

        # AI analyst filter
        ai_reasoning = ""
        ai_sl = None
        ai_tp = None
        ai_size_mult = None
        if self.ai_analyst.enabled:
            indicator_text = extract_indicator_values(df, self.config)
            candles_text = summarize_candles(df)
            risk_text = format_risk_text(self.config)
            verdict = await self.ai_analyst.analyze(
                signal=result.signal.value,
                score=result.score,
                details=result.details,
                indicator_text=indicator_text,
                candles_text=candles_text,
                risk_text=risk_text,
            )

            if verdict.error:
                # Fallback: AI unavailable — trade on technical signals
                logger.warning("AI unavailable (%s), using technical signals for %s", verdict.error, symbol)
            elif not verdict.confirmed:
                direction = "BUY" if side == "Buy" else "SELL"
                msg = (
                    f"🤖 AI отклонил {direction} {symbol}\n"
                    f"Уверенность: {verdict.confidence}/10\n"
                    f"Причина: {verdict.reasoning}"
                )
                logger.info(msg)
                await self._notify(msg)
                return
            else:
                ai_reasoning = f"🤖 AI ({verdict.confidence}/10): {verdict.reasoning}"
                ai_sl = verdict.stop_loss
                ai_tp = verdict.take_profit
                ai_size_mult = verdict.position_size
                logger.info("AI confirmed %s %s — confidence %d/10", side, symbol, verdict.confidence)

        await self._open_trade(
            symbol, side, category, result.score, result.details,
            ai_reasoning=ai_reasoning,
            ai_sl_pct=ai_sl,
            ai_tp_pct=ai_tp,
            ai_size_mult=ai_size_mult,
        )

    async def _open_trade(
        self, symbol: str, side: str, category: str, score: int, details: dict,
        ai_reasoning: str = "",
        ai_sl_pct: float | None = None,
        ai_tp_pct: float | None = None,
        ai_size_mult: float | None = None,
    ):
        price = self.client.get_last_price(symbol, category=category)
        balance = self.client.get_balance()

        info = self._get_instrument_info(symbol, category)
        qty = self.risk.calculate_position_size(
            balance=balance,
            price=price,
            qty_step=info["qty_step"],
            min_qty=info["min_qty"],
        )

        # AI position size multiplier
        if ai_size_mult is not None and 0.1 <= ai_size_mult <= 2.0:
            import math
            qty = math.floor((qty * ai_size_mult) / info["qty_step"]) * info["qty_step"]
            qty = round(qty, 8)

        if qty <= 0:
            logger.info("Position size too small for %s, skipping", symbol)
            return

        # AI-adjusted or default SL/TP
        if ai_sl_pct is not None and 0.3 <= ai_sl_pct <= 5.0:
            sl = price * (1 - ai_sl_pct / 100) if side == "Buy" else price * (1 + ai_sl_pct / 100)
            sl = round(sl, 6)
        else:
            ai_sl_pct = None
            sl = self.risk.calculate_sl_tp(price, side)[0]

        if ai_tp_pct is not None and 0.5 <= ai_tp_pct <= 10.0:
            tp = price * (1 + ai_tp_pct / 100) if side == "Buy" else price * (1 - ai_tp_pct / 100)
            tp = round(tp, 6)
        else:
            ai_tp_pct = None
            tp = self.risk.calculate_sl_tp(price, side)[1]

        order = self.client.place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            category=category,
            stop_loss=sl if category == "linear" else None,
            take_profit=tp if category == "linear" else None,
        )

        order_id = order.get("orderId", "")
        await self.db.insert_trade(
            symbol=symbol,
            side=side,
            category=category,
            qty=qty,
            entry_price=price,
            stop_loss=sl,
            take_profit=tp,
            order_id=order_id,
        )

        # Set trailing stop for futures
        trailing_msg = ""
        if category == "linear":
            trail_pct = self.config["risk"].get("trailing_stop", 0)
            trail_activation_pct = self.config["risk"].get("trailing_activation", 0)
            if trail_pct > 0:
                trailing_distance = price * trail_pct / 100
                if side == "Buy":
                    active_price = price * (1 + trail_activation_pct / 100)
                else:
                    active_price = price * (1 - trail_activation_pct / 100)
                self.client.set_trailing_stop(
                    symbol=symbol,
                    trailing_stop=trailing_distance,
                    active_price=active_price,
                )
                trailing_msg = f"\n📐 Trailing SL: {trail_pct}% (активация при {trail_activation_pct}% прибыли)"

        direction = "ЛОНГ 📈" if side == "Buy" else "ШОРТ 📉"
        pos_value = qty * price
        sl_note = f" (AI: {ai_sl_pct}%)" if ai_sl_pct else ""
        tp_note = f" (AI: {ai_tp_pct}%)" if ai_tp_pct else ""
        size_note = f" (AI: x{ai_size_mult})" if ai_size_mult else ""
        msg = (
            f"{'🟢' if side == 'Buy' else '🔴'} Открыл {direction} {symbol}\n"
            f"Цена: {price}\n"
            f"Объём: {qty}{size_note} (~{pos_value:,.0f} USDT)\n"
            f"SL: {sl}{sl_note} | TP: {tp}{tp_note}{trailing_msg}\n"
            f"Баланс: {balance:,.0f} USDT"
        )
        if ai_reasoning:
            msg += f"\n{ai_reasoning}"
        logger.info(msg)
        await self._notify(msg)

    # ── AI strategy review ──────────────────────────────────

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
        )

        if update.error:
            logger.warning("AI review failed: %s", update.error)
            return

        if not update.changes:
            logger.info("AI review: no changes needed")
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
        self.signal_gen = SignalGenerator(self.config)
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
        logger.info(msg)
        await self._notify(msg)

    # ── Monitor closed trades ─────────────────────────────────

    async def _check_closed_trades(self):
        open_trades = await self.db.get_open_trades()
        if not open_trades:
            return

        for trade in open_trades:
            if trade["category"] != "linear":
                continue
            try:
                await self._check_trade_closed(trade)
            except Exception:
                logger.exception("Error checking trade %s", trade["id"])

    async def _check_trade_closed(self, trade: dict):
        symbol = trade["symbol"]
        positions = self.client.get_positions(symbol=symbol, category="linear")

        # Check if we still have a position for this symbol+side
        still_open = False
        for p in positions:
            if p["symbol"] == symbol and p["size"] > 0:
                still_open = True
                break

        if still_open:
            return

        # Position closed — calculate PnL
        entry_price = trade["entry_price"]
        qty = trade["qty"]
        side = trade["side"]

        # Get current/last price as approximation of exit price
        exit_price = self.client.get_last_price(symbol, category="linear")

        # Try to get actual closed PnL from exchange
        try:
            pnl = self._get_closed_pnl(symbol, trade["order_id"])
        except Exception:
            # Fallback: calculate manually
            if side == "Buy":
                pnl = (exit_price - entry_price) * qty
            else:
                pnl = (entry_price - exit_price) * qty

        # Update DB
        balance = self.client.get_balance()
        await self.db.close_trade(trade["id"], exit_price, pnl)
        await self.db.update_daily_pnl(pnl)
        self.risk.record_pnl(pnl, balance)

        # Send notification
        if pnl >= 0:
            emoji = "💰"
            result_text = f"заработал: +{pnl:,.2f} USDT"
        else:
            emoji = "💸"
            result_text = f"просрал: {pnl:,.2f} USDT"

        direction = "ЛОНГ" if side == "Buy" else "ШОРТ"
        daily_pnl = await self.db.get_daily_pnl()
        total_pnl = await self.db.get_total_pnl()

        msg = (
            f"{emoji} Закрыл {direction} {symbol}\n"
            f"Вход: {entry_price} → Выход: {exit_price}\n"
            f"Результат: {result_text}\n"
            f"─────────────\n"
            f"За день: {daily_pnl:+,.2f} USDT\n"
            f"Всего: {total_pnl:+,.2f} USDT\n"
            f"Баланс: {balance:,.0f} USDT"
        )
        logger.info(msg)
        await self._notify(msg)

    def _get_closed_pnl(self, symbol: str, order_id: str) -> float:
        resp = self.session_closed_pnl(symbol)
        # This is a best-effort; fallback handles failures
        raise NotImplementedError

    def _get_instrument_info(self, symbol: str, category: str) -> dict:
        key = f"{symbol}_{category}"
        if key not in self._instrument_cache:
            self._instrument_cache[key] = self.client.get_instrument_info(symbol, category)
        return self._instrument_cache[key]

    async def _notify(self, text: str):
        if self.notifier:
            try:
                await self.notifier(text)
            except Exception:
                logger.exception("Failed to send notification")

    # ── Status info ──────────────────────────────────────────

    async def get_status(self) -> str:
        balance = self.client.get_balance()
        open_trades = await self.db.get_open_trades()
        daily = await self.db.get_daily_pnl()
        total = await self.db.get_total_pnl()

        lines = [
            f"{'🟢 РАБОТАЕТ' if self._running else '🔴 ОСТАНОВЛЕН'}",
            f"{'⛔ СТОП — дневной лимит потерь' if self.risk.is_halted else ''}",
            f"Баланс: {balance:,.2f} USDT",
            f"Открытых сделок: {len(open_trades)}",
            f"За день: {daily:+,.2f} USDT",
            f"Всего: {total:+,.2f} USDT",
            f"Пары: {', '.join(self.pairs)}",
            f"Таймфрейм: {self.timeframe}м | Плечо: {self.leverage}x",
        ]
        return "\n".join(line for line in lines if line)

    async def close_all_positions(self) -> str:
        categories = self._get_categories()
        closed = []
        for cat in categories:
            if cat != "linear":
                continue
            positions = self.client.get_positions(category=cat)
            for p in positions:
                try:
                    close_side = "Sell" if p["side"] == "Buy" else "Buy"
                    self.client.place_order(
                        symbol=p["symbol"],
                        side=close_side,
                        qty=p["size"],
                        category=cat,
                    )
                    closed.append(f"{p['symbol']} ({p['side']})")
                except Exception:
                    logger.exception("Failed to close position %s", p["symbol"])

        # Mark DB trades as closed
        open_trades = await self.db.get_open_trades()
        for t in open_trades:
            try:
                exit_price = self.client.get_last_price(t["symbol"], category=t["category"])
                if t["side"] == "Buy":
                    pnl = (exit_price - t["entry_price"]) * t["qty"]
                else:
                    pnl = (t["entry_price"] - exit_price) * t["qty"]
                await self.db.close_trade(t["id"], exit_price, pnl)
                await self.db.update_daily_pnl(pnl)
                balance = self.client.get_balance()
                self.risk.record_pnl(pnl, balance)
            except Exception:
                logger.exception("Failed to update DB for %s", t["symbol"])

        if closed:
            msg = f"❌ Закрыто {len(closed)} позиций:\n" + "\n".join(f"  • {c}" for c in closed)
        else:
            msg = "Нет позиций для закрытия."
        logger.info(msg)
        await self._notify(msg)
        return msg

    async def get_positions_text(self) -> str:
        categories = self._get_categories()
        lines = []
        total_pnl = 0.0
        for cat in categories:
            if cat == "linear":
                positions = self.client.get_positions(category=cat)
                for p in positions:
                    direction = "ЛОНГ" if p["side"] == "Buy" else "ШОРТ"
                    upnl = p["unrealised_pnl"]
                    total_pnl += upnl
                    entry = p["entry_price"]
                    # PnL в процентах от входа
                    if entry > 0 and p["size"] > 0:
                        pnl_pct = (upnl / (entry * p["size"])) * 100
                    else:
                        pnl_pct = 0.0
                    emoji = "🟢" if upnl >= 0 else "🔴"
                    lines.append(
                        f"{emoji} {direction} {p['symbol']}\n"
                        f"   Вход: {entry} | Объём: {p['size']}\n"
                        f"   PnL: {upnl:+,.2f} USDT ({pnl_pct:+.2f}%)"
                    )
        if lines:
            total_emoji = "🟢" if total_pnl >= 0 else "🔴"
            lines.append(f"\n{total_emoji} Итого PnL: {total_pnl:+,.2f} USDT")
        return "\n".join(lines) if lines else "Нет открытых позиций."

    async def get_pnl_text(self) -> str:
        daily = await self.db.get_daily_pnl()
        total = await self.db.get_total_pnl()
        recent = await self.db.get_recent_trades(5)
        lines = [
            f"За день: {daily:+,.2f} USDT",
            f"Всего: {total:+,.2f} USDT",
            "",
            "Последние сделки:",
        ]
        for t in recent:
            pnl = t.get("pnl") or 0
            direction = "ЛОНГ" if t["side"] == "Buy" else "ШОРТ"
            if pnl >= 0:
                result = f"+{pnl:,.2f}"
            else:
                result = f"{pnl:,.2f}"
            lines.append(
                f"  {direction} {t['symbol']} | {result} USDT | {t['status']}"
            )
        return "\n".join(lines)
