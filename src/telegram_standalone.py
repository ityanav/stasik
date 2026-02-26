"""Standalone Telegram bot — reads all instance DBs directly, no TradingEngine."""

import asyncio
import logging
import signal
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

logger = logging.getLogger(__name__)

MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [
        [KeyboardButton("📊 Статус"), KeyboardButton("📈 Позиции")],
        [KeyboardButton("▶️ Старт"), KeyboardButton("🛑 Стоп")],
    ],
    resize_keyboard=True,
)

# Instance definitions: name, db_path, service, currency, config_path, exchange_type
INSTANCES = [
    {"name": "FIBA", "db": "/root/stasik/data/fiba.db", "service": "stasik-fiba", "currency": "USDT", "config": "/root/stasik/config/fiba.yaml", "exchange": "bybit"},
    {"name": "BUBA", "db": "/root/stasik/data/buba.db", "service": "stasik-buba", "currency": "USDT", "config": "/root/stasik/config/buba.yaml", "exchange": "bybit"},
    {"name": "TBANK-SCALP", "db": "/root/stasik/data/tbank_scalp.db", "service": "stasik-tbank-scalp", "currency": "RUB", "config": "/root/stasik/config/tbank_scalp.yaml", "exchange": "tbank"},
    {"name": "TBANK-SWING", "db": "/root/stasik/data/tbank_swing.db", "service": "stasik-tbank-swing", "currency": "RUB", "config": "/root/stasik/config/tbank_swing.yaml", "exchange": "tbank"},
    {"name": "MIDAS", "db": "/root/stasik/data/midas.db", "service": "stasik-midas", "currency": "RUB", "config": "/root/stasik/config/midas.yaml", "exchange": "tbank"},
    {"name": "FIN", "db": None, "service": "stasik-fin", "currency": None, "config": None, "exchange": None},
]


def _query_db(db_path: str, sql: str, params: tuple = ()) -> list:
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("DB query failed (%s): %s", db_path, e)
        return []


def _is_service_active(service: str) -> bool:
    try:
        r = subprocess.run(["systemctl", "is-active", service], capture_output=True, text=True, timeout=3)
        return r.stdout.strip() == "active"
    except Exception:
        return False


def _systemctl(action: str, service: str) -> bool:
    try:
        r = subprocess.run(["systemctl", action, service], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except Exception:
        logger.exception("systemctl %s %s failed", action, service)
        return False


def _get_status() -> str:
    lines = ["🤖 Stasik Status\n"]
    for inst in INSTANCES:
        active = _is_service_active(inst["service"])
        icon = "🟢" if active else "🔴"
        name = inst["name"]

        if inst["db"] and Path(inst["db"]).exists():
            rows = _query_db(inst["db"], "SELECT COUNT(*) as cnt FROM trades WHERE status='open'")
            open_cnt = rows[0]["cnt"] if rows else 0
            lines.append(f"{icon} {name}: {'active' if active else 'stopped'} | {open_cnt} open")
        else:
            lines.append(f"{icon} {name}: {'active' if active else 'stopped'}")
    return "\n".join(lines)


def _get_positions() -> tuple[str, list[dict]]:
    all_positions = []
    for inst in INSTANCES:
        if not inst["db"] or not Path(inst["db"]).exists():
            continue
        rows = _query_db(
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

    lines = [f"📈 Позиции ({len(all_positions)})\n"]
    for p in all_positions:
        d = "L" if p["side"] == "Buy" else "S"
        pnl = p.get("pnl") or 0
        sign = "+" if pnl >= 0 else ""
        lines.append(f"[{p['instance']}] {p['symbol']} {d} @ {p['entry_price']} | {sign}{pnl:,.2f} {p['currency']}")

    return "\n".join(lines), all_positions


def _close_bybit_position(config_path: str, symbol: str) -> str:
    """Close a Bybit position via API."""
    try:
        from pybit.unified_trading import HTTP

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        bybit_cfg = cfg["bybit"]
        http_kwargs = {
            "api_key": bybit_cfg["api_key"],
            "api_secret": bybit_cfg["api_secret"],
        }
        if bybit_cfg.get("demo"):
            http_kwargs["demo"] = True
        else:
            http_kwargs["testnet"] = bybit_cfg.get("testnet", False)

        session = HTTP(**http_kwargs)
        resp = session.get_positions(category="linear", symbol=symbol)
        for p in resp["result"]["list"]:
            size = float(p["size"])
            if size > 0:
                close_side = "Sell" if p["side"] == "Buy" else "Buy"
                session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=close_side,
                    orderType="Market",
                    qty=str(size),
                    reduceOnly=True,
                )
                return f"✅ {symbol} закрыт (market {close_side} {size})"
        return f"⚠️ {symbol} — позиция не найдена на бирже"
    except Exception as e:
        return f"❌ Ошибка закрытия {symbol}: {e}"


def _close_tbank_position(config_path: str, symbol: str) -> str:
    """Close a TBank position via API."""
    try:
        from src.exchange.tbank_client import TBankClient

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        client = TBankClient(cfg)
        positions = client.get_positions(symbol=symbol)
        for p in positions:
            if p["symbol"] == symbol and p["size"] > 0:
                close_side = "Sell" if p["side"] == "Buy" else "Buy"
                client.place_order(symbol=symbol, side=close_side, qty=p["size"])
                return f"✅ {symbol} закрыт (market {close_side} {p['size']})"
        return f"⚠️ {symbol} — позиция не найдена на бирже"
    except Exception as e:
        err_str = str(e)
        if "30079" in err_str or "not available for trading" in err_str.lower():
            return f"⏸ Биржа MOEX закрыта — {symbol} нельзя закрыть сейчас"
        return f"❌ Ошибка закрытия {symbol}: {e}"


def _update_db_closed(db_path: str, trade_id: int):
    """Mark trade as closed in DB."""
    try:
        conn = sqlite3.connect(db_path)
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("UPDATE trades SET status='closed', closed_at=? WHERE id=?", (now, trade_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("DB update failed for trade %s: %s", trade_id, e)


def _find_instance(name: str) -> Optional[dict]:
    for inst in INSTANCES:
        if inst["name"] == name:
            return inst
    return None


class StandaloneTelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = str(chat_id)
        self.app: Application | None = None
        self._started = False

    async def start(self):
        self.app = Application.builder().token(self.token).build()
        self._register_handlers()
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(
            drop_pending_updates=True,
            allowed_updates=["message", "callback_query"],
        )
        self._started = True
        logger.info("Standalone Telegram bot started")

    async def stop(self):
        if self._started and self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            self._started = False
            logger.info("Standalone Telegram bot stopped")

    def _register_handlers(self):
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("positions", self._cmd_positions))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_button))
        self.app.add_handler(CallbackQueryHandler(self._callback_handler))
        self.app.add_error_handler(self._error_handler)

    @staticmethod
    async def _error_handler(update, ctx: ContextTypes.DEFAULT_TYPE):
        logger.error("Telegram handler error: %s", ctx.error, exc_info=ctx.error)

    def _check_auth(self, update: Update) -> bool:
        return str(update.effective_chat.id) == self.chat_id

    # ── Button handler ──

    async def _handle_button(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        text = update.message.text.strip()
        handlers = {
            "📊 Статус": self._cmd_status,
            "📈 Позиции": self._cmd_positions,
            "▶️ Старт": self._cmd_run,
            "🛑 Стоп": self._cmd_stop,
        }
        handler = handlers.get(text)
        if handler:
            await handler(update, ctx)

    # ── Commands ──

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        await update.message.reply_text(
            "🤖 Stasik Trading Bot\n\n"
            "📊 Статус — состояние ботов\n"
            "📈 Позиции — открытые сделки + закрытие\n"
            "▶️ Старт — запустить бота\n"
            "🛑 Стоп — остановить бота",
            reply_markup=MAIN_KEYBOARD,
        )

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        await update.message.reply_text(_get_status(), reply_markup=MAIN_KEYBOARD)

    async def _cmd_positions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        text, positions = _get_positions()

        if positions:
            buttons = []
            for p in positions:
                d = "L" if p["side"] == "Buy" else "S"
                pnl = p.get("pnl") or 0
                sign = "+" if pnl >= 0 else ""
                label = f"❌ {p['instance']} {p['symbol']} {d} ({sign}{pnl:.2f})"
                cb_data = f"close_{p['instance']}_{p['id']}_{p['symbol']}"
                buttons.append([InlineKeyboardButton(label, callback_data=cb_data)])
            buttons.append([InlineKeyboardButton("❌ Закрыть ВСЕ", callback_data="close_all")])
            buttons.append([InlineKeyboardButton("Отмена", callback_data="cancel")])
            await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(buttons))
        else:
            await update.message.reply_text(text, reply_markup=MAIN_KEYBOARD)

    async def _cmd_stop(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        buttons = []
        for inst in INSTANCES:
            active = _is_service_active(inst["service"])
            icon = "🟢" if active else "🔴"
            buttons.append([InlineKeyboardButton(f"🛑 {icon} {inst['name']}", callback_data=f"stop_{inst['service']}")])
        buttons.append([InlineKeyboardButton("🛑 Всё", callback_data="stop_all")])
        buttons.append([InlineKeyboardButton("Отмена", callback_data="cancel")])
        await update.message.reply_text("Что остановить?", reply_markup=InlineKeyboardMarkup(buttons))

    async def _cmd_run(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        buttons = []
        for inst in INSTANCES:
            active = _is_service_active(inst["service"])
            icon = "🟢" if active else "🔴"
            buttons.append([InlineKeyboardButton(f"▶️ {icon} {inst['name']}", callback_data=f"start_{inst['service']}")])
        buttons.append([InlineKeyboardButton("▶️ Всё", callback_data="start_all")])
        buttons.append([InlineKeyboardButton("Отмена", callback_data="cancel")])
        await update.message.reply_text("Что запустить?", reply_markup=InlineKeyboardMarkup(buttons))

    # ── Inline callbacks ──

    async def _callback_handler(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if str(query.from_user.id) != self.chat_id:
            await query.answer("Нет доступа")
            return
        await query.answer()
        data = query.data

        # ── Stop service ──
        if data.startswith("stop_stasik-"):
            service = data[len("stop_"):]
            name = service.replace("stasik-", "").upper()
            ok = _systemctl("stop", service)
            await query.edit_message_text(f"{'🛑' if ok else '❌'} {name} {'остановлен' if ok else 'ошибка'}")

        elif data == "stop_all":
            results = []
            for inst in INSTANCES:
                ok = _systemctl("stop", inst["service"])
                results.append(f"{'🛑' if ok else '❌'} {inst['name']} {'остановлен' if ok else 'ошибка'}")
            await query.edit_message_text("\n".join(results))

        # ── Start service ──
        elif data.startswith("start_stasik-"):
            service = data[len("start_"):]
            name = service.replace("stasik-", "").upper()
            if _is_service_active(service):
                await query.edit_message_text(f"🟢 {name} уже работает")
            else:
                ok = _systemctl("start", service)
                await query.edit_message_text(f"{'▶️' if ok else '❌'} {name} {'запущен' if ok else 'ошибка'}")

        elif data == "start_all":
            results = []
            for inst in INSTANCES:
                if _is_service_active(inst["service"]):
                    results.append(f"🟢 {inst['name']} уже работает")
                else:
                    ok = _systemctl("start", inst["service"])
                    results.append(f"{'▶️' if ok else '❌'} {inst['name']} {'запущен' if ok else 'ошибка'}")
            await query.edit_message_text("\n".join(results))

        # ── Close single position ──
        elif data.startswith("close_") and data != "close_all":
            # Format: close_{INSTANCE}_{TRADE_ID}_{SYMBOL}
            parts = data.split("_", 3)
            if len(parts) >= 4:
                inst_name = parts[1]
                trade_id = int(parts[2])
                symbol = parts[3]

                await query.edit_message_text(f"⏳ Закрываю {symbol}...")

                inst = _find_instance(inst_name)
                if not inst or not inst["config"]:
                    await query.edit_message_text(f"❌ Инстанс {inst_name} не найден")
                    return

                loop = asyncio.get_event_loop()
                if inst["exchange"] == "bybit":
                    result = await loop.run_in_executor(None, _close_bybit_position, inst["config"], symbol)
                else:
                    result = await loop.run_in_executor(None, _close_tbank_position, inst["config"], symbol)

                if result.startswith("✅"):
                    _update_db_closed(inst["db"], trade_id)

                await query.edit_message_text(result)

        # ── Close all positions ──
        elif data == "close_all":
            _, positions = _get_positions()
            if not positions:
                await query.edit_message_text("Нет открытых позиций.")
                return

            await query.edit_message_text(f"⏳ Закрываю {len(positions)} позиций...")

            results = []
            loop = asyncio.get_event_loop()
            for p in positions:
                inst = _find_instance(p["instance"])
                if not inst or not inst["config"]:
                    results.append(f"❌ {p['symbol']} — инстанс не найден")
                    continue

                if inst["exchange"] == "bybit":
                    result = await loop.run_in_executor(None, _close_bybit_position, inst["config"], p["symbol"])
                else:
                    result = await loop.run_in_executor(None, _close_tbank_position, inst["config"], p["symbol"])

                if result.startswith("✅"):
                    _update_db_closed(inst["db"], p["id"])
                results.append(result)

            await query.edit_message_text("\n".join(results))

        elif data == "cancel":
            await query.edit_message_text("Отменено.")

    async def send_message(self, text: str):
        if not self._started or not self.app:
            return
        try:
            await self.app.bot.send_message(chat_id=self.chat_id, text=text)
        except Exception:
            logger.exception("Failed to send Telegram message")


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config_path = Path(__file__).resolve().parent.parent / "config" / "telegram.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        token = cfg["telegram"]["token"]
        chat_id = cfg["telegram"]["chat_id"]
    else:
        fallback = Path(__file__).resolve().parent.parent / "config" / "smc.yaml"
        with open(fallback) as f:
            cfg = yaml.safe_load(f)
        token = cfg["telegram"]["token"]
        chat_id = cfg["telegram"]["chat_id"]

    bot = StandaloneTelegramBot(token, chat_id)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await bot.start()
    logger.info("Waiting for commands...")
    await stop_event.wait()
    await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
