"""Standalone Telegram bot — reads all instance DBs directly, no TradingEngine."""

import asyncio
import logging
import signal
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

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
        [KeyboardButton("📊 Статус"), KeyboardButton("💰 PnL")],
        [KeyboardButton("📈 Позиции"), KeyboardButton("❓ Помощь")],
        [KeyboardButton("▶️ Старт"), KeyboardButton("🛑 Стоп")],
    ],
    resize_keyboard=True,
)

# Instance definitions: name, db_path, service, currency
INSTANCES = [
    {"name": "FIBA", "db": "/root/stasik/data/fiba.db", "service": "stasik-fiba", "currency": "USDT"},
    {"name": "TBANK-SCALP", "db": "/root/stasik/data/tbank_scalp.db", "service": "stasik-tbank-scalp", "currency": "RUB"},
    {"name": "TBANK-SWING", "db": "/root/stasik/data/tbank_swing.db", "service": "stasik-tbank-swing", "currency": "RUB"},
    {"name": "MIDAS", "db": "/root/stasik/data/midas.db", "service": "stasik-midas", "currency": "RUB"},
    {"name": "FIN", "db": None, "service": "stasik-fin", "currency": None},
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


def _get_pnl() -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = ["💰 PnL\n"]

    total_usdt = 0.0
    total_rub = 0.0
    today_usdt = 0.0
    today_rub = 0.0

    for inst in INSTANCES:
        if not inst["db"] or not Path(inst["db"]).exists():
            continue

        # All-time
        rows = _query_db(inst["db"], "SELECT COUNT(*) as cnt, COALESCE(SUM(pnl),0) as total FROM trades WHERE status='closed'")
        cnt = rows[0]["cnt"] if rows else 0
        total_pnl = rows[0]["total"] if rows else 0

        # Today
        rows_today = _query_db(inst["db"], "SELECT COUNT(*) as cnt, COALESCE(SUM(pnl),0) as total FROM trades WHERE status='closed' AND closed_at LIKE ?", (today + "%",))
        today_cnt = rows_today[0]["cnt"] if rows_today else 0
        today_pnl = rows_today[0]["total"] if rows_today else 0

        cur = inst["currency"]
        sign = "+" if total_pnl >= 0 else ""
        today_sign = "+" if today_pnl >= 0 else ""

        lines.append(f"📌 {inst['name']}: {sign}{total_pnl:,.2f} {cur} ({cnt} trades)")
        if today_cnt > 0:
            lines.append(f"   Today: {today_sign}{today_pnl:,.2f} {cur} ({today_cnt})")

        if cur == "USDT":
            total_usdt += total_pnl
            today_usdt += today_pnl
        else:
            total_rub += total_pnl
            today_rub += today_pnl

    lines.append("")
    if total_usdt != 0 or today_usdt != 0:
        lines.append(f"Σ USDT: {'+' if total_usdt >= 0 else ''}{total_usdt:,.2f} (today: {'+' if today_usdt >= 0 else ''}{today_usdt:,.2f})")
    if total_rub != 0 or today_rub != 0:
        lines.append(f"Σ RUB: {'+' if total_rub >= 0 else ''}{total_rub:,.2f} (today: {'+' if today_rub >= 0 else ''}{today_rub:,.2f})")

    return "\n".join(lines)


def _get_positions() -> tuple[str, list[dict]]:
    all_positions = []
    for inst in INSTANCES:
        if not inst["db"] or not Path(inst["db"]).exists():
            continue
        rows = _query_db(
            inst["db"],
            "SELECT symbol, side, entry_price, qty, pnl, opened_at FROM trades WHERE status='open'",
        )
        for r in rows:
            r["instance"] = inst["name"]
            r["currency"] = inst["currency"]
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
        self.app.add_handler(CommandHandler("pnl", self._cmd_pnl))
        self.app.add_handler(CommandHandler("positions", self._cmd_positions))
        self.app.add_handler(CommandHandler("stop", self._cmd_stop))
        self.app.add_handler(CommandHandler("help", self._cmd_help))
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
            "💰 PnL": self._cmd_pnl,
            "📈 Позиции": self._cmd_positions,
            "▶️ Старт": self._cmd_run,
            "🛑 Стоп": self._cmd_stop,
            "❓ Помощь": self._cmd_help,
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
            "/status — Статус сервисов\n"
            "/pnl — Прибыль и убытки\n"
            "/positions — Открытые позиции\n"
            "/stop — Остановить сервис\n"
            "/help — Помощь",
            reply_markup=MAIN_KEYBOARD,
        )

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        await update.message.reply_text(_get_status(), reply_markup=MAIN_KEYBOARD)

    async def _cmd_pnl(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        await update.message.reply_text(_get_pnl(), reply_markup=MAIN_KEYBOARD)

    async def _cmd_positions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        text, positions = _get_positions()
        await update.message.reply_text(text, reply_markup=MAIN_KEYBOARD)

    async def _cmd_stop(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        buttons = []
        for inst in INSTANCES:
            buttons.append([InlineKeyboardButton(f"🛑 {inst['name']}", callback_data=f"stop_{inst['service']}")])
        buttons.append([InlineKeyboardButton("🛑 Всё", callback_data="stop_all")])
        buttons.append([InlineKeyboardButton("Отмена", callback_data="cancel")])
        await update.message.reply_text("Что остановить?", reply_markup=InlineKeyboardMarkup(buttons))

    async def _cmd_run(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        buttons = []
        for inst in INSTANCES:
            buttons.append([InlineKeyboardButton(f"▶️ {inst['name']}", callback_data=f"start_{inst['service']}")])
        buttons.append([InlineKeyboardButton("▶️ Всё", callback_data="start_all")])
        buttons.append([InlineKeyboardButton("Отмена", callback_data="cancel")])
        await update.message.reply_text("Что запустить?", reply_markup=InlineKeyboardMarkup(buttons))

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        await update.message.reply_text(
            "🤖 Stasik Trading Bot\n\n"
            "📊 Статус — состояние сервисов\n"
            "💰 PnL — прибыль/убытки\n"
            "📈 Позиции — открытые сделки\n"
            "▶️ Старт — запустить сервис\n"
            "🛑 Стоп — остановить сервис\n"
            "❓ Помощь — эта справка",
            reply_markup=MAIN_KEYBOARD,
        )

    # ── Inline callbacks ──

    async def _callback_handler(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if str(query.from_user.id) != self.chat_id:
            await query.answer("Нет доступа")
            return
        await query.answer()
        data = query.data

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

        elif data.startswith("start_stasik-"):
            service = data[len("start_"):]
            name = service.replace("stasik-", "").upper()
            if _is_service_active(service):
                await query.edit_message_text(f"{name} уже работает")
            else:
                ok = _systemctl("start", service)
                await query.edit_message_text(f"{'▶️' if ok else '❌'} {name} {'запущен' if ok else 'ошибка'}")

        elif data == "start_all":
            results = []
            for inst in INSTANCES:
                if _is_service_active(inst["service"]):
                    results.append(f"{inst['name']} уже работает")
                else:
                    ok = _systemctl("start", inst["service"])
                    results.append(f"{'▶️' if ok else '❌'} {inst['name']} {'запущен' if ok else 'ошибка'}")
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
        # Fallback: read from smc.yaml
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
