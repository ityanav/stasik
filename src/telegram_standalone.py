"""Standalone Telegram bot — reads all instance DBs directly, no TradingEngine."""

import asyncio
import logging
import signal
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

from src.telegram_data import INSTANCES, is_service_active, find_instance
from src.telegram_formatters import format_dashboard, format_positions
from src.telegram_actions import (
    systemctl_action,
    close_bybit_position,
    close_tbank_position,
    update_db_closed,
)
from src.telegram_analytics import get_all_trades_with_scores, analyze_trades

logger = logging.getLogger(__name__)

MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [
        [KeyboardButton("📊 Статус"), KeyboardButton("📈 Позиции")],
        [KeyboardButton("▶️ Старт"), KeyboardButton("🛑 Стоп")],
        [KeyboardButton("🔬 Аналитик")],
    ],
    resize_keyboard=True,
)


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
            "🔬 Аналитик": self._cmd_analytics,
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
            "📊 Статус — дашборд с балансом и PnL\n"
            "📈 Позиции — открытые сделки + закрытие\n"
            "▶️ Старт — запустить бота\n"
            "🛑 Стоп — остановить бота\n"
            "🔬 Аналитик — AI-анализ сделок",
            reply_markup=MAIN_KEYBOARD,
        )

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, format_dashboard)
        await update.message.reply_text(text, reply_markup=MAIN_KEYBOARD)

    async def _cmd_analytics(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        await update.message.reply_text("🔬 Анализирую сделки...", reply_markup=MAIN_KEYBOARD)
        loop = asyncio.get_event_loop()
        trades = await loop.run_in_executor(None, get_all_trades_with_scores)
        text = await loop.run_in_executor(None, analyze_trades, trades)
        # Telegram limit 4096 chars — split if needed
        for i in range(0, len(text), 4096):
            await update.message.reply_text(text[i:i + 4096], reply_markup=MAIN_KEYBOARD)

    async def _cmd_positions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        loop = asyncio.get_event_loop()
        text, positions = await loop.run_in_executor(None, format_positions)

        if positions:
            buttons = []
            for p in positions:
                label = f"❌ {p['symbol']}"
                cb_data = f"close_{p['instance']}_{p['id']}_{p['symbol']}"
                buttons.append([InlineKeyboardButton(label, callback_data=cb_data)])
            buttons.append([InlineKeyboardButton("❌ Закрыть ВСЕ", callback_data="close_all")])
            buttons.append([InlineKeyboardButton("Отмена", callback_data="cancel")])
            await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(buttons))
        else:
            await update.message.reply_text("📈 Нет открытых позиций.", reply_markup=MAIN_KEYBOARD)

    async def _cmd_stop(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        buttons = []
        for inst in INSTANCES:
            active = is_service_active(inst["service"])
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
            active = is_service_active(inst["service"])
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
            ok = systemctl_action("stop", service)
            await query.edit_message_text(f"{'🛑' if ok else '❌'} {name} {'остановлен' if ok else 'ошибка'}")

        elif data == "stop_all":
            results = []
            for inst in INSTANCES:
                ok = systemctl_action("stop", inst["service"])
                results.append(f"{'🛑' if ok else '❌'} {inst['name']} {'остановлен' if ok else 'ошибка'}")
            await query.edit_message_text("\n".join(results))

        # ── Start service ──
        elif data.startswith("start_stasik-"):
            service = data[len("start_"):]
            name = service.replace("stasik-", "").upper()
            if is_service_active(service):
                await query.edit_message_text(f"🟢 {name} уже работает")
            else:
                ok = systemctl_action("start", service)
                await query.edit_message_text(f"{'▶️' if ok else '❌'} {name} {'запущен' if ok else 'ошибка'}")

        elif data == "start_all":
            results = []
            for inst in INSTANCES:
                if is_service_active(inst["service"]):
                    results.append(f"🟢 {inst['name']} уже работает")
                else:
                    ok = systemctl_action("start", inst["service"])
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

                inst = find_instance(inst_name)
                if not inst or not inst["config"]:
                    await query.edit_message_text(f"❌ Инстанс {inst_name} не найден")
                    return

                loop = asyncio.get_event_loop()
                if inst["exchange"] == "bybit":
                    result = await loop.run_in_executor(None, close_bybit_position, inst["config"], symbol)
                else:
                    result = await loop.run_in_executor(None, close_tbank_position, inst["config"], symbol)

                if result.startswith("✅"):
                    update_db_closed(inst["db"], trade_id)

                await query.edit_message_text(result)

        # ── Close all positions ──
        elif data == "close_all":
            _, positions = format_positions()
            if not positions:
                await query.edit_message_text("Нет открытых позиций.")
                return

            await query.edit_message_text(f"⏳ Закрываю {len(positions)} позиций...")

            results = []
            loop = asyncio.get_event_loop()
            for p in positions:
                inst = find_instance(p["instance"])
                if not inst or not inst["config"]:
                    results.append(f"❌ {p['symbol']} — инстанс не найден")
                    continue

                if inst["exchange"] == "bybit":
                    result = await loop.run_in_executor(None, close_bybit_position, inst["config"], p["symbol"])
                else:
                    result = await loop.run_in_executor(None, close_tbank_position, inst["config"], p["symbol"])

                if result.startswith("✅"):
                    update_db_closed(inst["db"], p["id"])
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
