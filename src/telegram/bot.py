import logging
import subprocess

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

logger = logging.getLogger(__name__)

# Постоянная клавиатура внизу чата
MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [
        [KeyboardButton("📊 Статус"), KeyboardButton("💰 PnL")],
        [KeyboardButton("📈 Позиции"), KeyboardButton("🪙 Пары")],
        [KeyboardButton("❌ Закрыть сделки"), KeyboardButton("❓ Помощь")],
        [KeyboardButton("▶️ Старт"), KeyboardButton("🛑 Стоп")],
    ],
    resize_keyboard=True,
)


class TelegramBot:
    def __init__(self, config: dict, engine, notify_only: bool = False):
        self.token = config["telegram"]["token"]
        self.chat_id = str(config["telegram"]["chat_id"])
        self.engine = engine
        self.config = config
        self.app: Application | None = None
        self._started = False
        self._notify_only = notify_only

        # Other instances for start/stop control
        self._other_instances = config.get("other_instances", [])

    async def start(self):
        if not self.token or not self.chat_id:
            logger.warning("Telegram token/chat_id not configured, bot disabled")
            return

        self.app = Application.builder().token(self.token).build()

        if not self._notify_only:
            self._register_handlers()

        await self.app.initialize()
        await self.app.start()

        if not self._notify_only:
            await self.app.updater.start_polling(
                drop_pending_updates=True,
                allowed_updates=["message", "callback_query"],
            )

        self._started = True
        mode = "notify-only" if self._notify_only else "full"
        logger.info("Telegram bot started (%s)", mode)

    async def stop(self):
        if self._started and self.app:
            if not self._notify_only:
                await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            self._started = False
            logger.info("Telegram bot stopped")

    def _register_handlers(self):
        # Slash-команды
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("stop", self._cmd_stop))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("pnl", self._cmd_pnl))
        self.app.add_handler(CommandHandler("positions", self._cmd_positions))
        self.app.add_handler(CommandHandler("pairs", self._cmd_pairs))
        self.app.add_handler(CommandHandler("help", self._cmd_help))

        # Кнопки клавиатуры (текстовые сообщения)
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_button))

        # Inline-кнопки (подтверждение стопа, закрытие позиций)
        self.app.add_handler(CallbackQueryHandler(self._callback_handler))

        # Error handler
        self.app.add_error_handler(self._error_handler)

    @staticmethod
    async def _error_handler(update, ctx: ContextTypes.DEFAULT_TYPE):
        logger.error("Telegram handler error: %s", ctx.error, exc_info=ctx.error)

    def _check_auth(self, update: Update) -> bool:
        return str(update.effective_chat.id) == self.chat_id

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _systemctl(action: str, service: str) -> bool:
        """Run systemctl action. Returns True on success."""
        try:
            result = subprocess.run(
                ["systemctl", action, service],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            logger.exception("systemctl %s %s failed", action, service)
            return False

    @staticmethod
    def _is_service_active(service: str) -> bool:
        try:
            result = subprocess.run(
                ["systemctl", "is-active", service],
                capture_output=True, text=True, timeout=3,
            )
            return result.stdout.strip() == "active"
        except Exception:
            return False

    # ── Button handler ─────────────────────────────────────────

    async def _handle_button(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return

        text = update.message.text.strip()

        handlers = {
            "📊 Статус": self._cmd_status,
            "💰 PnL": self._cmd_pnl,
            "📈 Позиции": self._cmd_positions,
            "🪙 Пары": self._cmd_pairs,
            "❌ Закрыть сделки": self._cmd_close_all,
            "▶️ Старт": self._cmd_run,
            "🛑 Стоп": self._cmd_stop,
            "❓ Помощь": self._cmd_help,
        }

        handler = handlers.get(text)
        if handler:
            await handler(update, ctx)

    # ── Commands ─────────────────────────────────────────────

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        await update.message.reply_text(
            "🤖 Stasik Trading Bot\n\n"
            "Используй кнопки внизу или команды:\n"
            "/status — Статус бота\n"
            "/pnl — Прибыль и убытки\n"
            "/positions — Открытые позиции\n"
            "/pairs — Торговые пары\n"
            "/stop — Остановить бота",
            reply_markup=MAIN_KEYBOARD,
        )

    async def _cmd_stop(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return

        instance_name = self.engine.instance_name or "SCALP"
        buttons = [
            [InlineKeyboardButton(f"🛑 {instance_name}", callback_data="stop_scalp")],
        ]
        for inst in self._other_instances:
            name = inst.get("name", "???")
            service = inst.get("service", "")
            if service:
                buttons.append([InlineKeyboardButton(f"🛑 {name}", callback_data=f"stop_inst_{service}")])
        buttons.append([InlineKeyboardButton("🛑 Всё", callback_data="stop_all")])
        buttons.append([InlineKeyboardButton("Отмена", callback_data="cancel")])

        keyboard = InlineKeyboardMarkup(buttons)
        await update.message.reply_text("Что остановить?", reply_markup=keyboard)

    async def _cmd_run(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return

        instance_name = self.engine.instance_name or "SCALP"
        buttons = [
            [InlineKeyboardButton(f"▶️ {instance_name}", callback_data="start_scalp")],
        ]
        for inst in self._other_instances:
            name = inst.get("name", "???")
            service = inst.get("service", "")
            if service:
                buttons.append([InlineKeyboardButton(f"▶️ {name}", callback_data=f"start_inst_{service}_{name}")])
        buttons.append([InlineKeyboardButton("▶️ Всё", callback_data="start_all")])
        buttons.append([InlineKeyboardButton("Отмена", callback_data="cancel")])

        keyboard = InlineKeyboardMarkup(buttons)
        await update.message.reply_text("Что запустить?", reply_markup=keyboard)

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        text = await self.engine.get_status()
        await update.message.reply_text(text, reply_markup=MAIN_KEYBOARD)

    async def _cmd_pnl(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        text = await self.engine.get_pnl_text()
        await update.message.reply_text(text, reply_markup=MAIN_KEYBOARD)

    async def _cmd_positions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        text = await self.engine.get_positions_text()
        await update.message.reply_text(text, reply_markup=MAIN_KEYBOARD)

    async def _cmd_pairs(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        pairs = self.engine.pairs
        text = "🪙 Торговые пары:\n" + "\n".join(f"  • {p}" for p in pairs)
        await update.message.reply_text(text, reply_markup=MAIN_KEYBOARD)

    async def _cmd_close_all(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        # Show positions with individual close buttons
        text = await self.engine.get_positions_text()
        if "Нет открытых позиций" in text:
            await update.message.reply_text("Нет открытых позиций.", reply_markup=MAIN_KEYBOARD)
            return

        # Build buttons: one per position + close all
        positions = await self.engine.get_open_positions_list()
        buttons = []
        for p in positions:
            direction = "L" if p["side"] == "Buy" else "S"
            upnl = p.get("upnl", 0)
            pnl_str = f"+{upnl:,.0f}" if upnl >= 0 else f"{upnl:,.0f}"
            label = f"❌ {p['symbol']} {direction} ({pnl_str})"
            buttons.append([InlineKeyboardButton(label, callback_data=f"close_{p['symbol']}")])

        buttons.append([
            InlineKeyboardButton("💥 Закрыть ВСЕ", callback_data="confirm_close_all"),
            InlineKeyboardButton("Отмена", callback_data="cancel"),
        ])

        keyboard = InlineKeyboardMarkup(buttons)
        await update.message.reply_text(
            f"{text}\n\nВыбери сделку для закрытия:",
            reply_markup=keyboard,
        )

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        await update.message.reply_text(
            "🤖 Stasik Trading Bot\n\n"
            "📊 Статус — баланс, состояние\n"
            "💰 PnL — прибыль/убытки\n"
            "📈 Позиции — открытые сделки\n"
            "🪙 Пары — торговые пары\n"
            "❌ Закрыть сделки — закрыть все позиции\n"
            "▶️ Старт — запустить бота\n"
            "🛑 Стоп — остановить бота\n"
            "❓ Помощь — эта справка",
            reply_markup=MAIN_KEYBOARD,
        )

    # ── Callback (inline buttons) ────────────────────────────

    async def _callback_handler(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        logger.info("Callback received: data=%s from user=%s", query.data, query.from_user.id)

        if str(query.from_user.id) != self.chat_id:
            await query.answer("Нет доступа")
            return

        await query.answer()
        data = query.data

        # ── Stop handlers ──
        if data == "stop_scalp":
            await self.engine.stop()
            await query.edit_message_text("⏸ SCALP остановлен")

        elif data.startswith("stop_inst_"):
            service = data[len("stop_inst_"):]
            name = service.replace("stasik-", "").upper()
            ok = self._systemctl("stop", service)
            if ok:
                await query.edit_message_text(f"🛑 {name} остановлен")
            else:
                await query.edit_message_text(f"❌ Не удалось остановить {name}")

        elif data == "stop_all":
            await self.engine.stop()
            results = ["⏸ SCALP остановлен"]
            for inst in self._other_instances:
                service = inst.get("service", "")
                name = inst.get("name", "???")
                if service:
                    ok = self._systemctl("stop", service)
                    results.append(f"{'🛑' if ok else '❌'} {name} {'остановлен' if ok else 'ошибка'}")
            await query.edit_message_text("\n".join(results))

        # ── Start handlers ──
        elif data == "start_scalp":
            if self.engine._running:
                await query.edit_message_text("SCALP уже работает")
            else:
                await self.engine.resume()
                await query.edit_message_text("▶️ SCALP запущен")

        elif data.startswith("start_inst_"):
            parts = data[len("start_inst_"):].split("_", 1)
            service = parts[0]
            name = parts[1] if len(parts) > 1 else service.replace("stasik-", "").upper()
            if self._is_service_active(service):
                await query.edit_message_text(f"{name} уже работает")
            else:
                ok = self._systemctl("start", service)
                if ok:
                    await query.edit_message_text(f"▶️ {name} запущен")
                else:
                    await query.edit_message_text(f"❌ Не удалось запустить {name}")

        elif data == "start_all":
            results = []
            if self.engine._running:
                results.append("SCALP уже работает")
            else:
                await self.engine.resume()
                results.append("▶️ SCALP запущен")
            for inst in self._other_instances:
                service = inst.get("service", "")
                name = inst.get("name", "???")
                if service:
                    if self._is_service_active(service):
                        results.append(f"{name} уже работает")
                    else:
                        ok = self._systemctl("start", service)
                        results.append(f"{'▶️' if ok else '❌'} {name} {'запущен' if ok else 'ошибка'}")
            await query.edit_message_text("\n".join(results))

        # ── Legacy / other handlers ──
        elif data == "confirm_stop":
            await self.engine.stop()
            await query.edit_message_text("⏸ Торговля остановлена.")

        elif data == "confirm_close_all":
            logger.info("Closing all positions...")
            await query.edit_message_text("⏳ Закрываю позиции...")
            try:
                result = await self.engine.close_all_positions()
                logger.info("Close result: %s", result)
                await self.app.bot.send_message(chat_id=self.chat_id, text=result, reply_markup=MAIN_KEYBOARD)
            except Exception:
                logger.exception("Error closing positions")
                await self.app.bot.send_message(
                    chat_id=self.chat_id,
                    text="❌ Ошибка при закрытии позиций. Проверь логи.",
                    reply_markup=MAIN_KEYBOARD,
                )

        elif data.startswith("close_"):
            symbol = data[6:]  # "close_BTCUSDT" -> "BTCUSDT"
            logger.info("Closing position %s...", symbol)
            await query.edit_message_text(f"⏳ Закрываю {symbol}...")
            try:
                result = await self.engine.close_position(symbol)
                await self.app.bot.send_message(chat_id=self.chat_id, text=result, reply_markup=MAIN_KEYBOARD)
            except Exception:
                logger.exception("Error closing position %s", symbol)
                await self.app.bot.send_message(
                    chat_id=self.chat_id,
                    text=f"❌ Ошибка при закрытии {symbol}. Проверь логи.",
                    reply_markup=MAIN_KEYBOARD,
                )

        elif data == "cancel":
            await query.edit_message_text("Отменено.")

    # ── Send notification ────────────────────────────────────

    async def send_message(self, text: str):
        if not self._started or not self.app:
            return
        try:
            await self.app.bot.send_message(chat_id=self.chat_id, text=text)
        except Exception:
            logger.exception("Failed to send Telegram message")
