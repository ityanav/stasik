import asyncio
import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

logger = logging.getLogger(__name__)


class TelegramBot:
    def __init__(self, config: dict, engine):
        self.token = config["telegram"]["token"]
        self.chat_id = str(config["telegram"]["chat_id"])
        self.engine = engine
        self.app: Application | None = None
        self._started = False

    async def start(self):
        if not self.token or not self.chat_id:
            logger.warning("Telegram token/chat_id not configured, bot disabled")
            return

        self.app = Application.builder().token(self.token).build()
        self._register_handlers()

        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        self._started = True
        logger.info("Telegram bot started")

    async def stop(self):
        if self._started and self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            self._started = False
            logger.info("Telegram bot stopped")

    def _register_handlers(self):
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("stop", self._cmd_stop))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("pnl", self._cmd_pnl))
        self.app.add_handler(CommandHandler("positions", self._cmd_positions))
        self.app.add_handler(CommandHandler("pairs", self._cmd_pairs))
        self.app.add_handler(CommandHandler("help", self._cmd_help))
        self.app.add_handler(CallbackQueryHandler(self._callback_handler))

    def _check_auth(self, update: Update) -> bool:
        return str(update.effective_chat.id) == self.chat_id

    # ── Commands ─────────────────────────────────────────────

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        await update.message.reply_text(
            "Bybit Scalper Bot\n\n"
            "Commands:\n"
            "/status — Bot status\n"
            "/pnl — Profit & Loss\n"
            "/positions — Open positions\n"
            "/pairs — Active pairs\n"
            "/stop — Stop trading\n"
            "/help — Help"
        )

    async def _cmd_stop(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Yes, stop", callback_data="confirm_stop"),
                InlineKeyboardButton("Cancel", callback_data="cancel"),
            ]
        ])
        await update.message.reply_text("Stop trading?", reply_markup=keyboard)

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        text = await self.engine.get_status()
        await update.message.reply_text(text)

    async def _cmd_pnl(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        text = await self.engine.get_pnl_text()
        await update.message.reply_text(text)

    async def _cmd_positions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        text = await self.engine.get_positions_text()
        await update.message.reply_text(text)

    async def _cmd_pairs(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        pairs = self.engine.pairs
        text = "Active pairs:\n" + "\n".join(f"  • {p}" for p in pairs)
        await update.message.reply_text(text)

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not self._check_auth(update):
            return
        await update.message.reply_text(
            "Bybit Scalper Bot\n\n"
            "/start — Welcome\n"
            "/status — Balance, PnL, state\n"
            "/pnl — Detailed PnL\n"
            "/positions — Open positions\n"
            "/pairs — Active trading pairs\n"
            "/stop — Stop the bot\n"
            "/help — This message"
        )

    # ── Callback (inline buttons) ────────────────────────────

    async def _callback_handler(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if str(query.from_user.id) != self.chat_id:
            await query.answer("Unauthorized")
            return

        await query.answer()
        data = query.data

        if data == "confirm_stop":
            await self.engine.stop()
            await query.edit_message_text("Trading stopped.")
        elif data == "cancel":
            await query.edit_message_text("Cancelled.")

    # ── Send notification ────────────────────────────────────

    async def send_message(self, text: str):
        if not self._started or not self.app:
            return
        try:
            await self.app.bot.send_message(chat_id=self.chat_id, text=text)
        except Exception:
            logger.exception("Failed to send Telegram message")
