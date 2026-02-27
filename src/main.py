import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

import yaml

from src.core.engine import TradingEngine
from src.dashboard.app import Dashboard
from src.telegram.bot import TelegramBot


def load_config(config_path: str | None = None) -> dict:
    if config_path:
        path = Path(config_path)
    else:
        path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    instance = config.get("instance_name", "stasik")
    log_file = Path(__file__).resolve().parent.parent / f"{instance}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Config loaded: %s", args.config or "config/config.yaml")

    # Database path from config
    db_path = config.get("database", {}).get("path")

    engine = TradingEngine(config, db_path=db_path)

    notify_only = config.get("telegram", {}).get("notify_only", False)
    tg_bot = TelegramBot(config, engine, notify_only=notify_only)

    # Wire notifier: engine -> telegram
    engine.notifier = tg_bot.send_message

    # Dashboard
    dashboard = None
    if config.get("dashboard", {}).get("enabled", False):
        dashboard = Dashboard(config, engine.db, engine)


    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # Start Telegram bot
    await tg_bot.start()

    # Start dashboard
    if dashboard:
        await dashboard.start()

    # Start engine in background
    engine_task = asyncio.create_task(engine.start())

    # Wait for stop signal
    await stop_event.wait()

    # Graceful shutdown
    logger.info("Shutting down...")
    await engine.shutdown()
    engine_task.cancel()
    try:
        await engine_task
    except asyncio.CancelledError:
        pass
    if dashboard:
        await dashboard.stop()
    await tg_bot.stop()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
