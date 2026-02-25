"""Standalone dashboard — runs independently of any trading engine."""

import asyncio
import logging
import signal
from pathlib import Path

import yaml

from src.dashboard.app import Dashboard
from src.storage.database import Database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


async def main():
    config_path = Path(__file__).resolve().parent.parent / "config" / "fiba.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("Dashboard standalone starting")

    # DB for main instance (SCALP)
    db_path = config.get("database", {}).get("path")
    if not db_path:
        db_path = str(Path(__file__).resolve().parent.parent / "data" / "trades.db")
    db = Database(Path(db_path))
    await db.connect()

    dashboard = Dashboard(config, db, engine=None)
    await dashboard.start()

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await stop_event.wait()

    logger.info("Shutting down dashboard...")
    await dashboard.stop()
    logger.info("Dashboard stopped")


if __name__ == "__main__":
    asyncio.run(main())
