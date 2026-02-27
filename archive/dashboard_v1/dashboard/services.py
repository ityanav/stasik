"""Dashboard utility functions and constants."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_ARCHIVE_MAP = {
    "trades.db": "archive_scalp.db",
    "degen.db": "archive_degen.db",
    "tbank_scalp.db": "archive_tbank_scalp.db",
    "tbank_swing.db": "archive_tbank_swing.db",
}


def _get_db_path(live_path: str, source: str) -> str:
    """Return archive DB path if source='archive', else the live path."""
    if source != "archive":
        return live_path
    p = Path(live_path)
    archive_name = _ARCHIVE_MAP.get(p.name)
    if archive_name:
        return str(p.parent / archive_name)
    return live_path


def _other_instances(config: dict) -> list[dict]:
    """Return other_instances from config."""
    return config.get("other_instances", [])
