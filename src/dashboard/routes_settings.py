import asyncio
import json
import logging
from pathlib import Path

import aiosqlite
from aiohttp import web

logger = logging.getLogger(__name__)


def _other_instances(config: dict) -> list[dict]:
    """Return other_instances from config."""
    return config.get("other_instances", [])


class RouteSettingsMixin:
    """Dashboard mixin: night/sale settings, SSE events, helper queries."""

    async def _api_night_settings_get(self, request: web.Request) -> web.Response:
        """Return night mode settings from server file."""
        try:
            if self._night_settings_path.exists():
                data = json.loads(self._night_settings_path.read_text())
            else:
                data = {"enabled": False, "target": 0}
        except Exception:
            data = {"enabled": False, "target": 0}
        return web.json_response(data)

    async def _api_night_settings_post(self, request: web.Request) -> web.Response:
        """Save night mode settings to server file."""
        try:
            body = await request.json()
            data = {
                "enabled": bool(body.get("enabled", False)),
                "target": int(body.get("target", 0)),
            }
            self._night_settings_path.parent.mkdir(parents=True, exist_ok=True)
            self._night_settings_path.write_text(json.dumps(data))
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def _api_sale_settings_get(self, request: web.Request) -> web.Response:
        try:
            if self._sale_settings_path.exists():
                data = json.loads(self._sale_settings_path.read_text())
            else:
                data = {"enabled": False, "target": 0}
        except Exception:
            data = {"enabled": False, "target": 0}
        return web.json_response(data)

    async def _api_sale_settings_post(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            data = {
                "enabled": bool(body.get("enabled", False)),
                "target": int(body.get("target", 0)),
            }
            self._sale_settings_path.parent.mkdir(parents=True, exist_ok=True)
            self._sale_settings_path.write_text(json.dumps(data))
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def _api_events(self, request: web.Request) -> web.StreamResponse:
        """SSE endpoint — pushes events when trades open or close."""
        resp = web.StreamResponse()
        resp.content_type = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"
        await resp.prepare(request)

        last_id = await self._get_last_trade_id()
        last_open = await self._get_open_count()

        try:
            while True:
                await asyncio.sleep(3)
                cur_id = await self._get_last_trade_id()
                cur_open = await self._get_open_count()
                if cur_id != last_id:
                    # New trade appeared or status changed
                    if cur_open < last_open:
                        await resp.write(b"event: trade_closed\ndata: {}\n\n")
                    else:
                        await resp.write(b"event: trade_opened\ndata: {}\n\n")
                    last_id = cur_id
                    last_open = cur_open
                elif cur_open != last_open:
                    if cur_open < last_open:
                        await resp.write(b"event: trade_closed\ndata: {}\n\n")
                    else:
                        await resp.write(b"event: trade_opened\ndata: {}\n\n")
                    last_open = cur_open
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        return resp

    async def _get_last_trade_id(self) -> int:
        """Get max trade id across all instances for fast change detection."""
        max_id = 0
        _sql = "SELECT MAX(id) FROM trades"
        try:
            async with aiosqlite.connect(str(self.db.db_path)) as db:
                cur = await db.execute(_sql)
                row = await cur.fetchone()
                if row and row[0]:
                    max_id = max(max_id, row[0])
        except Exception:
            pass
        for inst in _other_instances(self.config):
            db_path = inst.get("db_path", "")
            if db_path and Path(db_path).exists():
                try:
                    async with aiosqlite.connect(db_path) as db:
                        cur = await db.execute(_sql)
                        row = await cur.fetchone()
                        if row and row[0]:
                            max_id = max(max_id, row[0])
                except Exception:
                    pass
        return max_id

    async def _get_open_count(self) -> int:
        """Fast open trade count across all instances."""
        cnt = 0
        _sql = "SELECT COUNT(*) FROM trades WHERE status='open'"
        try:
            async with aiosqlite.connect(str(self.db.db_path)) as db:
                cur = await db.execute(_sql)
                row = await cur.fetchone()
                if row:
                    cnt += row[0]
        except Exception:
            pass
        for inst in _other_instances(self.config):
            db_path = inst.get("db_path", "")
            if db_path and Path(db_path).exists():
                try:
                    async with aiosqlite.connect(db_path) as db:
                        cur = await db.execute(_sql)
                        row = await cur.fetchone()
                        if row:
                            cnt += row[0]
                except Exception:
                    pass
        return cnt
