import logging
import subprocess
from datetime import date
from pathlib import Path

import aiosqlite
from aiohttp import web

from src.dashboard.auth import AuthManager, COOKIE_NAME
from src.storage.database import Database

logger = logging.getLogger(__name__)


_ARCHIVE_MAP = {
    "trades.db": "archive_scalp_pre_kotegawa.db",
    "degen.db": "archive_degen_pre_kotegawa.db",
    "swing.db": "archive_swing_pre_kotegawa.db",
    "tbank_scalp.db": "archive_tbank_scalp_pre_kotegawa.db",
    "tbank_swing.db": "archive_tbank_swing_pre_kotegawa.db",
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


class Dashboard:
    def __init__(self, config: dict, db: Database, engine=None):
        self.config = config
        self.db = db
        self.engine = engine
        self.port = config.get("dashboard", {}).get("port", 8080)
        self.auth = AuthManager(config)
        self.app = web.Application(middlewares=[self.auth.middleware()])
        self._setup_routes()
        self._runner: web.AppRunner | None = None

    def _setup_routes(self):
        self.app.router.add_get("/login", self._login_page)
        self.app.router.add_post("/login", self._login_post)
        self.app.router.add_get("/logout", self._logout)
        self.app.router.add_get("/", self._index)
        self.app.router.add_get("/api/stats", self._api_stats)
        self.app.router.add_get("/api/trades", self._api_trades)
        self.app.router.add_get("/api/pnl", self._api_pnl)
        self.app.router.add_get("/api/positions", self._api_positions)
        self.app.router.add_get("/api/instances", self._api_instances)

    async def start(self):
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", self.port)
        await site.start()
        logger.info("Dashboard started on 127.0.0.1:%d", self.port)

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()
            logger.info("Dashboard stopped")

    # --- Auth routes ---

    async def _login_page(self, request: web.Request) -> web.Response:
        ip = request.headers.get("X-Real-IP", request.remote)
        error = ""
        blocked = self.auth.is_blocked(ip)
        if blocked:
            error = '<div class="error">IP заблокирован. Подождите 15 минут.</div>'
        question, token = self.auth.generate_captcha()
        html = _LOGIN_HTML.replace("{{error}}", error)
        html = html.replace("{{captcha_question}}", question)
        html = html.replace("{{captcha_token}}", token)
        html = html.replace("{{blocked}}", "disabled" if blocked else "")
        return web.Response(text=html, content_type="text/html")

    async def _login_post(self, request: web.Request) -> web.Response:
        ip = request.headers.get("X-Real-IP", request.remote)
        if self.auth.is_blocked(ip):
            raise web.HTTPFound("/login")

        data = await request.post()
        username = data.get("username", "")
        password = data.get("password", "")
        captcha_answer = data.get("captcha", "")
        captcha_token = data.get("captcha_token", "")

        if not self.auth.verify_captcha(captcha_token, captcha_answer):
            self.auth.record_fail(ip)
            raise web.HTTPFound("/login")

        if not self.auth.check_password(username, password):
            self.auth.record_fail(ip)
            raise web.HTTPFound("/login")

        self.auth.clear_fails(ip)
        cookie_value = self.auth.create_session_cookie(username)
        resp = web.HTTPFound("/")
        resp.set_cookie(
            COOKIE_NAME,
            cookie_value,
            max_age=86400,
            httponly=True,
            secure=True,
            samesite="Lax",
        )
        return resp

    async def _logout(self, request: web.Request) -> web.Response:
        resp = web.HTTPFound("/login")
        resp.del_cookie(COOKIE_NAME)
        return resp

    # --- API routes ---

    async def _index(self, request: web.Request) -> web.Response:
        return web.Response(text=_DASHBOARD_HTML, content_type="text/html")

    async def _api_stats(self, request: web.Request) -> web.Response:
        source = request.query.get("source", "live")
        is_archive = source == "archive"

        if is_archive:
            archive_path = _get_db_path(str(self.db.db_path), source)
            daily_pnl = 0.0
            total_pnl = 0.0
            open_count = 0
            total = 0
            wins = 0
            losses = 0
            if Path(archive_path).exists():
                try:
                    async with aiosqlite.connect(archive_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute("SELECT COALESCE(SUM(pnl), 0) as total FROM daily_pnl")
                        row = await cur.fetchone()
                        if row:
                            total_pnl = float(row["total"])
                        cur = await db.execute("SELECT COUNT(*) as cnt FROM trades WHERE status = 'open'")
                        row = await cur.fetchone()
                        if row:
                            open_count = int(row["cnt"])
                        cur = await db.execute(
                            "SELECT COUNT(*) as total, "
                            "COALESCE(SUM(CASE WHEN pnl >= 0 THEN 1 ELSE 0 END), 0) as wins, "
                            "COALESCE(SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END), 0) as losses "
                            "FROM trades WHERE status = 'closed'"
                        )
                        row = await cur.fetchone()
                        if row:
                            total = int(row["total"])
                            wins = int(row["wins"])
                            losses = int(row["losses"])
                except Exception:
                    logger.warning("Failed to read archive stats from %s", archive_path)
        else:
            daily_pnl = await self.db.get_daily_pnl()
            total_pnl = await self.db.get_total_pnl()
            open_trades = await self.db.get_open_trades()
            stats = await self.db.get_trade_stats()
            total = stats["total"]
            wins = stats["wins"]
            losses = stats["losses"]
            open_count = len(open_trades)

        balance = 0.0
        running = False
        tbank_balance = 0.0
        if not is_archive and self.engine:
            try:
                balance = self.engine.client.get_balance()
            except Exception:
                pass
            running = self.engine._running

        exchange = self.config.get("exchange", "bybit")
        currency = "RUB" if exchange == "tbank" else "USDT"

        # Total trades across all instances
        all_trades = total
        for inst in self.config.get("other_instances", []):
            db_path = _get_db_path(inst.get("db_path", ""), source)
            inst_name = inst.get("name", "")
            if db_path and Path(db_path).exists():
                try:
                    async with aiosqlite.connect(db_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute("SELECT COUNT(*) as cnt FROM trades WHERE status='closed'")
                        row = await cur.fetchone()
                        if row:
                            all_trades += int(row["cnt"])
                except Exception:
                    pass
            # Get TBank balance (only for live mode)
            if not is_archive and "TBANK" in inst_name.upper() and self.engine:
                try:
                    from src.exchange.tbank_client import TBankClient
                    import yaml
                    svc = inst.get("service", "")
                    cfg_map = {"stasik-tbank-scalp": "config/tbank_scalp.yaml", "stasik-tbank-swing": "config/tbank_swing.yaml"}
                    cfg_path = cfg_map.get(svc)
                    if cfg_path and Path(cfg_path).exists() and tbank_balance == 0:
                        with open(cfg_path) as f:
                            tcfg = yaml.safe_load(f)
                        tc = TBankClient(tcfg)
                        tbank_balance = tc.get_balance()
                except Exception:
                    pass

        data = {
            "balance": balance,
            "tbank_balance": tbank_balance,
            "daily_pnl": daily_pnl,
            "total_pnl": total_pnl,
            "open_positions": open_count,
            "total_trades": total,
            "all_trades": all_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / total * 100) if total > 0 else 0,
            "running": running,
            "currency": currency,
            "source": source,
        }
        return web.json_response(data)

    async def _api_trades(self, request: web.Request) -> web.Response:
        page = int(request.query.get("page", "1"))
        per_page = int(request.query.get("per_page", "20"))
        source = request.query.get("source", "live")

        instance_name = self.config.get("instance_name", "SCALP")

        # Collect all trades from all instances
        all_trades = []

        # Main instance trades
        main_db_path = _get_db_path(str(self.db.db_path), source)
        if source == "archive":
            if Path(main_db_path).exists():
                try:
                    async with aiosqlite.connect(main_db_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute("SELECT * FROM trades ORDER BY id DESC LIMIT 1000")
                        rows = await cur.fetchall()
                        for r in rows:
                            t = dict(r)
                            t["instance"] = t.get("instance") or instance_name
                            all_trades.append(t)
                except Exception:
                    logger.warning("Failed to read archive trades from %s", main_db_path)
        else:
            main_trades = await self.db.get_recent_trades(1000)
            for t in main_trades:
                t["instance"] = t.get("instance") or instance_name
                all_trades.append(t)

        # Other instances trades
        for inst in self.config.get("other_instances", []):
            db_path = _get_db_path(inst.get("db_path", ""), source)
            inst_name = inst.get("name", "???")
            if db_path and Path(db_path).exists():
                try:
                    async with aiosqlite.connect(db_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute(
                            "SELECT * FROM trades ORDER BY id DESC LIMIT 1000"
                        )
                        rows = await cur.fetchall()
                        for r in rows:
                            t = dict(r)
                            t["instance"] = t.get("instance") or inst_name
                            all_trades.append(t)
                except Exception:
                    logger.warning("Failed to read trades from %s", inst_name)

        # Sort by opened_at descending
        all_trades.sort(key=lambda t: t.get("opened_at", ""), reverse=True)

        # Paginate
        offset = (page - 1) * per_page
        page_trades = all_trades[offset:offset + per_page + 1]
        has_next = len(page_trades) > per_page
        if has_next:
            page_trades = page_trades[:per_page]

        return web.json_response({"trades": page_trades, "page": page, "has_next": has_next})

    async def _api_pnl(self, request: web.Request) -> web.Response:
        days = int(request.query.get("days", "30"))
        tf = request.query.get("tf", "1D")  # 10m, 30m, 1H, 1D, 1M
        source = request.query.get("source", "live")

        instance_name = self.config.get("instance_name", "SCALP")

        if tf in ("1D", "1M"):
            return await self._pnl_daily(days, tf, instance_name, source)
        else:
            return await self._pnl_intraday(tf, instance_name, source)

    async def _pnl_daily(self, days: int, tf: str, instance_name: str, source: str = "live") -> web.Response:
        """Daily or monthly PnL from daily_pnl table."""
        limit = days if tf == "1D" else days * 30

        pnl_by_key: dict[str, dict] = {}

        # Main instance
        main_db_path = _get_db_path(str(self.db.db_path), source)
        if source == "archive":
            if Path(main_db_path).exists():
                try:
                    async with aiosqlite.connect(main_db_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute(
                            "SELECT trade_date, pnl, trades_count FROM daily_pnl ORDER BY trade_date DESC LIMIT ?",
                            (limit,),
                        )
                        rows = await cur.fetchall()
                        for r in rows:
                            raw_date = r["trade_date"]
                            key = raw_date[:7] if tf == "1M" else raw_date
                            if key not in pnl_by_key:
                                pnl_by_key[key] = {"trade_date": key, "pnl": 0, "trades_count": 0}
                            pnl_by_key[key]["pnl"] += r["pnl"]
                            pnl_by_key[key]["trades_count"] += r["trades_count"]
                            pnl_by_key[key][instance_name] = pnl_by_key[key].get(instance_name, 0) + r["pnl"]
                except Exception:
                    logger.warning("Failed to read archive PnL from %s", main_db_path)
        else:
            main_data = await self.db.get_daily_pnl_history(limit)
            for d in main_data:
                raw_date = d["trade_date"]
                key = raw_date[:7] if tf == "1M" else raw_date
                if key not in pnl_by_key:
                    pnl_by_key[key] = {"trade_date": key, "pnl": 0, "trades_count": 0}
                pnl_by_key[key]["pnl"] += d["pnl"]
                pnl_by_key[key]["trades_count"] += d["trades_count"]
                pnl_by_key[key][instance_name] = pnl_by_key[key].get(instance_name, 0) + d["pnl"]

        # Other instances
        for inst in self.config.get("other_instances", []):
            db_path = _get_db_path(inst.get("db_path", ""), source)
            inst_name = inst.get("name", "???")
            if db_path and Path(db_path).exists():
                try:
                    async with aiosqlite.connect(db_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute(
                            "SELECT trade_date, pnl, trades_count FROM daily_pnl ORDER BY trade_date DESC LIMIT ?",
                            (limit,),
                        )
                        rows = await cur.fetchall()
                        for r in rows:
                            raw_date = r["trade_date"]
                            key = raw_date[:7] if tf == "1M" else raw_date
                            if key not in pnl_by_key:
                                pnl_by_key[key] = {"trade_date": key, "pnl": 0, "trades_count": 0}
                            pnl_by_key[key]["pnl"] += r["pnl"]
                            pnl_by_key[key]["trades_count"] += r["trades_count"]
                            pnl_by_key[key][inst_name] = pnl_by_key[key].get(inst_name, 0) + r["pnl"]
                except Exception:
                    logger.warning("Failed to read PnL from %s", inst_name)

        result = sorted(pnl_by_key.values(), key=lambda x: x["trade_date"])
        return web.json_response(result)

    async def _pnl_intraday(self, tf: str, instance_name: str, source: str = "live") -> web.Response:
        """Intraday PnL from trades table (10m, 30m, 1H buckets)."""
        from datetime import datetime, timedelta

        minutes = {"1m": 1, "5m": 5, "10m": 10, "30m": 30, "1H": 60}.get(tf, 60)
        # How far back to look — for archive use much larger window
        if source == "archive":
            lookback_hours = {"1m": 24, "5m": 72, "10m": 168, "30m": 720, "1H": 2160}.get(tf, 720)
        else:
            lookback_hours = {"1m": 2, "5m": 4, "10m": 6, "30m": 24, "1H": 72}.get(tf, 24)
        since = (datetime.utcnow() - timedelta(hours=lookback_hours)).strftime("%Y-%m-%d %H:%M:%S")

        main_db_path = _get_db_path(str(self.db.db_path), source)
        all_dbs = [(main_db_path, instance_name)]
        for inst in self.config.get("other_instances", []):
            db_path = _get_db_path(inst.get("db_path", ""), source)
            if db_path and Path(db_path).exists():
                all_dbs.append((db_path, inst.get("name", "???")))

        pnl_by_bucket: dict[str, dict] = {}

        for db_path, inst_name in all_dbs:
            try:
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    cur = await db.execute(
                        "SELECT closed_at, pnl FROM trades WHERE status='closed' AND closed_at >= ? ORDER BY closed_at",
                        (since,),
                    )
                    rows = await cur.fetchall()
                    for r in rows:
                        ts = r["closed_at"]
                        if not ts:
                            continue
                        try:
                            dt = datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            try:
                                dt = datetime.fromisoformat(ts)
                            except Exception:
                                continue
                        # Round down to bucket
                        bucket_min = (dt.minute // minutes) * minutes
                        bucket_dt = dt.replace(minute=bucket_min, second=0, microsecond=0)
                        key = bucket_dt.strftime("%Y-%m-%d %H:%M")

                        if key not in pnl_by_bucket:
                            pnl_by_bucket[key] = {"trade_date": key, "pnl": 0, "trades_count": 0}
                        pnl_by_bucket[key]["pnl"] += r["pnl"] or 0
                        pnl_by_bucket[key]["trades_count"] += 1
                        pnl_by_bucket[key][inst_name] = pnl_by_bucket[key].get(inst_name, 0) + (r["pnl"] or 0)
            except Exception:
                logger.warning("Failed to read intraday PnL from %s", inst_name)

        result = sorted(pnl_by_bucket.values(), key=lambda x: x["trade_date"])
        return web.json_response(result)

    async def _api_positions(self, request: web.Request) -> web.Response:
        positions = []
        instance_name = self.config.get("instance_name", "SCALP")
        source = request.query.get("source", "live")
        is_archive = source == "archive"

        if is_archive:
            # Archive: show open trades from archive DB for main instance
            main_archive = _get_db_path(str(self.db.db_path), source)
            if Path(main_archive).exists():
                try:
                    async with aiosqlite.connect(main_archive) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute(
                            "SELECT symbol, side, entry_price, qty FROM trades WHERE status = 'open'"
                        )
                        rows = await cur.fetchall()
                        for r in rows:
                            positions.append({
                                "symbol": r["symbol"],
                                "side": r["side"],
                                "size": float(r["qty"]),
                                "entry_price": float(r["entry_price"]),
                                "unrealised_pnl": 0.0,
                                "instance": instance_name,
                            })
                except Exception:
                    logger.warning("Failed to read archive positions from %s", main_archive)
        else:
            # Live: current engine positions
            if self.engine:
                try:
                    exchange = self.config.get("exchange", "bybit")
                    if exchange == "tbank":
                        raw = self.engine.client.get_positions()
                    else:
                        raw = self.engine.client.get_positions(category="linear")
                    for p in raw:
                        positions.append({
                            "symbol": p["symbol"],
                            "side": p["side"],
                            "size": p["size"],
                            "entry_price": p["entry_price"],
                            "unrealised_pnl": p["unrealised_pnl"],
                            "instance": instance_name,
                        })
                except Exception:
                    pass

        # Other instances — open trades from DB
        for inst in self.config.get("other_instances", []):
            inst_name = inst.get("name", "???")
            db_path = _get_db_path(inst.get("db_path", ""), source)
            if db_path and Path(db_path).exists():
                try:
                    async with aiosqlite.connect(db_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute(
                            "SELECT symbol, side, entry_price, qty FROM trades WHERE status = 'open'"
                        )
                        rows = await cur.fetchall()
                        for r in rows:
                            positions.append({
                                "symbol": r["symbol"],
                                "side": r["side"],
                                "size": float(r["qty"]),
                                "entry_price": float(r["entry_price"]),
                                "unrealised_pnl": 0.0,
                                "instance": inst_name,
                            })
                except Exception:
                    logger.warning("Failed to read positions from %s", inst_name)

        return web.json_response(positions)

    async def _api_instances(self, request: web.Request) -> web.Response:
        source = request.query.get("source", "live")
        is_archive = source == "archive"
        instances = []

        # Current instance (scalper)
        if is_archive:
            archive_path = _get_db_path(str(self.db.db_path), source)
            scalp_daily = 0.0
            scalp_total = 0.0
            scalp_open_count = 0
            scalp_total_trades = 0
            scalp_wins = 0
            scalp_losses = 0
            if Path(archive_path).exists():
                try:
                    async with aiosqlite.connect(archive_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute("SELECT COALESCE(SUM(pnl), 0) as total FROM daily_pnl")
                        row = await cur.fetchone()
                        if row:
                            scalp_total = float(row["total"])
                        cur = await db.execute("SELECT COUNT(*) as cnt FROM trades WHERE status = 'open'")
                        row = await cur.fetchone()
                        if row:
                            scalp_open_count = int(row["cnt"])
                        cur = await db.execute(
                            "SELECT COUNT(*) as total, "
                            "COALESCE(SUM(CASE WHEN pnl >= 0 THEN 1 ELSE 0 END), 0) as wins, "
                            "COALESCE(SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END), 0) as losses "
                            "FROM trades WHERE status = 'closed'"
                        )
                        row = await cur.fetchone()
                        if row:
                            scalp_total_trades = int(row["total"])
                            scalp_wins = int(row["wins"])
                            scalp_losses = int(row["losses"])
                except Exception:
                    logger.warning("Failed to read archive instances from %s", archive_path)

            instances.append({
                "name": self.config.get("instance_name", "SCALP"),
                "running": False,
                "daily_pnl": scalp_daily,
                "total_pnl": scalp_total,
                "open_positions": scalp_open_count,
                "total_trades": scalp_total_trades,
                "wins": scalp_wins,
                "losses": scalp_losses,
                "win_rate": (scalp_wins / scalp_total_trades * 100) if scalp_total_trades > 0 else 0,
                "timeframe": str(self.config["trading"]["timeframe"]),
                "leverage": self.config["trading"].get("leverage", 1),
                "pairs": self.config["trading"]["pairs"],
            })
        else:
            scalp_daily = await self.db.get_daily_pnl()
            scalp_total = await self.db.get_total_pnl()
            scalp_stats = await self.db.get_trade_stats()
            scalp_open = await self.db.get_open_trades()
            running = self.engine._running if self.engine else False

            instances.append({
                "name": self.config.get("instance_name", "SCALP"),
                "running": running,
                "daily_pnl": scalp_daily,
                "total_pnl": scalp_total,
                "open_positions": len(scalp_open),
                "total_trades": scalp_stats["total"],
                "wins": scalp_stats["wins"],
                "losses": scalp_stats["losses"],
                "win_rate": (scalp_stats["wins"] / scalp_stats["total"] * 100) if scalp_stats["total"] > 0 else 0,
                "timeframe": str(self.config["trading"]["timeframe"]),
                "leverage": self.config["trading"].get("leverage", 1),
                "pairs": self.config["trading"]["pairs"],
            })

        # Other instances
        for inst in self.config.get("other_instances", []):
            data = await self._read_instance_data(inst, source)
            instances.append(data)

        return web.json_response(instances)

    async def _read_instance_data(self, inst: dict, source: str = "live") -> dict:
        name = inst.get("name", "???")
        db_path = _get_db_path(inst.get("db_path", ""), source)
        service = inst.get("service", "")

        # Check service status (only for live)
        running = False
        if service and source != "archive":
            try:
                result = subprocess.run(
                    ["systemctl", "is-active", service],
                    capture_output=True, text=True, timeout=3,
                )
                running = result.stdout.strip() == "active"
            except Exception:
                pass

        daily = 0.0
        total = 0.0
        open_count = 0
        total_trades = 0
        wins = 0
        losses = 0

        if db_path and Path(db_path).exists():
            try:
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    cur = await db.execute(
                        "SELECT pnl FROM daily_pnl WHERE trade_date = ?",
                        (date.today().isoformat(),),
                    )
                    row = await cur.fetchone()
                    if row:
                        daily = float(row["pnl"])

                    cur = await db.execute("SELECT COALESCE(SUM(pnl), 0) as total FROM daily_pnl")
                    row = await cur.fetchone()
                    if row:
                        total = float(row["total"])

                    cur = await db.execute("SELECT COUNT(*) as cnt FROM trades WHERE status = 'open'")
                    row = await cur.fetchone()
                    if row:
                        open_count = int(row["cnt"])

                    cur = await db.execute(
                        "SELECT COUNT(*) as total, "
                        "COALESCE(SUM(CASE WHEN pnl >= 0 THEN 1 ELSE 0 END), 0) as wins, "
                        "COALESCE(SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END), 0) as losses "
                        "FROM trades WHERE status = 'closed'"
                    )
                    row = await cur.fetchone()
                    if row:
                        total_trades = int(row["total"])
                        wins = int(row["wins"])
                        losses = int(row["losses"])
            except Exception:
                logger.warning("Failed to read DB for instance %s", name)

        return {
            "name": name,
            "running": running,
            "daily_pnl": daily,
            "total_pnl": total,
            "open_positions": open_count,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / total_trades * 100) if total_trades > 0 else 0,
            "timeframe": inst.get("timeframe", "?"),
            "leverage": inst.get("leverage", "?"),
            "pairs": inst.get("pairs", []),
        }


# ── Login page ────────────────────────────────────────────────────

_LOGIN_HTML = """<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="utf-8">
<title>Stasik — Вход</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:'Segoe UI',-apple-system,sans-serif;
  background:linear-gradient(135deg,#0f0f23 0%,#1a1a3e 50%,#0f0f23 100%);
  min-height:100vh;display:flex;align-items:center;justify-content:center;
  color:#e0e0e0;
}
.login-box{
  background:rgba(26,26,62,0.85);backdrop-filter:blur(20px);
  border:1px solid rgba(255,255,255,0.06);
  border-radius:20px;padding:40px 36px;width:100%;max-width:400px;
  box-shadow:0 20px 60px rgba(0,0,0,0.5);
}
.logo{text-align:center;margin-bottom:28px}
.logo .icon{font-size:48px;display:block;margin-bottom:8px}
.logo h1{font-size:22px;font-weight:700;color:#fff;letter-spacing:1px}
.logo p{font-size:13px;color:#666;margin-top:4px}
.error{
  background:rgba(255,82,82,0.15);color:#ff5252;
  padding:10px 14px;border-radius:10px;font-size:13px;
  margin-bottom:16px;text-align:center;
}
label{font-size:12px;color:#888;text-transform:uppercase;letter-spacing:1px;display:block;margin-bottom:6px}
input[type=text],input[type=password]{
  width:100%;padding:12px 14px;border:1px solid rgba(255,255,255,0.08);
  border-radius:10px;background:rgba(15,15,35,0.6);color:#fff;
  font-size:15px;outline:none;transition:border 0.2s;margin-bottom:16px;
}
input:focus{border-color:#6366f1}
.captcha-row{display:flex;align-items:center;gap:10px;margin-bottom:16px}
.captcha-q{
  background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.2);
  border-radius:10px;padding:10px 16px;font-size:16px;font-weight:600;
  color:#a5b4fc;white-space:nowrap;
}
.captcha-row input{margin-bottom:0;flex:1}
button{
  width:100%;padding:13px;border:none;border-radius:10px;
  background:linear-gradient(135deg,#6366f1,#8b5cf6);
  color:#fff;font-size:15px;font-weight:600;cursor:pointer;
  transition:opacity 0.2s,transform 0.1s;
}
button:hover{opacity:0.9}
button:active{transform:scale(0.98)}
button:disabled{opacity:0.4;cursor:not-allowed}
</style>
</head><body>
<div class="login-box">
  <div class="logo">
    <div class="icon">🤖</div>
    <h1>Stasik Trading Bot</h1>
    <p>Авторизация в панели управления</p>
  </div>
  {{error}}
  <form method="POST" action="/login">
    <label>Логин</label>
    <input type="text" name="username" autocomplete="username" required {{blocked}}>
    <label>Пароль</label>
    <input type="password" name="password" autocomplete="current-password" required {{blocked}}>
    <label>Капча: сколько будет {{captcha_question}} ?</label>
    <div class="captcha-row">
      <div class="captcha-q">{{captcha_question}} = ?</div>
      <input type="text" name="captcha" inputmode="numeric" required {{blocked}}>
    </div>
    <input type="hidden" name="captcha_token" value="{{captcha_token}}">
    <button type="submit" {{blocked}}>Войти</button>
  </form>
</div>
</body></html>"""


# ── Main dashboard HTML ───────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="utf-8">
<title>Stasik Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2/dist/chartjs-plugin-datalabels.min.js"></script>
<style>
:root{
  --bg:#0f0f23;--bg2:#1a1a3e;--bg3:#12122e;
  --green:#00e676;--red:#ff5252;--purple:#6366f1;
  --text:#e0e0e0;--muted:#888;--border:rgba(255,255,255,0.06);
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',-apple-system,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}

.header{
  background:linear-gradient(90deg,var(--bg2),rgba(99,102,241,0.15));
  border-bottom:1px solid var(--border);
  padding:16px 24px;display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:100;backdrop-filter:blur(12px);
}
.header-left{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.header .icon{font-size:32px}
.header h1{font-size:20px;font-weight:700;color:#fff;white-space:nowrap}
.header h1 span{color:var(--purple);font-weight:400}
.header-stats{font-size:13px;color:var(--muted);display:flex;align-items:center;gap:6px;border-left:1px solid rgba(255,255,255,0.1);padding-left:12px;white-space:nowrap}
.header-stats .hs-val{color:#fff;font-weight:600}
.header-stats .hs-sep{color:rgba(255,255,255,0.15);margin:0 4px}
.header-right{display:flex;align-items:center;gap:16px}
#status-badge{
  padding:5px 14px;border-radius:20px;font-size:12px;font-weight:600;letter-spacing:0.5px;
}
#status-badge.on{background:rgba(0,230,118,0.15);color:var(--green);border:1px solid rgba(0,230,118,0.3)}
#status-badge.off{background:rgba(255,82,82,0.15);color:var(--red);border:1px solid rgba(255,82,82,0.3)}
.logout-btn{
  background:rgba(255,255,255,0.06);border:1px solid var(--border);
  color:#aaa;padding:6px 16px;border-radius:10px;font-size:13px;
  text-decoration:none;transition:all 0.2s;
}
.logout-btn:hover{background:rgba(255,82,82,0.15);color:var(--red);border-color:rgba(255,82,82,0.3)}

.container{max-width:1200px;margin:0 auto;padding:24px}

.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-bottom:24px}
.card{
  background:var(--bg2);border:1px solid var(--border);border-radius:16px;
  padding:20px;transition:transform 0.2s,box-shadow 0.2s;position:relative;overflow:hidden;
}
.card:hover{transform:translateY(-2px);box-shadow:0 8px 30px rgba(0,0,0,0.3)}
.card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--purple),transparent);border-radius:16px 16px 0 0;
}
.card .card-icon{font-size:24px;margin-bottom:8px;display:block}
.card h3{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1.2px}
.card .val{font-size:26px;font-weight:700;margin-top:6px;transition:all 0.3s}
.g{color:var(--green)}.r{color:var(--red)}

.chart-section{
  background:var(--bg2);border:1px solid var(--border);border-radius:16px;
  padding:24px;margin-bottom:24px;
}
.chart-section h2{font-size:16px;color:#fff;margin-bottom:16px;display:flex;align-items:center;gap:8px}

.section{
  background:var(--bg3);border:1px solid var(--border);border-radius:16px;
  padding:20px;margin-bottom:24px;
}
.section h2{font-size:16px;color:#fff;margin-bottom:14px;display:flex;align-items:center;gap:8px}

table{width:100%;border-collapse:collapse}
th{text-align:left;padding:10px 12px;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:1px;border-bottom:2px solid var(--bg2)}
td{padding:10px 12px;font-size:13px;border-bottom:1px solid rgba(255,255,255,0.03)}
tr{transition:background 0.2s}
tr:hover{background:rgba(99,102,241,0.06)}
.side-long{color:var(--green);font-weight:600}
.side-short{color:var(--red);font-weight:600}
.status-closed{color:var(--muted)}
.status-open{color:var(--purple);font-weight:600}
.inst-tag{font-size:10px;padding:2px 8px;border-radius:8px;font-weight:600;letter-spacing:0.5px}
.inst-scalp{background:rgba(99,102,241,0.15);color:#a5b4fc}
.inst-swing{background:rgba(245,158,11,0.15);color:#fbbf24}
.inst-degen{background:rgba(236,72,153,0.15);color:#f472b6}
.inst-tbank{background:rgba(16,185,129,0.15);color:#34d399}
.tbl-wrap{overflow-x:auto}

.pagination{display:flex;align-items:center;justify-content:center;gap:12px;margin-top:14px}
.pagination button{
  background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.2);
  color:var(--purple);padding:8px 18px;border-radius:8px;font-size:13px;
  cursor:pointer;transition:all 0.2s;
}
.pagination button:hover{background:rgba(99,102,241,0.25)}
.pagination button:disabled{opacity:0.3;cursor:not-allowed}
.pagination .page-info{color:var(--muted);font-size:13px}

.instances{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;margin-bottom:20px;
}
.instance-card{
  background:var(--bg2);border:1px solid var(--border);border-radius:16px;
  padding:20px;position:relative;overflow:hidden;
}
.instance-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:16px 16px 0 0;
}
.instance-card.scalp::before{background:linear-gradient(90deg,#6366f1,#a78bfa)}
.instance-card.swing::before{background:linear-gradient(90deg,#f59e0b,#fbbf24)}
.instance-card.degen::before{background:linear-gradient(90deg,#ec4899,#f472b6)}
.instance-card.tbank::before{background:linear-gradient(90deg,#10b981,#34d399)}
.instance-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}
.instance-name{font-size:18px;font-weight:700;color:#fff;display:flex;align-items:center;gap:8px}
.instance-name .badge{
  font-size:10px;padding:3px 10px;border-radius:12px;font-weight:600;letter-spacing:0.5px;
}
.badge-scalp{background:rgba(99,102,241,0.15);color:#a5b4fc;border:1px solid rgba(99,102,241,0.3)}
.badge-swing{background:rgba(245,158,11,0.15);color:#fbbf24;border:1px solid rgba(245,158,11,0.3)}
.badge-degen{background:rgba(236,72,153,0.15);color:#f472b6;border:1px solid rgba(236,72,153,0.3)}
.badge-tbank{background:rgba(16,185,129,0.15);color:#34d399;border:1px solid rgba(16,185,129,0.3)}
.instance-status{font-size:12px;font-weight:600;padding:4px 12px;border-radius:12px}
.instance-status.on{background:rgba(0,230,118,0.12);color:var(--green)}
.instance-status.off{background:rgba(255,82,82,0.12);color:var(--red)}
.instance-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.instance-stat h4{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
.instance-stat .val{font-size:20px;font-weight:700}
.instance-meta{margin-top:12px;font-size:11px;color:var(--muted);display:flex;gap:14px;flex-wrap:wrap}

@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.fade-in{animation:fadeIn 0.4s ease}
.tf-buttons{display:flex;gap:4px}
.tf-buttons button{
  background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);
  color:#a5b4fc;padding:6px 14px;border-radius:8px;font-size:12px;font-weight:600;
  cursor:pointer;transition:all 0.2s;
}
.tf-buttons button:hover{background:rgba(99,102,241,0.25)}
.tf-buttons button.active{background:rgba(99,102,241,0.35);border-color:#6366f1;color:#fff}

.chart-dual-wrap{display:flex;flex-direction:column;gap:16px}
.chart-half{background:var(--bg3);border:1px solid var(--border);border-radius:12px;padding:16px;position:relative}
.chart-half-label{font-size:12px;font-weight:600;margin-bottom:10px;display:flex;align-items:center;gap:6px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.dot-bybit{background:#818cf8}
.dot-degen{background:#f472b6}
.dot-tbank{background:#34d399}
.chart-legend-custom{display:flex;gap:12px;font-size:12px;color:var(--muted)}
.chart-legend-custom .leg-item{display:flex;align-items:center;gap:5px}
.leg-bar{width:12px;height:10px;border-radius:2px}
.leg-line{width:14px;height:2px;border-radius:1px}

.source-toggle{display:flex;gap:0;border-radius:10px;overflow:hidden;border:1px solid rgba(99,102,241,0.2)}
.source-toggle button{
  background:rgba(99,102,241,0.08);border:none;color:#a5b4fc;padding:6px 16px;
  font-size:12px;font-weight:600;cursor:pointer;transition:all 0.2s;
}
.source-toggle button:hover{background:rgba(99,102,241,0.2)}
.source-toggle button.active{background:rgba(99,102,241,0.35);color:#fff}
.source-toggle button.archive-active{background:rgba(245,158,11,0.35);color:#fbbf24}
.archive-banner{
  background:linear-gradient(90deg,rgba(245,158,11,0.15),rgba(245,158,11,0.05));
  border:1px solid rgba(245,158,11,0.3);border-radius:10px;
  padding:10px 20px;margin-bottom:16px;text-align:center;
  color:#fbbf24;font-size:14px;font-weight:600;
}
body.archive-mode{background:linear-gradient(135deg,#1a1508 0%,#1a1a3e 50%,#1a1508 100%)}
body.archive-mode .header{background:linear-gradient(90deg,var(--bg2),rgba(245,158,11,0.15))}

.last-update{text-align:center;color:#555;font-size:11px;padding:16px}

@media(max-width:600px){
  .container{padding:14px}
  .cards{grid-template-columns:repeat(2,1fr);gap:10px}
  .card{padding:14px}
  .card .val{font-size:20px}
  .header{padding:12px 16px;flex-wrap:wrap;gap:10px}
  .header h1{font-size:16px}
}
</style>
</head><body>

<div class="header">
  <div class="header-left">
    <div class="icon">🤖</div>
    <h1>Stasik <span>Trading Bot</span></h1>
    <div class="header-stats" id="header-stats"></div>
  </div>
  <div class="header-right">
    <div class="source-toggle" id="source-toggle">
      <button class="active" onclick="setSource('live')" data-source="live">Live</button>
      <button onclick="setSource('archive')" data-source="archive">Архив</button>
    </div>
    <div id="status-badge" class="off">...</div>
    <a href="/logout" class="logout-btn">Выход</a>
  </div>
</div>

<div class="container">
  <div class="archive-banner" id="archive-banner" style="display:none">АРХИВ — данные до перехода на стратегию Котегавы</div>
  <div class="instances" id="instances"></div>

  <div class="chart-section">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;flex-wrap:wrap;gap:10px">
      <h2 style="margin:0">📈 PnL</h2>
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
        <div class="chart-legend-custom" id="chart-legend"></div>
        <div class="tf-buttons" id="tf-buttons">
          <button class="active" onclick="setTF('1m')" data-tf="1m">1м</button>
          <button onclick="setTF('5m')" data-tf="5m">5м</button>
          <button onclick="setTF('10m')" data-tf="10m">10м</button>
          <button onclick="setTF('30m')" data-tf="30m">30м</button>
          <button onclick="setTF('1H')" data-tf="1H">1ч</button>
          <button onclick="setTF('1D')" data-tf="1D">1д</button>
          <button onclick="setTF('1M')" data-tf="1M">1мес</button>
        </div>
      </div>
    </div>
    <div class="chart-dual-wrap" id="chart-wrap">
      <div class="chart-half"><div class="chart-half-label"><span class="dot dot-bybit"></span>Bybit (USDT)</div><canvas id="pnlChartBybit"></canvas></div>
      <div class="chart-half" id="degen-chart-half" style="display:none"><div class="chart-half-label"><span class="dot dot-degen"></span>DEGEN Memes (USDT)</div><canvas id="pnlChartDegen"></canvas></div>
      <div class="chart-half" id="tbank-chart-half" style="display:none"><div class="chart-half-label"><span class="dot dot-tbank"></span>T-Bank (RUB)</div><canvas id="pnlChartTbank"></canvas></div>
    </div>
  </div>

  <div class="section">
    <h2>📊 Открытые позиции</h2>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Бот</th><th>Пара</th><th>Сторона</th><th>Размер</th><th>Вход</th><th>Unrealized PnL</th></tr></thead>
        <tbody id="pos-body"></tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <h2>📋 История сделок</h2>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Бот</th><th>Пара</th><th>Сторона</th><th>Вход</th><th>Выход</th><th>PnL</th><th>Статус</th>
        </tr></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
    <div class="pagination">
      <button id="prev-btn" disabled onclick="changePage(-1)">← Назад</button>
      <span class="page-info" id="page-info">Стр. 1</span>
      <button id="next-btn" disabled onclick="changePage(1)">Вперёд →</button>
    </div>
  </div>

  <div class="last-update" id="last-update"></div>
</div>

<script>
let currentPage=1,hasNext=false,currentTF='1m',currentSource='live';
function fmt(v){return(v>=0?'+':'')+v.toFixed(2)}
function cls(v){return v>=0?'g':'r'}
function setTF(tf){
  currentTF=tf;
  document.querySelectorAll('.tf-buttons button').forEach(b=>b.classList.toggle('active',b.dataset.tf===tf));
  loadChart();
}
function setSource(src){
  currentSource=src;
  currentPage=1;
  document.querySelectorAll('.source-toggle button').forEach(b=>{
    b.classList.remove('active','archive-active');
    if(b.dataset.source===src){b.classList.add(src==='archive'?'archive-active':'active')}
  });
  const banner=document.getElementById('archive-banner');
  const badge=document.getElementById('status-badge');
  if(src==='archive'){
    banner.style.display='';
    badge.style.display='none';
    document.body.classList.add('archive-mode');
  }else{
    banner.style.display='none';
    badge.style.display='';
    document.body.classList.remove('archive-mode');
  }
  loadAll();
}

async function loadInstances(){
  try{
    const list=await(await fetch('/api/instances?source='+currentSource)).json();
    _hasTbankInst=list.some(i=>i.name.toLowerCase().includes('tbank'));
    _hasDegenInst=list.some(i=>i.name.toLowerCase().includes('degen'));
    const el=document.getElementById('instances');
    el.innerHTML=list.map(i=>{
      const key=i.name.toLowerCase();
      const isTbank=key.includes('tbank');
      const isDegen=key.includes('degen');
      const isScalp=key.includes('scalp')&&!isTbank;
      const isSwing=key.includes('swing')&&!isTbank;
      const cardCls=isTbank?'tbank':isDegen?'degen':isScalp?'scalp':'swing';
      const badgeCls=isTbank?'badge-tbank':isDegen?'badge-degen':isScalp?'badge-scalp':'badge-swing';
      const tf=(isScalp||isDegen)?i.timeframe+'м':i.timeframe;
      const cur=isTbank?'RUB':'USDT';
      const icon=isTbank?'🏦':isDegen?'🎰':isScalp?'⚡':'🌊';
      return`<div class="instance-card ${cardCls} fade-in">
        <div class="instance-header">
          <div class="instance-name">
            ${icon}
            <span>${i.name}</span>
            <span class="badge ${badgeCls}">${tf} / ${i.leverage}x</span>
          </div>
          <div class="instance-status ${i.running?'on':'off'}">${i.running?'РАБОТАЕТ':'СТОП'}</div>
        </div>
        <div class="instance-stats">
          <div class="instance-stat"><h4>За день</h4><div class="val ${cls(i.daily_pnl)}">${fmt(i.daily_pnl)} ${cur}</div></div>
          <div class="instance-stat"><h4>Всего PnL</h4><div class="val ${cls(i.total_pnl)}">${fmt(i.total_pnl)} ${cur}</div></div>
          <div class="instance-stat"><h4>Win Rate</h4><div class="val">${i.total_trades?i.win_rate.toFixed(1)+'%':'—'}</div></div>
        </div>
        <div class="instance-meta">
          <span>📊 Позиции: ${i.open_positions}</span>
          <span>🔢 Сделок: ${i.total_trades} (${i.wins}W / ${i.losses}L)</span>
          <span>🪙 ${i.pairs.length} пар</span>
        </div>
      </div>`;
    }).join('');
  }catch(e){console.error('instances',e)}
}

async function loadStats(){
  try{
    const s=await(await fetch('/api/stats?source='+currentSource)).json();
    const b=document.getElementById('status-badge');
    b.className=s.running?'on':'off';
    b.textContent=s.running?'РАБОТАЕТ':'ОСТАНОВЛЕН';
    const tb=s.tbank_balance||0;
    const hs=document.getElementById('header-stats');
    hs.innerHTML=`Баланс: <span class="hs-val">${s.balance.toLocaleString('ru-RU',{maximumFractionDigits:0})} USDT</span>`
      +(tb?` / <span class="hs-val">${tb.toLocaleString('ru-RU',{maximumFractionDigits:0})} RUB</span>`:'')
      +`<span class="hs-sep">|</span>сделок: <span class="hs-val">${s.all_trades||s.total_trades}</span>`;
  }catch(e){console.error('stats',e)}
}

let chartBybit=null, chartDegen=null, chartTbank=null, _hasTbankInst=false, _hasDegenInst=false;

function buildWaterfall(canvasId, labels, dailyArr, posColor, negColor, currency, sym){
  const ctx=document.getElementById(canvasId).getContext('2d');

  // Build floating bar ranges: each bar floats from cumBefore → cumAfter
  const ranges=[];
  let cum=0;
  dailyArr.forEach(v=>{
    const from=cum;
    cum+=v;
    ranges.push([Math.min(from,cum), Math.max(from,cum)]);
  });
  const colors=dailyArr.map(v=>v>=0?posColor:negColor);

  // Connector lines dataset (thin lines connecting bars)
  const connectors=[];
  let c2=0;
  dailyArr.forEach((v,i)=>{
    c2+=v;
    connectors.push(c2);
  });

  return new Chart(ctx,{type:'bar',
    plugins:[ChartDataLabels,{
      id:'waterfall-connectors',
      afterDatasetsDraw(chart){
        const meta=chart.getDatasetMeta(0);
        const ctx2=chart.ctx;
        ctx2.save();
        ctx2.setLineDash([3,3]);
        ctx2.strokeStyle='rgba(255,255,255,0.15)';
        ctx2.lineWidth=1;
        for(let i=0;i<meta.data.length-1;i++){
          const bar=meta.data[i];
          const next=meta.data[i+1];
          // connect end of current bar to start of next
          const endY=dailyArr[i]>=0?bar.y:bar.base;
          ctx2.beginPath();
          ctx2.moveTo(bar.x,endY);
          ctx2.lineTo(next.x,endY);
          ctx2.stroke();
        }
        ctx2.restore();
      }
    }],
    data:{labels,datasets:[{
      data:ranges,
      backgroundColor:colors,
      borderRadius:3,
      borderSkipped:false,
      datalabels:{
        display:c=>dailyArr[c.dataIndex]!==0,
        formatter:(v,c)=>{const d=dailyArr[c.dataIndex];return(d>=0?'+':'')+d.toFixed(0)},
        color:c=>dailyArr[c.dataIndex]>=0?'#00e676':'#ff5252',
        anchor:c=>dailyArr[c.dataIndex]>=0?'end':'start',
        align:c=>dailyArr[c.dataIndex]>=0?'top':'bottom',
        font:{size:10,weight:'bold'},
      }
    }]},
    options:{responsive:true,maintainAspectRatio:true,
      interaction:{intersect:false,mode:'index'},
      plugins:{
        legend:{display:false},
        datalabels:{},
        tooltip:{
          backgroundColor:'rgba(26,26,62,0.95)',titleColor:'#fff',bodyColor:'#e0e0e0',
          borderColor:'rgba(99,102,241,0.3)',borderWidth:1,cornerRadius:10,padding:12,
          callbacks:{
            label:c=>{
              const i=c.dataIndex;
              const d=dailyArr[i];
              const total=connectors[i];
              return[
                'PnL: '+(d>=0?'+':'')+d.toFixed(2)+' '+currency,
                'Итого: '+(total>=0?'+':'')+total.toFixed(2)+' '+currency
              ];
            }
          }
        }
      },
      scales:{
        x:{ticks:{color:'#555',maxRotation:45,font:{size:11}},grid:{display:false}},
        y:{
          ticks:{color:'#888',callback:v=>(v>=0?'+':'')+v.toFixed(0)+' '+sym,font:{size:10}},
          grid:{color:c=>c.tick.value===0?'rgba(255,255,255,0.35)':'rgba(255,255,255,0.04)',lineWidth:c=>c.tick.value===0?2:1},
        }
      }
    }
  });
}

function buildLineChart(canvasId, labels, dailyArr, lineColor, fillColor, currency, sym){
  const ctx=document.getElementById(canvasId).getContext('2d');
  // Build cumulative PnL
  const cumulative=[];
  let cum=0;
  dailyArr.forEach(v=>{cum+=v;cumulative.push(cum)});

  // Gradient fill
  const gradient=ctx.createLinearGradient(0,0,0,300);
  gradient.addColorStop(0,fillColor);
  gradient.addColorStop(1,'rgba(0,0,0,0)');

  return new Chart(ctx,{type:'line',
    plugins:[ChartDataLabels],
    data:{labels,datasets:[{
      data:cumulative,
      borderColor:lineColor,
      backgroundColor:gradient,
      fill:true,
      tension:0.3,
      borderWidth:2.5,
      pointRadius:dailyArr.length>20?0:4,
      pointHoverRadius:6,
      pointBackgroundColor:lineColor,
      pointBorderColor:'#1a1a3e',
      pointBorderWidth:2,
      datalabels:{
        display:c=>{
          const len=cumulative.length;
          if(len<=10)return true;
          if(len<=20)return c.dataIndex%2===0||c.dataIndex===len-1;
          return c.dataIndex===0||c.dataIndex===len-1||c.dataIndex%Math.ceil(len/8)===0;
        },
        formatter:v=>(v>=0?'+':'')+v.toFixed(0),
        color:c=>cumulative[c.dataIndex]>=0?'#00e676':'#ff5252',
        anchor:'end',align:'top',
        font:{size:10,weight:'bold'},
        offset:4,
      }
    },{
      // Daily PnL as thin bars behind the line
      type:'bar',
      data:dailyArr,
      backgroundColor:dailyArr.map(v=>v>=0?'rgba(244,114,182,0.3)':'rgba(255,82,82,0.3)'),
      borderRadius:2,
      datalabels:{display:false},
    }]},
    options:{responsive:true,maintainAspectRatio:true,
      interaction:{intersect:false,mode:'index'},
      plugins:{
        legend:{display:false},
        datalabels:{},
        tooltip:{
          backgroundColor:'rgba(26,26,62,0.95)',titleColor:'#fff',bodyColor:'#e0e0e0',
          borderColor:'rgba(236,72,153,0.3)',borderWidth:1,cornerRadius:10,padding:12,
          callbacks:{
            label:c=>{
              if(c.datasetIndex===0){
                return 'Кумулятивный: '+(c.raw>=0?'+':'')+c.raw.toFixed(2)+' '+currency;
              }
              const d=dailyArr[c.dataIndex];
              return 'За период: '+(d>=0?'+':'')+d.toFixed(2)+' '+currency;
            }
          }
        }
      },
      scales:{
        x:{ticks:{color:'#555',maxRotation:45,font:{size:11}},grid:{display:false}},
        y:{
          ticks:{color:'#888',callback:v=>(v>=0?'+':'')+v.toFixed(0)+' '+sym,font:{size:10}},
          grid:{color:c=>c.tick.value===0?'rgba(255,255,255,0.35)':'rgba(255,255,255,0.04)',lineWidth:c=>c.tick.value===0?2:1},
        }
      }
    }
  });
}

async function loadChart(){
  try{
    const pnl=await(await fetch('/api/pnl?days=30&tf='+currentTF+'&source='+currentSource)).json();
    // Always destroy old charts first
    if(chartBybit){chartBybit.destroy();chartBybit=null}
    if(chartDegen){chartDegen.destroy();chartDegen=null}
    if(chartTbank){chartTbank.destroy();chartTbank=null}
    if(!pnl.length){
      // No data — show placeholder text on canvases
      ['pnlChartBybit','pnlChartDegen','pnlChartTbank'].forEach(id=>{
        const c=document.getElementById(id);
        const cx=c.getContext('2d');cx.clearRect(0,0,c.width,c.height);
        cx.fillStyle='#555';cx.font='14px sans-serif';cx.textAlign='center';
        cx.fillText('Нет данных за выбранный период',c.width/2,c.height/2);
      });
      document.getElementById('degen-chart-half').style.display=_hasDegenInst?'':'none';
      document.getElementById('tbank-chart-half').style.display=_hasTbankInst?'':'none';
      return;
    }
    const labels=pnl.map(d=>{
      const dt=d.trade_date;
      if(currentTF==='1M')return dt;
      if(currentTF==='1D')return dt.slice(5);
      return dt.length>10?dt.slice(11,16):dt;
    });

    const bybitDaily=[], degenDaily=[], tbankDaily=[];
    let hasTbank=false, hasDegen=false;
    pnl.forEach(d=>{
      let bPnl=0, dPnl=0, tPnl=0;
      Object.keys(d).forEach(k=>{
        if(k==='trade_date'||k==='pnl'||k==='trades_count')return;
        const ku=k.toUpperCase();
        if(ku.includes('TBANK')){tPnl+=d[k];hasTbank=true}
        else if(ku.includes('DEGEN')){dPnl+=d[k];hasDegen=true}
        else{bPnl+=d[k]}
      });
      if(!Object.keys(d).some(k=>!['trade_date','pnl','trades_count'].includes(k))){bPnl=d.pnl}
      bybitDaily.push(bPnl);degenDaily.push(dPnl);tbankDaily.push(tPnl);
    });

    chartBybit=buildWaterfall('pnlChartBybit',labels,bybitDaily,
      'rgba(129,140,248,0.75)','rgba(255,82,82,0.75)','USDT','$');

    const dgHalf=document.getElementById('degen-chart-half');
    if(hasDegen||_hasDegenInst){
      dgHalf.style.display='';
      if(hasDegen){
        chartDegen=buildLineChart('pnlChartDegen',labels,degenDaily,
          '#f472b6','rgba(244,114,182,0.15)','USDT','$');
      }else{
        const c=document.getElementById('pnlChartDegen');
        const cx=c.getContext('2d');cx.clearRect(0,0,c.width,c.height);
        cx.fillStyle='#555';cx.font='14px sans-serif';cx.textAlign='center';
        cx.fillText('DEGEN ещё не торговал',c.width/2,c.height/2);
      }
    }else{dgHalf.style.display='none'}

    const tbHalf=document.getElementById('tbank-chart-half');
    if(hasTbank||_hasTbankInst){
      tbHalf.style.display='';
      if(hasTbank){
        chartTbank=buildWaterfall('pnlChartTbank',labels,tbankDaily,
          'rgba(52,211,153,0.75)','rgba(255,82,82,0.75)','RUB','₽');
      }else{
        const c=document.getElementById('pnlChartTbank');
        const cx=c.getContext('2d');cx.clearRect(0,0,c.width,c.height);
        cx.fillStyle='#555';cx.font='14px sans-serif';cx.textAlign='center';
        cx.fillText('Нет данных TBank за выбранный период',c.width/2,c.height/2);
      }
    }else{
      tbHalf.style.display='none';
    }

    // Legend
    document.getElementById('chart-legend').innerHTML=`
      <span class="leg-item"><span class="leg-bar" style="background:rgba(129,140,248,0.75)"></span>Профит</span>
      <span class="leg-item"><span class="leg-bar" style="background:rgba(255,82,82,0.75)"></span>Убыток</span>
      <span class="leg-item"><span class="leg-line" style="background:#f472b6"></span>DEGEN кумулятив</span>
    `;
  }catch(e){console.error('chart',e)}
}

async function loadPositions(){
  try{
    const pos=await(await fetch('/api/positions?source='+currentSource)).json();
    const body=document.getElementById('pos-body');
    if(!pos.length){body.innerHTML='<tr><td colspan="6" style="text-align:center;color:#555;padding:20px">Нет открытых позиций</td></tr>';return}
    body.innerHTML=pos.map(p=>{
      const pnl=parseFloat(p.unrealised_pnl)||0;
      const inst=(p.instance||'').toUpperCase();
      const isCls=inst.includes('TBANK')?'inst-tbank':inst.includes('DEGEN')?'inst-degen':inst.includes('SCALP')?'inst-scalp':'inst-swing';
      const iLabel=inst.includes('TBANK-SCALP')?'TB-SCALP':inst.includes('TBANK-SWING')?'TB-SWING':inst.includes('DEGEN')?'DEGEN':inst.includes('SCALP')?'SCALP':'SWING';
      const cur=inst.includes('TBANK')?'RUB':'USDT';
      return`<tr class="fade-in">
        <td><span class="inst-tag ${isCls}">${iLabel}</span></td>
        <td><strong>${p.symbol}</strong></td>
        <td class="${p.side==='Buy'?'side-long':'side-short'}">${p.side==='Buy'?'ЛОНГ':'ШОРТ'}</td>
        <td>${p.size}</td><td>${parseFloat(p.entry_price).toFixed(2)}</td>
        <td class="${cls(pnl)}"><strong>${fmt(pnl)} ${cur}</strong></td></tr>`;
    }).join('');
  }catch(e){console.error('positions',e)}
}

async function loadTrades(page){
  try{
    const r=await(await fetch('/api/trades?page='+page+'&per_page=20&source='+currentSource)).json();
    currentPage=r.page;hasNext=r.has_next;
    document.getElementById('prev-btn').disabled=currentPage<=1;
    document.getElementById('next-btn').disabled=!hasNext;
    document.getElementById('page-info').textContent='Стр. '+currentPage;
    document.getElementById('tbody').innerHTML=r.trades.map(t=>{
      const p=t.pnl||0;
      const inst=(t.instance||'').toUpperCase();
      const isCls=inst.includes('TBANK')?'inst-tbank':inst.includes('DEGEN')?'inst-degen':inst.includes('SCALP')?'inst-scalp':'inst-swing';
      const iLabel=inst.includes('TBANK-SCALP')?'TB-SCALP':inst.includes('TBANK-SWING')?'TB-SWING':inst.includes('DEGEN')?'DEGEN':inst.includes('SCALP')?'SCALP':'SWING';
      return`<tr class="fade-in">
        <td><span class="inst-tag ${isCls}">${iLabel}</span></td>
        <td><strong>${t.symbol}</strong></td>
        <td class="${t.side==='Buy'?'side-long':'side-short'}">${t.side==='Buy'?'ЛОНГ':'ШОРТ'}</td>
        <td>${t.entry_price||'-'}</td><td>${t.exit_price||'-'}</td>
        <td class="${cls(p)}"><strong>${p?p.toFixed(2):'-'}</strong></td>
        <td class="${t.status==='closed'?'status-closed':'status-open'}">${t.status}</td></tr>`;
    }).join('');
  }catch(e){console.error('trades',e)}
}

function changePage(d){loadTrades(currentPage+d)}

async function loadAll(){
  await loadInstances();
  await Promise.all([loadStats(),loadChart(),loadPositions(),loadTrades(currentPage)]);
  document.getElementById('last-update').textContent=
    'Обновлено: '+new Date().toLocaleTimeString('ru-RU')+' (автообновление каждые 30 сек)';
}
loadAll();
setInterval(async()=>{await loadInstances();loadStats();loadChart();loadPositions()},30000);
</script>
</body></html>"""
