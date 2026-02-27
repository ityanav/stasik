import asyncio
import logging
import subprocess
import time
from pathlib import Path

from aiohttp import web

from src.dashboard.auth import AuthManager, COOKIE_NAME
from src.dashboard.night_mode import NightModeMixin
from src.dashboard.sale_mode import SaleModeMixin
from src.dashboard.routes_stats import RouteStatsMixin
from src.dashboard.routes_control import RouteControlMixin
from src.dashboard.routes_settings import RouteSettingsMixin
from src.dashboard.services import _get_db_path, _other_instances, _ARCHIVE_MAP
from src.storage.database import Database

logger = logging.getLogger(__name__)

# Load HTML templates once at import time
_TEMPLATES_DIR = Path(__file__).parent / "templates"
_LOGIN_HTML = (_TEMPLATES_DIR / "login.html").read_text(encoding="utf-8")
_DASHBOARD_HTML = (_TEMPLATES_DIR / "dashboard.html").read_text(encoding="utf-8")


class Dashboard(
    NightModeMixin,
    SaleModeMixin,
    RouteStatsMixin,
    RouteControlMixin,
    RouteSettingsMixin,
):
    def __init__(self, config: dict, db: Database, engine=None):
        self.config = config
        self.db = db
        self.engine = engine
        self.port = config.get("dashboard", {}).get("port", 8080)
        self.auth = AuthManager(config)
        self.app = web.Application(middlewares=[self.auth.middleware()])
        self._setup_routes()
        self._runner: web.AppRunner | None = None
        self._night_settings_path = Path("config/night_settings.json")
        self._sale_settings_path = Path("config/sale_settings.json")

        # Standalone Bybit client (used when engine is None)
        self._client = None
        if not engine and config.get("bybit"):
            try:
                from src.exchange.client import BybitClient
                self._client = BybitClient(config)
                logger.info("Dashboard: standalone BybitClient initialized")
            except Exception:
                logger.warning("Dashboard: failed to init standalone BybitClient")

        # Cached TBank client (avoid re-init on every request)
        self._tbank_client = None
        self._init_tbank_client()

        # TTL caches for expensive operations
        self._cache = {}  # key -> (value, expires_at)
        self._cache_time = time

    # ── Client management ────────────────────────────────────────────

    def _get_client(self):
        """Get Bybit client from engine or standalone."""
        if self.engine:
            return self.engine.client
        return self._client

    def _init_tbank_client(self):
        """Init cached TBank client from first available config."""
        try:
            from src.exchange.tbank_client import TBankClient
            import yaml as _yaml
            for cfg_path in ("config/tbank_scalp.yaml", "config/midas.yaml"):
                if Path(cfg_path).exists():
                    with open(cfg_path) as f:
                        tcfg = _yaml.safe_load(f)
                    self._tbank_client = TBankClient(tcfg)
                    logger.info("Dashboard: standalone TBankClient initialized")
                    return
        except Exception as e:
            logger.warning("Dashboard: failed to init TBankClient: %s", e)

    def _get_tbank_client(self):
        """Get cached TBank client."""
        return self._tbank_client

    # ── TTL Cache ────────────────────────────────────────────────────

    def _cache_get(self, key: str):
        """Get value from TTL cache, returns None if expired/missing."""
        entry = self._cache.get(key)
        if entry and self._cache_time.time() < entry[1]:
            return entry[0]
        return None

    def _cache_set(self, key: str, value, ttl: float = 5.0):
        """Set value in TTL cache."""
        self._cache[key] = (value, self._cache_time.time() + ttl)

    def _get_bybit_positions_cached(self):
        """Get Bybit positions with 5s cache."""
        cached = self._cache_get("bybit_positions")
        if cached is not None:
            return cached
        client = self._get_client()
        if not client:
            return []
        try:
            raw = client.get_positions(category="linear")
            self._cache_set("bybit_positions", raw, 5.0)
            return raw
        except Exception:
            return []

    def _get_tbank_positions_cached(self):
        """Get TBank positions with 5s cache."""
        cached = self._cache_get("tbank_positions")
        if cached is not None:
            return cached
        tc = self._get_tbank_client()
        if not tc:
            return []
        try:
            raw = tc.get_positions()
            self._cache_set("tbank_positions", raw, 5.0)
            return raw
        except Exception:
            return []

    def _get_bybit_balance_cached(self):
        """Get Bybit balance with 10s cache."""
        cached = self._cache_get("bybit_balance")
        if cached is not None:
            return cached
        client = self._get_client()
        if not client:
            return 0.0
        try:
            bal = client.get_balance()
            self._cache_set("bybit_balance", bal, 10.0)
            return bal
        except Exception:
            return 0.0

    def _get_tbank_balance_cached(self):
        """Get TBank balance with 10s cache."""
        cached = self._cache_get("tbank_balance")
        if cached is not None:
            return cached
        tc = self._get_tbank_client()
        if not tc:
            return 0.0
        try:
            bal = tc.get_balance()
            self._cache_set("tbank_balance", bal, 10.0)
            return bal
        except Exception:
            return 0.0

    def _check_service_active(self, service: str) -> bool:
        """Check systemctl is-active with 10s cache."""
        cache_key = f"svc_{service}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        try:
            result = subprocess.run(
                ["systemctl", "is-active", service],
                capture_output=True, text=True, timeout=3,
            )
            active = result.stdout.strip() == "active"
            self._cache_set(cache_key, active, 10.0)
            return active
        except Exception:
            self._cache_set(cache_key, False, 10.0)
            return False

    # ── Routes setup ─────────────────────────────────────────────────

    def _setup_routes(self):
        self.app.router.add_get("/login", self._login_page)
        self.app.router.add_post("/login", self._login_post)
        self.app.router.add_get("/logout", self._logout)
        self.app.router.add_get("/", self._index)
        self.app.router.add_get("/api/stats", self._api_stats)
        self.app.router.add_get("/api/trades", self._api_trades)
        self.app.router.add_get("/api/pnl", self._api_pnl)
        self.app.router.add_get("/api/positions", self._api_positions)
        self.app.router.add_post("/api/toggle-service", self._api_toggle_service)
        self.app.router.add_post("/api/close-position", self._api_close_position)
        self.app.router.add_post("/api/close-all", self._api_close_all)
        self.app.router.add_post("/api/double-position", self._api_double_position)
        self.app.router.add_post("/api/set-sl", self._api_set_sl)
        self.app.router.add_post("/api/set-tp", self._api_set_tp)
        self.app.router.add_get("/api/instances", self._api_instances)
        self.app.router.add_get("/api/pair-pnl", self._api_pair_pnl)
        self.app.router.add_post("/api/set-leverage", self._api_set_leverage)
        self.app.router.add_get("/api/events", self._api_events)
        self.app.router.add_get("/api/night-settings", self._api_night_settings_get)
        self.app.router.add_post("/api/night-settings", self._api_night_settings_post)
        self.app.router.add_get("/api/sale-settings", self._api_sale_settings_get)
        self.app.router.add_post("/api/sale-settings", self._api_sale_settings_post)

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self):
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", self.port)
        await site.start()
        logger.info("Dashboard started on 127.0.0.1:%d", self.port)
        self._night_task = asyncio.create_task(self._night_loop())
        self._sale_task = asyncio.create_task(self._sale_loop())

    async def stop(self):
        if hasattr(self, "_night_task") and self._night_task:
            self._night_task.cancel()
        if hasattr(self, "_sale_task") and self._sale_task:
            self._sale_task.cancel()
        if self._runner:
            await self._runner.cleanup()
            logger.info("Dashboard stopped")

    # ── Auth routes ──────────────────────────────────────────────────

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

    async def _index(self, request: web.Request) -> web.Response:
        return web.Response(text=_DASHBOARD_HTML, content_type="text/html",
                            headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
