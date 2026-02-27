import hashlib
import hmac
import logging
import random
import time

import bcrypt
from aiohttp import web

logger = logging.getLogger(__name__)

COOKIE_NAME = "stasik_session"
BLOCK_ATTEMPTS = 5
BLOCK_DURATION = 900  # 15 minutes


class AuthManager:
    def __init__(self, config: dict):
        dash = config.get("dashboard", {})
        self.username = dash.get("username", "admin")
        self.password_hash = dash.get("password_hash", "").encode()
        self.secret = dash.get("secret", "changeme").encode()
        self._failed: dict[str, list] = {}  # ip -> [timestamps]
        self._captchas: dict[str, int] = {}  # session_token -> answer

    # --- IP blocking ---

    def is_blocked(self, ip: str) -> bool:
        if ip not in self._failed:
            return False
        attempts = self._failed[ip]
        # clean old attempts
        now = time.time()
        self._failed[ip] = [t for t in attempts if now - t < BLOCK_DURATION]
        return len(self._failed[ip]) >= BLOCK_ATTEMPTS

    def record_fail(self, ip: str):
        if ip not in self._failed:
            self._failed[ip] = []
        self._failed[ip].append(time.time())

    def clear_fails(self, ip: str):
        self._failed.pop(ip, None)

    # --- Captcha ---

    def generate_captcha(self) -> tuple[str, str]:
        """Returns (question_text, token)."""
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        op = random.choice([("+", a + b), ("-", abs(a - b))])
        if op[0] == "-" and a < b:
            a, b = b, a
        question = f"{a} {op[0]} {b}"
        answer = op[1]
        token = hashlib.sha256(f"{time.time()}{random.random()}".encode()).hexdigest()[:16]
        self._captchas[token] = answer
        # cleanup old tokens (keep max 1000)
        if len(self._captchas) > 1000:
            keys = list(self._captchas.keys())
            for k in keys[:500]:
                self._captchas.pop(k, None)
        return question, token

    def verify_captcha(self, token: str, user_answer: str) -> bool:
        expected = self._captchas.pop(token, None)
        if expected is None:
            return False
        try:
            return int(user_answer.strip()) == expected
        except (ValueError, AttributeError):
            return False

    # --- Password ---

    def check_password(self, username: str, password: str) -> bool:
        if username != self.username:
            return False
        try:
            return bcrypt.checkpw(password.encode(), self.password_hash)
        except Exception:
            return False

    # --- Session cookie ---

    def create_session_cookie(self, username: str) -> str:
        ts = str(int(time.time()))
        payload = f"{username}:{ts}"
        sig = hmac.new(self.secret, payload.encode(), hashlib.sha256).hexdigest()
        return f"{payload}:{sig}"

    def verify_session(self, cookie_value: str) -> str | None:
        """Returns username if valid, None otherwise."""
        if not cookie_value:
            return None
        parts = cookie_value.split(":")
        if len(parts) != 3:
            return None
        username, ts, sig = parts
        payload = f"{username}:{ts}"
        expected = hmac.new(self.secret, payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        # session valid for 24 hours
        try:
            if time.time() - int(ts) > 86400:
                return None
        except ValueError:
            return None
        return username

    # --- Middleware ---

    def middleware(self):
        auth = self

        @web.middleware
        async def auth_middleware(request: web.Request, handler):
            # allow login page and its POST
            if request.path in ("/login", "/favicon.ico"):
                return await handler(request)

            cookie = request.cookies.get(COOKIE_NAME)
            user = auth.verify_session(cookie)
            if not user:
                raise web.HTTPFound("/login")

            return await handler(request)

        return auth_middleware
