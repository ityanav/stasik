import logging
import sqlite3
import time
from pathlib import Path

import httpx
import pandas as pd
import yaml

from src.strategy.signals import Trend

logger = logging.getLogger(__name__)


class DataFetcherMixin:
    """Mixin for data fetching: tickers, klines, funding, fear-greed, margin, T-Bank."""

    # ── Tickers / Liquidity ─────────────────────────────────────────────

    def _get_tickers_cached(self) -> dict:
        """Return all tickers, cached for 5 minutes."""
        now = time.time()
        if now - self._tickers_cache_ts > 300:
            try:
                self._tickers_cache = self.client.get_all_tickers()
                self._tickers_cache_ts = now
            except Exception:
                logger.warning("Failed to fetch tickers for liquidity filter", exc_info=True)
        return self._tickers_cache

    def _check_liquidity(self, symbol: str) -> bool:
        """Pre-trade liquidity filter. Returns True if symbol passes."""
        if self._min_turnover_24h == 0 and self._max_spread_pct == 0:
            return True

        tickers = self._get_tickers_cached()
        ticker = tickers.get(symbol)
        if not ticker:
            return True  # no data — allow trade (conservative)

        turnover = ticker["turnover24h"]
        last_price = ticker["last_price"]
        bid = ticker["bid1"]
        ask = ticker["ask1"]

        passed = True
        reasons = []

        if self._min_turnover_24h > 0 and turnover < self._min_turnover_24h:
            passed = False
            reasons.append(f"turnover=${turnover/1e6:.1f}M<${self._min_turnover_24h/1e6:.0f}M")

        if self._max_spread_pct > 0 and last_price > 0 and bid > 0 and ask > 0:
            spread_pct = (ask - bid) / last_price * 100
            if spread_pct > self._max_spread_pct:
                passed = False
                reasons.append(f"spread={spread_pct:.3f}%>{self._max_spread_pct}%")

        if not passed:
            now = time.time()
            last_log = self._liquidity_log_ts.get(symbol, 0)
            if now - last_log > 1800:  # log once per 30 min per symbol
                self._liquidity_log_ts[symbol] = now
                logger.info("Liquidity filter: skip %s (%s)", symbol, ", ".join(reasons))

        return passed

    # ── Margin ──────────────────────────────────────────────────────────

    def _check_margin_limit(self) -> bool:
        """Check if instance margin limit is exceeded. Returns True if OK to trade."""
        if self._max_margin_usdt <= 0:
            return True  # disabled
        now = time.time()
        cached_margin, cached_ts = self._margin_cache
        if now - cached_ts < self._margin_cache_ttl:
            used = cached_margin
        else:
            try:
                used = self.client.get_used_margin(self.pairs)
                self._margin_cache = (used, now)
            except Exception as e:
                logger.warning("Margin check failed: %s — allowing trade", e)
                return True
        if used >= self._max_margin_usdt:
            logger.info(
                "Margin limit: %s used $%.0f / $%.0f — blocking new entry",
                self.instance_name, used, self._max_margin_usdt,
            )
            return False
        return True

    # ── Multi-timeframe / HTF ───────────────────────────────────────────

    def _get_mtf_data(self, symbol: str, category: str) -> dict[str, pd.DataFrame]:
        """Fetch multi-timeframe klines for AI context, with per-TF caching."""
        mtf_data: dict[str, pd.DataFrame] = {}
        now = time.time()
        for tf in self._extra_timeframes:
            cache_key = f"{symbol}_{tf}"
            cached = self._mtf_cache.get(cache_key)
            cache_ttl = self._timeframe_to_seconds(tf)  # cache for one candle period
            if cached and now - cached[1] < cache_ttl:
                mtf_data[tf] = cached[0]
                continue
            try:
                tf_df = self.client.get_klines(
                    symbol=symbol, interval=tf, limit=100, category=category
                )
                tf_df.attrs["symbol"] = symbol
                mtf_data[tf] = tf_df
                self._mtf_cache[cache_key] = (tf_df, now)
            except Exception:
                logger.warning("Failed to fetch %sm klines for %s", tf, symbol)
        return mtf_data

    async def _get_fear_greed(self) -> int | None:
        """Fetch Fear & Greed Index from alternative.me, cached for 1 hour."""
        now = time.time()
        if self._fng_cache and now - self._fng_cache[1] < 3600:
            return self._fng_cache[0]

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get("https://api.alternative.me/fng/?limit=1")
                data = resp.json()
                value = int(data["data"][0]["value"])
                self._fng_cache = (value, now)
                logger.info("Fear & Greed Index: %d (%s)", value, data["data"][0].get("value_classification", ""))
                return value
        except Exception as e:
            logger.warning("Failed to fetch Fear & Greed Index: %s", e)
            return None

    def _get_funding_rate_cached(self, symbol: str, category: str) -> float:
        """Get funding rate with 30 min cache."""
        now = time.time()
        cached = self._funding_cache.get(symbol)
        if cached and now - cached[1] < 1800:
            return cached[0]

        rate = self.client.get_funding_rate(symbol, category)
        self._funding_cache[symbol] = (rate, now)
        return rate

    def _get_htf_data(self, symbol: str, category: str) -> tuple[Trend, float, float]:
        """Get higher timeframe trend + ADX + SMA deviation, cached for 5 minutes."""
        from src.strategy.indicators import calculate_adx, calculate_sma_deviation
        now = time.time()
        cached = self._htf_cache.get(symbol)
        if cached and now - cached[3] < 300:  # 5 min cache
            return cached[0], cached[1], cached[2]

        try:
            htf_df = self.client.get_klines(
                symbol=symbol, interval=self.htf_timeframe, limit=100, category=category
            )
            htf_df.attrs["symbol"] = symbol
            trend = self.signal_gen.get_htf_trend(htf_df)
            adx = calculate_adx(htf_df, self._adx_period)
            htf_sma_dev = calculate_sma_deviation(htf_df, 25)
        except Exception:
            logger.warning("Failed to get HTF data for %s, allowing trade", symbol)
            trend = Trend.NEUTRAL
            adx = 25.0  # default: allow trading
            htf_sma_dev = 0.0

        self._htf_cache[symbol] = (trend, adx, htf_sma_dev, now)
        return trend, adx, htf_sma_dev

    def _get_instrument_info(self, symbol: str, category: str) -> dict:
        key = f"{symbol}_{category}"
        if key not in self._instrument_cache:
            self._instrument_cache[key] = self.client.get_instrument_info(symbol, category)
        return self._instrument_cache[key]

    # ── T-Bank helpers ──────────────────────────────────────────────────

    def _get_tbank_balance(self) -> float | None:
        """Get T-Bank balance from tbank config (if available)."""
        # Find a tbank config from other_instances
        for inst in self.config.get("other_instances", []):
            if "TBANK" not in inst.get("name", "").upper():
                continue
            # Try to find config path for this instance
            service = inst.get("service", "")
            if not service:
                continue
            # Read config from systemd service ExecStart
            for cfg_name in ("config/tbank_scalp.yaml", "config/tbank_swing.yaml"):
                cfg_path = Path("/root/stasik") / cfg_name
                if not cfg_path.exists():
                    continue
                try:
                    with open(cfg_path) as f:
                        tbank_cfg = yaml.safe_load(f)
                    token = tbank_cfg.get("tbank", {}).get("token", "")
                    if not token or token == "YOUR_TOKEN_HERE":
                        continue
                    sandbox = tbank_cfg.get("tbank", {}).get("sandbox", True)
                    from t_tech.invest import Client
                    from t_tech.invest.constants import INVEST_GRPC_API, INVEST_GRPC_API_SANDBOX
                    target = INVEST_GRPC_API_SANDBOX if sandbox else INVEST_GRPC_API
                    with Client(token, target=target) as client:
                        accounts = client.users.get_accounts()
                        if not accounts.accounts:
                            continue
                        acc_id = accounts.accounts[0].id
                        if sandbox:
                            portfolio = client.sandbox.get_sandbox_portfolio(account_id=acc_id)
                        else:
                            portfolio = client.operations.get_portfolio(account_id=acc_id)
                        total = portfolio.total_amount_portfolio
                        return float(total.units) + float(total.nano) / 1e9
                except Exception:
                    logger.debug("Failed to get T-Bank balance", exc_info=True)
            break  # Only try once
        return None

    def _get_other_instance_prices(self, symbols: list[str], is_tbank: bool) -> dict[str, float]:
        """Get live prices for symbols from another instance's exchange."""
        prices = {}
        if not symbols:
            return prices
        try:
            if is_tbank:
                prices = self._get_tbank_prices(symbols)
            else:
                # Same exchange (Bybit) — use our client
                for sym in symbols:
                    try:
                        prices[sym] = self.client.get_last_price(sym)
                    except Exception:
                        pass
        except Exception:
            logger.debug("Failed to get prices for other instance", exc_info=True)
        return prices

    def _get_tbank_prices(self, symbols: list[str]) -> dict[str, float]:
        """Get live prices from T-Bank API for given tickers."""
        for cfg_name in ("config/tbank_scalp.yaml", "config/tbank_swing.yaml"):
            cfg_path = Path("/root/stasik") / cfg_name
            if not cfg_path.exists():
                continue
            try:
                with open(cfg_path) as f:
                    tbank_cfg = yaml.safe_load(f)
                token = tbank_cfg.get("tbank", {}).get("token", "")
                if not token or token == "YOUR_TOKEN_HERE":
                    continue
                sandbox = tbank_cfg.get("tbank", {}).get("sandbox", True)
                from t_tech.invest import Client
                from t_tech.invest.constants import INVEST_GRPC_API, INVEST_GRPC_API_SANDBOX
                target = INVEST_GRPC_API_SANDBOX if sandbox else INVEST_GRPC_API
                with Client(token, target=target) as client:
                    # Load instruments to get FIGI mapping
                    all_shares = client.instruments.shares(instrument_status=1).instruments
                    all_futures = client.instruments.futures(instrument_status=1).instruments
                    figi_map = {}
                    for inst in list(all_shares) + list(all_futures):
                        if inst.ticker in symbols:
                            figi_map[inst.ticker] = inst.figi
                    if not figi_map:
                        continue
                    figi_list = list(figi_map.values())
                    resp = client.market_data.get_last_prices(instrument_id=figi_list)
                    prices = {}
                    figi_to_ticker = {v: k for k, v in figi_map.items()}
                    for lp in resp.last_prices:
                        ticker = figi_to_ticker.get(lp.figi)
                        if ticker:
                            price = float(lp.price.units) + float(lp.price.nano) / 1e9
                            prices[ticker] = price
                    return prices
            except Exception:
                logger.debug("Failed to get T-Bank prices from %s", cfg_name, exc_info=True)
        return {}

    def _get_tbank_klines(self, symbol: str, interval: str, limit: int = 100):
        """Get klines for a T-Bank symbol by borrowing token from tbank config."""
        for cfg_name in ("config/tbank_scalp.yaml", "config/tbank_swing.yaml"):
            cfg_path = Path("/root/stasik") / cfg_name
            if not cfg_path.exists():
                continue
            try:
                with open(cfg_path) as f:
                    tbank_cfg = yaml.safe_load(f)
                token = tbank_cfg.get("tbank", {}).get("token", "")
                if not token or token == "YOUR_TOKEN_HERE":
                    continue
                sandbox = tbank_cfg.get("tbank", {}).get("sandbox", True)
                from src.exchange.tbank_client import TBankClient
                # Minimal config just for klines
                mini_cfg = {
                    "tbank": {"token": token, "sandbox": sandbox, "account_id": "", "commission_rate": 0.0004},
                    "trading": {"pairs": [symbol], "instrument_type": "share"},
                }
                tc = TBankClient(mini_cfg)
                return tc.get_klines(symbol, interval, limit=limit)
            except Exception:
                logger.debug("Failed to get T-Bank klines for %s from %s", symbol, cfg_name, exc_info=True)
        return None

    # ── Instance lookup ─────────────────────────────────────────────────

    def _find_instance_for_symbol(self, symbol: str) -> dict | None:
        """Find which other_instance owns a symbol by checking its DB."""
        for inst in self.config.get("other_instances", []):
            db_path = inst.get("db_path", "")
            if db_path and Path(db_path).exists():
                try:
                    conn = sqlite3.connect(db_path)
                    conn.row_factory = sqlite3.Row
                    row = conn.execute(
                        "SELECT id FROM trades WHERE symbol = ? AND status = 'open' LIMIT 1",
                        (symbol,),
                    ).fetchone()
                    conn.close()
                    if row:
                        return inst
                except Exception:
                    pass
        return None

    def _get_tbank_client_for_instance(self, inst: dict):
        """Create a temporary TBankClient for an other_instance."""
        config_map = {
            "stasik-tbank-scalp": "config/tbank_scalp.yaml",
            "stasik-tbank-swing": "config/tbank_swing.yaml",
        }
        service = inst.get("service", "")
        config_path = config_map.get(service)
        if not config_path or not Path(config_path).exists():
            return None
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        from src.exchange.tbank_client import TBankClient
        return TBankClient(cfg)
