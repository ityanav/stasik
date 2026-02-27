import logging
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path

import aiosqlite
import yaml
from aiohttp import web

logger = logging.getLogger(__name__)

# Combo type detection — load combo sets from config/combos.yaml
_COMBO_KEY_MAP = {
    "fib_zone": "fib", "liq_sweep": "sweep", "fvg": "fvg",
    "order_block": "ob", "cluster_bonus": "cluster",
    "cum_delta": "cd", "ote_bonus": "ote", "murray": "mm",
    "displacement": "mom", "volume": "vol", "rsi_div": "rsi_div",
    "pivot_bonus": "pivot", "vol_profile": "vp",
}
_ALL_DETAIL_KEYS = list(_COMBO_KEY_MAP.keys())


def _load_combo_sets() -> dict[str, list]:
    """Load combo sets from config/combos.yaml."""
    path = Path("config/combos.yaml")
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _get_combo_type(details_json: str | None) -> str:
    """Determine combo type ('lev' or 'shakal') from signal_scores details JSON."""
    if not details_json:
        return "lev"
    try:
        import json
        d = json.loads(details_json)
        active = frozenset(
            _COMBO_KEY_MAP[k] for k in _ALL_DETAIL_KEYS if d.get(k, 0) != 0
        )
        sets = _load_combo_sets()
        # Reverse map: short name → internal key (for combo definitions)
        fwd = {v: k for k, v in _COMBO_KEY_MAP.items()}
        for set_name, combos in sets.items():
            allowed = [frozenset(x for x in combo) for combo in combos]
            if active in allowed:
                return set_name
        return "lev"
    except Exception:
        return "lev"


def _get_db_path(live_path: str, source: str) -> str:
    """Return archive DB path if source='archive', else the live path."""
    from src.dashboard.services import _get_db_path as _orig
    return _orig(live_path, source)


def _other_instances(config: dict) -> list[dict]:
    """Return other_instances from config."""
    return config.get("other_instances", [])


class RouteStatsMixin:
    """Dashboard mixin: stats, trades, PnL, positions, instances endpoints."""

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
                        cur = await db.execute(
                            "SELECT COALESCE(SUM(pnl), 0) as total FROM trades WHERE status = 'closed'"
                        )
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
        if not is_archive:
            balance = self._get_bybit_balance_cached()
            _all_services = [
                "stasik-fiba", "stasik-tbank-scalp", "stasik-tbank-swing",
                "stasik-midas", "stasik-fin",
            ]
            running = any(self._check_service_active(s) for s in _all_services)

        exchange = self.config.get("exchange", "bybit")
        currency = "RUB" if exchange == "tbank" else "USDT"

        # Total trades across all instances + today's net PnL per currency
        all_trades = total
        today_pnl_usdt = daily_pnl  # main instance (SCALP) is USDT
        today_pnl_rub = 0.0
        _today_sql = (
            "SELECT COALESCE(SUM(pnl), 0) as total FROM trades "
            "WHERE status = 'closed' AND date(closed_at) = date('now')"
        )
        _combined_sql = (
            "SELECT COUNT(*) as cnt, "
            "COALESCE(SUM(CASE WHEN date(closed_at) = date('now') THEN pnl ELSE 0 END), 0) as today "
            "FROM trades WHERE status='closed'"
        )
        for inst in _other_instances(self.config):
            db_path = _get_db_path(inst.get("db_path", ""), source)
            inst_name = inst.get("name", "")
            is_rub = "TBANK" in inst_name.upper() or "MIDAS" in inst_name.upper()
            if db_path and Path(db_path).exists():
                try:
                    async with aiosqlite.connect(db_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute(_combined_sql)
                        row = await cur.fetchone()
                        if row:
                            all_trades += int(row["cnt"])
                            if not is_archive:
                                pnl_val = float(row["today"])
                                if is_rub:
                                    today_pnl_rub += pnl_val
                                else:
                                    today_pnl_usdt += pnl_val
                except Exception:
                    pass
            if not is_archive and "TBANK" in inst_name.upper() and tbank_balance == 0:
                tbank_balance = self._get_tbank_balance_cached()

        data = {
            "balance": balance,
            "tbank_balance": tbank_balance,
            "daily_pnl": daily_pnl,
            "total_pnl": total_pnl,
            "today_pnl_usdt": round(today_pnl_usdt, 2),
            "today_pnl_rub": round(today_pnl_rub, 2),
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
        date_param = request.query.get("date", "")
        range_param = request.query.get("range", "day")

        instance_name = self.config.get("instance_name", "SCALP")

        # Build date filter for SQL
        if date_param:
            if range_param == "week":
                try:
                    end = (datetime.strptime(date_param, "%Y-%m-%d") + timedelta(days=7)).strftime("%Y-%m-%d")
                except ValueError:
                    end = date_param
                date_where = f" AND date(closed_at) >= '{date_param}' AND date(closed_at) < '{end}'"
            else:
                date_where = f" AND date(closed_at) = '{date_param}'"
        elif source == "archive":
            date_where = ""
        else:
            # Live, today only
            date_where = " AND date(closed_at) = date('now')"

        # Collect all trades from all instances
        all_trades = []

        # Main instance trades — always use direct SQL for date filtering
        main_db_path = _get_db_path(str(self.db.db_path), source)
        if source == "archive":
            if Path(main_db_path).exists():
                try:
                    async with aiosqlite.connect(main_db_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute(
                            f"SELECT * FROM trades WHERE 1=1{date_where} ORDER BY id DESC LIMIT 1000"
                        )
                        rows = await cur.fetchall()
                        for r in rows:
                            t = dict(r)
                            t["instance"] = t.get("instance") or instance_name
                            all_trades.append(t)
                except Exception:
                    logger.warning("Failed to read archive trades from %s", main_db_path)
        else:
            try:
                async with aiosqlite.connect(str(self.db.db_path)) as db:
                    db.row_factory = aiosqlite.Row
                    cur = await db.execute(
                        f"SELECT * FROM trades WHERE 1=1{date_where} ORDER BY id DESC LIMIT 1000"
                    )
                    rows = await cur.fetchall()
                    for r in rows:
                        t = dict(r)
                        t["instance"] = t.get("instance") or instance_name
                        all_trades.append(t)
            except Exception:
                logger.warning("Failed to read trades from main DB")

        # Other instances trades
        for inst in _other_instances(self.config):
            db_path = _get_db_path(inst.get("db_path", ""), source)
            inst_name = inst.get("name", "???")
            if db_path and Path(db_path).exists():
                try:
                    async with aiosqlite.connect(db_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute(
                            f"SELECT * FROM trades WHERE 1=1{date_where} ORDER BY id DESC LIMIT 1000"
                        )
                        rows = await cur.fetchall()
                        for r in rows:
                            t = dict(r)
                            t["instance"] = t.get("instance") or inst_name
                            all_trades.append(t)
                except Exception:
                    logger.warning("Failed to read trades from %s", inst_name)

        # Sort: open trades on top (newest first), then closed (newest first)
        all_trades.sort(key=lambda t: (
            0 if t.get("status") == "open" else 1,
            t.get("closed_at") or t.get("opened_at") or "",
        ), reverse=False)
        # Reverse within groups: newest open first, newest closed first
        open_trades = [t for t in all_trades if t.get("status") == "open"]
        closed_trades = [t for t in all_trades if t.get("status") != "open"]
        open_trades.sort(key=lambda t: t.get("opened_at") or "", reverse=True)
        closed_trades.sort(key=lambda t: t.get("closed_at") or "", reverse=True)
        all_trades = open_trades + closed_trades

        # Enrich with leverage from config
        for t in all_trades:
            if "leverage" not in t or not t.get("leverage"):
                t["leverage"] = self._get_instance_config_leverage(t.get("instance", ""))

        # Paginate
        offset = (page - 1) * per_page
        page_trades = all_trades[offset:offset + per_page + 1]
        has_next = len(page_trades) > per_page
        if has_next:
            page_trades = page_trades[:per_page]

        # Enrich with combo_type from signal_scores
        await self._enrich_combo_type(page_trades, source)

        return web.json_response({"trades": page_trades, "page": page, "has_next": has_next})

    async def _enrich_combo_type(self, trades: list[dict], source: str = "live"):
        """Add combo_type ('lev'/'shakal') to trades by looking up signal_scores."""
        # Group trade IDs by DB path
        db_trades: dict[str, list[dict]] = {}
        instance_name = self.config.get("instance_name", "SCALP")
        for t in trades:
            inst = (t.get("instance") or "").upper()
            if inst == instance_name.upper() or not inst:
                db_path = _get_db_path(str(self.db.db_path), source)
            else:
                found = False
                for oi in _other_instances(self.config):
                    if oi.get("name", "").upper() == inst:
                        db_path = _get_db_path(oi.get("db_path", ""), source)
                        found = True
                        break
                if not found:
                    t["combo_type"] = "lev"
                    continue
            db_trades.setdefault(db_path, []).append(t)

        for db_path, group in db_trades.items():
            if not Path(db_path).exists():
                for t in group:
                    t["combo_type"] = "lev"
                continue
            ids = [t["id"] for t in group if t.get("id")]
            if not ids:
                for t in group:
                    t["combo_type"] = "lev"
                continue
            try:
                async with aiosqlite.connect(db_path) as db:
                    placeholders = ",".join("?" * len(ids))
                    cur = await db.execute(
                        f"SELECT trade_id, details FROM signal_scores WHERE trade_id IN ({placeholders})",
                        ids,
                    )
                    rows = await cur.fetchall()
                    score_map = {r[0]: r[1] for r in rows}
                for t in group:
                    t["combo_type"] = _get_combo_type(score_map.get(t.get("id")))
            except Exception:
                for t in group:
                    t["combo_type"] = "lev"

    async def _api_pnl(self, request: web.Request) -> web.Response:
        days = int(request.query.get("days", "30"))
        tf = request.query.get("tf", "1D")  # 10m, 30m, 1H, 1D, 1M
        source = request.query.get("source", "live")
        date_param = request.query.get("date", "")
        range_param = request.query.get("range", "day")

        instance_name = self.config.get("instance_name", "SCALP")

        return await self._pnl_intraday(tf, instance_name, source, date_param, range_param)

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
        for inst in _other_instances(self.config):
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

    async def _pnl_intraday(self, tf: str, instance_name: str, source: str = "live",
                             date_param: str = "", range_param: str = "day") -> web.Response:
        """Intraday PnL from trades table. Filters by date for live mode."""

        minutes = {"1m": 1, "5m": 5, "10m": 10, "30m": 30, "1H": 60, "1D": 1440, "1M": 43200}.get(tf, 60)
        per_trade = True  # always show each trade as a separate point

        # Build date filter
        if date_param:
            if range_param == "week":
                try:
                    end = (datetime.strptime(date_param, "%Y-%m-%d") + timedelta(days=7)).strftime("%Y-%m-%d")
                except ValueError:
                    end = date_param
                date_where = f" AND date(closed_at) >= '{date_param}' AND date(closed_at) < '{end}'"
            else:
                date_where = f" AND date(closed_at) = '{date_param}'"
        elif source == "archive":
            date_where = ""
        else:
            # Live, today only
            date_where = " AND date(closed_at) = date('now')"

        main_db_path = _get_db_path(str(self.db.db_path), source)
        all_dbs = [(main_db_path, instance_name)]
        for inst in _other_instances(self.config):
            db_path = _get_db_path(inst.get("db_path", ""), source)
            if db_path and Path(db_path).exists():
                all_dbs.append((db_path, inst.get("name", "???")))

        all_trades: list[dict] = []
        pnl_by_bucket: dict[str, dict] = {}

        for db_path, inst_name in all_dbs:
            try:
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    cur = await db.execute(
                        f"SELECT closed_at, pnl FROM trades WHERE status='closed'{date_where} ORDER BY closed_at",
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

                        if per_trade:
                            key = dt.strftime("%Y-%m-%d %H:%M:%S")
                            all_trades.append({
                                "trade_date": key, "pnl": r["pnl"] or 0,
                                "trades_count": 1, inst_name: r["pnl"] or 0,
                            })
                        else:
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

        if per_trade:
            all_trades.sort(key=lambda x: x["trade_date"])
            return web.json_response(all_trades)
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
                            "SELECT id, symbol, side, entry_price, qty, stop_loss, take_profit, partial_closed, opened_at FROM trades WHERE status = 'open' ORDER BY opened_at DESC"
                        )
                        rows = await cur.fetchall()
                        for r in rows:
                            positions.append({
                                "id": r["id"],
                                "symbol": r["symbol"],
                                "side": r["side"],
                                "size": float(r["qty"]),
                                "entry_price": float(r["entry_price"]),
                                "stop_loss": float(r["stop_loss"] if r["stop_loss"] else 0),
                                "take_profit": float(r["take_profit"] if r["take_profit"] else 0),
                                "unrealised_pnl": 0.0,
                                "instance": instance_name,
                                "partial_closed": int(r["partial_closed"] or 0),
                                "opened_at": r["opened_at"] or "",
                            })
                except Exception:
                    logger.warning("Failed to read archive positions from %s", main_archive)
        else:
            # Live: read open trades from main DB (same as other instances)
            main_open = await self.db.get_open_trades()
            for t in main_open:
                positions.append({
                    "id": t["id"],
                    "symbol": t["symbol"],
                    "side": t["side"],
                    "size": float(t["qty"]),
                    "entry_price": float(t["entry_price"]),
                    "stop_loss": float(t.get("stop_loss") or 0),
                    "take_profit": float(t.get("take_profit") or 0),
                    "unrealised_pnl": 0.0,
                    "instance": instance_name,
                    "partial_closed": t.get("partial_closed", 0) or 0,
                    "opened_at": t.get("opened_at") or "",
                })

        # Other instances — open trades from DB
        for inst in _other_instances(self.config):
            inst_name = inst.get("name", "???")
            db_path = _get_db_path(inst.get("db_path", ""), source)
            if db_path and Path(db_path).exists():
                try:
                    async with aiosqlite.connect(db_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute(
                            "SELECT id, symbol, side, entry_price, qty, stop_loss, take_profit, partial_closed, opened_at FROM trades WHERE status = 'open' ORDER BY opened_at DESC"
                        )
                        rows = await cur.fetchall()
                        for r in rows:
                            positions.append({
                                "id": r["id"],
                                "symbol": r["symbol"],
                                "side": r["side"],
                                "size": float(r["qty"]),
                                "entry_price": float(r["entry_price"]),
                                "stop_loss": float(r["stop_loss"] if r["stop_loss"] else 0),
                                "take_profit": float(r["take_profit"] if r["take_profit"] else 0),
                                "unrealised_pnl": 0.0,
                                "instance": inst_name,
                                "partial_closed": int(r["partial_closed"] or 0),
                                "opened_at": r["opened_at"] or "",
                            })
                except Exception:
                    logger.warning("Failed to read positions from %s", inst_name)

        # Sort all positions: newest first (by opened_at descending)
        positions.sort(key=lambda p: p.get("opened_at") or "", reverse=True)

        # Enrich positions with live unrealised PnL (calculated per-instance)
        if not is_archive and positions:
            # Bybit positions (cached 5s)
            if self._get_client():
                try:
                    raw = self._get_bybit_positions_cached()
                    live_mark = {p["symbol"]: p.get("mark_price", 0) for p in raw}
                    live_leverage = {p["symbol"]: p.get("leverage", "?") for p in raw}
                    for pos in positions:
                        inst = (pos.get("instance") or "").upper()
                        if "TBANK" in inst or "MIDAS" in inst:
                            continue
                        sym = pos["symbol"]
                        pos["leverage"] = live_leverage.get(sym, "?")
                        mark = 0
                        if sym in live_mark and live_mark[sym]:
                            mark = float(live_mark[sym])
                        else:
                            try:
                                mark = self._get_client().get_last_price(sym, category="linear")
                            except Exception:
                                pass
                        if mark > 0:
                            entry = float(pos["entry_price"])
                            qty = float(pos["size"])
                            direction = 1 if pos["side"] == "Buy" else -1
                            pos["unrealised_pnl"] = round((mark - entry) * qty * direction, 2)
                except Exception:
                    pass

            # TBank/Midas positions — get mark prices via TBankClient
            tbank_positions = [p for p in positions
                               if ("TBANK" in (p.get("instance") or "").upper()
                                   or "MIDAS" in (p.get("instance") or "").upper())]
            if tbank_positions:
                try:
                    tc = self._get_tbank_client()
                    if tc:
                        # Get live positions from TBank (cached 5s)
                        tbank_raw = self._get_tbank_positions_cached()
                        tbank_mark = {p["symbol"]: p for p in tbank_raw}
                        for pos in tbank_positions:
                            pos["leverage"] = "1"
                            sym = pos["symbol"]
                            live = tbank_mark.get(sym)
                            if live:
                                pos["unrealised_pnl"] = round(float(live.get("unrealised_pnl", 0)), 2)
                            else:
                                # Position not on exchange — calc from last price
                                try:
                                    mark = tc.get_last_price(sym)
                                    if mark > 0:
                                        entry = float(pos["entry_price"])
                                        qty = float(pos["size"])
                                        direction = 1 if pos["side"] == "Buy" else -1
                                        pos["unrealised_pnl"] = round((mark - entry) * qty * direction, 2)
                                except Exception:
                                    pass
                except Exception as e:
                    logger.debug("TBank position enrichment failed: %s", e)

        for pos in positions:
            entry = float(pos["entry_price"])
            qty = float(pos["size"])
            sl = float(pos.get("stop_loss") or 0)
            tp = float(pos.get("take_profit") or 0)
            pos["entry_amount"] = round(entry * qty, 2)
            # SL PnL: loss in currency when stop-loss triggers
            if sl > 0:
                if pos["side"] == "Buy":
                    pos["sl_pnl"] = round((sl - entry) * qty, 2)
                else:
                    pos["sl_pnl"] = round((entry - sl) * qty, 2)
            else:
                pos["sl_pnl"] = None
            # TP PnL: profit in currency when take-profit triggers
            if tp > 0:
                if pos["side"] == "Buy":
                    pos["tp_pnl"] = round((tp - entry) * qty, 2)
                else:
                    pos["tp_pnl"] = round((entry - tp) * qty, 2)
            else:
                pos["tp_pnl"] = None

        # Enrich with combo_type
        await self._enrich_combo_type_positions(positions, source)

        return web.json_response(positions)

    async def _enrich_combo_type_positions(self, positions: list[dict], source: str = "live"):
        """Add combo_type to open positions from signal_scores."""
        # Reuse _enrich_combo_type — positions have same structure (id, instance)
        await self._enrich_combo_type(positions, source)

    async def _api_pair_pnl(self, request: web.Request) -> web.Response:
        """Per-pair PnL across all instances."""
        pairs: dict[str, dict] = {}
        db_cfg = self.config.get("database", {})
        main_db = db_cfg.get("path", "data/trades.db") if isinstance(db_cfg, dict) else str(db_cfg)
        other = _other_instances(self.config)
        all_dbs = [{"db_path": main_db, "name": self.config.get("instance_name", "MAIN")}] + other
        for inst in all_dbs:
            db_path = inst.get("db_path", "")
            inst_name = inst.get("name", "")
            if not db_path or not Path(db_path).exists():
                continue
            is_tbank = "TBANK" in inst_name.upper() or "MIDAS" in inst_name.upper()
            cur = "RUB" if is_tbank else "USDT"
            try:
                async with aiosqlite.connect(db_path) as db:
                    async_cur = await db.execute(
                        "SELECT symbol, COUNT(*) as cnt, COALESCE(SUM(pnl),0) as total "
                        "FROM trades WHERE status='closed' GROUP BY symbol"
                    )
                    rows = await async_cur.fetchall()
                for symbol, cnt, total in rows:
                    key = f"{symbol}_{cur}"
                    if key not in pairs:
                        pairs[key] = {"symbol": symbol, "cur": cur, "trades": 0, "pnl": 0.0}
                    pairs[key]["trades"] += cnt
                    pairs[key]["pnl"] += total
            except Exception:
                continue
        result = sorted(pairs.values(), key=lambda x: x["pnl"], reverse=True)
        return web.json_response(result)

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
                        cur = await db.execute(
                            "SELECT COALESCE(SUM(pnl), 0) as total FROM trades WHERE status = 'closed'"
                        )
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

            inst_name = self.config.get("instance_name", "SCALP")
            instances.append({
                "name": inst_name,
                "service": "stasik-" + inst_name.lower(),
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
            inst_name = self.config.get("instance_name", "SCALP")
            main_service = "stasik-" + inst_name.lower()
            running = self._check_service_active(main_service)

            instances.append({
                "name": inst_name,
                "service": main_service,
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
        for inst in _other_instances(self.config):
            data = await self._read_instance_data(inst, source)
            instances.append(data)

        # Enrich with live leverage from exchange
        if not is_archive and self._get_client():
            try:
                lev_map = self._get_client().get_all_leverage(category="linear")
                for inst_data in instances:
                    if "TBANK" in inst_data["name"].upper():
                        continue
                    pairs = inst_data.get("pairs", [])
                    pair_levs = [lev_map.get(p) for p in pairs if p in lev_map]
                    if pair_levs:
                        inst_data["leverage"] = pair_levs[0]
            except Exception:
                pass

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
                        "SELECT COALESCE(SUM(pnl), 0) as total FROM trades "
                        "WHERE status = 'closed' AND date(closed_at) = ?",
                        (date.today().isoformat(),),
                    )
                    row = await cur.fetchone()
                    if row:
                        daily = float(row["total"])

                    cur = await db.execute(
                        "SELECT COALESCE(SUM(pnl), 0) as total FROM trades WHERE status = 'closed'"
                    )
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
            "service": service,
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
