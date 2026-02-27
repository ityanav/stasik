import logging
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

import aiosqlite
from aiohttp import web

logger = logging.getLogger(__name__)


def _other_instances(config: dict) -> list[dict]:
    """Return other_instances from config."""
    return config.get("other_instances", [])


class RouteControlMixin:
    """Dashboard mixin: service toggle, close/double positions, SL/TP, leverage."""

    async def _api_toggle_service(self, request: web.Request) -> web.Response:
        """Start or stop a systemd service."""
        try:
            data = await request.json()
            service = data.get("service", "")
            action = data.get("action", "")  # "start" or "stop"
            if not service or action not in ("start", "stop", "restart"):
                return web.json_response({"ok": False, "error": "Bad request"}, status=400)
            # Whitelist: only known stasik services
            allowed = {"stasik", "stasik-degen", "stasik-tbank-scalp", "stasik-tbank-swing", "stasik-dashboard", "stasik-midas", "stasik-turtle", "stasik-turtle-tbank", "stasik-smc", "stasik-fiba", "stasik-buba", "stasik-shakal", "stasik-shakal1h", "stasik-fin"}
            if service not in allowed:
                return web.json_response({"ok": False, "error": "Unknown service"}, status=400)

            result = subprocess.run(
                ["systemctl", action, service],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return web.json_response({"ok": False, "error": result.stderr.strip()}, status=500)
            return web.json_response({"ok": True, "message": f"{service} {action}ed"})
        except Exception as e:
            logger.exception("toggle-service error")
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    async def _api_close_position(self, request: web.Request) -> web.Response:
        """Close a single position by symbol."""
        if self.engine:
            try:
                data = await request.json()
                symbol = data.get("symbol", "")
                if not symbol:
                    return web.json_response({"ok": False, "error": "No symbol"}, status=400)
                result = await self.engine.close_position(symbol)
                return web.json_response({"ok": True, "message": result})
            except Exception as e:
                logger.exception("close-position error")
                return web.json_response({"ok": False, "error": str(e)}, status=500)
        # Standalone mode — close via exchange client directly
        try:
            data = await request.json()
            symbol = data.get("symbol", "")
            instance = data.get("instance", "")
            if not symbol:
                return web.json_response({"ok": False, "error": "No symbol"}, status=400)
            # Determine exchange type from instance name
            is_tbank = any(k in (instance or "").upper() for k in ("TBANK", "MIDAS"))
            exchange_closed = False
            if is_tbank:
                try:
                    from src.exchange.tbank_client import TBankClient
                    # Find matching tbank config
                    for inst in _other_instances(self.config):
                        if inst.get("name", "").upper() == (instance or "").upper():
                            cfg_path = inst.get("config_path", "")
                            if cfg_path and Path(cfg_path).exists():
                                import yaml
                                with open(cfg_path) as f:
                                    tcfg = yaml.safe_load(f)
                                tc = TBankClient(tcfg)
                                positions = tc.get_positions(symbol=symbol)
                                for p in positions:
                                    if p["symbol"] == symbol and p["size"] > 0:
                                        close_side = "Sell" if p["side"] == "Buy" else "Buy"
                                        tc.place_order(symbol=symbol, side=close_side, qty=p["size"])
                                        exchange_closed = True
                            break
                except Exception as e:
                    err_str = str(e)
                    if "30079" in err_str or "not available" in err_str.lower():
                        return web.json_response({"ok": False, "error": "MOEX закрыта"}, status=400)
                    logger.warning("Failed to close %s on TBank: %s", symbol, e)
            else:
                client = self._get_client()
                if client:
                    try:
                        positions = client.get_positions(symbol=symbol, category="linear")
                        for p in positions:
                            if p["symbol"] == symbol and p["size"] > 0:
                                close_side = "Sell" if p["side"] == "Buy" else "Buy"
                                client.place_order(symbol=symbol, side=close_side, qty=p["size"], category="linear", reduce_only=True)
                                exchange_closed = True
                    except Exception:
                        logger.warning("Failed to close %s on Bybit", symbol)
            # Close in DB (get current price for PnL)
            db_path = self._resolve_instance_db(instance)
            if not db_path:
                # Fallback: find which DB has this symbol
                all_dbs = [(str(self.db.db_path), self.config.get("instance_name", "SCALP"))]
                for inst in _other_instances(self.config):
                    all_dbs.append((inst.get("db_path", ""), inst.get("name", "")))
                for dp, _ in all_dbs:
                    if dp and Path(dp).exists():
                        try:
                            async with aiosqlite.connect(dp) as db:
                                cur = await db.execute(
                                    "SELECT id FROM trades WHERE symbol=? AND status='open'", (symbol,))
                                if await cur.fetchone():
                                    db_path = dp
                                    break
                        except Exception:
                            pass
            if db_path and Path(db_path).exists():
                try:
                    mark = 0
                    try:
                        mark = client.get_last_price(symbol, category="linear")
                    except Exception:
                        pass
                    async with aiosqlite.connect(db_path) as db:
                        db.row_factory = aiosqlite.Row
                        cur = await db.execute(
                            "SELECT id, side, entry_price, qty FROM trades WHERE symbol=? AND status='open'",
                            (symbol,))
                        rows = await cur.fetchall()
                        for r in rows:
                            entry = float(r["entry_price"])
                            qty = float(r["qty"])
                            pnl = 0.0
                            if mark > 0:
                                direction = 1 if r["side"] == "Buy" else -1
                                pnl = round((mark - entry) * qty * direction, 2)
                            await db.execute(
                                "UPDATE trades SET exit_price=?, pnl=?, status='closed', closed_at=? WHERE id=?",
                                (mark if mark > 0 else entry, pnl, datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%dT%H:%M:%S.%f"), r["id"]))
                        await db.commit()
                        logger.info("Closed %d trade(s) in DB for %s (pnl per trade calculated)", len(rows), symbol)
                except Exception:
                    logger.exception("Failed to close %s in DB", symbol)
            msg = f"Закрыто: {symbol}" + ("" if exchange_closed else " (только в БД)")
            return web.json_response({"ok": True, "message": msg})
        except Exception as e:
            logger.exception("close-position error")
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    async def _api_close_all(self, request: web.Request) -> web.Response:
        """Close all open positions."""
        if self.engine:
            try:
                result = await self.engine.close_all_positions()
                return web.json_response({"ok": True, "message": result})
            except Exception as e:
                logger.exception("close-all error")
                return web.json_response({"ok": False, "error": str(e)}, status=500)
        # Standalone mode
        count = 0
        errors = []

        # 1. Close Bybit positions
        client = self._get_client()
        if client:
            try:
                positions = client.get_positions(category="linear")
                for p in positions:
                    if p["size"] > 0:
                        close_side = "Sell" if p["side"] == "Buy" else "Buy"
                        client.place_order(symbol=p["symbol"], side=close_side, qty=p["size"], category="linear", reduce_only=True)
                        count += 1
            except Exception as e:
                logger.exception("close-all Bybit error")
                errors.append(f"Bybit: {e}")

        # 2. Close TBank/Midas positions
        for inst in _other_instances(self.config):
            inst_name = (inst.get("name") or "").upper()
            if not any(k in inst_name for k in ("TBANK", "MIDAS")):
                continue
            cfg_path = inst.get("config_path", "")
            if not cfg_path or not Path(cfg_path).exists():
                continue
            try:
                import yaml
                from src.exchange.tbank_client import TBankClient
                with open(cfg_path) as f:
                    tcfg = yaml.safe_load(f)
                tc = TBankClient(tcfg)
                positions = tc.get_positions()
                for p in positions:
                    if p["size"] > 0:
                        close_side = "Sell" if p["side"] == "Buy" else "Buy"
                        tc.place_order(symbol=p["symbol"], side=close_side, qty=p["size"])
                        count += 1
            except Exception as e:
                err_str = str(e)
                if "30079" in err_str or "not available" in err_str.lower():
                    errors.append(f"{inst_name}: MOEX закрыта")
                else:
                    logger.exception("close-all %s error", inst_name)
                    errors.append(f"{inst_name}: {e}")

        # 3. Close all open trades in all DBs
        all_dbs = [(str(self.db.db_path), self.config.get("instance_name", "SCALP"))]
        for inst in _other_instances(self.config):
            all_dbs.append((inst.get("db_path", ""), inst.get("name", "")))
        for db_path, db_inst in all_dbs:
            if not db_path or not Path(db_path).exists():
                continue
            try:
                async with aiosqlite.connect(db_path) as db:
                    await db.execute(
                        "UPDATE trades SET status='closed', closed_at=? WHERE status='open'",
                        (datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%dT%H:%M:%S.%f"),))
                    await db.commit()
            except Exception:
                logger.warning("close-all: failed to close trades in DB %s", db_path)

        msg = f"Закрыто {count} позиций"
        if errors:
            msg += " (ошибки: " + "; ".join(errors) + ")"
        return web.json_response({"ok": True, "message": msg})

    async def _api_double_position(self, request: web.Request) -> web.Response:
        """Open a new position with 2x size in the same direction."""
        client = self._get_client()
        if not client:
            return web.json_response({"ok": False, "error": "No client"}, status=503)
        try:
            data = await request.json()
            symbol = data.get("symbol", "")
            side = data.get("side", "")
            qty = float(data.get("qty", 0))
            instance = data.get("instance", "")
            if not symbol or not side or qty <= 0:
                return web.json_response({"ok": False, "error": "Missing symbol/side/qty"}, status=400)
            # Add same qty to double the position (existing + new = 2x)
            client.place_order(symbol=symbol, side=side, qty=qty, category="linear")
            # Update qty in the correct DB so dashboard reflects the change
            db_path = self._resolve_instance_db(instance)
            if db_path and Path(db_path).exists():
                try:
                    async with aiosqlite.connect(db_path) as db:
                        await db.execute(
                            "UPDATE trades SET qty = qty + ? WHERE symbol = ? AND status = 'open'",
                            (qty, symbol),
                        )
                        await db.commit()
                except Exception:
                    logger.warning("Failed to update qty in DB for %s", symbol)
            return web.json_response({"ok": True, "message": f"X2: {side} +{qty} {symbol}"})
        except Exception as e:
            logger.exception("double-position error")
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    async def _api_set_sl(self, request: web.Request) -> web.Response:
        """Set stop-loss by loss amount (USDT). Calculates SL price and sets on exchange + DB."""
        client = self._get_client()
        if not client:
            return web.json_response({"ok": False, "error": "No client"}, status=503)
        try:
            data = await request.json()
            symbol = data.get("symbol", "")
            loss_amount = float(data.get("amount", 0))
            instance = data.get("instance", "")
            side = data.get("side", "")
            entry_price = float(data.get("entry_price", 0))
            qty = float(data.get("qty", 0))
            if not symbol or loss_amount <= 0 or not side or entry_price <= 0 or qty <= 0:
                return web.json_response({"ok": False, "error": "Missing params"}, status=400)

            # Calculate SL price from loss amount
            # Buy: sl_price = entry - loss / qty
            # Sell: sl_price = entry + loss / qty
            if side == "Buy":
                sl_price = entry_price - loss_amount / qty
            else:
                sl_price = entry_price + loss_amount / qty

            if sl_price <= 0:
                return web.json_response({"ok": False, "error": "SL price <= 0"}, status=400)

            sl_price = round(sl_price, 6)

            # Update SL in DB first
            db_path = self._resolve_instance_db(instance)
            if db_path and Path(db_path).exists():
                try:
                    async with aiosqlite.connect(db_path) as db:
                        await db.execute(
                            "UPDATE trades SET stop_loss = ? WHERE symbol = ? AND status = 'open'",
                            (sl_price, symbol),
                        )
                        await db.commit()
                except Exception:
                    logger.warning("Failed to update SL in DB for %s", symbol)

            # Set SL on Bybit (may fail if no position on exchange)
            exchange_ok = False
            try:
                client.session.set_trading_stop(
                    category="linear", symbol=symbol,
                    stopLoss=str(sl_price), positionIdx=client._pos_idx(side),
                )
                exchange_ok = True
            except Exception:
                logger.warning("set_trading_stop failed for %s (no position on exchange?)", symbol)

            logger.info("SL set: %s %s @ %s (loss=%s, exchange=%s)", side, symbol, sl_price, loss_amount, exchange_ok)
            return web.json_response({"ok": True, "sl_price": sl_price, "exchange": exchange_ok})
        except Exception as e:
            logger.exception("set-sl error")
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    async def _api_set_tp(self, request: web.Request) -> web.Response:
        """Set take-profit by profit amount (USDT). Calculates TP price and sets on exchange + DB."""
        client = self._get_client()
        if not client:
            return web.json_response({"ok": False, "error": "No client"}, status=503)
        try:
            data = await request.json()
            symbol = data.get("symbol", "")
            profit_amount = float(data.get("amount", 0))
            instance = data.get("instance", "")
            side = data.get("side", "")
            entry_price = float(data.get("entry_price", 0))
            qty = float(data.get("qty", 0))
            if not symbol or profit_amount <= 0 or not side or entry_price <= 0 or qty <= 0:
                return web.json_response({"ok": False, "error": "Missing params"}, status=400)

            # Calculate TP price from profit amount
            # Buy: tp_price = entry + profit / qty
            # Sell: tp_price = entry - profit / qty
            if side == "Buy":
                tp_price = entry_price + profit_amount / qty
            else:
                tp_price = entry_price - profit_amount / qty

            if tp_price <= 0:
                return web.json_response({"ok": False, "error": "TP price <= 0"}, status=400)

            tp_price = round(tp_price, 6)

            # Update TP in DB first
            db_path = self._resolve_instance_db(instance)
            if db_path and Path(db_path).exists():
                try:
                    async with aiosqlite.connect(db_path) as db:
                        await db.execute(
                            "UPDATE trades SET take_profit = ? WHERE symbol = ? AND status = 'open'",
                            (tp_price, symbol),
                        )
                        await db.commit()
                except Exception:
                    logger.warning("Failed to update TP in DB for %s", symbol)

            # Set TP on Bybit (may fail if no position on exchange)
            exchange_ok = False
            try:
                client.session.set_trading_stop(
                    category="linear", symbol=symbol,
                    takeProfit=str(tp_price), positionIdx=client._pos_idx(side),
                )
                exchange_ok = True
            except Exception:
                logger.warning("set_trading_stop (TP) failed for %s (no position on exchange?)", symbol)

            logger.info("TP set: %s %s @ %s (profit=%s, exchange=%s)", side, symbol, tp_price, profit_amount, exchange_ok)
            return web.json_response({"ok": True, "tp_price": tp_price, "exchange": exchange_ok})
        except Exception as e:
            logger.exception("set-tp error")
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    def _resolve_instance_db(self, instance: str) -> str | None:
        """Map instance name to its DB path."""
        instance_upper = (instance or "").upper()
        main_name = self.config.get("instance_name", "SCALP").upper()
        if not instance_upper or instance_upper == main_name:
            return str(self.db.db_path)
        for inst in _other_instances(self.config):
            if inst.get("name", "").upper() == instance_upper:
                return inst.get("db_path", "")
        return None

    _ALLOWED_LEVERAGE = {1, 2, 3, 5, 10, 15, 20}

    def _get_instance_pairs(self, instance: str) -> list[str]:
        instance_upper = (instance or "").upper()
        main_name = self.config.get("instance_name", "SCALP").upper()
        if instance_upper == main_name:
            return self.config.get("trading", {}).get("pairs", [])
        for inst in _other_instances(self.config):
            if inst.get("name", "").upper() == instance_upper:
                return inst.get("pairs", [])
        return []

    def _get_instance_config_leverage(self, instance: str) -> int:
        """Get configured leverage for an instance from its YAML config."""
        import yaml as _yaml
        instance_upper = (instance or "").upper()
        main_name = self.config.get("instance_name", "SCALP").upper()
        if instance_upper == main_name:
            return int(self.config.get("trading", {}).get("leverage", 10))
        for inst in _other_instances(self.config):
            if inst.get("name", "").upper() == instance_upper:
                # Try loading the actual config file for accurate leverage
                svc = inst.get("service", "")
                cfg_map = {
                    "stasik-degen": "config/degen.yaml",
                }
                cfg_path = cfg_map.get(svc, "")
                if cfg_path:
                    try:
                        with open(cfg_path) as f:
                            inst_cfg = _yaml.safe_load(f)
                        return int(inst_cfg.get("trading", {}).get("leverage", 10))
                    except Exception:
                        pass
                # Fallback: parse from "10x" string
                lev_str = str(inst.get("leverage", "10x")).replace("x", "")
                try:
                    return int(lev_str)
                except ValueError:
                    return 10
        return 10

    async def _api_set_leverage(self, request: web.Request) -> web.Response:
        client = self._get_client()
        if not client:
            return web.json_response({"ok": False, "error": "No client"}, status=503)
        try:
            data = await request.json()
            raw_leverage = data.get("leverage", 0)
            restore_config = raw_leverage == "config"
            leverage = 0 if restore_config else int(raw_leverage)
            symbol = data.get("symbol", "")
            instance = data.get("instance", "")

            if not restore_config and leverage not in self._ALLOWED_LEVERAGE:
                return web.json_response(
                    {"ok": False, "error": f"Invalid leverage: {leverage}"},
                    status=400,
                )

            results = {"ok": [], "failed": []}

            if symbol:
                try:
                    client.set_leverage(symbol, leverage, category="linear")
                    results["ok"].append(symbol)
                except Exception as e:
                    results["failed"].append({"symbol": symbol, "error": str(e)})
            elif instance:
                pairs = self._get_instance_pairs(instance)
                if not pairs:
                    return web.json_response(
                        {"ok": False, "error": f"Unknown instance: {instance}"},
                        status=400,
                    )
                inst_leverage = self._get_instance_config_leverage(instance) if restore_config else leverage
                for pair in pairs:
                    try:
                        client.set_leverage(pair, inst_leverage, category="linear")
                        results["ok"].append(pair)
                    except Exception as e:
                        if "leverage not modified" not in str(e).lower():
                            results["failed"].append({"symbol": pair, "error": str(e)})
                        else:
                            results["ok"].append(pair)
                leverage = inst_leverage
            else:
                return web.json_response(
                    {"ok": False, "error": "Provide 'symbol' or 'instance'"},
                    status=400,
                )

            success = len(results["failed"]) == 0
            logger.info("set-leverage: %sx, ok=%s, failed=%s", leverage, results["ok"], results["failed"])
            return web.json_response({"ok": success, "leverage": leverage, "results": results})
        except Exception as e:
            logger.exception("set-leverage error")
            return web.json_response({"ok": False, "error": str(e)}, status=500)
