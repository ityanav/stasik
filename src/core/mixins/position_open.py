import logging
import math
from datetime import datetime, timedelta, timezone

from src.strategy.signals import SMCGenerator

logger = logging.getLogger(__name__)


class PositionOpenMixin:
    """Methods for opening new positions, extracted from TradingEngine."""

    async def _open_trade(
        self, symbol: str, side: str, category: str, score: int, details: dict,
        ai_reasoning: str = "",
        ai_sl_pct: float | None = None,
        ai_tp_pct: float | None = None,
        ai_size_mult: float | None = None,
        atr: float | None = None,
        is_scale_in: bool = False,
        df=None,
    ):
        price = self.client.get_last_price(symbol, category=category)
        balance = self.client.get_balance()

        # ATR-based SL/TP (default), AI overrides take priority
        atr_sl_pct = None
        atr_tp_pct = None
        if atr and atr > 0:
            _, _, atr_sl_pct, atr_tp_pct = self.risk.calculate_sl_tp_atr(
                price, side, atr, self._atr_sl_mult, self._atr_tp_mult
            )

        info = self._get_instrument_info(symbol, category)
        sizing_sl = atr_sl_pct if atr_sl_pct else None
        qty = self.risk.calculate_position_size(
            balance=balance,
            price=price,
            qty_step=info["qty_step"],
            min_qty=info["min_qty"],
            sl_pct=sizing_sl,
            leverage=self.leverage,
        )

        # AI position size multiplier
        if ai_size_mult is not None and 0.1 <= ai_size_mult <= 2.0:
            qty = math.floor((qty * ai_size_mult) / info["qty_step"]) * info["qty_step"]
            qty = round(qty, 8)

        # Cap qty by instrument max
        max_qty = info.get("max_qty", 0)
        if max_qty > 0 and qty > max_qty:
            logger.info("Qty capped: %.2f -> %.2f (max_qty %s)", qty, max_qty, symbol)
            qty = math.floor(max_qty / info["qty_step"]) * info["qty_step"]
            qty = round(qty, 8)

        if qty <= 0:
            logger.info("Position size too small for %s, skipping", symbol)
            return

        # SL/TP priority: SMC (swept_level+ATR) > AI > ATR > fixed
        sl_source = "fixed"
        tp_source = "fixed"

        # SMC SL/TP: behind swept_level + 0.5 ATR buffer; TP from Fib extension
        smc_sl = None
        smc_tp = None
        if isinstance(self.signal_gen, SMCGenerator) and details:
            sweep_level = details.get("sweep_level", 0)
            tp1_level = details.get("tp1_level", 0)
            atr_buf = atr * 0.5 if atr and atr > 0 else 0
            if sweep_level and sweep_level > 0:
                if side == "Buy":
                    smc_sl = round(sweep_level - atr_buf, 6)
                else:
                    smc_sl = round(sweep_level + atr_buf, 6)
                # Enforce minimum SL distance (1 ATR)
                sl_dist = abs(price - smc_sl)
                min_sl_dist = atr if atr and atr > 0 else price * 0.005
                if sl_dist < min_sl_dist:
                    if side == "Buy":
                        smc_sl = round(price - min_sl_dist, 6)
                    else:
                        smc_sl = round(price + min_sl_dist, 6)
                    logger.info("FIBA SL widened: %s %.6f → %.6f (min 1 ATR=%.6f)",
                                symbol, sweep_level - atr_buf if side == "Buy" else sweep_level + atr_buf,
                                smc_sl, min_sl_dist)
            if tp1_level and tp1_level > 0:
                smc_tp = round(tp1_level, 6)
                # Cap SMC TP by ATR multiplier so it doesn't overshoot
                if atr and atr > 0:
                    max_tp_dist = atr * self._atr_tp_mult
                    if side == "Buy":
                        atr_cap = round(price + max_tp_dist, 6)
                        if smc_tp > atr_cap:
                            logger.info("FIBA TP capped: %s %.6f → %.6f (ATR cap %.2f%%)",
                                        symbol, smc_tp, atr_cap, max_tp_dist / price * 100)
                            smc_tp = atr_cap
                    else:
                        atr_cap = round(price - max_tp_dist, 6)
                        if smc_tp < atr_cap:
                            logger.info("FIBA TP capped: %s %.6f → %.6f (ATR cap %.2f%%)",
                                        symbol, smc_tp, atr_cap, max_tp_dist / price * 100)
                            smc_tp = atr_cap

        if smc_sl is not None:
            sl = smc_sl
            sl_dist_pct = abs(price - sl) / price * 100
            sl_source = f"FIBA:sweep({sl_dist_pct:.2f}%)"
        elif ai_sl_pct is not None and 0.3 <= ai_sl_pct <= 5.0:
            sl = price * (1 - ai_sl_pct / 100) if side == "Buy" else price * (1 + ai_sl_pct / 100)
            sl = round(sl, 6)
            sl_source = f"AI:{ai_sl_pct:.1f}%"
        elif atr_sl_pct:
            sl = self.risk.calculate_sl_tp_atr(price, side, atr, self._atr_sl_mult, self._atr_tp_mult)[0]
            sl_source = f"ATR:{atr_sl_pct*100:.2f}%"
        else:
            sl = self.risk.calculate_sl_tp(price, side)[0]

        # Clamp SL by max dollar loss (max_sl_usd)
        max_sl_usd = self.config.get("risk", {}).get("max_sl_usd", 0)
        if max_sl_usd > 0 and qty > 0:
            sl_loss = abs(sl - price) * qty
            if sl_loss > max_sl_usd:
                max_sl_dist = max_sl_usd / qty
                if side == "Buy":
                    sl = round(price - max_sl_dist, 6)
                else:
                    sl = round(price + max_sl_dist, 6)
                sl_source += f"→cap${max_sl_usd}"
                logger.info("SL capped: %s $%.0f → $%.0f (max_sl_usd=%d)", symbol, sl_loss, max_sl_usd, max_sl_usd)

        if smc_tp is not None:
            tp = smc_tp
            tp_dist_pct = abs(tp - price) / price * 100
            tp_source = f"FIBA:fib_ext({tp_dist_pct:.2f}%)"
        elif ai_tp_pct is not None and 0.5 <= ai_tp_pct <= 10.0:
            tp = price * (1 + ai_tp_pct / 100) if side == "Buy" else price * (1 - ai_tp_pct / 100)
            tp = round(tp, 6)
            tp_source = f"AI:{ai_tp_pct:.1f}%"
        elif atr_tp_pct:
            tp = self.risk.calculate_sl_tp_atr(price, side, atr, self._atr_sl_mult, self._atr_tp_mult)[1]
            tp_source = f"ATR:{atr_tp_pct*100:.2f}%"
        else:
            tp = self.risk.calculate_sl_tp(price, side)[1]

        # TP с учётом комиссии: чистая прибыль >= min_tp_net после round-trip fee
        min_tp_net = self.config.get("risk", {}).get("min_tp_net", 0)
        if min_tp_net > 0 and qty > 0:
            position_value = price * qty
            round_trip_fee = position_value * self.risk.commission_rate * 2
            min_gross_profit = min_tp_net + round_trip_fee
            current_tp_profit = abs(tp - price) * qty
            if current_tp_profit < min_gross_profit:
                min_tp_dist = min_gross_profit / qty
                if side == "Buy":
                    tp = round(price + min_tp_dist, 6)
                else:
                    tp = round(price - min_tp_dist, 6)
                tp_source += f"→min${min_tp_net}"
                logger.info("TP adjusted for commission: %s $%.0f → $%.0f (net=$%.0f + fee=$%.2f)",
                            symbol, current_tp_profit, min_gross_profit, min_tp_net, round_trip_fee)

        # SL cap: SL distance must not exceed 50% of TP distance (R:R >= 2:1)
        if tp > 0 and sl > 0:
            sl_dist = abs(price - sl)
            tp_dist = abs(tp - price)
            max_sl_dist = tp_dist * 0.5
            if tp_dist > 0 and sl_dist > max_sl_dist:
                old_sl = sl
                if side == "Buy":
                    sl = round(price - max_sl_dist, 6)
                else:
                    sl = round(price + max_sl_dist, 6)
                sl_source += f"→50%TP"
                logger.info("SL tightened to 50%% TP: %s %.6f → %.6f (SL %.2f%% → %.2f%%, TP %.2f%%)",
                            symbol, old_sl, sl, sl_dist / price * 100, max_sl_dist / price * 100, tp_dist / price * 100)

        # Place order with retry on qty rejection
        order = None
        for attempt in range(3):
            try:
                if self.exchange_type == "tbank":
                    order = self.client.place_order(
                        symbol=symbol, side=side, qty=qty,
                        stop_loss=sl, take_profit=tp,
                    )
                else:
                    order = self.client.place_order(
                        symbol=symbol, side=side, qty=qty, category=category,
                        stop_loss=sl if category == "linear" else None,
                        take_profit=tp if category == "linear" else None,
                    )
                break
            except Exception as e:
                err_msg = str(e).lower()
                if "max_qty" in err_msg or "exceeds maximum" in err_msg or "too large" in err_msg:
                    qty = math.floor(qty * 0.5 / info["qty_step"]) * info["qty_step"]
                    qty = round(qty, 8)
                    if qty <= 0:
                        logger.warning("Order rejected for %s: qty too large, cannot reduce further", symbol)
                        return
                    logger.info("Order qty rejected for %s, retrying with qty=%.2f (attempt %d)", symbol, qty, attempt + 2)
                else:
                    raise
        if order is None:
            logger.warning("Failed to place order for %s after retries", symbol)
            return

        order_id = order.get("orderId", "")
        await self.db.insert_trade(
            symbol=symbol,
            side=side,
            category=category,
            qty=qty,
            entry_price=price,
            stop_loss=sl,
            take_profit=tp,
            order_id="scale_in" if is_scale_in else order_id,
        )

        # Set trailing stop for futures / tbank (ATR-based or fixed)
        # Skip exchange-level trailing if multiple positions per symbol
        trailing_msg = ""
        open_trades = await self.db.get_open_trades()
        symbol_count = sum(1 for t in open_trades if t["symbol"] == symbol)
        if category in ("linear", "tbank") and symbol_count <= 1:
            if atr and atr > 0:
                trailing_distance = self.risk.calculate_trailing_distance_atr(
                    atr, price, self._atr_trail_mult
                )
                if trailing_distance > 0:
                    trail_pct = trailing_distance / price * 100
                    trail_activation_pct = self.config["risk"].get("trailing_activation", 0)
                    if side == "Buy":
                        active_price = price * (1 + trail_activation_pct / 100)
                    else:
                        active_price = price * (1 - trail_activation_pct / 100)
                    # Ensure active_price differs from entry (Bybit requirement)
                    if active_price == price:
                        tick = price * 0.001  # 0.1% minimum activation
                        active_price = price + tick if side == "Buy" else price - tick
                    self.client.set_trailing_stop(
                        symbol=symbol,
                        trailing_stop=trailing_distance,
                        active_price=active_price,
                    )
                    trailing_msg = f"\n📐 Trailing SL: {trail_pct:.2f}% ATR (активация при {trail_activation_pct}% прибыли)"
            else:
                trail_pct = self.config["risk"].get("trailing_stop", 0)
                trail_activation_pct = self.config["risk"].get("trailing_activation", 0)
                if trail_pct > 0:
                    trailing_distance = price * trail_pct / 100
                    if side == "Buy":
                        active_price = price * (1 + trail_activation_pct / 100)
                    else:
                        active_price = price * (1 - trail_activation_pct / 100)
                    self.client.set_trailing_stop(
                        symbol=symbol,
                        trailing_stop=trailing_distance,
                        active_price=active_price,
                    )
                    trailing_msg = f"\n📐 Trailing SL: {trail_pct}% (активация при {trail_activation_pct}% прибыли)"
        elif symbol_count > 1:
            trailing_msg = "\n📐 Trailing: пропущен (мульти-позиция)"

        direction = "LONG" if side == "Buy" else "SHORT"
        pos_value = qty * price
        from datetime import datetime, timezone, timedelta
        msk = datetime.now(timezone(timedelta(hours=3)))
        msk_time = msk.strftime("%H:%M")
        exchange = self.config.get("exchange", "bybit")
        cur = "RUB" if exchange == "tbank" else "USDT"
        ps_fmt = f"{pos_value:,.0f}" if pos_value < 100_000 else f"{pos_value/1000:,.1f}k"
        inst = self.instance_name or "BOT"
        lev = self.leverage if hasattr(self, 'leverage') and self.leverage else 1
        msg = f"📥 {inst} | {direction} {symbol}\n   {ps_fmt} {cur} × {lev}x | {msk_time} MSK"
        logger.info(msg)
        await self._notify(msg)

    async def _open_turtle_trade(self, symbol: str, side: str, category: str,
                                  n_value: float, system: int | None, df):
        """Open first Turtle unit with N-based sizing, SL=2N, no TP."""
        price = self.client.get_last_price(symbol, category=category)
        balance = self.client.get_balance()
        if price <= 0 or n_value <= 0:
            return

        # Unit size: (risk_pct% of equity) / N per unit
        dollar_risk = balance * (self._turtle_risk_pct / 100)
        qty_raw = dollar_risk / n_value
        info = self._get_instrument_info(symbol, category)
        qty = math.floor(qty_raw / info["qty_step"]) * info["qty_step"]
        qty = round(qty, 8)
        if qty <= 0:
            logger.info("Turtle: position size too small for %s (N=%.4f)", symbol, n_value)
            return

        # SL = 2N from entry
        sl_dist = self._turtle_sl_n_mult * n_value
        if side == "Buy":
            sl = round(price - sl_dist, 6)
        else:
            sl = round(price + sl_dist, 6)

        # No TP — exit only via Donchian exit channel
        try:
            order = self.client.place_order(
                symbol=symbol, side=side, qty=qty, category=category,
                stop_loss=sl, take_profit=None,
            )
        except Exception:
            logger.exception("Turtle: failed to place order for %s", symbol)
            return

        order_id = order.get("orderId", "")
        await self.db.insert_trade(
            symbol=symbol, side=side, category=category,
            qty=qty, entry_price=price, stop_loss=sl, take_profit=0,
            order_id=order_id,
        )

        # Track turtle state
        self._turtle_state[symbol] = {
            "units": 1,
            "entries": [{"price": price, "qty": qty}],
            "side": side,
            "system": system or 1,
            "last_add_price": price,
            "entry_n": n_value,
        }

        direction = "LONG" if side == "Buy" else "SHORT"
        pos_value = qty * price
        daily_total = await self._get_daily_total_pnl()
        day_arrow = "▲" if daily_total >= 0 else "▼"
        msg = (
            f"🐢 Turtle {direction} {symbol} (S{system or '?'})\n"
            f"Price: {price:.2f} | N: {n_value:.4f}\n"
            f"SL: {sl:.2f} ({self._turtle_sl_n_mult}N) | TP: exit channel\n"
            f"Unit 1/{self._turtle_max_units} | Size: {qty} (~${pos_value:,.0f})\n"
            f"Balance: {balance:,.0f} USDT ({day_arrow} {daily_total:+,.0f})"
        )
        logger.info(msg)
        await self._notify(msg)

    async def _check_turtle_pyramid(self, symbol: str, df, state: dict,
                                     n_value: float, category: str):
        """Add unit every 0.5N price movement in profit direction (max 4 units)."""
        if state["units"] >= self._turtle_max_units:
            return
        if n_value <= 0:
            return

        price = df["close"].iloc[-1]
        last_add = state["last_add_price"]
        side = state["side"]
        threshold = self._turtle_pyramid_n_mult * n_value

        # Check if price moved enough since last unit addition
        if side == "Buy" and price < last_add + threshold:
            return
        if side == "Sell" and price > last_add - threshold:
            return

        if not self._check_margin_limit():
            return

        balance = self.client.get_balance()
        dollar_risk = balance * (self._turtle_risk_pct / 100)
        qty_raw = dollar_risk / n_value
        info = self._get_instrument_info(symbol, category)
        qty = math.floor(qty_raw / info["qty_step"]) * info["qty_step"]
        qty = round(qty, 8)
        if qty <= 0:
            return

        # New SL for ALL units: 2N from this new entry
        sl_dist = self._turtle_sl_n_mult * n_value
        if side == "Buy":
            new_sl = round(price - sl_dist, 6)
        else:
            new_sl = round(price + sl_dist, 6)

        try:
            self.client.place_order(
                symbol=symbol, side=side, qty=qty, category=category,
                stop_loss=new_sl, take_profit=None,
            )
        except Exception:
            logger.exception("Turtle pyramid: failed for %s", symbol)
            return

        await self.db.insert_trade(
            symbol=symbol, side=side, category=category,
            qty=qty, entry_price=price, stop_loss=new_sl, take_profit=0,
            order_id=f"pyramid_{state['units'] + 1}",
        )

        # Update SL on all existing trades for this symbol
        await self._turtle_update_trailing_sl(symbol, new_sl)

        state["units"] += 1
        state["entries"].append({"price": price, "qty": qty})
        state["last_add_price"] = price
        state["entry_n"] = n_value

        unit_num = state["units"]
        logger.info(
            "🐢 Turtle PYRAMID %s %s: unit %d/%d @ %.2f (SL moved to %.2f)",
            side, symbol, unit_num, self._turtle_max_units, price, new_sl,
        )
        await self._notify(
            f"🐢 Pyramid {side} {symbol}: unit {unit_num}/{self._turtle_max_units} @ {price:.2f}\n"
            f"SL all units → {new_sl:.2f}"
        )
