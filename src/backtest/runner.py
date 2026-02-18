"""
Backtester for Stasik trading strategy.
Usage: python3 -m src.backtest.runner --symbol BTCUSDT --days 7
"""
import argparse
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from src.exchange.client import BybitClient
from src.strategy.indicators import calculate_adx
from src.strategy.signals import Signal, SignalGenerator, Trend

logger = logging.getLogger(__name__)


@dataclass
class VirtualTrade:
    symbol: str
    side: str
    entry_price: float
    qty: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    exit_price: float | None = None
    exit_time: datetime | None = None
    pnl: float = 0.0
    exit_reason: str = ""
    partial_closed: bool = False


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    period_days: int
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trades: list[VirtualTrade] = field(default_factory=list)
    signals_generated: int = 0
    signals_filtered: int = 0


class Backtester:
    def __init__(self, config: dict):
        self.config = config
        self.client = BybitClient(config)
        self.signal_gen = SignalGenerator(config)
        self.risk_config = config["risk"]
        self.sl_pct = self.risk_config["stop_loss"] / 100
        self.tp_pct = self.risk_config["take_profit"] / 100
        self.partial_close_enabled = self.risk_config.get("partial_close_enabled", True)
        self.partial_trigger = self.risk_config.get("partial_close_trigger", 50) / 100
        self.adx_min = config["strategy"].get("adx_min", 20)
        self.adx_period = config["strategy"].get("adx_period", 14)

    def run(self, symbol: str, days: int = 30, timeframe: str = "1") -> BacktestResult:
        result = BacktestResult(symbol=symbol, timeframe=timeframe, period_days=days)

        df = self._fetch_historical(symbol, timeframe, days)
        if len(df) < 200:
            logger.error("Not enough historical data: %d candles", len(df))
            return result

        logger.info("Backtesting %s on %d candles (%d days)", symbol, len(df), days)

        # HTF data for trend filter
        htf_tf = str(self.config["trading"].get("htf_timeframe", "15"))
        htf_df = self.client.get_klines(symbol=symbol, interval=htf_tf, limit=1000, category="linear")

        open_trade: VirtualTrade | None = None
        balance = 10000.0
        peak_balance = balance
        max_dd = 0.0
        pnl_list: list[float] = []

        for i in range(200, len(df)):
            window = df.iloc[:i + 1].copy()
            window.attrs["symbol"] = symbol
            candle = df.iloc[i]

            # Check open trade
            if open_trade:
                hit = self._check_trade_exit(open_trade, candle)
                if hit:
                    open_trade.exit_time = candle["timestamp"]
                    balance += open_trade.pnl
                    pnl_list.append(open_trade.pnl)
                    result.trades.append(open_trade)
                    if open_trade.pnl >= 0:
                        result.wins += 1
                    else:
                        result.losses += 1
                    result.total_pnl += open_trade.pnl
                    peak_balance = max(peak_balance, balance)
                    dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
                    max_dd = max(max_dd, dd)
                    open_trade = None
                continue

            # Generate signal
            sig = self.signal_gen.generate(window)
            if sig.signal == Signal.HOLD:
                continue

            result.signals_generated += 1

            # ADX filter
            try:
                adx = calculate_adx(window, self.adx_period)
                if adx < self.adx_min:
                    result.signals_filtered += 1
                    continue
            except Exception:
                pass

            # HTF trend filter
            htf_trend = self.signal_gen.get_htf_trend(htf_df)
            if htf_trend == Trend.BEARISH and sig.signal == Signal.BUY:
                result.signals_filtered += 1
                continue
            if htf_trend == Trend.BULLISH and sig.signal == Signal.SELL:
                result.signals_filtered += 1
                continue

            # Open virtual trade
            side = "Buy" if sig.signal == Signal.BUY else "Sell"
            price = candle["close"]
            risk_amount = balance * (self.risk_config["risk_per_trade"] / 100)
            position_value = risk_amount / self.sl_pct
            qty = position_value / price

            if side == "Buy":
                sl = price * (1 - self.sl_pct)
                tp = price * (1 + self.tp_pct)
            else:
                sl = price * (1 + self.sl_pct)
                tp = price * (1 - self.tp_pct)

            open_trade = VirtualTrade(
                symbol=symbol, side=side, entry_price=price,
                qty=qty, stop_loss=sl, take_profit=tp,
                entry_time=candle["timestamp"],
            )

        # Close remaining trade
        if open_trade:
            last = df.iloc[-1]
            open_trade.exit_price = last["close"]
            open_trade.exit_time = last["timestamp"]
            if open_trade.side == "Buy":
                open_trade.pnl = (last["close"] - open_trade.entry_price) * open_trade.qty
            else:
                open_trade.pnl = (open_trade.entry_price - last["close"]) * open_trade.qty
            open_trade.exit_reason = "end"
            balance += open_trade.pnl
            pnl_list.append(open_trade.pnl)
            result.trades.append(open_trade)

        # Calculate metrics
        result.total_trades = len(result.trades)
        result.max_drawdown = max_dd * 100

        if result.wins > 0:
            result.avg_win = sum(t.pnl for t in result.trades if t.pnl >= 0) / result.wins
        if result.losses > 0:
            result.avg_loss = sum(t.pnl for t in result.trades if t.pnl < 0) / result.losses
        if result.total_trades > 0:
            result.win_rate = result.wins / result.total_trades * 100

        total_wins = sum(t.pnl for t in result.trades if t.pnl >= 0)
        total_losses = abs(sum(t.pnl for t in result.trades if t.pnl < 0))
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        if len(pnl_list) >= 2:
            avg_pnl = statistics.mean(pnl_list)
            std_pnl = statistics.stdev(pnl_list)
            result.sharpe_ratio = (avg_pnl / std_pnl) * (252 ** 0.5) if std_pnl > 0 else 0

        return result

    def _check_trade_exit(self, trade: VirtualTrade, candle) -> bool:
        high = candle["high"]
        low = candle["low"]

        if trade.side == "Buy":
            if low <= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                trade.pnl = (trade.stop_loss - trade.entry_price) * trade.qty
                trade.exit_reason = "sl"
                return True
            if self.partial_close_enabled and not trade.partial_closed:
                tp_dist = trade.take_profit - trade.entry_price
                trigger = trade.entry_price + tp_dist * self.partial_trigger
                if high >= trigger:
                    half = trade.qty * 0.5
                    trade.pnl += (trigger - trade.entry_price) * half
                    trade.qty -= half
                    trade.stop_loss = trade.entry_price
                    trade.partial_closed = True
            if high >= trade.take_profit:
                trade.exit_price = trade.take_profit
                trade.pnl += (trade.take_profit - trade.entry_price) * trade.qty
                trade.exit_reason = "tp"
                return True
        else:
            if high >= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                trade.pnl = (trade.entry_price - trade.stop_loss) * trade.qty
                trade.exit_reason = "sl"
                return True
            if self.partial_close_enabled and not trade.partial_closed:
                tp_dist = trade.entry_price - trade.take_profit
                trigger = trade.entry_price - tp_dist * self.partial_trigger
                if low <= trigger:
                    half = trade.qty * 0.5
                    trade.pnl += (trade.entry_price - trigger) * half
                    trade.qty -= half
                    trade.stop_loss = trade.entry_price
                    trade.partial_closed = True
            if low <= trade.take_profit:
                trade.exit_price = trade.take_profit
                trade.pnl += (trade.entry_price - trade.take_profit) * trade.qty
                trade.exit_reason = "tp"
                return True

        return False

    def _fetch_historical(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        all_dfs = []
        tf_minutes = int(timeframe)
        candles_needed = (days * 24 * 60) // tf_minutes
        batch_size = 1000
        end_time = None

        while candles_needed > 0:
            limit = min(batch_size, candles_needed)
            try:
                if end_time:
                    resp = self.client.session.get_kline(
                        category="linear", symbol=symbol,
                        interval=timeframe, limit=limit, end=str(end_time),
                    )
                    rows = resp["result"]["list"]
                    df_batch = pd.DataFrame(
                        rows,
                        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
                    )
                    for col in ["open", "high", "low", "close", "volume", "turnover"]:
                        df_batch[col] = df_batch[col].astype(float)
                    df_batch["timestamp"] = pd.to_datetime(df_batch["timestamp"].astype(int), unit="ms")
                    df_batch = df_batch.sort_values("timestamp").reset_index(drop=True)
                else:
                    df_batch = self.client.get_klines(symbol, timeframe, limit, "linear")

                all_dfs.append(df_batch)
                candles_needed -= len(df_batch)

                if len(df_batch) > 0:
                    end_time = int(df_batch["timestamp"].iloc[0].timestamp() * 1000)
                else:
                    break

                if len(df_batch) < limit:
                    break

            except Exception:
                logger.exception("Failed to fetch batch for %s", symbol)
                break

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"])
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        logger.info("Fetched %d historical candles for %s", len(combined), symbol)
        return combined


def format_result(r: BacktestResult) -> str:
    lines = [
        f"{'=' * 50}",
        f"  BACKTEST: {r.symbol} ({r.timeframe}м, {r.period_days} дней)",
        f"{'=' * 50}",
        f"  Всего сделок:      {r.total_trades}",
        f"  Побед / Поражений: {r.wins} / {r.losses}",
        f"  Win Rate:          {r.win_rate:.1f}%",
        f"  Общий PnL:         {r.total_pnl:+,.2f} USDT",
        f"  Profit Factor:     {r.profit_factor:.2f}",
        f"  Max Drawdown:      {r.max_drawdown:.1f}%",
        f"  Sharpe Ratio:      {r.sharpe_ratio:.2f}",
        f"  Средний выигрыш:   {r.avg_win:+,.2f} USDT",
        f"  Средний проигрыш:  {r.avg_loss:+,.2f} USDT",
        f"  Сигналов:          {r.signals_generated} (отфильтровано: {r.signals_filtered})",
        f"{'=' * 50}",
    ]

    if r.trades:
        lines.append("\n  Последние 10 сделок:")
        for t in r.trades[-10:]:
            emoji = "+" if t.pnl >= 0 else "-"
            exit_p = f"{t.exit_price:.2f}" if t.exit_price else "?"
            lines.append(
                f"    {emoji} {t.side} {t.entry_price:.2f} -> {exit_p} "
                f"| {t.pnl:+,.2f} USDT ({t.exit_reason})"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Stasik Backtester")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--days", type=int, default=7, help="Period in days")
    parser.add_argument("--timeframe", default="1", help="Candle timeframe (minutes)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    bt = Backtester(config)
    result = bt.run(args.symbol, args.days, args.timeframe)
    print(format_result(result))


if __name__ == "__main__":
    main()
