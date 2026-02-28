"""
Mass SMC Strategy Optimizer — Sampled approach.

Checks signals every SIGNAL_STEP candles (not every bar) but checks
trade exits every bar. This reduces computation ~4x while preserving
accuracy for ranking configs.

Usage:
    python3 -m src.backtest.optimizer [--days 30] [--top 20]
"""
import argparse
import copy
import itertools
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from src.exchange.client import BybitClient
from src.strategy.indicators import calculate_atr, calculate_ema
from src.strategy.signals import SMCGenerator, Signal, Trend

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]  # 3 top-volume symbols
SIGNAL_STEP = 12   # check signals every 12th candle (= 3h on 15m TF)
WINDOW = 100       # sliding window size


def fetch_data(config, symbol, tf, days):
    """Fetch historical klines from Bybit."""
    client = BybitClient(config)
    all_dfs = []
    candles_needed = (days * 24 * 60) // int(tf)
    end_time = None
    while candles_needed > 0:
        limit = min(1000, candles_needed)
        try:
            if end_time:
                resp = client.session.get_kline(
                    category="linear", symbol=symbol, interval=tf, limit=limit, end=str(end_time))
                rows = resp["result"]["list"]
                b = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","turnover"])
                for c in ["open","high","low","close","volume","turnover"]:
                    b[c] = b[c].astype(float)
                b["timestamp"] = pd.to_datetime(b["timestamp"].astype(int), unit="ms")
                b = b.sort_values("timestamp").reset_index(drop=True)
            else:
                b = client.get_klines(symbol, tf, limit, "linear")
            all_dfs.append(b)
            candles_needed -= len(b)
            if len(b) > 0:
                end_time = int(b["timestamp"].iloc[0].timestamp() * 1000)
            else:
                break
            if len(b) < limit:
                break
        except Exception:
            break
    if not all_dfs:
        return pd.DataFrame()
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def make_config(base, ms, fib_req, fib_prem, zam, sw, fvg=0.5, disp=0.3, vol=1.5, atr_sl=1.25, atr_tp=2.5):
    """Build config from base + param overrides."""
    c = copy.deepcopy(base)
    c["strategy"]["min_score"] = ms
    c["strategy"]["vol_threshold"] = vol
    c["smc"]["fib_required"] = fib_req
    c["smc"]["fib_premium_only"] = fib_prem
    c["smc"]["zigzag_atr_mult"] = zam
    c["smc"]["zigzag_enabled"] = True
    c["smc"]["sweep_lookback"] = sw
    c["smc"]["fvg_proximity_pct"] = fvg
    c["smc"]["displacement_body_pct"] = disp
    c["smc"]["cluster_enabled"] = True
    c["risk"]["atr_sl_multiplier"] = atr_sl
    c["risk"]["atr_tp_multiplier"] = atr_tp
    return c


def run_backtest(df, htf_df, symbol, config):
    """Run sampled backtest: signals every SIGNAL_STEP bars, exits every bar."""
    if len(df) < WINDOW + 50:
        return None

    smc = config.get("smc", {})
    risk = config["risk"]

    # Build SMC generator
    gen = SMCGenerator(config)

    # HTF structure
    from src.strategy.indicators.smc import detect_swing_points_zigzag
    swings = detect_swing_points_zigzag(
        htf_df, atr_period=14, atr_mult=smc.get("zigzag_atr_mult", 2.0))
    gen.update_structure(symbol, swings, htf_df)

    # HTF trend
    htf_trend = gen.get_htf_trend(htf_df)

    # Risk params
    atr_sl_mult = risk.get("atr_sl_multiplier", 1.25)
    atr_tp_mult = risk.get("atr_tp_multiplier", 2.5)
    comm = risk.get("commission_rate", 0.055) / 100
    sl_pct_fb = risk.get("stop_loss", 1.5) / 100
    tp_pct_fb = risk.get("take_profit", 4.0) / 100

    # Sim state
    balance = 10000.0
    peak = balance
    max_dd = 0.0
    pnls = []
    open_trade = None  # (side, entry, qty, sl, tp)
    wins = losses = signals = filtered = 0

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    for i in range(WINDOW, n):
        h = float(highs[i])
        l = float(lows[i])
        c = float(closes[i])

        # Check exit every bar
        if open_trade:
            side, entry, qty, sl, tp = open_trade
            pnl = None
            if side == "Buy":
                if l <= sl:
                    pnl = (sl - entry) * qty - entry * qty * comm * 2
                elif h >= tp:
                    pnl = (tp - entry) * qty - entry * qty * comm * 2
            else:
                if h >= sl:
                    pnl = (entry - sl) * qty - entry * qty * comm * 2
                elif l <= tp:
                    pnl = (entry - tp) * qty - entry * qty * comm * 2

            if pnl is not None:
                balance += pnl
                pnls.append(pnl)
                if pnl >= 0:
                    wins += 1
                else:
                    losses += 1
                peak = max(peak, balance)
                dd = (peak - balance) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
                open_trade = None
            continue  # still in trade if no exit

        # Check signals only every SIGNAL_STEP bars
        if (i - WINDOW) % SIGNAL_STEP != 0:
            continue

        start = max(0, i - WINDOW + 1)
        window = df.iloc[start:i + 1]
        window.attrs["symbol"] = symbol

        sig = gen.generate(window, symbol)
        if sig.signal == Signal.HOLD:
            continue

        signals += 1

        # HTF filter
        if htf_trend == Trend.BEARISH and sig.signal == Signal.BUY:
            filtered += 1
            continue
        if htf_trend == Trend.BULLISH and sig.signal == Signal.SELL:
            filtered += 1
            continue

        # ATR SL/TP
        try:
            atr = calculate_atr(window, 14)
            if atr > 0 and c > 0:
                sl_d = max(0.003, min(0.05, (atr * atr_sl_mult) / c))
                tp_d = max(0.005, min(0.05, (atr * atr_tp_mult) / c))
            else:
                sl_d, tp_d = sl_pct_fb, tp_pct_fb
        except Exception:
            sl_d, tp_d = sl_pct_fb, tp_pct_fb

        risk_amt = balance * (risk.get("risk_per_trade", 0.5) / 100)
        qty = (risk_amt / sl_d) / c if sl_d > 0 and c > 0 else 0
        if qty <= 0:
            continue

        side = "Buy" if sig.signal == Signal.BUY else "Sell"
        if side == "Buy":
            open_trade = (side, c, qty, c * (1 - sl_d), c * (1 + tp_d))
        else:
            open_trade = (side, c, qty, c * (1 + sl_d), c * (1 - tp_d))

    # Close remaining
    if open_trade:
        side, entry, qty, sl, tp = open_trade
        last = float(closes[-1])
        pnl = ((last - entry) if side == "Buy" else (entry - last)) * qty - entry * qty * comm * 2
        balance += pnl
        pnls.append(pnl)
        if pnl >= 0: wins += 1
        else: losses += 1

    tc = wins + losses
    wr = wins / tc * 100 if tc > 0 else 0
    tw = sum(p for p in pnls if p >= 0)
    tl = abs(sum(p for p in pnls if p < 0))
    pf = tw / tl if tl > 0 else (999 if tw > 0 else 0)
    sharpe = 0
    if len(pnls) >= 2:
        m = statistics.mean(pnls)
        s = statistics.stdev(pnls)
        sharpe = (m / s) * (252**0.5) if s > 0 else 0

    return {
        "trades": tc, "wins": wins, "losses": losses,
        "pnl": sum(pnls), "win_rate": wr, "pf": pf, "sharpe": sharpe,
        "max_dd": max_dd * 100,
        "exp": sum(pnls) / tc if tc > 0 else 0,
        "avg_win": tw / wins if wins > 0 else 0,
        "avg_loss": -tl / losses if losses > 0 else 0,
        "signals": signals, "filtered": filtered,
    }


@dataclass
class AggResult:
    cid: str
    params: dict
    trades: int = 0
    pnl: float = 0.0
    wr: float = 0.0
    pf: float = 0.0
    sharpe: float = 0.0
    dd: float = 0.0
    exp: float = 0.0
    score: float = 0.0
    per_sym: dict = field(default_factory=dict)


def aggregate(results):
    """results: {cid: {sym: result, '_params': {...}}}"""
    agg = []
    for cid, data in results.items():
        params = data.get("_params", {})
        active = {s: r for s, r in data.items() if s != "_params" and r and r["trades"] > 0}
        if not active:
            continue
        a = AggResult(
            cid=cid, params=params,
            trades=sum(r["trades"] for r in active.values()),
            pnl=sum(r["pnl"] for r in active.values()),
            wr=statistics.mean([r["win_rate"] for r in active.values()]),
            pf=statistics.mean([min(r["pf"], 10) for r in active.values()]),
            sharpe=statistics.mean([r["sharpe"] for r in active.values()]),
            dd=max(r["max_dd"] for r in active.values()),
            exp=statistics.mean([r["exp"] for r in active.values()]),
            per_sym=active,
        )
        # Composite score
        wr_f = a.wr / 100
        pf_f = min(a.pf, 5) / 5
        dd_f = max(0, 1 - a.dd / 30)
        t_f = min(a.trades / 20, 1)
        pnl_f = max(0, min(1, (a.pnl + 2000) / 4000))
        a.score = (pnl_f * 0.30 + wr_f * 0.25 + pf_f * 0.20 + dd_f * 0.15 + t_f * 0.10) * 100
        agg.append(a)
    agg.sort(key=lambda x: x.score, reverse=True)
    return agg


def print_top(label, agg_list, n=5):
    print(f"\n  Топ-{min(n, len(agg_list))} {label}:", flush=True)
    for i, a in enumerate(agg_list[:n]):
        p = a.params
        fib = "P" if p.get("fp") else ("R" if p.get("fr") else "N")
        print(f"    {i+1}. Score={a.score:.1f} PnL={a.pnl:+,.0f} WR={a.wr:.1f}% "
              f"PF={min(a.pf,99):.2f} DD={a.dd:.1f}% T={a.trades} | "
              f"ms={p.get('ms')} fib={fib} z={p.get('z')} sw={p.get('sw')} "
              f"fvg={p.get('fvg',0.5)} d={p.get('d',0.3)} v={p.get('v',1.5)} "
              f"sl={p.get('sl',1.25)} tp={p.get('tp',2.5)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    t0 = time.time()

    cfg_path = Path(__file__).resolve().parent.parent.parent / "config" / "fiba.yaml"
    with open(cfg_path) as f:
        base = yaml.safe_load(f)

    cache_dir = Path("/root/stasik/data/optim_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("  STASIK SMC OPTIMIZER (sampled)", flush=True)
    print(f"  {args.days}д | {', '.join(args.symbols)} | signal_step={SIGNAL_STEP}", flush=True)
    print("=" * 70, flush=True)

    # ── Load data ──
    print("\n[1/4] Данные...", flush=True)
    data = {}
    for sym in args.symbols:
        dp = cache_dir / f"{sym}_15_{args.days}d.csv"
        hp = cache_dir / f"{sym}_htf60.csv"
        if dp.exists():
            df = pd.read_csv(str(dp), parse_dates=["timestamp"])
        else:
            print(f"  {sym} 15м: скачиваю...", end=" ", flush=True)
            df = fetch_data(base, sym, "15", args.days)
            if len(df) == 0:
                print("FAIL", flush=True); continue
            df.to_csv(str(dp), index=False)
            print(f"OK ({len(df)})", flush=True)

        if hp.exists():
            htf = pd.read_csv(str(hp), parse_dates=["timestamp"])
        else:
            print(f"  {sym} HTF: скачиваю...", end=" ", flush=True)
            client = BybitClient(base)
            htf = client.get_klines(symbol=sym, interval="60", limit=1000, category="linear")
            if len(htf) == 0:
                print("FAIL", flush=True); continue
            htf.to_csv(str(hp), index=False)
            print(f"OK ({len(htf)})", flush=True)

        data[sym] = (df, htf)
        print(f"  {sym}: {len(df)} свечей", flush=True)

    # ── Layer 1: Macro ──
    L1 = list(itertools.product(
        [2, 3, 4],                                          # min_score
        [(True, True), (True, False), (False, False)],      # fib_req, fib_prem
        [1.5, 2.0, 3.0],                                   # zigzag_atr_mult
        [20, 30],                                           # sweep_lookback
    ))
    total = len(L1) * len(data)
    print(f"\n[2/4] Layer 1: {len(L1)} конфигов × {len(data)} символов = {total} бэктестов", flush=True)

    l1_res = {}
    done = 0
    for (ms, (fr, fp), z, sw) in L1:
        cid = f"ms{ms}_fr{fr}_fp{fp}_z{z}_sw{sw}"
        l1_res[cid] = {"_params": {"ms": ms, "fr": fr, "fp": fp, "z": z, "sw": sw}}
        cfg = make_config(base, ms, fr, fp, z, sw)
        for sym in data:
            df, htf = data[sym]
            r = run_backtest(df, htf, sym, cfg)
            l1_res[cid][sym] = r
            done += 1
            if done % 20 == 0 or done == total:
                elapsed = time.time() - t0
                eta = (elapsed / done) * (total - done) if done > 0 else 0
                print(f"  L1: {done}/{total} ({done*100//total}%) ETA {eta:.0f}s", flush=True)

    l1_agg = aggregate(l1_res)
    print_top("Layer 1", l1_agg)

    # ── Layer 2: Micro on top 5 ──
    L2 = list(itertools.product(
        [0.3, 0.5, 1.0],    # fvg_proximity
        [0.2, 0.3, 0.5],    # displacement_body
        [1.2, 1.5, 2.0],    # vol_threshold
    ))
    top5 = l1_agg[:5]
    total2 = len(top5) * len(L2) * len(data)
    print(f"\n[3/4] Layer 2: {len(top5)}×{len(L2)}×{len(data)} = {total2} бэктестов", flush=True)

    l2_res = {}
    done = 0
    t2 = time.time()
    for top in top5:
        p = top.params
        for (fvg, d, v) in L2:
            cid = f"ms{p['ms']}_fr{p['fr']}_fp{p['fp']}_z{p['z']}_sw{p['sw']}_fvg{fvg}_d{d}_v{v}"
            l2_res[cid] = {"_params": {**p, "fvg": fvg, "d": d, "v": v}}
            cfg = make_config(base, p["ms"], p["fr"], p["fp"], p["z"], p["sw"], fvg, d, v)
            for sym in data:
                df, htf = data[sym]
                r = run_backtest(df, htf, sym, cfg)
                l2_res[cid][sym] = r
                done += 1
                if done % 20 == 0 or done == total2:
                    el = time.time() - t2
                    eta = (el / done) * (total2 - done) if done > 0 else 0
                    print(f"  L2: {done}/{total2} ({done*100//total2}%) ETA {eta:.0f}s", flush=True)

    l2_agg = aggregate(l2_res)
    print_top("Layer 2", l2_agg)

    # ── Layer 3: Risk on top 5 ──
    L3 = list(itertools.product(
        [0.75, 1.0, 1.25, 1.5, 2.0],   # atr_sl
        [1.5, 2.0, 2.5, 3.0, 4.0],     # atr_tp
    ))
    best = (l2_agg if l2_agg else l1_agg)[:5]
    total3 = len(best) * len(L3) * len(data)
    print(f"\n[4/4] Layer 3: {len(best)}×{len(L3)}×{len(data)} = {total3} бэктестов", flush=True)

    l3_res = {}
    done = 0
    t3 = time.time()
    for top in best:
        p = top.params
        for (sl, tp) in L3:
            cid = f"ms{p['ms']}_z{p['z']}_sw{p['sw']}_fvg{p.get('fvg',0.5)}_d{p.get('d',0.3)}_v{p.get('v',1.5)}_sl{sl}_tp{tp}"
            l3_res[cid] = {"_params": {**p, "sl": sl, "tp": tp}}
            cfg = make_config(base, p["ms"], p["fr"], p["fp"], p["z"], p["sw"],
                            p.get("fvg",0.5), p.get("d",0.3), p.get("v",1.5), sl, tp)
            for sym in data:
                df, htf = data[sym]
                r = run_backtest(df, htf, sym, cfg)
                l3_res[cid][sym] = r
                done += 1
                if done % 20 == 0 or done == total3:
                    el = time.time() - t3
                    eta = (el / done) * (total3 - done) if done > 0 else 0
                    print(f"  L3: {done}/{total3} ({done*100//total3}%) ETA {eta:.0f}s", flush=True)

    l3_agg = aggregate(l3_res)
    print_top("Layer 3", l3_agg)

    # ── Final ──
    all_res = {**l1_res, **l2_res, **l3_res}
    final = aggregate(all_res)

    elapsed = time.time() - t0
    n_show = min(args.top, len(final))

    print(flush=True)
    print("=" * 70, flush=True)
    print(f"  ИТОГИ ({elapsed:.0f} сек, {len(all_res)} конфигураций)", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    print(f"  {'#':>3} {'Score':>6} {'PnL':>10} {'WR%':>6} {'PF':>5} {'Shrp':>6} {'DD%':>5} {'T':>4} | Параметры", flush=True)
    print(f"  {'─'*3} {'─'*6} {'─'*10} {'─'*6} {'─'*5} {'─'*6} {'─'*5} {'─'*4} + {'─'*55}", flush=True)

    for i, a in enumerate(final[:n_show]):
        p = a.params
        fib = "P" if p.get("fp") else ("R" if p.get("fr") else "N")
        pf_s = f"{a.pf:.2f}" if a.pf < 99 else "INF"
        ps = (f"ms={p.get('ms')} fib={fib} z={p.get('z')} sw={p.get('sw')} "
              f"fvg={p.get('fvg',0.5)} d={p.get('d',0.3)} v={p.get('v',1.5)} "
              f"sl={p.get('sl',1.25)} tp={p.get('tp',2.5)}")
        print(f"  {i+1:>3} {a.score:>6.1f} {a.pnl:>+10,.0f} {a.wr:>6.1f} {pf_s:>5} "
              f"{a.sharpe:>6.2f} {a.dd:>5.1f} {a.trades:>4} | {ps}", flush=True)

    # Per-symbol for top 3
    for i, a in enumerate(final[:3]):
        print(f"\n  #{i+1} по символам:", flush=True)
        for sym, r in a.per_sym.items():
            pfs = f"{r['pf']:.2f}" if r['pf'] < 99 else "INF"
            print(f"    {sym:>10}: PnL={r['pnl']:>+8,.0f} WR={r['win_rate']:>5.1f}% PF={pfs:>5} "
                  f"DD={r['max_dd']:>5.1f}% T={r['trades']:>3}", flush=True)

    # Save JSON
    out = Path("/root/stasik/data/optimization_results.json")
    jdata = {
        "timestamp": datetime.now().isoformat(),
        "days": args.days, "symbols": list(data.keys()),
        "elapsed_sec": round(elapsed, 1), "total_configs": len(final),
        "top": [{
            "rank": i+1, "cid": a.cid, "params": a.params,
            "score": round(a.score,2), "pnl": round(a.pnl,2),
            "wr": round(a.wr,2), "pf": round(min(a.pf,999),2),
            "sharpe": round(a.sharpe,2), "dd": round(a.dd,2),
            "trades": a.trades, "exp": round(a.exp,2),
            "per_sym": {s: {k: round(v,2) if isinstance(v,float) else v for k,v in r.items()} for s,r in a.per_sym.items()},
        } for i, a in enumerate(final[:50])],
    }
    with open(out, "w") as f:
        json.dump(jdata, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON: {out}", flush=True)

    # Best config
    if final:
        b = final[0]
        p = b.params
        print(flush=True)
        print("=" * 70, flush=True)
        print("  ЛУЧШАЯ КОНФИГУРАЦИЯ", flush=True)
        print("=" * 70, flush=True)
        print(f"  min_score:            {p.get('ms')}", flush=True)
        print(f"  fib_required:         {p.get('fr')}", flush=True)
        print(f"  fib_premium_only:     {p.get('fp')}", flush=True)
        print(f"  zigzag_atr_mult:      {p.get('z')}", flush=True)
        print(f"  sweep_lookback:       {p.get('sw')}", flush=True)
        print(f"  fvg_proximity_pct:    {p.get('fvg', 0.5)}", flush=True)
        print(f"  displacement_body_pct:{p.get('d', 0.3)}", flush=True)
        print(f"  vol_threshold:        {p.get('v', 1.5)}", flush=True)
        print(f"  atr_sl_multiplier:    {p.get('sl', 1.25)}", flush=True)
        print(f"  atr_tp_multiplier:    {p.get('tp', 2.5)}", flush=True)
        print(f"  ─────────────────────", flush=True)
        print(f"  PnL:          {b.pnl:>+10,.2f} USDT", flush=True)
        print(f"  Win Rate:     {b.wr:>10.1f}%", flush=True)
        print(f"  Profit Factor:{b.pf:>10.2f}", flush=True)
        print(f"  Sharpe:       {b.sharpe:>10.2f}", flush=True)
        print(f"  Max DD:       {b.dd:>10.1f}%", flush=True)
        print(f"  Trades:       {b.trades:>10}", flush=True)
        print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
