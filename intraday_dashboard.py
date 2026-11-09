#!/usr/bin/env python3
"""
Intraday Desk - NSE intraday trading dashboard
==============================================
A local web page built on TradingView's Lightweight Charts library, fed with live NSE
intraday candles (Yahoo Finance) by this small Python server.

What you get in the browser:
  * Candlestick chart with VWAP, EMA 9/21, Supertrend(10,3), volume, RSI, and today's
    key levels (opening range, previous-day high/low/close, CPR, pivots R1/R2/S1/S2)
  * Intraday bias score (bullish / bearish) from VWAP, EMAs, Supertrend, RSI, CPR, OR
  * Live signals from three intraday strategies, with entry, stop-loss and target:
      1. Opening Range Breakout (15-min OR + VWAP + volume)
      2. VWAP-EMA pullback (trend continuation)
      3. Supertrend flip + VWAP filter
  * Intraday backtest of each strategy on this stock (up to 60 days of 5-min data):
    win % / loss %, profit factor, net P&L after Zerodha-style intraday charges and
    slippage, max drawdown, green-day %, equity curve and every trade on the chart
  * Watchlist scanner, position-size calculator, charges & breakeven calculator
  * "Open in TradingView" button for the full TradingView chart of the same symbol

Why not the embedded TradingView widget? TradingView's free embeddable widgets refuse NSE
symbols and give BSE only daily candles, so they cannot show Indian intraday charts.

Setup:  pip install yfinance pandas numpy
Run:    python intraday_dashboard.py            -> opens http://127.0.0.1:8050
        python intraday_dashboard.py --port 9000 --no-browser

Notes: Yahoo data can lag the exchange slightly; 1-min history covers 7 days, 2/5/15-min
history covers 60 days. Backtests assume next-candle-open entries, stop-loss checked before
target inside a candle, and square-off at 3:15 PM. Educational tool, not investment advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    raise SystemExit("Missing dependency. Run:  pip install yfinance pandas numpy")

IST = "Asia/Kolkata"
OPEN_MIN, CLOSE_MIN = 9 * 60 + 15, 15 * 60 + 30
OR_MINUTES = 15
OR_END = OPEN_MIN + OR_MINUTES          # 09:30
LAST_ENTRY = 14 * 60 + 30               # no new trades after 2:30 PM
SQ_OFF = 15 * 60 + 15                   # square-off 3:15 PM
PERIOD = {"1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d"}
BAR_MIN = {"1m": 1, "2m": 2, "5m": 5, "15m": 15}

ALIASES = {"NIFTY": "^NSEI", "NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK", "NIFTYBANK": "^NSEBANK",
           "FINNIFTY": "NIFTY_FIN_SERVICE.NS", "SENSEX": "^BSESN"}
TV_SYMBOL = {"^NSEI": "NSE:NIFTY", "^NSEBANK": "NSE:BANKNIFTY", "NIFTY_FIN_SERVICE.NS": "NSE:CNXFINANCE",
             "^BSESN": "BSE:SENSEX"}
WATCHLIST = [
    "NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK",
    "BANKBARODA", "PNB", "BAJFINANCE", "SHRIRAMFIN", "JIOFIN", "INFY", "TCS", "HCLTECH", "WIPRO", "LT", "ITC",
    "BHARTIARTL", "HINDUNILVR", "MARUTI", "M&M", "BAJAJ-AUTO", "TMPV", "SUNPHARMA", "TITAN", "TRENT", "ETERNAL",
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA", "ONGC", "NTPC", "POWERGRID", "TATAPOWER",
    "ADANIENT", "ADANIPORTS", "ULTRACEMCO", "ASIANPAINT", "DLF", "BEL",
]

STRATS = {
    "orb": dict(name="Opening Range Breakout", short="ORB",
                desc="After 9:30, trade the first close beyond the 15-min opening range, only on the "
                     "VWAP side and with volume >= 1.2x average. Stop: other side of the range (max 2 ATR). "
                     "Target 2R. Exit early if price closes back through VWAP. One long and one short a day.",
                max_day=2, one_side=True, rr=2.0),
    "pullback": dict(name="VWAP-EMA Pullback", short="Pullback",
                     desc="Trend day continuation: EMA9 above EMA21, price above VWAP, Supertrend up, then a "
                          "candle dips to EMA9 and closes green above it with RSI 50-70 (mirror for shorts). "
                          "Stop below the candle low. Target 2R. Exit if price closes beyond EMA21. One trade a day.",
                     max_day=1, one_side=False, rr=2.0),
    "supertrend": dict(name="Supertrend + VWAP", short="Supertrend",
                       desc="Enter within 3 candles of a Supertrend(10,3) flip, once per leg, only if price is on the "
                            "same side of VWAP and EMA9/EMA21 agree. Stop at the Supertrend line. Target 2R. "
                            "Exit on the opposite flip.",
                       max_day=3, one_side=False, rr=2.0),
}


# ------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------
def resolve(sym: str) -> str:
    s = sym.strip().upper()
    if s in ALIASES:
        return ALIASES[s]
    if s.startswith("^") or s.endswith(".NS") or s.endswith(".BO"):
        return s
    return s + ".NS"


def display_symbol(t: str) -> str:
    rev = {v: k for k, v in ALIASES.items() if k in ("NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX")}
    return rev.get(t, t.replace(".NS", "").replace(".BO", ""))


def tv_symbol(t: str) -> str:
    if t in TV_SYMBOL:
        return TV_SYMBOL[t]
    ex = "BSE" if t.endswith(".BO") else "NSE"
    return f"{ex}:" + t.replace(".NS", "").replace(".BO", "").replace("&", "_").replace("-", "_")


def now_ist() -> dt.datetime:
    return pd.Timestamp.now(tz=IST).tz_localize(None).to_pydatetime()


def market_open(now: dt.datetime) -> bool:
    m = now.hour * 60 + now.minute
    return now.weekday() < 5 and OPEN_MIN <= m < CLOSE_MIN


def num(x, nd=2):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(x) or math.isinf(x)) else round(x, nd)


def arr(s, nd=2):
    return [num(v, nd) for v in (s.to_numpy() if hasattr(s, "to_numpy") else s)]


def charges(buy_v: float, sell_v: float, slip_pct: float = 0.0) -> dict:
    """NSE equity intraday, Zerodha-style: brokerage 0.03% or Rs 20 per order (lower),
    STT 0.025% sell, exchange txn 0.00307%, SEBI Rs 10/crore, stamp 0.003% buy, GST 18%."""
    turnover = buy_v + sell_v
    brokerage = min(20.0, 0.0003 * buy_v) + min(20.0, 0.0003 * sell_v)
    stt = 0.00025 * sell_v
    txn = 0.0000307 * turnover
    sebi = turnover * 10 / 1e7
    stamp = 0.00003 * buy_v
    gst = 0.18 * (brokerage + txn + sebi)
    slip = slip_pct / 100 * turnover
    total = brokerage + stt + txn + sebi + stamp + gst + slip
    return dict(brokerage=brokerage, stt=stt, txn=txn, sebi=sebi, stamp=stamp, gst=gst, slippage=slip, total=total)


# ------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------
_cache: dict = {}
_lock = threading.Lock()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    idx = pd.to_datetime(df.index)
    idx = idx.tz_convert(IST).tz_localize(None) if idx.tz is not None else idx
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index()
    m = df.index.hour * 60 + df.index.minute
    df = df[(m >= OPEN_MIN) & (m < CLOSE_MIN)]
    df["Volume"] = df["Volume"].fillna(0).astype(float)
    return df


def fetch(t: str, interval: str, ttl: int = 25) -> pd.DataFrame:
    key = (t, interval)
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    raw = yf.Ticker(t).history(period=PERIOD[interval], interval=interval, auto_adjust=False, prepost=False)
    if raw is None or raw.empty:
        raise ValueError(f"No intraday data for {display_symbol(t)}. Check the NSE symbol (e.g. RELIANCE, SBIN, NIFTY).")
    df = clean(raw)
    if len(df) < 60:
        raise ValueError(f"Not enough intraday data for {display_symbol(t)}.")
    with _lock:
        _cache[key] = (time.time(), df)
    return df


# ------------------------------------------------------------------------------------------
# Indicators
# ------------------------------------------------------------------------------------------
def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def _supertrend(h, l, c, tr, n=10, mult=3.0):
    atr = _rma(tr, n).to_numpy()
    hl2 = ((h + l) / 2).to_numpy()
    cc = c.to_numpy()
    ub, lb = hl2 + mult * atr, hl2 - mult * atr
    fub, flb = ub.copy(), lb.copy()
    d = np.ones(len(cc))
    line = np.full(len(cc), np.nan)
    line[0] = flb[0]
    for i in range(1, len(cc)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or cc[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or cc[i - 1] < flb[i - 1]) else flb[i - 1]
        if d[i - 1] < 0 and cc[i] > fub[i - 1]:
            d[i] = 1
        elif d[i - 1] > 0 and cc[i] < flb[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
        line[i] = flb[i] if d[i] > 0 else fub[i]
    return d, line


def compute(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = d.index.date
    d["tod"] = d.index.hour * 60 + d.index.minute
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    g = d.groupby("date")

    tp = (h + l + c) / 3
    cumv = v.groupby(d["date"]).cumsum()
    vwap = (tp * v).groupby(d["date"]).cumsum() / cumv.replace(0, np.nan)
    tp_avg = tp.groupby(d["date"]).cumsum() / (g.cumcount() + 1)       # indices have no volume
    d["VWAP"] = vwap.fillna(tp_avg)

    d["EMA9"] = c.ewm(span=9, adjust=False).mean()
    d["EMA21"] = c.ewm(span=21, adjust=False).mean()
    delta = c.diff()
    gain, loss = _rma(delta.clip(lower=0), 14), _rma((-delta).clip(lower=0), 14)
    d["RSI"] = 100 * gain / (gain + loss).replace(0, np.nan)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    d["ATR"] = _rma(tr, 14)
    d["ST_DIR"], d["ST_LINE"] = _supertrend(h, l, c, tr, 10, 3.0)
    d["VOL_RATIO"] = v / v.rolling(20).mean().replace(0, np.nan)

    in_or = d["tod"] < OR_END
    orh = h.where(in_or).groupby(d["date"]).transform("max")
    orl = l.where(in_or).groupby(d["date"]).transform("min")
    d["OR_H"] = orh.where(d["tod"] >= OR_END)
    d["OR_L"] = orl.where(d["tod"] >= OR_END)

    daily = d.groupby("date").agg(H=("High", "max"), L=("Low", "min"), C=("Close", "last"))
    p = daily.shift(1)
    P = (p.H + p.L + p.C) / 3
    BC = (p.H + p.L) / 2
    TC = 2 * P - BC
    lv = pd.DataFrame(dict(PDH=p.H, PDL=p.L, PDC=p.C, P=P, TC=np.maximum(TC, BC), BC=np.minimum(TC, BC),
                           R1=2 * P - p.L, S1=2 * P - p.H, R2=P + (p.H - p.L), S2=P - (p.H - p.L)),
                      index=daily.index)
    d = d.join(lv, on="date")
    return d


def bias_series(d: pd.DataFrame) -> pd.Series:
    """Vectorised version of bias(): the same -100..+100 score on every candle."""
    c = d["Close"]
    with np.errstate(invalid="ignore"):
        s = 25 * np.sign(c - d["VWAP"]) + 20 * np.sign(d["EMA9"] - d["EMA21"]) + 20 * d["ST_DIR"]
        s = s + 15 * ((d["RSI"] > 55).astype(int) - (d["RSI"] < 45).astype(int))
        s = s + 10 * ((c > d["TC"]).astype(int) - (c < d["BC"]).astype(int))
        s = s + 10 * ((c > d["OR_H"]).astype(int) - (c < d["OR_L"]).astype(int))
    return s.fillna(0)


NSE_HOLIDAYS = {dt.date(2026, 11, 10), dt.date(2026, 11, 24), dt.date(2026, 12, 25)}
LEVEL_KEYS = ("PDH", "PDL", "PDC", "P", "TC", "BC", "R1", "S1", "R2", "S2", "OR_H", "OR_L")


def next_session(after: dt.date) -> dt.date:
    d = after + dt.timedelta(days=1)
    while d.weekday() >= 5 or d in NSE_HOLIDAYS:
        d += dt.timedelta(days=1)
    return d


def session_window(now: dt.datetime) -> bool:
    """True on NSE trading days from 09:15 to 15:40 IST (10 min past the close, to catch the last candles)."""
    m = now.hour * 60 + now.minute
    return now.weekday() < 5 and now.date() not in NSE_HOLIDAYS and OPEN_MIN <= m <= CLOSE_MIN + 10


def next_levels(d: pd.DataFrame) -> dict:
    """Pivot/CPR levels for the session after the last one in the data."""
    last = d[d["date"] == d["date"].iloc[-1]]
    H, L, C = last["High"].max(), last["Low"].min(), last["Close"].iloc[-1]
    P = (H + L + C) / 3
    BC = (H + L) / 2
    TC = 2 * P - BC
    return {k: num(v) for k, v in dict(PDH=H, PDL=L, PDC=C, P=P, TC=max(TC, BC), BC=min(TC, BC), R1=2 * P - L,
                                         S1=2 * P - H, R2=P + (H - L), S2=P - (H - L), OR_H=None, OR_L=None).items()}


# ------------------------------------------------------------------------------------------
# Strategies & simulator
# ------------------------------------------------------------------------------------------
def signals(d: pd.DataFrame, key: str):
    c, o, h, l, atr, tod = d["Close"], d["Open"], d["High"], d["Low"], d["ATR"], d["tod"]
    if key == "orb":
        t_ok = (tod >= OR_END) & (tod <= LAST_ENTRY)
        vol_ok = d["VOL_RATIO"].fillna(9) >= 1.2
        L = t_ok & (c > d["OR_H"]) & (c.shift(1) <= d["OR_H"]) & (c > d["VWAP"]) & vol_ok
        S = t_ok & (c < d["OR_L"]) & (c.shift(1) >= d["OR_L"]) & (c < d["VWAP"]) & vol_ok
        RL = np.maximum(c - np.maximum(d["OR_L"], c - 2 * atr), 0.5 * atr)
        RS = np.maximum(np.minimum(d["OR_H"], c + 2 * atr) - c, 0.5 * atr)
        XL, XS = c < d["VWAP"], c > d["VWAP"]
    elif key == "pullback":
        t_ok = (tod >= OPEN_MIN + 30) & (tod <= LAST_ENTRY)
        L = t_ok & (d["EMA9"] > d["EMA21"]) & (c > d["VWAP"]) & (d["ST_DIR"] > 0) & (l <= d["EMA9"]) & \
            (c > d["EMA9"]) & (c > o) & d["RSI"].between(50, 70)
        S = t_ok & (d["EMA9"] < d["EMA21"]) & (c < d["VWAP"]) & (d["ST_DIR"] < 0) & (h >= d["EMA9"]) & \
            (c < d["EMA9"]) & (c < o) & d["RSI"].between(30, 50)
        RL = np.maximum(c - l, 0.5 * atr) + 0.1 * atr
        RS = np.maximum(h - c, 0.5 * atr) + 0.1 * atr
        XL, XS = c < d["EMA21"], c > d["EMA21"]
    else:  # supertrend
        t_ok = (tod >= OR_END) & (tod <= LAST_ENTRY)
        flip = d["ST_DIR"] != d["ST_DIR"].shift(1)
        leg = flip.cumsum()                                    # one id per Supertrend leg
        fresh = d.groupby(leg).cumcount() <= 3                 # within 3 candles of the flip
        rl, rs = c - d["ST_LINE"], d["ST_LINE"] - c
        qL = t_ok & fresh & (d["ST_DIR"] > 0) & (c > d["VWAP"]) & (d["EMA9"] > d["EMA21"]) & (rl <= 3 * atr)
        qS = t_ok & fresh & (d["ST_DIR"] < 0) & (c < d["VWAP"]) & (d["EMA9"] < d["EMA21"]) & (rs <= 3 * atr)
        L = qL & (qL.astype(int).groupby(leg).cumsum() == 1)  # first qualifying candle of the leg
        S = qS & (qS.astype(int).groupby(leg).cumsum() == 1)
        RL, RS = np.maximum(rl, 0.5 * atr), np.maximum(rs, 0.5 * atr)
        XL, XS = d["ST_DIR"] < 0, d["ST_DIR"] > 0
    f = lambda s: s.fillna(False).to_numpy(dtype=bool) if s.dtype != float else s.to_numpy()
    return (f(L), f(S), np.asarray(RL, dtype=float), np.asarray(RS, dtype=float), f(XL), f(XS))


def simulate(d: pd.DataFrame, key: str, capital: float, slip: float):
    sp = STRATS[key]
    L, S, RL, RS, XL, XS = signals(d, key)
    B = bias_series(d).to_numpy()
    o, h, l, c = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    tod, dates, idx = d["tod"].to_numpy(), d["date"].to_numpy(), d.index
    ts = idx.values.astype("datetime64[s]").astype(np.int64)
    n = len(d)
    trades, pos, pending, pend_exit, cur = [], 0, None, False, None
    day_n, day_sides = {}, set()

    def close(i, px, reason):
        nonlocal pos, cur, pend_exit
        t, side, qty = cur, cur["side"], cur["qty"]
        buy_v, sell_v = (t["entry"] * qty, px * qty) if side > 0 else (px * qty, t["entry"] * qty)
        ch = charges(buy_v, sell_v, slip)["total"]
        gross = side * (px - t["entry"]) * qty
        t.update(exit=round(float(px), 2), exit_time=idx[i].strftime("%d %b %H:%M"), exit_ts=int(ts[i]),
                 reason=reason, gross=round(gross, 2), charges=round(ch, 2), net=round(gross - ch, 2),
                 r=round(side * (px - t["entry"]) / t["risk"], 2), status="closed")
        trades.append(t)
        pos, cur, pend_exit = 0, None, False

    for i in range(1, n):
        if dates[i] != dates[i - 1]:
            if pos:
                close(i - 1, c[i - 1], "End of day")
            pending = None
        if pending is not None and not pos:
            side, risk, si = pending
            pending = None
            e = float(o[i])
            cur = dict(side=side, date=str(dates[i]), entry=round(e, 2), entry_time=idx[i].strftime("%d %b %H:%M"),
                       entry_ts=int(ts[i]), risk=float(risk), stop=round(e - side * risk, 2),
                       target=round(e + side * sp["rr"] * risk, 2), qty=max(1, int(capital // e)), bias=int(B[si]))
            pos = side
        if pos:
            st, tg = cur["stop"], cur["target"]
            if tod[i] >= SQ_OFF:
                close(i, o[i], "Square-off 3:15")
            elif pend_exit:
                close(i, o[i], "Exit signal")
            elif pos > 0 and l[i] <= st:
                close(i, min(o[i], st), "Stop-loss")
            elif pos > 0 and h[i] >= tg:
                close(i, max(o[i], tg), "Target")
            elif pos < 0 and h[i] >= st:
                close(i, max(o[i], st), "Stop-loss")
            elif pos < 0 and l[i] <= tg:
                close(i, min(o[i], tg), "Target")
            elif (pos > 0 and XL[i]) or (pos < 0 and XS[i]):
                pend_exit = True
        if not pos and pending is None:
            k = dates[i]
            if day_n.get(k, 0) < sp["max_day"]:
                if L[i] and not (sp["one_side"] and (k, 1) in day_sides) and RL[i] > 0:
                    pending = (1, RL[i], i)
                elif S[i] and not (sp["one_side"] and (k, -1) in day_sides) and RS[i] > 0:
                    pending = (-1, RS[i], i)
                if pending:
                    day_n[k] = day_n.get(k, 0) + 1
                    day_sides.add((k, pending[0]))
    signal = None
    if pending is not None:
        side, risk, _ = pending
        ref = float(c[-1])
        signal = dict(side=side, entry=round(ref, 2), stop=round(ref - side * risk, 2),
                      target=round(ref + side * sp["rr"] * risk, 2), time=idx[-1].strftime("%d %b %H:%M"))
    return trades, cur, signal


def trade_stats(trades: list, capital: float) -> dict:
    if not trades:
        return dict(trades=0, wins=0, losses=0, win_pct=0, loss_pct=0, net=0, gross=0, charges=0, ret_pct=0,
                    pf=None, avg_win=0, avg_loss=0, expectancy=0, avg_r=0, max_dd=0, max_dd_pct=0, days=0,
                    green_days=0, best_day=0, worst_day=0, long_n=0, long_win=None, short_n=0, short_win=None,
                    exits={})
    net = np.array([t["net"] for t in trades])
    wins, losses = net[net > 0], net[net <= 0]
    cum = np.cumsum(net)
    dd = float((cum - np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]).min())
    by_day = pd.Series(net, index=[t["date"] for t in trades]).groupby(level=0).sum()
    longs = [t for t in trades if t["side"] > 0]
    shorts = [t for t in trades if t["side"] < 0]
    wr = lambda ts_: round(sum(t["net"] > 0 for t in ts_) / len(ts_) * 100, 1) if ts_ else None
    return dict(
        trades=len(trades), wins=int(len(wins)), losses=int(len(losses)),
        win_pct=round(len(wins) / len(trades) * 100, 1), loss_pct=round(len(losses) / len(trades) * 100, 1),
        net=round(float(net.sum()), 2), gross=round(sum(t["gross"] for t in trades), 2),
        charges=round(sum(t["charges"] for t in trades), 2), ret_pct=round(float(net.sum()) / capital * 100, 2),
        pf=round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
        avg_win=round(float(wins.mean()), 2) if len(wins) else 0, avg_loss=round(float(losses.mean()), 2) if len(losses) else 0,
        expectancy=round(float(net.mean()), 2), avg_r=round(float(np.mean([t["r"] for t in trades])), 2),
        max_dd=round(dd, 2), max_dd_pct=round(dd / capital * 100, 2), days=int(len(by_day)),
        green_days=round(float((by_day > 0).mean() * 100), 1), best_day=round(float(by_day.max()), 2),
        worst_day=round(float(by_day.min()), 2), long_n=len(longs), long_win=wr(longs), short_n=len(shorts),
        short_win=wr(shorts), exits=pd.Series([t["reason"] for t in trades]).value_counts().to_dict(),
    )


def bias(row: pd.Series) -> dict:
    items, score = [], 0
    c = row["Close"]

    def add(ok, pts, good, bad, neutral=None):
        nonlocal score
        if ok is None:
            items.append(dict(state="neutral", text=neutral))
            return
        score += pts if ok else -pts
        items.append(dict(state="bull" if ok else "bear", text=good if ok else bad))

    add(c > row["VWAP"], 25, f"Above VWAP {row['VWAP']:.2f}", f"Below VWAP {row['VWAP']:.2f}")
    add(row["EMA9"] > row["EMA21"], 20, "EMA 9 above EMA 21", "EMA 9 below EMA 21")
    add(row["ST_DIR"] > 0, 20, "Supertrend bullish", "Supertrend bearish")
    r = row["RSI"]
    add(None if 45 <= r <= 55 else r > 55, 15, f"RSI {r:.0f} - buyers in control", f"RSI {r:.0f} - sellers in control",
        f"RSI {r:.0f} - neutral")
    if pd.notna(row.get("TC")):
        add(None if row["BC"] <= c <= row["TC"] else c > row["TC"], 10, "Above CPR", "Below CPR", "Inside CPR")
    if pd.notna(row.get("OR_H")):
        add(None if row["OR_L"] <= c <= row["OR_H"] else c > row["OR_H"], 10, "Above opening range",
            "Below opening range", "Inside opening range")
    label = ("Strong bullish" if score >= 50 else "Bullish" if score >= 20 else "Strong bearish" if score <= -50
             else "Bearish" if score <= -20 else "Neutral / range")
    return dict(score=int(score), label=label, items=items)


def strat_status(trades, open_t, signal, last_row, last_date, capital, slip):
    if open_t and open_t["date"] == str(last_date):
        side, qty = open_t["side"], open_t["qty"]
        px = float(last_row["Close"])
        buy_v, sell_v = (open_t["entry"] * qty, px * qty) if side > 0 else (px * qty, open_t["entry"] * qty)
        unreal = side * (px - open_t["entry"]) * qty - charges(buy_v, sell_v, slip)["total"]
        return dict(state="LONG" if side > 0 else "SHORT", active=True, entry=open_t["entry"], stop=open_t["stop"],
                    target=open_t["target"], since=open_t["entry_time"], qty=qty, unreal=round(unreal, 2))
    if signal:
        return dict(state="NEW LONG" if signal["side"] > 0 else "NEW SHORT", active=False, entry=signal["entry"],
                    stop=signal["stop"], target=signal["target"], since=signal["time"])
    todays = [t for t in trades if t["date"] == str(last_date)]
    return dict(state="WAIT", active=False, today_trades=len(todays),
                today_net=round(sum(t["net"] for t in todays), 2))


# ------------------------------------------------------------------------------------------
# API
# ------------------------------------------------------------------------------------------
def analyze(sym: str, interval: str, capital: float, slip: float) -> dict:
    interval = interval if interval in PERIOD else "5m"
    t = resolve(sym)
    d = compute(fetch(t, interval))
    now = now_ist()
    is_open = market_open(now)
    last_t = d.index[-1].to_pydatetime()
    forming = is_open and last_t.date() == now.date() and now < last_t + dt.timedelta(minutes=BAR_MIN[interval])
    dc = d.iloc[:-1] if forming and len(d) > 2 else d
    last = dc.iloc[-1]
    last_date = dc["date"].iloc[-1]

    strategies = {}
    for k, sp in STRATS.items():
        trades, open_t, sig = simulate(dc, k, capital, slip)
        strategies[k] = dict(name=sp["name"], short=sp["short"], desc=sp["desc"], stats=trade_stats(trades, capital),
                             trades=trades, open=open_t,
                             status=strat_status(trades, open_t, sig, last, last_date, capital, slip))

    ts = d.index.values.astype("datetime64[s]").astype(np.int64).tolist()
    first_of_day = (d["date"] != d["date"].shift(1)).to_numpy()
    vwap = d["VWAP"].where(~first_of_day)
    lvl = {k: num(last.get(k)) for k in ("PDH", "PDL", "PDC", "P", "TC", "BC", "R1", "S1", "R2", "S2", "OR_H", "OR_L")}
    cpr_w = (abs(lvl["TC"] - lvl["BC"]) / lvl["P"] * 100) if lvl["P"] else None
    ltp = float(d["Close"].iloc[-1])
    has_vol = bool(d["Volume"].tail(200).sum() > 0)
    return dict(
        symbol=display_symbol(t), ticker=t, tv_symbol=tv_symbol(t), interval=interval, bar_min=BAR_MIN[interval],
        bars_per_day=int(375 / BAR_MIN[interval]), market_open=is_open, forming=bool(forming),
        updated=now.strftime("%d %b %Y %H:%M:%S"), last_bar=d.index[-1].strftime("%d %b %H:%M"),
        session_date=str(last_date), ltp=round(ltp, 2), pdc=lvl["PDC"],
        chg_pct=num((ltp / lvl["PDC"] - 1) * 100) if lvl["PDC"] else None, has_volume=has_vol,
        t=ts, o=arr(d["Open"]), h=arr(d["High"]), l=arr(d["Low"]), c=arr(d["Close"]), v=arr(d["Volume"], 0),
        vwap=arr(vwap), ema9=arr(d["EMA9"]), ema21=arr(d["EMA21"]), rsi=arr(d["RSI"], 1),
        st_up=arr(d["ST_LINE"].where(d["ST_DIR"] > 0)), st_dn=arr(d["ST_LINE"].where(d["ST_DIR"] < 0)),
        levels=lvl, cpr_width=num(cpr_w, 3), atr=num(last["ATR"]),
        bias=bias(last), strategies=strategies, capital=capital, slip=slip,
    )


_scan_cache: dict = {}


def scan(interval: str, capital: float, slip: float) -> dict:
    interval = interval if interval in PERIOD else "5m"
    key = (interval, capital, slip)
    hit = _scan_cache.get(key)
    if hit and time.time() - hit[0] < 60:
        return hit[1]
    tickers = [resolve(s) for s in WATCHLIST]
    raw = yf.download(tickers, period="5d", interval=interval, group_by="ticker", threads=True,
                      progress=False, auto_adjust=False)
    now = now_ist()
    is_open = market_open(now)
    rows = []
    for s, t in zip(WATCHLIST, tickers):
        try:
            df = clean(raw[t])
            if len(df) < 60:
                raise ValueError("no data")
            d = compute(df)
            last_t = d.index[-1].to_pydatetime()
            forming = is_open and last_t.date() == now.date() and now < last_t + dt.timedelta(minutes=BAR_MIN[interval])
            dc = d.iloc[:-1] if forming else d
            last = dc.iloc[-1]
            ltp = float(d["Close"].iloc[-1])
            sigs = {}
            for k in STRATS:
                trades, open_t, sig = simulate(dc, k, capital, slip)
                sigs[k] = strat_status(trades, open_t, sig, last, dc["date"].iloc[-1], capital, slip)
            b = bias(last)
            rows.append(dict(symbol=s, ltp=round(ltp, 2),
                             chg=num((ltp / last["PDC"] - 1) * 100) if pd.notna(last["PDC"]) else None,
                             bias=b["score"], label=b["label"], vwap=num((ltp / last["VWAP"] - 1) * 100),
                             rsi=num(last["RSI"], 0), signals=sigs))
        except Exception as e:  # noqa: BLE001
            rows.append(dict(symbol=s, error=str(e)[:80]))
    out = dict(rows=rows, updated=now.strftime("%d %b %H:%M:%S"), market_open=is_open, interval=interval)
    _scan_cache[key] = (time.time(), out)
    return out


TRIGGERS = {
    "orb": {1: "After 9:30, buy if a candle closes above the 9:15-9:30 high and above VWAP, with volume at least 1.2x "
               "average. Stop at the 9:15-9:30 low (max 2 ATR). Target 2R. Exit if price closes back below VWAP.",
            -1: "After 9:30, short if a candle closes below the 9:15-9:30 low and below VWAP, with volume at least 1.2x "
                "average. Stop at the 9:15-9:30 high (max 2 ATR). Target 2R. Exit if price closes back above VWAP."},
    "pullback": {1: "From 9:45, with price above VWAP, EMA9 above EMA21 and Supertrend up: buy the first green candle "
                    "that dips to EMA9 and closes above it (RSI 50-70). Stop below that candle. Target 2R.",
                 -1: "From 9:45, with price below VWAP, EMA9 below EMA21 and Supertrend down: short the first red candle "
                     "that pokes up to EMA9 and closes below it (RSI 30-50). Stop above that candle. Target 2R."},
    "supertrend": {1: "Buy within 3 candles of Supertrend flipping up, if price is above VWAP and EMA9 is above EMA21. "
                      "Stop at the Supertrend line. Target 2R. Exit if it flips down.",
                   -1: "Short within 3 candles of Supertrend flipping down, if price is below VWAP and EMA9 is below "
                       "EMA21. Stop at the Supertrend line. Target 2R. Exit if it flips up."},
}
_batch_cache: dict = {}


def batch(interval: str, ttl: int = 120):
    """60 days of candles for the whole watchlist in one download (shared by the plan and big-money tabs)."""
    hit = _batch_cache.get(interval)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    raw = yf.download([resolve(x) for x in WATCHLIST], period=PERIOD[interval], interval=interval,
                      group_by="ticker", threads=True, progress=False, auto_adjust=False)
    _batch_cache[interval] = (time.time(), raw)
    return raw


PRIOR = 20          # weight (in trades) given to the all-stocks average when estimating one stock's odds
_plan_cache: dict = {}


def plan(interval: str, capital: float, slip: float) -> dict:
    """Rank every stock x setup for the current (or next) session by expected profit after costs."""
    interval = interval if interval in PERIOD else "5m"
    key = (interval, capital, slip)
    hit = _plan_cache.get(key)
    if hit and time.time() - hit[0] < 300:
        return hit[1]
    tickers = [resolve(x) for x in WATCHLIST]
    raw = batch(interval)
    now = now_ist()
    is_open = market_open(now)
    per, pool, last_bars = [], {}, []
    for sym, t in zip(WATCHLIST, tickers):
        try:
            df = clean(raw[t])
            if len(df) < 200:
                continue
            d = compute(df)
            last_t = d.index[-1].to_pydatetime()
            forming = is_open and last_t.date() == now.date() and now < last_t + dt.timedelta(minutes=BAR_MIN[interval])
            dc = d.iloc[:-1] if forming else d
            last, last_date = dc.iloc[-1], dc["date"].iloc[-1]
            live_today = is_open and last_date == now.date()
            lv = {k: num(last.get(k)) for k in LEVEL_KEYS} if live_today else next_levels(dc)
            b = bias(last)
            strat = {}
            for k in STRATS:
                trades, open_t, sig = simulate(dc, k, capital, slip)
                for tr in trades:
                    pool.setdefault((k, tr["side"]), []).append(tr["net"])
                    pool.setdefault((k, 0), []).append(tr["net"])
                strat[k] = dict(trades=[(tr["side"], tr["bias"], tr["net"], tr["risk"]) for tr in trades],
                                status=strat_status(trades, open_t, sig, last, last_date, capital, slip))
            ltp = float(d["Close"].iloc[-1])
            per.append(dict(symbol=sym, ltp=round(ltp, 2), atr=float(last["ATR"]), bias=b["score"], label=b["label"],
                            chg=num((ltp / last["PDC"] - 1) * 100) if pd.notna(last["PDC"]) else None,
                            levels=lv, strat=strat, live_today=live_today))
            last_bars.append(last_t)
        except Exception:  # noqa: BLE001
            continue
    if not per:
        raise ValueError("No data came back for the watchlist. Try again in a minute.")

    pooled = {}
    for k in STRATS:
        for sd_ in (1, -1, 0):
            nets = np.array(pool.get((k, sd_), []))
            pooled[(k, sd_)] = dict(n=int(len(nets)), win=float((nets > 0).mean()) if len(nets) else 0.0,
                                    ev=float(nets.mean()) if len(nets) else 0.0)

    cands = []
    for p in per:
        side = 1 if p["bias"] >= 20 else -1 if p["bias"] <= -20 else 0
        for k, sp in STRATS.items():
            trs = p["strat"][k]["trades"]
            rel = [x for x in trs if side == 0 or x[0] == side]       # same-direction trades on this stock
            n, w, tot = len(rel), sum(1 for x in rel if x[2] > 0), sum(x[2] for x in rel)
            p0, ev0 = pooled[(k, side)]["win"], pooled[(k, side)]["ev"]
            prob = (w + PRIOR * p0) / (n + PRIOR)
            ev = (tot + PRIOR * ev0) / (n + PRIOR)
            sd = math.sqrt(prob * (1 - prob) / (n + PRIOR + 1))
            risk = float(np.median([x[3] for x in trs])) if trs else p["atr"]
            st = p["strat"][k]["status"]
            live = st["state"] != "WAIT" and (side == 0 or (("LONG" in st["state"]) == (side > 0)))
            cands.append(dict(
                symbol=p["symbol"], strategy=k, setup=sp["name"], side=side, bias=p["bias"], label=p["label"],
                ltp=p["ltp"], chg=p["chg"], prob=round(prob * 100, 1), lo=round(max(0, prob - 1.645 * sd) * 100, 1),
                hi=round(min(1, prob + 1.645 * sd) * 100, 1), ev=round(ev, 0), n=n,
                stock_win=round(w / n * 100, 1) if n else None, risk=round(risk, 2),
                risk_pct=round(risk / p["ltp"] * 100, 2), qty=max(1, int(capital // p["ltp"])),
                trigger=TRIGGERS[k][side] if side else "No clear direction yet. " + TRIGGERS[k][1].split(".")[0] +
                "; or the mirror image for a short. Let the first 15-30 minutes pick the side.",
                levels=p["levels"], live=live, status=st))
    cands.sort(key=lambda c: (0 if (c["live"] and c["ev"] > 0) else 1, -c["ev"], -c["prob"]))
    best, seen = [], set()
    for c in cands:
        if c["symbol"] not in seen:
            seen.add(c["symbol"])
            best.append(c)
    live_session = any(p["live_today"] for p in per)
    last_bar = max(last_bars)
    plan_date = now.date() if live_session else next_session(last_bar.date())
    out = dict(
        generated=now.strftime("%d %b %Y %H:%M"), data_until=last_bar.strftime("%d %b %Y, %H:%M"),
        live_session=live_session, plan_date=plan_date.strftime("%a, %d %b %Y"),
        session_label=("Today's plan, " if live_session else "Plan for ") + plan_date.strftime("%a %d %b %Y"),
        capital=capital, slip=slip, interval=interval, prior=PRIOR, symbols=len(per),
        positive=sum(1 for c in best if c["ev"] > 0), top=best[:5], best=best,
        edge=[dict(strategy=k, name=STRATS[k]["name"],
                   **{nm: dict(n=pooled[(k, sd_)]["n"], win=round(pooled[(k, sd_)]["win"] * 100, 1),
                               ev=round(pooled[(k, sd_)]["ev"], 0))
                      for nm, sd_ in (("all", 0), ("longs", 1), ("shorts", -1))}) for k in STRATS],
    )
    _plan_cache[key] = (time.time(), out)
    return out


# ------------------------------------------------------------------------------------------
# Big money: NSE bulk/block deals, FII/DII flows, unusual-volume candles
# ------------------------------------------------------------------------------------------
NSE_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
INSTITUTION_RE = re.compile(
    r"\b(FUND|FUNDS|MUTUAL|MF|INSURANCE|ASSURANCE|PENSION|BANK|BANKING|SOCIETE|GOLDMAN|MORGAN|NOMURA|CITIGROUP|"
    r"BNP|UBS|JPMORGAN|MERRILL|BOFA|FPI|AIF|LIC|NIPPON|MIRAE|INVESCO|VANGUARD|BLACKROCK|NORGES|GOVERNMENT|"
    r"ABU DHABI|MONETARY AUTHORITY|PLC|LLC|LP|FOREIGN|PORTFOLIO|ASSET MANAGEMENT|AMC|TRUSTEE|INVESTMENT|"
    r"INVESTMENTS|CAPITAL|SECURITIES|HOLDINGS|PARTNERS|MASTER|OPPORTUNITIES)\b", re.I)


def nse_json(path: str):
    req = urllib.request.Request("https://www.nseindia.com/api/" + path,
                                 headers={"User-Agent": NSE_UA, "Referer": "https://www.nseindia.com/",
                                          "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def _deals(rows, wl):
    out = []
    for r in rows or []:
        try:
            qty, px = float(r.get("qty") or 0), float(r.get("watp") or 0)
        except ValueError:
            continue
        name = (r.get("clientName") or "").strip()
        out.append(dict(date=r.get("date"), symbol=r.get("symbol"), company=r.get("name"), client=name,
                        side=(r.get("buySell") or "").upper(), qty=int(qty), price=round(px, 2),
                        value_cr=round(qty * px / 1e7, 2), institution=bool(INSTITUTION_RE.search(name)),
                        watch=(r.get("symbol") in wl)))
    out.sort(key=lambda x: -x["value_cr"])
    return out


def volume_spikes(interval: str = "5m", sessions: int = 5, min_ratio: float = 4.0) -> list:
    """Candles whose volume is many times the usual volume for that time of day (last few sessions)."""
    raw = batch(interval)
    spikes = []
    for sym in WATCHLIST:
        t = resolve(sym)
        try:
            d = clean(raw[t])
        except Exception:  # noqa: BLE001
            continue
        if d["Volume"].tail(200).sum() == 0 or len(d) < 300:
            continue                                                      # indices have no volume
        d["date"] = d.index.date
        d["tod"] = d.index.hour * 60 + d.index.minute
        usual = d.groupby("tod")["Volume"].transform(lambda x: x.shift(1).rolling(20, min_periods=8).median())
        d["ratio"] = d["Volume"] / usual.replace(0, np.nan)
        recent = sorted(d["date"].unique())[-sessions:]
        hit = d[(d["date"].isin(recent)) & (d["ratio"] >= min_ratio)]
        for ts, r in hit.iterrows():
            px = float(r["Close"])
            spikes.append(dict(symbol=sym, time=ts.strftime("%d %b %H:%M"), ts=int(pd.Timestamp(ts).value // 10**9),
                               qty=int(r["Volume"]), ratio=round(float(r["ratio"]), 1), price=round(px, 2),
                               value_cr=round(float(r["Volume"]) * px / 1e7, 2),
                               move=round((px / float(r["Open"]) - 1) * 100, 2),
                               dir=1 if px > r["Open"] else -1 if px < r["Open"] else 0))
    spikes.sort(key=lambda x: (-x["ts"] // 86400, -x["value_cr"]))
    return spikes[:80]


_inst_cache: dict = {}


def institutional(interval: str = "5m") -> dict:
    hit = _inst_cache.get(interval)
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    wl = set(WATCHLIST)
    out = dict(generated=now_ist().strftime("%d %b %Y %H:%M"), errors=[])
    try:
        out["fiidii"] = [dict(category=r.get("category"), date=r.get("date"), buy=float(r.get("buyValue") or 0),
                              sell=float(r.get("sellValue") or 0), net=float(r.get("netValue") or 0))
                         for r in nse_json("fiidiiTradeReact")]
    except Exception as e:  # noqa: BLE001
        out["fiidii"], _ = [], out["errors"].append(f"FII/DII flows unavailable ({e.__class__.__name__})")
    try:
        ld = nse_json("snapshot-capital-market-largedeal")
        out["as_on"] = ld.get("as_on_date")
        out["block"] = _deals(ld.get("BLOCK_DEALS_DATA"), wl)
        out["bulk"] = _deals(ld.get("BULK_DEALS_DATA"), wl)
    except Exception as e:  # noqa: BLE001
        out["block"], out["bulk"] = [], []
        out["errors"].append(f"Bulk/block deals unavailable ({e.__class__.__name__})")
    try:
        out["spikes"] = volume_spikes(interval)
    except Exception as e:  # noqa: BLE001
        out["spikes"] = []
        out["errors"].append(f"Volume scan unavailable ({e.__class__.__name__})")
    _inst_cache[interval] = (time.time(), out)
    return out


# ------------------------------------------------------------------------------------------
# Monthly picks (factor research: docs/FINDINGS.md sections 4-5) and the paper-trading journal
# ------------------------------------------------------------------------------------------
PICK_N = 10
PICK_COST_PCT = 0.30                    # assumed round trip per rebalance (0.15% per side), before tax
JOURNAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "journal", "paper_journal.json")
PICK_STRATS = {
    "momentum": dict(name="Momentum top 10", months=tuple(range(1, 13)), every="monthly",
                     rule="Ranked by return from 12 months to 1 month before the last month-end. Buy the top 10; keep a "
                          "holding until it falls below rank 20 (banding: about 60% less trading and short-term tax). "
                          "Checked every month."),
    "lowvol": dict(name="Low volatility top 10", months=(3, 6, 9, 12), every="quarterly",
                   rule="The 10 stocks with the lowest 1-year daily volatility at the last month-end. "
                        "Rebalanced every quarter (after March, June, September and December)."),
}
_daily_cache: dict = {}
_picks_cache: dict = {}


def daily_closes(ttl: int = 3600) -> pd.DataFrame:
    """About 2 years of daily closes (dividend-adjusted) for the watchlist stocks plus NIFTY."""
    hit = _daily_cache.get("c")
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    syms = [s for s in WATCHLIST if s not in ("NIFTY", "BANKNIFTY")]
    raw = yf.download([resolve(s) for s in syms] + ["^NSEI"], period="2y", interval="1d", auto_adjust=True,
                      group_by="ticker", threads=True, progress=False)
    cols = {}
    for s in syms + ["NIFTY"]:
        try:
            c = raw["^NSEI" if s == "NIFTY" else resolve(s)]["Close"].dropna()
        except KeyError:
            continue
        idx = pd.to_datetime(c.index)
        c.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        if len(c) > 260:
            cols[s] = c[~c.index.duplicated(keep="last")]
    if "NIFTY" not in cols:
        raise ValueError("No daily data for the Nifty from Yahoo right now.")
    df = pd.DataFrame(cols).sort_index()
    _daily_cache["c"] = (time.time(), df)
    return df


PICK_BAND = 20      # banding: a momentum holding is kept until it falls below this rank (docs/FINDINGS.md s.6)
CORE_PCT = 70       # core-satellite plan: % in a Nifty 50 index fund, the rest split over the momentum picks


def compute_picks(C: pd.DataFrame, n: int = PICK_N, held=None, band: int = PICK_BAND) -> dict:
    """Rank on closes up to the last completed month-end; prices are the latest close.
    Momentum uses banding: current holdings (`held`) stay while ranked within `band`, then the top fills the rest."""
    S = C.drop(columns="NIFTY")
    last = S.index[-1]
    form = S.index[S.index.to_period("M") < last.to_period("M")][-1]
    H = S.loc[:form]
    vol = H.pct_change().tail(252).std() * math.sqrt(252) * 100
    mom = (H.iloc[-22] / H.iloc[-253] - 1) * 100 if len(H) >= 253 else pd.Series(np.nan, index=H.columns)
    ok = mom.notna() & vol.notna() & (H.tail(253).count() >= 253)
    px = S.iloc[-1]

    ranked = mom[ok].sort_values(ascending=False)
    mrank = {s: i + 1 for i, s in enumerate(ranked.index)}
    keep = [s for s in (held or []) if mrank.get(s, band + 1) <= band][:n]
    for s in ranked.index:
        if len(keep) >= n:
            break
        if s not in keep:
            keep.append(s)
    keep.sort(key=lambda s: mrank[s])
    lv = vol[ok].sort_values().head(n)

    def row(s, rank, kept=False):
        return dict(rank=rank, symbol=s, price=num(px[s]), mom=num(mom[s], 1), vol=num(vol[s], 1), kept=kept)
    m = int(last.month)
    q = next(x for x in (3, 6, 9, 12) if x >= m)
    return dict(formed=str(form.date()), formation=str(form.to_period("M")), asof=str(last.date()),
                universe=int(ok.sum()), band=band,
                momentum=[row(s, mrank[s], kept=mrank[s] > n) for s in keep],
                lowvol=[row(s, i + 1) for i, s in enumerate(lv.index)],
                prices={s: num(px[s]) for s in S.columns if pd.notna(px[s])}, nifty=num(C["NIFTY"].iloc[-1]),
                next=dict(momentum=dt.date(2000, m, 1).strftime("%B"), lowvol=dt.date(2000, q, 1).strftime("%B")))


def update_journal(J: dict, P: dict, logged: str) -> dict:
    """Close the open paper period and open a new one at today's prices when a rebalance is due."""
    J = json.loads(json.dumps(J))
    J.setdefault("started", logged)
    fm, month = P["formation"], int(P["formation"][-2:])
    for k, sp in PICK_STRATS.items():
        periods = J.setdefault(k, [])
        cur = periods[-1] if periods and not periods[-1].get("closed") else None
        if cur is not None and not (cur["formation"] < fm and month in sp["months"]):
            continue
        if cur is not None:
            cur.update(closed=logged, exit={s: P["prices"].get(s) for s in cur["entry"]}, nifty_exit=P["nifty"])
        periods.append(dict(formation=fm, formed=P["formed"], opened=logged, closed=None,
                            entry={r["symbol"]: r["price"] for r in P[k]}, nifty_entry=P["nifty"]))
    return J


def journal_summary(J: dict, P: dict) -> dict:
    out = {}
    for k, sp in PICK_STRATS.items():
        rows, cum, ncum, prev = [], 1.0, 1.0, set()
        for p in J.get(k, []):
            new = set(p["entry"])
            changed = len(new - prev) / len(new) if new else 1.0     # kept holdings are not traded again
            prev = new
            closed = bool(p.get("closed"))
            ex = p.get("exit") if closed else {s: P["prices"].get(s) for s in p["entry"]}
            nx = p.get("nifty_exit") if closed else P["nifty"]
            picks_ = [dict(symbol=s, entry=e, now=ex.get(s), ret=num((ex[s] / e - 1) * 100) if e and ex.get(s) else None)
                      for s, e in p["entry"].items()]
            rs_ = [x["ret"] for x in picks_ if x["ret"] is not None]
            gross = float(np.mean(rs_)) if rs_ else 0.0
            net = gross - PICK_COST_PCT * changed
            nifty = (nx / p["nifty_entry"] - 1) * 100 if nx and p.get("nifty_entry") else 0.0
            cum *= 1 + net / 100
            ncum *= 1 + nifty / 100
            rows.append(dict(formation=p["formation"], opened=p["opened"], closed=p.get("closed"), open=not closed,
                             gross=num(gross), net=num(net), nifty=num(nifty), changed=num(changed * 100, 0),
                             picks=picks_))
        out[k] = dict(name=sp["name"], every=sp["every"], started=rows[0]["opened"] if rows else None,
                      periods=rows, cum_net=num((cum - 1) * 100), cum_nifty=num((ncum - 1) * 100))
    return out


def load_journal() -> dict:
    try:
        with open(JOURNAL_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def picks(write_journal: bool = False) -> dict:
    hit = _picks_cache.get("p")
    if hit and time.time() - hit[0] < 300 and not write_journal:
        return hit[1]
    J = load_journal()
    open_mom = [p for p in J.get("momentum", []) if not p.get("closed")]
    P = compute_picks(daily_closes(), held=list(open_mom[-1]["entry"]) if open_mom else None)
    if write_journal:
        J = update_journal(J, P, now_ist().strftime("%Y-%m-%d %H:%M"))
        os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
        with open(JOURNAL_PATH, "w", encoding="utf-8") as f:
            json.dump(J, f, indent=1)
    out = {k: P[k] for k in ("formed", "formation", "asof", "universe", "band", "momentum", "lowvol", "next")}
    out.update(generated=now_ist().strftime("%d %b %Y %H:%M"), cost_pct=PICK_COST_PCT, core_pct=CORE_PCT,
               rules={k: v["rule"] for k, v in PICK_STRATS.items()}, journal=journal_summary(J, P))
    _picks_cache["p"] = (time.time(), out)
    return out


# ------------------------------------------------------------------------------------------
# Options lab: NSE end-of-day option chains (bhavcopy) + live underlying prices; maths in options_lab.py
# ------------------------------------------------------------------------------------------
_opt_cache: dict = {}
OPT_BT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "options_backtest.json")


def options_payload(ttl: int = 600) -> dict:
    import options_lab as ol
    hit = _opt_cache.get("o")
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    bhav = ol.latest_bhavcopy(now_ist().date())
    if bhav is None:
        raise ValueError("NSE's option price file could not be downloaded right now.")
    try:
        raw = batch("5m")
    except Exception:  # noqa: BLE001
        raw = None
    chains = {}
    for s in WATCHLIST:
        ch = ol.chain(bhav, s)
        if not ch:
            continue
        try:
            d = clean(raw[resolve(s)])
            d["date"] = d.index.date
            daily = d.groupby("date").agg(O=("Open", "first"), H=("High", "max"), L=("Low", "min"), C=("Close", "last")).tail(21)
            rets = daily["C"].pct_change().dropna()
            ch.update(spot_now=num(d["Close"].iloc[-1]), spot_time=d.index[-1].strftime("%d %b %H:%M"),
                      hv20=num(rets.std() * math.sqrt(252) * 100, 1),
                      range_pct=num(((daily.H - daily.L) / daily.O).tail(20).mean() * 100),
                      move_pct=num(((daily.C - daily.O).abs() / daily.O).tail(20).mean() * 100))
        except Exception:  # noqa: BLE001
            ch.update(spot_now=ch["spot"], spot_time=ch["asof"], hv20=None, range_pct=None, move_pct=None)
        ch["atm_iv"] = ol.atm_iv(ch)
        chains[s] = ch
    bt = None
    try:
        with open(OPT_BT_PATH, encoding="utf-8") as f:
            bt = json.load(f)
    except (OSError, ValueError):
        pass
    out = dict(asof=str(bhav["date"].iloc[0]), generated=now_ist().strftime("%d %b %Y %H:%M"), r=ol.RISK_FREE,
               symbols=list(chains), chains=chains, backtest=bt)
    _opt_cache["o"] = (time.time(), out)
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            capital = max(1000.0, float(q.get("capital", 100000)))
            slip = min(1.0, max(0.0, float(q.get("slip", 0.02))))
            if u.path in ("/", "/index.html"):
                return self._send(200, PAGE, "text/html")
            if u.path == "/api/analyze":
                return self._send(200, json.dumps(analyze(q.get("symbol", "RELIANCE"), q.get("interval", "5m"),
                                                          capital, slip)))
            if u.path == "/api/scan":
                return self._send(200, json.dumps(scan(q.get("interval", "5m"), capital, slip)))
            if u.path == "/api/plan":
                return self._send(200, json.dumps(plan(q.get("interval", "5m"), capital, slip)))
            if u.path == "/api/institutional":
                return self._send(200, json.dumps(institutional(q.get("interval", "5m"))))
            if u.path == "/api/options":
                return self._send(200, json.dumps(options_payload()))
            if u.path == "/api/picks":
                return self._send(200, json.dumps(picks()))
            if u.path == "/api/watchlist":
                return self._send(200, json.dumps(WATCHLIST))
            return self._send(404, json.dumps({"error": "not found"}))
        except Exception as e:  # noqa: BLE001
            return self._send(400, json.dumps({"error": str(e)}))


# ------------------------------------------------------------------------------------------
# Page
# ------------------------------------------------------------------------------------------
LIB = '<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>'

CSS = r"""/* Layout: sticky control bar; chart column + signal rail; tools row. Dark-first trading-terminal palette. */
:root{--bg:#0b0e13;--panel:#12161d;--panel2:#181d26;--line:#232a35;--grid:#1a2029;--ink:#e7eaf0;--mute:#8790a0;--up:#22c55e;--dn:#f05252;--upbg:rgba(34,197,94,.12);--dnbg:rgba(240,82,82,.12);--acc:#f5a524;--blue:#5aa2ff;--violet:#b78bff;--info:rgba(90,162,255,.1);color-scheme:dark}
@media (prefers-color-scheme:light){:root:not([data-theme="dark"]){--bg:#f4f5f7;--panel:#fff;--panel2:#f7f8fa;--line:#e2e5ea;--grid:#eef0f3;--ink:#14171c;--mute:#646c79;--up:#15803d;--dn:#d23434;--upbg:rgba(21,128,61,.09);--dnbg:rgba(210,52,52,.09);--acc:#c27803;--blue:#2563eb;--violet:#7c3aed;--info:rgba(37,99,235,.08);color-scheme:light}}
:root[data-theme="light"]{--bg:#f4f5f7;--panel:#fff;--panel2:#f7f8fa;--line:#e2e5ea;--grid:#eef0f3;--ink:#14171c;--mute:#646c79;--up:#15803d;--dn:#d23434;--upbg:rgba(21,128,61,.09);--dnbg:rgba(210,52,52,.09);--acc:#c27803;--blue:#2563eb;--violet:#7c3aed;--info:rgba(37,99,235,.08);color-scheme:light}
[hidden]{display:none!important}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
font-variant-numeric:tabular-nums}
body::before{content:"";position:fixed;inset:0 0 auto;height:env(safe-area-inset-top,0px);z-index:10;background:var(--bg)}
header{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding-block:10px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));
display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.brand{font-weight:750;letter-spacing:.3px;margin-right:6px}.brand span{color:var(--acc)}
input,select,button{font:inherit;color:var(--ink);background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:7px 10px}
input:focus,select:focus{outline:2px solid var(--blue);outline-offset:-1px}
button{cursor:pointer}button.primary{background:var(--acc);border-color:var(--acc);color:#111;font-weight:650}
button.ghost{background:transparent}
a.btn{color:var(--ink);text-decoration:none;border:1px solid var(--line);border-radius:8px;padding:7px 10px;background:var(--panel2)}
#sym{width:150px;text-transform:uppercase}
.spacer{flex:1}
.pill{display:inline-flex;align-items:center;gap:6px;padding:3px 9px;border-radius:999px;font-size:12px;font-weight:650;border:1px solid var(--line)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--mute)}
.pill.open .dot{background:var(--up);box-shadow:0 0 0 3px var(--upbg)}
.tabs{display:flex;gap:4px}.tabs button{background:transparent;border-color:transparent}.tabs button.on{background:var(--panel2);border-color:var(--line)}
main{max-width:1500px;margin:0 auto;padding-block:14px 40px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.snapnote{margin-bottom:12px;padding:10px 12px;border-radius:10px;background:var(--info);border:1px solid var(--line);font-size:13px}
.grid>*{min-width:0}
.quote{display:flex;flex-wrap:wrap;align-items:baseline;gap:14px;margin:2px 0 12px}
.quote h1{margin:0;font-size:22px}.ltp{font-size:22px;font-weight:700}.mute{color:var(--mute)}.small{font-size:12px}
.up{color:var(--up)}.dn{color:var(--dn)}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:14px}
@media (max-width:1050px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:14px}
.card h2{font-size:13px;text-transform:uppercase;letter-spacing:.6px;color:var(--mute);margin:0 0 10px;font-weight:650}
.row{display:flex;justify-content:space-between;gap:8px;align-items:center}
#chart{height:470px}#rsi{height:120px;margin-top:6px}#eq{height:200px}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--mute);margin-bottom:8px}
.legend i{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}
.legend label{cursor:pointer}
.score{font-size:34px;font-weight:750;line-height:1}
.meter{height:8px;border-radius:5px;background:linear-gradient(90deg,var(--dn),var(--panel2) 45%,var(--panel2) 55%,var(--up));position:relative;margin:10px 0}
.meter b{position:absolute;top:-4px;width:4px;height:16px;border-radius:2px;background:var(--ink);transform:translateX(-2px)}
ul.chk{list-style:none;margin:8px 0 0;padding:0}ul.chk li{padding:4px 0;border-bottom:1px dashed var(--line);font-size:13px}
ul.chk li:before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px;background:var(--mute)}
ul.chk li.bull:before{background:var(--up)}ul.chk li.bear:before{background:var(--dn)}
.sig{border:1px solid var(--line);border-radius:10px;padding:10px;margin-bottom:8px;background:var(--panel2)}
.tag{font-size:11px;font-weight:750;padding:2px 8px;border-radius:6px;letter-spacing:.4px;background:var(--line);color:var(--mute)}
.tag.long{background:var(--upbg);color:var(--up)}.tag.short{background:var(--dnbg);color:var(--dn)}
.kv{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:8px;font-size:12px}.kv div span{display:block;color:var(--mute);font-size:11px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--mute);font-weight:600;font-size:12px}th:first-child,td:first-child{text-align:left}
tr.sel td{background:var(--panel2)}tr.click{cursor:pointer}tr.click:hover td{background:var(--panel2)}
.scroll{overflow-x:auto}.tall{max-height:360px;overflow-y:auto}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:8px;margin:10px 0}
.tile{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:10px}
.tile .v{font-size:17px;font-weight:700;white-space:nowrap}.tile .k{font-size:11px;color:var(--mute);text-transform:uppercase;letter-spacing:.4px}
.tools{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.form{display:grid;grid-template-columns:1fr 1fr;gap:8px}.form label{font-size:12px;color:var(--mute);display:grid;gap:4px}
.form input,.form select{width:100%}
.out{margin-top:10px;background:var(--panel2);border-radius:10px;padding:10px;font-size:13px}
.out .row{padding:2px 0}
.note{font-size:12px;color:var(--mute)}
.banner{display:none;margin-bottom:12px;padding:10px 12px;border-radius:10px;background:var(--dnbg);color:var(--dn);border:1px solid var(--dn)}
.rules li{margin:5px 0}
.hidden{display:none!important}
.planhead{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:8px;margin:2px 0 12px}
.planhead h1{margin:0;font-size:22px;text-wrap:balance}
.verdict{padding:12px 14px;border-radius:10px;margin-bottom:14px;border:1px solid var(--line);background:var(--panel2);font-size:13.5px}
.verdict.warn{border-color:var(--acc);background:color-mix(in srgb,var(--acc) 10%,var(--panel))}
.verdict.ok{border-color:var(--up);background:var(--upbg)}
.picks{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:14px;margin-bottom:14px}
@media (max-width:900px){.picks{grid-template-columns:1fr}}
.runners{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;align-content:start}
.pick{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;display:grid;gap:8px;min-width:0}
.pick.hero{border-color:var(--acc);padding:18px}
.pick .sym{font-size:20px;font-weight:750}.pick.hero .sym{font-size:26px}
.rank{font-size:11px;font-weight:750;color:var(--acc);letter-spacing:.6px;text-transform:uppercase}
.probrow{display:flex;align-items:center;gap:12px}
.probbig{font-size:34px;font-weight:750;line-height:1}.pick:not(.hero) .probbig{font-size:24px}
.pbar{position:relative;height:10px;border-radius:6px;background:var(--panel2);border:1px solid var(--line)}
.pbar span{position:absolute;top:1px;bottom:1px;border-radius:4px;background:color-mix(in srgb,var(--blue) 35%,transparent)}
.pbar b{position:absolute;top:-3px;width:3px;height:14px;border-radius:2px;background:var(--blue);transform:translateX(-1px)}
.pbar i{position:absolute;top:-4px;bottom:-4px;left:50%;border-left:1px dashed var(--mute)}
.trigger{font-size:13px;margin:0;background:var(--panel2);border-radius:8px;padding:9px 10px}
.lvchips{display:flex;flex-wrap:wrap;gap:6px;font-size:12px}
.lvchips span{border:1px solid var(--line);border-radius:6px;padding:2px 7px;color:var(--mute)}.lvchips b{color:var(--ink);font-weight:600}
#plTable th:nth-child(2),#plTable td:nth-child(2),#plTable th:nth-child(3),#plTable td:nth-child(3){text-align:left}
#inBlock th:nth-child(-n+3),#inBlock td:nth-child(-n+3),#inBulk th:nth-child(-n+3),#inBulk td:nth-child(-n+3),#inSpikes th:nth-child(-n+2),#inSpikes td:nth-child(-n+2){text-align:left}
.minibar{display:inline-block;width:60px;height:6px;border-radius:3px;background:var(--panel2);vertical-align:middle;margin-left:6px;position:relative;overflow:hidden}
.minibar span{position:absolute;left:0;top:0;bottom:0;background:var(--blue)}
.loading{opacity:.55;transition:opacity .2s}
"""

BODY = r"""<header>
  <div class="brand">Intraday <span>Desk</span></div>
  <div class="tabs"><button id="tabPlan">Today's plan</button><button id="tabDesk" class="on">Chart</button><button id="tabScan">Scanner</button><button id="tabInst">Big money</button><button id="tabPicks">Monthly picks</button><button id="tabOpt">Options lab</button></div>
  <input id="sym" list="wl" placeholder="Symbol" autocomplete="off"><datalist id="wl"></datalist>
  <select id="iv"><option value="1m">1 min</option><option value="2m">2 min</option><option value="5m" selected>5 min</option><option value="15m">15 min</option></select>
  <button class="primary" id="go">Load</button>
  <label class="small mute" id="autoWrap"><input type="checkbox" id="auto" checked> Auto-refresh</label>
  <a class="btn" id="tv" target="_blank" rel="noopener">Open in TradingView ↗</a>
  <div class="spacer"></div>
  <span class="pill" id="mkt"><span class="dot"></span><span id="mktTxt">—</span></span>
  <span class="mono" id="clock">--:--:--</span>
</header>
<main>
<div class="snapnote" id="snap" hidden></div>
<div class="banner" id="err"></div>

<section id="plan" class="hidden">
  <div class="planhead"><h1 id="plTitle">Today's plan</h1><span class="mute small" id="plSub">Ranking every stock and setup…</span></div>
  <div class="verdict" id="plVerdict">Building the plan from 60 days of 5-minute data for the watchlist. This takes about 10-60 seconds.</div>
  <div class="picks"><div id="plHero"></div><div class="runners" id="plRunners"></div></div>
  <div class="card"><div class="row"><h2>All stocks, best setup for each</h2><span class="note">Sorted by expected profit per trade after costs</span></div>
    <div class="scroll tall"><table id="plTable"></table></div></div>
  <div class="grid">
    <div class="card"><h2>How each setup has done across all stocks</h2><div class="scroll"><table id="plEdge"></table></div>
      <p class="note">Last 60 days, 5-minute candles, ₹1,00,000 per trade, Zerodha-style charges and 0.02% slippage per side. A win means a net profit after charges.</p></div>
    <div class="card"><h2>How to read this</h2><ul class="small" style="margin:0;padding-left:18px;display:grid;gap:6px">
      <li><b>Probability of profit</b> is how often this setup, traded in this direction, ended in a net profit on this stock, blended with the result across all stocks. Small samples are pulled toward the average, and the range shows the uncertainty (90%).</li>
      <li><b>Expected profit</b> is the average result per trade after charges. It ranks the list because a 40% win rate can still pay if wins are twice the size of losses.</li>
      <li><b>Direction</b> comes from the bias at the last close. Recheck it after 9:30 before acting on any trigger.</li>
      <li>These are past results on a short window, not a forecast. When nothing has a positive expected profit, the best trade is no trade.</li></ul></div>
  </div>
</section>

<section id="desk">
  <div class="quote">
    <h1 id="qSym">—</h1><span class="ltp" id="qLtp">—</span><span id="qChg"></span>
    <span class="mute small" id="qUpd"></span>
  </div>
  <div class="grid">
    <div>
      <div class="card">
        <div class="legend">
          <span><i style="background:var(--acc)"></i>VWAP</span><span><i style="background:var(--blue)"></i>EMA 9</span>
          <span><i style="background:var(--violet)"></i>EMA 21</span><span><i style="background:var(--up)"></i>Supertrend</span>
          <label><input type="checkbox" id="showLv" checked> Levels</label>
          <label><input type="checkbox" id="showMk" checked> Backtest trades</label>
          <span id="mkName"></span>
        </div>
        <div id="chart"></div><div id="rsi"></div>
      </div>

      <div class="card">
        <div class="row"><h2>Intraday backtest on this stock</h2><span class="note" id="btNote"></span></div>
        <div class="scroll"><table id="cmp"></table></div>
        <p class="note">Read "Before costs" first: if it is near zero or negative, the setup has no edge on this stock and charges turn it into a loser. Click a row to see its trades on the chart.</p>
        <div id="detail"></div>
        <div id="eq"></div>
        <h2 style="margin-top:14px">Trades <span class="note" id="trN"></span></h2>
        <div class="scroll tall"><table id="trades"></table></div>
      </div>
    </div>

    <aside>
      <div class="card">
        <h2>Intraday bias</h2>
        <div class="row"><div class="score" id="bScore">—</div><div id="bLabel" style="font-weight:650"></div></div>
        <div class="meter"><b id="bMark" style="left:50%"></b></div>
        <ul class="chk" id="bItems"></ul>
      </div>
      <div class="card"><h2>Strategy signals</h2><div id="sigs"></div>
        <p class="note">Signals use completed candles only. A NEW signal means: enter at the next candle's open. Confirm on the chart before trading.</p></div>
      <div class="card"><h2>Key levels</h2><table id="lv"></table><p class="note" id="cprNote"></p></div>
      <div class="card"><h2>Backtest settings</h2>
        <div class="form">
          <label>Capital per trade (₹)<input id="cap" type="number" min="1000" step="1000" value="100000"></label>
          <label>Slippage per side (%)<input id="slip" type="number" min="0" max="1" step="0.01" value="0.02"></label>
        </div>
        <p class="note">Charges: Zerodha-style NSE intraday (brokerage 0.03% or ₹20, STT 0.025% sell, exchange 0.00307%, SEBI, stamp 0.003% buy, 18% GST).</p>
      </div>
    </aside>
  </div>

  <div class="tools">
    <div class="card"><h2>Position size</h2>
      <div class="form">
        <label>Capital (₹)<input id="pCap" type="number" value="100000"></label>
        <label>Risk per trade (%)<input id="pRisk" type="number" value="1" step="0.25"></label>
        <label>Entry<input id="pEntry" type="number" step="0.05"></label>
        <label>Stop-loss<input id="pStop" type="number" step="0.05"></label>
        <label>MIS leverage (x)<input id="pLev" type="number" value="5" step="1"></label>
        <label>Target (R multiple)<input id="pRR" type="number" value="2" step="0.5"></label>
      </div><div class="out" id="pOut">Enter entry and stop-loss.</div></div>
    <div class="card"><h2>Charges &amp; breakeven</h2>
      <div class="form">
        <label>Direction<select id="cSide"><option value="1">Buy then sell</option><option value="-1">Short then cover</option></select></label>
        <label>Quantity<input id="cQty" type="number" value="100"></label>
        <label>Entry price<input id="cEntry" type="number" step="0.05"></label>
        <label>Exit price<input id="cExit" type="number" step="0.05"></label>
      </div><div class="out" id="cOut">Enter prices.</div></div>
    <div class="card"><h2>Discipline rules</h2>
      <ol class="rules small">
        <li>Risk at most 1% of capital per trade; size from the stop, not from leverage.</li>
        <li>Stop for the day after 2 losing trades or a 2% loss.</li>
        <li>Only take setups that agree with the bias and VWAP side.</li>
        <li>No new entries after 2:30 PM; be flat by 3:15 PM (your broker auto-squares MIS a few minutes later, with a fee).</li>
        <li>Never move a stop further away or average a losing intraday trade.</li>
        <li>Prefer strategies whose backtest on this stock is positive after charges.</li>
      </ol></div>
  </div>
</section>

<section id="inst" class="hidden">
  <div class="planhead"><h1>Big money</h1><span class="mute small" id="inSub">Loading NSE data…</span></div>
  <div class="verdict">NSE publishes <b>block deals</b> (large trades in the 8:45-9:00 AM and 2:05-2:20 PM windows) and <b>bulk deals</b> (one client trading 0.5% or more of a company's shares in a day) with the client's name, quantity and price, plus <b>FII/DII totals</b> for the whole market. It does not publish the exact time of these trades, or which institution is behind any single trade on the screen. The last table fills that gap the only way possible: it shows the exact 5-minute candles where volume was many times normal, without the name of who traded.</div>
  <div class="banner" id="inErr"></div>
  <div class="tiles" id="inFlows"></div>
  <div class="card"><div class="row"><h2>Block deals</h2><span class="note" id="inBlockN"></span></div>
    <div class="scroll tall"><table id="inBlock"></table></div>
    <p class="note">"Likely institution" is a guess from the client's name (fund, bank, insurer, FPI and similar words). Watchlist stocks are marked ★.</p></div>
  <div class="card"><div class="row"><h2>Bulk deals</h2><span class="note" id="inBulkN"></span></div>
    <div class="scroll tall"><table id="inBulk"></table></div></div>
  <div class="card"><div class="row"><h2>Unusual volume on the watchlist (last 5 sessions)</h2><span class="note">5-min candles with volume 4× or more the usual for that time of day</span></div>
    <div class="scroll tall"><table id="inSpikes"></table></div>
    <p class="note">The quantity is everyone's total in that candle, not one buyer. Candles at 9:15 and 3:25 often spike because of the open and close auctions and index rebalancing. Click a row to open its chart.</p></div>
</section>

<section id="picks" class="hidden">
  <div class="planhead"><h1>Monthly picks</h1><span class="mute small" id="pkSub">Loading daily data…</span></div>
  <div class="verdict warn"><b>Longer-term lists, not intraday trades.</b> These follow the two factor rules tested in docs/FINDINGS.md
    (2016–2026, 43 stocks, after costs and tax). Momentum beat simply holding the same stocks by only about 1% a year, and
    not in every period; low volatility earned less but fell less (−28% worst fall vs −41%). Both results are flattered by
    hindsight because the list is today's large caps. The journal below tracks the picks on paper from the day they were
    logged, so you can see whether they keep working before risking money.</div>
  <div class="card"><div class="row"><h2>Portfolio plan (core-satellite)</h2><span class="note">How large fund managers combine a cheap index core with a factor tilt</span></div>
    <div class="form" style="max-width:420px"><label>Amount to invest (₹)<input id="pfAmt" type="number" min="1000" step="1000" value="100000"></label></div>
    <div id="pfPlan" class="out"></div>
    <p class="note">Backtest 2016–2026 after costs and tax (docs/FINDINGS.md section 6): 70% Nifty + 30% banded momentum made about 14.1% a year
      against 10.2% for the Nifty alone (price index; add ~1.3% a year for dividends) with a similar worst fall (−38%). All-momentum made more
      (about 22%) but is far riskier and the most flattered by hindsight. The core is an index fund or ETF you hold for years; only the
      satellite changes each month. Long-term holdings, not intraday trades. Not investment advice.</p></div>
  <div class="tools">
    <div class="card"><div class="row"><h2>Momentum top 10</h2><span class="note" id="pkMomNext"></span></div>
      <p class="note" id="pkMomRule"></p><div class="scroll"><table id="pkMom"></table></div></div>
    <div class="card"><div class="row"><h2>Low volatility top 10</h2><span class="note" id="pkLowNext"></span></div>
      <p class="note" id="pkLowRule"></p><div class="scroll"><table id="pkLow"></table></div></div>
  </div>
  <div class="card"><div class="row"><h2>Paper-trading journal</h2><span class="note" id="pkJNote"></span></div>
    <div id="pkJournal"></div></div>
</section>

<section id="opt" class="hidden">
  <div class="planhead"><h1>Options lab</h1><span class="mute small" id="olSub">Loading NSE option prices…</span></div>
  <div class="verdict warn"><b>A calculator and practice tool, not tips.</b> About 9 in 10 individual F&amp;O traders lost money in
    FY24 and FY25 (SEBI). Buying an option needs the right direction, a big enough move <i>and</i> fast enough timing, because the
    premium shrinks every minute. The backtest below shows how buying options on this dashboard's own signals would have done.</div>
  <div class="grid">
    <div>
      <div class="card"><h2>Build a trade</h2>
        <div class="form" style="grid-template-columns:repeat(auto-fit,minmax(130px,1fr))">
          <label>Underlying<select id="olSym"></select></label>
          <label>Expiry<select id="olExp"></select></label>
          <label>Call / put<select id="olCp"><option value="CE">Call (CE)</option><option value="PE">Put (PE)</option></select></label>
          <label>Buy / sell<select id="olSide"><option value="1">Buy</option><option value="-1">Sell (write)</option></select></label>
          <label>Strike<select id="olK"></select></label>
          <label>Lots<input id="olLots" type="number" min="1" step="1" value="1"></label>
          <label>Price now (₹)<input id="olSpot" type="number" step="0.05"></label>
          <label>Implied vol (%)<input id="olIV" type="number" step="0.1"></label>
          <label>Hold for (hours)<input id="olHold" type="number" min="0" step="0.25" value="1"></label>
        </div>
        <div id="olOut" class="out"></div>
        <div class="row" style="margin-top:8px"><span class="note" id="olNote"></span><button class="primary" id="olPaper">Paper-trade this</button></div>
      </div>
      <div class="card"><div class="row"><h2>Profit / loss</h2><span class="note">solid: on expiry day · dashed: after your holding time · after charges</span></div>
        <div id="olPay"></div></div>
    </div>
    <aside>
      <div class="card"><div class="row"><h2>Option chain</h2><span class="note" id="olChainNote"></span></div>
        <div class="scroll tall"><table id="olChain"></table></div>
        <p class="note">Closing prices from NSE's end-of-day file. Click a price to load that option.</p></div>
    </aside>
  </div>
  <div class="card"><div class="row"><h2>My paper trades</h2><span class="note" id="olPaperNote">Saved in this browser only</span></div>
    <div class="scroll"><table id="olPaperT"></table></div></div>
  <div class="card"><div class="row"><h2>Backtest: buying options on this dashboard's signals</h2><span class="note" id="olBtNote"></span></div>
    <div class="scroll"><table id="olBt"></table></div><p class="note" id="olBtText"></p></div>
</section>

<section id="scanner" class="hidden">
  <div class="card">
    <div class="row"><h2>Watchlist scanner</h2><div><span class="note" id="scUpd"></span> <button id="scGo" class="primary">Scan now</button></div></div>
    <p class="note">Bias score and live strategy states for liquid F&amp;O stocks and indices. Click a row to open its chart. <span id="scanTime">Takes ~20-40 seconds.</span></p>
    <div class="scroll"><table id="scT"></table></div>
  </div>
</section>

<p class="note">Data: Yahoo Finance intraday (may lag NSE slightly). Charts: TradingView Lightweight Charts. Backtests assume next-candle-open entries, stop checked before target inside a candle, and 3:15 PM square-off, with costs. Past results don't predict future results. Educational tool, not investment advice.</p>
</main>"""

JS = r"""
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (x, d=2) => x == null || isNaN(x) ? '—' : Number(x).toLocaleString('en-IN', {minimumFractionDigits:d, maximumFractionDigits:d});
const rs = (x, d=0) => x == null ? '—' : (x < 0 ? '-₹' : '₹') + fmt(Math.abs(x), d);
const sgn = (x, d=2, suf='') => x == null ? '—' : (x > 0 ? '+' : '') + fmt(x, d) + suf;
const cls = x => x > 0 ? 'up' : x < 0 ? 'dn' : '';
const store = {get(k, d){try{return localStorage.getItem('idesk_'+k) ?? d}catch(e){return d}}, set(k, v){try{localStorage.setItem('idesk_'+k, v)}catch(e){}}};
const STATIC = window.SNAP_INDEX || null;
const S = {symbol: store.get('sym','RELIANCE'), iv: store.get('iv','5m'), strat: store.get('strat','orb'), D: null, busy: false};
$('#sym').value = S.symbol; $('#iv').value = S.iv;
$('#cap').value = store.get('cap', 100000); $('#slip').value = store.get('slip', 0.02);

function showErr(m){ const b = $('#err'); b.textContent = m; b.style.display = m ? 'block' : 'none'; }
if (!window.LightweightCharts) showErr('Chart library failed to load (needs internet access to unpkg.com).');
const LWC = window.LightweightCharts;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const C = {up: css('--up'), dn: css('--dn'), acc: css('--acc'), blue: css('--blue'), violet: css('--violet'), mute: css('--mute'), line: css('--line'), grid: css('--grid')};
const hexA = (h, a) => { const n = parseInt(h.slice(1), 16); return `rgba(${n>>16&255},${n>>8&255},${n&255},${a})`; };
const opts = () => ({autoSize: true, layout: {background: {type: 'solid', color: 'transparent'}, textColor: C.mute, fontFamily: 'inherit'},
  grid: {vertLines: {color: C.grid}, horzLines: {color: C.grid}}, rightPriceScale: {borderColor: C.line},
  timeScale: {borderColor: C.line, timeVisible: true, secondsVisible: false, rightOffset: 4},
  crosshair: {mode: LWC.CrosshairMode.Normal}, localization: {priceFormatter: p => fmt(p, 2)}});

const main = LWC.createChart($('#chart'), opts());
const candle = main.addCandlestickSeries({upColor: C.up, downColor: C.dn, borderVisible: false, wickUpColor: C.up, wickDownColor: C.dn});
candle.priceScale().applyOptions({scaleMargins: {top: .06, bottom: .2}});
const volS = main.addHistogramSeries({priceFormat: {type: 'volume'}, priceScaleId: 'vol', lastValueVisible: false, priceLineVisible: false});
main.priceScale('vol').applyOptions({scaleMargins: {top: .84, bottom: 0}});
const line = (color, w=1.6, style=0) => main.addLineSeries({color, lineWidth: w, lineStyle: style, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false});
const vwapS = line(C.acc, 2), ema9S = line(C.blue, 1.2), ema21S = line(C.violet, 1.2);
const stUp = line(C.up, 1, 2), stDn = line(C.dn, 1, 2);
const rsiC = LWC.createChart($('#rsi'), opts());
const rsiS = rsiC.addLineSeries({color: C.blue, lineWidth: 1.4, priceLineVisible: false});
[70, 30].forEach(p => rsiS.createPriceLine({price: p, color: p > 50 ? C.dn : C.up, lineWidth: 1, lineStyle: 2, axisLabelVisible: false}));
const eqC = LWC.createChart($('#eq'), opts());
const eqS = eqC.addBaselineSeries({baseValue: {type: 'price', price: 0}, topLineColor: C.up, bottomLineColor: C.dn,
  topFillColor1: hexA(C.up, .25), topFillColor2: hexA(C.up, .02), bottomFillColor1: hexA(C.dn, .02), bottomFillColor2: hexA(C.dn, .25), lineWidth: 2});
let syncing = false;
const link = (a, b) => a.timeScale().subscribeVisibleLogicalRangeChange(r => { if (syncing || !r) return; syncing = true; b.timeScale().setVisibleLogicalRange(r); syncing = false; });
link(main, rsiC); link(rsiC, main);
let priceLines = [];

const ws = (T, a) => a.map((v, i) => v == null ? {time: T[i]} : {time: T[i], value: v});

function drawLevels(D){
  priceLines.forEach(p => candle.removePriceLine(p)); priceLines = [];
  if (!$('#showLv').checked) return;
  const L = D.levels, add = (p, c, t, st=2) => { if (p != null) priceLines.push(candle.createPriceLine({price: p, color: c, lineWidth: 1, lineStyle: st, axisLabelVisible: true, title: t})); };
  add(L.OR_H, C.blue, 'OR H', 0); add(L.OR_L, C.blue, 'OR L', 0);
  add(L.PDH, C.mute, 'PDH'); add(L.PDL, C.mute, 'PDL');
  add(L.TC, C.violet, 'TC', 1); add(L.BC, C.violet, 'BC', 1);
  add(L.R1, C.dn, 'R1', 3); add(L.S1, C.up, 'S1', 3);
}

function drawMarkers(D){
  const st = D.strategies[S.strat];
  $('#mkName').textContent = $('#showMk').checked ? `Trades shown: ${st.name}` : '';
  if (!$('#showMk').checked) { candle.setMarkers([]); return; }
  const m = [];
  const all = st.trades.concat(st.open ? [st.open] : []);
  const t0 = D.t[0];
  for (const t of all) {
    if (t.entry_ts < t0) continue;
    m.push({time: t.entry_ts, position: t.side > 0 ? 'belowBar' : 'aboveBar', color: t.side > 0 ? C.up : C.dn,
            shape: t.side > 0 ? 'arrowUp' : 'arrowDown', text: t.side > 0 ? 'B' : 'S'});
    if (t.exit_ts) m.push({time: t.exit_ts, position: t.side > 0 ? 'aboveBar' : 'belowBar', color: t.net > 0 ? C.up : C.dn,
            shape: 'circle', text: (t.net > 0 ? '+' : '') + Math.round(t.net)});
  }
  m.sort((a, b) => a.time - b.time);
  candle.setMarkers(m);
}

function render(D, keep){
  const T = D.t, range = keep ? main.timeScale().getVisibleLogicalRange() : null;
  candle.setData(T.map((t, i) => ({time: t, open: D.o[i], high: D.h[i], low: D.l[i], close: D.c[i]})));
  volS.setData(T.map((t, i) => ({time: t, value: D.v[i] || 0, color: D.c[i] >= D.o[i] ? hexA(C.up, .35) : hexA(C.dn, .35)})));
  vwapS.setData(ws(T, D.vwap)); ema9S.setData(ws(T, D.ema9)); ema21S.setData(ws(T, D.ema21));
  stUp.setData(ws(T, D.st_up)); stDn.setData(ws(T, D.st_dn)); rsiS.setData(ws(T, D.rsi));
  drawLevels(D); drawMarkers(D);
  if (range) main.timeScale().setVisibleLogicalRange(range);
  else main.timeScale().setVisibleLogicalRange({from: T.length - D.bars_per_day * 1.7, to: T.length + 3});

  $('#qSym').textContent = D.symbol;
  $('#qLtp').textContent = '₹' + fmt(D.ltp);
  $('#qChg').innerHTML = D.chg_pct == null ? '' : `<span class="${cls(D.chg_pct)}" style="font-weight:650">${sgn(D.chg_pct, 2, '%')}</span>`;
  $('#qUpd').textContent = `Session ${D.session_date} · last candle ${D.last_bar}${D.forming ? ' (forming)' : ''} · updated ${D.updated.slice(-8)} IST` + (D.has_volume ? '' : ' · index: VWAP uses price only');
  $('#tv').href = 'https://in.tradingview.com/chart/?symbol=' + encodeURIComponent(D.tv_symbol);
  renderBias(D); renderSignals(D); renderLevels(D); renderBacktest(D);
}

function renderBias(D){
  const b = D.bias;
  $('#bScore').textContent = sgn(b.score, 0); $('#bScore').className = 'score ' + cls(b.score);
  $('#bLabel').textContent = b.label; $('#bMark').style.left = (50 + b.score / 2) + '%';
  $('#bItems').innerHTML = b.items.map(i => `<li class="${i.state}">${esc(i.text)}</li>`).join('');
}

function renderSignals(D){
  $('#sigs').innerHTML = Object.entries(D.strategies).map(([k, s]) => {
    const st = s.status, long = st.state.includes('LONG'), short = st.state.includes('SHORT');
    const tg = `<span class="tag ${long ? 'long' : short ? 'short' : ''}">${st.state}</span>`;
    let body = '';
    if (st.entry != null) {
      body = `<div class="kv"><div><span>${st.active ? 'Entry' : 'Entry ≈'}</span>₹${fmt(st.entry)}</div><div><span>Stop-loss</span>₹${fmt(st.stop)}</div><div><span>Target</span>₹${fmt(st.target)}</div></div>
      <div class="row small" style="margin-top:6px"><span class="mute">${st.active ? 'Since ' + esc(st.since) : 'Signal at ' + esc(st.since)}</span>
      ${st.active ? `<span class="${cls(st.unreal)}">${rs(st.unreal)} open P&amp;L</span>` : ''}
      <button class="ghost small" data-size="${k}">Size it</button></div>`;
    } else {
      body = `<div class="small mute" style="margin-top:6px">No setup on the last candle.${st.today_trades ? ` ${st.today_trades} trade(s) this session: <span class="${cls(st.today_net)}">${rs(st.today_net)}</span>` : ''}</div>`;
    }
    const w = s.stats;
    return `<div class="sig"><div class="row"><b>${esc(s.name)}</b>${tg}</div>${body}
      <div class="small mute" style="margin-top:6px">Backtest here: ${w.trades} trades · ${fmt(w.win_pct, 1)}% win · <span class="${cls(w.net)}">${rs(w.net)}</span></div></div>`;
  }).join('');
  document.querySelectorAll('[data-size]').forEach(b => b.onclick = () => {
    const st = S.D.strategies[b.dataset.size].status;
    $('#pEntry').value = st.entry; $('#pStop').value = st.stop; calcPos();
    $('#pEntry').scrollIntoView({behavior: 'smooth', block: 'center'});
  });
}

function renderLevels(D){
  const L = D.levels, names = {R2:'R2', R1:'R1', PDH:'Prev day high', TC:'CPR top', P:'Pivot', BC:'CPR bottom', PDC:'Prev close', PDL:'Prev day low', S1:'S1', S2:'S2', OR_H:'Opening range high', OR_L:'Opening range low'};
  const rows = Object.keys(names).filter(k => L[k] != null).map(k => ({k, n: names[k], p: L[k]}));
  rows.push({k: 'LTP', n: 'Last price', p: D.ltp});
  rows.sort((a, b) => b.p - a.p);
  $('#lv').innerHTML = '<tr><th>Level</th><th>Price</th><th>Dist.</th></tr>' + rows.map(r => r.k === 'LTP'
    ? `<tr><td><b>▶ Last price</b></td><td><b>${fmt(r.p)}</b></td><td></td></tr>`
    : `<tr><td>${r.n}</td><td>${fmt(r.p)}</td><td class="${cls(r.p - D.ltp)}">${sgn((r.p / D.ltp - 1) * 100, 2, '%')}</td></tr>`).join('');
  const w = D.cpr_width;
  $('#cprNote').textContent = w == null ? '' : `CPR width ${fmt(w, 3)}% — ` + (w < 0.25 ? 'narrow: trending day more likely.' : w > 0.5 ? 'wide: range-bound day more likely.' : 'average.') + ` ATR ₹${fmt(D.atr)}.`;
}

function renderBacktest(D){
  const ent = Object.entries(D.strategies), any = ent[0][1].stats;
  $('#btNote').textContent = `${any.days || 0} sessions · ${D.interval} candles · ₹${fmt(D.capital, 0)} per trade · slippage ${D.slip}%/side`;
  $('#cmp').innerHTML = '<tr><th>Strategy</th><th>Trades</th><th>Win %</th><th>Loss %</th><th>Profit factor</th><th>Before costs</th><th>Net P&amp;L</th><th>Return</th><th>Max DD</th><th>Green days</th></tr>' +
    ent.map(([k, s]) => { const w = s.stats; return `<tr class="click ${k === S.strat ? 'sel' : ''}" data-k="${k}"><td><b>${esc(s.name)}</b></td><td>${w.trades}</td>
    <td class="up">${fmt(w.win_pct, 1)}%</td><td class="dn">${fmt(w.loss_pct, 1)}%</td><td>${w.pf == null ? (w.wins ? '∞' : '—') : fmt(w.pf)}</td>
    <td class="${cls(w.gross)}">${rs(w.gross)}</td><td class="${cls(w.net)}"><b>${rs(w.net)}</b></td><td class="${cls(w.ret_pct)}">${sgn(w.ret_pct, 2, '%')}</td><td class="dn">${rs(w.max_dd)}</td><td>${fmt(w.green_days, 0)}%</td></tr>`; }).join('');
  document.querySelectorAll('#cmp tr[data-k]').forEach(r => r.onclick = () => { S.strat = r.dataset.k; store.set('strat', S.strat); renderBacktest(S.D); drawMarkers(S.D); });

  const s = D.strategies[S.strat], w = s.stats;
  const ex = Object.entries(w.exits).map(([k, v]) => `${esc(k)} ${v}`).join(' · ');
  $('#detail').innerHTML = `<p class="small" style="margin:12px 0 0"><b>${esc(s.name)}.</b> <span class="mute">${esc(s.desc)}</span></p>
   <div class="tiles">
    <div class="tile"><div class="k">Win rate</div><div class="v up">${fmt(w.win_pct, 1)}%</div><div class="small mute">${w.wins} wins</div></div>
    <div class="tile"><div class="k">Loss rate</div><div class="v dn">${fmt(w.loss_pct, 1)}%</div><div class="small mute">${w.losses} losses</div></div>
    <div class="tile"><div class="k">Net after costs</div><div class="v ${cls(w.net)}">${rs(w.net)}</div><div class="small mute">gross ${rs(w.gross)} · costs ${rs(w.charges)}</div></div>
    <div class="tile"><div class="k">Avg win / loss</div><div class="v">${rs(w.avg_win)} / ${rs(w.avg_loss)}</div><div class="small mute">avg ${sgn(w.avg_r, 2)}R per trade</div></div>
    <div class="tile"><div class="k">Expectancy</div><div class="v ${cls(w.expectancy)}">${rs(w.expectancy)}</div><div class="small mute">per trade</div></div>
    <div class="tile"><div class="k">Longs / shorts win</div><div class="v">${w.long_win == null ? '—' : fmt(w.long_win, 0) + '%'} / ${w.short_win == null ? '—' : fmt(w.short_win, 0) + '%'}</div><div class="small mute">${w.long_n} L · ${w.short_n} S</div></div>
    <div class="tile"><div class="k">Best / worst day</div><div class="v">${rs(w.best_day)} / ${rs(w.worst_day)}</div><div class="small mute">${fmt(w.green_days, 0)}% green days</div></div>
   </div><p class="note">Exits: ${ex || '—'}</p>`;
  let cum = 0;
  eqS.setData(s.trades.map(t => ({time: t.exit_ts, value: Math.round(cum += t.net)})));
  eqC.timeScale().fitContent();
  $('#trN').textContent = `(${s.trades.length}, newest first)`;
  $('#trades').innerHTML = '<tr><th>Entry</th><th>Side</th><th>Entry ₹</th><th>Exit</th><th>Exit ₹</th><th>Qty</th><th>R</th><th>Costs</th><th>Net</th><th>Reason</th></tr>' +
    s.trades.slice().reverse().map(t => `<tr><td>${t.entry_time}</td><td class="${t.side > 0 ? 'up' : 'dn'}">${t.side > 0 ? 'LONG' : 'SHORT'}</td><td>${fmt(t.entry)}</td>
    <td>${t.exit_time.slice(-5)}</td><td>${fmt(t.exit)}</td><td>${t.qty}</td><td class="${cls(t.r)}">${sgn(t.r, 2)}</td><td>${rs(t.charges)}</td>
    <td class="${cls(t.net)}"><b>${rs(t.net)}</b></td><td class="mute">${t.reason}</td></tr>`).join('');
}

async function load(keep=false){
  if (S.busy) return; S.busy = true; $('#desk').classList.add('loading');
  const sym = ($('#sym').value || 'RELIANCE').trim().toUpperCase(), iv = $('#iv').value;
  const sameView = keep && S.D && S.D.symbol === sym && S.D.interval === iv;
  try {
    let D;
    if (STATIC) {
      const f = STATIC.files[sym];
      if (!f) throw new Error(`${sym} isn't in this snapshot. Pick one of the ${STATIC.symbols.length} symbols in the list (or open the Scanner).`);
      const r = await fetch(f);
      if (!r.ok) throw new Error('Could not load the data for ' + sym + '.');
      D = await r.json();
    } else {
      const q = new URLSearchParams({symbol: sym, interval: iv, capital: $('#cap').value || 100000, slip: $('#slip').value || 0.02});
      const r = await fetch('/api/analyze?' + q);
      D = await r.json();
      if (!r.ok) throw new Error(D.error || 'Request failed');
    }
    S.D = D; S.symbol = sym; S.iv = iv; store.set('sym', sym); store.set('iv', iv);
    showErr(''); render(D, sameView);
  } catch (e) { showErr(e.message.includes('fetch') ? 'Cannot reach the local server. Is intraday_dashboard.py still running?' : e.message); }
  finally { S.busy = false; $('#desk').classList.remove('loading'); }
}

// Scanner
async function runScan(){
  $('#scGo').disabled = true; $('#scGo').textContent = 'Scanning…';
  try {
    const q = new URLSearchParams({interval: $('#iv').value === '1m' ? '1m' : $('#iv').value, capital: $('#cap').value, slip: $('#slip').value});
    const r = await fetch(STATIC ? STATIC.scan : '/api/scan?' + q), D = await r.json();
    if (!r.ok) throw new Error(D.error || 'Scan failed');
    const ok = D.rows.filter(x => !x.error).sort((a, b) => Math.abs(b.bias) - Math.abs(a.bias));
    const tg = st => `<span class="tag ${st.state.includes('LONG') ? 'long' : st.state.includes('SHORT') ? 'short' : ''}">${st.state}</span>`;
    $('#scT').innerHTML = '<tr><th>Symbol</th><th>LTP</th><th>Chg</th><th>Bias</th><th>vs VWAP</th><th>RSI</th><th>ORB</th><th>Pullback</th><th>Supertrend</th></tr>' +
      ok.map(x => `<tr class="click" data-s="${esc(x.symbol)}"><td><b>${esc(x.symbol)}</b></td><td>${fmt(x.ltp)}</td><td class="${cls(x.chg)}">${sgn(x.chg, 2, '%')}</td>
      <td class="${cls(x.bias)}"><b>${sgn(x.bias, 0)}</b> <span class="small">${esc(x.label)}</span></td><td class="${cls(x.vwap)}">${sgn(x.vwap, 2, '%')}</td><td>${fmt(x.rsi, 0)}</td>
      <td>${tg(x.signals.orb)}</td><td>${tg(x.signals.pullback)}</td><td>${tg(x.signals.supertrend)}</td></tr>`).join('') +
      D.rows.filter(x => x.error).map(x => `<tr><td>${esc(x.symbol)}</td><td colspan="8" class="mute" style="text-align:left">${esc(x.error)}</td></tr>`).join('');
    $('#scUpd').textContent = 'Updated ' + D.updated + ' IST';
    document.querySelectorAll('#scT tr[data-s]').forEach(r => r.onclick = () => { $('#sym').value = r.dataset.s; tab('desk'); load(); });
  } catch (e) { showErr(e.message); }
  finally { $('#scGo').disabled = false; $('#scGo').textContent = STATIC ? 'Show list' : 'Scan now'; }
}
let PLAN = null;
const sideTag = s => s > 0 ? '<span class="tag long">LONG</span>' : s < 0 ? '<span class="tag short">SHORT</span>' : '<span class="tag">EITHER</span>';
const pbar = c => `<div class="pbar" title="Range ${fmt(c.lo, 0)}-${fmt(c.hi, 0)}%"><i></i><span style="left:${c.lo}%;width:${Math.max(1, c.hi - c.lo)}%"></span><b style="left:${c.prob}%"></b></div>`;
function lvChips(c){
  const L = c.levels, k = c.side >= 0 ? [['PDH','Prev high'],['R1','R1'],['TC','CPR top'],['BC','CPR bottom'],['PDL','Prev low'],['S1','S1']]
                                     : [['PDL','Prev low'],['S1','S1'],['BC','CPR bottom'],['TC','CPR top'],['PDH','Prev high'],['R1','R1']];
  return '<div class="lvchips">' + k.filter(([x]) => L[x] != null).map(([x, n]) => `<span>${n} <b>${fmt(L[x])}</b></span>`).join('') + '</div>';
}
function pickCard(c, i){
  const hero = i === 0, st = c.status;
  const liveTxt = c.live ? `<div class="small"><span class="tag ${st.state.includes('LONG') ? 'long' : 'short'}">${st.state}</span> entry ₹${fmt(st.entry)} · stop ₹${fmt(st.stop)} · target ₹${fmt(st.target)}</div>` : '';
  return `<div class="pick ${hero ? 'hero' : ''}">
    <div class="row"><span class="rank">${hero ? 'Top setup by expected profit' : '#' + (i + 1)}</span>${sideTag(c.side)}</div>
    <div class="row"><div><span class="sym">${esc(c.symbol)}</span> <span class="mute small">₹${fmt(c.ltp)} <span class="${cls(c.chg)}">${sgn(c.chg, 2, '%')}</span></span></div></div>
    <div><b>${esc(c.setup)}</b> <span class="mute small">· bias ${sgn(c.bias, 0)} ${esc(c.label)}</span></div>
    <div class="probrow"><div class="probbig ${c.prob >= 50 ? 'up' : ''}">${fmt(c.prob, 0)}%</div>
      <div class="small">probability of profit<br><span class="mute">range ${fmt(c.lo, 0)}-${fmt(c.hi, 0)}% · ${c.n} past trades on this stock</span></div></div>
    ${pbar(c)}
    <div class="kv"><div><span>Expected / trade</span><b class="${cls(c.ev)}">${rs(c.ev)}</b></div><div><span>Typical stop</span>₹${fmt(c.risk)} (${fmt(c.risk_pct)}%)</div><div><span>Target (2R)</span>₹${fmt(2 * c.risk)}</div></div>
    ${liveTxt}
    ${hero ? `<p class="trigger"><b>Trigger.</b> ${esc(c.trigger)}</p>${lvChips(c)}` : ''}
    <div class="row small"><span class="mute">~${c.qty} shares at ₹${fmt(PLAN.capital, 0)} per trade</span><button class="ghost small" data-open="${esc(c.symbol)}" data-k="${c.strategy}">Open chart</button></div>
  </div>`;
}
function renderPlan(P){
  PLAN = P;
  $('#plTitle').textContent = P.session_label;
  $('#plSub').textContent = `${P.symbols} symbols · data up to ${P.data_until}` + (P.live_session ? ' · live session' : ' · built before the open');
  const top = P.top[0], v = $('#plVerdict');
  if (top && top.ev > 0) {
    v.className = 'verdict ' + (top.prob >= 50 ? 'ok' : 'warn');
    v.innerHTML = `<b>${P.positive} of ${P.symbols} stocks have a setup with positive expected profit after costs.</b> The best is ${esc(top.setup)} ${top.side > 0 ? 'long' : top.side < 0 ? 'short' : ''} on ${esc(top.symbol)}: about ${rs(top.ev)} per trade with a ${fmt(top.prob, 0)}% chance of profit (range ${fmt(top.lo, 0)}-${fmt(top.hi, 0)}%). ` +
      (top.lo < 50 && top.ev < 100 ? 'The edge is small and the range is wide, so treat it as a candidate to watch, not a sure thing.' : '');
  } else {
    v.className = 'verdict warn';
    v.innerHTML = `<b>No setup has a positive expected profit after costs on any of the ${P.symbols} stocks right now.</b> The best trade may be no trade. The list below shows the least-bad options if you still want to watch something.`;
  }
  $('#plHero').innerHTML = top ? pickCard(top, 0) : '';
  $('#plRunners').innerHTML = P.top.slice(1).map((c, i) => pickCard(c, i + 1)).join('');
  $('#plTable').innerHTML = '<tr><th>#</th><th>Symbol</th><th>Best setup</th><th>Side</th><th>Bias</th><th>Probability of profit</th><th>Range</th><th>Expected / trade</th><th>Trades</th><th>Now</th></tr>' +
    P.best.map((c, i) => `<tr class="click" data-open="${esc(c.symbol)}" data-k="${c.strategy}"><td>${i + 1}</td><td><b>${esc(c.symbol)}</b></td><td>${esc(c.setup)}</td>
      <td>${sideTag(c.side)}</td><td class="${cls(c.bias)}">${sgn(c.bias, 0)}</td><td>${fmt(c.prob, 0)}%<span class="minibar"><span style="width:${c.prob}%"></span></span></td>
      <td class="mute">${fmt(c.lo, 0)}-${fmt(c.hi, 0)}%</td><td class="${cls(c.ev)}"><b>${rs(c.ev)}</b></td><td>${c.n}</td><td>${c.live ? `<span class="tag ${c.status.state.includes('LONG') ? 'long' : 'short'}">${c.status.state}</span>` : '<span class="mute">wait</span>'}</td></tr>`).join('');
  $('#plEdge').innerHTML = '<tr><th>Setup</th><th>Trades</th><th>Win % (all)</th><th>Expected (all)</th><th>Longs win</th><th>Shorts win</th></tr>' +
    P.edge.map(e => `<tr><td><b>${esc(e.name)}</b></td><td>${e.all.n}</td><td>${fmt(e.all.win, 1)}%</td><td class="${cls(e.all.ev)}">${rs(e.all.ev)}</td>
      <td>${fmt(e.longs.win, 1)}% <span class="mute small">(${rs(e.longs.ev)})</span></td><td>${fmt(e.shorts.win, 1)}% <span class="mute small">(${rs(e.shorts.ev)})</span></td></tr>`).join('');
  document.querySelectorAll('#plan [data-open]').forEach(el => el.onclick = () => {
    $('#sym').value = el.dataset.open; S.strat = el.dataset.k; store.set('strat', S.strat); tab('desk'); load();
  });
}
async function loadPlan(){
  try {
    const q = new URLSearchParams({interval: '5m', capital: $('#cap').value || 100000, slip: $('#slip').value || 0.02});
    const r = await fetch(STATIC ? STATIC.plan : '/api/plan?' + q), P = await r.json();
    if (!r.ok) throw new Error(P.error || 'Could not build the plan');
    renderPlan(P);
  } catch (e) { $('#plVerdict').className = 'verdict warn'; $('#plVerdict').textContent = e.message.includes('fetch') ? 'Cannot reach the local server. Is intraday_dashboard.py still running?' : e.message; }
}
let INST = null;
const lakh = q => q >= 1e7 ? fmt(q / 1e7, 2) + ' cr' : q >= 1e5 ? fmt(q / 1e5, 2) + ' L' : fmt(q, 0);
function dealRows(rows){
  return '<tr><th>Date</th><th>Symbol</th><th>Client</th><th>Side</th><th>Quantity</th><th>Price</th><th>Value</th><th>Type</th></tr>' + (rows.length ? rows.map(d =>
    `<tr${d.watch ? ' class="click" data-open="' + esc(d.symbol) + '"' : ''}><td>${esc(d.date)}</td><td><b>${esc(d.symbol)}</b>${d.watch ? ' ★' : ''}<div class="small mute">${esc(d.company)}</div></td>
     <td style="text-align:left;white-space:normal;min-width:180px">${esc(d.client)}</td><td><span class="tag ${d.side === 'BUY' ? 'long' : 'short'}">${esc(d.side)}</span></td>
     <td>${lakh(d.qty)}<div class="small mute">${fmt(d.qty, 0)}</div></td><td>₹${fmt(d.price)}</td><td><b>₹${fmt(d.value_cr, 2)} cr</b></td>
     <td class="small ${d.institution ? '' : 'mute'}">${d.institution ? 'Likely institution' : 'Other'}</td></tr>`).join('')
    : '<tr><td colspan="8" class="mute" style="text-align:left">No deals reported for the latest session.</td></tr>');
}
function renderInst(I){
  INST = I;
  $('#inSub').textContent = `NSE data as on ${I.as_on || '—'} · fetched ${I.generated} IST`;
  const e = $('#inErr'); e.textContent = (I.errors || []).join(' · '); e.style.display = (I.errors || []).length ? 'block' : 'none';
  $('#inFlows').innerHTML = (I.fiidii || []).map(f => `<div class="tile"><div class="k">${esc(f.category)} net · ${esc(f.date)}</div>
     <div class="v ${cls(f.net)}">${f.net < 0 ? '-' : '+'}₹${fmt(Math.abs(f.net), 2)} cr</div><div class="small mute">bought ₹${fmt(f.buy, 0)} cr · sold ₹${fmt(f.sell, 0)} cr (cash market, all stocks)</div></div>`).join('')
     || '<div class="tile"><div class="k">FII / DII</div><div class="small mute">Not available right now.</div></div>';
  $('#inBlockN').textContent = `${I.block.length} trades · ${I.block.filter(d => d.institution).length} likely institutional`;
  $('#inBulkN').textContent = `${I.bulk.length} trades`;
  $('#inBlock').innerHTML = dealRows(I.block); $('#inBulk').innerHTML = dealRows(I.bulk);
  $('#inSpikes').innerHTML = '<tr><th>Time (IST)</th><th>Symbol</th><th>Shares in candle</th><th>× normal</th><th>Price</th><th>Value</th><th>Candle</th></tr>' + (I.spikes.length ? I.spikes.map(x =>
    `<tr class="click" data-open="${esc(x.symbol)}"><td>${esc(x.time)}</td><td><b>${esc(x.symbol)}</b></td><td>${lakh(x.qty)}<div class="small mute">${fmt(x.qty, 0)}</div></td>
     <td><b>${fmt(x.ratio, 1)}×</b></td><td>₹${fmt(x.price)}</td><td>₹${fmt(x.value_cr, 2)} cr</td>
     <td class="${x.dir > 0 ? 'up' : x.dir < 0 ? 'dn' : 'mute'}">${x.dir > 0 ? 'Green' : x.dir < 0 ? 'Red' : 'Flat'} ${sgn(x.move, 2, '%')}</td></tr>`).join('')
    : '<tr><td colspan="7" class="mute" style="text-align:left">No unusual volume found.</td></tr>');
  document.querySelectorAll('#inst [data-open]').forEach(el => el.onclick = () => { $('#sym').value = el.dataset.open; tab('desk'); load(); });
}
async function loadInst(){
  try {
    const r = await fetch(STATIC ? STATIC.inst : '/api/institutional'), I = await r.json();
    if (!r.ok) throw new Error(I.error || 'Could not load NSE data');
    renderInst(I);
  } catch (e) { const b = $('#inErr'); b.textContent = e.message.includes('fetch') ? 'Cannot reach the local server. Is intraday_dashboard.py still running?' : e.message; b.style.display = 'block'; }
}
let PICKS = null;
function renderPortfolio(){
  if (!PICKS) return;
  const amt = Math.max(0, +$('#pfAmt').value || 0), core = PICKS.core_pct / 100, n = PICKS.momentum.length || 1;
  const each = amt * (1 - core) / n;
  $('#pfPlan').innerHTML = `<div class="row"><span><b>Core: Nifty 50 index fund / ETF</b> (${fmt(PICKS.core_pct, 0)}%)</span><b>${rs(amt * core)}</b></div>
    <div class="row"><span><b>Satellite: momentum picks</b> (${fmt(100 - PICKS.core_pct, 0)}%, ${n} stocks)</span><b>${rs(amt * (1 - core))}</b></div>
    <div class="lvchips" style="margin-top:6px">${PICKS.momentum.map(r => `<span>${esc(r.symbol)} <b>${rs(each)}</b> ≈ ${r.price ? Math.floor(each / r.price) : 0} sh</span>`).join('')}</div>`;
}
function pickRows(rows){
  return '<tr><th>#</th><th>Symbol</th><th>Price</th><th>12-1m return</th><th>1y volatility</th></tr>' + rows.map(r =>
    `<tr class="click" data-open="${esc(r.symbol)}"><td>${r.rank}${r.kept ? ' <span class="tag" title="Kept by banding: still within the top ' + (PICKS ? PICKS.band : 20) + '">kept</span>' : ''}</td><td><b>${esc(r.symbol)}</b></td><td>₹${fmt(r.price)}</td>
     <td class="${cls(r.mom)}">${sgn(r.mom, 1, '%')}</td><td>${fmt(r.vol, 1)}%</td></tr>`).join('');
}
function journalCard(j){
  if (!j.periods.length) return `<div class="tile" style="margin-bottom:10px"><div class="k">${esc(j.name)}</div>
    <div class="small mute">No entries yet. The first picks are logged at the first update during market hours, at that moment's prices.</div></div>`;
  const cur = j.periods[j.periods.length - 1], past = j.periods.filter(p => !p.open).reverse();
  const live = cur.open ? `<p class="small" style="margin:8px 0 4px"><b>Open since ${esc(cur.opened)}</b> (ranked on ${esc(cur.formation)} closes):
      <span class="${cls(cur.net)}">${sgn(cur.net, 2, '%')}</span> after costs vs Nifty <span class="${cls(cur.nifty)}">${sgn(cur.nifty, 2, '%')}</span></p>
    <div class="scroll"><table><tr><th>Symbol</th><th>Entry</th><th>Now</th><th>Return</th></tr>${cur.picks.map(p =>
      `<tr><td><b>${esc(p.symbol)}</b></td><td>₹${fmt(p.entry)}</td><td>₹${fmt(p.now)}</td><td class="${cls(p.ret)}">${sgn(p.ret, 2, '%')}</td></tr>`).join('')}</table></div>` : '';
  const hist = past.length ? `<div class="scroll" style="margin-top:8px"><table><tr><th>Ranked on</th><th>Held</th><th>Before costs</th><th>After costs</th><th>Nifty</th><th>Beat Nifty?</th></tr>${past.map(p =>
      `<tr><td>${esc(p.formation)}</td><td class="mute">${esc(p.opened.slice(0, 10))} → ${esc(p.closed.slice(0, 10))}</td><td class="${cls(p.gross)}">${sgn(p.gross, 2, '%')}</td>
       <td class="${cls(p.net)}"><b>${sgn(p.net, 2, '%')}</b></td><td class="${cls(p.nifty)}">${sgn(p.nifty, 2, '%')}</td><td>${p.net > p.nifty ? '<span class="up">Yes</span>' : '<span class="dn">No</span>'}</td></tr>`).join('')}</table></div>` : '';
  return `<div class="tile" style="margin-bottom:10px"><div class="row"><div class="k">${esc(j.name)} · rebalanced ${esc(j.every)}</div><span class="small mute">since ${esc(j.started)}</span></div>
    <div class="v"><span class="${cls(j.cum_net)}">${sgn(j.cum_net, 2, '%')}</span> <span class="small mute">after costs, before tax · Nifty ${sgn(j.cum_nifty, 2, '%')}</span></div>${live}${hist}</div>`;
}
function renderPicks(P){
  PICKS = P;
  $('#pkSub').textContent = `Ranked on closes up to ${P.formed} · prices as of ${P.asof} · ${P.universe} stocks`;
  $('#pkMomRule').textContent = P.rules.momentum; $('#pkLowRule').textContent = P.rules.lowvol;
  $('#pkMomNext').textContent = `next re-rank after ${P.next.momentum} ends`; $('#pkLowNext').textContent = `next re-rank after ${P.next.lowvol} ends`;
  $('#pkMom').innerHTML = pickRows(P.momentum); $('#pkLow').innerHTML = pickRows(P.lowvol);
  $('#pkJNote').textContent = `Equal money in each pick · about ${fmt(P.cost_pct, 2)}% costs on the stocks that change · tax not included · paper only`;
  renderPortfolio();
  $('#pkJournal').innerHTML = journalCard(P.journal.momentum) + journalCard(P.journal.lowvol);
  document.querySelectorAll('#picks [data-open]').forEach(el => el.onclick = () => { $('#sym').value = el.dataset.open; tab('desk'); load(); });
}
async function loadPicks(){
  try {
    if (STATIC && !STATIC.picks) throw new Error('This snapshot was built before the Monthly picks tab existed.');
    const r = await fetch(STATIC ? STATIC.picks : '/api/picks'), P = await r.json();
    if (!r.ok) throw new Error(P.error || 'Could not load the picks');
    renderPicks(P);
  } catch (e) { $('#pkSub').textContent = e.message.includes('fetch') ? 'Cannot reach the local server. Is intraday_dashboard.py still running?' : e.message; }
}
// Options lab. Pricing and charges duplicate options_lab.py - change both together.
let OPT = null;
const erf = x => { const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a)); };
const Nc = x => 0.5 * (1 + erf(x / Math.SQRT2)), nd = x => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
function bs(S, K, T, v, cp, r){
  if (T <= 0 || v <= 0) return cp === 'CE' ? Math.max(0, S - K) : Math.max(0, K - S);
  const st = v * Math.sqrt(T), d1 = (Math.log(S / K) + (r + v * v / 2) * T) / st, d2 = d1 - st;
  return cp === 'CE' ? S * Nc(d1) - K * Math.exp(-r * T) * Nc(d2) : K * Math.exp(-r * T) * Nc(-d2) - S * Nc(-d1);
}
function grk(S, K, T, v, cp, r){
  T = Math.max(T, 1 / 525600);
  const st = v * Math.sqrt(T), d1 = (Math.log(S / K) + (r + v * v / 2) * T) / st, d2 = d1 - st, c = -S * nd(d1) * v / (2 * Math.sqrt(T));
  return {delta: cp === 'CE' ? Nc(d1) : Nc(d1) - 1, vega: S * nd(d1) * Math.sqrt(T) / 100,
          theta: (cp === 'CE' ? c - r * K * Math.exp(-r * T) * Nc(d2) : c + r * K * Math.exp(-r * T) * Nc(-d2)) / 365};
}
function optCharges(buyP, sellP){ const t = buyP + sellP, br = 40, stt = .001 * sellP, txn = .0003503 * t, sebi = t * 10 / 1e7, stamp = .00003 * buyP;
  return br + stt + txn + sebi + stamp + .18 * (br + txn + sebi); }
const istNow = () => new Date(Date.now() + 5.5 * 3600e3);                       // IST wall clock held in UTC fields
const yrsTo = (exp, from) => Math.max(0, (Date.parse(exp + 'T15:30:00Z') - from.getTime()) / 6e4 / 525600);
const istStr = () => istNow().toISOString().slice(0, 16).replace('T', ' ');
const olChainOf = () => OPT && OPT.chains[$('#olSym').value];
const olExpOf = () => { const c = olChainOf(); return c && c.expiries[+$('#olExp').value || 0]; };
const olRowOf = () => { const e = olExpOf(); return e && e.strikes.find(s => s.strike === +$('#olK').value); };
function olFillSym(){
  const c = olChainOf(); if (!c) return;
  $('#olExp').innerHTML = c.expiries.map((e, i) => `<option value="${i}">${esc(e.expiry)} (${fmt(e.days, 0)} days)</option>`).join('');
  $('#olSpot').value = c.spot_now ?? c.spot; olFillExp();
}
function olFillExp(){
  const e = olExpOf(); if (!e || !e.strikes.length) return;
  const S = +$('#olSpot').value, atm = e.strikes.reduce((a, b) => Math.abs(b.strike - S) < Math.abs(a.strike - S) ? b : a);
  $('#olK').innerHTML = e.strikes.map(s => `<option value="${s.strike}"${s === atm ? ' selected' : ''}>${fmt(s.strike, s.strike % 1 ? 2 : 0)}</option>`).join('');
  olFillIV(); olRenderChain();
}
function olFillIV(){ const r = olRowOf(), c = olChainOf(), cp = $('#olCp').value;
  $('#olIV').value = (r && r[cp] && r[cp].iv) || c.atm_iv || c.hv20 || 20; olCalc(); }
function olState(){
  const c = olChainOf(), e = olExpOf(); if (!c || !e) return null;
  const S = +$('#olSpot').value, K = +$('#olK').value, cp = $('#olCp').value, side = +$('#olSide').value, r = OPT.r;
  const lots = Math.max(1, Math.round(+$('#olLots').value || 1)), qty = c.lot * lots, v = Math.max(0.01, (+$('#olIV').value || 20) / 100);
  const hold = Math.max(0, +$('#olHold').value || 0), T = yrsTo(e.expiry, istNow()), T2 = Math.max(0, T - hold / 8760);
  return {c, e, S, K, cp, side, r, lots, qty, v, hold, T, T2, p: bs(S, K, T, v, cp, r)};
}
function olCalc(){
  const st = olState(); if (!st) return;
  const {c, e, S, K, cp, side, r, lots, qty, v, hold, T, T2, p} = st, g = grk(S, K, T, v, cp, r), row = olRowOf();
  const prem = p * qty, ch = optCharges(prem, prem), close = row && row[cp] ? row[cp].close : null;
  const fav = (side > 0) === (cp === 'CE') ? 1 : -1;                              // price direction that helps this position
  const pnlAt = x => side * (bs(S * (1 + fav * x), K, T2, v, cp, r) - p) * qty - ch;
  const flat = pnlAt(0);
  let be = null;
  if (flat >= 0) be = 0;
  else if (pnlAt(0.25) > 0) { let lo = 0, hi = 0.25; for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (pnlAt(m) > 0) hi = m; else lo = m; } be = hi * 100; }
  const typ = c.move_pct, rng = c.range_pct;
  const beTxt = be == null ? '<span class="dn">more than 25% — practically out of reach</span>'
    : be === 0 ? '<span class="up">none: time decay pays you while the price stays put</span>'
    : `<b>${fmt(be, 2)}%</b> (₹${fmt(S * be / 100)}) ${fav > 0 ? 'up' : 'down'}`;
  $('#olOut').innerHTML = `
    <div class="row"><span class="mute">Premium per share</span><span><b>₹${fmt(p)}</b> now (estimate)${close != null ? ` · ₹${fmt(close)} at NSE close ${esc(OPT.asof)}` : ''}</span></div>
    <div class="row"><span class="mute">${side > 0 ? 'You pay' : 'You receive'} (${lots} lot${lots > 1 ? 's' : ''} × ${c.lot})</span><b>${rs(prem)}</b></div>
    <div class="row"><span class="mute">Delta · theta · vega</span><span>${fmt(g.delta * side, 2)} · <span class="${cls(g.theta * side)}">${rs(g.theta * qty * side)}/day</span> · ${rs(g.vega * qty * side)} per vol point</span></div>
    <div class="row"><span class="mute">Charges, in and out</span><span>${rs(ch)}</span></div>
    <div class="row"><span class="mute">If price doesn't move for ${fmt(hold, 2)} h</span><span class="${cls(flat)}">${rs(flat)}</span></div>
    <div class="row"><span class="mute">Move needed in ${fmt(hold, 2)} h to break even</span><span>${beTxt}</span></div>
    <div class="row"><span class="mute">Typical day for ${esc(c.symbol)} (last 20)</span><span>${typ == null ? '—' : `open→close ${fmt(typ, 2)}% · high-low range ${fmt(rng, 2)}%`}</span></div>
    ${side < 0 ? '<div class="small dn" style="margin-top:6px"><b>Selling (writing) options:</b> the most you can earn is the premium, but losses can be many times larger if the price jumps. Brokers block a large margin (often ₹1 lakh+ per lot). This lab does not calculate margin.</div>' : ''}`;
  $('#olNote').textContent = `${fmt(T * 365, 2)} days to expiry · IV ${fmt(v * 100, 1)}% · 20-day realised vol ${c.hv20 == null ? '—' : fmt(c.hv20, 1) + '%'}`;
  olDrawPay(st, ch);
}
function olDrawPay(st, ch){
  const {S, K, cp, side, qty, v, T2, r, p} = st, W = 640, H = 240, pad = 44, n = 120, lo = S * 0.92, hi = S * 1.08;
  const xs = Array.from({length: n + 1}, (_, i) => lo + (hi - lo) * i / n);
  const yE = xs.map(x => side * ((cp === 'CE' ? Math.max(0, x - K) : Math.max(0, K - x)) - p) * qty - ch);
  const yH = xs.map(x => side * (bs(x, K, T2, v, cp, r) - p) * qty - ch);
  const all = yE.concat(yH), y0 = Math.min(0, ...all), y1 = Math.max(0, ...all), span = (y1 - y0) || 1;
  const X = x => pad + (x - lo) / (hi - lo) * (W - 2 * pad), Y = y => H - pad / 1.5 - (y - y0) / span * (H - pad * 1.4);
  const path = ys => xs.map((x, i) => `${i ? 'L' : 'M'}${X(x).toFixed(1)},${Y(ys[i]).toFixed(1)}`).join('');
  $('#olPay').innerHTML = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto" role="img" aria-label="Profit and loss chart">
    <line x1="${pad}" x2="${W - pad}" y1="${Y(0)}" y2="${Y(0)}" style="stroke:var(--line);stroke-width:1.5"/>
    <line x1="${X(S)}" x2="${X(S)}" y1="10" y2="${H - pad / 1.5}" style="stroke:var(--mute);stroke-dasharray:3 3"/>
    <line x1="${X(K)}" x2="${X(K)}" y1="10" y2="${H - pad / 1.5}" style="stroke:var(--violet);stroke-dasharray:1 4"/>
    <path d="${path(yE)}" style="fill:none;stroke:var(--blue);stroke-width:2.2"/>
    <path d="${path(yH)}" style="fill:none;stroke:var(--acc);stroke-width:2;stroke-dasharray:6 4"/>
    <text x="${pad - 6}" y="${Y(y1) + 4}" text-anchor="end" style="fill:var(--up);font-size:11px">${rs(y1)}</text>
    <text x="${pad - 6}" y="${Y(y0) + 4}" text-anchor="end" style="fill:var(--dn);font-size:11px">${rs(y0)}</text>
    <text x="${pad - 6}" y="${Y(0) + 4}" text-anchor="end" style="fill:var(--mute);font-size:11px">₹0</text>
    <text x="${X(lo)}" y="${H - 6}" style="fill:var(--mute);font-size:11px">${fmt(lo, 0)} (−8%)</text>
    <text x="${X(S)}" y="${H - 6}" text-anchor="middle" style="fill:var(--ink);font-size:11px">now ${fmt(S, 0)}</text>
    <text x="${X(hi)}" y="${H - 6}" text-anchor="end" style="fill:var(--mute);font-size:11px">${fmt(hi, 0)} (+8%)</text>
    <text x="${X(K) + 4}" y="20" style="fill:var(--violet);font-size:11px">strike ${fmt(K, 0)}</text></svg>`;
}
function olRenderChain(){
  const c = olChainOf(), e = olExpOf(); if (!c || !e) return;
  const S = +$('#olSpot').value || c.spot, atm = e.strikes.reduce((a, b) => Math.abs(b.strike - S) < Math.abs(a.strike - S) ? b : a);
  const cell = (s, cp) => s[cp] ? `<td class="click" data-k="${s.strike}" data-cp="${cp}"><b>${fmt(s[cp].close)}</b><div class="small mute">IV ${s[cp].iv == null ? '—' : fmt(s[cp].iv, 1)} · OI ${lakh(s[cp].oi)}</div></td>` : '<td class="mute">—</td>';
  $('#olChainNote').textContent = `${c.symbol} ${e.expiry} · lot ${c.lot}`;
  $('#olChain').innerHTML = '<tr><th>Call</th><th style="text-align:center">Strike</th><th>Put</th></tr>' + e.strikes.map(s =>
    `<tr${s === atm ? ' class="sel"' : ''}>${cell(s, 'CE')}<td style="text-align:center"><b>${fmt(s.strike, s.strike % 1 ? 2 : 0)}</b>${s === atm ? '<div class="small mute">at the money</div>' : ''}</td>${cell(s, 'PE')}</tr>`).join('');
  document.querySelectorAll('#olChain td[data-k]').forEach(td => td.onclick = () => { $('#olCp').value = td.dataset.cp; $('#olK').value = td.dataset.k; olFillIV(); });
}
const olPapers = () => { try { return JSON.parse(store.get('optpaper', '[]')) || []; } catch (e) { return []; } };
const olSavePapers = a => store.set('optpaper', JSON.stringify(a));
function olMark(t){
  if (t.closed) return t.exit;
  const c = OPT && OPT.chains[t.sym], S = c ? (c.spot_now ?? c.spot) : t.spot;
  return bs(S, t.K, yrsTo(t.exp, istNow()), t.iv, t.cp, OPT ? OPT.r : 0.065);
}
function olPnl(t, px){ const q = t.lot * t.lots, ent = t.entry * q, now = px * q;
  return t.side * (now - ent) - (t.side > 0 ? optCharges(ent, now) : optCharges(now, ent)); }
function olRenderPaper(){
  const a = olPapers();
  if (!a.length) { $('#olPaperT').innerHTML = '<tr><td class="mute" style="text-align:left">No paper trades yet. Build a trade above and press "Paper-trade this".</td></tr>'; return; }
  const closed = a.filter(t => t.closed), wins = closed.filter(t => olPnl(t, t.exit) > 0).length, tot = closed.reduce((s, t) => s + olPnl(t, t.exit), 0);
  $('#olPaperNote').textContent = `Saved in this browser only · ${closed.length} closed: ${wins} won, total ${rs(tot)} after charges · open trades valued at the model price`;
  $('#olPaperT').innerHTML = '<tr><th>Opened (IST)</th><th>Option</th><th>Side</th><th>Lots</th><th>Entry ₹</th><th>Now / exit ₹</th><th>P&amp;L after charges</th><th></th></tr>' +
    a.slice().reverse().map(t => { const px = olMark(t), pl = olPnl(t, px);
      return `<tr><td>${esc(t.opened)}</td><td><b>${esc(t.sym)} ${fmt(t.K, t.K % 1 ? 2 : 0)} ${t.cp}</b><div class="small mute">expiry ${esc(t.exp)}</div></td>
        <td class="${t.side > 0 ? 'up' : 'dn'}">${t.side > 0 ? 'BUY' : 'SELL'}</td><td>${t.lots}</td><td>${fmt(t.entry)}</td><td>${fmt(px)}${t.closed ? '<div class="small mute">closed ' + esc(t.closed) + '</div>' : ''}</td>
        <td class="${cls(pl)}"><b>${rs(pl)}</b></td><td>${t.closed ? '' : `<button class="ghost small" data-close="${t.id}">Close</button>`} <button class="ghost small" data-del="${t.id}">Remove</button></td></tr>`; }).join('');
  document.querySelectorAll('[data-close]').forEach(b => b.onclick = () => { const a2 = olPapers(), t = a2.find(x => x.id == b.dataset.close);
    if (t) { t.exit = olMark(t); t.closed = istStr(); olSavePapers(a2); olRenderPaper(); } });
  document.querySelectorAll('[data-del]').forEach(b => b.onclick = () => { olSavePapers(olPapers().filter(x => x.id != b.dataset.del)); olRenderPaper(); });
}
function olRenderBt(B){
  if (!B) { $('#olBtText').textContent = 'Run research/options_backtest.py to fill this table.'; return; }
  $('#olBtNote').textContent = `${B.sessions} sessions, ${B.period} · 1 lot, at-the-money, nearest expiry · bid-ask ${B.slippage_pct.index}% index / ${B.slippage_pct.stock}% stock per side`;
  const tr = (name, g, x) => x ? `<tr><td><b>${esc(name)}</b></td><td>${g}</td><td>${x.trades}</td><td>${fmt(x.win_net, 1)}%</td><td>${rs(x.premium)}</td>
    <td class="${cls(x.gross)}">${rs(x.gross)}</td><td>${rs(x.charges)}</td><td>${rs(x.slip)}</td><td class="${cls(x.net)}"><b>${rs(x.net)}</b></td><td class="${cls(x.net_pct_premium)}">${sgn(x.net_pct_premium, 1, '%')}</td></tr>` : '';
  $('#olBt').innerHTML = '<tr><th>Setup</th><th>Options on</th><th>Trades</th><th>Win (net)</th><th>Premium / lot</th><th>Before costs</th><th>Charges</th><th>Bid-ask</th><th>Net / trade</th><th>Net % of premium</th></tr>' +
    Object.values(B.setups).map(s => tr(s.name, 'indices', s.index) + tr(s.name, 'stocks', s.stock)).join('') + tr('All signals', 'all', B.all);
  const a = B.all, x = B.expiry_day;
  $('#olBtText').innerHTML = `Buying options on these signals lost about <b>${rs(-a.net)}</b> per trade after costs (${sgn(a.net_pct_premium, 1, '%')} of the premium paid), winning ${fmt(a.win_net, 0)}% of the time. ` +
    (x ? `On expiry days it was worse: ${sgn(x.net_pct_premium, 1, '%')} of premium. ` : '') +
    `The same signals traded in shares made ${rs(a.cash_gross)} before costs and ${rs(a.cash_net)} after, per ₹1 lakh. Options don't create an edge the signal doesn't have; they add time decay and wider spreads. Model estimate (Black-Scholes with each option's previous-day implied volatility), not actual fills.`;
}
async function loadOpt(){
  try {
    if (STATIC && !STATIC.options) throw new Error('This snapshot was built before the Options lab existed.');
    const r = await fetch(STATIC ? STATIC.options : '/api/options'), O = await r.json();
    if (!r.ok) throw new Error(O.error || 'Could not load option prices');
    OPT = O;
    const idx = O.symbols.filter(s => O.chains[s].index), rest = O.symbols.filter(s => !O.chains[s].index);
    $('#olSym').innerHTML = idx.concat(rest).map(s => `<option>${esc(s)}</option>`).join('');
    $('#olSym').value = O.chains.NIFTY ? 'NIFTY' : O.symbols[0];
    $('#olSub').textContent = `Option closes from NSE for ${O.asof} · underlying prices up to ${(O.chains[$('#olSym').value] || {}).spot_time || '—'} · ${O.symbols.length} underlyings`;
    olFillSym(); olRenderPaper(); olRenderBt(O.backtest);
  } catch (e) { $('#olSub').textContent = e.message.includes('fetch') ? 'Cannot reach the local server. Is intraday_dashboard.py still running?' : e.message; }
}
$('#olSym').onchange = olFillSym; $('#olExp').onchange = olFillExp; $('#olCp').onchange = olFillIV; $('#olK').onchange = olFillIV;
['#olSide', '#olLots', '#olSpot', '#olIV', '#olHold'].forEach(s => $(s).oninput = olCalc);
$('#olPaper').onclick = () => { const st = olState(); if (!st) return;
  const a = olPapers(); a.push({id: Date.now(), sym: st.c.symbol, exp: st.e.expiry, K: st.K, cp: st.cp, side: st.side, lots: st.lots, lot: st.c.lot,
    entry: Math.round(st.p * 100) / 100, iv: st.v, spot: st.S, opened: istStr()}); olSavePapers(a); olRenderPaper(); };

function tab(n){
  [['plan', '#plan', '#tabPlan'], ['desk', '#desk', '#tabDesk'], ['scan', '#scanner', '#tabScan'], ['inst', '#inst', '#tabInst'], ['picks', '#picks', '#tabPicks'], ['opt', '#opt', '#tabOpt']].forEach(([k, sec, btn]) => {
    $(sec).classList.toggle('hidden', n !== k); $(btn).classList.toggle('on', n === k); });
  store.set('tab', n);
  if (n === 'opt' && !OPT) loadOpt();
  if (n === 'picks' && !PICKS) loadPicks();
  if (n === 'scan' && !$('#scT').innerHTML) runScan();
  if (n === 'plan' && !PLAN) loadPlan();
  if (n === 'inst' && !INST) loadInst();
  if (n === 'desk' && S.D) requestAnimationFrame(() => render(S.D, false));
}
$('#tabOpt').onclick = () => tab('opt'); $('#tabPicks').onclick = () => tab('picks'); $('#pfAmt').oninput = renderPortfolio; $('#tabInst').onclick = () => tab('inst'); $('#tabPlan').onclick = () => tab('plan'); $('#tabDesk').onclick = () => tab('desk'); $('#tabScan').onclick = () => tab('scan'); $('#scGo').onclick = runScan;

// Calculators (same charge model as the backtest)
function charges(buyV, sellV){ const t = buyV + sellV, br = Math.min(20, .0003 * buyV) + Math.min(20, .0003 * sellV), stt = .00025 * sellV,
  txn = .0000307 * t, sebi = t * 10 / 1e7, stamp = .00003 * buyV, gst = .18 * (br + txn + sebi);
  return {br, stt, txn, sebi, stamp, gst, total: br + stt + txn + sebi + stamp + gst}; }
function calcPos(){
  const cap = +$('#pCap').value, risk = +$('#pRisk').value, e = +$('#pEntry').value, s = +$('#pStop').value, lev = +$('#pLev').value || 1, rr = +$('#pRR').value || 2;
  const per = Math.abs(e - s);
  if (!(e > 0 && s > 0 && per > 0)) { $('#pOut').textContent = 'Enter entry and stop-loss.'; return; }
  const long = s < e, qR = Math.floor(cap * risk / 100 / per), qM = Math.floor(cap * lev / e), q = Math.max(0, Math.min(qR, qM));
  const tgt = long ? e + rr * per : e - rr * per, ch = charges(e * q, tgt * q).total;
  $('#pOut').innerHTML = `<div class="row"><span>${long ? 'LONG' : 'SHORT'} quantity</span><b style="font-size:18px">${q}</b></div>
   <div class="row"><span class="mute">Risk if stopped</span><span class="dn">${rs(q * per)} (${fmt(q * per / cap * 100, 2)}%)</span></div>
   <div class="row"><span class="mute">Target (${rr}R)</span><span>₹${fmt(tgt)} → <span class="up">${rs(q * per * rr - ch)}</span> after costs</span></div>
   <div class="row"><span class="mute">Exposure / margin</span><span>${rs(q * e)} / ${rs(q * e / lev)}</span></div>
   ${qM < qR ? '<div class="small dn" style="margin-top:4px">Capped by leverage, not by risk.</div>' : ''}`;
}
function calcCh(){
  const side = +$('#cSide').value, q = +$('#cQty').value, e = +$('#cEntry').value, x = +$('#cExit').value;
  if (!(q > 0 && e > 0)) { $('#cOut').textContent = 'Enter prices.'; return; }
  const ex = x > 0 ? x : e, buyV = (side > 0 ? e : ex) * q, sellV = (side > 0 ? ex : e) * q, c = charges(buyV, sellV);
  const gross = side * (ex - e) * q, be = c.total / q;
  $('#cOut').innerHTML = `<div class="row"><span class="mute">Brokerage</span><span>${rs(c.br, 2)}</span></div>
   <div class="row"><span class="mute">STT</span><span>${rs(c.stt, 2)}</span></div><div class="row"><span class="mute">Exchange + SEBI</span><span>${rs(c.txn + c.sebi, 2)}</span></div>
   <div class="row"><span class="mute">Stamp duty</span><span>${rs(c.stamp, 2)}</span></div><div class="row"><span class="mute">GST</span><span>${rs(c.gst, 2)}</span></div>
   <div class="row"><b>Total charges</b><b>${rs(c.total, 2)}</b></div>
   <div class="row"><span class="mute">Breakeven move</span><span>₹${fmt(be)} / share (${fmt(be / e * 100, 3)}%)</span></div>
   ${x > 0 ? `<div class="row"><b>Net P&amp;L</b><b class="${cls(gross - c.total)}">${rs(gross - c.total, 2)}</b></div>` : ''}`;
}
['#pCap','#pRisk','#pEntry','#pStop','#pLev','#pRR'].forEach(s => $(s).oninput = calcPos);
['#cSide','#cQty','#cEntry','#cExit'].forEach(s => $(s).oninput = calcCh);

// Clock & market status (IST)
function ist(){ const p = new Intl.DateTimeFormat('en-GB', {timeZone: 'Asia/Kolkata', hour12: false, weekday: 'short', hour: '2-digit', minute: '2-digit', second: '2-digit'}).formatToParts(new Date());
  const g = t => p.find(x => x.type === t).value; return {wd: g('weekday'), h: +g('hour'), m: +g('minute'), s: +g('second')}; }
function tick(){
  const t = ist(), mins = t.h * 60 + t.m, wk = !['Sat', 'Sun'].includes(t.wd);
  $('#clock').textContent = `${String(t.h).padStart(2, '0')}:${String(t.m).padStart(2, '0')}:${String(t.s).padStart(2, '0')} IST`;
  let txt = 'Market closed', open = false;
  if (wk && mins >= 540 && mins < 555) txt = 'Pre-open';
  else if (wk && mins >= 555 && mins < 930) { open = true;
    const left = 915 - mins; txt = left > 0 ? `Open · square-off in ${Math.floor(left / 60)}h ${left % 60}m` : 'Open · square off now'; }
  $('#mkt').classList.toggle('open', open); $('#mktTxt').textContent = txt;
  return open;
}
setInterval(tick, 1000); tick();
setInterval(() => { if (!STATIC && $('#auto').checked && tick() && !$('#desk').classList.contains('hidden')) load(true); }, 60000);

$('#go').onclick = () => load();
$('#sym').onkeydown = e => { if (e.key === 'Enter') load(); };
$('#iv').onchange = () => load();
$('#showLv').onchange = () => S.D && drawLevels(S.D);
$('#showMk').onchange = () => S.D && drawMarkers(S.D);
['#cap', '#slip'].forEach(s => $(s).onchange = () => { store.set(s.slice(1), $(s).value); load(true); });
(STATIC ? Promise.resolve(STATIC.symbols) : fetch('/api/watchlist').then(r => r.json()))
  .then(w => $('#wl').innerHTML = w.map(s => `<option value="${esc(s)}">`).join('')).catch(() => {});
if (STATIC) {
  $('#autoWrap').hidden = true; $('#iv').hidden = true; $('#cap').disabled = true; $('#slip').disabled = true;
  $('#scanTime').textContent = ''; $('#scGo').textContent = 'Show list';
  const n = $('#snap'); n.hidden = false;
  const age = STATIC.gen_ts ? Math.round((Date.now() / 1000 - STATIC.gen_ts) / 60) : null;
  n.innerHTML = (age != null && age <= 15 && tick()
    ? `<b>Auto-updating.</b> Rebuilt ${age < 1 ? 'just now' : age + ' min ago'} (${esc(STATIC.generated)} IST); a new copy arrives about every 5 minutes while the market is open. `
    : `<b>Snapshot, not live.</b> Last rebuilt ${esc(STATIC.generated)} IST. `) +
    `Candles and signals run up to <b>${esc(STATIC.data_until)}</b> for ${STATIC.symbols.length} symbols. Yahoo data can lag NSE by a few minutes.`;
  if (!STATIC.files[S.symbol]) { S.symbol = STATIC.symbols.find(x => x === 'RELIANCE') || STATIC.symbols[0]; $('#sym').value = S.symbol; }
}
load();
tab(store.get('tab', 'plan'));
setInterval(() => { if (!STATIC && PLAN && PLAN.live_session && !$('#plan').classList.contains('hidden') && tick()) { PLAN = null; loadPlan(); } }, 300000);
"""


def local_page() -> str:
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Intraday Desk</title>'
            + LIB + '<style>' + CSS + '</style></head><body>' + BODY + '<script>' + JS + '</script></body></html>')


def hosted_page(index: dict) -> str:
    """Snapshot page for publishing: the hosted skeleton adds doctype/head/body itself."""
    return ('<title>Intraday Desk</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<style>' + CSS + '</style>\n' + LIB + '\n' + BODY
            + '\n<script>window.SNAP_INDEX = ' + json.dumps(index) + ';</script>\n<script>' + JS + '</script>')


def standalone_page(index: dict) -> str:
    """Snapshot page for ordinary static hosts (GitHub Pages etc.): a complete HTML document."""
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>Intraday Desk</title>'
            + LIB + '<style>' + CSS + '</style></head><body>' + BODY
            + '<script>window.SNAP_INDEX = ' + json.dumps(index) + ';</script><script>' + JS + '</script></body></html>')


PAGE = local_page()


CHART_KEYS = ("t", "o", "h", "l", "c", "v", "vwap", "ema9", "ema21", "rsi", "st_up", "st_dn")


def build_snapshot(outdir: str, interval: str = "5m", sessions: int = 6, capital: float = 100000, slip: float = 0.02,
                   standalone: bool = False, journal: bool = False):
    """Write a self-contained snapshot (index.html + data/*.json) that works without this server."""
    import os
    os.makedirs(os.path.join(outdir, "data"), exist_ok=True)
    files, rows, last_bars = {}, [], []
    for sym in WATCHLIST:
        try:
            D = analyze(sym, interval, capital, slip)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {sym}: {e}")
            continue
        days = [t // 86400 for t in D["t"]]
        keep = sorted(set(days))[-sessions:]
        i0 = days.index(keep[0])
        for k in CHART_KEYS:
            D[k] = D[k][i0:]
        name = re.sub(r"[^A-Za-z0-9_-]", "_", sym) + ".json"
        with open(os.path.join(outdir, "data", name), "w", encoding="utf-8") as f:
            json.dump(D, f, separators=(",", ":"))
        files[sym] = "data/" + name
        last_vwap = next((x for x in reversed(D["vwap"]) if x is not None), None)
        rows.append(dict(symbol=sym, ltp=D["ltp"], chg=D["chg_pct"], bias=D["bias"]["score"], label=D["bias"]["label"],
                         vwap=num((D["ltp"] / last_vwap - 1) * 100) if last_vwap else None,
                         rsi=next((x for x in reversed(D["rsi"]) if x is not None), None),
                         signals={k: v["status"] for k, v in D["strategies"].items()}))
        last_bars.append(D["session_date"] + " " + D["last_bar"][-5:])
        print(f"  {sym:<12} ok")
    gen = now_ist().strftime("%d %b %Y %H:%M")
    with open(os.path.join(outdir, "data", "_scan.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, updated=gen, market_open=False, interval=interval), f, separators=(",", ":"))
    P = plan(interval, capital, slip)
    with open(os.path.join(outdir, "data", "_plan.json"), "w", encoding="utf-8") as f:
        json.dump(P, f, separators=(",", ":"))
    print(f"  plan: {P['symbols']} symbols, {P['positive']} with positive expected value")
    I = institutional(interval)
    with open(os.path.join(outdir, "data", "_inst.json"), "w", encoding="utf-8") as f:
        json.dump(I, f, separators=(",", ":"))
    print(f"  big money: {len(I['block'])} block, {len(I['bulk'])} bulk deals, {len(I['spikes'])} volume spikes")
    picks_file = None
    try:
        K = picks(write_journal=journal)
        with open(os.path.join(outdir, "data", "_picks.json"), "w", encoding="utf-8") as f:
            json.dump(K, f, separators=(",", ":"))
        picks_file = "data/_picks.json"
        print(f"  monthly picks: ranked on {K['formed']}, journal {'updated' if journal else 'read only'}")
    except Exception as e:  # noqa: BLE001
        print(f"  monthly picks skipped: {e}")
    options_file = None
    try:
        O = options_payload()
        with open(os.path.join(outdir, "data", "_options.json"), "w", encoding="utf-8") as f:
            json.dump(O, f, separators=(",", ":"))
        options_file = "data/_options.json"
        print(f"  options lab: {len(O['chains'])} option chains from NSE close of {O['asof']}")
    except Exception as e:  # noqa: BLE001
        print(f"  options lab skipped: {e}")
    latest = max(last_bars) if last_bars else ""
    try:
        until = dt.datetime.strptime(latest, "%Y-%m-%d %H:%M").strftime("%d %b %Y, %H:%M")
    except ValueError:
        until = latest
    index = dict(generated=gen, gen_ts=int(time.time()), data_until=until, symbols=list(files), files=files, scan="data/_scan.json",
                 plan="data/_plan.json", inst="data/_inst.json", picks=picks_file, options=options_file)
    with open(os.path.join(outdir, "index.html"), "w", encoding="utf-8") as f:
        f.write(standalone_page(index) if standalone else hosted_page(index))
    print(f"Snapshot written to {outdir} ({len(files)} symbols, data up to {until})")


def main():
    ap = argparse.ArgumentParser(description="Intraday Desk - local NSE intraday dashboard")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--snapshot", metavar="DIR", help="write a static snapshot page to DIR instead of serving")
    ap.add_argument("--standalone", action="store_true",
                    help="with --snapshot: write a complete HTML page for static hosts like GitHub Pages")
    ap.add_argument("--if-open", action="store_true",
                    help="with --snapshot: do nothing unless NSE is in session (09:15-15:40 IST, not a holiday)")
    ap.add_argument("--journal", action="store_true",
                    help="with --snapshot: log the monthly picks in journal/paper_journal.json when a rebalance is due")
    a = ap.parse_args()
    if a.snapshot:
        if a.if_open and not session_window(now_ist()):
            return print("Market closed - snapshot skipped.")
        return build_snapshot(a.snapshot, standalone=a.standalone, journal=a.journal)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}"
    print(f"Intraday Desk running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
