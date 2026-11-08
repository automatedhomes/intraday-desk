#!/usr/bin/env python3
"""
Indian Stock Analyzer  (NSE / BSE)
==================================
Scores a stock on three pillars and turns them into a BUY / SELL / HOLD call:

  1. Technicals    trend (SMA 50/200), MACD, RSI, Supertrend, ADX/DI, Bollinger, OBV & volume
  2. Fundamentals  P/E, forward P/E, P/B, PEG, ROE, debt/equity, current ratio, net margin,
                   revenue & earnings growth, analyst consensus target
  3. News/market   headline sentiment (Google News India + Yahoo Finance, recency weighted)
                   plus Nifty 50 trend and market-headline sentiment

Then it backtests the technical engine on years of daily data and reports:
  win % / loss %, average win/loss, profit factor, CAGR, max drawdown, vs buy & hold,
  and how often past BUY and SELL calls were right after 5 / 10 / 20 trading days.

Setup:   pip install -r requirements.txt        (yfinance, pandas, numpy)
Run:     python stock_analyzer.py RELIANCE
         python stock_analyzer.py TCS INFY HDFCBANK --years 8
         python stock_analyzer.py SBIN --bse                 (BSE price feed)
         python stock_analyzer.py TATASTEEL --capital 500000 --buy-th 30
Output:  console report + report_<SYMBOL>.html (interactive charts) + trades_<SYMBOL>.csv

Educational tool, NOT investment advice. Data comes from Yahoo Finance / Google News and can
be delayed, incomplete or wrong. Fundamentals and news are present-day snapshots, so only the
technical engine can be backtested without look-ahead bias.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency. Run:  pip install yfinance pandas numpy")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    years: int = 5              # backtest length
    exchange: str = "NS"        # NS = NSE, BO = BSE
    buy_th: float = 25.0        # technical score that triggers a BUY
    sell_th: float = -25.0      # technical score that triggers a SELL call
    exit_th: float = -25.0      # backtest: close long when score drops below this
    sl_atr: float = 2.0         # stop-loss distance in ATRs
    tp_atr: float = 3.0         # target distance in ATRs
    max_hold: int = 40          # max trading days in a trade
    cost_pct: float = 0.15      # per side: brokerage + STT + exchange/GST/stamp + slippage
    capital: float = 100_000.0  # for position-size suggestion
    risk_pct: float = 1.0       # % of capital risked per trade
    w_tech: float = 0.45
    w_fund: float = 0.35
    w_news: float = 0.20
    news: bool = True
    html: bool = True


TECH_WEIGHTS = dict(trend=25, macd=15, rsi=15, supertrend=15, adx=10, bollinger=10, volume=10)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def to_ticker(symbol: str, exchange: str) -> str:
    s = symbol.strip().upper()
    if s.startswith("^") or "." in s:
        return s
    return f"{s}.{exchange}"


def fetch_prices(ticker: str, years: float):
    tk = yf.Ticker(ticker)
    start = (dt.date.today() - dt.timedelta(days=int(365.25 * (years + 1.3)))).isoformat()
    df = tk.history(start=start, interval="1d", auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No price data for {ticker}. Use NSE symbols like RELIANCE, TCS, HDFCBANK.")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    return df, tk


def _num(info: dict, key: str):
    v = info.get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def safe_info(tk) -> dict:
    try:
        return tk.info or {}
    except Exception:
        return {}


# --------------------------------------------------------------------------------------
# Technical indicators
# --------------------------------------------------------------------------------------
def _rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (used by RSI, ATR, ADX)."""
    return s.ewm(alpha=1 / n, adjust=False).mean()


def _supertrend(h, l, c, tr, n=10, mult=3.0):
    atr = _rma(tr, n).to_numpy()
    hl2 = ((h + l) / 2).to_numpy()
    cc = c.to_numpy()
    ub, lb = hl2 + mult * atr, hl2 - mult * atr
    fub, flb = ub.copy(), lb.copy()
    direction = np.ones(len(cc))
    line = np.full(len(cc), np.nan)
    for i in range(1, len(cc)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or cc[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or cc[i - 1] < flb[i - 1]) else flb[i - 1]
        if direction[i - 1] < 0 and cc[i] > fub[i - 1]:
            direction[i] = 1
        elif direction[i - 1] > 0 and cc[i] < flb[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
        line[i] = flb[i] if direction[i] > 0 else fub[i]
    return direction, line


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"].astype(float)

    d["SMA20"] = c.rolling(20).mean()
    d["SMA50"] = c.rolling(50).mean()
    d["SMA200"] = c.rolling(200).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_SIG"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_HIST"] = d["MACD"] - d["MACD_SIG"]

    delta = c.diff()
    gain = _rma(delta.clip(lower=0), 14)
    loss = _rma((-delta).clip(lower=0), 14)
    d["RSI"] = 100 * gain / (gain + loss).replace(0, np.nan)

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    d["ATR"] = _rma(tr, 14)

    std = c.rolling(20).std()
    d["BB_UP"] = d["SMA20"] + 2 * std
    d["BB_LO"] = d["SMA20"] - 2 * std
    d["BB_PCT"] = (c - d["BB_LO"]) / (d["BB_UP"] - d["BB_LO"]).replace(0, np.nan)

    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index)
    atr = d["ATR"].replace(0, np.nan)
    d["PDI"] = 100 * _rma(pdm, 14) / atr
    d["MDI"] = 100 * _rma(mdm, 14) / atr
    dx = 100 * (d["PDI"] - d["MDI"]).abs() / (d["PDI"] + d["MDI"]).replace(0, np.nan)
    d["ADX"] = _rma(dx, 14)

    d["ST_DIR"], d["ST_LINE"] = _supertrend(h, l, c, tr, 10, 3.0)

    d["OBV"] = (np.sign(c.diff()).fillna(0) * v).cumsum()
    d["OBV_SLOPE"] = d["OBV"].diff(20)
    d["VOL_RATIO"] = v / v.rolling(20).mean().replace(0, np.nan)
    d["ROC20"] = c.pct_change(20) * 100
    d["HI52"] = h.rolling(252, min_periods=120).max()
    d["LO52"] = l.rolling(252, min_periods=120).min()
    return d


def technical_components(d: pd.DataFrame):
    """Each component is in [-1, +1]; weighted sum gives a score in [-100, +100]."""
    with np.errstate(invalid="ignore"):
        c = d["Close"].to_numpy()
        sma50, sma200 = d["SMA50"].to_numpy(), d["SMA200"].to_numpy()
        trend = 0.4 * np.sign(c - sma200) + 0.3 * np.sign(sma50 - sma200) + 0.3 * np.sign(c - sma50)

        hist = d["MACD_HIST"].to_numpy()
        hist_prev = d["MACD_HIST"].shift(1).to_numpy()
        macd = np.where(d["MACD"].to_numpy() > d["MACD_SIG"].to_numpy(), 0.6, -0.6) + \
            np.where(hist > hist_prev, 0.4, -0.4)

        r = d["RSI"].to_numpy()
        rsi = np.where(r < 30, 0.6, np.where(r < 45, -0.4, np.where(r < 55, 0.0, np.where(r < 70, 0.7, -0.6))))

        st = d["ST_DIR"].to_numpy()
        adx = np.sign(d["PDI"].to_numpy() - d["MDI"].to_numpy()) * np.clip(d["ADX"].to_numpy() / 25, 0.2, 1.0)

        b = d["BB_PCT"].to_numpy()
        bb = np.where(b < 0.05, 0.6, np.where(b > 0.95, -0.6, (b - 0.5) * 0.4))

        day_dir = np.sign(np.diff(c, prepend=np.nan))
        vol = 0.6 * np.sign(d["OBV_SLOPE"].to_numpy()) + 0.4 * (d["VOL_RATIO"].to_numpy() > 1.5) * day_dir

    comp = pd.DataFrame(dict(trend=trend, macd=macd, rsi=rsi, supertrend=st, adx=adx,
                             bollinger=bb, volume=vol), index=d.index)
    score = sum(comp[k] * w for k, w in TECH_WEIGHTS.items())
    valid = d[["SMA200", "ADX", "BB_PCT", "RSI", "OBV_SLOPE", "VOL_RATIO"]].notna().all(axis=1)
    return comp, score.where(valid).clip(-100, 100)


def describe_technicals(row: pd.Series) -> list[str]:
    out = []
    c = row["Close"]
    out.append(f"Price {'above' if c > row['SMA200'] else 'below'} 200-DMA ({row['SMA200']:.2f}) -> "
               f"{'long-term uptrend' if c > row['SMA200'] else 'long-term downtrend'}")
    if row["SMA50"] > row["SMA200"]:
        out.append("50-DMA above 200-DMA (golden-cross regime)")
    else:
        out.append("50-DMA below 200-DMA (death-cross regime)")
    out.append(f"MACD {'above' if row['MACD'] > row['MACD_SIG'] else 'below'} signal line "
               f"(histogram {row['MACD_HIST']:+.2f})")
    r = row["RSI"]
    zone = "oversold" if r < 30 else "weak" if r < 45 else "neutral" if r < 55 else "bullish momentum" if r < 70 else "overbought"
    out.append(f"RSI(14) = {r:.1f} ({zone})")
    out.append(f"Supertrend(10,3) is {'BULLISH' if row['ST_DIR'] > 0 else 'BEARISH'} (line {row['ST_LINE']:.2f})")
    strength = "strong" if row["ADX"] >= 25 else "weak/no"
    out.append(f"ADX = {row['ADX']:.1f} ({strength} trend), +DI {row['PDI']:.1f} vs -DI {row['MDI']:.1f}")
    out.append(f"Bollinger %B = {row['BB_PCT']:.2f} (0 = lower band, 1 = upper band)")
    out.append(f"OBV 20-day slope {'rising' if row['OBV_SLOPE'] > 0 else 'falling'}, "
               f"volume {row['VOL_RATIO']:.2f}x its 20-day average")
    if pd.notna(row.get("HI52")):
        out.append(f"52-week range {row['LO52']:.2f} - {row['HI52']:.2f} "
                   f"(price is {(c / row['HI52'] - 1) * 100:+.1f}% from the high)")
    return out


# --------------------------------------------------------------------------------------
# Fundamentals
# --------------------------------------------------------------------------------------
def _band(x: float, rules, otherwise: float) -> float:
    for upper, score in rules:
        if x < upper:
            return float(score)
    return float(otherwise)


def fundamental_score(info: dict):
    rows = []

    def add(metric, disp, score, weight, note):
        rows.append(dict(metric=metric, value=disp, score=round(float(score), 2), weight=weight, note=note))

    sector = info.get("sector") or ""
    financial = "financial" in sector.lower()

    pe = _num(info, "trailingPE")
    if pe is not None:
        s = -1.0 if pe <= 0 else _band(pe, [(15, 1), (25, 0.5), (40, 0), (60, -0.5)], -1)
        add("P/E (TTM)", f"{pe:.1f}", s, 12, "<15 cheap, 15-25 fair, >40 expensive")
    fpe = _num(info, "forwardPE")
    if pe and fpe and pe > 0 and fpe > 0:
        s = 0.6 if fpe < 0.9 * pe else (-0.6 if fpe > 1.1 * pe else 0.0)
        add("Forward P/E", f"{fpe:.1f}", s, 5, "below TTM P/E = earnings expected to grow")
    pb = _num(info, "priceToBook")
    if pb is not None and pb > 0:
        add("Price / Book", f"{pb:.2f}", _band(pb, [(1.5, 1), (3, 0.5), (6, 0), (10, -0.5)], -1), 8,
            "<3 reasonable; banks usually 1-4")
    peg = _num(info, "trailingPegRatio") or _num(info, "pegRatio")
    if peg is not None and peg > 0:
        add("PEG ratio", f"{peg:.2f}", _band(peg, [(1, 1), (1.5, 0.5), (2.5, 0)], -0.6), 6,
            "<1 = growth at a reasonable price")
    roe = _num(info, "returnOnEquity")
    if roe is not None:
        r = roe * 100
        add("Return on equity", f"{r:.1f}%", _band(r, [(0, -1), (10, -0.3), (15, 0.2), (20, 0.6)], 1), 15,
            ">15% good, >20% excellent")
    if not financial:
        de = _num(info, "debtToEquity")
        if de is not None:
            x = de / 100  # Yahoo reports D/E in percent
            add("Debt / Equity", f"{x:.2f}x", _band(x, [(0.3, 1), (0.8, 0.5), (1.5, -0.2)], -0.8), 10,
                "<0.5x comfortable, >1.5x risky")
        cr = _num(info, "currentRatio")
        if cr is not None:
            add("Current ratio", f"{cr:.2f}", _band(cr, [(1, -0.5), (1.5, 0.2)], 0.5), 5, ">1.5 healthy liquidity")
    pm = _num(info, "profitMargins")
    if pm is not None:
        m = pm * 100
        add("Net profit margin", f"{m:.1f}%", _band(m, [(0, -1), (5, -0.3), (10, 0), (20, 0.5)], 1), 10,
            ">10% good, >20% excellent")
    rg = _num(info, "revenueGrowth")
    if rg is not None:
        g = rg * 100
        add("Revenue growth (YoY)", f"{g:+.1f}%", _band(g, [(0, -0.7), (8, 0), (15, 0.5)], 1), 12,
            ">15% strong")
    eg = _num(info, "earningsGrowth")
    if eg is None:
        eg = _num(info, "earningsQuarterlyGrowth")
    if eg is not None:
        g = eg * 100
        add("Earnings growth (YoY)", f"{g:+.1f}%", _band(g, [(0, -0.8), (10, 0), (20, 0.5)], 1), 12,
            ">20% strong")
    target = _num(info, "targetMeanPrice")
    price = _num(info, "currentPrice") or _num(info, "regularMarketPrice")
    n_an = _num(info, "numberOfAnalystOpinions") or 0
    if target and price and n_an >= 3:
        upside = (target / price - 1) * 100
        add("Analyst target upside", f"{upside:+.1f}% ({int(n_an)} analysts)",
            _band(upside, [(-5, -1), (5, 0), (15, 0.5)], 1), 10, "consensus mean target vs price")

    mcap = _num(info, "marketCap")
    meta = dict(
        name=info.get("longName") or info.get("shortName") or "",
        short=info.get("shortName") or info.get("longName") or "",
        sector=sector, industry=info.get("industry") or "",
        mcap_cr=(mcap / 1e7) if mcap else None,
        div_yield=(_num(info, "trailingAnnualDividendYield") or 0) * 100,
        coverage=sum(r["weight"] for r in rows),
    )
    if not rows:
        return None, rows, meta
    score = sum(r["score"] * r["weight"] for r in rows) / meta["coverage"] * 100
    return score, rows, meta


# --------------------------------------------------------------------------------------
# News & market sentiment
# --------------------------------------------------------------------------------------
POS_WORDS = {
    "beat", "beats", "surge", "surges", "surged", "soar", "soars", "soared", "jump", "jumps", "jumped",
    "rally", "rallies", "rallied", "gain", "gains", "gained", "rise", "rises", "rose", "record", "upgrade",
    "upgrades", "upgraded", "outperform", "outperforms", "bullish", "strong", "robust", "growth", "wins",
    "bags", "secures", "secured", "expansion", "expands", "approval", "approves", "approved", "dividend",
    "bonus", "buyback", "breakout", "upbeat", "positive", "boost", "boosts", "boosted", "multibagger",
    "partnership", "launches", "accumulate", "overweight", "rebound", "rebounds", "recovers", "highest",
    "profit", "profits", "climbs", "climbed", "advance", "advances", "optimistic", "momentum", "inflows",
}
NEG_WORDS = {
    "miss", "misses", "missed", "fall", "falls", "fell", "plunge", "plunges", "plunged", "slump", "slumps",
    "crash", "crashes", "drop", "drops", "dropped", "decline", "declines", "declined", "loss", "losses",
    "downgrade", "downgrades", "downgraded", "underperform", "bearish", "weak", "weaker", "probe", "raid",
    "penalty", "fined", "fraud", "scam", "default", "defaults", "resigns", "resignation", "lawsuit", "ban",
    "banned", "slashes", "warning", "warns", "concern", "concerns", "pressure", "tumble", "tumbles",
    "tumbled", "pledge", "selloff", "layoffs", "strike", "recall", "delay", "delays", "lowest", "sinks",
    "sank", "slips", "slipped", "slides", "underweight", "reduce", "outflows", "fears", "volatile", "cuts",
}
POS_PHRASES = ["all-time high", "record high", "52-week high", "beats estimates", "order win", "target raised",
               "raises target", "buy rating", "strong buy", "upper circuit", "bonus issue", "share buyback",
               "net profit rises", "profit jumps", "margin expansion"]
NEG_PHRASES = ["52-week low", "lower circuit", "misses estimates", "target cut", "cuts target", "sell rating",
               "profit falls", "net loss", "show cause", "sebi notice", "tax demand", "income tax raid",
               "promoter pledge", "rating downgrade", "profit booking", "profit-taking", "margin pressure"]


def headline_sentiment(title: str) -> float:
    t = title.lower()
    p = n = 0
    for ph in POS_PHRASES:
        if ph in t:
            p += 2
            t = t.replace(ph, " ")
    for ph in NEG_PHRASES:
        if ph in t:
            n += 2
            t = t.replace(ph, " ")
    for w in re.findall(r"[a-z]+", t):
        if w in POS_WORDS:
            p += 1
        elif w in NEG_WORDS:
            n += 1
    return 0.0 if p + n == 0 else (p - n) / (p + n)


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def google_news(query: str, days: int = 30, limit: int = 25) -> list[dict]:
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": f"{query} when:{days}d", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        root = ET.fromstring(r.read())
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        src = (it.findtext("source") or "").strip()
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3]
        try:
            ts = parsedate_to_datetime(it.findtext("pubDate")).astimezone(dt.timezone.utc).replace(tzinfo=None)
        except Exception:
            ts = None
        items.append(dict(title=title, link=it.findtext("link") or "", source=src or "Google News", time=ts))
    return items[:limit]


def yahoo_news(tk, limit: int = 15) -> list[dict]:
    try:
        raw = tk.news or []
    except Exception:
        return []
    out = []
    for n in raw[:limit]:
        c = n.get("content", n) if isinstance(n, dict) else {}
        title = c.get("title") or ""
        if not title:
            continue
        link = c.get("canonicalUrl", {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else c.get("link", "")
        src = c.get("provider", {}).get("displayName") if isinstance(c.get("provider"), dict) else c.get("publisher", "")
        ts = None
        try:
            if c.get("pubDate"):
                ts = pd.Timestamp(c["pubDate"]).tz_convert(None).to_pydatetime()
            elif n.get("providerPublishTime"):
                ts = dt.datetime.fromtimestamp(n["providerPublishTime"], dt.timezone.utc).replace(tzinfo=None)
        except Exception:
            ts = None
        out.append(dict(title=title, link=link or "", source=src or "Yahoo Finance", time=ts))
    return out


def aggregate_sentiment(items: list[dict], half_life: float = 7.0) -> float | None:
    if not items:
        return None
    now = _utcnow()
    num = den = 0.0
    for it in items:
        s = headline_sentiment(it["title"])
        it["sentiment"] = round(s, 2)
        age = (now - it["time"]).total_seconds() / 86400 if it.get("time") else 15
        w = 0.5 ** (max(age, 0) / half_life)
        if s == 0:
            w *= 0.5
        num += s * w
        den += w
    return num / den * 100 if den else None


def _dedupe(items):
    seen, out = set(), []
    for it in items:
        k = re.sub(r"[^a-z0-9]", "", it["title"].lower())[:80]
        if k and k not in seen:
            seen.add(k)
            out.append(it)
    return out


def market_context(cfg: Config) -> dict:
    ctx = dict(regime=None, headline=None, items=[], text="Nifty data unavailable")
    try:
        nifty, _ = fetch_prices("^NSEI", 1)
        c = nifty["Close"]
        s50, s200 = c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
        last = c.iloc[-1]
        regime = (0.4 * np.sign(last - s200) + 0.3 * np.sign(s50 - s200) + 0.3 * np.sign(last - s50)) * 100
        ctx["regime"] = float(regime)
        ctx["nifty"] = float(last)
        ctx["nifty_1m"] = float((last / c.iloc[-22] - 1) * 100) if len(c) > 22 else None
        ctx["text"] = (f"Nifty 50 at {last:,.0f}, {'above' if last > s200 else 'below'} its 200-DMA, "
                       f"{ctx['nifty_1m']:+.1f}% over 1 month")
    except Exception as e:
        ctx["text"] = f"Nifty data unavailable ({e.__class__.__name__})"
    if cfg.news:
        try:
            items = _dedupe(google_news("Sensex Nifty stock market", days=7, limit=20))
            ctx["headline"] = aggregate_sentiment(items)
            ctx["items"] = items
        except Exception:
            pass
    return ctx


def news_pillar(tk, meta: dict, symbol: str, market: dict, cfg: Config) -> dict:
    items = []
    if cfg.news:
        name = re.sub(r"\b(limited|ltd\.?|ltd)\b", "", meta.get("short") or symbol, flags=re.I).strip(" .,")
        name = name or symbol
        try:
            items += google_news(f'"{name}" (shares OR stock OR NSE OR BSE)', days=30)
        except Exception:
            pass
        items += yahoo_news(tk)
        items = _dedupe(items)
        items.sort(key=lambda x: x.get("time") or dt.datetime.min, reverse=True)
    stock = aggregate_sentiment(items)

    mparts = [(market.get("regime"), 0.6), (market.get("headline"), 0.4)]
    mparts = [(s, w) for s, w in mparts if s is not None]
    mkt = sum(s * w for s, w in mparts) / sum(w for _, w in mparts) if mparts else None

    if stock is not None and mkt is not None:
        score = 0.7 * stock + 0.3 * mkt
    else:
        score = stock if stock is not None else mkt
    return dict(score=score, stock=stock, market=mkt, items=items)


# --------------------------------------------------------------------------------------
# Combine into a call
# --------------------------------------------------------------------------------------
def verdict(score: float) -> str:
    if score >= 40:
        return "STRONG BUY"
    if score >= 15:
        return "BUY"
    if score > -15:
        return "HOLD / NEUTRAL"
    if score > -40:
        return "SELL"
    return "STRONG SELL"


def combine(tech, fund, fund_cov, news, cfg: Config):
    w_fund = cfg.w_fund * min(1.0, fund_cov / 60) if fund is not None else 0
    parts = [(tech, cfg.w_tech), (fund, w_fund), (news, cfg.w_news if news is not None else 0)]
    parts = [(s, w) for s, w in parts if s is not None and w > 0]
    total_w = sum(w for _, w in parts)
    score = sum(s * w for s, w in parts) / total_w
    weights = dict(technical=cfg.w_tech / total_w, fundamental=w_fund / total_w,
                   news=(cfg.w_news if news is not None else 0) / total_w)
    notes = []
    if fund is not None and tech is not None and np.sign(fund) != np.sign(tech) and abs(fund - tech) > 50:
        notes.append("Technicals and fundamentals disagree strongly - treat the call with extra caution.")
    if fund is None:
        notes.append("No fundamentals available - call is based on technicals and news only.")
    elif fund_cov < 60:
        notes.append("Limited fundamental data - fundamental weight was reduced.")
    return score, weights, notes


def trade_plan(d: pd.DataFrame, call: str, cfg: Config) -> dict:
    last = d.iloc[-1]
    px, atr = float(last["Close"]), float(last["ATR"])
    sup, res = float(d["Low"].tail(20).min()), float(d["High"].tail(20).max())
    plan = dict(price=px, atr=atr, support=sup, resistance=res)
    if "BUY" in call:
        sl = px - cfg.sl_atr * atr
        plan.update(side="LONG", entry=px, stop=sl, target1=px + cfg.tp_atr * atr,
                    target2=px + 2 * cfg.tp_atr * atr)
        risk = px - sl
    elif "SELL" in call:
        sl = px + cfg.sl_atr * atr
        plan.update(side="EXIT / SHORT", entry=px, stop=sl, target1=px - cfg.tp_atr * atr,
                    target2=px - 2 * cfg.tp_atr * atr)
        risk = sl - px
    else:
        plan.update(side="WAIT", entry=None, stop=None, target1=None, target2=None)
        risk = None
    if risk and risk > 0:
        qty = int(cfg.capital * cfg.risk_pct / 100 / risk)
        qty = min(qty, int(cfg.capital / px))
        plan.update(qty=qty, risk_rs=qty * risk, rr=cfg.tp_atr / cfg.sl_atr)
    return plan


# --------------------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------------------
def backtest(d: pd.DataFrame, score: pd.Series, cfg: Config, start: pd.Timestamp) -> dict:
    """Long-only, enter next day's open after the score crosses above buy_th.
    Exit on stop-loss (sl_atr x ATR), target (tp_atr x ATR), score < exit_th (next open),
    or max_hold days. Same-day stop & target -> stop assumed (conservative). Costs both sides."""
    o, h, l, c = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    atr, s, dates = d["ATR"].to_numpy(), score.to_numpy(), d.index
    n = len(d)
    i0 = max(int(np.searchsorted(dates, start)), int(np.argmax(~np.isnan(s))) + 1)
    cost = cfg.cost_pct / 100

    equity = np.full(n, np.nan)
    in_pos = np.zeros(n, dtype=bool)
    trades = []
    eq = 1.0
    pos = pending_entry = pending_exit = False
    entry_px = stop = target = eq_entry = sig_atr = sig_score = 0.0
    entry_i = 0

    for i in range(i0, n):
        if pending_entry and not pos:
            entry_px, entry_i = o[i], i
            stop, target = entry_px - cfg.sl_atr * sig_atr, entry_px + cfg.tp_atr * sig_atr
            eq_entry, pos, pending_entry = eq, True, False

        if pos:
            exit_px = reason = None
            if pending_exit:
                exit_px, reason = o[i], "Signal reversal"
            elif l[i] <= stop:
                exit_px, reason = min(o[i], stop), "Stop-loss"
            elif h[i] >= target:
                exit_px, reason = max(o[i], target), "Target hit"
            elif i - entry_i >= cfg.max_hold:
                exit_px, reason = c[i], "Time exit"

            if exit_px is not None:
                net = (exit_px * (1 - cost)) / (entry_px * (1 + cost)) - 1
                eq = eq_entry * (1 + net)
                trades.append(dict(entry_date=dates[entry_i].date(), entry=round(entry_px, 2),
                                   exit_date=dates[i].date(), exit=round(exit_px, 2),
                                   days=i - entry_i, entry_score=round(sig_score, 1),
                                   gross_pct=round((exit_px / entry_px - 1) * 100, 2),
                                   net_pct=round(net * 100, 2), result="WIN" if net > 0 else "LOSS",
                                   reason=reason, status="Closed"))
                pos = pending_exit = False
            else:
                in_pos[i] = True
                if s[i] < cfg.exit_th:
                    pending_exit = True

        if not pos and not pending_entry and i < n - 1 and i > 0:
            if s[i] >= cfg.buy_th and not (s[i - 1] >= cfg.buy_th):
                pending_entry, sig_atr, sig_score = True, atr[i], s[i]

        equity[i] = eq_entry * (c[i] * (1 - cost)) / (entry_px * (1 + cost)) if pos else eq

    if pos:
        net = (c[-1] * (1 - cost)) / (entry_px * (1 + cost)) - 1
        trades.append(dict(entry_date=dates[entry_i].date(), entry=round(entry_px, 2),
                           exit_date=dates[-1].date(), exit=round(c[-1], 2), days=n - 1 - entry_i,
                           entry_score=round(sig_score, 1), gross_pct=round((c[-1] / entry_px - 1) * 100, 2),
                           net_pct=round(net * 100, 2), result="OPEN", reason="Still open (marked to market)",
                           status="Open"))

    eq_s = pd.Series(equity, index=dates).dropna()
    closed = [t for t in trades if t["status"] == "Closed"]
    r = np.array([t["net_pct"] for t in closed]) / 100 if closed else np.array([])
    wins, losses = r[r > 0], r[r <= 0]
    yrs = max((eq_s.index[-1] - eq_s.index[0]).days / 365.25, 1e-9)
    bh = d["Close"].iloc[i0:]
    bh_ret = bh.iloc[-1] / bh.iloc[0] - 1
    daily = eq_s.pct_change().dropna()

    stats = dict(
        period=f"{eq_s.index[0].date()} to {eq_s.index[-1].date()}",
        years=yrs,
        trades=len(closed),
        wins=len(wins), losses=len(losses),
        win_pct=len(wins) / len(closed) * 100 if closed else 0.0,
        loss_pct=len(losses) / len(closed) * 100 if closed else 0.0,
        avg_win=wins.mean() * 100 if len(wins) else 0.0,
        avg_loss=losses.mean() * 100 if len(losses) else 0.0,
        best=r.max() * 100 if len(r) else 0.0,
        worst=r.min() * 100 if len(r) else 0.0,
        profit_factor=(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("inf") if len(wins) else 0.0,
        expectancy=r.mean() * 100 if len(r) else 0.0,
        avg_days=float(np.mean([t["days"] for t in closed])) if closed else 0.0,
        total_return=(eq_s.iloc[-1] - 1) * 100,
        cagr=(eq_s.iloc[-1] ** (1 / yrs) - 1) * 100 if eq_s.iloc[-1] > 0 else -100.0,
        max_dd=((eq_s / eq_s.cummax()) - 1).min() * 100,
        sharpe=daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0,
        exposure=in_pos[i0:].mean() * 100,
        bh_return=bh_ret * 100,
        bh_cagr=((1 + bh_ret) ** (1 / yrs) - 1) * 100,
        bh_max_dd=((bh / bh.cummax()) - 1).min() * 100,
        exit_reasons=pd.Series([t["reason"] for t in closed]).value_counts().to_dict() if closed else {},
    )
    return dict(stats=stats, trades=trades, equity=eq_s, start_i=i0)


def signal_accuracy(d: pd.DataFrame, score: pd.Series, cfg: Config, start_i: int, horizons=(5, 10, 20)):
    """How often a BUY call (score crossing up through buy_th) was followed by a higher price,
    and a SELL call (crossing down through sell_th) by a lower price, N trading days later."""
    s, c = score, d["Close"]
    buy = (s >= cfg.buy_th) & (s.shift(1) < cfg.buy_th)
    sell = (s <= cfg.sell_th) & (s.shift(1) > cfg.sell_th)
    window = pd.Series(np.arange(len(d)) >= start_i, index=d.index)
    rows = []
    for hzn in horizons:
        fwd = c.shift(-hzn) / c - 1
        ok = window & fwd.notna()
        b, sl = fwd[buy & ok], fwd[sell & ok]
        base = fwd[ok]
        rows.append(dict(
            horizon=hzn,
            buy_n=len(b), buy_success=(b > 0).mean() * 100 if len(b) else None,
            buy_avg=b.mean() * 100 if len(b) else None,
            sell_n=len(sl), sell_success=(sl < 0).mean() * 100 if len(sl) else None,
            sell_avg=sl.mean() * 100 if len(sl) else None,
            base_up=(base > 0).mean() * 100 if len(base) else None,
        ))
    return rows


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def _fmt(x, nd=2, suffix="", plus=False):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "inf" if isinstance(x, float) and math.isinf(x) else "n/a"
    return f"{x:+,.{nd}f}{suffix}" if plus else f"{x:,.{nd}f}{suffix}"


def print_report(R: dict, cfg: Config):
    line = "=" * 78
    st, plan = R["bt"]["stats"], R["plan"]
    m = R["fund_meta"]
    print("\n" + line)
    print(f" {R['ticker']}  {m.get('name', '')}")
    if m.get("sector"):
        mc = f" | Mkt cap Rs {m['mcap_cr']:,.0f} cr" if m.get("mcap_cr") else ""
        print(f" {m['sector']} / {m['industry']}{mc}")
    print(f" Last close: Rs {plan['price']:,.2f}  ({R['last_date']})")
    print(line)
    print(f"\n  >>> CALL: {R['call']}   (composite score {R['score']:+.1f} / 100)\n")
    print(f"  Technical   {_fmt(R['tech'], 1, plus=True):>7}   weight {R['weights']['technical']*100:.0f}%")
    print(f"  Fundamental {_fmt(R['fund'], 1, plus=True):>7}   weight {R['weights']['fundamental']*100:.0f}%")
    print(f"  News/market {_fmt(R['news']['score'], 1, plus=True):>7}   weight {R['weights']['news']*100:.0f}%")
    for n in R["notes"]:
        print(f"  ! {n}")

    print("\n  TRADE PLAN")
    if plan["side"] == "WAIT":
        print(f"  No trade. Watch support Rs {plan['support']:,.2f} / resistance Rs {plan['resistance']:,.2f}")
    else:
        label = "Exit longs; short only via F&O/intraday" if plan["side"] != "LONG" else "Long"
        print(f"  {label}: entry ~Rs {plan['entry']:,.2f} | stop-loss Rs {plan['stop']:,.2f} | "
              f"target 1 Rs {plan['target1']:,.2f} | target 2 Rs {plan['target2']:,.2f}")
        if plan.get("qty"):
            print(f"  Size for {cfg.risk_pct:g}% risk on Rs {cfg.capital:,.0f}: {plan['qty']} shares "
                  f"(risk Rs {plan['risk_rs']:,.0f}, reward:risk {plan['rr']:.1f})")
    print(f"  ATR(14) Rs {plan['atr']:,.2f} | 20-day support Rs {plan['support']:,.2f} | "
          f"resistance Rs {plan['resistance']:,.2f}")

    print("\n  TECHNICALS")
    for t in R["tech_notes"]:
        print(f"   - {t}")

    print("\n  FUNDAMENTALS")
    if R["fund_rows"]:
        for r in R["fund_rows"]:
            tag = "+" if r["score"] > 0.2 else "-" if r["score"] < -0.2 else "~"
            print(f"   [{tag}] {r['metric']:<24}{r['value']:<28}{r['note']}")
    else:
        print("   No fundamental data returned by Yahoo Finance.")

    print("\n  NEWS & MARKET")
    print(f"   Market: {R['market']['text']}")
    print(f"   Stock headline sentiment: {_fmt(R['news']['stock'], 0, plus=True)} | "
          f"market sentiment: {_fmt(R['news']['market'], 0, plus=True)}")
    for it in R["news"]["items"][:8]:
        when = it["time"].strftime("%d %b") if it.get("time") else "      "
        print(f"   {when}  [{it.get('sentiment', 0):+.1f}] {it['title'][:90]}")

    print("\n  BACKTEST (technical engine, long-only, costs included)")
    print(f"   Period {st['period']}  ({st['years']:.1f} yrs)")
    print(f"   Trades {st['trades']} | Wins {st['wins']} ({st['win_pct']:.1f}%) | "
          f"Losses {st['losses']} ({st['loss_pct']:.1f}%)")
    print(f"   Avg win {st['avg_win']:+.2f}% | Avg loss {st['avg_loss']:+.2f}% | "
          f"Profit factor {_fmt(st['profit_factor'])} | Expectancy {st['expectancy']:+.2f}%/trade")
    print(f"   Strategy return {st['total_return']:+.1f}% (CAGR {st['cagr']:+.1f}%, max DD {st['max_dd']:.1f}%, "
          f"Sharpe {st['sharpe']:.2f}, in market {st['exposure']:.0f}% of days)")
    print(f"   Buy & hold      {st['bh_return']:+.1f}% (CAGR {st['bh_cagr']:+.1f}%, max DD {st['bh_max_dd']:.1f}%)")
    beat = st["total_return"] > st["bh_return"]
    print(f"   Beat buy & hold? {'YES' if beat else 'NO'}"
          + ("" if beat else " - simply holding the stock would have done better over this period"))
    if st["exit_reasons"]:
        print("   Exits: " + ", ".join(f"{k} {v}" for k, v in st["exit_reasons"].items()))
    print("\n   Signal accuracy (was the call right N trading days later?)")
    print(f"   {'N':>4} {'BUY calls':>10} {'success':>8} {'avg ret':>8} | {'SELL calls':>10} {'success':>8} "
          f"{'avg ret':>8} | {'any day up':>10}")
    for a in R["acc"]:
        print(f"   {a['horizon']:>4} {a['buy_n']:>10} {_fmt(a['buy_success'], 1, '%'):>8} "
              f"{_fmt(a['buy_avg'], 2, '%', True):>8} | {a['sell_n']:>10} {_fmt(a['sell_success'], 1, '%'):>8} "
              f"{_fmt(a['sell_avg'], 2, '%', True):>8} | {_fmt(a['base_up'], 1, '%'):>10}")
    print("   ('any day up' = baseline: % of all days the stock was higher N days later.)")
    print("\n  Not investment advice. Past performance does not guarantee future results.")
    print(line)


def _ser(x):
    return [None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(float(v), 2) for v in x]


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#e4e4e0;--pos:#15803d;--neg:#b91c1c;--neu:#a16207}
@media (prefers-color-scheme:dark){:root{--bg:#121214;--card:#1c1c1f;--ink:#ececee;--mute:#9a9aa2;--line:#2c2c31;--pos:#4ade80;--neg:#f87171;--neu:#facc15}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 60px}
h1{font-size:26px;margin:0}h2{font-size:17px;margin:0 0 12px}.sub{color:var(--mute);margin:4px 0 0}
.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));margin-top:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-top:16px;overflow-x:auto}
.grid .card{margin-top:0}
.call{font-size:30px;font-weight:700;letter-spacing:.5px}
.pos{color:var(--pos)}.neg{color:var(--neg)}.neu{color:var(--neu)}
.big{font-size:24px;font-weight:650}.lbl{color:var(--mute);font-size:13px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--mute);font-weight:600}td.wrap{white-space:normal}
.bar{height:8px;border-radius:4px;background:var(--line);position:relative;margin-top:6px}
.bar span{position:absolute;top:0;bottom:0;border-radius:4px}
ul{margin:0;padding-left:18px}.note{color:var(--mute);font-size:13px;margin-top:10px}
a{color:inherit}
</style></head><body><main>
__BODY__
<div class="card"><div id="price"></div></div>
<div class="card"><div id="score"></div></div>
<div class="card"><div id="rsi"></div></div>
<div class="card"><div id="equity"></div></div>
__TABLES__
<p class="note">Educational tool, not investment advice. Data from Yahoo Finance and Google News may be delayed or incomplete. Fundamentals and news are current snapshots; the backtest covers the technical engine only. Generated __GEN__.</p>
</main>
<script>
const D = __DATA__;
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
const ink = dark ? '#ddd' : '#222', grid = dark ? '#2c2c31' : '#e8e8e4';
const base = (h, t) => ({height:h, title:{text:t, font:{size:14}}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  font:{color:ink}, margin:{l:55,r:15,t:40,b:30}, legend:{orientation:'h', y:-0.12},
  xaxis:{gridcolor:grid, rangeslider:{visible:false}}, yaxis:{gridcolor:grid}});
const cfg = {responsive:true, displaylogo:false};
const hline = (y, c) => ({type:'line', xref:'paper', x0:0, x1:1, y0:y, y1:y, line:{color:c, dash:'dot', width:1}});
Plotly.newPlot('price', [
  {type:'candlestick', x:D.dates, open:D.open, high:D.high, low:D.low, close:D.close, name:'Price'},
  {type:'scatter', x:D.dates, y:D.sma50, name:'SMA 50', line:{width:1.3, color:'#3b82f6'}},
  {type:'scatter', x:D.dates, y:D.sma200, name:'SMA 200', line:{width:1.3, color:'#f59e0b'}},
  {type:'scatter', x:D.dates, y:D.st, name:'Supertrend', line:{width:1, color:'#a855f7', dash:'dot'}, visible:'legendonly'},
  {type:'scatter', mode:'markers', x:D.entries.x, y:D.entries.y, name:'Entry', marker:{symbol:'triangle-up', size:11, color:'#16a34a'}},
  {type:'scatter', mode:'markers', x:D.exits.x, y:D.exits.y, text:D.exits.t, name:'Exit', marker:{symbol:'triangle-down', size:11, color:D.exits.c}}
], base(480, 'Price, moving averages and backtest trades'), cfg);
Plotly.newPlot('score', [{type:'scatter', x:D.dates, y:D.score, name:'Technical score', fill:'tozeroy', line:{color:'#6366f1', width:1}}],
  Object.assign(base(260, 'Technical score (-100 to +100)'), {shapes:[hline(D.buy_th,'#16a34a'), hline(D.sell_th,'#dc2626')]}), cfg);
Plotly.newPlot('rsi', [
  {type:'scatter', x:D.dates, y:D.rsi, name:'RSI 14', line:{color:'#0ea5e9', width:1.2}},
  {type:'bar', x:D.dates, y:D.hist, name:'MACD hist', yaxis:'y2', marker:{color:D.hist.map(v => v >= 0 ? '#16a34a' : '#dc2626')}, opacity:.5}
], Object.assign(base(260, 'RSI and MACD histogram'), {yaxis:{gridcolor:grid, range:[0,100], title:'RSI'},
  yaxis2:{overlaying:'y', side:'right', showgrid:false, title:'MACD'}, shapes:[hline(70,'#dc2626'), hline(30,'#16a34a')]}), cfg);
Plotly.newPlot('equity', [
  {type:'scatter', x:D.eq_dates, y:D.eq, name:'Strategy', line:{color:'#16a34a', width:2}},
  {type:'scatter', x:D.eq_dates, y:D.bh, name:'Buy & hold', line:{color:'#94a3b8', width:1.5}}
], base(320, 'Backtest equity (start = 100)'), cfg);
</script></body></html>"""


def _cls(x):
    return "neu" if x is None else "pos" if x >= 15 else "neg" if x <= -15 else "neu"


def _bar(x):
    if x is None:
        return ""
    x = max(-100, min(100, x))
    left, width = (50, x / 2) if x >= 0 else (50 + x / 2, -x / 2)
    color = "var(--pos)" if x >= 15 else "var(--neg)" if x <= -15 else "var(--neu)"
    return f'<div class="bar"><span style="left:{left}%;width:{width}%;background:{color}"></span></div>'


def write_html(R: dict, cfg: Config, path: str):
    d, bt, st, plan = R["d"], R["bt"], R["bt"]["stats"], R["plan"]
    m, e = R["fund_meta"], html.escape
    view = d.iloc[bt["start_i"]:]
    eq = bt["equity"]
    bh = view["Close"] / view["Close"].iloc[0] * 100
    closed = [t for t in bt["trades"] if t["status"] == "Closed"]
    payload = dict(
        dates=[x.strftime("%Y-%m-%d") for x in view.index],
        open=_ser(view["Open"]), high=_ser(view["High"]), low=_ser(view["Low"]), close=_ser(view["Close"]),
        sma50=_ser(view["SMA50"]), sma200=_ser(view["SMA200"]), st=_ser(view["ST_LINE"]),
        score=_ser(R["tscore"].iloc[bt["start_i"]:]), rsi=_ser(view["RSI"]), hist=_ser(view["MACD_HIST"]),
        buy_th=cfg.buy_th, sell_th=cfg.sell_th,
        entries=dict(x=[str(t["entry_date"]) for t in bt["trades"]], y=[t["entry"] for t in bt["trades"]]),
        exits=dict(x=[str(t["exit_date"]) for t in closed], y=[t["exit"] for t in closed],
                   t=[f"{t['reason']}: {t['net_pct']:+.2f}%" for t in closed],
                   c=["#16a34a" if t["net_pct"] > 0 else "#dc2626" for t in closed]),
        eq_dates=[x.strftime("%Y-%m-%d") for x in eq.index], eq=_ser(eq * 100),
        bh=_ser(bh.reindex(eq.index)),
    )

    mc = f" · Mkt cap ₹{m['mcap_cr']:,.0f} cr" if m.get("mcap_cr") else ""
    if plan["side"] == "WAIT":
        plan_html = (f"<p>No new position. Watch 20-day support <b>₹{plan['support']:,.2f}</b> and resistance "
                     f"<b>₹{plan['resistance']:,.2f}</b>.</p>")
    else:
        side = "Long entry" if plan["side"] == "LONG" else "Exit longs (short only via F&amp;O/intraday)"
        qty = (f"<tr><td class='wrap'>Size ({cfg.risk_pct:g}% risk on ₹{cfg.capital:,.0f})</td><td class='wrap'>{plan['qty']} shares · "
               f"risk ₹{plan['risk_rs']:,.0f} · R:R {plan['rr']:.1f}</td></tr>") if plan.get("qty") else ""
        plan_html = (f"<table><tr><td>{side}</td><td>₹{plan['entry']:,.2f}</td></tr>"
                     f"<tr><td>Stop-loss</td><td>₹{plan['stop']:,.2f}</td></tr>"
                     f"<tr><td>Target 1</td><td>₹{plan['target1']:,.2f}</td></tr>"
                     f"<tr><td>Target 2</td><td>₹{plan['target2']:,.2f}</td></tr>{qty}</table>")
    notes = "".join(f"<p class='note'>⚠ {e(n)}</p>" for n in R["notes"])

    def pillar(name, val, w):
        return (f"<div><div class='lbl'>{name} · weight {w*100:.0f}%</div>"
                f"<div class='big {_cls(val)}'>{_fmt(val, 0, plus=True)}</div>{_bar(val)}</div>")

    body = f"""
<h1>{e(R['ticker'])} <span style="font-weight:400">{e(m.get('name',''))}</span></h1>
<p class="sub">{e(m.get('sector',''))} {('/ ' + e(m['industry'])) if m.get('industry') else ''}{mc} · Last close ₹{plan['price']:,.2f} on {R['last_date']}</p>
<div class="grid">
 <div class="card"><div class="lbl">Call</div><div class="call {_cls(R['score'])}">{R['call']}</div>
  <div class="lbl">Composite score {R['score']:+.1f} / 100</div>{_bar(R['score'])}{notes}</div>
 <div class="card" style="display:grid;gap:12px">{pillar('Technical', R['tech'], R['weights']['technical'])}
  {pillar('Fundamental', R['fund'], R['weights']['fundamental'])}{pillar('News & market', R['news']['score'], R['weights']['news'])}</div>
 <div class="card"><h2>Trade plan</h2>{plan_html}<p class="note">ATR(14) ₹{plan['atr']:,.2f}</p></div>
</div>
<div class="grid" style="margin-top:16px">
 <div class="card"><div class="lbl">Backtest win rate</div><div class="big pos">{st['win_pct']:.1f}%</div>
  <div class="lbl">{st['wins']} wins · {st['losses']} losses ({st['loss_pct']:.1f}%) of {st['trades']} trades</div></div>
 <div class="card"><div class="lbl">Strategy return ({st['years']:.1f} yrs)</div><div class="big {'pos' if st['total_return']>0 else 'neg'}">{st['total_return']:+.1f}%</div>
  <div class="lbl">CAGR {st['cagr']:+.1f}% · max drawdown {st['max_dd']:.1f}%</div></div>
 <div class="card"><div class="lbl">Buy &amp; hold, same period</div><div class="big">{st['bh_return']:+.1f}%</div>
  <div class="lbl">CAGR {st['bh_cagr']:+.1f}% · max drawdown {st['bh_max_dd']:.1f}%</div>
  <div class="lbl {'pos' if st['total_return'] > st['bh_return'] else 'neg'}">Strategy {'beat' if st['total_return'] > st['bh_return'] else 'did not beat'} buy &amp; hold</div></div>
 <div class="card"><div class="lbl">Profit factor · expectancy</div><div class="big">{_fmt(st['profit_factor'])}</div>
  <div class="lbl">{st['expectancy']:+.2f}% per trade · avg {st['avg_days']:.0f} days held</div></div>
</div>"""

    tech_li = "".join(f"<li>{e(t)}</li>" for t in R["tech_notes"])
    fund_tr = "".join(
        f"<tr><td>{e(r['metric'])}</td><td>{e(r['value'])}</td>"
        f"<td class='{'pos' if r['score']>0.2 else 'neg' if r['score']<-0.2 else 'neu'}'>{r['score']:+.1f}</td>"
        f"<td class='wrap lbl'>{e(r['note'])}</td></tr>" for r in R["fund_rows"]) or \
        "<tr><td colspan=4>No fundamental data available.</td></tr>"
    news_tr = "".join(
        f"<tr><td>{it['time'].strftime('%d %b') if it.get('time') else ''}</td>"
        f"<td class='{'pos' if it.get('sentiment',0)>0 else 'neg' if it.get('sentiment',0)<0 else 'neu'}'>{it.get('sentiment',0):+.1f}</td>"
        f"<td class='wrap'><a href='{e(it['link'])}' target='_blank' rel='noopener'>{e(it['title'])}</a></td>"
        f"<td class='lbl'>{e(it['source'] or '')}</td></tr>" for it in R["news"]["items"][:20]) or \
        "<tr><td colspan=4>No recent headlines found.</td></tr>"
    acc_tr = "".join(
        f"<tr><td>{a['horizon']} days</td><td>{a['buy_n']}</td><td>{_fmt(a['buy_success'],1,'%')}</td>"
        f"<td>{_fmt(a['buy_avg'],2,'%',True)}</td><td>{a['sell_n']}</td><td>{_fmt(a['sell_success'],1,'%')}</td>"
        f"<td>{_fmt(a['sell_avg'],2,'%',True)}</td><td>{_fmt(a['base_up'],1,'%')}</td></tr>" for a in R["acc"])
    trade_tr = "".join(
        f"<tr><td>{t['entry_date']}</td><td>₹{t['entry']:,.2f}</td><td>{t['exit_date']}</td><td>₹{t['exit']:,.2f}</td>"
        f"<td>{t['days']}</td><td class='{'pos' if t['net_pct']>0 else 'neg'}'>{t['net_pct']:+.2f}%</td>"
        f"<td>{t['result']}</td><td>{e(t['reason'])}</td></tr>" for t in reversed(bt["trades"]))
    exits = " · ".join(f"{e(k)}: {v}" for k, v in st["exit_reasons"].items())

    tables = f"""
<div class="grid"><div class="card"><h2>Technical read</h2><ul>{tech_li}</ul></div>
<div class="card"><h2>Fundamentals</h2><table><tr><th>Metric</th><th>Value</th><th>Score</th><th>Guide</th></tr>{fund_tr}</table></div></div>
<div class="card"><h2>News &amp; market</h2><p class="lbl">{e(R['market']['text'])} · stock headline sentiment {_fmt(R['news']['stock'],0,plus=True)} · market sentiment {_fmt(R['news']['market'],0,plus=True)}</p>
<table><tr><th>Date</th><th>Tone</th><th>Headline</th><th>Source</th></tr>{news_tr}</table></div>
<div class="card"><h2>Signal accuracy</h2><p class="lbl">How often past BUY and SELL calls were right N trading days later. "Any day up" is the baseline to beat.</p>
<table><tr><th>Horizon</th><th>BUY calls</th><th>Success</th><th>Avg return</th><th>SELL calls</th><th>Success</th><th>Avg return</th><th>Any day up</th></tr>{acc_tr}</table></div>
<div class="card"><h2>Backtest trades</h2><p class="lbl">{st['period']} · long-only · entry next open after score ≥ {cfg.buy_th:g} · stop {cfg.sl_atr:g}×ATR · target {cfg.tp_atr:g}×ATR · exit if score &lt; {cfg.exit_th:g} or after {cfg.max_hold} days · {cfg.cost_pct:g}% cost per side · Sharpe {st['sharpe']:.2f} · in market {st['exposure']:.0f}% of days<br>Exits: {exits}</p>
<table><tr><th>Entry</th><th>Price</th><th>Exit</th><th>Price</th><th>Days</th><th>Net</th><th>Result</th><th>Reason</th></tr>{trade_tr}</table></div>"""

    out = (HTML_TEMPLATE.replace("__TITLE__", e(f"{R['ticker']} Analysis"))
           .replace("__BODY__", body).replace("__TABLES__", tables)
           .replace("__GEN__", dt.datetime.now().strftime("%d %b %Y %H:%M"))
           .replace("__DATA__", json.dumps(payload)))
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def analyze(symbol: str, cfg: Config, market: dict) -> dict:
    ticker = to_ticker(symbol, cfg.exchange)
    print(f"\nFetching {ticker} ...")
    raw, tk = fetch_prices(ticker, cfg.years)
    if len(raw) < 260:
        raise ValueError(f"{ticker}: need at least ~1 year of history (got {len(raw)} days)")
    d = add_indicators(raw)
    comp, tscore = technical_components(d)
    tech = float(tscore.iloc[-1])

    info = safe_info(tk)
    fund, fund_rows, fund_meta = fundamental_score(info)
    if not fund_meta["name"]:
        fund_meta["name"] = fund_meta["short"] = symbol.upper()
    news = news_pillar(tk, fund_meta, symbol.split(".")[0], market, cfg)

    score, weights, notes = combine(tech, fund, fund_meta["coverage"], news["score"], cfg)
    call = verdict(score)
    plan = trade_plan(d, call, cfg)

    start = d.index[-1] - pd.DateOffset(years=cfg.years)
    bt = backtest(d, tscore, cfg, start)
    acc = signal_accuracy(d, tscore, cfg, bt["start_i"])

    return dict(ticker=ticker, d=d, tscore=tscore, comp=comp, tech=tech, fund=fund, fund_rows=fund_rows,
                fund_meta=fund_meta, news=news, market=market, score=score, weights=weights, notes=notes,
                call=call, plan=plan, bt=bt, acc=acc, tech_notes=describe_technicals(d.iloc[-1]),
                last_date=d.index[-1].strftime("%d %b %Y"))


def main(argv=None):
    p = argparse.ArgumentParser(description="Analyse Indian stocks: fundamentals + technicals + news -> call + backtest")
    p.add_argument("symbols", nargs="+", help="NSE symbols, e.g. RELIANCE TCS HDFCBANK (or full Yahoo tickers)")
    p.add_argument("--bse", action="store_true", help="use BSE (.BO) instead of NSE (.NS)")
    p.add_argument("--years", type=int, default=5, help="backtest length in years (default 5)")
    p.add_argument("--buy-th", type=float, default=25, help="technical score for BUY (default 25)")
    p.add_argument("--sell-th", type=float, default=-25, help="technical score for SELL (default -25)")
    p.add_argument("--exit-th", type=float, default=-25, help="backtest exit score (default -25)")
    p.add_argument("--sl-atr", type=float, default=2.0, help="stop-loss in ATRs (default 2)")
    p.add_argument("--tp-atr", type=float, default=3.0, help="target in ATRs (default 3)")
    p.add_argument("--max-hold", type=int, default=40, help="max days per trade (default 40)")
    p.add_argument("--cost", type=float, default=0.15, help="cost %% per side (default 0.15)")
    p.add_argument("--capital", type=float, default=100000, help="capital for sizing (default 100000)")
    p.add_argument("--risk", type=float, default=1.0, help="%% of capital risked per trade (default 1)")
    p.add_argument("--no-news", action="store_true", help="skip news fetching")
    p.add_argument("--no-html", action="store_true", help="skip HTML report")
    a = p.parse_args(argv)

    cfg = Config(years=a.years, exchange="BO" if a.bse else "NS", buy_th=a.buy_th, sell_th=a.sell_th,
                 exit_th=a.exit_th, sl_atr=a.sl_atr, tp_atr=a.tp_atr, max_hold=a.max_hold, cost_pct=a.cost,
                 capital=a.capital, risk_pct=a.risk, news=not a.no_news, html=not a.no_html)

    print("Reading market context (Nifty 50) ...")
    market = market_context(cfg)
    summary = []
    for sym in a.symbols:
        try:
            R = analyze(sym, cfg, market)
        except Exception as ex:
            print(f"\n[{sym}] failed: {ex}")
            continue
        print_report(R, cfg)
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", R["ticker"])
        pd.DataFrame(R["bt"]["trades"]).to_csv(f"trades_{safe}.csv", index=False)
        if cfg.html:
            write_html(R, cfg, f"report_{safe}.html")
            print(f"  Saved report_{safe}.html and trades_{safe}.csv")
        st = R["bt"]["stats"]
        summary.append((R["ticker"], R["call"], R["score"], st["win_pct"], st["trades"], st["cagr"], st["bh_cagr"]))

    if len(summary) > 1:
        print("\nSUMMARY")
        print(f"{'Ticker':<16}{'Call':<16}{'Score':>7}{'Win %':>8}{'Trades':>8}{'CAGR':>8}{'B&H':>8}")
        for t, c, s, w, n, cg, bh in sorted(summary, key=lambda x: -x[2]):
            print(f"{t:<16}{c:<16}{s:>+7.1f}{w:>7.1f}%{n:>8}{cg:>+7.1f}%{bh:>+7.1f}%")


if __name__ == "__main__":
    main()
