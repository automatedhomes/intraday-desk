"""
Options lab helpers: Black-Scholes prices and Greeks, implied volatility, NSE F&O option charges, and the
NSE end-of-day F&O "bhavcopy" (UDiFF format) as a free source of option closes, lot sizes and expiries.

Used by intraday_dashboard.py (Options lab tab) and research/options_backtest.py. Educational estimates:
real premiums move with bid-ask spreads and changing implied volatility that this model does not capture.
The JS in intraday_dashboard.py duplicates bs_price/greeks/option_charges - change both together.
"""
from __future__ import annotations

import datetime as dt
import io
import math
import os
import urllib.request
import zipfile

import pandas as pd

RISK_FREE = 0.065            # approx. Indian 91-day T-bill rate; has little effect on short-dated options
MIN_YEAR = 525600.0          # minutes in a year (calendar time)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "fo")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/130.0 Safari/537.36", "Referer": "https://www.nseindia.com/", "Accept": "*/*"}
BHAV_URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d}_F_0000.csv.zip"
INDEX_UNDERLYINGS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}


# ------------------------------------------------------------------------------------------------
# Pricing
# ------------------------------------------------------------------------------------------------
def _N(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _n(x: float) -> float:
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def bs_price(S: float, K: float, T: float, sigma: float, cp: str, r: float = RISK_FREE) -> float:
    """Black-Scholes price of a European call ('CE') or put ('PE'). T in years."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if cp == "CE" else max(0.0, K - S)
    st = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + sigma * sigma / 2) * T) / st
    d2 = d1 - st
    if cp == "CE":
        return S * _N(d1) - K * math.exp(-r * T) * _N(d2)
    return K * math.exp(-r * T) * _N(-d2) - S * _N(-d1)


def greeks(S: float, K: float, T: float, sigma: float, cp: str, r: float = RISK_FREE) -> dict:
    """Delta, gamma, theta (rupees per calendar day) and vega (rupees per 1 point of volatility)."""
    T = max(T, 1 / MIN_YEAR)
    st = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + sigma * sigma / 2) * T) / st
    d2 = d1 - st
    common = -S * _n(d1) * sigma / (2 * math.sqrt(T))
    if cp == "CE":
        delta, theta = _N(d1), common - r * K * math.exp(-r * T) * _N(d2)
    else:
        delta, theta = _N(d1) - 1, common + r * K * math.exp(-r * T) * _N(-d2)
    return dict(delta=delta, gamma=_n(d1) / (S * st), theta=theta / 365, vega=S * _n(d1) * math.sqrt(T) / 100)


def implied_vol(price: float, S: float, K: float, T: float, cp: str, r: float = RISK_FREE):
    """Volatility that reproduces `price`, by bisection; None when the price is below intrinsic value."""
    if not (price > 0 and S > 0 and K > 0 and T > 0):
        return None
    intrinsic = max(0.0, S - K * math.exp(-r * T)) if cp == "CE" else max(0.0, K * math.exp(-r * T) - S)
    if price <= intrinsic + 1e-9:
        return None
    lo, hi = 0.005, 5.0
    if bs_price(S, K, T, hi, cp, r) < price:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        if bs_price(S, K, T, mid, cp, r) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def years_to_expiry(now: dt.datetime, expiry: dt.date) -> float:
    """Calendar time from `now` (naive IST) to 15:30 IST on the expiry day, in years."""
    end = dt.datetime.combine(expiry, dt.time(15, 30))
    return max(0.0, (end - now).total_seconds() / 60 / MIN_YEAR)


# ------------------------------------------------------------------------------------------------
# Charges (Zerodha-style NSE options, per the published schedule in Nov 2026)
# ------------------------------------------------------------------------------------------------
def option_charges(buy_prem: float, sell_prem: float, orders: int = 2, slip_pct: float = 0.0) -> dict:
    """buy_prem / sell_prem = premium value in rupees (price x quantity) on each side."""
    turnover = buy_prem + sell_prem
    brokerage = 20.0 * orders
    stt = 0.001 * sell_prem                     # 0.1% of premium on the sell side
    txn = 0.0003503 * turnover                  # NSE exchange transaction charge on premium
    sebi = turnover * 10 / 1e7
    stamp = 0.00003 * buy_prem
    gst = 0.18 * (brokerage + txn + sebi)
    slip = slip_pct / 100 * turnover
    total = brokerage + stt + txn + sebi + stamp + gst + slip
    return dict(brokerage=brokerage, stt=stt, txn=txn, sebi=sebi, stamp=stamp, gst=gst, slippage=slip, total=total)


# ------------------------------------------------------------------------------------------------
# NSE F&O bhavcopy
# ------------------------------------------------------------------------------------------------
def _parse_bhav(raw: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(raw))
    df = pd.read_csv(z.open(z.namelist()[0]))
    df = df[df["FinInstrmTp"].isin(["IDO", "STO"])]                   # index and stock options
    out = pd.DataFrame(dict(
        symbol=df["TckrSymb"].astype(str), expiry=pd.to_datetime(df["XpryDt"]).dt.date,
        strike=df["StrkPric"].astype(float), cp=df["OptnTp"].astype(str), close=df["ClsPric"].astype(float),
        settle=df["SttlmPric"].astype(float), spot=df["UndrlygPric"].astype(float), oi=df["OpnIntrst"].astype(float),
        volume=df["TtlTradgVol"].astype(float), lot=df["NewBrdLotQty"].astype(float)))
    out["date"] = pd.to_datetime(df["TradDt"]).dt.date.values
    return out.reset_index(drop=True)


def bhavcopy(day: dt.date, cache: bool = True):
    """Option rows for one trading day, or None if NSE has no file for it (holiday/weekend/not yet published)."""
    d = day.strftime("%Y%m%d")
    path = os.path.join(CACHE_DIR, f"fo_{d}.zip")
    raw = None
    if cache and os.path.exists(path):
        with open(path, "rb") as f:
            raw = f.read()
    if raw is None:
        try:
            with urllib.request.urlopen(urllib.request.Request(BHAV_URL.format(d=d), headers=UA), timeout=40) as r:
                raw = r.read()
        except Exception:  # noqa: BLE001  (404 on non-trading days)
            return None
        if not raw.startswith(b"PK"):
            return None
        if cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(path, "wb") as f:
                f.write(raw)
    return _parse_bhav(raw)


def latest_bhavcopy(today: dt.date, lookback: int = 8):
    for k in range(lookback):
        df = bhavcopy(today - dt.timedelta(days=k))
        if df is not None and len(df):
            return df
    return None


def nse_symbol(sym: str) -> str:
    """Watchlist name -> NSE F&O ticker (the same for nearly all, e.g. M&M, BAJAJ-AUTO, NIFTY)."""
    return sym


def chain(bhav: pd.DataFrame, sym: str, n_exp: int = 2, width: float = 0.12) -> dict | None:
    """Strikes within +-width of the underlying for the next n_exp expiries, with closes and implied vols."""
    rows = bhav[bhav["symbol"] == nse_symbol(sym)]
    if rows.empty:
        return None
    day = rows["date"].iloc[0]
    spot = float(rows["spot"].median())
    when = dt.datetime.combine(day, dt.time(15, 30))
    exps = sorted(e for e in rows["expiry"].unique() if e > day)[:n_exp]      # expiry-day contracts are settled
    out = []
    for e in exps:
        r = rows[(rows["expiry"] == e) & (rows["strike"].between(spot * (1 - width), spot * (1 + width)))]
        T = years_to_expiry(when, e)
        strikes = []
        for k, g in r.groupby("strike"):
            row = dict(strike=float(k))
            for cp in ("CE", "PE"):
                x = g[g["cp"] == cp]
                if len(x):
                    px = float(x["close"].iloc[0])
                    iv = implied_vol(px, spot, float(k), T, cp) if T > 0 else None
                    row[cp] = dict(close=round(px, 2), oi=int(x["oi"].iloc[0]), vol=int(x["volume"].iloc[0]),
                                   iv=round(iv * 100, 1) if iv else None)
            strikes.append(row)
        out.append(dict(expiry=str(e), days=round(T * 365, 2), strikes=strikes))
    return dict(symbol=sym, asof=str(day), spot=round(spot, 2), lot=int(rows["lot"].iloc[0]),
                index=sym in INDEX_UNDERLYINGS, expiries=out)


def atm_iv(ch: dict, expiry_i: int = 0):
    """Average of call and put implied vol at the strike nearest the underlying (percent)."""
    try:
        e = ch["expiries"][expiry_i]
    except (IndexError, KeyError, TypeError):
        return None
    if not e["strikes"]:
        return None
    s = min(e["strikes"], key=lambda r: abs(r["strike"] - ch["spot"]))
    ivs = [s[c]["iv"] for c in ("CE", "PE") if c in s and s[c]["iv"]]
    return round(sum(ivs) / len(ivs), 1) if ivs else None
