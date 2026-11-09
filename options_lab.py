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


# ------------------------------------------------------------------------------------------------
# Probability screener: hold-to-expiry outcomes of simple option setups under a lognormal model
# ------------------------------------------------------------------------------------------------
# Model (fixed in advance): price at expiry is lognormal with NO directional drift (mean = today's price) and
# volatility = the underlying's realised 20-day volatility. Option prices are NSE closes. If realised volatility
# equals implied volatility, every setup's expected value is about -(charges + bid-ask): an edge can only come
# from the gap between the two (and from the volatility skew). Exits at expiry at intrinsic value.
import numpy as np  # noqa: E402
from statistics import NormalDist  # noqa: E402

SLIP_PCT = {"index": 0.5, "stock": 2.0}          # bid-ask allowance per side, % of premium
_ZG = np.array([NormalDist().inv_cdf((i + 0.5) / 801) for i in range(801)])   # equal-probability grid
SETUPS = {
    "long_call": "Buy call", "long_put": "Buy put",
    "bull_call": "Bull call spread", "bear_put": "Bear put spread",
    "bull_put": "Bull put spread (credit)", "bear_call": "Bear call spread (credit)",
}


def terminal_prices(S: float, sigma: float, T: float) -> np.ndarray:
    return S * np.exp(-0.5 * sigma * sigma * T + sigma * math.sqrt(max(T, 0.0)) * _ZG)


def _charges_vec(buy_val, sell_val, orders):
    turnover = buy_val + sell_val
    br = 20.0 * orders
    txn = 0.0003503 * turnover
    sebi = turnover * 10 / 1e7
    return br + 0.001 * sell_val + txn + sebi + 0.00003 * buy_val + 0.18 * (br + txn + sebi)


def evaluate_legs(legs, S: float, T: float, sigma: float, lot: int, slip_pct: float) -> dict:
    """legs: [dict(cp, K, side (+1 buy / -1 sell), price)]. Profit per lot at expiry, after charges and bid-ask."""
    ST = terminal_prices(S, sigma, T)
    pnl = np.zeros_like(ST)
    net = 0.0
    for g in legs:
        intr = np.maximum(ST - g["K"], 0) if g["cp"] == "CE" else np.maximum(g["K"] - ST, 0)
        entry, exit_ = g["price"] * lot, intr * lot
        orders = 1 + (intr > 0)                                   # worthless legs simply expire
        if g["side"] > 0:
            ch = _charges_vec(entry, exit_, orders)
        else:
            ch = _charges_vec(exit_, entry, orders)
        pnl += g["side"] * (exit_ - entry) - ch - slip_pct / 100 * (entry + exit_)
        net += g["side"] * g["price"]
    win, lose = pnl[pnl > 0], pnl[pnl <= 0]
    kelly = kelly_fraction(pnl)
    grid = np.linspace(S * 0.6, S * 1.4, 2001)
    pg = sum(g["side"] * ((np.maximum(grid - g["K"], 0) if g["cp"] == "CE" else np.maximum(g["K"] - grid, 0)) - g["price"])
             for g in legs)
    be = [float(grid[i] + (grid[i + 1] - grid[i]) * (-pg[i]) / (pg[i + 1] - pg[i]))
          for i in range(len(grid) - 1) if pg[i] * pg[i + 1] < 0]
    return dict(ev=float(pnl.mean()), pop=float((pnl > 0).mean()), avg_win=float(win.mean()) if len(win) else 0.0,
                avg_loss=float(lose.mean()) if len(lose) else 0.0, worst=float(pnl.min()), best=float(pnl.max()),
                net=net, breakevens=be, kelly=kelly)


def kelly_fraction(pnl: np.ndarray) -> float:
    """Share of capital to put at risk (= max loss) per trade that maximises expected log growth, for the given
    outcome distribution. 0 when the expected value is not positive. Use a fraction (e.g. 1/4) in practice:
    the distribution itself is only an estimate."""
    worst = float(pnl.min())
    if worst >= 0 or pnl.mean() <= 0:
        return 0.0
    x = pnl / -worst                                     # outcome per rupee at risk, >= -1
    fs = np.linspace(0, 0.999, 1000)
    g = np.array([np.mean(np.log1p(f * x)) for f in fs])
    return float(fs[int(g.argmax())])


def setup_legs(kind: str, exp: dict, S: float, iv: float, T: float):
    """Choose strikes by a fixed rule: long legs at the money, other legs at fractions of the 1-sigma move."""
    rows = [r for r in exp["strikes"]]
    if not rows or T <= 0:
        return None
    em = S * iv * math.sqrt(T)

    def pick(cp, target, side):
        ok = [r for r in rows if cp in r and r[cp]["close"] >= 0.5 and r[cp]["vol"] > 0]
        if not ok:
            return None
        r = min(ok, key=lambda x: abs(x["strike"] - target))
        return dict(cp=cp, K=r["strike"], side=side, price=r[cp]["close"])
    L = {
        "long_call": [("CE", S, 1)], "long_put": [("PE", S, 1)],
        "bull_call": [("CE", S, 1), ("CE", S + em, -1)], "bear_put": [("PE", S, 1), ("PE", S - em, -1)],
        "bull_put": [("PE", S - 0.5 * em, -1), ("PE", S - 1.5 * em, 1)],
        "bear_call": [("CE", S + 0.5 * em, -1), ("CE", S + 1.5 * em, 1)],
    }[kind]
    legs = [pick(cp, tgt, sd) for cp, tgt, sd in L]
    if any(x is None for x in legs):
        return None
    if len(legs) == 2 and legs[0]["K"] == legs[1]["K"]:
        return None                                            # strikes too coarse for this spread
    return legs


def screen_chain(ch: dict, hv: float, min_days: float = 2.0, as_of: dt.datetime | None = None) -> list:
    """All setups for one underlying's nearest expiry with at least `min_days` left. hv/iv as fractions."""
    if not ch or not hv:
        return []
    exp = next((e for e in ch["expiries"] if e["days"] >= min_days), None)
    if exp is None:
        return []
    S = ch["spot"]
    when = as_of or dt.datetime.combine(dt.date.fromisoformat(ch["asof"]), dt.time(15, 30))
    T = years_to_expiry(when, dt.date.fromisoformat(exp["expiry"]))
    iv = (atm_iv(dict(ch, expiries=[exp])) or hv * 100) / 100
    slip = SLIP_PCT["index" if ch["index"] else "stock"]
    out = []
    for kind in SETUPS:
        legs = setup_legs(kind, exp, S, iv, T)
        if not legs:
            continue
        r = evaluate_legs(legs, S, T, hv, ch["lot"], slip)
        risk = max(1.0, -r["worst"])
        out.append(dict(symbol=ch["symbol"], index=ch["index"], kind=kind, setup=SETUPS[kind], expiry=exp["expiry"],
                        days=exp["days"], spot=S, lot=ch["lot"], iv=round(iv * 100, 1), hv=round(hv * 100, 1),
                        legs=legs, net=round(r["net"], 2), net_lot=round(r["net"] * ch["lot"], 0),
                        limit=round(r["net"] + r["ev"] / ch["lot"], 2), ev=round(r["ev"], 0), pop=round(r["pop"] * 100, 1),
                        avg_win=round(r["avg_win"], 0), avg_loss=round(r["avg_loss"], 0), max_loss=round(-r["worst"], 0),
                        best=round(r["best"], 0), ev_per_risk=round(r["ev"] / risk * 100, 2), kelly=round(r["kelly"], 4),
                        breakevens=[round(b, 2) for b in r["breakevens"]]))
    return out
