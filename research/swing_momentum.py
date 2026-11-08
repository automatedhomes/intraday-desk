"""Study: three swing/positional ideas vs buy-and-hold on the watchlist (daily data, 2016 onward). Needs network.

Rules were fixed BEFORE looking at results (no tuning). Costs 0.15% per side (delivery STT 0.1%, stamp,
exchange, GST, DP charge and slippage, roughly). Results are shown before and after costs.

  1. Pullback   close > SMA200 and RSI14 < 40 -> buy next open. Exit next open after RSI14 > 55,
                stop 2 x ATR14 below entry (intraday), or close of day 15.
  2. Breakout   close > highest high of the previous 50 days and close > SMA200 -> buy next open.
                Exit next open after close < highest close since entry - 3 x ATR14.
  3. Momentum   at each month end rank stocks by 12-1 month return (t-252 .. t-21), hold the top 10
                equally for the next month. Variant "+filter": hold cash when Nifty < its SMA200.
                Sensitivity grid (reported, not used to pick a winner): top 5/10/15, 6-1 and 12-1 months.

Portfolios: each stock gets an equal share of capital; idle cash earns 0% (conservative).
CAVEAT: the universe is TODAY's large caps, so anything that holds them (buy-and-hold, momentum) is
flattered by survivorship bias. Compare strategies with the equal-weight hold of the same stocks first.

Run from the project root:  python research/swing_momentum.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intraday_dashboard as idd  # noqa: E402

COST = 0.0015
WARM, START = "2014-01-01", "2016-01-01"
HALVES = (("2016-2020", "2016-01-01", "2020-12-31"), ("2021-now", "2021-01-01", "2100-01-01"))
SYMS = [s for s in idd.WATCHLIST if s not in ("NIFTY", "BANKNIFTY")]


def load():
    tick = {s: idd.resolve(s) for s in SYMS}
    extra = {"NIFTY": "^NSEI", "NIFTYBEES": "NIFTYBEES.NS"}
    raw = yf.download(list(tick.values()) + list(extra.values()), start=WARM, interval="1d", auto_adjust=True,
                      group_by="ticker", threads=True, progress=False)
    out = {}
    for s, t in {**tick, **extra}.items():
        try:
            df = raw[t][["Open", "High", "Low", "Close"]].dropna()
        except KeyError:
            continue
        idx = pd.to_datetime(df.index)
        df.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        df = df[(df > 0).all(axis=1)]
        if len(df) > 300:
            out[s] = df
    return out


def indicators(df):
    d = df.copy()
    c = d["Close"]
    d["SMA200"] = c.rolling(200).mean()
    delta = c.diff()
    g = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    lo = (-delta).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI"] = 100 * g / (g + lo)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - c.shift()).abs(), (d["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["HH50"] = d["High"].rolling(50).max().shift(1)
    return d


def sleeve(d, kind, cost):
    """Daily equity of one stock's sleeve (1 when flat at start) and its closed trades."""
    o, h, l, c = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    sma, rsi, atr, hh = (d[k].to_numpy() for k in ("SMA200", "RSI", "ATR", "HH50"))
    idx, n = d.index, len(d)
    ok = np.where((idx >= pd.Timestamp(START)) & ~np.isnan(sma))[0]
    E = np.full(n, np.nan)
    if not len(ok):
        return E, []
    eq, inpos, pend_in, pend_out = 1.0, False, False, False
    entry = stop = peak = 0.0
    ei = 0
    trades = []
    for i in range(ok[0], n):
        ref = c[i - 1]
        if pend_in and not inpos:
            entry, ei, inpos, pend_in = o[i], i, True, False
            eq *= 1 - cost
            ref = o[i]
            stop, peak = entry - 2 * atr[i - 1], entry
        if inpos:
            px = None
            if pend_out:
                px = o[i]
            elif kind == "pullback" and l[i] <= stop:
                px = min(o[i], stop)
            elif kind == "pullback" and i - ei >= 15:
                px = c[i]
            if px is not None:
                eq *= px / ref * (1 - cost)
                trades.append((px / entry - 1, px * (1 - cost) ** 2 / entry - 1, i - ei))
                inpos, pend_out = False, False
            else:
                eq *= c[i] / ref
                if kind == "pullback" and rsi[i] > 55:
                    pend_out = True
                if kind == "breakout":
                    peak = max(peak, c[i])
                    if c[i] < peak - 3 * atr[i]:
                        pend_out = True
        E[i] = eq
        if not inpos and not pend_in and i < n - 1 and c[i] > sma[i]:
            if (kind == "pullback" and rsi[i] < 40) or (kind == "breakout" and c[i] > hh[i]):
                pend_in = True
    return E, trades


def stats(r):
    r = r.dropna()
    if r.empty:
        return dict(cagr=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan)
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    return dict(cagr=(eq.iloc[-1] ** (1 / yrs) - 1) * 100, vol=r.std() * np.sqrt(252) * 100,
                sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
                maxdd=((eq / eq.cummax()) - 1).min() * 100)


def momentum(C, nifty, k, look, filt, cost):
    rets = C.pct_change()
    score = (C.shift(21) / C.shift(look) - 1).shift(1)            # known at the previous close
    trend_ok = (nifty > nifty.rolling(200).mean()).reindex(C.index).ffill().shift(1)
    days = C.index[C.index >= pd.Timestamp(START)]
    month_end = pd.Series(days, index=days).groupby(days.to_period("M")).last().tolist()
    w = pd.Series(0.0, index=C.columns)
    out, turn = pd.Series(0.0, index=days), []
    me = set(month_end)
    for t in days:
        out[t] = float((w * rets.loc[t].fillna(0)).sum())
        if t in me:
            s = score.loc[t].dropna()
            new = pd.Series(0.0, index=C.columns)
            if len(s) >= k and not (filt and not bool(trend_ok.get(t, True))):
                new[s.nlargest(k).index] = 1 / k
            tv = float((new - w).abs().sum())
            out[t] -= tv * cost
            turn.append(tv)
            w = new
    return out, float(np.mean(turn)) * 12 / 2 * 100 if turn else 0.0     # one-way turnover %/yr


def fmt_row(name, r, extra=""):
    full = stats(r)
    parts = [f"{name:<34}{full['cagr']:>7.1f}%{full['maxdd']:>8.1f}%{full['sharpe']:>7.2f}"]
    for _, a, b in HALVES:
        parts.append(f"{stats(r[(r.index >= a) & (r.index <= b)])['cagr']:>9.1f}%")
    return "".join(parts) + ("  " + extra if extra else "")


def main():
    data = load()
    stocks = {s: indicators(df) for s, df in data.items() if s in SYMS}
    bench = data["NIFTY"]   # NIFTYBEES has a bad Yahoo split adjustment in Dec 2019 (-90% then +897%)
    print(f"{len(stocks)} stocks, daily data from {START} to {max(d.index[-1] for d in stocks.values()).date()}")
    print(f"Costs {COST * 100:.2f}% per side. Idle cash earns 0%.\n")
    head = f"{'':<34}{'CAGR':>8}{'Max DD':>9}{'Sharpe':>7}" + "".join(f"{h[0]:>10}" for h in HALVES)

    # equal-weight hold of the same stocks, from the date each has a 200-day history (same as the sleeves)
    ew = pd.DataFrame({s: d["Close"].pct_change().where(d["SMA200"].notna() & (d.index >= pd.Timestamp(START)))
                       for s, d in stocks.items()}).mean(axis=1)
    b = bench["Close"].pct_change()
    b = b[b.index >= pd.Timestamp(START)]
    print(head)
    print(fmt_row("Nifty 50 index, hold (no divs)", b, "add ~1.3%/yr for dividends"))
    print(fmt_row("Equal-weight hold, these stocks", ew))
    print()

    for kind in ("pullback", "breakout"):
        for label, cost in (("before costs", 0.0), ("after costs", COST)):
            sl, tr = {}, []
            for s, d in stocks.items():
                E, t = sleeve(d, kind, cost)
                sl[s] = pd.Series(E, index=d.index).pct_change()
                tr += t
            port = pd.DataFrame(sl).mean(axis=1)
            g = np.array([x[0] for x in tr]) * 100
            net = np.array([x[1] for x in tr]) * 100
            days = np.mean([x[2] for x in tr]) if tr else 0
            extra = ""
            if cost:
                extra = (f"{len(tr)} trades, win {np.mean(net > 0) * 100:.0f}%, avg/trade gross {g.mean():+.2f}% "
                         f"net {net.mean():+.2f}%, held {days:.0f}d")
            print(fmt_row(f"{kind.title()} ({label})", port, extra))
    print()

    C = pd.DataFrame({s: d["Close"] for s, d in stocks.items()})
    nifty = data["NIFTY"]["Close"]
    for filt in (False, True):
        for look, ln in ((252, "12-1m"), (126, "6-1m")):
            for k in (5, 10, 15):
                g, _ = momentum(C, nifty, k, look, filt, 0.0)
                r, to = momentum(C, nifty, k, look, filt, COST)
                star = " <- pre-chosen" if (k == 10 and look == 252) else ""
                print(fmt_row(f"Momentum top{k} {ln}{' +filter' if filt else ''}", r,
                              f"before costs {stats(g)['cagr']:.1f}%/yr, turnover {to:.0f}%/yr{star}"))
    print("\nAll returns after costs unless marked. Past results on today's large caps (survivorship bias); not advice.")


if __name__ == "__main__":
    main()
