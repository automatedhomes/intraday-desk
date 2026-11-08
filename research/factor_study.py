"""Study: published factor rules (NSE-style momentum, low volatility, momentum + low volatility) on the
watchlist, after costs AND Indian capital-gains tax, vs holding the Nifty or the same stocks. Needs network.

Rules are taken from public index methodologies, not tuned here:
  * NSE momentum score (Nifty200 Momentum 30 style): average of cross-sectional z-scores of 6- and 12-month
    price return divided by 1-year volatility. Top 10, rebalanced every 6 months (June, December).
  * Low volatility (Nifty Low Volatility 30 style): the 10 stocks with the lowest 1-year daily volatility,
    rebalanced every quarter.
  * Momentum + low volatility (Alpha Low-Volatility style, alpha replaced by the momentum score):
    average z-score of the two, top 10, every 6 months.
  * Reference: plain 12-1 month momentum, top 10, monthly (from research/swing_momentum.py).
Signals use the previous close; trades at the rebalance day's close; 0.15% cost per side.
Tax: 20% short-term (held < 365 days), 12.5% long-term, losses offset gains and carry forward, settled
each 31 Dec, everything sold and taxed at the end. The Rs 1.25 lakh LTCG exemption is ignored.
CAVEAT: today's large caps = survivorship bias. Compare with "hold the same stocks" first.

Run from the project root:  python research/factor_study.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swing_momentum import COST, HALVES, START, SYMS, load, stats  # noqa: E402

ST_TAX, LT_TAX = 0.20, 0.125


def schedule(days, months):
    s = pd.Series(days, index=days)
    last = s.groupby(days.to_period("M")).last()
    return {d for d in last if d.month in months}


def simulate(C, rebal, target, cost=COST, tax=True):
    """Portfolio value per day (pre-tax marks) and the final value after selling everything and paying tax."""
    days = C.index[C.index >= pd.Timestamp(START)]
    units, basis, since = {}, {}, {}
    cash, gains, carry, traded = 1.0, {"st": 0.0, "lt": 0.0}, 0.0, 0.0
    vals = pd.Series(np.nan, index=days)

    def port_value(px):
        return max(1e-12, cash + sum(units[s] * px[s] for s in units))

    def realise(s, frac, t, price, V):
        nonlocal cash, traded
        v = frac * units[s] * price
        g = v - frac * basis[s]
        gains["lt" if (t - since[s]).days >= 365 else "st"] += g
        cash += v * (1 - cost)
        traded += v / V
        units[s] *= 1 - frac
        basis[s] *= 1 - frac
        if frac >= 1 - 1e-12:
            del units[s], basis[s], since[s]

    def settle():
        nonlocal carry
        st, lt = gains["st"] + carry, gains["lt"]
        carry = 0.0
        if st < 0:
            lt, st = lt + st, 0.0
        if lt < 0:
            carry, lt = lt, 0.0
        gains["st"] = gains["lt"] = 0.0
        return ST_TAX * st + LT_TAX * lt if tax else 0.0

    year = days[0].year
    for t in days:
        px = C.loc[t]
        if t.year != year:
            due, year = settle(), t.year
            hold = sum(units[s] * px[s] for s in units)
            if due > 0 and hold > 0:
                take = min(1.0, max(0.0, (due - cash) / hold)) if due > cash else 0.0
                cash -= due
                for s in list(units):
                    if take > 0:
                        cash += take * units[s] * px[s]
                        units[s] *= 1 - take
                        basis[s] *= 1 - take
        if t in rebal:
            w = target(t)
            V = cash + sum(units[s] * px[s] for s in units)
            for s in list(units):
                cv, tv = units[s] * px[s], V * float(w.get(s, 0.0))
                if cv > tv + 1e-12:
                    realise(s, (cv - tv) / cv, t, px[s], V)
            buys = {s: V * ws - units.get(s, 0.0) * px[s] for s, ws in w.items() if ws > 0}
            buys = {s: b for s, b in buys.items() if b > 1e-12}
            need = sum(buys.values())
            scale = min(1.0, cash / need) if need > 0 else 0.0
            V = port_value(px)
            for s, b in buys.items():
                b *= scale
                if s not in units:
                    units[s], basis[s], since[s] = 0.0, 0.0, t
                units[s] += b * (1 - cost) / px[s]
                basis[s] += b
                cash -= b
                traded += b / V
        vals[t] = cash + sum(units[s] * px[s] for s in units)
    t, px = days[-1], C.loc[days[-1]]
    V = port_value(px)
    for s in list(units):
        realise(s, 1.0, t, px[s], V)
    final = cash - settle()
    yrs = (days[-1] - days[0]).days / 365.25
    return vals, final, traded / 2 / yrs * 100


def zs(x):
    return (x - x.mean()) / x.std()


def main():
    data = load()
    C = pd.DataFrame({s: data[s]["Close"] for s in SYMS if s in data}).ffill(limit=5)
    nifty = data["NIFTY"]["Close"].reindex(C.index).ffill()
    rets = C.pct_change()
    vol = (rets.rolling(252).std() * np.sqrt(252)).shift(1)
    r6, r12 = (C / C.shift(126) - 1).shift(1), (C / C.shift(252) - 1).shift(1)
    m121 = (C.shift(21) / C.shift(252) - 1).shift(1)
    days = C.index[C.index >= pd.Timestamp(START)]
    monthly, quarterly, semi = schedule(days, range(1, 13)), schedule(days, (3, 6, 9, 12)), schedule(days, (6, 12))
    first = days[0]

    def top(score, k):
        s = score.dropna()
        if len(s) < k:
            return pd.Series(dtype=float)
        return pd.Series(1 / k, index=s.nlargest(k).index)

    def nse_mom(t):
        ok = vol.loc[t].notna() & r12.loc[t].notna()
        return (zs((r6.loc[t] / vol.loc[t])[ok]) + zs((r12.loc[t] / vol.loc[t])[ok])) / 2

    strategies = {
        "NSE momentum, top 10, 6-monthly": (semi, lambda t, k=10: top(nse_mom(t), k)),
        "NSE momentum, top 15, 6-monthly": (semi, lambda t: top(nse_mom(t), 15)),
        "Low volatility, top 10, quarterly": (quarterly, lambda t: top(-vol.loc[t][r12.loc[t].notna()], 10)),
        "Low volatility, top 15, quarterly": (quarterly, lambda t: top(-vol.loc[t][r12.loc[t].notna()], 15)),
        "Momentum + low vol, top 10, 6-mthly": (semi, lambda t: top((zs(nse_mom(t)) + zs(-vol.loc[t][nse_mom(t).index])) / 2, 10)),
        "Momentum + low vol, top 15, 6-mthly": (semi, lambda t: top((zs(nse_mom(t)) + zs(-vol.loc[t][nse_mom(t).index])) / 2, 15)),
        "12-1m momentum, top 10, monthly": (monthly, lambda t: top(m121.loc[t], 10)),
    }
    start_ok = C.loc[first].notna()
    bench = {
        "Nifty 50 index, hold (no dividends)": (pd.DataFrame({"N": nifty}), {first}, lambda t: pd.Series({"N": 1.0})),
        "Hold the same stocks (equal, no rebal.)": (C, {first}, lambda t: pd.Series(1 / start_ok.sum(), index=start_ok[start_ok].index)),
    }

    yrs = (days[-1] - first).days / 365.25
    print(f"{C.shape[1]} stocks, {first.date()} to {days[-1].date()} ({yrs:.1f} years). Costs {COST * 100:.2f}%/side, "
          f"tax {ST_TAX * 100:.0f}% short / {LT_TAX * 100:.1f}% long term.\n")
    print(f"{'':<40}{'Before costs':>13}{'After costs':>12}{'After tax':>10}{'Max DD':>8}"
          + "".join(f"{h[0]:>10}" for h in HALVES) + f"{'Turnover':>10}")
    rows = list(bench.items()) + [(k, (C, *v)) for k, v in strategies.items()]
    for name, (P, rb, fn) in rows:
        g, _, _ = simulate(P, rb, fn, cost=0.0, tax=False)
        v, _, to = simulate(P, rb, fn, tax=False)
        _, final, _ = simulate(P, rb, fn)
        r = v.pct_change()
        halves = "".join(f"{stats(r[(r.index >= a) & (r.index <= b)])['cagr']:>9.1f}%" for _, a, b in HALVES)
        print(f"{name:<40}{stats(g.pct_change())['cagr']:>12.1f}%{stats(r)['cagr']:>11.1f}%"
              f"{(final ** (1 / yrs) - 1) * 100:>9.1f}%{stats(r)['maxdd']:>7.0f}%{halves}{to:>9.0f}%")
    print("\nCAGR per year. 'After tax' = everything sold and taxed at the end. Halves are after costs, before tax.")
    print("Survivorship bias: today's large caps. Past results, not advice.")


if __name__ == "__main__":
    main()
