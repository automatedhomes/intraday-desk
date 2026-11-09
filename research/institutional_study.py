"""Study: institutional portfolio techniques applied to the factor ideas, after costs and Indian tax. Needs network.

Techniques (rules fixed before running, no tuning):
  * Banding (turnover buffer): buy the top 10 by 12-1m momentum, but keep a holding until it falls below
    rank 20. Large managers use buffers like this to cut trading costs and short-term tax.
  * Two factor sleeves: 50% momentum (banded, monthly) + 50% low volatility (top 10, re-ranked quarterly),
    rebalanced monthly. Diversifying across factors that crash at different times.
  * Core-satellite: 70% Nifty 50 index + 30% satellite (banded momentum, or both sleeves). The core keeps
    costs and tax low; the satellite is where any factor edge would come from.
Same engine as research/factor_study.py: 0.15%/side, 20% short-term / 12.5% long-term tax, settled yearly.
The Nifty here is the price index (no dividends, ~1.3%/yr), which understates every core-satellite result.
CAVEAT: today's large caps = survivorship bias.

Run from the project root:  python research/institutional_study.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from factor_study import HALVES, schedule, simulate  # noqa: E402
from swing_momentum import START, SYMS, load, stats  # noqa: E402

K, BAND = 10, 20


def main():
    data = load()
    C = pd.DataFrame({s: data[s]["Close"] for s in SYMS if s in data}).ffill(limit=5)
    C["NIFTY"] = data["NIFTY"]["Close"].reindex(C.index).ffill()
    S = C.drop(columns="NIFTY")
    vol = (S.pct_change().rolling(252).std() * np.sqrt(252)).shift(1)
    m121 = (S.shift(21) / S.shift(252) - 1).shift(1)
    r12 = (S / S.shift(252) - 1).shift(1)
    days = C.index[C.index >= pd.Timestamp(START)]
    monthly = schedule(days, range(1, 13))
    first = days[0]

    def eq(names, w):
        return pd.Series(w / len(names), index=list(names)) if len(names) else pd.Series(dtype=float)

    def plain_mom():
        return lambda t: eq(m121.loc[t].dropna().nlargest(K).index, 1.0)

    def banded_mom_picker():
        held = []

        def pick(t):
            s = m121.loc[t].dropna()
            if len(s) < BAND:
                return []
            rank = s.rank(ascending=False)
            keep = [x for x in held if x in rank.index and rank[x] <= BAND]
            for x in s.sort_values(ascending=False).index:
                if len(keep) >= K:
                    break
                if x not in keep:
                    keep.append(x)
            held[:] = keep[:K]
            return list(held)
        return pick

    def lowvol_picker():
        held = []

        def pick(t):
            if t.month in (3, 6, 9, 12) or not held:
                s = -vol.loc[t][r12.loc[t].notna()].dropna()
                if len(s) >= K:
                    held[:] = list(s.nlargest(K).index)
            return list(held)
        return pick

    def banded_mom(core=0.0):
        def make():
            mp = banded_mom_picker()

            def target(t):
                w = eq(mp(t), 1 - core)
                return pd.concat([w, pd.Series({"NIFTY": core})]) if core else w
            return target
        return make

    def sleeves(core=0.0):
        def make():
            mp, lp = banded_mom_picker(), lowvol_picker()

            def target(t):
                half = (1 - core) / 2
                w = eq(mp(t), half).add(eq(lp(t), half), fill_value=0)
                return pd.concat([w, pd.Series({"NIFTY": core})]) if core else w
            return target
        return make

    start_ok = S.loc[first].notna()
    rows = {
        "Nifty 50 index, hold (no dividends)": ({first}, lambda: (lambda t: pd.Series({"NIFTY": 1.0}))),
        "Hold the same stocks (equal, no rebal.)": ({first}, lambda: (lambda t: eq(start_ok[start_ok].index, 1.0))),
        "12-1m momentum top 10, monthly (before)": (monthly, plain_mom),
        "+ banding (buy top 10, sell below 20)": (monthly, banded_mom()),
        "Two sleeves: 50% momentum + 50% low vol": (monthly, sleeves()),
        "Core 70% Nifty + 30% banded momentum": (monthly, banded_mom(core=0.7)),
        "Core 70% Nifty + 30% two sleeves": (monthly, sleeves(core=0.7)),
    }
    yrs = (days[-1] - first).days / 365.25
    print(f"{S.shape[1]} stocks + Nifty, {first.date()} to {days[-1].date()} ({yrs:.1f} years).\n")
    print(f"{'':<42}{'Before costs':>13}{'After costs':>12}{'After tax':>10}{'Max DD':>8}{'Vol':>6}"
          + "".join(f"{h[0]:>10}" for h in HALVES) + f"{'Turnover':>10}")
    for name, (rb, make) in rows.items():
        g, _, _ = simulate(C, rb, make(), cost=0.0, tax=False)
        v, _, to = simulate(C, rb, make(), tax=False)
        _, final, _ = simulate(C, rb, make())
        r = v.pct_change()
        st = stats(r)
        halves = "".join(f"{stats(r[(r.index >= a) & (r.index <= b)])['cagr']:>9.1f}%" for _, a, b in HALVES)
        print(f"{name:<42}{stats(g.pct_change())['cagr']:>12.1f}%{st['cagr']:>11.1f}%"
              f"{(final ** (1 / yrs) - 1) * 100:>9.1f}%{st['maxdd']:>7.0f}%{st['vol']:>5.0f}%{halves}{to:>9.0f}%")
    print("\nCAGR per year. Vol = yearly volatility. Halves after costs, before tax. Past results, not advice.")

    # Robustness, not selection: does banding help across nearby settings too?
    global K, BAND
    print("\nBanding sensitivity (after tax CAGR; 'none' = sell as soon as it leaves the top K):")
    for k in (10, 15):
        cells = []
        for band in (None, 15, 20, 25, 30):
            if band is not None and band <= k:
                cells.append(f"{'-':>8}")
                continue
            K, BAND = k, (band or k)
            _, final, _ = simulate(C, monthly, banded_mom()())
            cells.append(f"{(final ** (1 / yrs) - 1) * 100:>7.1f}%")
        print(f"  top {k:<3} sell below: none / 15 / 20 / 25 / 30 ->" + "".join(cells))
    K, BAND = 10, 20


if __name__ == "__main__":
    main()
