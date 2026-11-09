"""Study: does the probability screener's expected value predict real results? Needs network (bhavcopies cache).

Every trading day in the cached NSE F&O bhavcopies, for every watchlist underlying, score the six setups with
options_lab.screen_chain() (NSE closes for prices, realised 20-day volatility for the model) exactly as the
dashboard does. Then settle each one at expiry using the underlying's actual close on the expiry day from that
day's bhavcopy, after the same charges and bid-ask allowance. Only setups whose expiry is inside the data count.

Checks: (1) realised result of everything, of positive-EV setups and of each day's top 10;
(2) realised result by predicted-EV decile; (3) calibration: predicted chance of profit vs actual win rate.

Run from the project root:  python research/option_screener_test.py      (writes docs/option_screener_test.json)
"""
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import intraday_dashboard as idd  # noqa: E402
import options_lab as ol  # noqa: E402

START, END = dt.date(2026, 8, 12), dt.date(2026, 11, 9)


def main():
    days, bh = [], {}
    d = START
    while d <= END:
        b = ol.bhavcopy(d)
        if b is not None and len(b):
            bh[d] = b
            days.append(d)
        d += dt.timedelta(days=1)
    print(f"{len(days)} bhavcopies {days[0]} .. {days[-1]}", flush=True)
    tick = {s: idd.resolve(s) for s in idd.WATCHLIST}
    raw = yf.download(list(tick.values()), start="2026-05-01", end=str(END + dt.timedelta(days=1)), interval="1d",
                      auto_adjust=False, group_by="ticker", threads=True, progress=False)
    hv = {}
    for s, t in tick.items():
        try:
            c = raw[t]["Close"].dropna()
        except KeyError:
            continue
        c.index = pd.to_datetime(c.index).date
        hv[s] = (np.log(c).diff().rolling(20).std() * math.sqrt(252))
    spot_on = {d: bh[d].groupby("symbol")["spot"].median() for d in days}

    rows = []
    for d in days:
        for s in idd.WATCHLIST:
            h = hv.get(s)
            if h is None:
                continue
            hh = h[[x for x in h.index if x <= d]]
            if hh.dropna().empty:
                continue
            ch = ol.chain(bh[d], s)
            for c in ol.screen_chain(ch, float(hh.dropna().iloc[-1])):
                E = dt.date.fromisoformat(c["expiry"])
                if E not in spot_on or s not in spot_on[E]:
                    continue                                     # expiry outside the data, or a holiday shift
                SE = float(spot_on[E][s])
                real = ol.evaluate_legs(c["legs"], SE, 0.0, 0.0, c["lot"], ol.SLIP_PCT["index" if c["index"] else "stock"])["ev"]
                rows.append(dict(date=str(d), symbol=s, index=c["index"], kind=c["kind"], ev=c["ev"], pop=c["pop"],
                                 epr=c["ev_per_risk"], risk=c["max_loss"], real=real, win=real > 0))
    R = pd.DataFrame(rows)
    print(f"{len(R)} setups settled at expiry", flush=True)

    def stat(df):
        return dict(n=int(len(df)), pred=round(float(df.ev.mean()), 0), real=round(float(df.real.mean()), 0),
                    win=round(float(df.win.mean() * 100), 1), pred_pop=round(float(df["pop"].mean()), 1),
                    real_pct_risk=round(float(df.real.sum() / df.risk.sum() * 100), 2)) if len(df) else None
    pos = R[R.ev > 0]
    top = (pos.sort_values("epr", ascending=False).groupby("date").apply(lambda g: g.drop_duplicates("symbol").head(10))
           .reset_index(drop=True))
    R["decile"] = pd.qcut(R.epr.rank(method="first"), 10, labels=False) + 1
    dec = [dict(decile=int(k), **stat(g)) for k, g in R.groupby("decile")]
    R["pop_bucket"] = pd.cut(R["pop"], [0, 30, 40, 50, 60, 70, 80, 100])
    cal = [dict(bucket=str(k), **stat(g)) for k, g in R.groupby("pop_bucket", observed=True)]
    corr = float(np.corrcoef(R.epr, R.real / R.risk.clip(lower=1))[0, 1])
    # Direction-neutral check: within each setup type and day, compare the top and bottom half by predicted EV.
    # A falling market helps every put the same, so only a ranking that adds value beats this comparison.
    R["half"] = R.groupby(["date", "kind"]).epr.rank(pct=True).gt(0.5).map({True: "top", False: "bottom"})
    within = {k: {h: stat(g[g.half == h]) for h in ("top", "bottom")} for k, g in R.groupby("kind")}
    ra = R.real / R.risk.clip(lower=1)
    within_corr = {k: round(float(np.corrcoef(g.epr, ra[g.index])[0, 1]), 3) for k, g in R.groupby("kind")}
    R.to_csv(ROOT / "cache" / "option_screener_rows.csv", index=False)
    out = dict(generated=dt.datetime.now().strftime("%d %b %Y %H:%M"), period=f"{days[0]} to {days[-1]}",
               sessions=len(days), all=stat(R), positive_ev=stat(pos), top10_daily=stat(top),
               by_setup={k: stat(g) for k, g in R.groupby("kind")}, index=stat(R[R["index"]]), stock=stat(R[~R["index"]]),
               deciles=dec, calibration=cal, corr_pred_real=round(corr, 3), within_setup=within,
               within_setup_corr=within_corr,
               nifty_move_pct=round(float(spot_on[days[-1]]["NIFTY"] / spot_on[days[0]]["NIFTY"] * 100 - 100), 2))
    (ROOT / "docs" / "option_screener_test.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"\n{'':<28}{'n':>6}{'Pred EV':>9}{'Real':>8}{'Win%':>7}{'Pred POP':>9}{'Real % risk':>12}")
    for name in ("all", "positive_ev", "top10_daily", "index", "stock"):
        x = out[name]
        print(f"{name:<28}{x['n']:>6}{x['pred']:>9.0f}{x['real']:>8.0f}{x['win']:>6.1f}%{x['pred_pop']:>8.1f}%{x['real_pct_risk']:>11.2f}%")
    for k, x in out["by_setup"].items():
        print(f"{'  ' + ol.SETUPS[k][:24]:<28}{x['n']:>6}{x['pred']:>9.0f}{x['real']:>8.0f}{x['win']:>6.1f}%{x['pred_pop']:>8.1f}%{x['real_pct_risk']:>11.2f}%")
    print("\nPredicted-EV deciles (1 = lowest):")
    for x in dec:
        print(f"  {x['decile']:>2}: n {x['n']:>5}  pred {x['pred']:>7.0f}  real {x['real']:>7.0f}  win {x['win']:>5.1f}%")
    print("Chance-of-profit calibration (predicted bucket -> actual win rate):")
    for x in cal:
        print(f"  {x['bucket']:<10} n {x['n']:>5}  predicted {x['pred_pop']:>5.1f}%  actual {x['win']:>5.1f}%")
    print(f"Correlation of predicted EV/risk with realised P&L/risk: {corr:.3f}")
    print(f"Nifty over the period: {out['nifty_move_pct']:+.2f}%")
    print("Within each setup type (same day): top half vs bottom half by predicted EV -> realised per lot, correlation")
    for k, v in within.items():
        print(f"  {ol.SETUPS[k][:26]:<28} top {v['top']['real']:>7.0f}  bottom {v['bottom']['real']:>7.0f}   corr {within_corr[k]:+.3f}")


if __name__ == "__main__":
    main()
