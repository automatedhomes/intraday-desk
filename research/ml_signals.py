"""Study: machine-learning stock selection, walk-forward, after costs and tax. Needs network and scikit-learn.

Rules fixed before running (no tuning):
  * 43 watchlist stocks, daily data from 2014. 16 features known at each close: log returns over 1/5/20/60/120
    days, 20/60-day volatility, RSI 14, distance from 20- and 200-day averages, 20-day average range, volume vs
    its 20-day average, cross-sectional ranks of the 5/20/60-day returns, Nifty 5- and 20-day returns.
  * Target: the stock's next 5-day return MINUS the average of all stocks (market direction removed).
  * Models: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=4) and Ridge(alpha=1)
    on standardised features. Walk-forward: retrained every January on all earlier data with a 10-day gap,
    predicting that calendar year only (test years 2018-2026).
  * Portfolio: every 5th trading day hold the 10 stocks with the highest prediction (features from the previous
    close), equal weight; 0.15% cost per side and 20%/12.5% tax via research/factor_study.simulate.
  * Benchmarks: hold the same stocks equally (no rebalancing), and the Nifty.
Checks: information coefficient (rank correlation of prediction with the next 5-day excess return) and the
return spread between the top and bottom deciles, per year.
CAVEAT: today's large caps (survivorship bias) flatter every long-only result; the excess-return target does not.

Run from the project root:  python research/ml_signals.py      (writes docs/ml_signals.json)
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import intraday_dashboard as idd  # noqa: E402
from factor_study import simulate  # noqa: E402
from swing_momentum import stats  # noqa: E402

START, FIRST_TEST, HORIZON, K, GAP = "2014-01-01", 2018, 5, 10, 10
FEATS = ["r1", "r5", "r20", "r60", "r120", "v20", "v60", "rsi", "d20", "d200", "rng20", "volr",
         "x5", "x20", "x60", "m5", "m20"]


def load():
    syms = [s for s in idd.WATCHLIST if s not in ("NIFTY", "BANKNIFTY")]
    raw = yf.download([idd.resolve(s) for s in syms] + ["^NSEI"], start=START, interval="1d", auto_adjust=True,
                      group_by="ticker", threads=True, progress=False)
    px = {}
    for s in syms + ["NIFTY"]:
        t = "^NSEI" if s == "NIFTY" else idd.resolve(s)
        try:
            df = raw[t][["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        except KeyError:
            continue
        idx = pd.to_datetime(df.index)
        df.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        if len(df) > 400:
            px[s] = df[df["Close"] > 0]
    return px


def features(px):
    nifty = px["NIFTY"]["Close"]
    m5, m20 = np.log(nifty).diff(5), np.log(nifty).diff(20)
    frames = []
    for s, df in px.items():
        if s == "NIFTY":
            continue
        c, lc = df["Close"], np.log(df["Close"])
        r = lc.diff()
        delta = c.diff()
        g = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        lo = (-delta).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        f = pd.DataFrame(dict(
            r1=r, r5=lc.diff(5), r20=lc.diff(20), r60=lc.diff(60), r120=lc.diff(120),
            v20=r.rolling(20).std(), v60=r.rolling(60).std(), rsi=100 * g / (g + lo),
            d20=c / c.rolling(20).mean() - 1, d200=c / c.rolling(200).mean() - 1,
            rng20=((df["High"] - df["Low"]) / c).rolling(20).mean(),
            volr=df["Volume"] / df["Volume"].rolling(20).mean().replace(0, np.nan),
            m5=m5.reindex(df.index), m20=m20.reindex(df.index),
            fwd=lc.shift(-HORIZON) - lc), index=df.index)
        f["symbol"] = s
        frames.append(f)
    F = pd.concat(frames).reset_index().rename(columns={"index": "date", "Date": "date"})
    for a, b in (("x5", "r5"), ("x20", "r20"), ("x60", "r60")):
        F[a] = F.groupby("date")[b].rank(pct=True)
    F["y"] = F["fwd"] - F.groupby("date")["fwd"].transform("mean")          # excess over the average stock
    return F.replace([np.inf, -np.inf], np.nan)


def walk_forward(F, make_model):
    preds = []
    for yr in range(FIRST_TEST, pd.Timestamp.today().year + 1):
        test = F[F.date.dt.year == yr].dropna(subset=FEATS)
        cutoff = pd.Timestamp(yr, 1, 1) - pd.Timedelta(days=GAP + HORIZON * 2)
        train = F[(F.date < cutoff)].dropna(subset=FEATS + ["y"])
        if test.empty or len(train) < 5000:
            continue
        m = make_model().fit(train[FEATS], train["y"])
        preds.append(test.assign(pred=m.predict(test[FEATS]))[["date", "symbol", "pred", "y"]])
    return pd.concat(preds)


def evaluate(P, name):
    P = P.dropna(subset=["pred"])
    ic = P.dropna(subset=["y"]).groupby("date").apply(lambda g: g["pred"].rank().corr(g["y"].rank()))
    P = P.dropna(subset=["y"]).copy()
    P["dec"] = P.groupby("date")["pred"].rank(pct=True)
    spread = P[P.dec > 0.9].groupby("date").y.mean() - P[P.dec <= 0.1].groupby("date").y.mean()
    by_year = {int(y): dict(ic=round(float(v.mean()), 4)) for y, v in ic.groupby(ic.index.year)}
    for y, v in spread.groupby(spread.index.year):
        by_year[int(y)]["spread_5d_pct"] = round(float(v.mean() * 100), 3)
    return dict(model=name, ic_mean=round(float(ic.mean()), 4), ic_tstat=round(float(ic.mean() / ic.std() * np.sqrt(len(ic) / HORIZON)), 2),
                spread_5d_pct=round(float(spread.mean() * 100), 3), by_year=by_year)


def main():
    t0 = dt.datetime.now()
    px = load()
    F = features(px)
    print(f"{F.symbol.nunique()} stocks, {len(F):,} stock-days; walk-forward test years {FIRST_TEST}+", flush=True)
    models = {
        "Gradient boosting": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=4, random_state=0),
        "Linear (ridge)": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }
    C = pd.DataFrame({s: df["Close"] for s, df in px.items() if s != "NIFTY"}).ffill(limit=5)
    C["NIFTY"] = px["NIFTY"]["Close"].reindex(C.index).ffill()
    S = C.drop(columns="NIFTY")
    test_days = C.index[C.index >= pd.Timestamp(FIRST_TEST, 1, 1)]
    rebal = set(test_days[::HORIZON])
    first = test_days[0]
    yrs = (test_days[-1] - first).days / 365.25
    out = dict(generated=t0.strftime("%d %b %Y %H:%M"), period=f"{first.date()} to {test_days[-1].date()}", models={})

    def row(name, rb, target):
        # simulate() starts in 2016 (its own START); the portfolio holds cash until the first rebalance here, so
        # measure every column over the test period only. Turnover is scaled to the test years for the same reason.
        g, _, _ = simulate(C, rb, target, cost=0.0, tax=False)
        v, _, to = simulate(C, rb, target, tax=False)
        _, final, _ = simulate(C, rb, target)
        g, v = g[g.index >= first], v[v.index >= first]
        st = stats(v.pct_change())
        sim_yrs = (test_days[-1] - pd.Timestamp("2016-01-01")).days / 365.25
        return dict(name=name, before_costs=round(stats(g.pct_change())["cagr"], 1), after_costs=round(st["cagr"], 1),
                    after_tax=round((final ** (1 / yrs) - 1) * 100, 1), max_dd=round(st["maxdd"], 0),
                    turnover=round(to * sim_yrs / yrs, 0))
    ok0 = S.loc[first].notna()
    out["benchmarks"] = [row("Nifty 50 index, hold (no dividends)", {first}, lambda t: pd.Series({"NIFTY": 1.0})),
                         row("Hold the same stocks (equal, no rebal.)", {first}, lambda t: pd.Series(1 / ok0.sum(), index=ok0[ok0].index))]
    for name, mk in models.items():
        P = walk_forward(F, mk)
        ev = evaluate(P, name)
        piv = P.pivot_table(index="date", columns="symbol", values="pred").reindex(C.index).shift(1)   # previous close

        def target(t, piv=piv):
            s = piv.loc[t].dropna() if t in piv.index else pd.Series(dtype=float)
            return pd.Series(1 / K, index=s.nlargest(K).index) if len(s) >= K else pd.Series(dtype=float)
        ev["portfolio"] = row(f"{name}: top {K}, weekly", rebal, target)
        out["models"][name] = ev
        print(f"  {name}: IC {ev['ic_mean']:+.4f} (t {ev['ic_tstat']:.1f}), top-bottom decile {ev['spread_5d_pct']:+.3f}% per 5 days", flush=True)
    (ROOT / "docs" / "ml_signals.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"\n{'':<44}{'Before costs':>13}{'After costs':>12}{'After tax':>10}{'Max DD':>8}{'Turnover':>10}")
    for r in out["benchmarks"] + [m["portfolio"] for m in out["models"].values()]:
        print(f"{r['name']:<44}{r['before_costs']:>12.1f}%{r['after_costs']:>11.1f}%{r['after_tax']:>9.1f}%{r['max_dd']:>7.0f}%{r['turnover']:>9.0f}%")
    for name, m in out["models"].items():
        print(f"\n{name} by year (IC, top-minus-bottom decile % per 5 days):")
        print("  " + "  ".join(f"{y}: {v['ic']:+.3f}/{v.get('spread_5d_pct', 0):+.2f}" for y, v in m["by_year"].items()))
    print(f"\nTook {(dt.datetime.now() - t0).seconds}s. Past results on today's large caps; not advice.")


if __name__ == "__main__":
    main()
