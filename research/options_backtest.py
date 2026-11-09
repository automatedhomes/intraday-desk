"""Study: buying options on the dashboard's intraday signals, after F&O charges and bid-ask cost. Needs network.

Rules fixed before running (no tuning):
  * Signals and exits: exactly the intraday backtest in intraday_dashboard.simulate() (5-min candles, ~60 days,
    entry next candle's open, stop/target/exit signal/3:15 square-off).
  * Instrument: long signal -> buy the at-the-money call, short -> buy the at-the-money put, nearest expiry on
    or after the trade day, strike nearest to the entry price, 1 lot (NSE lot size from the bhavcopy).
  * Premium: Black-Scholes at the entry and exit prices and times, using that contract's implied volatility from
    the previous trading day's NSE close (bhavcopy). Volatility is held constant through the trade.
  * Costs: Zerodha-style option charges (options_lab.option_charges) + bid-ask allowance per side:
    0.5% of premium for index options, 2% for stock options.
Model limits: real fills, intraday IV changes and very wide stock-option spreads are not captured.

Run from the project root:  python research/options_backtest.py      (writes docs/options_backtest.json)
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import intraday_dashboard as idd  # noqa: E402
import options_lab as ol  # noqa: E402

SLIP = {"index": 0.5, "stock": 2.0}


def main():
    t0 = dt.datetime.now()
    data, dates = {}, set()
    for s in idd.WATCHLIST:
        try:
            d = idd.compute(idd.fetch(idd.resolve(s), "5m"))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {s}: {e}")
            continue
        data[s] = d
        dates |= set(d["date"])
    dates = sorted(dates)
    print(f"{len(data)} symbols, {len(dates)} sessions {dates[0]} .. {dates[-1]}; loading NSE bhavcopies ...", flush=True)
    bhav = {}
    day = dates[0] - dt.timedelta(days=1)
    while len(bhav) < 1 and day > dates[0] - dt.timedelta(days=10):            # the session before the first one
        b = ol.bhavcopy(day)
        if b is not None:
            bhav[day] = b
        day -= dt.timedelta(days=1)
    for d in dates:
        b = ol.bhavcopy(d)
        if b is not None:
            bhav[d] = b
    bdays = sorted(bhav)
    by_sym = {d: dict(tuple(bhav[d].groupby("symbol"))) for d in bdays}
    print(f"  {len(bhav)} bhavcopies loaded", flush=True)

    rows, skipped = [], {}
    for s, d in data.items():
        group = "index" if s in ol.INDEX_UNDERLYINGS else "stock"
        for k in idd.STRATS:
            trades, _, _ = idd.simulate(d, k, 100_000, 0.02)
            for t in trades:
                D = dt.date.fromisoformat(t["date"])
                prev = [x for x in bdays if x < D]
                if not prev or s not in by_sym[prev[-1]]:
                    skipped["no previous bhavcopy"] = skipped.get("no previous bhavcopy", 0) + 1
                    continue
                P = by_sym[prev[-1]][s]
                cp = "CE" if t["side"] > 0 else "PE"
                exps = sorted(e for e in P["expiry"].unique() if e >= D)
                if not exps:
                    skipped["no expiry"] = skipped.get("no expiry", 0) + 1
                    continue
                E = exps[0]
                c = P[(P["expiry"] == E) & (P["cp"] == cp)]
                if c.empty:
                    skipped["no contracts"] = skipped.get("no contracts", 0) + 1
                    continue
                K = float(c.iloc[(c["strike"] - t["entry"]).abs().argmin()]["strike"])
                row = c[c["strike"] == K].iloc[0]
                T_prev = ol.years_to_expiry(dt.datetime.combine(prev[-1], dt.time(15, 30)), E)
                iv = ol.implied_vol(float(row["close"]), float(row["spot"]), K, T_prev, cp)
                if iv is None:
                    skipped["no implied vol (stale/illiquid close)"] = skipped.get("no implied vol (stale/illiquid close)", 0) + 1
                    continue
                t_in = dt.datetime.utcfromtimestamp(t["entry_ts"])
                t_out = dt.datetime.utcfromtimestamp(t["exit_ts"])
                p_in = ol.bs_price(t["entry"], K, ol.years_to_expiry(t_in, E), iv, cp)
                p_out = ol.bs_price(t["exit"], K, ol.years_to_expiry(t_out, E), iv, cp)
                if p_in < 0.5:
                    skipped["premium under Rs 0.50"] = skipped.get("premium under Rs 0.50", 0) + 1
                    continue
                lot = int(row["lot"])
                gross = (p_out - p_in) * lot
                ch = ol.option_charges(p_in * lot, p_out * lot)["total"]
                slip = SLIP[group] / 100 * (p_in + p_out) * lot
                rows.append(dict(symbol=s, group=group, strategy=k, side=t["side"], date=t["date"], expiry=str(E),
                                 dte=(E - D).days, iv=iv * 100, premium=p_in * lot, gross=gross, charges=ch, slip=slip,
                                 net=gross - ch - slip, cash_net=t["net"], cash_gross=t["gross"]))
    R = pd.DataFrame(rows)

    def summary(df):
        if df.empty:
            return None
        return dict(trades=int(len(df)), win_gross=round(float((df.gross > 0).mean() * 100), 1),
                    win_net=round(float((df.net > 0).mean() * 100), 1), premium=round(float(df.premium.mean()), 0),
                    gross=round(float(df.gross.mean()), 0), charges=round(float(df.charges.mean()), 0),
                    slip=round(float(df.slip.mean()), 0), net=round(float(df.net.mean()), 0),
                    net_pct_premium=round(float(df.net.sum() / df.premium.sum() * 100), 1),
                    total_net=round(float(df.net.sum()), 0), cash_gross=round(float(df.cash_gross.mean()), 0),
                    cash_net=round(float(df.cash_net.mean()), 0))
    out = dict(generated=dt.datetime.now().strftime("%d %b %Y %H:%M"), sessions=len(dates),
               period=f"{dates[0]} to {dates[-1]}", symbols=len(data), slippage_pct=SLIP, skipped=skipped,
               setups={k: dict(name=idd.STRATS[k]["name"], **{g: summary(R[(R.strategy == k) & ((R.group == g) if g != "all" else True)])
                                                              for g in ("index", "stock", "all")}) for k in idd.STRATS},
               all=summary(R), expiry_day=summary(R[R.dte == 0]), not_expiry_day=summary(R[R.dte > 0]))
    (ROOT / "docs" / "options_backtest.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"\n{len(R)} option trades priced, skipped {sum(skipped.values())} {skipped}")
    print(f"{'':<36}{'Trades':>7}{'Win net':>8}{'Premium':>9}{'Gross':>8}{'Charges':>8}{'Bid-ask':>8}{'Net':>8}{'Net %':>7}")
    for k, v in out["setups"].items():
        for g in ("index", "stock"):
            x = v[g]
            if x:
                print(f"{v['name'][:24] + ' / ' + g:<36}{x['trades']:>7}{x['win_net']:>7.1f}%{x['premium']:>9.0f}{x['gross']:>8.0f}"
                      f"{x['charges']:>8.0f}{x['slip']:>8.0f}{x['net']:>8.0f}{x['net_pct_premium']:>6.1f}%")
    a = out["all"]
    print(f"{'ALL':<36}{a['trades']:>7}{a['win_net']:>7.1f}%{a['premium']:>9.0f}{a['gross']:>8.0f}{a['charges']:>8.0f}"
          f"{a['slip']:>8.0f}{a['net']:>8.0f}{a['net_pct_premium']:>6.1f}%")
    print(f"Expiry-day trades: {out['expiry_day']}\nSame signals in shares (Rs 1 lakh): gross {a['cash_gross']}, net {a['cash_net']} per trade")
    print(f"Rs per lot per trade; Net % = total net / total premium paid. Took {(dt.datetime.now() - t0).seconds}s.")


if __name__ == "__main__":
    main()
