"""Offline smoke tests on synthetic data (no network needed).

Run:  python -m pytest -q tests      (or)      python tests/test_smoke.py
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intraday_dashboard as idd  # noqa: E402
import stock_analyzer as sa  # noqa: E402


def synthetic_intraday(days=30, seed=7, start_px=1000.0, volume=True):
    """5-minute candles, 09:15-15:25 IST, random walk with a little trend."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-09-01", periods=days)
    idx = [d + pd.Timedelta(minutes=9 * 60 + 15 + 5 * k) for d in dates for k in range(75)]
    rets = rng.normal(0.00005, 0.0015, len(idx))
    close = start_px * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_px], close[:-1]]) * (1 + rng.normal(0, 0.0003, len(idx)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0008, len(idx))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0008, len(idx))))
    vol = rng.integers(20_000, 200_000, len(idx)).astype(float) if volume else np.zeros(len(idx))
    return pd.DataFrame(dict(Open=open_, High=high, Low=low, Close=close, Volume=vol), index=pd.DatetimeIndex(idx))


def synthetic_daily(n=1600, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    close = 500 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    open_ = np.concatenate([[500], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    return pd.DataFrame(dict(Open=open_, High=high, Low=low, Close=close,
                             Volume=rng.integers(1e5, 1e6, n).astype(float)), index=idx)


# ---------------------------------------------------------------- intraday_dashboard
def test_compute_columns_and_vwap_reset():
    d = idd.compute(synthetic_intraday())
    for col in ("VWAP", "EMA9", "EMA21", "RSI", "ATR", "ST_DIR", "ST_LINE", "OR_H", "OR_L", "PDH", "TC", "BC"):
        assert col in d.columns, col
    first = d.groupby("date").head(1)                      # VWAP resets each session to the typical price
    tp = (first["High"] + first["Low"] + first["Close"]) / 3
    assert np.allclose(first["VWAP"], tp)
    assert d.loc[d["tod"] < idd.OR_END, "OR_H"].isna().all()   # opening range only known after 9:30
    assert (d["TC"].dropna() >= d["BC"].dropna()).all()


def test_index_without_volume_uses_price_vwap():
    d = idd.compute(synthetic_intraday(volume=False))
    assert d["VWAP"].notna().all()


def test_simulate_all_strategies():
    d = idd.compute(synthetic_intraday(days=40))
    for k in idd.STRATS:
        trades, open_t, sig = idd.simulate(d, k, 100_000, 0.02)
        assert isinstance(trades, list)
        for t in trades:
            assert t["status"] == "closed"
            assert t["reason"] in {"Stop-loss", "Target", "Exit signal", "Square-off 3:15", "End of day"}
            assert -100 <= t["bias"] <= 100
            assert t["charges"] > 0
            assert abs(t["net"] - (t["gross"] - t["charges"])) < 0.05
            assert t["entry_ts"] <= t["exit_ts"]                 # stop/target can hit inside the entry candle
        st = idd.trade_stats(trades, 100_000)
        assert st["trades"] == len(trades)
        if trades:
            assert abs(st["win_pct"] + st["loss_pct"] - 100) < 0.2


def test_no_trades_after_square_off():
    d = idd.compute(synthetic_intraday(days=40, seed=11))
    for k in idd.STRATS:
        trades, _, _ = idd.simulate(d, k, 100_000, 0.0)
        for t in trades:
            hh, mm = map(int, t["entry_time"][-5:].split(":"))
            assert hh * 60 + mm <= idd.LAST_ENTRY + 15, t


def test_charges_known_case():
    c = idd.charges(100_000, 101_000, 0.0)
    assert abs(c["brokerage"] - 40) < 1e-9                  # Rs 20 cap on both legs
    assert abs(c["stt"] - 25.25) < 1e-9                     # 0.025% of sell value
    assert abs(c["stamp"] - 3.0) < 1e-9                     # 0.003% of buy value
    assert 70 < c["total"] < 85


def test_bias_series_matches_bias():
    d = idd.compute(synthetic_intraday(days=10))
    row = d.iloc[-1]
    assert int(idd.bias_series(d).iloc[-1]) == idd.bias(row)["score"]


def test_next_session_skips_weekend_and_holiday():
    assert idd.next_session(dt.date(2026, 11, 6)) == dt.date(2026, 11, 9)   # Fri -> Mon
    assert idd.next_session(dt.date(2026, 11, 9)) == dt.date(2026, 11, 11)  # Tue 10 Nov is a holiday


def test_pages_build():
    assert idd.PAGE.startswith("<!doctype html>")
    hosted = idd.hosted_page(dict(generated="x", data_until="y", symbols=["A"], files={"A": "data/A.json"},
                                  scan="data/_scan.json", plan="data/_plan.json", inst="data/_inst.json"))
    assert "<!doctype" not in hosted.lower() and "SNAP_INDEX" in hosted and "<title>" in hosted


def test_standalone_page_and_session_window():
    page = idd.standalone_page(dict(generated="x", gen_ts=0, data_until="y", symbols=["A"], files={"A": "data/A.json"},
                                    scan="data/_scan.json", plan="data/_plan.json", inst="data/_inst.json"))
    assert page.startswith("<!doctype html>") and "SNAP_INDEX" in page and page.rstrip().endswith("</html>")
    assert idd.session_window(dt.datetime(2026, 11, 9, 9, 15))          # Monday open
    assert idd.session_window(dt.datetime(2026, 11, 9, 15, 40))         # last rebuild after close
    assert not idd.session_window(dt.datetime(2026, 11, 9, 15, 41))
    assert not idd.session_window(dt.datetime(2026, 11, 9, 9, 14))
    assert not idd.session_window(dt.datetime(2026, 11, 10, 11, 0))     # holiday
    assert not idd.session_window(dt.datetime(2026, 11, 8, 11, 0))      # Sunday


def synthetic_closes(days=330, end="2026-11-09", n=14, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=days)
    drift = np.linspace(-0.001, 0.002, n)                       # stock k trends harder the higher k is
    scale = np.linspace(0.005, 0.03, n)[::-1]                   # stock 0 is the most volatile
    rets = rng.normal(drift, scale, (days, n))
    C = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=[f"S{k}" for k in range(n)])
    C["NIFTY"] = 20000 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, days)))
    return C


def test_compute_picks_uses_last_month_end_only():
    C = synthetic_closes()
    P = idd.compute_picks(C, n=5)
    assert P["formed"] == "2026-10-30" and P["formation"] == "2026-10" and P["asof"] == "2026-11-09"
    assert len(P["momentum"]) == 5 and len(P["lowvol"]) == 5
    vols = [r["vol"] for r in P["lowvol"]]
    assert vols == sorted(vols)
    moms = [r["mom"] for r in P["momentum"]]
    assert moms == sorted(moms, reverse=True)
    C2 = C.copy()
    C2.loc["2026-11-02":, "S0"] *= 50                            # a November jump must not change the ranking
    assert [r["symbol"] for r in idd.compute_picks(C2, n=5)["momentum"]] == [r["symbol"] for r in P["momentum"]]


def test_momentum_banding_keeps_holdings_within_band():
    C = synthetic_closes(n=14)
    base = idd.compute_picks(C, n=5, band=8)
    order = [r["symbol"] for r in base["momentum"]]
    full = idd.compute_picks(C, n=14, band=14)["momentum"]               # every stock, in rank order
    rank = {r["symbol"]: r["rank"] for r in full}
    inside = next(s for s, k in rank.items() if 5 < k <= 8)              # outside the top 5, inside the band
    outside = next(s for s, k in rank.items() if k > 8)
    P = idd.compute_picks(C, n=5, held=[inside, outside], band=8)
    syms = [r["symbol"] for r in P["momentum"]]
    assert len(syms) == 5 and inside in syms and outside not in syms
    assert next(r for r in P["momentum"] if r["symbol"] == inside)["kept"]
    assert syms[:4] == order[:4]                                         # the rest filled from the top


def test_journal_rebalance_rules():
    P = idd.compute_picks(synthetic_closes(), n=5)
    J = idd.update_journal({}, P, "2026-11-09 09:20")
    assert len(J["momentum"]) == 1 and len(J["lowvol"]) == 1
    assert idd.update_journal(J, P, "2026-11-09 09:25") == J            # same month: nothing changes
    P12 = dict(P, formation="2026-11", formed="2026-11-30")
    J2 = idd.update_journal(J, P12, "2026-12-01 09:20")
    assert len(J2["momentum"]) == 2 and J2["momentum"][0]["closed"] == "2026-12-01 09:20"
    assert len(J2["lowvol"]) == 1                                        # quarterly: November is not a quarter end
    J3 = idd.update_journal(J2, dict(P, formation="2026-12", formed="2026-12-31"), "2027-01-01 09:20")
    assert len(J3["lowvol"]) == 2
    S = idd.journal_summary(J2, P12)["momentum"]
    first = S["periods"][0]
    assert not first["open"] and abs(first["net"] - (first["gross"] - idd.PICK_COST_PCT)) < 0.02


def test_options_pricing_parity_iv_and_charges():
    import options_lab as ol
    S, K, T, v, r = 25000.0, 25100.0, 7 / 365, 0.16, ol.RISK_FREE
    c, p = ol.bs_price(S, K, T, v, "CE"), ol.bs_price(S, K, T, v, "PE")
    assert abs((c - p) - (S - K * np.exp(-r * T))) < 1e-6                       # put-call parity
    assert abs(ol.implied_vol(c, S, K, T, "CE") - v) < 1e-4                     # IV round trip
    g = ol.greeks(S, K, T, v, "CE")
    assert 0 < g["delta"] < 1 and g["theta"] < 0 and g["vega"] > 0
    assert ol.greeks(S, K, T, v, "PE")["delta"] < 0
    assert ol.bs_price(S, K, 0, v, "PE") == 100.0                               # expiry = intrinsic value
    assert ol.implied_vol(10.0, S, 24000.0, T, "CE") is None                    # below intrinsic
    ch = ol.option_charges(10_000, 12_000)
    assert abs(ch["brokerage"] - 40) < 1e-9 and abs(ch["stt"] - 12.0) < 1e-9 and abs(ch["stamp"] - 0.3) < 1e-9
    assert 60 < ch["total"] < 70


def test_option_chain_from_bhavcopy_rows():
    import options_lab as ol
    day = dt.date(2026, 11, 9)
    rows = []
    for e in (day, dt.date(2026, 11, 17)):
        for k in (24900.0, 25000.0, 25100.0):
            for cp in ("CE", "PE"):
                T = ol.years_to_expiry(dt.datetime(2026, 11, 9, 15, 30), e)
                px = ol.bs_price(25010.0, k, T, 0.15, cp) if T > 0 else 1.0
                rows.append(dict(symbol="NIFTY", expiry=e, strike=k, cp=cp, close=round(px, 2), settle=px, spot=25010.0,
                                 oi=1000.0, volume=10.0, lot=65.0, date=day))
    ch = ol.chain(pd.DataFrame(rows), "NIFTY")
    assert ch["lot"] == 65 and ch["index"] and [e["expiry"] for e in ch["expiries"]] == ["2026-11-17"]   # expired one dropped
    assert abs(ol.atm_iv(ch) - 15.0) < 0.3


def test_screener_maths():
    import options_lab as ol
    S, T, v, lot = 25000.0, 7 / 365, 0.16, 65
    c = ol.bs_price(S, 25000.0, T, v, "CE", r=0.0)
    # fair price + same volatility in the model -> expected value is just minus the costs
    r = ol.evaluate_legs([dict(cp="CE", K=25000.0, side=1, price=c)], S, T, v, lot, 0.0)
    costs = ol.option_charges(c * lot, c * lot)["total"]
    assert -1.6 * costs < r["ev"] < -0.3 * costs
    assert 0.2 < r["pop"] < 0.5 and r["breakevens"] and abs(r["breakevens"][0] - (25000 + c)) < 1
    # debit spread: loss is capped at the debit (+ costs), gain at width - debit
    legs = [dict(cp="CE", K=25000.0, side=1, price=200.0), dict(cp="CE", K=25300.0, side=-1, price=80.0)]
    s = ol.evaluate_legs(legs, S, T, v, lot, 0.5)
    assert -120 * lot - 200 < s["worst"] < -120 * lot and s["best"] < 180 * lot
    # settling at a known price (T=0) gives one exact outcome
    up = ol.evaluate_legs(legs, 25500.0, 0.0, 0.0, lot, 0.0)
    assert up["pop"] == 1.0 and 180 * lot - 250 < up["ev"] < 180 * lot          # width 300 - debit 120, minus charges
    assert abs(up["best"] - up["worst"]) < 1e-9


def test_option_journal_logs_once_and_settles(tmp_path=None, monkeypatch=None):
    import tempfile
    import options_lab as ol
    path = Path(tempfile.mkdtemp()) / "oj.json"
    old_path, old_bhav = idd.OPT_JOURNAL_PATH, ol.bhavcopy
    idd.OPT_JOURNAL_PATH = str(path)
    try:
        legs = [dict(cp="CE", K=100.0, side=1, price=2.0)]
        pick = dict(symbol="ABC", index=False, kind="long_call", setup="Buy call", expiry="2026-11-17", legs=legs, lot=100,
                    spot=100.0, ev=50.0, pop=40.0, max_loss=250.0, kelly=0.05, iv=20.0, hv=30.0)
        O = dict(asof="2026-11-09", screener=dict(top=[pick]), chains={"ABC": dict(spot_now=101.0)})
        J = idd.update_option_journal(O)
        J = idd.update_option_journal(O)                                       # same day again: no duplicate
        assert len(J["entries"]) == 1 and J["entries"][0]["real"] is None
        ol.bhavcopy = lambda d, cache=True: pd.DataFrame(dict(symbol=["ABC"], spot=[110.0]))
        J = idd.update_option_journal(dict(O, asof="2026-11-17", screener=dict(top=[])))
        e = J["entries"][0]
        assert e["settle_spot"] == 110.0 and 700 < e["real"] < 800              # (10 - 2) x 100 minus costs
        S = idd.option_journal_summary(J, O)
        assert S["settled"] == 1 and S["wins"] == 1 and S["days"] == 2
    finally:
        idd.OPT_JOURNAL_PATH, ol.bhavcopy = old_path, old_bhav


def test_kelly_fraction():
    import options_lab as ol
    # classic bet: win +1 with p=0.6, lose -1 with p=0.4 -> Kelly = 2p-1 = 0.2
    pnl = np.array([1.0] * 60 + [-1.0] * 40)
    assert abs(ol.kelly_fraction(pnl) - 0.2) < 0.002
    assert ol.kelly_fraction(np.array([1.0] * 40 + [-1.0] * 60)) == 0.0        # negative edge -> no bet
    assert ol.kelly_fraction(np.array([5.0, -1.0, -1.0, -1.0])) > 0              # long shot with positive EV


def test_resolve_and_tv_symbols():
    assert idd.resolve("nifty") == "^NSEI" and idd.resolve("SBIN") == "SBIN.NS"
    assert idd.tv_symbol("M&M.NS") == "NSE:M_M" and idd.tv_symbol("^NSEBANK") == "NSE:BANKNIFTY"


# ---------------------------------------------------------------- stock_analyzer
def test_daily_engine_and_backtest():
    d = sa.add_indicators(synthetic_daily())
    comp, score = sa.technical_components(d)
    assert score.dropna().between(-100, 100).all()
    cfg = sa.Config(years=3)
    bt = sa.backtest(d, score, cfg, d.index[-1] - pd.DateOffset(years=3))
    st = bt["stats"]
    assert st["trades"] == st["wins"] + st["losses"]
    acc = sa.signal_accuracy(d, score, cfg, bt["start_i"])
    assert [a["horizon"] for a in acc] == [5, 10, 20]


def test_fundamental_and_news_scoring():
    info = dict(trailingPE=18, forwardPE=15, priceToBook=2.5, returnOnEquity=0.22, debtToEquity=40,
                profitMargins=0.15, revenueGrowth=0.12, earningsGrowth=0.25, currentRatio=1.8, sector="Industrials")
    score, rows, meta = sa.fundamental_score(info)
    assert score is not None and score > 30 and len(rows) >= 8
    assert sa.headline_sentiment("Shares surge to record high after strong Q2 profit") > 0
    assert sa.headline_sentiment("Stock plunges after SEBI notice, profit falls") < 0


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} tests passed")
