"""Study: intraday setups before vs after costs on 20 stocks (5-min, 60 days). Needs network.

Run from the project root:  python research/intraday_costs.py
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import intraday_dashboard as m
syms = "RELIANCE HDFCBANK ICICIBANK SBIN AXISBANK INFY TCS LT ITC BHARTIARTL MARUTI SUNPHARMA TITAN TATASTEEL HINDALCO NTPC ONGC BAJFINANCE ADANIENT TRENT".split()
D = {}
for s in syms:
    D[s] = m.compute(m.fetch(m.resolve(s), "5m"))
def evaluate(label, slip=0.02):
    agg = []
    for s, d in D.items():
        for k in m.STRATS:
            tr, _, _ = m.simulate(d, k, 100000, slip)
            st = m.trade_stats(tr, 100000)
            agg.append((k, st["trades"], st["wins"], st["net"], st["gross"], st["charges"]))
    df = pd.DataFrame(agg, columns=["k","n","w","net","gross","ch"]).groupby("k").sum()
    df["win%"] = (df.w/df.n*100).round(1); df["net/tr"] = (df.net/df.n).round(0); df["gross/tr"]=(df.gross/df.n).round(0); df["cost/tr"]=(df.ch/df.n).round(0)
    print(f"--- {label}"); print(df[["n","win%","gross/tr","cost/tr","net/tr","net"]].to_string())
evaluate("baseline (20 stocks, 60 days, 5m)")
evaluate("zero slippage", slip=0.0)
