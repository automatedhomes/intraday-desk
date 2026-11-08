"""Study: intraday variants (minimum stop %, 1 trade/day, 15-min candles, larger size). Needs network.

Run from the project root:  python research/intraday_variants.py
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import intraday_dashboard as m
syms = "RELIANCE HDFCBANK ICICIBANK SBIN AXISBANK INFY TCS LT ITC BHARTIARTL MARUTI SUNPHARMA TITAN TATASTEEL HINDALCO NTPC ONGC BAJFINANCE ADANIENT TRENT".split()
data = {iv: {s: m.compute(m.fetch(m.resolve(s), iv)) for s in syms} for iv in ("5m","15m")}
orig_signals = m.signals
orig_strats = {k: dict(v) for k, v in m.STRATS.items()}
def run(label, iv="5m", min_risk_pct=0.0, rr=None, max_day=None, cap=100000):
    def sig(d, key):
        L,S,RL,RS,XL,XS = orig_signals(d, key)
        floor = d["Close"].to_numpy() * min_risk_pct / 100
        return L,S,np.maximum(RL,floor),np.maximum(RS,floor),XL,XS
    m.signals = sig
    for k in m.STRATS:
        m.STRATS[k] = dict(orig_strats[k]); 
        if rr: m.STRATS[k]["rr"] = rr
        if max_day: m.STRATS[k]["max_day"] = max_day
    rows=[]
    for s,d in data[iv].items():
        for k in m.STRATS:
            tr,_,_ = m.simulate(d,k,cap,0.02); st=m.trade_stats(tr,cap)
            rows.append((k,s,st["trades"],st["wins"],st["net"],st["gross"]))
    df=pd.DataFrame(rows,columns=["k","s","n","w","net","gross"])
    g=df.groupby("k").agg(n=("n","sum"),w=("w","sum"),net=("net","sum"),gross=("gross","sum"))
    pos=df[df.n>0].groupby("k").apply(lambda x:(x.net>0).mean()*100, include_groups=False)
    out=" | ".join(f"{k}: n{int(r.n)} win{r.w/r.n*100:4.1f}% g/tr{r.gross/r.n:+5.0f} net/tr{r.net/r.n:+5.0f} +stocks{pos.get(k,0):3.0f}%" for k,r in g.iterrows())
    print(f"{label:<34} {out}")
    m.signals = orig_signals
run("5m baseline")
run("5m risk>=0.3%", min_risk_pct=0.3)
run("5m risk>=0.5%", min_risk_pct=0.5)
run("5m risk>=0.5% maxday1", min_risk_pct=0.5, max_day=1)
run("15m baseline", iv="15m")
run("15m risk>=0.5%", iv="15m", min_risk_pct=0.5)
run("15m risk>=0.5% rr1.5", iv="15m", min_risk_pct=0.5, rr=1.5)
run("5m risk>=0.5% cap 5L", min_risk_pct=0.5, cap=500000)
