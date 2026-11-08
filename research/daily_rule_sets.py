"""Study: classic daily rule sets (RSI/Bollinger dips, Supertrend, 50/200-DMA) vs buy-and-hold on 20 stocks. Needs network.

Run from the project root:  python research/daily_rule_sets.py
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import stock_analyzer as sa
syms = "RELIANCE TCS HDFCBANK INFY ITC TATASTEEL SBIN BAJFINANCE LT HINDUNILVR MARUTI SUNPHARMA TITAN ASIANPAINT AXISBANK ONGC NTPC TATAPOWER DLF ADANIPORTS".split()
data={}
for s in syms:
    raw,_=sa.fetch_prices(s+".NS",8); d=sa.add_indicators(raw); data[s]=d
def sig(entry, exit_):
    sc = pd.Series(0.0, index=entry.index); sc[exit_]= -100; sc[entry & ~exit_]=100; return sc
def run(name, f, **kw):
    for yrs in (5,7):
        rows=[]
        for s,d in data.items():
            e,x = f(d); cfg=sa.Config(years=yrs, buy_th=50, exit_th=-50, **kw)
            st=sa.backtest(d, sig(e.fillna(False),x.fillna(False)), cfg, d.index[-1]-pd.DateOffset(years=yrs))["stats"]
            rows.append((st["win_pct"],st["trades"],st["cagr"],st["bh_cagr"],st["max_dd"],min(st["profit_factor"],5),st["exposure"]))
        a=np.array(rows)
        print(f"{name:<40}{yrs}y win {a[:,0].mean():5.1f}% tr {a[:,1].mean():5.1f} CAGR {a[:,2].mean():+5.1f}% B&H {a[:,3].mean():+5.1f}% DD {a[:,4].mean():6.1f}% PF {np.median(a[:,5]):.2f} expo {a[:,6].mean():3.0f}% beat {np.mean(a[:,2]>a[:,3])*100:3.0f}%")
up = lambda d: d.Close > d.SMA200
run("P1 RSI<40 in uptrend, exit >SMA20", lambda d:(up(d)&(d.RSI<40), d.Close>d.SMA20), tp_atr=99, max_hold=20)
run("P2 RSI<35 in uptrend, exit RSI>55", lambda d:(up(d)&(d.RSI<35), d.RSI>55), tp_atr=99, max_hold=30)
run("P3 %B<0.1 in uptrend, exit %B>0.6", lambda d:(up(d)&(d.BB_PCT<0.1), d.BB_PCT>0.6), tp_atr=99, max_hold=25)
run("T1 Supertrend up & >SMA200, exit ST down", lambda d:((d.ST_DIR>0)&up(d), d.ST_DIR<0), tp_atr=99, sl_atr=4, max_hold=500)
run("T2 SMA50>SMA200 trend hold, exit cross", lambda d:((d.SMA50>d.SMA200)&up(d), d.Close<d.SMA200*0.97), tp_atr=99, sl_atr=99, max_hold=2000)
