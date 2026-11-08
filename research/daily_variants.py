"""Study: daily engine variants (trend filter, exits, targets) across 20 stocks, 5 and 7 years. Needs network; takes a few minutes.

Run from the project root:  python research/daily_variants.py
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import stock_analyzer as sa
syms = "RELIANCE TCS HDFCBANK INFY ITC TATASTEEL SBIN BAJFINANCE LT HINDUNILVR MARUTI SUNPHARMA TITAN ASIANPAINT AXISBANK ONGC NTPC TATAPOWER DLF ADANIPORTS".split()
data = {}
for s in syms:
    try:
        raw,_ = sa.fetch_prices(s+".NS", 8); d = sa.add_indicators(raw); _, sc = sa.technical_components(d); data[s]=(d,sc)
    except Exception as e: print(s, e)
def run(name, cfg, filt=None):
    rows=[]
    for s,(d,sc) in data.items():
        score = sc.copy()
        if filt is not None:
            m = filt(d)
            score = score.where(m | (score < cfg.buy_th), cfg.buy_th - 1)  # block entries when filter false
        start = d.index[-1] - pd.DateOffset(years=cfg.years)
        st = sa.backtest(d, score, cfg, start)["stats"]
        rows.append((st["win_pct"], st["trades"], st["cagr"], st["bh_cagr"], st["max_dd"], st["profit_factor"]))
    a = np.array(rows, dtype=float); a[:,5]=np.where(np.isinf(a[:,5]),3,a[:,5])
    print(f"{name:<34} win {a[:,0].mean():5.1f}%  trades {a[:,1].mean():5.1f}  CAGR {a[:,2].mean():+5.1f}% (B&H {a[:,3].mean():+5.1f}%)  maxDD {a[:,4].mean():6.1f}%  PF {np.median(a[:,5]):.2f}  beat-B&H {np.mean(a[:,2]>a[:,3])*100:.0f}%")
C = sa.Config
trend = lambda d: (d.Close > d.SMA200) & (d.SMA50 > d.SMA200)
for yrs in (5, 7):
    print(f"--- {yrs} years, {len(data)} stocks")
    run("A default", C(years=yrs))
    run("B trend filter", C(years=yrs), trend)
    run("C exit -25", C(years=yrs, exit_th=-25))
    run("D trend + exit -25", C(years=yrs, exit_th=-25), trend)
    run("E trend + exit -25 + tp 4", C(years=yrs, exit_th=-25, tp_atr=4), trend)
    run("F trend + exit -25 + hold 60", C(years=yrs, exit_th=-25, max_hold=60), trend)
    run("G trend + no signal exit, hold 60", C(years=yrs, exit_th=-101, max_hold=60), trend)
