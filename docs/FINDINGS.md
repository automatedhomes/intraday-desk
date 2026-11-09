# Backtest findings (Nov 2026)

All results use real NSE data from Yahoo Finance and include trading costs. They describe the past
on a limited window; they are not forecasts.

## 1. Daily swing engine (`stock_analyzer.py`)

Long-only, enter next open after the technical score crosses +25, stop 2×ATR, target 3×ATR,
exit if score < −25 (default; −10 whipsawed more), max 40 days, 0.15% cost per side.

- **RELIANCE, 5 years:** 25 trades, 40% wins, strategy −5.4% vs buy-and-hold +16.5%.
- **7 large caps (TCS, HDFCBANK, INFY, ITC, TATASTEEL, SBIN, BAJFINANCE):** win rates 20–45%.
- **20 large caps, many variants** (trend filter above 200-DMA, exit −25, wider targets, longer holds):
  the default wins about 42–45% of trades and roughly breaks even. Buy-and-hold made about
  +9.5% a year over 5 years and about +15% a year over 7 years. No variant beat it on most stocks.
- **Classic rule sets** (RSI dip-buying in an uptrend, Bollinger dips, Supertrend trend-following,
  50/200-DMA trend holding): dip-buying wins ~55% of trades but makes little money because it is
  in the market rarely; none beat buy-and-hold on most stocks.
- Scripts: `research/daily_variants.py`, `research/daily_rule_sets.py`.

## 2. Intraday setups (`intraday_dashboard.py`)

5-minute candles, last 60 days, ₹1,00,000 per trade, Zerodha-style charges + 0.02% slippage per side.

20 large caps (`research/intraday_costs.py`):

| Setup | Trades | Win % | Before costs / trade | Costs / trade | Net / trade |
|---|---|---|---|---|---|
| Opening Range Breakout | 776 | 30.4% | +₹20 | ₹122 | −₹102 |
| VWAP-EMA Pullback (1/day) | 1,187 | 30.6% | +₹1 | ₹122 | −₹121 |
| Supertrend + VWAP | 1,074 | 33.1% | −₹9 | ₹122 | −₹131 |

All 45 watchlist symbols (plan engine, pooled): ORB 1,842 trades, 29.8% wins, −₹96/trade;
Pullback 2,668, 29.3%, −₹122; Supertrend 2,369, 33.9%, −₹114. Shorts did a little better than
longs in this window (e.g. ORB shorts 32.6% vs longs 27.3%).

Variants tried (`research/intraday_variants.py`): minimum stop of 0.3%/0.5% of price, max one
trade a day, 15-minute candles, ₹5 lakh per trade. Before-costs edge stayed between about −₹10
and +₹57 per trade; nothing survived costs. Conclusion: these classic setups have no meaningful
edge on liquid large caps in this window; costs decide the outcome.

This matches SEBI's FY23 study: about 7 in 10 individual intraday traders in the equity cash
segment lost money.

## 3. What the plan tab showed for Mon 09 Nov 2026

Only 3 of 45 symbols had a setup with positive expected profit after costs, all small and uncertain:
WIPRO Supertrend short (48% probability of profit, range 36–60%, +₹36/trade, 26 trades),
ADANIENT ORB long (39%, +₹28), BHARTIARTL ORB long (41%, +₹20).

## 4. Swing and momentum ideas, 2016 – Nov 2026 (`research/swing_momentum.py`)

43 watchlist stocks, daily data, rules fixed before running, 0.15% cost per side, idle cash earns 0%.
Each stock gets 1/43 of capital in the per-stock strategies.

| Strategy | CAGR after costs | Max drawdown | 2016-20 | 2021-now | Notes |
|---|---|---|---|---|---|
| Nifty 50 index, hold (price only) | 11.1% (+~1.3% dividends) | −38% | 12.0% | 10.4% | benchmark |
| Equal-weight hold of the 43 stocks | 20.2% | −42% | 17.1% | 23.0% | inflated by survivorship bias |
| Trend pullback (RSI<40 above 200-DMA) | 0.0% (0.8% before costs) | −12% | −1.6% | 1.4% | 1,288 trades, 49% wins, +0.30% gross/trade: no edge |
| 50-day breakout, 3-ATR trailing stop | 5.8% (6.4% before) | −10% | 5.7% | 5.9% | 954 trades, 43% wins, +3.5% net/trade, held ~41 days |
| Momentum top 10, 12-1 months (pre-chosen) | 22.6% (23.6% before) | −39% | 20.1% | 24.8% | ~260% one-way turnover a year |
| Same + cash when Nifty < 200-DMA | 17.6% | −26% | 15.3% | 19.5% | smaller drawdown, lower return |

- Pullback dip-buying has no edge, consistent with section 1.
- Breakout has a real per-trade edge that survives costs (wins are large), but it sits in cash most of the
  time, so its total return is well below just holding. Not tested: putting idle cash in a liquid fund.
- Momentum beat the equal-weight hold by ~2.4%/yr in both halves, and every lookback/size variant with
  12-1 months beat it too; 6-1 months did not. The margin is small, and short-term capital gains tax (20%)
  on monthly rotation versus long-term tax (12.5%) on holding would likely take most or all of it.
- The universe is today's large caps, so holding them looks far better than the Nifty (20% vs 11%): that gap
  is mostly hindsight, not skill. Yahoo's NIFTYBEES series has a bad split in Dec 2019; use `^NSEI`.

## 5. Published factor rules, after costs and tax, 2016 – Nov 2026 (`research/factor_study.py`)

Web research (Nov 2026: NSE index whitepapers and factsheets, S&P DJI, commercial backtests) says momentum,
low volatility and quality are the factors with the best Indian record, and that momentum's edge is eaten
by turnover and has crashed up to ~70%. Quality needs historical fundamentals (not in free data): not tested.
Rules below follow NSE's published index methods; 0.15%/side costs; tax 20% short-term, 12.5% long-term.

| Strategy | Before costs | After costs | After tax | Max DD | 2016-20 | 2021-now |
|---|---|---|---|---|---|---|
| Nifty 50 index, hold (price only) | 11.1% | 11.1% | 10.2% (+~1.3% dividends) | −38% | 12.0% | 10.4% |
| Hold the same 43 stocks | 20.0% | 20.0% | 18.7% | −41% | 18.4% | 21.3% |
| NSE momentum, top 10, 6-monthly | 22.1% | 21.7% | 19.1% | −41% | 13.6% | 29.0% |
| Low volatility, top 10, quarterly | 14.0% | 13.7% | 12.0% | −28% | 15.0% | 12.7% |
| Momentum + low vol, top 10, 6-monthly | 16.1% | 15.8% | 13.9% | −37% | 12.1% | 19.0% |
| 12-1m momentum, top 10, monthly | 23.9% | 22.8% | 19.8% | −39% | 19.9% | 25.4% |

- After tax, the best (monthly momentum) beats holding the same stocks by ~1.1%/yr; NSE-style momentum by
  ~0.4%/yr and only in 2021-now (it lagged in 2016-20). That is too small and too unstable to call an edge.
- Low volatility gives up return for a smaller crash (−28%); it slightly beat the Nifty after tax.
- Tax costs momentum about 2-3%/yr versus ~1.3%/yr for holding, which is why holding is so hard to beat.
- Every portfolio of these stocks beats the Nifty mainly because of survivorship bias (today's winners).

## 6. Institutional portfolio techniques, after costs and tax (`research/institutional_study.py`)

Rules fixed before running; same engine and tax as section 5.

| Portfolio | After costs | After tax | Max DD | Vol | 2016-20 | 2021-now | Turnover |
|---|---|---|---|---|---|---|---|
| Nifty 50 index, hold (price only) | 11.1% | 10.2% | −38% | 16% | 12.0% | 10.4% | 9% |
| Hold the same 43 stocks | 19.9% | 18.6% | −41% | 21% | 18.4% | 21.2% | 9% |
| Momentum top 10, monthly | 22.7% | 19.7% | −39% | 22% | 19.9% | 25.2% | 293% |
| **+ banding (buy top 10, sell below rank 20)** | 24.9% | **22.1%** | −38% | 21% | 23.2% | 26.3% | 123% |
| 50% banded momentum + 50% low volatility | 19.5% | 17.3% | −33% | 17% | 19.5% | 19.5% | 110% |
| **70% Nifty + 30% banded momentum** | 15.6% | **14.1%** | −38% | 17% | 16.1% | 15.1% | 47% |
| 70% Nifty + 30% both sleeves | 14.0% | 12.6% | −36% | 16% | 15.0% | 13.1% | 42% |

Banding sensitivity (after tax): top 10 with no band 19.7%, sell below 15 22.1%, 20 22.2%, 25 25.5%, 30 26.0%;
top 15: none 20.6%, 20 21.7%, 25 21.3%, 30 22.4%. Banding helped in every setting, so the direction is robust.
Wider bands look even better partly because they hold past winners longer, which survivorship bias flatters:
keep the standard 10/20 and don't widen it on this evidence.

What changed in the dashboard: Monthly picks now uses banding (10/20) for momentum, the journal charges costs
only on picks that change, and the tab shows a 70/30 core-satellite plan. Intraday setups are unchanged: no
institutional technique fixes a strategy with no edge before costs.

## 7. Buying options on the intraday signals (`research/options_backtest.py`, `docs/options_backtest.json`)

60 sessions (13 Aug – 9 Nov 2026), every signal of the three intraday setups on 45 symbols. Long signal →
buy the at-the-money call, short → the put, nearest expiry, 1 lot. Premium by Black-Scholes at the actual
entry/exit prices and times, with each contract's implied volatility from NSE's previous-day close.
Costs: option charges (~₹68 a trade) + bid-ask allowance 0.5% (index) / 2% (stock) of premium per side.

| Setup / options on | Trades | Win (net) | Premium/lot | Before costs | Net / trade | Net % of premium |
|---|---|---|---|---|---|---|
| ORB / indices | 142 | 31.7% | ₹11,655 | +₹238 | +₹51 | +0.4% |
| ORB / stocks | 1,642 | 27.3% | ₹11,579 | +₹128 | −₹407 | −3.5% |
| Pullback / indices | 119 | 31.1% | ₹11,760 | −₹81 | −₹268 | −2.3% |
| Pullback / stocks | 2,507 | 28.2% | ₹11,310 | +₹18 | −₹503 | −4.4% |
| Supertrend / indices | 100 | 23.0% | ₹13,101 | −₹451 | −₹650 | −5.0% |
| Supertrend / stocks | 2,199 | 31.5% | ₹11,094 | +₹87 | −₹427 | −3.8% |
| **All** | **6,709** | **29.1%** | ₹11,347 | +₹63 | **−₹441** | **−3.9%** |

- Before costs the signals are roughly break-even in options too (+₹63 a trade); charges (₹68) and the
  bid-ask spread (₹436, mostly wide stock-option spreads) turn that into a loss. Expiry-day trades: −13% of premium.
- ORB on index options is the only line near zero (+₹51, 142 trades): too few trades and too small to call an edge.
- The same signals in shares: +₹6 before and −₹115 after costs per ₹1 lakh. Options add time decay and spreads;
  they don't create an edge the signal lacks.
- Model limits: constant implied volatility within the day, model prices instead of actual fills.

## How to use these findings

- Judge any new idea on its before-costs edge across many stocks first.
- Show "before costs" and "after costs" side by side, always.
- A good process outcome is often "no trade".
