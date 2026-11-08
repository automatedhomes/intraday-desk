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

## How to use these findings

- Judge any new idea on its before-costs edge across many stocks first.
- Show "before costs" and "after costs" side by side, always.
- A good process outcome is often "no trade".
