# Intraday Desk

Tools for analysing NSE stocks, built with Claude.

- **`intraday_dashboard.py`** — a web dashboard for intraday trading: TradingView-style charts with
  VWAP, EMAs, Supertrend and key levels; a ranked plan for the day with probability of profit;
  live signals for three setups; after-cost backtests; a watchlist scanner; and a Big money tab
  with NSE block/bulk deals, FII/DII flows and unusual-volume candles.
- **Options lab** tab — payoff charts, premiums and Greeks from NSE option prices, paper option trades, and a
  backtest of buying options on the dashboard's signals (after F&O charges).
- **`stock_analyzer.py`** — a swing analyzer: fundamentals + technicals + news → a buy/sell call,
  with a 5-year backtest and an HTML report.

Educational tools, not investment advice. See `docs/FINDINGS.md` for what the backtests showed.

## Move it into Claude Code

1. Unzip this folder somewhere easy, for example your Documents folder.
2. Install Claude Code, or use the **Code** tab in the Claude desktop app (no install needed there):
   - macOS / Linux: `curl -fsSL https://claude.ai/install.sh | bash`
   - Windows PowerShell: `irm https://claude.ai/install.ps1 | iex`
3. Open a terminal in the folder and start it:
   ```
   cd intraday-desk
   claude
   ```
   (In the desktop app: Code tab → choose the `intraday-desk` folder.)
4. Paste the prompt from `KICKOFF_PROMPT.md`. Claude Code reads `CLAUDE.md` automatically, so it
   starts with the full project history and rules.

## Run it yourself (without Claude Code)

Needs Python 3.10+.

```
pip install -r requirements.txt
python intraday_dashboard.py        # opens http://127.0.0.1:8050 (keep the window open)
python stock_analyzer.py RELIANCE   # writes report_RELIANCE_NS.html
python tests/test_smoke.py          # offline checks
```

If port 8050 is busy: `python intraday_dashboard.py --port 9000`, then open http://127.0.0.1:9000.

## Folder

```
intraday-desk/
├── CLAUDE.md             project memory for Claude Code
├── KICKOFF_PROMPT.md     first message to paste into Claude Code
├── README.md
├── requirements.txt
├── intraday_dashboard.py
├── stock_analyzer.py
├── tests/test_smoke.py   offline tests
├── research/             backtest studies behind the findings
├── docs/FINDINGS.md      backtest results
├── snapshot/             last static snapshot (data up to 06 Nov 2026)
└── samples/              example Reliance report and trade list
```
