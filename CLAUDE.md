# Intraday Desk — project memory

Indian stock market (NSE/BSE) analysis and intraday trading tools, built in a claude.ai chat
(Nov 2026) and moved here. The owner is a retail trader in Delhi (IST, UTC+5:30) who is not a
programmer: explain commands and results in plain words, run things for them, and give them a
URL or file they can open rather than instructions to follow.

## What's here

| Path | What it is |
|---|---|
| `stock_analyzer.py` | Swing/positional analyzer (daily data). Fundamentals + technicals + news → STRONG BUY…STRONG SELL call, trade plan, 5-year backtest, signal accuracy. Writes `report_<TICKER>.html` + `trades_<TICKER>.csv`. |
| `intraday_dashboard.py` | Intraday web app (local server on 127.0.0.1:8050). Tabs: Today's plan, Chart, Scanner, Big money. Also builds a static snapshot (`--snapshot DIR`). |
| `tests/test_smoke.py` | Offline tests on synthetic data (no network). Run after every change. |
| `research/` | One-off backtest studies that produced `docs/FINDINGS.md` (need network, take minutes). |
| `snapshot/` | Last built static snapshot (data up to Fri 06 Nov 2026 15:25). Regenerable. |
| `samples/` | Example output of `stock_analyzer.py RELIANCE`. |
| `docs/FINDINGS.md` | Backtest results and what they mean. Read before changing strategies. |

## Commands

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # yfinance, pandas, numpy (+ pytest for tests)
python tests/test_smoke.py                            # or: python -m pytest -q tests
python intraday_dashboard.py                          # opens http://127.0.0.1:8050
python intraday_dashboard.py --port 9000 --no-browser
python intraday_dashboard.py --snapshot snapshot      # static page + data/*.json, no server needed
python stock_analyzer.py RELIANCE                     # also: TCS INFY --years 8, SBIN --bse, --no-news
```

## intraday_dashboard.py architecture (single file, by design)

- **Data**: `fetch()` (one symbol, yfinance `history`, cached 25 s) and `batch()` (whole `WATCHLIST`,
  `yf.download`, cached 120 s). `clean()` converts to naive IST and keeps 09:15–15:29 only.
  Yahoo limits: 1m → 7 days, 2m/5m/15m → 60 days. Indices (`^NSEI`, `^NSEBANK`) have zero volume.
- **Indicators**: `compute()` adds VWAP (resets daily; price-only fallback for indices), EMA9/21, RSI14,
  ATR14 (Wilder), Supertrend(10,3), volume ratio, 15-min opening range (`OR_H/OR_L`, NaN before 9:30),
  previous-day levels, CPR (`TC/BC/P`), pivots R1/R2/S1/S2.
- **Bias**: `bias()` (one row, with explanations) and `bias_series()` (vectorised). Keep them identical
  (a test checks this). Score −100..+100 from VWAP 25, EMA 20, Supertrend 20, RSI 15, CPR 10, OR 10.
- **Strategies**: `STRATS` + `signals()` → ORB (15-min OR + VWAP + volume ≥1.2×), VWAP-EMA pullback
  (max 1/day), Supertrend + VWAP (within 3 candles of a flip, once per leg). All target 2R.
- **Backtest**: `simulate()` — signal on candle close, entry next candle's open, stop checked before
  target inside a candle, exit signals execute next open, no entries after 14:30, square-off 15:15,
  per-trade notional = capital (default ₹1,00,000). Each trade records `bias` at the signal candle.
- **Costs**: `charges()` = Zerodha-style NSE intraday: brokerage min(₹20, 0.03%) per leg, STT 0.025%
  sell, exchange 0.00307%, SEBI ₹10/crore, stamp 0.003% buy, GST 18% on brokerage+exchange+SEBI,
  plus slippage % per side (default 0.02). The JS calculator duplicates this — change both together.
- **Plan engine**: `plan()` ranks every symbol × setup by expected net ₹/trade. Probability of profit =
  stock's same-direction win rate shrunk toward the all-stocks rate with `PRIOR = 20` trades; range is a
  90% normal approximation. Direction = bias at last close (≥20 long, ≤−20 short, else either).
  `next_session()` uses `NSE_HOLIDAYS` — **update that set every year** from NSE's holiday circular.
- **Big money**: `institutional()` → NSE `fiidiiTradeReact` (market-wide FII/DII cash flows, latest day),
  `snapshot-capital-market-largedeal` (bulk + block deals, latest day; client names), and
  `volume_spikes()` (5-min candles ≥4× the usual volume for that time of day, last 5 sessions).
  `INSTITUTION_RE` guesses "likely institution" from client names.
- **Web page**: built from `LIB` + `CSS` + `BODY` + `JS` strings. `local_page()` (served at `/`) fetches
  `/api/analyze`, `/api/scan`, `/api/plan`, `/api/institutional`, `/api/watchlist`. `hosted_page()`
  sets `window.SNAP_INDEX` and the same JS reads `data/*.json` instead (static mode: `STATIC` in JS).
  Charts use TradingView Lightweight Charts **v4.2.0** from unpkg (v5 changed the API — don't bump
  casually). Chart times are IST wall-clock encoded as UTC epoch seconds.
- CSS is dark-first with light-mode tokens under `prefers-color-scheme` and `[data-theme]`; keep
  colours as tokens on `:root`.

## Hard constraints learned the hard way

- **TradingView embeddable widgets refuse NSE symbols** and give BSE only daily candles, so they
  can't show Indian intraday charts. Use Lightweight Charts with our own data; link out to
  `https://in.tradingview.com/chart/?symbol=NSE:<SYM>` (TradingView uses `M_M`, `BAJAJ_AUTO`).
- **Yahoo Finance blocks browser (CORS) requests**, so a static page can't fetch live prices. Live
  data needs the Python server. A hosted snapshot is stale by design and says so in a banner.
- **NSE API** works with a browser User-Agent + `Referer: https://www.nseindia.com/`. The
  `historical/bulk-deals` and `historical/block-deals` endpoints return an HTML block page to
  scripts, so only the latest session's deals are available.
- NSE does **not** publish the time of bulk/block deals or who is behind any on-screen trade. Never
  present volume spikes as "institutional buying" — they are total volume, buyer unknown.
- Recent symbol changes: Zomato → `ETERNAL`; Tata Motors demerged → `TMPV` (passenger vehicles).

## Rules for changing strategies and results

1. Read `docs/FINDINGS.md`. Before costs the intraday setups make roughly ₹0–20 per ₹1 lakh trade;
   costs are ~₹80–120, so all three lose after costs. The daily engine doesn't beat buy-and-hold.
2. **No curve-fitting.** Don't tune thresholds until a 60-day backtest looks good. Any new rule must
   be tested across ≥20 stocks and both 5m and 15m, and judged on *before-costs* edge first.
3. Always show results **after charges**, next to before-charges. Never call anything "profitable"
   or a "sure trade". "Probability of profit" means historical net win rate, with its range.
4. When nothing has positive expected value, the honest output is "the best trade may be no trade".
5. These are educational tools, not investment advice. Keep the disclaimers on every page.
6. Run `python tests/test_smoke.py` after every change; add a test for any new calculation.

## Hosted snapshot

A snapshot of the dashboard was published from claude.ai as a private artifact:
https://claude.ai/artifact/D5RPGKpLiHQDEvNR51CvgC (owner's account; data up to 06 Nov 2026).
From here, regenerate with `--snapshot` and host the folder on any static host (GitHub Pages,
Netlify, Cloudflare Pages) if the owner wants a shareable link.

## Live hosting (set up 09 Nov 2026)

- **GitHub Pages:** https://automatedhomes.github.io/intraday-desk/ (public repo `automatedhomes/intraday-desk`).
  GitHub's cron proved unreliable (1 run on 09 Nov), so cron-job.org POSTs workflow_dispatch every 5 min;
  non-forced dispatches act like scheduled runs. `gh workflow run live-snapshot.yml -f force=true` builds any time.
  `.github/workflows/live-snapshot.yml` runs every 5 min 03:45-10:10 UTC Mon-Fri and builds with
  `--snapshot site --standalone --if-open` (`session_window()` skips holidays); "Run workflow" builds anytime.
  GitHub's cron often runs late. Public repos' schedules pause after 60 days without a commit.
- **claude.ai artifact:** https://claude.ai/artifact/HkKNBuUQGBrM9dZCsp1444, republished by a session-only
  Claude cron job via `live_tick.ps1` (not in git); needs the PC and Claude session open, expires after 7 days.

## Monthly picks tab and paper journal (added 09 Nov 2026)

- `picks()` → `/api/picks`, `data/_picks.json`: momentum top 10 (12-1m, monthly) and low volatility top 10
  (1-year vol, quarterly), ranked on closes up to the last completed month-end (`compute_picks`).
- `journal/paper_journal.json` is the forward paper record. Only the scheduled GitHub run writes it
  (`--if-open --journal`) and commits it, at live prices; local runs read it. **`git pull --rebase` before pushing.**
  `update_journal` opens a period when a rebalance is due and closes the previous one at the same prices.
- Momentum uses banding (`PICK_BAND = 20`): `picks()` passes the open journal holdings to `compute_picks(held=...)`.
  `CORE_PCT = 70` drives the core-satellite card. Evidence: docs/FINDINGS.md section 6.

## Backlog ideas (ask the owner before starting)

- Scheduled snapshot rebuild during market hours (hourly) + static hosting, for a near-live phone link.
- Split `intraday_dashboard.py` into a package only if it keeps growing; keep `python intraday_dashboard.py` working.
- Paper-trading journal: log the plan's top pick each day and its real outcome, to measure live accuracy.
- Options/F&O, Telegram alerts for new signals, broker API (Kite Connect) for real-time ticks.
