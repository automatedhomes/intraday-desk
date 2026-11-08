Paste everything below the line into Claude Code as your first message, from inside the `intraday-desk` folder.

---

I'm moving my Indian stock market project here from a claude.ai chat. This folder has everything:
a swing analyzer (`stock_analyzer.py`), an intraday web dashboard (`intraday_dashboard.py`) with
Today's plan, Chart, Scanner and Big money tabs, tests, research scripts, the last data snapshot,
and notes. I'm not a programmer, so please do the work yourself and explain results in plain words.

First, read `CLAUDE.md` and `docs/FINDINGS.md`. Then:

1. Set up a Python virtual environment in this folder and install `requirements.txt`.
2. Run the offline tests (`python tests/test_smoke.py`) and tell me if they pass.
3. Start the live dashboard (`python intraday_dashboard.py`). Check that `/api/analyze?symbol=RELIANCE`,
   `/api/plan` and `/api/institutional` return data, then tell me the exact link to open in my browser.
   If the market is open, tell me what the Today's plan tab shows as the top setup and its
   probability of profit. If it's closed, say which session the plan is for.
4. Run `python stock_analyzer.py RELIANCE` and summarise the call and the backtest in 3-4 lines.

Rules for this project:
- Don't tune strategies just to make a backtest look profitable. Always show results after charges,
  next to before charges, and say plainly when nothing has an edge.
- Keep it educational, not investment advice. Never place trades or connect to my broker account
  without asking me first.
- If something fails (missing package, port busy, data source blocked), fix it or tell me exactly
  what you need from me.

When all of that works, suggest the 3 most useful next improvements (for example a near-live link
I can open on my phone, a paper-trading journal that tracks the plan's top pick each day, or
Telegram alerts for new signals) and wait for me to choose.
