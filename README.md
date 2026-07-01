# kalshi-paper-live

Kalshi 15-minute crypto up/down markets — **extreme-tails mean-reversion strategy**,
live **paper trading** engine (no real orders, ever) with real Kalshi prices and real
settlement results.

- Engine: GitHub Actions (`.github/workflows/engine.yml`), self-chains up to a 12h campaign.
- State: pushed to the `state` branch (`state.json`, `trades.csv`).
- Dashboard: GitHub Pages (`docs/index.html`) reading the state branch. No auth, read-only.

Strategy (from reverse-engineering Polymarket/Kalshi opening prices):
market makers price short-term mean-reversion at open but **underprice it at extremes**.
Bet only when the previous 15m return is in the trailing 10% tails AND a logistic model
(mean-reversion + order-flow features, trained on ~90d of 15m data) confirms direction.
Entry = real Kalshi ask captured in the first 2.5 minutes of the window.

This is research tooling; nothing here is financial advice.
