# Shark Supertrend 14/3 - FAST AUTO GitHub Backtest

One-click GitHub Actions backtest for **all Shark Exchange USDT pairs**.

## Fixed automatic settings
- Market: USDT
- Symbols: ALL available Shark USDT pairs
- Timeframe: 5m
- History: latest 864 candles per pair (~3 days)
- Supertrend: 14 / 3
- Margin: INR 100 per trade
- Leverage: 10x
- Simultaneous positions: Unlimited
- Fees: Off
- Slippage: 0
- Real trading: **Never**

## Run
GitHub → Actions → **Shark Supertrend 14/3 - FAST AUTO Backtest** → **Run workflow**.

No inputs are required.

The workflow downloads one kline request per pair and paces requests at about 1.05 seconds to stay around Shark's public API request limit. Results are uploaded automatically as the `shark-backtest-results` artifact.

Output files:
- `backtest_trades.csv`
- `backtest_equity.csv`
- `backtest_summary.json`
- `backtest_summary.txt`

Downloaded market data is not persisted between GitHub Actions runs; each run fetches fresh data.
