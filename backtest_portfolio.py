
import os
import time
import math
import json
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import numpy as np

BASE_URL = os.getenv("SHARK_BASE_URL", "https://api.sharkexchange.in")
INTERVAL = os.getenv("TIMEFRAME", "5m")
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "3"))
MARGIN_INR = float(os.getenv("MARGIN_INR", "100"))
LEVERAGE = float(os.getenv("LEVERAGE", "10"))
MAX_OPEN_POSITIONS = None  # Unlimited
PRICE_TYPE = os.getenv("PRICE_TYPE", "MARK_PRICE")
MARKET = os.getenv("MARKET", "USDT")
SYMBOLS_ENV = os.getenv("SYMBOLS", "ALL")
REQUEST_SLEEP = float(os.getenv("REQUEST_SLEEP", "1.01"))  # ~59 requests/min, within Shark public limit
AUTO_CANDLES = int(os.getenv("AUTO_CANDLES", "864"))  # automatic fast mode: latest 3 days on 5m; one API request/pair
STARTING_CAPITAL = float(os.getenv("STARTING_CAPITAL_INR", "1000"))
INCLUDE_FEES = os.getenv("INCLUDE_FEES", "false").lower() == "true"
FEE_RATE = float(os.getenv("FEE_RATE", "0.0005"))  # per side, optional
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "0"))  # optional

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Shark-GitHub-Backtest/1.0"})


def request_json(method, path, **kwargs):
    url = BASE_URL.rstrip("/") + path
    last = None
    for attempt in range(4):
        try:
            r = SESSION.request(method, url, timeout=30, **kwargs)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request failed: {url} -> {last}")


def unwrap_data(obj):
    if isinstance(obj, dict):
        if "data" in obj:
            return obj["data"]
        if "result" in obj:
            return obj["result"]
    return obj


def get_symbols():
    """
    Fetch actual Shark trading-pair names.

    Shark exchangeInfo contains both:
      name         -> actual pair, e.g. BTCUSDT
      contractName -> human-readable name, e.g. Bitcoin

    We must use `name` for Kline requests.
    """
    raw = request_json(
        "GET",
        "/v1/exchange/exchangeInfo",
        params={"market": MARKET},
    )
    raw = unwrap_data(raw)

    contracts = None
    if isinstance(raw, dict):
        if isinstance(raw.get("contracts"), list):
            contracts = raw["contracts"]
        else:
            for key in ("data", "result"):
                value = raw.get(key)
                if isinstance(value, dict) and isinstance(value.get("contracts"), list):
                    contracts = value["contracts"]
                    break
    elif isinstance(raw, list):
        contracts = raw

    if not contracts:
        raise RuntimeError(
            f"Shark exchangeInfo returned no contracts for market {MARKET}."
        )

    symbols = []
    for item in contracts:
        if isinstance(item, str):
            candidate = item.strip().upper()
        elif isinstance(item, dict):
            # IMPORTANT: never use contractName here; that is the human name.
            candidate = (
                item.get("name")
                or item.get("contractPair")
                or item.get("symbol")
                or item.get("pair")
            )
            candidate = str(candidate).strip().upper() if candidate else ""
        else:
            continue

        if candidate.endswith("USDT"):
            symbols.append(candidate)

    symbols = sorted(set(symbols))

    if SYMBOLS_ENV.upper() != "ALL":
        requested = [x.strip().upper() for x in SYMBOLS_ENV.split(",") if x.strip()]
        requested = [
            x if x.endswith("USDT") else x + "USDT"
            for x in requested
        ]
        symbols = [x for x in requested if x in symbols]

    if not symbols:
        raise RuntimeError("No valid USDT trading pairs found.")

    print(f"Valid {MARKET} trading pairs found: {len(symbols)}")
    print("Sample pairs:", ", ".join(symbols[:10]))
    return symbols

def fetch_klines(pair):
    """Fetch the latest automatic fast-backtest window.

    To keep the full ALL-USDT scan around 5-6 minutes, this mode deliberately
    requests <=1000 candles per pair, so every pair needs only ONE kline API
    request.  With 864 x 5-minute candles this is about 3 days of history.
    Shark's documented public limit is 60 requests/minute, so requests are
    paced at ~1.01 seconds.
    """
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{pair}_{INTERVAL}.csv")

    if os.path.exists(cache_file):
        try:
            cached = pd.read_csv(cache_file)
            cached = cached.sort_values("timestamp").drop_duplicates("timestamp")
            if len(cached) >= min(AUTO_CANDLES, ATR_PERIOD + 5):
                return cached.tail(AUTO_CANDLES).reset_index(drop=True)
        except Exception:
            pass

    payload = {"pair": pair, "interval": INTERVAL, "limit": min(1000, AUTO_CANDLES)}
    data = request_json(
        "POST",
        "/v1/market/klines",
        params={"priceType": PRICE_TYPE},
        json=payload,
    )
    data = unwrap_data(data)

    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    rows = []
    for x in data:
        if not isinstance(x, dict):
            continue
        try:
            rows.append({
                "timestamp": int(x.get("startTime", x.get("t"))),
                "open": float(x["open"] if "open" in x else x["o"]),
                "high": float(x["high"] if "high" in x else x["h"]),
                "low": float(x["low"] if "low" in x else x["l"]),
                "close": float(x["close"] if "close" in x else x["c"]),
                "volume": float(x.get("volume", x.get("v", 0))),
            })
        except (TypeError, ValueError, KeyError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("timestamp").drop_duplicates("timestamp")
    df = df.tail(AUTO_CANDLES).reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    return df


def atr_wilder(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def supertrend(df, period=14, multiplier=3.0):
    out = df.copy()
    atr = atr_wilder(out, period)
    hl2 = (out["high"] + out["low"]) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    trend = pd.Series(index=out.index, dtype="int64")
    st = pd.Series(index=out.index, dtype="float64")

    for i in range(len(out)):
        if i == 0:
            trend.iloc[i] = 1
            st.iloc[i] = np.nan
            continue

        if pd.isna(atr.iloc[i]):
            trend.iloc[i] = 1
            st.iloc[i] = np.nan
            continue

        if pd.isna(final_upper.iloc[i-1]) or pd.isna(final_lower.iloc[i-1]):
            final_upper.iloc[i] = basic_upper.iloc[i]
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_upper.iloc[i] = (
                basic_upper.iloc[i]
                if basic_upper.iloc[i] < final_upper.iloc[i-1]
                or out["close"].iloc[i-1] > final_upper.iloc[i-1]
                else final_upper.iloc[i-1]
            )
            final_lower.iloc[i] = (
                basic_lower.iloc[i]
                if basic_lower.iloc[i] > final_lower.iloc[i-1]
                or out["close"].iloc[i-1] < final_lower.iloc[i-1]
                else final_lower.iloc[i-1]
            )

        if pd.isna(st.iloc[i-1]):
            trend.iloc[i] = 1 if out["close"].iloc[i] >= hl2.iloc[i] else -1
        elif trend.iloc[i-1] == -1:
            trend.iloc[i] = 1 if out["close"].iloc[i] > final_upper.iloc[i] else -1
        else:
            trend.iloc[i] = -1 if out["close"].iloc[i] < final_lower.iloc[i] else 1

        st.iloc[i] = final_lower.iloc[i] if trend.iloc[i] == 1 else final_upper.iloc[i]

    out["atr"] = atr
    out["supertrend"] = st
    out["trend"] = trend
    return out


def slip_price(price, side, is_entry):
    # Positive bps makes execution worse.
    bps = SLIPPAGE_BPS / 10000.0
    if side == "LONG":
        return price * (1 + bps) if is_entry else price * (1 - bps)
    return price * (1 - bps) if is_entry else price * (1 + bps)


def trade_pnl(entry, exit_, side):
    if side == "LONG":
        pct = (exit_ - entry) / entry
    else:
        pct = (entry - exit_) / entry
    pnl = MARGIN_INR * LEVERAGE * pct

    if INCLUDE_FEES:
        notional_entry = MARGIN_INR * LEVERAGE
        notional_exit = MARGIN_INR * LEVERAGE * (exit_ / entry)
        pnl -= (notional_entry + notional_exit) * FEE_RATE

    return pnl, pct * 100


def build_signals(df, symbol):
    d = supertrend(df, ATR_PERIOD, ATR_MULTIPLIER)
    signals = []
    # i = crossing candle; i+1 = confirmation candle.
    for i in range(1, len(d) - 1):
        cross = d.iloc[i]
        confirm = d.iloc[i + 1]
        if pd.isna(cross.supertrend) or pd.isna(confirm.supertrend):
            continue

        long_cross = cross.close > cross.supertrend and d.iloc[i-1].close <= d.iloc[i-1].supertrend
        short_cross = cross.close < cross.supertrend and d.iloc[i-1].close >= d.iloc[i-1].supertrend

        if long_cross and confirm.close > cross.high:
            prev_low = d.iloc[i-1].low
            sl = min(cross.low, prev_low)
            signals.append({
                "symbol": symbol,
                "side": "LONG",
                "cross_i": i,
                "entry_i": i + 1,
                "entry_time": int(confirm.timestamp),
                "entry": float(confirm.close),
                "sl": float(sl),
            })

        if short_cross and confirm.close < cross.low:
            prev_high = d.iloc[i-1].high
            sl = max(cross.high, prev_high)
            signals.append({
                "symbol": symbol,
                "side": "SHORT",
                "cross_i": i,
                "entry_i": i + 1,
                "entry_time": int(confirm.timestamp),
                "entry": float(confirm.close),
                "sl": float(sl),
            })
    return d, signals


def simulate_symbol(df, signals):
    # Returns signal events; portfolio engine decides concurrency and exits.
    if df.empty:
        return []
    for s in signals:
        s["df"] = df
    return signals


def run_portfolio(data_by_symbol):
    all_signals = []
    for symbol, (df, signals) in data_by_symbol.items():
        all_signals.extend(simulate_symbol(df, signals))

    all_signals.sort(key=lambda x: (x["entry_time"], x["symbol"], x["side"]))

    by_time = {}
    for s in all_signals:
        by_time.setdefault(s["entry_time"], []).append(s)

    positions = {}
    trades = []
    equity = STARTING_CAPITAL
    equity_curve = []

    # Map timestamp -> row index for each symbol.
    index_maps = {
        sym: {int(ts): idx for idx, ts in enumerate(df.timestamp)}
        for sym, (df, _) in data_by_symbol.items()
    }

    all_times = sorted(set(
        int(ts)
        for df, _ in data_by_symbol.values()
        for ts in df.timestamp
    ))

    for ts in all_times:
        # 1) Manage existing positions using this candle.
        for key in list(positions.keys()):
            pos = positions[key]
            df = data_by_symbol[pos["symbol"]][0]
            idx = index_maps[pos["symbol"]].get(ts)
            if idx is None or idx <= pos["entry_i"]:
                continue

            row = df.iloc[idx]
            exit_price = None
            reason = None

            if pos["side"] == "LONG":
                if row.low <= pos["sl"]:
                    exit_price = pos["sl"]
                    reason = "SL"
                elif row.trend == -1 and row.close < row.supertrend:
                    exit_price = row.close
                    reason = "SUPERTREND_REVERSAL"
            else:
                if row.high >= pos["sl"]:
                    exit_price = pos["sl"]
                    reason = "SL"
                elif row.trend == 1 and row.close > row.supertrend:
                    exit_price = row.close
                    reason = "SUPERTREND_REVERSAL"

            if exit_price is not None:
                exit_price = slip_price(float(exit_price), pos["side"], False)
                pnl, move_pct = trade_pnl(pos["entry_exec"], exit_price, pos["side"])
                equity += pnl
                trades.append({
                    "symbol": pos["symbol"],
                    "side": pos["side"],
                    "entry_time_utc": datetime.fromtimestamp(pos["entry_time"]/1000, tz=timezone.utc).isoformat(),
                    "exit_time_utc": datetime.fromtimestamp(ts/1000, tz=timezone.utc).isoformat(),
                    "entry_price": pos["entry_exec"],
                    "exit_price": exit_price,
                    "sl_price": pos["sl"],
                    "margin_inr": MARGIN_INR,
                    "leverage": LEVERAGE,
                    "pnl_inr": pnl,
                    "move_pct": move_pct,
                    "exit_reason": reason,
                    "bars_held": idx - pos["entry_i"],
                })
                del positions[key]

        # 2) Add new entries. Signals at same timestamp are alphabetically deterministic.
        for s in by_time.get(ts, []):
            # Unlimited simultaneous positions.
            key = s["symbol"]
            if key in positions:
                continue
            entry_exec = slip_price(s["entry"], s["side"], True)
            positions[key] = {
                **s,
                "entry_exec": entry_exec,
            }

        # 3) Mark-to-market equity for drawdown.
        mtm = equity
        for pos in positions.values():
            df = data_by_symbol[pos["symbol"]][0]
            idx = index_maps[pos["symbol"]].get(ts)
            if idx is None:
                continue
            px = float(df.iloc[idx].close)
            if pos["side"] == "LONG":
                pct = (px - pos["entry_exec"]) / pos["entry_exec"]
            else:
                pct = (pos["entry_exec"] - px) / pos["entry_exec"]
            mtm += MARGIN_INR * LEVERAGE * pct
        equity_curve.append((ts, mtm, len(positions)))

    # Force-close positions at final available close.
    for key, pos in list(positions.items()):
        df = data_by_symbol[pos["symbol"]][0]
        row = df.iloc[-1]
        exit_price = slip_price(float(row.close), pos["side"], False)
        pnl, move_pct = trade_pnl(pos["entry_exec"], exit_price, pos["side"])
        equity += pnl
        trades.append({
            "symbol": pos["symbol"],
            "side": pos["side"],
            "entry_time_utc": datetime.fromtimestamp(pos["entry_time"]/1000, tz=timezone.utc).isoformat(),
            "exit_time_utc": datetime.fromtimestamp(int(row.timestamp)/1000, tz=timezone.utc).isoformat(),
            "entry_price": pos["entry_exec"],
            "exit_price": exit_price,
            "sl_price": pos["sl"],
            "margin_inr": MARGIN_INR,
            "leverage": LEVERAGE,
            "pnl_inr": pnl,
            "move_pct": move_pct,
            "exit_reason": "BACKTEST_END",
            "bars_held": max(0, len(df) - 1 - pos["entry_i"]),
        })

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity_curve, columns=["timestamp", "equity_mtm", "open_positions"])

    if eq_df.empty:
        max_dd = 0.0
    else:
        peak = eq_df.equity_mtm.cummax()
        dd = eq_df.equity_mtm - peak
        max_dd = float(dd.min())

    if trades_df.empty:
        summary = {
            "starting_capital_inr": STARTING_CAPITAL,
            "ending_capital_inr": STARTING_CAPITAL,
            "total_pnl_inr": 0.0,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_inr": max_dd,
            "max_simultaneous_positions": int(eq_df.open_positions.max()) if not eq_df.empty else 0,
        }
    else:
        wins = int((trades_df.pnl_inr > 0).sum())
        losses = int((trades_df.pnl_inr < 0).sum())
        gross_profit = float(trades_df.loc[trades_df.pnl_inr > 0, "pnl_inr"].sum())
        gross_loss = float(-trades_df.loc[trades_df.pnl_inr < 0, "pnl_inr"].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else math.inf
        summary = {
            "starting_capital_inr": STARTING_CAPITAL,
            "ending_capital_inr": float(STARTING_CAPITAL + trades_df.pnl_inr.sum()),
            "total_pnl_inr": float(trades_df.pnl_inr.sum()),
            "total_trades": len(trades_df),
            "wins": wins,
            "losses": losses,
            "win_rate_pct": wins / len(trades_df) * 100,
            "gross_profit_inr": gross_profit,
            "gross_loss_inr": gross_loss,
            "profit_factor": pf,
            "average_trade_inr": float(trades_df.pnl_inr.mean()),
            "best_trade_inr": float(trades_df.pnl_inr.max()),
            "worst_trade_inr": float(trades_df.pnl_inr.min()),
            "max_drawdown_inr": max_dd,
            "max_simultaneous_positions": int(eq_df.open_positions.max()) if not eq_df.empty else 0,
        }

    return trades_df, eq_df, summary


def main():
    print("=" * 70)
    print("SHARK EXCHANGE — SUPERTREND 14/3 — 5M PORTFOLIO BACKTEST")
    print("=" * 70)
    print(f"Market={MARKET} | Auto history={AUTO_CANDLES} candles/pair | Margin=₹{MARGIN_INR} | Leverage={LEVERAGE}x | Max positions=UNLIMITED")
    print("AUTO MODE: latest 2016 candles per pair (7 days on 5m).")
    print("Data is cached locally so reruns do not re-download completed pairs.")
    print("API pacing: {:.2f}s between requests.".format(REQUEST_SLEEP))
    print("No live trading. Public market data only.")

    symbols = get_symbols()
    print(f"Symbols selected: {len(symbols)}")

    data = {}
    failed = []

    for n, symbol in enumerate(symbols, 1):
        try:
            print(f"[{n}/{len(symbols)}] {symbol} ...", flush=True)
            df = fetch_klines(symbol)
            if len(df) < ATR_PERIOD + 5:
                print("  skipped: insufficient candles")
                continue
            d, signals = build_signals(df, symbol)
            data[symbol] = (d, signals)
            print(f"  candles={len(df)}, signals={len(signals)}")
        except Exception as e:
            failed.append((symbol, str(e)))
            print(f"  FAILED: {e}")

    if not data:
        raise RuntimeError("No symbol data was successfully downloaded.")

    trades, equity, summary = run_portfolio(data)
    trades.to_csv("backtest_trades.csv", index=False)
    equity.to_csv("backtest_equity.csv", index=False)

    with open("backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    with open("backtest_summary.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nsymbols_requested: {len(symbols)}\n")
        f.write(f"symbols_backtested: {len(data)}\n")
        f.write(f"symbols_failed: {len(failed)}\n")
        if failed:
            f.write("\nFailed symbols:\n")
            for s, e in failed:
                f.write(f"{s}: {e}\n")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\nFiles: backtest_trades.csv, backtest_equity.csv, backtest_summary.json, backtest_summary.txt")


if __name__ == "__main__":
    main()
