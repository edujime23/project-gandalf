#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest engine (remastered):
- Uses only items with sufficient history to support long features/signals.
- Execution modes:
  - mid (default): execute at mid price (average of buy/sell medians); good for signal quality sanity.
  - realistic: BUY at sell_median, SELL at buy_median (includes spread headwind).
- Optional slippage and fee.
- Single position per item with capped sizing.
- Stores summary metrics in backtest_results and per-trade PnL in profit_tracking.
"""

import os
import json
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# ------------- Configuration via env -------------
SUPABASE_URL = os.getenv('SUPABASE_URL', '').rstrip('/')
SUPABASE_SERVICE_ROLE = os.getenv('SUPABASE_SERVICE_ROLE') or os.getenv('SUPABASE_KEY')

# Backtest window
LOOKBACK_DAYS = int(os.getenv('BACKTEST_DAYS', os.getenv('TRAIN_DAYS', '30')))

# Execution
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "mid").lower()  # 'mid' or 'realistic'
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0.0"))       # e.g., 0.002 -> 0.2%
FEE_PCT = float(os.getenv("FEE_PCT", "0.0"))                 # e.g., 0.001 -> 0.1%

# Position sizing
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000"))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.05"))  # 5% of equity
MAX_POSITION_VALUE = float(os.getenv("MAX_POSITION_VALUE", "2000"))

# Strategy params
MIN_ITEM_POINTS = int(os.getenv("MIN_ITEM_POINTS", "50"))  # minimum points per item to even consider
MA_FAST = int(os.getenv("MA_FAST", "5"))
MA_MID = int(os.getenv("MA_MID", "20"))
MA_SLOW = int(os.getenv("MA_SLOW", "50"))
VOL_MA = int(os.getenv("VOL_MA", "20"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
VOL_SURGE = float(os.getenv("VOL_SURGE", "1.5"))           # volume > VOL_MA * VOL_SURGE

# Strategy identity
STRATEGY_NAME = os.getenv("BACKTEST_STRATEGY_NAME", "Enhanced_MA_Strategy")

# Required history gates (align with trainer’s long windows if you want)
REQUIRED_HISTORY_STEPS = int(os.getenv("REQUIRED_HISTORY_STEPS", "174"))  # 168 + horizon(6)
# ------------------------------------------------

def sb_headers_json():
    key = SUPABASE_SERVICE_ROLE
    return {
        'apikey': key,
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json'
    }

def require_env():
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE")

def fetch_market_data(days=LOOKBACK_DAYS):
    """Paginate and fetch ascending, typed DataFrame."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    url = f"{SUPABASE_URL}/rest/v1/market_data"
    headers = sb_headers_json()
    params = {'select': '*', 'timestamp': f'gte.{cutoff}', 'order': 'timestamp.asc'}

    all_rows, offset = [], 0
    while True:
        hdr = headers.copy()
        hdr['Range'] = f'{offset}-{offset+999}'
        hdr['Range-Unit'] = 'items'
        r = requests.get(url, headers=hdr, params=params, timeout=40)
        if not r.ok: break
        chunk = r.json()
        if not chunk: break
        all_rows.extend(chunk)
        if len(chunk) < 1000: break
        offset += 1000

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    # Coerce numeric
    for col in ['sell_median','buy_median','spread','sell_orders','buy_orders']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['timestamp','item']).sort_values('timestamp')
    return df

def rsi(series: pd.Series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def safe_mid(row):
    b = row.get('buy_median')
    s = row.get('sell_median')
    try:
        b = float(b) if b is not None else None
        s = float(s) if s is not None else None
    except Exception:
        return None
    if b and s: return (b + s) / 2.0
    return s if s else (b if b else None)

def exec_price(row, side):
    """Execution price with mode, slippage, and fee applied."""
    buy_m = row.get('buy_median')
    sell_m = row.get('sell_median')
    buy_m = float(buy_m) if buy_m is not None else None
    sell_m = float(sell_m) if sell_m is not None else None

    if EXECUTION_MODE == "realistic":
        base = sell_m if side == 'BUY' else buy_m
        if base is None:
            # fallback to mid if one side missing
            base = safe_mid(row)
    else:
        base = safe_mid(row)

    if not base or base <= 0:
        return 0.0

    # Slippage
    if SLIPPAGE_PCT > 0:
        base = base * (1 + SLIPPAGE_PCT) if side == 'BUY' else base * (1 - SLIPPAGE_PCT)

    # Fee
    if FEE_PCT > 0:
        base = base * (1 + FEE_PCT) if side == 'BUY' else base * (1 - FEE_PCT)

    return float(base)

class BacktestEngine:
    def __init__(self, initial_capital=INITIAL_CAPITAL):
        self.initial_capital = float(initial_capital)
        self.capital = float(initial_capital)
        self.positions = {}       # item -> {'quantity', 'avg_price', 'entry_time'}
        self.trades = []          # raw signal executions (for audit)
        self.realized_trades = [] # closed positions rows for profit_tracking
        self.equity_curve = []    # [{timestamp, equity, cash}]
        self.active_positions = set()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple MA crossover + RSI + volume surge. Requires sufficient history."""
        signals = []
        for item in df['item'].dropna().unique():
            item_df = df[df['item'] == item].copy()
            if len(item_df) < max(MIN_ITEM_POINTS, REQUIRED_HISTORY_STEPS):
                continue
            price = item_df['sell_median'].astype(float)
            item_df['ma_fast'] = price.rolling(MA_FAST, min_periods=MA_FAST).mean()
            item_df['ma_mid']  = price.rolling(MA_MID,  min_periods=MA_MID).mean()
            item_df['ma_slow'] = price.rolling(MA_SLOW, min_periods=MA_SLOW).mean()
            item_df['rsi'] = rsi(price, RSI_PERIOD)

            item_df['volume'] = (item_df['buy_orders'].fillna(0) + item_df['sell_orders'].fillna(0)).astype(float)
            item_df['volume_ma'] = item_df['volume'].rolling(VOL_MA, min_periods=VOL_MA).mean()

            # iterate starting at index where all indicators valid
            start_idx = max(MA_SLOW, VOL_MA, RSI_PERIOD)
            if len(item_df) <= start_idx:
                continue

            # Reset position state per item
            in_pos = False

            for i in range(start_idx, len(item_df)):
                row = item_df.iloc[i]
                prev = item_df.iloc[i-1]
                if any(pd.isna([row['ma_fast'], row['ma_mid'], row['ma_slow'], row['rsi'], row['volume_ma']])):
                    continue

                buy_score = 0
                # Golden cross fast over mid
                if prev['ma_fast'] <= prev['ma_mid'] and row['ma_fast'] > row['ma_mid']:
                    buy_score += 1
                # Uptrend vs slow
                if row['sell_median'] > row['ma_slow']:
                    buy_score += 1
                # RSI recovery
                if 30 < row['rsi'] < 50 and row['rsi'] > prev['rsi']:
                    buy_score += 1
                # Volume surge
                if row['volume_ma'] and row['volume'] > row['volume_ma'] * VOL_SURGE:
                    buy_score += 1

                # BUY
                if buy_score >= 3 and not in_pos:
                    signals.append({
                        'timestamp': row['timestamp'],
                        'item': item,
                        'type': 'BUY',
                        'price': row['sell_median'],  # informational; execution decided later
                        'reason': f'confirmations={buy_score}'
                    })
                    in_pos = True

                # SELL if in position
                if in_pos:
                    sell_score = 0
                    # Death cross
                    if prev['ma_fast'] >= prev['ma_mid'] and row['ma_fast'] < row['ma_mid']:
                        sell_score += 2
                    # Overbought exit
                    if row['rsi'] > 70:
                        sell_score += 1
                    # Protective stop vs slow MA
                    if row['sell_median'] < row['ma_slow'] * 0.97:
                        sell_score += 3

                    if sell_score >= 2:
                        signals.append({
                            'timestamp': row['timestamp'],
                            'item': item,
                            'type': 'SELL',
                            'price': row['buy_median'],
                            'reason': f'exits={sell_score}'
                        })
                        in_pos = False

        return pd.DataFrame(signals)

    def execute_backtest(self, df: pd.DataFrame, signals: pd.DataFrame):
        """Replay signals on DF timeline; mark to market per timestamp."""
        if signals.empty:
            return

        # Build map timestamp->list of signals
        # Use ISO timestamps for robust dict keys
        sig_by_ts = {}
        for _, s in signals.iterrows():
            ts = pd.to_datetime(s['timestamp'], utc=True)
            key = ts.isoformat()
            sig_by_ts.setdefault(key, []).append(s.to_dict())

        for ts_val in df['timestamp'].dropna().unique():
            ts = pd.to_datetime(ts_val, utc=True)
            tick = df[df['timestamp'] == ts]
            # Handle signals at this ts
            for sig in sig_by_ts.get(ts.isoformat(), []):
                self.process_signal(sig, tick)

            # Mark to market
            self.equity_curve.append({
                'timestamp': ts,
                'equity': self.calculate_equity(tick),
                'cash': self.capital
            })

    def process_signal(self, sig, tick_df: pd.DataFrame):
        item = sig['item']
        row = tick_df[tick_df['item'] == item]
        if row.empty:
            return
        row = row.iloc[0]

        if sig['type'] == 'BUY' and item not in self.positions:
            price = exec_price(row, 'BUY')
            if price <= 0:
                return
            # Size
            pos_cap = min(self.capital * MAX_POSITION_PCT, MAX_POSITION_VALUE)
            qty = int(pos_cap // price)
            if qty <= 0:
                return
            cost = qty * price
            self.capital -= cost
            self.positions[item] = {'quantity': qty, 'avg_price': price, 'entry_time': sig['timestamp']}
            self.active_positions.add(item)
            self.trades.append({'timestamp': sig['timestamp'], 'item': item, 'type': 'BUY', 'quantity': qty, 'price': price, 'cost': cost, 'reason': sig.get('reason')})

        elif sig['type'] == 'SELL' and item in self.positions:
            pos = self.positions[item]
            price = exec_price(row, 'SELL')
            if price <= 0:
                return
            qty = pos['quantity']
            revenue = qty * price
            cost = qty * pos['avg_price']
            profit = revenue - cost
            hold_hours = None
            try:
                open_ts = pd.to_datetime(pos.get('entry_time'), utc=True)
                close_ts = pd.to_datetime(sig['timestamp'], utc=True)
                hold_hours = (close_ts - open_ts).total_seconds() / 3600.0
            except Exception:
                pass
            self.capital += revenue
            self.trades.append({'timestamp': sig['timestamp'], 'item': item, 'type': 'SELL', 'quantity': qty, 'price': price, 'revenue': revenue, 'profit': profit, 'profit_pct': (profit/cost*100) if cost>0 else 0, 'hold_time_hours': hold_hours, 'reason': sig.get('reason')})

            # realized trade for profit_tracking
            self.realized_trades.append({
                'item': item,
                'entry_price': float(pos['avg_price']),
                'exit_price': float(price),
                'quantity': int(qty),
                'profit': float(profit),
                'profit_percentage': float((profit/cost*100) if cost>0 else 0),
                'executed_at': pd.to_datetime(pos.get('entry_time'), utc=True).isoformat() if pos.get('entry_time') else None,
                'closed_at': pd.to_datetime(sig['timestamp'], utc=True).isoformat() if sig.get('timestamp') else None,
                'strategy': STRATEGY_NAME
            })

            del self.positions[item]
            self.active_positions.discard(item)

    def calculate_equity(self, tick_df: pd.DataFrame) -> float:
        equity = self.capital
        for item, pos in self.positions.items():
            row = tick_df[tick_df['item'] == item]
            if row.empty:
                continue
            mark = safe_mid(row.iloc[0]) if EXECUTION_MODE == "mid" else safe_mid(row.iloc[0])
            # Use mid for mark-to-market to avoid systematic spread bias
            if mark and mark > 0:
                equity += pos['quantity'] * float(mark)
        return float(equity)

    def calculate_metrics(self):
        if not self.equity_curve:
            return {}
        eq = pd.DataFrame(self.equity_curve)
        eq = eq.dropna(subset=['equity'])
        if eq.empty:
            return {}
        final_equity = float(eq['equity'].iloc[-1])

        rets = eq['equity'].pct_change().dropna()
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        if len(rets) > 1 and rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
            cum = (1 + rets).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            max_dd = float(dd.min())
        else:
            sharpe = 0.0
            max_dd = 0.0

        closed = [t for t in self.trades if t['type'] == 'SELL']
        wins = [t for t in closed if t.get('profit', 0) > 0]
        losses = [t for t in closed if t.get('profit', 0) <= 0]
        win_rate = (len(wins) / len(closed)) if closed else 0

        avg_win = float(np.mean([t['profit'] for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([t['profit'] for t in losses])) if losses else 0.0
        tot_wins = float(sum([t['profit'] for t in wins])) if wins else 0.0
        tot_losses = abs(float(sum([t['profit'] for t in losses]))) if losses else 0.0
        profit_factor = float((tot_wins / tot_losses) if tot_losses > 0 else 0.0)
        avg_hold = float(np.mean([t.get('hold_time_hours', 0) for t in closed])) if closed else 0.0

        return {
            'initial_capital': float(self.initial_capital),
            'final_capital': final_equity,
            'total_return': float(total_return) * 100.0,
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd) * 100.0,
            'total_trades': len(self.trades),
            'completed_trades': len(closed),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': float(win_rate) * 100.0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'best_trade': float(max([t.get('profit', 0) for t in closed])) if closed else 0.0,
            'worst_trade': float(min([t.get('profit', 0) for t in closed])) if closed else 0.0,
            'avg_hold_time': avg_hold
        }

    def save_results(self):
        metrics = self.calculate_metrics()
        if not metrics:
            print("No metrics to save")
            return metrics

        os.makedirs('output', exist_ok=True)
        with open('output/backtest_results.json', 'w') as f:
            json.dump({
                'strategy': STRATEGY_NAME,
                'execution_mode': EXECUTION_MODE,
                'slippage_pct': SLIPPAGE_PCT,
                'fee_pct': FEE_PCT,
                'run_date': datetime.now(timezone.utc).isoformat(),
                'metrics': metrics
            }, f, indent=2, default=str)
        print("Detailed results saved to output/backtest_results.json")

        # Write summary row to backtest_results
        st_date = None
        en_date = None
        if self.equity_curve:
            st_date = pd.to_datetime(self.equity_curve[0]['timestamp'], utc=True).date().isoformat()
            en_date = pd.to_datetime(self.equity_curve[-1]['timestamp'], utc=True).date().isoformat()

        payload = {
            'strategy_name': STRATEGY_NAME,
            'start_date': st_date or datetime.now(timezone.utc).date().isoformat(),
            'end_date': en_date or datetime.now(timezone.utc).date().isoformat(),
            'initial_capital': metrics['initial_capital'],
            'final_capital': metrics['final_capital'],
            'total_return': metrics['total_return'],
            'win_rate': metrics['win_rate'],
            'total_trades': metrics['completed_trades'],
            'max_drawdown': metrics['max_drawdown'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'metadata': {
                'execution_mode': EXECUTION_MODE,
                'slippage_pct': SLIPPAGE_PCT,
                'fee_pct': FEE_PCT,
                'avg_win': metrics['avg_win'],
                'avg_loss': metrics['avg_loss'],
                'profit_factor': metrics['profit_factor'],
                'best_trade': metrics['best_trade'],
                'worst_trade': metrics['worst_trade'],
                'avg_hold_time_hours': metrics['avg_hold_time']
            }
        }
        try:
            r = requests.post(f"{SUPABASE_URL}/rest/v1/backtest_results", headers=sb_headers_json(), data=json.dumps(payload), timeout=40)
            if r.ok:
                print("Backtest results saved to database")
            else:
                print(f"Failed to save results: {r.status_code} - {r.text}")
        except Exception as e:
            print(f"Save summary error: {e}")

        # Bulk insert realized trades into profit_tracking (optional but useful)
        if self.realized_trades:
            try:
                rows = [{
                    'signal_id': None,
                    'item': rt['item'],
                    'entry_price': rt['entry_price'],
                    'exit_price': rt['exit_price'],
                    'quantity': rt['quantity'],
                    'profit': rt['profit'],
                    'profit_percentage': rt['profit_percentage'],
                    'executed_at': rt['executed_at'],
                    'closed_at': rt['closed_at'],
                    'strategy': rt['strategy']
                } for rt in self.realized_trades]
                r = requests.post(f"{SUPABASE_URL}/rest/v1/profit_tracking",
                                  headers=sb_headers_json(),
                                  data=json.dumps(rows),
                                  timeout=60)
                if r.ok:
                    print(f"Wrote {len(rows)} realized trades to profit_tracking")
                else:
                    print(f"Failed to store profit_tracking: {r.status_code} - {r.text}")
            except Exception as e:
                print(f"Save trades error: {e}")

        return metrics

def main():
    require_env()
    print("Starting enhanced backtest...")
    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)

    print("Fetching historical data...")
    df = fetch_market_data(days=LOOKBACK_DAYS)
    if df.empty:
        print("No historical data available")
        return
    print(f"Loaded {len(df)} data points from {df['timestamp'].min()} to {df['timestamp'].max()} (items={df['item'].nunique()})")

    # Keep only items that meet a minimum points gate so signals have enough context
    counts = df.groupby('item')['timestamp'].count()
    items_ok = set(counts[counts >= max(MIN_ITEM_POINTS, REQUIRED_HISTORY_STEPS)].index)
    if not items_ok:
        print("No items meet the required history steps. Exiting.")
        return
    df = df[df['item'].isin(items_ok)].copy()

    print("Generating signals...")
    signals = engine.generate_signals(df)
    print(f"Generated {len(signals)} signals")
    if signals.empty:
        print("No trading signals generated")
        return

    print("Running backtest...")
    engine.execute_backtest(df, signals)

    print("Results:")
    metrics = engine.calculate_metrics()
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    engine.save_results()

if __name__ == "__main__":
    main()