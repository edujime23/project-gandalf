#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest engine (dual-run):
- Runs both execution modes in one process: mid and realistic.
- mid: executes at mid price (avg of buy/sell medians); good for signal quality.
- realistic: BUY at sell_median, SELL at buy_median; includes slippage/fees.
- Uses only items with sufficient history to support long windows (e.g., lags up to 168).
- Single position per item with capped sizing.
- Writes summary to backtest_results and realized PnL to profit_tracking for each mode.
- Saves artifacts to output/backtest_results_<mode>.json
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

# Fallback per-mode params (you can override via env)
# These are only used if per-mode envs aren't provided.
FALLBACK_SLIPPAGE_PCT_MID = float(os.getenv("SLIPPAGE_PCT_MID", "0.0"))
FALLBACK_FEE_PCT_MID      = float(os.getenv("FEE_PCT_MID", "0.0"))
FALLBACK_SLIPPAGE_PCT_REAL= float(os.getenv("SLIPPAGE_PCT_REALISTIC", os.getenv("SLIPPAGE_PCT", "0.002")))
FALLBACK_FEE_PCT_REAL     = float(os.getenv("FEE_PCT_REALISTIC", os.getenv("FEE_PCT", "0.001")))

# Which modes to run in one go (default both)
EXEC_MODES = [m.strip().lower() for m in os.getenv("EXEC_MODES", "mid,realistic").split(",") if m.strip()]

# Position sizing
INITIAL_CAPITAL     = float(os.getenv("INITIAL_CAPITAL", "10000"))
MAX_POSITION_PCT    = float(os.getenv("MAX_POSITION_PCT", "0.05"))  # 5% of equity
MAX_POSITION_VALUE  = float(os.getenv("MAX_POSITION_VALUE", "2000"))

# Strategy params
MIN_ITEM_POINTS = int(os.getenv("MIN_ITEM_POINTS", "50"))  # minimum points per item to even consider
MA_FAST   = int(os.getenv("MA_FAST", "5"))
MA_MID    = int(os.getenv("MA_MID", "20"))
MA_SLOW   = int(os.getenv("MA_SLOW", "50"))
VOL_MA    = int(os.getenv("VOL_MA", "20"))
RSI_PERIOD= int(os.getenv("RSI_PERIOD", "14"))
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

class BacktestEngine:
    def __init__(self, *, exec_mode: str, slippage_pct: float, fee_pct: float, initial_capital: float):
        self.exec_mode = exec_mode
        self.slippage_pct = float(slippage_pct)
        self.fee_pct = float(fee_pct)
        self.initial_capital = float(initial_capital)
        self.capital = float(initial_capital)
        self.positions = {}       # item -> {'quantity', 'avg_price', 'entry_time'}
        self.trades = []          # raw executions (audit)
        self.realized_trades = [] # for profit_tracking
        self.equity_curve = []    # [{timestamp, equity, cash}]
        self.active_positions = set()

    def exec_price(self, row, side):
        """Execution price with mode, slippage, and fee applied."""
        buy_m = row.get('buy_median')
        sell_m = row.get('sell_median')
        buy_m = float(buy_m) if buy_m is not None else None
        sell_m = float(sell_m) if sell_m is not None else None

        if self.exec_mode == "realistic":
            base = sell_m if side == 'BUY' else buy_m
            if base is None:
                base = safe_mid(row)
        else:
            base = safe_mid(row)

        if not base or base <= 0:
            return 0.0

        # Slippage
        if self.slippage_pct > 0:
            base = base * (1 + self.slippage_pct) if side == 'BUY' else base * (1 - self.slippage_pct)

        # Fee
        if self.fee_pct > 0:
            base = base * (1 + self.fee_pct) if side == 'BUY' else base * (1 - self.fee_pct)

        return float(base)

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

            start_idx = max(MA_SLOW, VOL_MA, RSI_PERIOD)
            if len(item_df) <= start_idx:
                continue

            in_pos = False
            for i in range(start_idx, len(item_df)):
                row = item_df.iloc[i]
                prev = item_df.iloc[i-1]
                if any(pd.isna([row['ma_fast'], row['ma_mid'], row['ma_slow'], row['rsi'], row['volume_ma']])):
                    continue

                buy_score = 0
                if prev['ma_fast'] <= prev['ma_mid'] and row['ma_fast'] > row['ma_mid']:
                    buy_score += 1
                if row['sell_median'] > row['ma_slow']:
                    buy_score += 1
                if 30 < row['rsi'] < 50 and row['rsi'] > prev['rsi']:
                    buy_score += 1
                if row['volume_ma'] and row['volume'] > row['volume_ma'] * VOL_SURGE:
                    buy_score += 1

                if buy_score >= 3 and not in_pos:
                    signals.append({
                        'timestamp': row['timestamp'],
                        'item': item,
                        'type': 'BUY',
                        'price': row['sell_median'],  # informational; execution decided later
                        'reason': f'confirmations={buy_score}'
                    })
                    in_pos = True

                if in_pos:
                    sell_score = 0
                    if prev['ma_fast'] >= prev['ma_mid'] and row['ma_fast'] < row['ma_mid']:
                        sell_score += 2
                    if row['rsi'] > 70:
                        sell_score += 1
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

        sig_by_ts = {}
        for _, s in signals.iterrows():
            ts = pd.to_datetime(s['timestamp'], utc=True)
            sig_by_ts.setdefault(ts, []).append(s.to_dict())

        for ts in df['timestamp'].dropna().unique():
            ts = pd.to_datetime(ts, utc=True)
            tick = df[df['timestamp'] == ts]

            for sig in sig_by_ts.get(ts, []):
                self.process_signal(sig, tick)

            # Mark-to-market using mid to avoid spread bias in equity curve
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
            price = self.exec_price(row, 'BUY')
            if price <= 0:
                return
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
            price = self.exec_price(row, 'SELL')
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
            mark = safe_mid(row.iloc[0])  # Always mid for marking
            if mark and mark > 0:
                equity += pos['quantity'] * float(mark)
        return float(equity)

    def calculate_metrics(self):
        if not self.equity_curve:
            return {}
        eq = pd.DataFrame(self.equity_curve).dropna(subset=['equity'])
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

    def save_results(self, mode: str):
        metrics = self.calculate_metrics()
        if not metrics:
            print(f"[{mode}] No metrics to save")
            return metrics

        os.makedirs('output', exist_ok=True)
        with open(f'output/backtest_results_{mode}.json', 'w') as f:
            json.dump({
                'strategy': STRATEGY_NAME,
                'execution_mode': mode,
                'slippage_pct': self.slippage_pct,
                'fee_pct': self.fee_pct,
                'run_date': datetime.now(timezone.utc).isoformat(),
                'metrics': metrics
            }, f, indent=2, default=str)
        print(f"[{mode}] Saved output/backtest_results_{mode}.json")

        # Summary to backtest_results
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
                'execution_mode': mode,
                'slippage_pct': self.slippage_pct,
                'fee_pct': self.fee_pct,
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
                print(f"[{mode}] Backtest results saved to database")
            else:
                print(f"[{mode}] Failed to save results: {r.status_code} - {r.text}")
        except Exception as e:
            print(f"[{mode}] Save summary error: {e}")

        # Bulk insert realized trades into profit_tracking
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
                    print(f"[{mode}] Wrote {len(rows)} realized trades to profit_tracking")
                else:
                    print(f"[{mode}] Failed to store profit_tracking: {r.status_code} - {r.text}")
            except Exception as e:
                print(f"[{mode}] Save trades error: {e}")

        return metrics

def get_mode_params(mode: str) -> tuple[float, float]:
    mode = mode.lower()
    if mode == "realistic":
        sl = float(os.getenv("SLIPPAGE_PCT_REALISTIC", str(FALLBACK_SLIPPAGE_PCT_REAL)))
        fe = float(os.getenv("FEE_PCT_REALISTIC", str(FALLBACK_FEE_PCT_REAL)))
        return sl, fe
    # mid
    sl = float(os.getenv("SLIPPAGE_PCT_MID", str(FALLBACK_SLIPPAGE_PCT_MID)))
    fe = float(os.getenv("FEE_PCT_MID", str(FALLBACK_FEE_PCT_MID)))
    return sl, fe

def main():
    require_env()
    print("Starting dual-mode backtest...")
    df = fetch_market_data(days=LOOKBACK_DAYS)
    if df.empty:
        print("No historical data available")
        return
    print(f"Loaded {len(df)} data points from {df['timestamp'].min()} to {df['timestamp'].max()} (items={df['item'].nunique()})")

    # Gate items for sufficient history so signals have context
    counts = df.groupby('item')['timestamp'].count()
    items_ok = set(counts[counts >= max(MIN_ITEM_POINTS, REQUIRED_HISTORY_STEPS)].index)
    if not items_ok:
        print("No items meet the required history steps. Exiting.")
        return
    df = df[df['item'].isin(items_ok)].copy()

    # Generate signals once; replay under each execution mode
    # Signals are independent of execution assumptions.
    tmp_engine_for_signals = BacktestEngine(exec_mode="mid", slippage_pct=0.0, fee_pct=0.0, initial_capital=INITIAL_CAPITAL)
    signals = tmp_engine_for_signals.generate_signals(df)
    print(f"Signals generated (shared by runs): {len(signals)}")
    if signals.empty:
        print("No trading signals generated. Exiting.")
        return

    for mode in EXEC_MODES:
        if mode not in {"mid", "realistic"}:
            print(f"Skipping unknown mode: {mode}")
            continue
        sl, fe = get_mode_params(mode)
        print(f"Running mode={mode} slippage={sl} fee={fe}")
        engine = BacktestEngine(exec_mode=mode, slippage_pct=sl, fee_pct=fe, initial_capital=INITIAL_CAPITAL)

        engine.execute_backtest(df, signals)

        metrics = engine.calculate_metrics()
        print(f"[{mode}] Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        engine.save_results(mode)

    print("Dual-mode backtest done.")

if __name__ == "__main__":
    main()