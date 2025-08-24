#!/usr/bin/env python
import pandas as pd
import numpy as np
import requests
import os
import json
from datetime import datetime, timedelta

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_ROLE = os.getenv('SUPABASE_SERVICE_ROLE')

class BacktestEngine:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.active_positions = set()

    def fetch_historical_data(self, days=30):
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        url = f"{SUPABASE_URL}/rest/v1/market_data"
        headers = {
            'apikey': SUPABASE_SERVICE_ROLE,
            'Authorization': f'Bearer {SUPABASE_SERVICE_ROLE}'
        }
        params = { 'select': '*', 'timestamp': f'gte.{cutoff}', 'order': 'timestamp.asc' }
        all_rows, offset = [], 0
        while True:
            hdr = headers | {'Range': f'{offset}-{offset+999}', 'Range-Unit': 'items'}
            r = requests.get(url, headers=hdr, params=params, timeout=30)
            if not r.ok:
                break
            chunk = r.json()
            if not chunk:
                break
            all_rows.extend(chunk)
            if len(chunk) < 1000:
                break
            offset += 1000

        df = pd.DataFrame(all_rows)
        if df.empty:
            return df
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        for col in ['sell_median','buy_median','spread','sell_orders','buy_orders']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=['timestamp']).sort_values('timestamp')

    def calculate_rsi(self, s, period=14):
        delta = s.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def generate_signals(self, data):
        signals = []
        for item in data['item'].unique():
            item_df = data[data['item'] == item].copy()
            if len(item_df) < 50:  # need enough context
                continue

            # Indicators
            price = item_df['sell_median']
            item_df['ma_5'] = price.rolling(5).mean()
            item_df['ma_20'] = price.rolling(20).mean()
            item_df['ma_50'] = price.rolling(50).mean()
            item_df['rsi'] = self.calculate_rsi(price)
            item_df['volume'] = item_df['buy_orders'].fillna(0) + item_df['sell_orders'].fillna(0)
            item_df['volume_ma'] = item_df['volume'].rolling(20).mean()

            for i in range(50, len(item_df)):
                row = item_df.iloc[i]
                prev = item_df.iloc[i-1]
                if any(pd.isna([row['ma_5'],row['ma_20'],row['ma_50'],row['rsi'],row['volume_ma']])):
                    continue

                buy_score = 0
                if prev['ma_5'] <= prev['ma_20'] and row['ma_5'] > row['ma_20']:
                    buy_score += 1  # golden cross
                if row['sell_median'] > row['ma_50']:
                    buy_score += 1  # uptrend
                if 30 < row['rsi'] < 50 and row['rsi'] > prev['rsi']:
                    buy_score += 1  # recovering oversold
                if row['volume'] > row['volume_ma'] * 1.5:
                    buy_score += 1  # volume surge

                if buy_score >= 3 and item not in self.active_positions:
                    signals.append({
                        'timestamp': row['timestamp'], 'item': item, 'type': 'BUY',
                        'price': row['sell_median'], 'reason': f'confirmations={buy_score}'
                    })
                    self.active_positions.add(item)

                if item in self.active_positions:
                    sell_score = 0
                    if prev['ma_5'] >= prev['ma_20'] and row['ma_5'] < row['ma_20']:
                        sell_score += 2  # death cross strong
                    if row['rsi'] > 70:
                        sell_score += 1
                    if row['sell_median'] < row['ma_50'] * 0.97:  # stop
                        sell_score += 3
                    if sell_score >= 2:
                        signals.append({
                            'timestamp': row['timestamp'], 'item': item, 'type': 'SELL',
                            'price': row['buy_median'], 'reason': f'exits={sell_score}'
                        })
                        self.active_positions.discard(item)
        return pd.DataFrame(signals)

    def execute_backtest(self, data, signals):
        self.active_positions.clear()
        sig_map = {(s['timestamp'], s['item']): s for _, s in signals.iterrows()}
        for ts in data['timestamp'].unique():
            tick = data[data['timestamp'] == ts]
            for _, row in tick.iterrows():
                key = (ts, row['item'])
                if key in sig_map:
                    self.process_signal(sig_map[key], row)
            self.equity_curve.append({
                'timestamp': ts, 'equity': self.calculate_equity(tick), 'cash': self.capital
            })

    def process_signal(self, sig, market_row):
        item = sig['item']
        if sig['type'] == 'BUY':
            pos_size = min(self.capital * 0.05, self.capital * 0.10, 2000)
            price = float(sig['price'] or 0)
            if pos_size > 0 and price > 0:
                qty = int(pos_size / price)
                if qty <= 0:
                    return
                cost = qty * price
                self.capital -= cost
                if item in self.positions:
                    old = self.positions[item]
                    new_qty = old['quantity'] + qty
                    new_avg = (old['quantity'] * old['avg_price'] + cost) / new_qty
                    self.positions[item] = {'quantity': new_qty, 'avg_price': new_avg, 'entry_time': sig['timestamp']}
                else:
                    self.positions[item] = {'quantity': qty, 'avg_price': price, 'entry_time': sig['timestamp']}
                self.trades.append({'timestamp': sig['timestamp'], 'item': item, 'type': 'BUY', 'quantity': qty, 'price': price, 'cost': cost, 'reason': sig.get('reason')})

        elif sig['type'] == 'SELL' and item in self.positions:
            pos = self.positions[item]
            price = float(sig['price'] or 0)
            if price <= 0:
                return
            revenue = pos['quantity'] * price
            cost = pos['quantity'] * pos['avg_price']
            profit = revenue - cost
            hold_hours = (sig['timestamp'] - pos['entry_time']).total_seconds() / 3600 if pos.get('entry_time') else None
            self.capital += revenue
            self.trades.append({'timestamp': sig['timestamp'], 'item': item, 'type': 'SELL', 'quantity': pos['quantity'], 'price': price, 'revenue': revenue, 'profit': profit, 'profit_pct': (profit/cost*100) if cost>0 else 0, 'hold_time_hours': hold_hours, 'reason': sig.get('reason')})
            del self.positions[item]

    def calculate_equity(self, tick_df):
        equity = self.capital
        for item, pos in self.positions.items():
            row = tick_df[tick_df['item'] == item]
            if not row.empty and pd.notna(row.iloc[0]['sell_median']):
                equity += pos['quantity'] * float(row.iloc[0]['sell_median'])
        return equity

    def calculate_metrics(self):
        if not self.equity_curve:
            return {}
        eq = pd.DataFrame(self.equity_curve)
        final_equity = float(eq['equity'].iloc[-1])

        rets = eq['equity'].pct_change().dropna()
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 0 and rets.std() > 0 else 0

        if len(rets) > 0:
            cum = (1 + rets).cumprod()
            drawdown = (cum - cum.cummax()) / cum.cummax()
            max_dd = drawdown.min()
        else:
            max_dd = 0

        closed = [t for t in self.trades if t['type'] == 'SELL']
        wins = [t for t in closed if t.get('profit', 0) > 0]
        losses = [t for t in closed if t.get('profit', 0) <= 0]
        win_rate = len(wins) / len(closed) if closed else 0
        avg_win = float(np.mean([t['profit'] for t in wins])) if wins else 0
        avg_loss = float(np.mean([t['profit'] for t in losses])) if losses else 0
        tot_wins = sum([t['profit'] for t in wins]) if wins else 0
        tot_losses = abs(sum([t['profit'] for t in losses])) if losses else 0
        profit_factor = (tot_wins / tot_losses) if tot_losses > 0 else 0

        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_equity,
            'total_return': total_return * 100,
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd) * 100,
            'total_trades': len(self.trades),
            'completed_trades': len(closed),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': float(profit_factor),
            'best_trade': max([t.get('profit', 0) for t in closed]) if closed else 0,
            'worst_trade': min([t.get('profit', 0) for t in closed]) if closed else 0,
            'avg_hold_time': float(np.mean([t.get('hold_time_hours', 0) for t in closed])) if closed else 0
        }

    def save_results(self, strategy_name='Enhanced_MA_Strategy'):
        metrics = self.calculate_metrics()
        if not metrics:
            print("No metrics to save")
            return metrics

        os.makedirs('output', exist_ok=True)
        with open('output/backtest_results.json', 'w') as f:
            json.dump({'strategy': strategy_name, 'run_date': datetime.now().isoformat(), 'metrics': metrics}, f, indent=2, default=str)
        print("Detailed results saved to output/backtest_results.json")

        headers = {
            'apikey': SUPABASE_SERVICE_ROLE,
            'Authorization': f'Bearer {SUPABASE_SERVICE_ROLE}',
            'Content-Type': 'application/json',
            'Prefer': 'return=minimal'
        }
        payload = {
            'strategy_name': strategy_name,
            'start_date': self.equity_curve[0]['timestamp'].date().isoformat() if self.equity_curve else datetime.now().date().isoformat(),
            'end_date': self.equity_curve[-1]['timestamp'].date().isoformat() if self.equity_curve else datetime.now().date().isoformat(),
            'initial_capital': metrics['initial_capital'],
            'final_capital': metrics['final_capital'],
            'total_return': metrics['total_return'],
            'win_rate': metrics['win_rate'],
            'total_trades': metrics['completed_trades'],
            'max_drawdown': metrics['max_drawdown'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'metadata': {
                'winning_trades': metrics['winning_trades'],
                'losing_trades': metrics['losing_trades'],
                'best_trade': metrics['best_trade'],
                'worst_trade': metrics['worst_trade'],
                'profit_factor': metrics['profit_factor'],
                'avg_win': metrics['avg_win'],
                'avg_loss': metrics['avg_loss'],
                'avg_hold_time_hours': metrics['avg_hold_time']
            }
        }
        try:
            r = requests.post(f"{SUPABASE_URL}/rest/v1/backtest_results", headers=headers, data=json.dumps(payload), timeout=30)
            if r.ok:
                print("Backtest results saved to database")
            else:
                print(f"Failed to save results: {r.status_code} - {r.text}")
        except Exception as e:
            print(f"Save error: {e}")
        return metrics

def main():
    print("Starting enhanced backtest...")
    engine = BacktestEngine(initial_capital=10000)

    print("Fetching historical data...")
    data = engine.fetch_historical_data(days=30)
    if data.empty:
        print("No historical data available")
        return
    print(f"Loaded {len(data)} data points from {data['timestamp'].min()} to {data['timestamp'].max()} (items={data['item'].nunique()})")

    print("Generating signals...")
    signals = engine.generate_signals(data)
    print(f"Generated {len(signals)} signals")

    if signals.empty:
        print("No trading signals generated")
        return

    print("Running backtest...")
    engine.execute_backtest(data, signals)

    print("Results:")
    metrics = engine.calculate_metrics()
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    engine.save_results()

if __name__ == "__main__":
    main()