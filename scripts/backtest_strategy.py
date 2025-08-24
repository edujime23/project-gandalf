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
        self.active_positions = set()  # Track which items we own
        
    def fetch_historical_data(self, days=30):
        """Fetch historical market data"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        url = f"{SUPABASE_URL}/rest/v1/market_data"
        headers = {
            'apikey': SUPABASE_SERVICE_ROLE,
            'Authorization': f'Bearer {SUPABASE_SERVICE_ROLE}'
        }
        params = {
            'select': '*',
            'timestamp': f'gte.{cutoff}',
            'order': 'timestamp.asc'
        }
        
        all_data = []
        offset = 0
        
        while True:
            headers['Range'] = f"{offset}-{offset+999}"
            headers['Range-Unit'] = 'items'
            
            resp = requests.get(url, headers=headers, params=params)
            if not resp.ok:
                break
                
            data = resp.json()
            if not data:
                break
                
            all_data.extend(data)
            if len(data) < 1000:
                break
            offset += 1000
            
        df = pd.DataFrame(all_data)
        
        # Handle timestamp parsing
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        except:
            def parse_timestamp(ts):
                for fmt in ['%Y-%m-%dT%H:%M:%S.%f%z', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ']:
                    try:
                        return datetime.strptime(ts.replace('+00:00', 'Z'), fmt)
                    except:
                        continue
                return pd.to_datetime(ts)
            
            df['timestamp'] = df['timestamp'].apply(parse_timestamp)
        
        # Convert numeric columns
        numeric_cols = ['sell_median', 'buy_median', 'spread', 'sell_orders', 'buy_orders']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.sort_values('timestamp')
        
    def generate_signals(self, data):
        """Enhanced trading signals with better filters"""
        signals = []
        
        for item in data['item'].unique():
            item_data = data[data['item'] == item].copy()
            
            if len(item_data) < 50:  # Need more data for reliable signals
                continue
                
            # Calculate indicators
            item_data['ma_5'] = item_data['sell_median'].rolling(5).mean()
            item_data['ma_20'] = item_data['sell_median'].rolling(20).mean()
            item_data['ma_50'] = item_data['sell_median'].rolling(50).mean()
            item_data['rsi'] = self.calculate_rsi(item_data['sell_median'])
            
            # Bollinger Bands
            item_data['bb_middle'] = item_data['sell_median'].rolling(20).mean()
            item_data['bb_std'] = item_data['sell_median'].rolling(20).std()
            item_data['bb_upper'] = item_data['bb_middle'] + (2 * item_data['bb_std'])
            item_data['bb_lower'] = item_data['bb_middle'] - (2 * item_data['bb_std'])
            
            # Volatility filter
            item_data['volatility'] = item_data['sell_median'].rolling(20).std()
            item_data['avg_volatility'] = item_data['volatility'].rolling(50).mean()
            
            # Volume filter
            item_data['volume'] = item_data['buy_orders'].fillna(0) + item_data['sell_orders'].fillna(0)
            item_data['volume_ma'] = item_data['volume'].rolling(20).mean()
            
            # MACD
            exp1 = item_data['sell_median'].ewm(span=12, adjust=False).mean()
            exp2 = item_data['sell_median'].ewm(span=26, adjust=False).mean()
            item_data['macd'] = exp1 - exp2
            item_data['signal_line'] = item_data['macd'].ewm(span=9, adjust=False).mean()
            
            for i in range(50, len(item_data)):
                row = item_data.iloc[i]
                prev_row = item_data.iloc[i-1]
                
                # Skip if any critical indicator is NaN
                if any(pd.isna([row['ma_5'], row['ma_20'], row['ma_50'], row['rsi'], row['volume_ma']])):
                    continue
                
                # ENHANCED BUY CONDITIONS:
                # Multiple confirmations required
                buy_signals = 0
                
                # 1. Golden cross
                if prev_row['ma_5'] <= prev_row['ma_20'] and row['ma_5'] > row['ma_20']:
                    buy_signals += 1
                
                # 2. Price above MA50 (uptrend)
                if row['sell_median'] > row['ma_50']:
                    buy_signals += 1
                
                # 3. RSI oversold but recovering
                if 30 < row['rsi'] < 50 and row['rsi'] > prev_row['rsi']:
                    buy_signals += 1
                
                # 4. Volume surge
                if row['volume'] > row['volume_ma'] * 1.5:
                    buy_signals += 1
                
                # 5. Price near Bollinger Band lower
                if row['sell_median'] <= row['bb_lower'] * 1.02:
                    buy_signals += 1
                
                # 6. MACD bullish crossover
                if prev_row['macd'] <= prev_row['signal_line'] and row['macd'] > row['signal_line']:
                    buy_signals += 1
                
                # Need at least 3 confirmations to buy
                if buy_signals >= 3 and item not in self.active_positions:
                    signals.append({
                        'timestamp': row['timestamp'],
                        'item': item,
                        'type': 'BUY',
                        'price': row['sell_median'],
                        'reason': f'Multiple confirmations ({buy_signals}/6)',
                        'indicators': {
                            'rsi': row['rsi'],
                            'volume_ratio': row['volume'] / row['volume_ma'] if row['volume_ma'] > 0 else 1
                        }
                    })
                    self.active_positions.add(item)
                
                # ENHANCED SELL CONDITIONS:
                sell_signals = 0
                
                # Only check sell conditions if we have a position
                if item in self.active_positions:
                    # 1. Death cross
                    if prev_row['ma_5'] >= prev_row['ma_20'] and row['ma_5'] < row['ma_20']:
                        sell_signals += 2  # Strong signal
                    
                    # 2. RSI overbought
                    if row['rsi'] > 70:
                        sell_signals += 1
                    
                    # 3. Price at Bollinger Band upper
                    if row['sell_median'] >= row['bb_upper'] * 0.98:
                        sell_signals += 1
                    
                    # 4. MACD bearish crossover
                    if prev_row['macd'] >= prev_row['signal_line'] and row['macd'] < row['signal_line']:
                        sell_signals += 1
                    
                    # 5. Stop loss - price below MA50
                    if row['sell_median'] < row['ma_50'] * 0.97:  # 3% stop loss
                        sell_signals += 3  # Force sell
                    
                    # 6. Take profit - 15% gain
                    if item in self.positions:
                        entry_price = self.positions[item]['avg_price']
                        if row['sell_median'] > entry_price * 1.15:
                            sell_signals += 2
                    
                    # Sell if we have enough signals
                    if sell_signals >= 2:
                        signals.append({
                            'timestamp': row['timestamp'],
                            'item': item,
                            'type': 'SELL',
                            'price': row['buy_median'],
                            'reason': f'Exit signals ({sell_signals})',
                            'indicators': {
                                'rsi': row['rsi'],
                                'price_position': 'upper' if row['sell_median'] >= row['bb_upper'] else 'normal'
                            }
                        })
                        self.active_positions.discard(item)
                    
        return pd.DataFrame(signals)
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def execute_backtest(self, data, signals):
        """Run the backtest with improved position management"""
        print(f"Starting backtest with {self.initial_capital} platinum")
        
        # Reset position tracking
        self.active_positions.clear()
        
        # Convert to dict for faster lookup
        signal_dict = {}
        for _, signal in signals.iterrows():
            key = (signal['timestamp'], signal['item'])
            signal_dict[key] = signal
            
        # Process each timestamp
        timestamps = data['timestamp'].unique()
        
        for timestamp in timestamps:
            current_data = data[data['timestamp'] == timestamp]
            
            # Check for signals
            for _, row in current_data.iterrows():
                key = (timestamp, row['item'])
                
                if key in signal_dict:
                    signal = signal_dict[key]
                    self.process_signal(signal, row)
                    
            # Update equity
            equity = self.calculate_equity(current_data)
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'cash': self.capital,
                'positions_value': equity - self.capital
            })
            
    def process_signal(self, signal, market_data):
        """Process a trading signal with improved money management"""
        item = signal['item']
        
        if signal['type'] == 'BUY':
            # Dynamic position sizing based on confidence
            base_position_size = self.capital * 0.05  # 5% base position
            
            # Adjust based on signal strength (if available)
            if 'indicators' in signal and 'rsi' in signal['indicators']:
                if signal['indicators']['rsi'] < 30:  # Very oversold
                    base_position_size *= 1.5
                    
            position_size = min(base_position_size, self.capital * 0.1, 2000)  # Max 10% or 2000p
            
            if self.capital >= position_size and signal['price'] > 0:
                quantity = int(position_size / signal['price'])
                
                if quantity > 0:
                    cost = quantity * signal['price']
                    self.capital -= cost
                    
                    if item in self.positions:
                        # Average down
                        old_qty = self.positions[item]['quantity']
                        old_avg = self.positions[item]['avg_price']
                        new_qty = old_qty + quantity
                        new_avg = (old_qty * old_avg + cost) / new_qty
                        
                        self.positions[item] = {
                            'quantity': new_qty,
                            'avg_price': new_avg,
                            'entry_time': signal['timestamp']
                        }
                    else:
                        self.positions[item] = {
                            'quantity': quantity,
                            'avg_price': signal['price'],
                            'entry_time': signal['timestamp']
                        }
                        
                    self.trades.append({
                        'timestamp': signal['timestamp'],
                        'item': item,
                        'type': 'BUY',
                        'quantity': quantity,
                        'price': signal['price'],
                        'cost': cost,
                        'reason': signal['reason']
                    })
                    
        elif signal['type'] == 'SELL' and item in self.positions:
            # Sell position
            position = self.positions[item]
            revenue = position['quantity'] * signal['price']
            cost = position['quantity'] * position['avg_price']
            profit = revenue - cost
            hold_time = (signal['timestamp'] - position['entry_time']).total_seconds() / 3600  # hours
            
            self.capital += revenue
            
            self.trades.append({
                'timestamp': signal['timestamp'],
                'item': item,
                'type': 'SELL',
                'quantity': position['quantity'],
                'price': signal['price'],
                'revenue': revenue,
                'profit': profit,
                'profit_pct': (profit / cost) * 100 if cost > 0 else 0,
                'hold_time_hours': hold_time,
                'reason': signal['reason']
            })
            
            del self.positions[item]
            
    def calculate_equity(self, current_data):
        """Calculate current equity value"""
        equity = self.capital
        
        for item, position in self.positions.items():
            item_data = current_data[current_data['item'] == item]
            if not item_data.empty:
                current_price = item_data.iloc[0]['sell_median']
                if pd.notna(current_price):
                    equity += position['quantity'] * current_price
                
        return equity
        
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.equity_curve:
            return {}
            
        equity_df = pd.DataFrame(self.equity_curve)
        final_equity = equity_df['equity'].iloc[-1]
        
        # Calculate returns
        returns = equity_df['equity'].pct_change().dropna()
        
        # Calculate metrics
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        if len(returns) > 0:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
        
        # Win rate and trade analysis
        completed_trades = [t for t in self.trades if t['type'] == 'SELL']
        winning_trades = [t for t in completed_trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in completed_trades if t.get('profit', 0) <= 0]
        
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        
        # Average win/loss
        avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        total_wins = sum([t['profit'] for t in winning_trades]) if winning_trades else 0
        total_losses = abs(sum([t['profit'] for t in losing_trades])) if losing_trades else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_equity,
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown * 100,
            'total_trades': len(self.trades),
            'completed_trades': len(completed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'best_trade': max([t.get('profit', 0) for t in completed_trades]) if completed_trades else 0,
            'worst_trade': min([t.get('profit', 0) for t in completed_trades]) if completed_trades else 0,
            'avg_hold_time': np.mean([t.get('hold_time_hours', 0) for t in completed_trades]) if completed_trades else 0
        }
        
    def save_results(self, strategy_name='Enhanced_MA_Strategy'):
        """Save backtest results to database and file"""
        metrics = self.calculate_metrics()
        
        if not metrics:
            print("No metrics to save")
            return
            
        # Create output directory
        os.makedirs('output', exist_ok=True)
        
        # Save detailed results to file
        detailed_results = {
            'strategy': strategy_name,
            'run_date': datetime.now().isoformat(),
            'metrics': metrics,
            'configuration': {
                'initial_capital': self.initial_capital,
                'position_size': '5-10% dynamic',
                'stop_loss': '3% below MA50',
                'take_profit': '15% gain',
                'min_buy_signals': 3,
                'min_sell_signals': 2
            },
            'top_trades': sorted([t for t in self.trades if t['type'] == 'SELL'], 
                               key=lambda x: x.get('profit', 0), reverse=True)[:10],
            'worst_trades': sorted([t for t in self.trades if t['type'] == 'SELL'], 
                                 key=lambda x: x.get('profit', 0))[:10],
            'summary_stats': {
                'total_buy_trades': len([t for t in self.trades if t['type'] == 'BUY']),
                'total_sell_trades': len([t for t in self.trades if t['type'] == 'SELL']),
                'unique_items_traded': len(set([t['item'] for t in self.trades])),
                'avg_trade_size': np.mean([t.get('cost', t.get('revenue', 0)) for t in self.trades])
            }
        }
        
        with open('output/backtest_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print("Detailed results saved to output/backtest_results.json")
        
        # Prepare data for database
        result = {
            'strategy_name': strategy_name,
            'start_date': self.equity_curve[0]['timestamp'].date().isoformat() if self.equity_curve else datetime.now().date().isoformat(),
            'end_date': self.equity_curve[-1]['timestamp'].date().isoformat() if self.equity_curve else datetime.now().date().isoformat(),
            'initial_capital': metrics['initial_capital'],
            'final_capital': metrics['final_capital'],
            'total_return': metrics['total_return'],
            'win_rate': metrics['win_rate'],
            'total_trades': metrics['completed_trades'],  # Use completed trades for accuracy
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
        
        # Store in database
        headers = {
            'apikey': SUPABASE_SERVICE_ROLE,
            'Authorization': f'Bearer {SUPABASE_SERVICE_ROLE}',
            'Content-Type': 'application/json',
            'Prefer': 'return=minimal'
        }
        
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/backtest_results",
            headers=headers,
            json=result
        )
        
        if resp.ok:
            print("Backtest results saved to database")
        else:
            print(f"Failed to save results: {resp.status_code} - {resp.text}")
            
        return metrics

def main():
    print("Starting enhanced backtest...")
    
    engine = BacktestEngine(initial_capital=10000)
    
    # Fetch historical data
    print("Fetching historical data...")
    data = engine.fetch_historical_data(days=30)
    
    if data.empty:
        print("No historical data available")
        return
        
    print(f"Loaded {len(data)} data points")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"Items found: {data['item'].nunique()}")
    
    # Generate signals
    print("\nGenerating trading signals with enhanced strategy...")
    signals = engine.generate_signals(data)
    print(f"Generated {len(signals)} signals")
    
    if not signals.empty:
        print(f"Buy signals: {len(signals[signals['type'] == 'BUY'])}")
        print(f"Sell signals: {len(signals[signals['type'] == 'SELL'])}")
    
    if signals.empty:
        print("No trading signals generated - strategy may be too conservative")
        print("Consider adjusting signal thresholds")
        return
    
    # Run backtest
    print("\nRunning backtest...")
    engine.execute_backtest(data, signals)
    
    # Calculate and display metrics
    metrics = engine.calculate_metrics()
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS - ENHANCED STRATEGY")
    print("="*60)
    print(f"Initial Capital:    {metrics['initial_capital']:>10.2f}")
    print(f"Final Capital:      {metrics['final_capital']:>10.2f}")
    print(f"Total Return:       {metrics['total_return']:>10.2f}%")
    print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:       {metrics['max_drawdown']:>10.2f}%")
    print(f"Win Rate:           {metrics['win_rate']:>10.2f}%")
    print(f"Profit Factor:      {metrics.get('profit_factor', 0):>10.2f}")
    print("-"*60)
    print(f"Total Trades:       {metrics['total_trades']:>10}")
    print(f"Winning Trades:     {metrics['winning_trades']:>10}")
    print(f"Losing Trades:      {metrics['losing_trades']:>10}")
    print(f"Average Win:        {metrics.get('avg_win', 0):>10.2f}")
    print(f"Average Loss:       {metrics.get('avg_loss', 0):>10.2f}")
    print(f"Best Trade:         {metrics['best_trade']:>10.2f}")
    print(f"Worst Trade:        {metrics['worst_trade']:>10.2f}")
    print(f"Avg Hold Time:      {metrics.get('avg_hold_time', 0):>10.1f} hours")
    print("="*60)
    
    # Save results
    engine.save_results()

if __name__ == "__main__":
    main()