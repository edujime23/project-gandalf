import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

class BacktestEngine:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
    def fetch_historical_data(self, days=30):
        """Fetch historical market data"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        url = f"{SUPABASE_URL}/rest/v1/market_data"
        headers = {
            'apikey': SUPABASE_KEY,
            'Authorization': f'Bearer {SUPABASE_KEY}'
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
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df.sort_values('timestamp')
        
    def generate_signals(self, data):
        """Generate trading signals based on patterns"""
        signals = []
        
        # Group by item
        for item in data['item'].unique():
            item_data = data[data['item'] == item].copy()
            
            if len(item_data) < 20:
                continue
                
            # Calculate indicators
            item_data['ma_5'] = item_data['sell_median'].rolling(5).mean()
            item_data['ma_20'] = item_data['sell_median'].rolling(20).mean()
            item_data['rsi'] = self.calculate_rsi(item_data['sell_median'])
            
            # Generate signals
            for i in range(20, len(item_data)):
                row = item_data.iloc[i]
                prev_row = item_data.iloc[i-1]
                
                # Golden cross - BUY signal
                if prev_row['ma_5'] <= prev_row['ma_20'] and row['ma_5'] > row['ma_20']:
                    signals.append({
                        'timestamp': row['timestamp'],
                        'item': item,
                        'type': 'BUY',
                        'price': row['sell_median'],
                        'reason': 'Golden Cross'
                    })
                
                # Death cross - SELL signal
                elif prev_row['ma_5'] >= prev_row['ma_20'] and row['ma_5'] < row['ma_20']:
                    signals.append({
                        'timestamp': row['timestamp'],
                        'item': item,
                        'type': 'SELL',
                        'price': row['buy_median'],
                        'reason': 'Death Cross'
                    })
                
                # RSI oversold - BUY
                elif row['rsi'] < 30:
                    signals.append({
                        'timestamp': row['timestamp'],
                        'item': item,
                        'type': 'BUY',
                        'price': row['sell_median'],
                        'reason': 'RSI Oversold'
                    })
                
                # RSI overbought - SELL
                elif row['rsi'] > 70:
                    signals.append({
                        'timestamp': row['timestamp'],
                        'item': item,
                        'type': 'SELL',
                        'price': row['buy_median'],
                        'reason': 'RSI Overbought'
                    })
                    
        return pd.DataFrame(signals)
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def execute_backtest(self, data, signals):
        """Run the backtest"""
        print(f"Starting backtest with {self.initial_capital} platinum")
        
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
                'equity': equity
            })
            
    def process_signal(self, signal, market_data):
        """Process a trading signal"""
        item = signal['item']
        
        if signal['type'] == 'BUY':
            # Check if we have capital
            position_size = min(self.capital * 0.1, 1000)  # Max 10% per trade
            
            if self.capital >= position_size:
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
                            'avg_price': new_avg
                        }
                    else:
                        self.positions[item] = {
                            'quantity': quantity,
                            'avg_price': signal['price']
                        }
                        
                    self.trades.append({
                        'timestamp': signal['timestamp'],
                        'item': item,
                        'type': 'BUY',
                        'quantity': quantity,
                        'price': signal['price'],
                        'reason': signal['reason']
                    })
                    
        elif signal['type'] == 'SELL' and item in self.positions:
            # Sell position
            position = self.positions[item]
            revenue = position['quantity'] * signal['price']
            cost = position['quantity'] * position['avg_price']
            profit = revenue - cost
            
            self.capital += revenue
            
            self.trades.append({
                'timestamp': signal['timestamp'],
                'item': item,
                'type': 'SELL',
                'quantity': position['quantity'],
                'price': signal['price'],
                'profit': profit,
                'profit_pct': (profit / cost) * 100,
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
                equity += position['quantity'] * current_price
                
        return equity
        
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.equity_curve:
            return {}
            
        equity_df = pd.DataFrame(self.equity_curve)
        final_equity = equity_df['equity'].iloc[-1]
        
        # Calculate returns
        returns = equity_df['equity'].pct_change().dropna()
        
        # Calculate metrics
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = [t for t in self.trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('profit', 0) <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_equity,
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown * 100,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate * 100,
            'best_trade': max([t.get('profit', 0) for t in self.trades]) if self.trades else 0,
            'worst_trade': min([t.get('profit', 0) for t in self.trades]) if self.trades else 0
        }
        
    def save_results(self, strategy_name='MA_Crossover'):
        """Save backtest results to database"""
        metrics = self.calculate_metrics()
        
        if not metrics:
            print("No metrics to save")
            return
            
        # Prepare data for database
        result = {
            'strategy_name': strategy_name,
            'start_date': self.equity_curve[0]['timestamp'].date().isoformat(),
            'end_date': self.equity_curve[-1]['timestamp'].date().isoformat(),
            'initial_capital': metrics['initial_capital'],
            'final_capital': metrics['final_capital'],
            'total_return': metrics['total_return'],
            'win_rate': metrics['win_rate'],
            'total_trades': metrics['total_trades'],
            'max_drawdown': metrics['max_drawdown'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'metadata': {
                'winning_trades': metrics['winning_trades'],
                'losing_trades': metrics['losing_trades'],
                'best_trade': metrics['best_trade'],
                'worst_trade': metrics['worst_trade']
            }
        }
        
        # Store in database
        headers = {
            'apikey': SUPABASE_KEY,
            'Authorization': f'Bearer {SUPABASE_KEY}',
            'Content-Type': 'application/json'
        }
        
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/backtest_results",
            headers=headers,
            json=result
        )
        
        if resp.ok:
            print("Backtest results saved to database")
        else:
            print(f"Failed to save results: {resp.status_code}")
            
        return metrics

def main():
    print("Starting backtest...")
    
    engine = BacktestEngine(initial_capital=10000)
    
    # Fetch historical data
    print("Fetching historical data...")
    data = engine.fetch_historical_data(days=30)
    print(f"Loaded {len(data)} data points")
    
    # Generate signals
    print("Generating trading signals...")
    signals = engine.generate_signals(data)
    print(f"Generated {len(signals)} signals")
    
    # Run backtest
    print("Running backtest...")
    engine.execute_backtest(data, signals)
    
    # Calculate and display metrics
    metrics = engine.calculate_metrics()
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Save results
    engine.save_results()

if __name__ == "__main__":
    main()