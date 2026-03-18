"""VWAP Scalping Backtest Engine — 1 Week, $1000"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime
from backtest.data_fetcher import DataFetcher

print("=" * 70)
print("  VWAP Scalping — 1 Week Backtest Engine ($1000)")
print("=" * 70)

# Fetch 1m data (7 days = ~10080 bars)
fetcher = DataFetcher()
print("Fetching 7 days of 1m data...")

df = fetcher.fetch_ohlcv('TRX/USDT', '1m', limit=10000)
if df is None or len(df) < 1000:
    print("ERROR: Not enough 1m data, using last available...")
    df = fetcher.fetch_ohlcv('TRX/USDT', '1m', limit=2000)

print(f"Data: {len(df)} 1m bars | {df.index[0]} → {df.index[-1]}")

# VWAP calculation (intraday)
df['tp'] = (df['high'] + df['low'] + df['close']) / 3
df['tp_volume'] = df['tp'] * df['volume']
df['date'] = df.index.date
df['cum_tp_volume'] = df.groupby('date')['tp_volume'].cumsum()
df['cum_volume'] = df.groupby('date')['volume'].cumsum()
df['vwap'] = df['cum_tp_volume'] / df['cum_volume']

# Delta
df['body'] = df['close'] - df['open']
df['body_pct'] = df['body'] / df['open']
df['buy_volume'] = np.where(df['body'] > 0, df['volume'] * abs(df['body_pct']), 0)
df['sell_volume'] = np.where(df['body'] < 0, df['volume'] * abs(df['body_pct']), 0)
df['delta_ratio'] = df['buy_volume'].rolling(5).sum() / (df['sell_volume'].rolling(5).sum() + 1e-9)
df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
df['atr'] = df['close'].rolling(14).apply(lambda x: max(x) - min(x), raw=True)

# Backtest function
def run_backtest(df, params, initial_capital=1000):
    """Run VWAP scalping backtest with given parameters."""
    capital = initial_capital
    position = None
    trades = []
    cooldown = 0
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        close = float(row['close'])
        high = float(row['high'])
        low = float(row['low'])
        vwap = float(row['vwap'])
        delta = float(row.get('delta_ratio', 1.0))
        vol_ratio = float(row.get('volume_ratio', 1.0))
        atr = float(row.get('atr', close * 0.01))
        
        # Exit check
        if position:
            side = position['side']
            entry = position['entry']
            sl = position['sl']
            tp = position['tp']
            qty = position['qty']
            
            # TP hit
            if side == 'LONG' and high >= tp:
                pnl = (tp - entry) * qty
                capital += pnl
                trades.append({'side': side, 'pnl': pnl, 'reason': 'TP', 'entry': entry, 'exit': tp})
                position = None
                cooldown = i + params['cooldown_bars']
            elif side == 'SHORT' and low <= tp:
                pnl = (entry - tp) * qty
                capital += pnl
                trades.append({'side': side, 'pnl': pnl, 'reason': 'TP', 'entry': entry, 'exit': tp})
                position = None
                cooldown = i + params['cooldown_bars']
            # SL hit
            elif side == 'LONG' and low <= sl:
                pnl = (sl - entry) * qty
                capital += pnl
                trades.append({'side': side, 'pnl': pnl, 'reason': 'SL', 'entry': entry, 'exit': sl})
                position = None
                cooldown = i + params['cooldown_bars']
            elif side == 'SHORT' and high >= sl:
                pnl = (entry - sl) * qty
                capital += pnl
                trades.append({'side': side, 'pnl': pnl, 'reason': 'SL', 'entry': entry, 'exit': sl})
                position = None
                cooldown = i + params['cooldown_bars']
            # Timeout
            elif i - position['entry_idx'] >= params['timeout_bars']:
                pnl = (close - entry) * qty if side == 'LONG' else (entry - close) * qty
                capital += pnl
                trades.append({'side': side, 'pnl': pnl, 'reason': 'TIMEOUT', 'entry': entry, 'exit': close})
                position = None
                cooldown = i + params['cooldown_bars']
        
        # Entry check
        if not position and i >= cooldown:
            dist_pct = abs(close - vwap) / vwap * 100
            atr_pct = atr / close
            
            # Filters
            if dist_pct > params['max_pullback_pct']:
                continue
            if atr_pct < params['min_atr_pct']:
                continue
            
            # LONG setup
            if low <= vwap and close > vwap:
                if delta < params['min_delta_ratio']:
                    continue
                if vol_ratio < params['min_volume_surge']:
                    continue
                
                entry = close
                sl = entry * (1 - params['sl_pct'] / 100)
                tp = entry * (1 + params['tp_pct'] / 100)
                qty = capital * params['position_size'] / entry  # Position sizing
                
                position = {
                    'side': 'LONG', 'entry': entry, 'sl': sl, 'tp': tp,
                    'qty': qty, 'entry_idx': i
                }
            
            # SHORT setup
            elif high >= vwap and close < vwap:
                if delta > (1 / params['min_delta_ratio']):
                    continue
                if vol_ratio < params['min_volume_surge']:
                    continue
                
                entry = close
                sl = entry * (1 + params['sl_pct'] / 100)
                tp = entry * (1 - params['tp_pct'] / 100)
                qty = capital * params['position_size'] / entry
                
                position = {
                    'side': 'SHORT', 'entry': entry, 'sl': sl, 'tp': tp,
                    'qty': qty, 'entry_idx': i
                }
    
    return {
        'trades': trades,
        'final_capital': capital,
        'total_return': (capital - initial_capital) / initial_capital * 100,
        'win_rate': len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100 if trades else 0,
        'avg_pnl': np.mean([t['pnl'] for t in trades]) if trades else 0,
        'max_drawdown': min([t['pnl'] for t in trades]) if trades else 0,
    }

# Parameter grid search
param_sets = [
    # (delta, volume, sl%, tp%, position_size, pullback%, timeout)
    {'name': 'Conservative', 'min_delta_ratio': 1.3, 'min_volume_surge': 1.5, 'sl_pct': 0.10, 'tp_pct': 0.20, 'position_size': 0.05, 'max_pullback_pct': 0.20, 'min_atr_pct': 0.001, 'timeout_bars': 10, 'cooldown_bars': 3},
    {'name': 'Moderate', 'min_delta_ratio': 1.2, 'min_volume_surge': 1.3, 'sl_pct': 0.08, 'tp_pct': 0.16, 'position_size': 0.10, 'max_pullback_pct': 0.30, 'min_atr_pct': 0.0005, 'timeout_bars': 15, 'cooldown_bars': 2},
    {'name': 'Aggressive', 'min_delta_ratio': 1.1, 'min_volume_surge': 1.1, 'sl_pct': 0.06, 'tp_pct': 0.12, 'position_size': 0.15, 'max_pullback_pct': 0.50, 'min_atr_pct': 0.0003, 'timeout_bars': 20, 'cooldown_bars': 1},
    {'name': 'Very Aggressive', 'min_delta_ratio': 1.05, 'min_volume_surge': 1.0, 'sl_pct': 0.05, 'tp_pct': 0.10, 'position_size': 0.20, 'max_pullback_pct': 1.00, 'min_atr_pct': 0.0001, 'timeout_bars': 30, 'cooldown_bars': 1},
]

print("\nRunning parameter optimization...")
results = []

for params in param_sets:
    result = run_backtest(df, params)
    results.append({**params, **result})
    print(f"\n  {params['name']:20s}: "
          f"Return: {result['total_return']:+.1f}% | "
          f"Trades: {len(result['trades']):3d} | "
          f"WR: {result['win_rate']:4.1f}% | "
          f"Final: ${result['final_capital']:.2f}")

# Best result
best = max(results, key=lambda x: x['final_capital'])
print(f"\n{'='*70}")
print(f"  BEST PARAMETERS: {best['name']}")
print(f"{'='*70}")
print(f"  Final Capital: ${best['final_capital']:.2f} ({best['total_return']:+.1f}%)")
print(f"  Total Trades: {len(best['trades'])}")
print(f"  Win Rate: {best['win_rate']:.1f}%")
print(f"  Avg PnL: ${best['avg_pnl']:.2f}")
print(f"\n  Parameters:")
print(f"    Delta Ratio: {best['min_delta_ratio']}")
print(f"    Volume Surge: {best['min_volume_surge']}x")
print(f"    SL: {best['sl_pct']:.2f}% | TP: {best['tp_pct']:.2f}%")
print(f"    Position Size: {best['position_size']*100:.0f}%")
print(f"    Max Pullback: {best['max_pullback_pct']:.2f}%")
print(f"\n  Apply these to live_trader.py _VWAP_PARAMS")

# Show trade distribution
if best['trades']:
    wins = [t for t in best['trades'] if t['pnl'] > 0]
    losses = [t for t in best['trades'] if t['pnl'] <= 0]
    
    print(f"\n  Trade Breakdown ({best['name']}):")
    reasons = {}
    for t in best['trades']:
        r = t['reason']
        if r not in reasons:
            reasons[r] = {'count': 0, 'pnl': 0}
        reasons[r]['count'] += 1
        reasons[r]['pnl'] += t['pnl']
    
    for r, info in sorted(reasons.items(), key=lambda x: x[1]['pnl'], reverse=True):
        print(f"    {r:10s}: {info['count']:3d} trades | ${info['pnl']:+7.2f}")
