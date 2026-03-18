"""VWAP Scalping Deep Optimization — Multi-Parameter Grid Search"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from itertools import product
from backtest.data_fetcher import DataFetcher

print("=" * 70)
print("  VWAP Scalping — Deep Parameter Optimization")
print("=" * 70)

# Fetch 1m data
fetcher = DataFetcher()
df = fetcher.fetch_ohlcv('TRX/USDT', '1m', limit=10000)

if df is None or len(df) < 1000:
    print("ERROR: No data")
    sys.exit(1)

print(f"Data: {len(df)} bars | {df.index[0]} → {df.index[-1]}")

# Indicators
df['tp'] = (df['high'] + df['low'] + df['close']) / 3
df['date'] = df.index.date
df['vwap'] = df.groupby('date').apply(lambda x: (x['tp'] * x['volume']).cumsum() / x['volume'].cumsum()).reset_index(0, drop=True)
df['body'] = df['close'] - df['open']
df['buy_volume'] = np.where(df['body'] > 0, df['volume'] * abs(df['body']/df['open']), 0)
df['sell_volume'] = np.where(df['body'] < 0, df['volume'] * abs(df['body']/df['open']), 0)
df['delta_ratio'] = df['buy_volume'].rolling(5).sum() / (df['sell_volume'].rolling(5).sum() + 1e-9)
df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
df['atr'] = df['close'].rolling(14).apply(lambda x: max(x) - min(x), raw=True)

# Backtest function
def backtest(df, p, capital=1000):
    pos = None
    trades = []
    cooldown = 0
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        close, high, low, vwap = float(row['close']), float(row['high']), float(row['low']), float(row['vwap'])
        delta, vol_ratio, atr = float(row['delta_ratio']), float(row['volume_ratio']), float(row['atr'])
        
        if pos:
            # Exit
            if pos['side'] == 'LONG':
                if high >= pos['tp']:
                    pnl = (pos['tp'] - pos['entry']) * pos['qty']
                    trades.append({'pnl': pnl, 'reason': 'TP'})
                    capital += pnl
                    pos = None
                    cooldown = i + p['cooldown']
                elif low <= pos['sl']:
                    pnl = (pos['sl'] - pos['entry']) * pos['qty']
                    trades.append({'pnl': pnl, 'reason': 'SL'})
                    capital += pnl
                    pos = None
                    cooldown = i + p['cooldown']
                elif i - pos['idx'] >= p['timeout']:
                    pnl = (close - pos['entry']) * pos['qty']
                    trades.append({'pnl': pnl, 'reason': 'TO'})
                    capital += pnl
                    pos = None
                    cooldown = i + p['cooldown']
            else:  # SHORT
                if low <= pos['tp']:
                    pnl = (pos['entry'] - pos['tp']) * pos['qty']
                    trades.append({'pnl': pnl, 'reason': 'TP'})
                    capital += pnl
                    pos = None
                    cooldown = i + p['cooldown']
                elif high >= pos['sl']:
                    pnl = (pos['entry'] - pos['sl']) * pos['qty']
                    trades.append({'pnl': pnl, 'reason': 'SL'})
                    capital += pnl
                    pos = None
                    cooldown = i + p['cooldown']
                elif i - pos['idx'] >= p['timeout']:
                    pnl = (pos['entry'] - close) * pos['qty']
                    trades.append({'pnl': pnl, 'reason': 'TO'})
                    capital += pnl
                    pos = None
                    cooldown = i + p['cooldown']
        
        if not pos and i >= cooldown:
            # Entry
            dist = abs(close - vwap) / vwap * 100
            atr_pct = atr / close
            
            if dist > p['max_dist'] or atr_pct < p['min_atr']:
                continue
            
            # LONG
            if low <= vwap and close > vwap and delta >= p['delta'] and vol_ratio >= p['vol']:
                entry = close
                sl = entry * (1 - p['sl']/100)
                tp = entry * (1 + p['tp']/100)
                qty = capital * p['size'] / entry
                pos = {'side': 'LONG', 'entry': entry, 'sl': sl, 'tp': tp, 'qty': qty, 'idx': i}
            # SHORT
            elif high >= vwap and close < vwap and delta <= 1/p['delta'] and vol_ratio >= p['vol']:
                entry = close
                sl = entry * (1 + p['sl']/100)
                tp = entry * (1 - p['tp']/100)
                qty = capital * p['size'] / entry
                pos = {'side': 'SHORT', 'entry': entry, 'sl': sl, 'tp': tp, 'qty': qty, 'idx': i}
    
    wins = [t for t in trades if t['pnl'] > 0]
    return {
        'return': (capital - 1000) / 1000 * 100,
        'trades': len(trades),
        'wr': len(wins) / len(trades) * 100 if trades else 0,
        'final': capital,
        'avg_pnl': np.mean([t['pnl'] for t in trades]) if trades else 0,
        'max_dd': min([t['pnl'] for t in trades]) if trades else 0,
    }

# Grid search
print("\nRunning grid search (this may take a minute)...")

param_grid = {
    'delta': [1.0, 1.1, 1.2, 1.3],
    'vol': [1.0, 1.1, 1.2, 1.3],
    'sl': [0.05, 0.08, 0.10, 0.15],
    'tp': [0.10, 0.15, 0.20, 0.30],
    'size': [0.05, 0.10, 0.15, 0.20],
    'max_dist': [0.20, 0.30, 0.50, 1.00],
    'min_atr': [0.0003, 0.0005, 0.001],
    'timeout': [10, 15, 20],
    'cooldown': [1, 2, 3],
}

# Sample 100 random combinations
import random
random.seed(42)
keys = list(param_grid.keys())
results = []

for _ in range(100):
    p = {k: random.choice(param_grid[k]) for k in keys}
    r = backtest(df, p)
    if r['trades'] >= 10:  # Minimum 10 trades
        results.append({**p, **r})

# Sort by return
results.sort(key=lambda x: x['return'], reverse=True)

print(f"\n{'='*70}")
print("  TOP 10 PARAMETER COMBINATIONS")
print(f"{'='*70}")
print(f"{'Rank':>4} {'Return':>8} {'Trades':>7} {'WR':>6} {'Delta':>5} {'Vol':>4} {'SL':>4} {'TP':>4} {'Size':>5}")
print("-" * 70)

for i, r in enumerate(results[:10], 1):
    print(f"{i:>4} {r['return']:>+7.1f}% {r['trades']:>7} {r['wr']:>5.1f}% {r['delta']:>5.1f} {r['vol']:>4.1f} {r['sl']:>4.2f} {r['tp']:>4.2f} {r['size']*100:>4.0f}%")

if results:
    best = results[0]
    print(f"\n{'='*70}")
    print(f"  BEST PARAMETERS:")
    print(f"{'='*70}")
    print(f"  Return: {best['return']:+.1f}% (${best['final']:.2f})")
    print(f"  Trades: {best['trades']} | WR: {best['wr']:.1f}%")
    print(f"\n  Settings:")
    print(f"    Delta Ratio: {best['delta']}")
    print(f"    Volume Surge: {best['vol']}x")
    print(f"    SL: {best['sl']:.2f}% | TP: {best['tp']:.2f}%")
    print(f"    Position Size: {best['size']*100:.0f}%")
    print(f"    Max Distance: {best['max_dist']:.2f}%")
    print(f"    Min ATR: {best['min_atr']:.4f}")
    print(f"    Timeout: {best['timeout']} bars | Cooldown: {best['cooldown']}")
else:
    print("\nNo valid results found with minimum 10 trades")
