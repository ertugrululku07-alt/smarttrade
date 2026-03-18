"""Scalping Strategy Shootout — 3 Approaches, 5m, 1 Week, $1000"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from backtest.data_fetcher import DataFetcher

print("=" * 70)
print("  Scalping Shootout — Momentum vs Range vs Liquidity")
print("=" * 70)

# Fetch 5m data (1 week = ~2000 bars)
fetcher = DataFetcher()
print("Fetching 5m data...")
df = fetcher.fetch_ohlcv('TRX/USDT', '5m', limit=2000)

if df is None or len(df) < 500:
    print("ERROR: Not enough 5m data")
    sys.exit(1)

print(f"Data: {len(df)} 5m bars ({len(df)*5/60:.0f} hours)")

# Indicators
df['ema9'] = df['close'].ewm(span=9).mean()
df['ema21'] = df['close'].ewm(span=21).mean()
df['ema50'] = df['close'].ewm(span=50).mean()
df['rsi'] = df['close'].rolling(14).apply(
    lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).mean() / abs(x.diff().clip(upper=0)).mean()))), raw=False)
df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
df['volume_ma'] = df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['volume_ma']

# Swing highs/lows for liquidity
df['swing_high'] = df['high'].rolling(5).max().shift(-2)
df['swing_low'] = df['low'].rolling(5).min().shift(-2)

def backtest_strategy(name, entry_func, params, capital=1000):
    """Generic backtest."""
    pos = None
    trades = []
    
    for i in range(100, len(df)-1):
        row = df.iloc[i]
        close = float(row['close'])
        high = float(row['high'])
        low = float(row['low'])
        
        # Exit
        if pos:
            if pos['side'] == 'LONG':
                if high >= pos['tp']:
                    pnl = (pos['tp'] - pos['entry']) * pos['qty']
                    capital += pnl
                    trades.append({'pnl': pnl, 'side': 'LONG'})
                    pos = None
                elif low <= pos['sl']:
                    pnl = (pos['sl'] - pos['entry']) * pos['qty']
                    capital += pnl
                    trades.append({'pnl': pnl, 'side': 'LONG'})
                    pos = None
            else:
                if low <= pos['tp']:
                    pnl = (pos['entry'] - pos['tp']) * pos['qty']
                    capital += pnl
                    trades.append({'pnl': pnl, 'side': 'SHORT'})
                    pos = None
                elif high >= pos['sl']:
                    pnl = (pos['entry'] - pos['sl']) * pos['qty']
                    capital += pnl
                    trades.append({'pnl': pnl, 'side': 'SHORT'})
                    pos = None
        
        # Entry
        if not pos:
            side = entry_func(df, i, params)
            if side:
                entry = close
                sl = entry * (1 - params['sl_pct']/100) if side == 'LONG' else entry * (1 + params['sl_pct']/100)
                tp = entry * (1 + params['tp_pct']/100) if side == 'LONG' else entry * (1 - params['tp_pct']/100)
                qty = capital * params['position'] / entry
                pos = {'side': side, 'entry': entry, 'sl': sl, 'tp': tp, 'qty': qty}
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]
    
    return {
        'name': name,
        'return': (capital - 1000) / 1000 * 100,
        'trades': len(trades),
        'wr': len(wins) / len(trades) * 100 if trades else 0,
        'final': capital,
        'avg_win': np.mean([t['pnl'] for t in wins]) if wins else 0,
        'avg_loss': np.mean([abs(t['pnl']) for t in losses]) if losses else 0,
        'max_dd': min([t['pnl'] for t in trades]) if trades else 0,
    }

# Strategy 1: MOMENTUM BREAKOUT
def momentum_entry(df, i, p):
    """Trende katıl - momentum confirmation ile."""
    row = df.iloc[i]
    prev = df.iloc[i-1]
    
    close = float(row['close'])
    ema9 = float(row['ema9'])
    ema21 = float(row['ema21'])
    rsi = float(row['rsi'])
    vol = float(row['vol_ratio'])
    
    # Uptrend momentum
    if ema9 > ema21 and float(prev['ema9']) <= float(prev['ema21']):
        if 45 < rsi < 75 and vol > 1.3:
            return 'LONG'
    
    # Downtrend momentum  
    if ema9 < ema21 and float(prev['ema9']) >= float(prev['ema21']):
        if 25 < rsi < 55 and vol > 1.3:
            return 'SHORT'
    
    return None

# Strategy 2: RANGE SCALPING (Support/Resistance)
def range_entry(df, i, p):
    """Son 20 barın range'inde al/sat."""
    if i < 30:
        return None
    
    recent = df.iloc[i-20:i]
    range_high = float(recent['high'].max())
    range_low = float(recent['low'].min())
    range_size = range_high - range_low
    
    if range_size < float(df.iloc[i]['close']) * 0.005:  # Range çok dar
        return None
    
    close = float(df.iloc[i]['close'])
    rsi = float(df.iloc[i]['rsi'])
    vol = float(df.iloc[i]['vol_ratio'])
    
    # Range low'dan dönüş
    if close < range_low + range_size * 0.15 and rsi < 40 and vol > 1.2:
        return 'LONG'
    
    # Range high'dan dönüş
    if close > range_high - range_size * 0.15 and rsi > 60 and vol > 1.2:
        return 'SHORT'
    
    return None

# Strategy 3: LIQUIDITY GRAB (Equal High/Low Break)
def liquidity_entry(df, i, p):
    """Eşit high/low kırılımlarında fade."""
    if i < 30:
        return None
    
    # Son 20 barın swing high/low'ları
    swing_highs = []
    swing_lows = []
    
    for j in range(i-25, i-5):
        if df.iloc[j]['high'] == df.iloc[j-5:j+5]['high'].max():
            swing_highs.append(float(df.iloc[j]['high']))
        if df.iloc[j]['low'] == df.iloc[j-5:j+5]['low'].min():
            swing_lows.append(float(df.iloc[j]['low']))
    
    if not swing_highs or not swing_lows:
        return None
    
    # Eşit seviyeler (tolerance 0.1%)
    close = float(df.iloc[i]['close'])
    high = float(df.iloc[i]['high'])
    low = float(df.iloc[i]['low'])
    
    eq_high = None
    for h in swing_highs:
        if abs(high - h) / h < 0.001:  # Break equal high
            eq_high = h
            break
    
    eq_low = None
    for l in swing_lows:
        if abs(low - l) / l < 0.001:  # Break equal low
            eq_low = l
            break
    
    vol = float(df.iloc[i]['vol_ratio'])
    
    # Equal high break → fade short (trap)
    if eq_high and close < eq_high * 0.998 and vol > 1.5:
        return 'SHORT'
    
    # Equal low break → fade long (trap)
    if eq_low and close > eq_low * 1.002 and vol > 1.5:
        return 'LONG'
    
    return None

# Test all
params = {'sl_pct': 0.3, 'tp_pct': 0.6, 'position': 0.15}  # R:R 1:2, 15% risk

print("\nTesting MOMENTUM BREAKOUT...")
r1 = backtest_strategy("Momentum Breakout", momentum_entry, params)

print("Testing RANGE SCALPING...")
r2 = backtest_strategy("Range Scalping", range_entry, params)

print("Testing LIQUIDITY GRAB...")
r3 = backtest_strategy("Liquidity Grab", liquidity_entry, params)

# Results
print(f"\n{'='*70}")
print("  SHOOTOUT RESULTS")
print(f"{'='*70}")

for r in [r1, r2, r3]:
    rr = r['avg_win'] / r['avg_loss'] if r['avg_loss'] > 0 else 0
    status = "✅ WIN" if r['return'] > 2 else "⚠️ OK" if r['return'] > 0 else "❌ FAIL"
    print(f"\n  {r['name']} {status}")
    print(f"    Return: {r['return']:+.1f}% (${r['final']:.2f})")
    print(f"    Trades: {r['trades']} | WR: {r['wr']:.1f}% | R:R: {rr:.1f}:1")
    print(f"    Max DD: ${r['max_dd']:.2f}")

# Winner
winner = max([r1, r2, r3], key=lambda x: x['final'])
print(f"\n{'='*70}")
print(f"  🏆 WINNER: {winner['name']} (+{winner['return']:.1f}%)")
print(f"{'='*70}")

if winner['return'] > 2:
    print(f"\n  ✅ Profitable! Implement this strategy.")
    print(f"\n  Optimal parameters:")
    print(f"    SL: {params['sl_pct']:.1f}% | TP: {params['tp_pct']:.1f}%")
    print(f"    Position: {params['position']*100:.0f}%")
elif winner['return'] > 0:
    print(f"\n  ⚠️ Marginal. May work with optimization.")
else:
    print(f"\n  ❌ All strategies failed on TRX/USDT 5m.")
    print(f"\n  Try:")
    print(f"    - Different coin (BTC/ETH more liquid)")
    print(f"    - Longer timeframe (15m/30m)")
    print(f"    - Or stick with ICT/SMC only")
