"""BTC/USDT Scalping Test — 5m, Best Strategy from Shootout"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from backtest.data_fetcher import DataFetcher

print("=" * 70)
print("  BTC/USDT Scalping — Momentum Breakout (5m)")
print("=" * 70)

fetcher = DataFetcher()
df = fetcher.fetch_ohlcv('BTC/USDT', '5m', limit=2000)

if df is None or len(df) < 500:
    print("ERROR: No BTC data")
    sys.exit(1)

print(f"Data: {len(df)} 5m bars ({len(df)*5/60:.0f} hours)")

# Indicators
df['ema9'] = df['close'].ewm(span=9).mean()
df['ema21'] = df['close'].ewm(span=21).mean()
df['rsi'] = df['close'].rolling(14).apply(
    lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).mean() / abs(x.diff().clip(upper=0)).mean()))), raw=False)
df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
df['volume_ma'] = df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['volume_ma']

# Momentum breakout entry
def momentum_entry(df, i):
    row = df.iloc[i]
    prev = df.iloc[i-1]
    
    close = float(row['close'])
    ema9 = float(row['ema9'])
    ema21 = float(row['ema21'])
    rsi = float(row['rsi'])
    vol = float(row['vol_ratio'])
    
    # Golden cross with momentum
    if ema9 > ema21 and float(prev['ema9']) <= float(prev['ema21']):
        if 45 < rsi < 75 and vol > 1.2:
            return 'LONG'
    
    # Death cross
    if ema9 < ema21 and float(prev['ema9']) >= float(prev['ema21']):
        if 25 < rsi < 55 and vol > 1.2:
            return 'SHORT'
    
    return None

# Backtest
capital = 1000
pos = None
trades = []

# Optimized params for BTC
sl_pct = 0.2   # 0.2% SL (BTC less volatile than altcoins)
tp_pct = 0.6   # 0.6% TP (R:R 1:3)
position = 0.20  # 20% position

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
        side = momentum_entry(df, i)
        if side:
            entry = close
            sl = entry * (1 - sl_pct/100) if side == 'LONG' else entry * (1 + sl_pct/100)
            tp = entry * (1 + tp_pct/100) if side == 'LONG' else entry * (1 - tp_pct/100)
            qty = capital * position / entry
            pos = {'side': side, 'entry': entry, 'sl': sl, 'tp': tp, 'qty': qty}

# Results
wins = [t for t in trades if t['pnl'] > 0]
losses = [t for t in trades if t['pnl'] < 0]

print(f"\n{'='*70}")
print("  BTC/USDT RESULTS")
print(f"{'='*70}")
print(f"  Return: {(capital-1000)/1000*100:+.1f}% (${capital:.2f})")
print(f"  Trades: {len(trades)}")
print(f"  Win Rate: {len(wins)/len(trades)*100:.1f}%" if trades else "  Win Rate: N/A")
print(f"  Wins: {len(wins)} | Losses: {len(losses)}")
if wins and losses:
    avg_win = np.mean([t['pnl'] for t in wins])
    avg_loss = np.mean([abs(t['pnl']) for t in losses])
    print(f"  Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
    print(f"  R:R: {avg_win/avg_loss:.2f}:1")

if capital > 1005:
    print(f"\n  ✅ PROFITABLE! +${capital-1000:.2f}")
    print(f"  Strategy: Momentum Breakout on BTC/USDT 5m")
    print(f"  Params: SL={sl_pct}% TP={tp_pct}% Position={position*100}%")
elif capital > 1000:
    print(f"\n  ⚠️ Marginal profit")
else:
    print(f"\n  ❌ Unprofitable on BTC/USDT as well")
    print(f"\n  CONCLUSION: Scalping not viable on current market conditions")
    print(f"  Recommendation: Stick with ICT/SMC swing trading only")
