"""VWAP Scalping Debug 1m — TRX"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators

fetcher = DataFetcher()
df = fetcher.fetch_ohlcv('TRX/USDT', '1m', limit=2000)
df = add_all_indicators(df)

# VWAP intraday
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

print("=" * 80)
print("  VWAP Scalping 1m Debug — TRX/USDT")
print("=" * 80)

params = {
    'min_pullback_pct': 0.08,
    'max_pullback_pct': 0.30,
    'min_delta_ratio': 1.2,
    'min_volume_surge': 1.3,
    'min_atr_pct': 0.001,
}

start_ts = pd.Timestamp('2026-03-16 14:00:00')
signals_found = 0

print(f"\n  {'Time':12s} {'Close':8s} {'VWAP':8s} {'Dist%':6s} {'Touch':5s} {'Δ≥1.2':5s} {'Vol≥1.3':7s} {'ATR≥0.1':7s} {'Signal':7s}")
print("  " + "-" * 75)

for i in range(50, len(df)):
    row = df.iloc[i]
    ts = row.name
    if ts < start_ts:
        continue
    
    close = float(row['close'])
    vwap = float(row['vwap'])
    high = float(row['high'])
    low = float(row['low'])
    atr = float(row.get('atr', 0))
    
    dist_pct = abs(close - vwap) / vwap * 100
    touched = (low <= vwap and close > vwap) or (high >= vwap and close < vwap)
    delta = float(row.get('delta_ratio', 0))
    vol = float(row.get('volume_ratio', 0))
    atr_pct = atr / close * 100 if close > 0 else 0
    
    # Checks
    dist_ok = dist_pct <= params['max_pullback_pct']
    delta_ok_long = delta >= params['min_delta_ratio']
    delta_ok_short = delta <= (1 / params['min_delta_ratio'])
    vol_ok = vol >= params['min_volume_surge']
    atr_ok = atr_pct >= params['min_atr_pct']
    
    long_setup = low <= vwap and close > vwap
    short_setup = high >= vwap and close < vwap
    
    signal = ""
    if touched and dist_ok:
        if long_setup and delta_ok_long and vol_ok and atr_ok:
            signal = "<<< LONG"
            signals_found += 1
        elif short_setup and delta_ok_short and vol_ok and atr_ok:
            signal = "<<< SHORT"
            signals_found += 1
    
    # Show every 50th bar + signals
    if signal or (i % 50 == 0):
        print(f"  {ts.strftime('%H:%M'):12s} ${close:.4f} ${vwap:.4f} {dist_pct:5.2f}% "
              f"{'V' if touched else ' ':5s} "
              f"{'V' if delta_ok_long else ' ':5s} "
              f"{'V' if vol_ok else ' ':7s} "
              f"{'V' if atr_ok else ' ':7s} {signal:7s}")
    
    if signals_found >= 5:
        break

print(f"\n  Total signals found: {signals_found}")

if signals_found == 0:
    print("\n  ❌ NO SIGNALS — Possible reasons:")
    print("     1. TRX in strong trend (VWAP distance > 0.3%)")
    print("     2. No VWAP touches with right delta/volume")
    print("     3. Parameters too strict")
    print("\n  🔧 Trying relaxed parameters...")
    
    # Relaxed params
    params_relax = {
        'min_pullback_pct': 0.05,
        'max_pullback_pct': 0.50,
        'min_delta_ratio': 1.0,
        'min_volume_surge': 1.0,
        'min_atr_pct': 0.0005,
    }
    
    signals_found = 0
    for i in range(50, len(df)):
        row = df.iloc[i]
        ts = row.name
        if ts < start_ts:
            continue
        
        close = float(row['close'])
        vwap = float(row['vwap'])
        high = float(row['high'])
        low = float(row['low'])
        
        dist_pct = abs(close - vwap) / vwap * 100
        touched = (low <= vwap and close > vwap) or (high >= vwap and close < vwap)
        delta = float(row.get('delta_ratio', 0))
        vol = float(row.get('volume_ratio', 0))
        
        dist_ok = dist_pct <= params_relax['max_pullback_pct']
        delta_ok_long = delta >= params_relax['min_delta_ratio']
        delta_ok_short = delta <= (1 / params_relax['min_delta_ratio'])
        vol_ok = vol >= params_relax['min_volume_surge']
        
        long_setup = low <= vwap and close > vwap
        short_setup = high >= vwap and close < vwap
        
        if touched and dist_ok:
            reason = []
            if long_setup:
                reason.append("LONG setup")
                if not delta_ok_long:
                    reason.append(f"delta {delta:.2f} < 1.0")
                if not vol_ok:
                    reason.append(f"vol {vol:.2f}x < 1.0x")
            elif short_setup:
                reason.append("SHORT setup")
                if not delta_ok_short:
                    reason.append(f"delta {delta:.2f} > 1.0")
                if not vol_ok:
                    reason.append(f"vol {vol:.2f}x < 1.0x")
            else:
                continue
            
            if (long_setup and delta_ok_long and vol_ok) or (short_setup and delta_ok_short and vol_ok):
                signals_found += 1
                print(f"  🆕 RELAXED SIGNAL: {ts.strftime('%H:%M')} {reason[0]} "
                      f"dist={dist_pct:.2f}% delta={delta:.2f} vol={vol:.2f}x")
            else:
                print(f"  ❌ REJECTED: {ts.strftime('%H:%M')} {reason[0]} — {' | '.join(reason[1:])}")
        
        if signals_found >= 3:
            break
    
    if signals_found == 0:
        print("\n  Even with relaxed params — no signals.")
        print("  TRX trending too strongly for VWAP mean reversion.")
