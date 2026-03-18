"""VWAP Scalping Debug — TRX 16 Mart+"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators

fetcher = DataFetcher()
df = fetcher.fetch_ohlcv('TRX/USDT', '1h', limit=200)
df = add_all_indicators(df)

# VWAP
df['tp'] = (df['high'] + df['low'] + df['close']) / 3
df['tp_volume'] = df['tp'] * df['volume']
df['vwap'] = df['tp_volume'].cumsum() / df['volume'].cumsum()

# Delta
df['body'] = df['close'] - df['open']
df['body_pct'] = df['body'] / df['open']
df['buy_volume'] = np.where(df['body'] > 0, df['volume'] * abs(df['body_pct']), 0)
df['sell_volume'] = np.where(df['body'] < 0, df['volume'] * abs(df['body_pct']), 0)
df['delta_ratio'] = df['buy_volume'].rolling(5).sum() / (df['sell_volume'].rolling(5).sum() + 1e-9)
df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

start_ts = pd.Timestamp('2026-03-16 00:00:00')

print("=" * 70)
print("  VWAP Scalping Debug — Son 20 bar (16 Mart+)")
print("=" * 70)
print(f"\n  {'Time':12s} {'Close':8s} {'VWAP':8s} {'Dist%':6s} {'Touch':5s} {'Delta':6s} {'Vol':5s} {'ATR%':5s}")
print("  " + "-" * 65)

for i in range(len(df)-20, len(df)):
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
    touched = (low <= vwap <= high) or (low <= vwap and close > vwap) or (high >= vwap and close < vwap)
    delta = float(row.get('delta_ratio', 0))
    vol = float(row.get('volume_ratio', 0))
    atr_pct = atr / close * 100 if close > 0 else 0
    
    # Signal check
    long_ok = low <= vwap and close > vwap and delta >= 1.3 and vol >= 1.5 and atr_pct >= 0.3
    short_ok = high >= vwap and close < vwap and delta <= 0.77 and vol >= 1.5 and atr_pct >= 0.3
    
    mark = ""
    if long_ok:
        mark = "<<< LONG"
    elif short_ok:
        mark = "<<< SHORT"
    
    print(f"  {ts.strftime('%m/%d %H:%M'):12s} ${close:.4f} ${vwap:.4f} {dist_pct:5.2f}% "
          f"{'V' if touched else ' ':5s} {delta:5.2f} {vol:4.1f}x {atr_pct:4.2f}% {mark}")

print(f"\n  Açıklama:")
print(f"    Touch = Fiyat VWAP'a dokundu ve bounce/reject yaptı")
print(f"    Delta = Buy/Sell volume ratio (1.3+ = LONG, <0.77 = SHORT)")
print(f"    Vol = Volume vs 20-bar average (1.5x = surge)")
print(f"    ATR% = Minimum 0.3% gerekli")
print(f"\n  16 Mart-17 Mart periyodunda TRX güçlü uptrend ($0.295→$0.307)")
print(f"  VWAP scalping için ideal değil — daha az pullback, daha az fırsat")
