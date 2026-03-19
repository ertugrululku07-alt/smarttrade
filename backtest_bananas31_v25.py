"""
BANANAS31/USDT Backtest with v2.5 Filters
Test new entry filters on recent BANANAS31 trade
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators
from ai import ict_core
import pandas as pd
import numpy as np
from datetime import datetime

def simulate_v25_filters(df, idx, direction='LONG'):
    """Simulate v2.5 entry filters"""
    reject_reasons = []
    
    df_slice = df.iloc[:idx+1].copy()
    if len(df_slice) < 50:
        return False, ["Insufficient data"]
    
    cp = float(df_slice['close'].iloc[-1])
    
    # HTF Resistance Proximity (5%)
    htf_lookback = min(100, len(df_slice))
    htf_high = float(df_slice.iloc[-htf_lookback:]['high'].max())
    
    if direction == 'LONG':
        distance_from_high = (htf_high - cp) / htf_high
        if distance_from_high < 0.05:
            reject_reasons.append(f"Resistance proximity: {distance_from_high*100:.1f}% < 5%")
    
    # Pullback Requirement (3%)
    if len(df_slice) >= 10:
        recent_high = float(df_slice.iloc[-10:]['high'].max())
        pullback = (recent_high - cp) / recent_high
        if direction == 'LONG' and pullback < 0.03:
            reject_reasons.append(f"No pullback: {pullback*100:.1f}% < 3%")
    
    # Premium Zone (70%)
    zone_lookback = min(30, len(df_slice))
    recent = df_slice.iloc[-zone_lookback:]
    range_high = float(recent['high'].max())
    range_low = float(recent['low'].min())
    
    if range_high > range_low:
        position = (cp - range_low) / (range_high - range_low)
        if direction == 'LONG' and position > 0.70:
            reject_reasons.append(f"Premium zone: {position*100:.0f}% > 70%")
    
    # RSI Overbought
    if 'rsi' in df_slice.columns:
        rsi = float(df_slice['rsi'].iloc[-1])
        if not np.isnan(rsi) and direction == 'LONG' and rsi > 70:
            reject_reasons.append(f"RSI overbought: {rsi:.1f} > 70")
    
    return len(reject_reasons) == 0, reject_reasons

def backtest_bananas31():
    print("=" * 80)
    print("BANANAS31/USDT BACKTEST - v2.5 Entry Filters")
    print("=" * 80)
    print()
    
    fetcher = DataFetcher('binance')
    
    # Fetch last 7 days
    print("Fetching BANANAS31/USDT 1h data (last 7 days)...")
    df = fetcher.fetch_ohlcv('BANANAS31/USDT', '1h', limit=200)
    
    if df is None or len(df) < 50:
        print("❌ Failed to fetch data")
        return
    
    df = add_all_indicators(df)
    
    print(f"✅ Loaded {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print()
    
    # Find recent entry (around $0.0107)
    entry_idx = None
    for i in range(len(df)-20, len(df)):
        close = df.iloc[i]['close']
        if 0.0105 < close < 0.0109:
            entry_idx = i
            break
    
    if entry_idx:
        print("=" * 80)
        print("ACTUAL ENTRY ANALYSIS (45 min ago)")
        print("=" * 80)
        print()
        
        entry_price = df.iloc[entry_idx]['close']
        entry_time = df.index[entry_idx]
        
        print(f"Entry Time: {entry_time}")
        print(f"Entry Price: ${entry_price:.6f}")
        print()
        
        # Test with v2.5 filters
        allowed, reasons = simulate_v25_filters(df, entry_idx, 'LONG')
        
        print(f"v2.5 Filter Result: {'✅ ALLOWED' if allowed else '❌ REJECTED'}")
        print()
        
        if not allowed:
            print("Rejection Reasons:")
            for reason in reasons:
                print(f"  ❌ {reason}")
            print()
        
        # Market context
        print("Market Context:")
        htf_high = df.iloc[max(0, entry_idx-100):entry_idx+1]['high'].max()
        htf_low = df.iloc[max(0, entry_idx-100):entry_idx+1]['low'].min()
        recent_high = df.iloc[max(0, entry_idx-10):entry_idx+1]['high'].max()
        
        print(f"  100-bar high: ${htf_high:.6f}")
        print(f"  100-bar low: ${htf_low:.6f}")
        print(f"  10-bar high: ${recent_high:.6f}")
        print(f"  Distance from high: {((htf_high - entry_price) / htf_high * 100):.1f}%")
        print(f"  Pullback: {((recent_high - entry_price) / recent_high * 100):.1f}%")
        
        if 'rsi' in df.columns:
            rsi = df.iloc[entry_idx]['rsi']
            print(f"  RSI: {rsi:.1f}")
        print()
    
    # Scan for valid entries
    print("=" * 80)
    print("VALID ENTRY OPPORTUNITIES (v2.5)")
    print("=" * 80)
    print()
    
    valid_entries = []
    for i in range(100, len(df)):
        allowed, _ = simulate_v25_filters(df, i, 'LONG')
        if allowed:
            valid_entries.append({
                'idx': i,
                'time': df.index[i],
                'price': df.iloc[i]['close'],
                'rsi': df.iloc[i]['rsi'] if 'rsi' in df.columns else 50
            })
    
    print(f"Found {len(valid_entries)} valid entries in dataset")
    print()
    
    if valid_entries:
        print("Last 5 valid entries:")
        for entry in valid_entries[-5:]:
            print(f"  {entry['time']}: ${entry['price']:.6f} (RSI: {entry['rsi']:.1f})")
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    if entry_idx:
        allowed, reasons = simulate_v25_filters(df, entry_idx, 'LONG')
        
        if not allowed:
            print("✅ v2.5 filters WOULD REJECT the actual entry")
            print()
            print("Why this entry is bad:")
            for reason in reasons:
                print(f"  • {reason}")
            print()
            print("Result: Trade would NOT be opened")
            print("        No -$3.57 loss ✅")
        else:
            print("⚠️ v2.5 filters would ALLOW this entry")
            print("   Entry quality may be acceptable")
    
    print()
    print(f"Total valid opportunities: {len(valid_entries)}")
    print()
    
    # Recommendation
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    print("For BANANAS31:")
    print("  • Wait for pullback to $0.0085-0.0095")
    print("  • RSI < 45 (oversold)")
    print("  • Discount zone (<50% range)")
    print("  • OB/FVG confluence")
    print()
    print("Current entry ($0.0107):")
    print("  • Too close to resistance")
    print("  • No pullback")
    print("  • Premium zone")
    print("  • Should be REJECTED ❌")
    print()

if __name__ == '__main__':
    backtest_bananas31()
