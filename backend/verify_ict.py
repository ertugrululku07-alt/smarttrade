import pandas as pd
import numpy as np
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators
from ai.adaptive_engine import AdaptiveEngine, Regime
import os

def verify_ict():
    print("ICT Indicator & Strategy Verification")
    print("-" * 40)
    
    fetcher = DataFetcher('binance')
    symbol = "BTCUSDT"
    timeframe = "1h"
    limit = 500
    
    print(f"Fetching {limit} bars of {symbol} {timeframe}...")
    df = fetcher.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    if df is None or df.empty:
        print("Failed to fetch data.")
        return
        
    print("Calculating indicators (including ICT)...")
    df = add_all_indicators(df)
    
    # 1. Indicator Verification
    fvg_bull_count = df['fvg_bull'].sum()
    fvg_bear_count = df['fvg_bear'].sum()
    ob_bull_count = (df['ob_bull'] > 0).sum()
    ob_bear_count = (df['ob_bear'] > 0).sum()
    mss_up_count = df['mss_up'].sum()
    mss_down_count = df['mss_down'].sum()
    
    print(f"\n[INDICATORS FOUND]")
    print(f"Bullish FVGs: {fvg_bull_count}")
    print(f"Bearish FVGs: {fvg_bear_count}")
    print(f"Bullish Order Blocks: {ob_bull_count}")
    print(f"Bearish Order Blocks: {ob_bear_count}")
    print(f"MSS Up (Breakouts): {mss_up_count}")
    print(f"MSS Down (Breakouts): {mss_down_count}")
    
    # 2. Strategy Detection Verification
    print("\n[ENGINE/STRATEGY TEST]")
    engine = AdaptiveEngine(primary_tf=timeframe, use_meta_filter=False)
    
    # Rejimi HIGH_VOLATILE'a zorlayarak ICT stratejisini test edelim (eğer doğal oluşmadıysa)
    # Ya da son 50 barda sinyal var mı bakalım
    signals_found = 0
    for i in range(len(df) - 50, len(df)):
        chunk = df.iloc[:i+1]
        decision = engine.decide(chunk)
        
        if decision.action != 'HOLD' and 'ict_smc' in decision.reason.lower():
            print(f"Bar {df.index[i]}: {decision.action} | {decision.reason} | Score: {decision.soft_score}")
            signals_found += 1
            
    if signals_found == 0:
        print("No ICT signals found in the last 50 bars. (This depends on market conditions)")
    else:
        print(f"Total ICT signals found: {signals_found}")

if __name__ == "__main__":
    verify_ict()
