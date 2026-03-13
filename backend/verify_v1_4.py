import os
import sys
import pandas as pd
import numpy as np

# Add backend to path
sys.path.insert(0, os.getcwd())

from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators
from ai.xgboost_trainer import generate_features
from ai.adaptive_live_adapter import generate_signal

def verify():
    fetcher = DataFetcher('binance')
    symbol = 'BTCUSDT'
    print(f"\n--- Verifying v1.4 for {symbol} ---")
    
    df = fetcher.fetch_ohlcv(symbol, '15m', limit=500)
    df = add_all_indicators(df)
    df = generate_features(df)
    
    # 1. Verify ATR Rank
    if 'atr_rank_50' in df.columns:
        ranks = df['atr_rank_50'].tail(20).tolist()
        unique_ranks = len(df['atr_rank_50'].unique())
        print(f"ATR Rank Tail: {[round(r, 3) for r in ranks]}")
        print(f"Unique ATR Rank values: {unique_ranks}")
        if unique_ranks > 1:
            print("[OK] ATR Rank is dynamic (fixed).")
        else:
            print("[FAIL] ATR Rank is still static.")
    else:
        print("[FAIL] ATR Rank column missing.")

    # 2. Verify Signal Generation
    # We want to see if any strategy produces a signal in the last 10 bars
    signals_found = 0
    for i in range(1, 11):
        test_df = df.iloc[:-i].copy()
        sig = generate_signal(test_df, symbol, '15m')
        if sig['signal'] != 'HOLD':
            print(f"[OK] SIG FOUND at bar -{i}: {sig['signal']} | {sig['reason']}")
            signals_found += 1
    
    if signals_found == 0:
        print("ℹ[*] No signals in last 10 bars (Market is neutral).")
    else:
        print(f"Total signals in last 10 bars: {signals_found}")

if __name__ == "__main__":
    verify()
