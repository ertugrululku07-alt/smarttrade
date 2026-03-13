"""
Verification Script for Phase 1.5 Upgrades
Meta-Label Generator v1.3 + Meta-Trainer v1.2
"""

import os
import pandas as pd
import numpy as np
import time
from backtest.data_fetcher import DataFetcher
from ai.meta_labelling.meta_label_generator import generate_meta_labels_bulk
from ai.meta_labelling.meta_trainer import train_meta_models_bulk

def run_verification():
    print("\n" + "="*50)
    print("PHASE 1.5 PIPELINE VERIFICATION")
    print("="*50)
    
    # 1. Fetch Data
    fetcher = DataFetcher('binance')
    symbols = ['BTC/USDT', 'ETH/USDT']
    print(f"\n[1/4] Fetching data for {symbols}...")
    
    all_dfs = {}
    all_dfs_4h = {}
    
    for symbol in symbols:
        try:
            data = fetcher.fetch_multi_tf(symbol, timeframes=['4h', '1h'], limit=1000)
            if '1h' in data and '4h' in data:
                all_dfs[symbol] = data['1h']
                all_dfs_4h[symbol] = data['4h']
                print(f"      {symbol}: 1h={len(data['1h'])}, 4h={len(data['4h'])}")
        except:
            continue
        
    if not all_dfs:
        print("No data fetched.")
        return
        
    # 2. Add minimal indicators
    for sym, df in all_dfs.items():
        df['atr'] = df['close'].rolling(14).std().fillna(df['close'] * 0.01)
        df['rsi'] = 50.0
        df['adx'] = 25.0
    
    # 3. Generate Meta-Labels
    print(f"\n[2/4] Generating Meta-Labels (v1.3)...")
    
    # Low trials/lookback for speed
    meta_df, stats = generate_meta_labels_bulk(
        all_dfs,
        all_dfs_4h=all_dfs_4h,
        timeframe="1h",
        lookahead=12,
        regime_lookback=30,
        verbose=True
    )
    
    if meta_df.empty:
        print("\n[FAIL] No meta-labels generated. Pipeline broken or insufficient data.")
        return
    
    print(f"\n[OK] Meta-Labels generated: {len(meta_df)} samples")
    print(f"     HTF Features found: {[c for c in meta_df.columns if c.startswith('htf_')]}")
    
    # 4. Train Meta-Models
    print(f"\n[3/4] Training Meta-Models (v1.2)...")
    # Low trials for speed
    train_results = train_meta_models_bulk(
        meta_df,
        model_dir="models/meta_verify",
        n_trials=5 
    )
    
    if not train_results:
        print("\n[FAIL] Training produced no models. Check logs.")
        return
        
    print(f"\n[OK] Training completed for regimes: {list(train_results.keys())}")
    
    # 5. Result Inspect
    print(f"\n[4/4] Verifying saved artifacts...")
    model_files = os.listdir("models/meta_verify")
    print(f"      Saved models: {model_files}")
    
    print("\n" + "="*50)
    print("VERIFICATION SUCCESSFUL")
    print("="*50)

if __name__ == "__main__":
    run_verification()
