import os
import sys
import pandas as pd
from datetime import datetime

# Path setup
backend_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_path)

from backtest.data_fetcher import DataFetcher
from ai.adaptive_live_adapter import generate_signal
from backtest.signals import add_all_indicators
from ai.data_sources.futures_data import enrich_ohlcv_with_futures
from ai.xgboost_trainer import generate_features

def test_hybrid_flow():
    print("🚀 Starting Hybrid Live Smoke Test...")
    fetcher = DataFetcher('binance')
    symbol = "BTC/USDT"
    
    print(f"📥 Fetching 1H data for {symbol}...")
    df_1h = fetcher.fetch_ohlcv(symbol, '1h', limit=100)
    print(f"📥 Fetching 15m data for {symbol}...")
    df_15m = fetcher.fetch_ohlcv(symbol, '15m', limit=100)
    
    if df_1h.empty or df_15m.empty:
        print("❌ Error: Fetching failed.")
        return

    print("🔧 Adding indicators and features...")
    df_1h = add_all_indicators(df_1h)
    df_1h = generate_features(df_1h)
    
    df_15m = add_all_indicators(df_15m)
    df_15m = generate_features(df_15m)

    print("🧠 Generating Hybrid Signal...")
    result = generate_signal(
        df_1h, 
        df_secondary=df_15m, 
        symbol=symbol, 
        timeframe='1h', 
        secondary_tf='15m'
    )
    
    print("\n📊 TEST RESULTS:")
    print(f"  Symbol      : {symbol}")
    print(f"  Signal      : {result['signal']}")
    print(f"  Confidence  : {result['confidence']:.4f}")
    print(f"  Regime      : {result['regime']}")
    print(f"  Strategy    : {result['strategy']}")
    print(f"  Reason      : {result['reason']}")
    
    if result['regime'] != 'unknown':
        print("\n✅ Smoke test PASSED: Hybrid logic is wired up correctly.")
    else:
        print("\n❌ Smoke test FAILED: Regime is unknown.")

if __name__ == "__main__":
    test_hybrid_flow()
