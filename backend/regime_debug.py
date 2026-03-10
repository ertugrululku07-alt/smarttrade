# regime_debug.py olarak kaydet
import pandas as pd
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators
from ai.xgboost_trainer import generate_features
from ai.regime_detector import detect_regime, Regime

fetcher = DataFetcher('binance')
coins = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT']

for sym in coins:
    df = fetcher.fetch_ohlcv(sym, '15m', limit=500)
    df = add_all_indicators(df)
    df = generate_features(df)
    
    regime, details = detect_regime(df, lookback=50)
    
    adx = details.get('adx', 0)
    hurst = details.get('hurst', 0)
    atr_rank = details.get('atr_rank', 0)
    scores = details.get('scores', {})
    
    print(f"\n{'─'*50}")
    print(f"{sym}: {regime.value}")
    print(f"  ADX={adx:.1f} Hurst={hurst:.3f} ATR_rank={atr_rank:.3f}")
    print(f"  Scores: {scores}")