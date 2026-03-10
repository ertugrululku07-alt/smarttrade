import os
import sys
import pandas as pd

# Standardize encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        pass

# Add backend to path
sys.path.insert(0, os.getcwd())

try:
    from backtest.data_fetcher import DataFetcher
    from ai.adaptive_live_adapter import generate_signal, print_debug

    fetcher = DataFetcher('binance')
    test_coins = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
        'BNBUSDT', 'AVAXUSDT', 'LINKUSDT', 'ADAUSDT', 'DOTUSDT',
    ]

    print("\n  >>> Multi-Coin Adaptive AI Test (15m)")
    print(f"  {'-' * 60}")

    results = []
    for sym in test_coins:
        try:
            df = fetcher.fetch_ohlcv(sym, '15m', limit=500)
            if df is None or len(df) < 100:
                print(f"  SKIP {sym}: veri yok")
                continue
            
            signal = generate_signal(df, sym, '15m')
            
            status = "[SIGNAL]" if signal['signal'] != 'HOLD' else "[HOLD]"
            print(f"  {status} {sym:12} | {signal['signal']:5} | "
                  f"regime={signal['regime']:16} | "
                  f"conf={signal['confidence']:.2f} | "
                  f"reason={signal['reason']}")
            
            results.append(signal)
            
        except Exception as e:
            print(f"  ERROR {sym}: {e}")

    print(f"\n  Total: {len(results)} coins tested")
    signals = [r for r in results if r['signal'] != 'HOLD']
    print(f"  Signals: {len(signals)} / {len(results)}")
    print()
    print_debug()

except Exception as main_e:
    print(f"Main execution failed: {main_e}")
