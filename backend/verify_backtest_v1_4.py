
import os
import sys
import pandas as pd
from ai.adaptive_backtest import AdaptiveBacktest
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators

def run_verify():
    fetcher = DataFetcher('binance')
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'ADAUSDT', 'DOTUSDT']
    timeframe = '15m'
    limit = 4000
    initial_capital = 1000.0
    
    print(f"--- v1.4 Backtest Simulation ($1000 Capital) ---")
    print(f"Timeframe: {timeframe} | Limit: {limit} bars")
    
    results = []
    total_trades = 0
    total_pnl = 0.0
    wins = 0
    
    backtester = AdaptiveBacktest(timeframe=timeframe, use_meta_filter=True)
    
    for sym in symbols:
        try:
            df = fetcher.fetch_ohlcv(sym, timeframe, limit=limit)
            if df is None or len(df) < 100: continue
            
            df = add_all_indicators(df)
            
            res = backtester.run(df, symbol=sym)
            results.append((sym, res))
            
            total_trades += res.total_trades
            wins += res.winners
            total_pnl += res.total_pnl_pct
            
            print(f"  {sym:10} | Trades: {res.total_trades:3} | WR: {res.win_rate:5.1f}% | PnL: {res.total_pnl_pct:6.2f}%")
        except Exception as e:
            print(f"  {sym:10} | ERROR: {e}")
            
    print(f"--- Summary ---")
    print(f"Total Trades: {total_trades}")
    avg_wr = (wins / total_trades * 100) if total_trades > 0 else 0
    print(f"Average WR: {avg_wr:.2f}%")
    print(f"Total PnL: {total_pnl:.2f}%")

if __name__ == "__main__":
    run_verify()
