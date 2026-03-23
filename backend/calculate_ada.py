import sys, os
sys.path.append(os.getcwd())
from backtest.data_fetcher import DataFetcher
import pandas as pd

def main():
    fetcher = DataFetcher('binance')
    # Fetch enough hourly data to cover Jan 2026 to Feb 2026
    # 60 days * 24 hours = 1440. Let's fetch 1500 to be safe
    df = fetcher.fetch_ohlcv('ADA/USDT', '1d', limit=100)
    
    if df is None:
        print("Failed to fetch data")
        return
        
    print(df.tail(10))
    
    # Let's find Jan 14 and Feb 5
    try:
        entry_row = df.loc[df.index.strftime('%Y-%m-%d') == '2026-01-14']
        exit_row = df.loc[df.index.strftime('%Y-%m-%d') == '2026-02-05']
        
        if entry_row.empty or exit_row.empty:
            print("Could not find exacta dates. Here are the dates available:")
            print(df.index.strftime('%Y-%m-%d').tolist()[-30:])
            return
            
        entry_price = float(entry_row['open'].values[0])
        exit_price = float(exit_row['close'].values[0])
        
        days = (exit_row.index[0] - entry_row.index[0]).days
        
        pnl_pct = ((entry_price - exit_price) / entry_price) * 100
        leveraged_pnl = pnl_pct * 10
        
        print(f"Entry Date: 2026-01-14 | Entry Price (Open): {entry_price}")
        print(f"Exit Date:  2026-02-05 | Exit Price (Close): {exit_price}")
        print(f"Duration: {days} days")
        print(f"Unleveraged Profit: {pnl_pct:.2f}%")
        print(f"10x Leveraged Profit: {leveraged_pnl:.2f}%")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
