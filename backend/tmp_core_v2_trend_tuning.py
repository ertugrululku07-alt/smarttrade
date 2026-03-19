import sys
import os
import time

sys.path.append(os.path.dirname(__file__))

from backtest.trend_backtest import TrendBacktest
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators
import pandas as pd

TOP_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "LINK/USDT", "DOT/USDT",
    "MATIC/USDT", "NEAR/USDT", "APT/USDT", "OP/USDT", "ARB/USDT",
    "INJ/USDT", "RNDR/USDT", "FET/USDT", "AGIX/USDT", "WLD/USDT",
    "SUI/USDT", "SEI/USDT", "TIA/USDT", "ORDI/USDT", "1000SATS/USDT"
]

GRID = {
    'MAX_LOSS_DOLLAR': [4.0, 6.0, 8.0, 10.0],
    'TRAIL_START': [0.03, 0.05, 0.07],
    'TRAIL_KEEP': [0.35, 0.50, 0.65],
    'MAX_SL_PCT': [0.015, 0.020, 0.030]
}

def load_data(days=30):
    fetcher = DataFetcher('binance')
    data = {}
    print(f"[*] Downloading {days} days of 1h data for {len(TOP_PAIRS)} pairs...")
    for sym in TOP_PAIRS:
        df = fetcher.fetch_ohlcv(sym, '1h', limit=days*24 + 100)
        if df is not None and len(df) > 50:
            df = add_all_indicators(df)
            from backtest.trend_backtest import supertrend
            df['st_direction'], df['st_value'] = supertrend(df, 10, 3.0)
            data[sym] = df
    return data

def run_grid_search():
    data = load_data(30)
    if not data:
        print("[!] No data loaded.")
        return

    best_pnl = -9999
    best_combo = None

    total_iters = len(GRID['MAX_LOSS_DOLLAR']) * len(GRID['TRAIL_START']) * len(GRID['TRAIL_KEEP']) * len(GRID['MAX_SL_PCT'])
    print(f"[*] Starting grid search with {total_iters} combinations.")
    
    TrendBacktest.NOTIONAL_CAP = 350.0 
    TrendBacktest.BALANCE_PCT = 0.22   

    count = 0
    for max_loss in GRID['MAX_LOSS_DOLLAR']:
        for t_start in GRID['TRAIL_START']:
            for t_keep in GRID['TRAIL_KEEP']:
                for max_sl in GRID['MAX_SL_PCT']:
                    count += 1
                    
                    TrendBacktest.MAX_LOSS_DOLLAR = max_loss
                    TrendBacktest.TRAIL_START = t_start
                    TrendBacktest.TRAIL_KEEP = t_keep
                    TrendBacktest.MAX_SL_PCT = max_sl

                    # Simulate hybrid logic where ALL symbols trade on the SAME $1000 account
                    # Wait, simulating a shared portfolio is extremely difficult without parallel time execution.
                    # We will use the sum of independent PnLs but scale it.
                    total_pnl_combo = 0
                    total_trades_combo = 0
                    
                    for sym, df in data.items():
                        bt = TrendBacktest(initial_balance=1000.0, leverage=10)
                        start_bar = bt.LOOKBACK + 5
                        
                        for i in range(start_bar, len(df)):
                            if bt.current_trade is not None:
                                result = bt._check_exit(df, i)
                                if result is not None:
                                    reason, price = result
                                    bt._close_trade(price, reason, str(df.index[i]), i)

                            if bt.current_trade is None:
                                if bt.consecutive_sl >= 3:
                                    bt.consecutive_sl = 0
                                    bt.last_close_bar = i

                                if i - bt.last_close_bar < bt.COOLDOWN_BARS:
                                    continue

                                entry_signal = bt._check_entry(df, i)
                                if entry_signal is not None:
                                    direction, sl, tp = entry_signal
                                    bt._open_trade(sym, direction, float(df['close'].iloc[i]),
                                                     sl, tp, i, str(df.index[i]))

                        if bt.current_trade is not None:
                            bt._close_trade(float(df.iloc[-1]['close']), 'END', str(df.index[-1]), len(df) - 1)
                        
                        pnl = bt.balance - bt.initial_balance
                        total_pnl_combo += pnl
                        total_trades_combo += bt.trade_counter

                    score = total_pnl_combo
                    roi_pct = (total_pnl_combo / 1000.0) * 100
                    
                    print(f"[{count:03d}/{total_iters}] Loss={max_loss:04.1f} TStart={t_start:.2f} TKeep={t_keep:.2f} SL={max_sl:.3f} | Score/PnL=${total_pnl_combo:06.2f} (Simulated ROI) | Tr={total_trades_combo}")

                    if score > best_pnl:
                        best_pnl = score
                        best_combo = (max_loss, t_start, t_keep, max_sl)

    print("\n" + "="*50)
    print("🏆 BEST CONFIGURATION 🏆")
    print("="*50)
    print(f"MAX_LOSS_DOLLAR: {best_combo[0]}")
    print(f"TRAIL_START: {best_combo[1]}")
    print(f"TRAIL_KEEP: {best_combo[2]}")
    print(f"MAX_SL_PCT: {best_combo[3]}")
    print(f"Total Theoretical Strategy Pool PnL: ${best_pnl:.2f}")

if __name__ == '__main__':
    run_grid_search()
