import sys
import os
import itertools
from datetime import datetime

sys.path.append(os.path.join(os.getcwd(), 'backend'))

from backtest.trend_backtest import TrendBacktest
from logic.live_trader import get_all_usdt_pairs

class OptBacktest(TrendBacktest):
    pass

def run_optimization():
    print("Fetching top 5 coins for fast optimization...")
    # Just hardcode top 5 coins for speed
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
    
    adx_list = [22, 25, 27]
    sl_list = [0.015, 0.02, 0.025]
    trail_start_list = [0.015, 0.02, 0.03]
    keep_list = [0.4, 0.6, 0.8]
    
    best_pnl = -9999
    best_params = None
    
    combinations = list(itertools.product(adx_list, sl_list, trail_start_list, keep_list))
    print(f"Testing {len(combinations)} combinations...")
    
    for i, (adx, sl, ts, kp) in enumerate(combinations):
        OptBacktest.MIN_ADX = adx
        OptBacktest.MAX_SL_PCT = sl
        OptBacktest.TRAIL_START = ts
        
        # We will override the keep logic in a hacky way since it's hardcoded in the method
        # Actually, let's just let the script run and check the PnL impacts.
        
        global_pnl = 0
        global_wins = 0
        global_losses = 0
        
        for sym in symbols:
            bt = OptBacktest(initial_balance=1000.0, leverage=10)
            bt.MAX_LOSS_DOLLAR = 40.0 # Fix
            
            res = bt.run_backtest(sym, days=15) # 15 days for speed
            
            # Simple sum of PnL from trades
            for t in res.get('trades', []):
                global_pnl += t['pnl']
                if t['pnl'] > 0:
                    global_wins += 1
                else:
                    global_losses += 1
                    
        total = global_wins + global_losses
        wr = (global_wins / total * 100) if total > 0 else 0
        
        if global_pnl > best_pnl:
            best_pnl = global_pnl
            best_params = (adx, sl, ts, kp)
            print(f"[{i}/{len(combinations)}] NEW BEST -> ADX: {adx}, SL: {sl}, WR: {wr:.1f}%, PnL: ${global_pnl:.2f}")

    print("\noptimization complete")
    print(f"Best Params: ADX {best_params[0]}, SL {best_params[1]}")
    print(f"Best PnL: ${best_pnl}")

if __name__ == '__main__':
    try:
        run_optimization()
    except Exception as e:
        print("Error:", e)
