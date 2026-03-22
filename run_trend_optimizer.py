import sys
import os
import itertools
from datetime import datetime

sys.path.append(os.getcwd())

from run_trend_simulation import UserTrendBacktest
from backend.live_trader import get_all_usdt_pairs

def run_optimization():
    # symbols = get_all_usdt_pairs()[:5]
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
    
    # Grid Search space
    adx_list = [20, 25]
    sl_list = [0.015, 0.02, 0.03, 0.04]
    trail_list = [(0.015, 0.4), (0.025, 0.6), (0.04, 0.8)] # (start, keep)
    
    combinations = list(itertools.product(adx_list, sl_list, trail_list))
    print(f"Testing {len(combinations)} combinations across 5 coins for 30 days...")
    
    best_pnl = -9999
    best_params = None
    
    for idx, (adx, sl, (t_start, t_keep)) in enumerate(combinations):
        
        global_pnl = 0
        global_wins = 0
        global_losses = 0
        
        for sym in symbols:
            # We override class level to inject custom params dynamically
            UserTrendBacktest.MIN_ADX = adx
            UserTrendBacktest.MAX_SL_PCT = sl
            
            bt = UserTrendBacktest(initial_balance=1000.0, leverage=10)
            bt.TRAIL_START = t_start
            bt.TRAIL_KEEP = t_keep
            bt.MAX_LOSS_DOLLAR = 40.0
            
            # The backtester creates a 'dummy' instance sometimes. Let's just run it!
            res = bt.run_backtest(sym, days=30)
            
            if res and 'trades' in res:
                for t in res['trades']:
                    trade_pnl_dollar = (t['pnl'] / t['margin']) * 200.0 if t.get('margin', 0) > 0 else 0
                    global_pnl += trade_pnl_dollar
                    if trade_pnl_dollar > 0:
                        global_wins += 1
                    else:
                        global_losses += 1
                
        total = global_wins + global_losses
        wr = (global_wins / total * 100) if total > 0 else 0
        
        print(f"[{idx+1}/{len(combinations)}] ADX:{adx} SL:{sl} Trail:{t_start}|{t_keep} -> PnL: ${global_pnl:.2f} (WR: {wr:.1f}%)")
        
        if global_pnl > best_pnl:
            best_pnl = global_pnl
            best_params = {'adx': adx, 'sl': sl, 'start': t_start, 'keep': t_keep}
            
    print("\n=== OPTIMAL PARAMETERS FOUND ===")
    print(best_params)
    print(f"Best PnL: ${best_pnl:.2f}")

if __name__ == '__main__':
    run_optimization()
