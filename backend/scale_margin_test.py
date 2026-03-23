import sys, os
sys.path.append(os.getcwd())
from datetime import datetime
from backtest.hybrid_momentum_backtest import MomentumHybridBacktest
from live_trader import get_all_usdt_pairs

def run_portfolio_sim(all_trades, margin, max_conc, cb_sl, cb_cd):
    balance = 1000.0
    open_pos = []
    peak = balance
    max_dd = 0.0
    
    for tr in all_trades:
        now = tr['entry_time']
        
        still = []
        for op in open_pos:
            if op['exit_time'] <= now:
                balance += op['margin'] + op['pnl']
                if balance > peak: peak = balance
                dd = (peak - balance) / peak if peak > 0 else 0
                if dd > max_dd: max_dd = dd
            else:
                still.append(op)
        open_pos = still
        
        if len(open_pos) >= max_conc or balance < margin: continue
        
        orig_m = tr['margin']
        if orig_m <= 0: continue
        scaled_pnl = (tr['pnl'] / orig_m) * margin
        
        balance -= margin
        open_pos.append({'exit_time': tr['exit_time'], 'margin': margin, 'pnl': scaled_pnl})
        
    for op in sorted(open_pos, key=lambda x: x['exit_time']):
        balance += op['margin'] + op['pnl']
        if balance > peak: peak = balance
        dd = (peak - balance) / peak if peak > 0 else 0
        if dd > max_dd: max_dd = dd
        
    return {'pnl_pct': (balance-1000)/10, 'max_dd': max_dd*100}

def main():
    symbols = get_all_usdt_pairs()[:10]
    all_trades = []
    
    print("Fetching trades for Optimal Hybrid (BP=15, VM=1.2, ADX=20)...")
    for s in symbols:
        bt = MomentumHybridBacktest()
        bt.BREAKOUT_PERIOD = 15; bt.VOL_MULT = 1.2; bt.MIN_ADX = 20
        res = bt.run_backtest(s, days=180)
        if res and res.get('success'):
            for t in res.get('trades', []):
                try:
                    et = datetime.fromisoformat(str(t['entry_time']).replace('Z',''))
                    xt = datetime.fromisoformat(str(t['exit_time']).replace('Z',''))
                    m = float(t.get('margin',0)); p = float(t.get('pnl',0))
                    if m > 0: all_trades.append({'entry_time': et, 'exit_time': xt, 'margin': m, 'pnl': p})
                except: pass

    all_trades.sort(key=lambda x: x['entry_time'])
    print(f"Collected {len(all_trades)} trades. Testing Margin Scaling:")
    
    for margin in [50, 100, 150, 200, 250]:
        res = run_portfolio_sim(all_trades, margin, max_conc=3, cb_sl=3, cb_cd=24)
        print(f"Margin: ${margin} -> Account PnL: {res['pnl_pct']:+.1f}% | Max Drawdown: {res['max_dd']:.1f}%")

if __name__ == '__main__':
    main()
