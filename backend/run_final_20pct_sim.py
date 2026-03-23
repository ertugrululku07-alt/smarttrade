import sys, os
sys.path.append(os.getcwd())
from backtest.hybrid_momentum_backtest import MomentumHybridBacktest
from live_trader import get_all_usdt_pairs
from datetime import datetime, timedelta

def main():
    symbols = get_all_usdt_pairs()[:10]
    all_trades = []
    
    print("Collecting trades for Optimal Hybrid (BP=15, VM=1.2, ADX>=20) over 180 days...")
    for idx, s in enumerate(symbols, 1):
        try:
            bt = MomentumHybridBacktest()
            bt.BREAKOUT_PERIOD = 15
            bt.VOL_MULT = 1.2
            bt.MIN_ADX = 20
            res = bt.run_backtest(s, days=180)
            if not res or not res.get('success'):
                continue
            
            trades = res.get('trades', [])
            for t in trades:
                try:
                    et = datetime.fromisoformat(str(t['entry_time']).replace('Z', ''))
                    xt = datetime.fromisoformat(str(t['exit_time']).replace('Z', ''))
                    
                    # Original simulation generated PnL based on a calculated qty
                    m = float(t.get('margin', 0) or 0)
                    pnl = float(t.get('pnl', 0) or 0)
                    if m > 0:
                        all_trades.append({
                            'symbol': t['symbol'],
                            'entry_time': et, 'exit_time': xt,
                            'margin': m, 'pnl': pnl,
                            'direction': t['direction'],
                            'entry': t['entry_price'], 'exit': t['exit_price']
                        })
                except Exception as e:
                    pass
        except Exception as e:
            pass

    all_trades.sort(key=lambda x: (x['entry_time'], x['exit_time']))
    print(f"Collected {len(all_trades)} trades. Replaying with $200 Margin (20%)...")

    balance = 1000.0
    margin = 200.0
    max_conc = 3
    cb_sl = 3
    cb_cd = 24
    
    open_pos = []
    peak = balance
    max_dd = 0.0
    wins = losses = consec_sl = 0
    cd_until = None
    
    for tr in all_trades:
        now = tr['entry_time']
        
        still = []
        for op in open_pos:
            if op['exit_time'] <= now:
                balance += op['margin'] + op['pnl']
                if balance > peak: peak = balance
                dd = (peak - balance) / peak if peak > 0 else 0
                if dd > max_dd: max_dd = dd
                
                if op['pnl'] > 0:
                    wins += 1; consec_sl = 0
                else:
                    losses += 1; consec_sl += 1
                    if consec_sl >= cb_sl: cd_until = op['exit_time']
            else:
                still.append(op)
        open_pos = still
        
        if cd_until:
            if now < cd_until + timedelta(hours=cb_cd): continue
            cd_until = None; consec_sl = 0
            
        if len(open_pos) >= max_conc or balance < margin: continue
        
        # Scale PnL based on the new $200 Margin instead of original M
        scaled_pnl = (tr['pnl'] / tr['margin']) * margin
        
        balance -= margin
        open_pos.append({'exit_time': tr['exit_time'], 'margin': margin, 'pnl': scaled_pnl})
        
    for op in sorted(open_pos, key=lambda x: x['exit_time']):
        balance += op['margin'] + op['pnl']
        if balance > peak: peak = balance
        dd = (peak - balance) / peak if peak > 0 else 0
        if dd > max_dd: max_dd = dd
        if op['pnl'] > 0: wins += 1
        else: losses += 1
        
    total = wins + losses
    
    print(f"\n==================================================")
    print(f"   OPTIMAL HYBRID ENGINE (180 DAYS) - 20% RISK   ")
    print(f"==================================================")
    print(f"Parametreler:")
    print(f"- Baslangic Bakiyesi: $1000")
    print(f"- Islem Basina Buyukluk: ${margin} Margin (%20)")
    print(f"- Max Esanli Islem: {max_conc}")
    print(f"- Devre Kesici: {cb_sl} Ardisik SL -> {cb_cd}s Bekleme")
    print(f"--------------------------------------------------")
    print(f"Sonuclar:")
    print(f"- Toplam Islem Sayisi: {total} (Orijinal Sinyal: {len(all_trades)})")
    print(f"- Basarili: {wins} / Basarisiz: {losses}")
    wr = wins/total*100 if total else 0
    print(f"- Kazanma Orani: %{wr:.2f}")
    print(f"- Maksimum Dusus (Drawdown): %{max_dd*100:.2f}")
    print(f"--------------------------------------------------")
    print(f"- Baslangic Bakiyesi: $1000.00")
    print(f"- Bitis Bakiyesi: ${balance:.2f}")
    print(f"- Net Kar/Zarar (PnL): ${balance-1000:.2f} (%{(balance-1000)/10:.2f})")
    print(f"==================================================")
    
if __name__ == '__main__':
    main()
