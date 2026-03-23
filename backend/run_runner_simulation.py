import sys, os
sys.path.append(os.getcwd())
from backtest.runner_hybrid_backtest import RunnerBacktest
from live_trader import get_all_usdt_pairs
from datetime import datetime, timedelta

def main():
    symbols = get_all_usdt_pairs()[:10]
    out = []
    def log(msg):
        out.append(str(msg))
        
    log(f"Running Runner Engine simulation for {len(symbols)} coins, 180 days")
    
    all_trades = []
    
    for idx, s in enumerate(symbols, 1):
        try:
            bt = RunnerBacktest()
            res = bt.run_backtest(s, days=180)
            if not res or not res.get('success'):
                continue
            
            trades = res.get('trades', [])
            for t in trades:
                try:
                    et = datetime.fromisoformat(str(t['entry_time']).replace('Z', ''))
                    xt = datetime.fromisoformat(str(t['exit_time']).replace('Z', ''))
                    m = float(t.get('margin', 0) or 0)
                    pnl = float(t.get('pnl', 0) or 0)
                    if m > 0:
                        all_trades.append({
                            'symbol': t['symbol'],
                            'entry_time': et, 'exit_time': xt,
                            'margin': m, 'pnl': pnl,
                            'direction': t['direction'],
                            'entry': t['entry_price'],
                            'exit': t['exit_price'],
                            'orig_pnl': pnl
                        })
                except Exception as e:
                    pass
        except Exception as e:
            pass

    all_trades.sort(key=lambda x: (x['entry_time'], x['exit_time']))
    log(f"\nCollected {len(all_trades)} trades. Replaying with portfolio controls...")

    balance = 1000.0
    margin = 75.0
    max_conc = 3
    cb_sl = 3
    cb_cd = 24
    
    open_pos = []
    peak = balance
    max_dd = 0.0
    wins = losses = consec_sl = 0
    cd_until = None
    
    log("\n--- NOTABLE MEGA TRADES ---")
    for tr in all_trades:
        p_pct = tr['orig_pnl'] / tr['margin'] * 100
        if p_pct > 150:
            log(f"MEGA: {tr['symbol']} {tr['direction']} Entry {tr['entry_time'].strftime('%Y-%m-%d')} Exit {tr['exit_time'].strftime('%Y-%m-%d')} ROE: +{p_pct:.1f}%")

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
    
    log(f"\n==================================================")
    log(f"   LET WINNERS RIDE - RUNNER ENGINE (180 DAYS)   ")
    log(f"==================================================")
    log(f"Parametreler:")
    log(f"- Baslangic Bakiyesi: $1000")
    log(f"- Islem Basina Buyukluk: ${margin} Margin")
    log(f"- Max Esanli Islem: {max_conc}")
    log(f"- Devre Kesici: {cb_sl} Ardisik SL -> {cb_cd}s Bekleme")
    log(f"--------------------------------------------------")
    log(f"Sonuclar:")
    log(f"- Toplam Islem Sayisi: {total} (Tum Sinyaller: {len(all_trades)})")
    log(f"- Basarili: {wins} / Basarisiz: {losses}")
    wr = wins/total*100 if total else 0
    log(f"- Kazanma Orani: %{wr:.2f}")
    log(f"- Maksimum Dusus (Drawdown): %{max_dd*100:.2f}")
    log(f"--------------------------------------------------")
    log(f"- Baslangic Bakiyesi: $1000.00")
    log(f"- Bitis Bakiyesi: ${balance:.2f}")
    log(f"- Net Kar/Zarar (PnL): ${balance-1000:.2f} (%{(balance-1000)/10:.2f})")
    log(f"==================================================")
    
    with open('runner_result.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    
if __name__ == '__main__':
    main()
