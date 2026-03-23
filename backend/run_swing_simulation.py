import sys, os
sys.path.append(os.getcwd())
from backtest.swing_daily_backtest import SwingDailyBacktest
from live_trader import get_all_usdt_pairs

def main():
    symbols = get_all_usdt_pairs()[:10]
    out = []
    def log(msg): out.append(str(msg)); print(msg)
        
    log(f"Running Daily Swing Engine simulation for {len(symbols)} coins, 365 days")
    
    all_trades = []
    
    for idx, s in enumerate(symbols, 1):
        try:
            bt = SwingDailyBacktest()
            res = bt.run_backtest(s, days=365)
            if not res or not res.get('success'):
                continue
            
            trades = res.get('trades', [])
            for t in trades:
                m = float(t.get('margin', 0) or 0)
                pnl = float(t.get('pnl', 0) or 0)
                if m > 0:
                    all_trades.append({
                        'symbol': t['symbol'],
                        'entry_time': t['entry_time'], 'exit_time': t['exit_time'],
                        'margin': m, 'pnl': pnl,
                        'direction': t['direction'],
                        'entry': t['entry_price'], 'exit': t['exit_price'],
                        'orig_pnl': pnl
                    })
        except Exception as e:
            pass

    all_trades.sort(key=lambda x: str(x['entry_time']))
    log(f"\nCollected {len(all_trades)} trades.")

    balance = 1000.0
    wins = losses = 0
    peak = balance
    max_dd = 0.0
    
    log("\n--- TRADES ---")
    for tr in all_trades:
        p_pct = tr['orig_pnl'] / tr['margin'] * 100
        val = f"{tr['symbol']} {tr['direction']} Entry {str(tr['entry_time'])[:10]} Exit {str(tr['exit_time'])[:10]} ROE: {p_pct:+.1f}% PnL: ${tr['pnl']:.2f}"
        log(val)

        balance += tr['pnl']
        if balance > peak: peak = balance
        dd = (peak - balance) / peak if peak > 0 else 0
        if dd > max_dd: max_dd = dd
        
        if tr['pnl'] > 0: wins += 1
        else: losses += 1
        
    total = wins + losses
    
    log(f"\n==================================================")
    log(f"   SWING DAILY (1G) TREND ENGINE (365 DAYS)   ")
    log(f"==================================================")
    log(f"Parametreler:")
    log(f"- Baslangic Bakiyesi: $1000")
    log(f"- Islem Basina Buyukluk: $200 Notional (3x Kaldırac)")
    log(f"--------------------------------------------------")
    log(f"Sonuclar:")
    log(f"- Toplam Islem Sayisi: {total}")
    log(f"- Basarili: {wins} / Basarisiz: {losses}")
    wr = wins/total*100 if total else 0
    log(f"- Kazanma Orani: %{wr:.2f}")
    log(f"- Maksimum Dusus (Drawdown): %{max_dd*100:.2f}")
    log(f"--------------------------------------------------")
    log(f"- Baslangic Bakiyesi: $1000.00")
    log(f"- Bitis Bakiyesi: ${balance:.2f}")
    log(f"- Net Kar/Zarar (PnL): ${balance-1000:.2f} (%{(balance-1000)/10:.2f})")
    log(f"==================================================")
    
    with open('swing_result.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    
if __name__ == '__main__':
    main()
