import sys
import os
import json
import time

sys.path.append(os.getcwd())

from backtest.ict_full_backtest import ICTFullBacktest
from live_trader import get_all_usdt_pairs, CORE_V2_SYMBOL_LIMIT
import traceback
from datetime import datetime
import pandas as pd

class UserStrategyBacktest(ICTFullBacktest):
    # Override constraints to allow fixed size
    FIXED_MARGIN = 200.0

    def _open_trade(self, symbol: str, direction: str, entry_price: float,
                    sl_price: float, tp_price: float,
                    bar_idx: int, timestamp: str) -> bool:
        # Fixed margin
        margin = self.FIXED_MARGIN
        
        # Check if we have enough balance
        if margin > self.balance:
            return False
            
        notional = margin * self.leverage
        qty = notional / entry_price
        
        if qty <= 0:
            return False

        self.balance -= margin
        self.trade_counter += 1

        self.current_trade = {
            'id': self.trade_counter,
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'initial_risk': abs(entry_price - sl_price),
            'qty': qty,
            'margin': margin,
            'entry_bar': bar_idx,
            'entry_time': timestamp,
            'max_profit_pct': 0.0,
            'peak_price': entry_price,
            'trail_active': False,
        }
        return True

def run_user_simulation():
    # symbols = get_all_usdt_pairs()[:CORE_V2_SYMBOL_LIMIT]
    # To not take hours, let's limit to 10 most liquid coins for the simulation, 180 days
    symbols = get_all_usdt_pairs()[:10]
    
    print(f"Running simulation for {len(symbols)} coins, 180 days, $1000 balance, 10x lev, $200 per trade.")
    
    results = []
    
    for idx, s in enumerate(symbols, 1):
        try:
            print(f"[{idx}/{len(symbols)}] Backtesting {s} for 180 days...")
            bt = UserStrategyBacktest(initial_balance=1000.0, leverage=10)
            r = bt.run_backtest(s, days=180)
            r['symbol'] = s
            results.append(r)
        except Exception as e:
            results.append({'symbol': s, 'success': False, 'error': str(e)})

    oks = [x for x in results if x.get('success')]
    
    # Portfolio replay combining all trades chronologically
    trades = []
    for r in oks:
        sym = r.get('symbol')
        for t in r.get('trades', []):
            try:
                et = datetime.fromisoformat(str(t.get('entry_time')).replace('Z',''))
                xt = datetime.fromisoformat(str(t.get('exit_time')).replace('Z',''))
            except Exception:
                continue
            m = float(t.get('margin', 0) or 0)
            p = float(t.get('pnl', 0) or 0)
            if m <= 0:
                continue
            trades.append({'symbol': sym, 'entry_time': et, 'exit_time': xt, 'margin': m, 'pnl': p, 'dir': t.get('direction', '')})

    # Sort chronological
    trades.sort(key=lambda x: (x['entry_time'], x['exit_time']))
    
    balance = 1000.0
    open_pos = []
    accepted = 0
    skipped = 0
    peak = balance
    max_dd = 0.0
    total_wins = 0
    total_losses = 0

    for tr in trades:
        now = tr['entry_time']
        
        # Close positions that exited before 'now'
        still = []
        for op in open_pos:
            if op['exit_time'] <= now:
                balance += op['margin'] + op['pnl']
                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
                if op['pnl'] > 0:
                    total_wins += 1
                else:
                    total_losses += 1
            else:
                still.append(op)
        open_pos = still

        if balance < 200.0:
            skipped += 1
            continue

        # Open new trade
        balance -= 200.0
        open_pos.append({
            'exit_time': tr['exit_time'],
            'margin': 200.0, 
            'pnl': (tr['pnl'] / tr['margin']) * 200.0 # scale PnL just in case margin varied, though it shouldn't
        })
        accepted += 1

    # Close remaining
    for op in sorted(open_pos, key=lambda x: x['exit_time']):
        balance += op['margin'] + op['pnl']
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        if op['pnl'] > 0:
            total_wins += 1
        else:
            total_losses += 1

    total_pnl = balance - 1000.0
    win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
    
    report = []
    report.append("==================================================")
    report.append("         SIMULATION REPORT (180 DAYS)          ")
    report.append("==================================================")
    report.append(f"Parametreler:")
    report.append(f"- Süre: 180 Gün")
    report.append(f"- Başlangıç Bakiyesi: $1000")
    report.append(f"- Kaldıraç: 10x")
    report.append(f"- İşlem Başına Büyüklük: $200 Margin ($2000 Notional)")
    report.append(f"- Sembol Sayısı: {len(symbols)} En Yüksek Hacimli Coin")
    report.append("--------------------------------------------------")
    report.append(f"Sonuçlar:")
    report.append(f"- Toplam İşlem Sayısı: {accepted} (Yetersiz Bakiye Nedeniyle Atlanan: {skipped})")
    report.append(f"- Başarılı İşlem: {total_wins}")
    report.append(f"- Başarısız İşlem: {total_losses}")
    report.append(f"- Kazanma Oranı: %{win_rate:.2f}")
    report.append(f"- Maksimum Düşüş (Drawdown): %{max_dd*100:.2f}")
    report.append("--------------------------------------------------")
    report.append(f"- Başlangıç Bakiyesi: $1000.00")
    report.append(f"- Bitiş Bakiyesi: ${balance:.2f}")
    report.append(f"- Net Kar/Zarar (PnL): ${total_pnl:.2f} (%{(total_pnl/1000)*100:.2f})")
    report.append("==================================================")

    out_str = "\n".join(report)
    print(out_str)
    
    with open("180_day_simulation_report.txt", "w", encoding="utf-8") as f:
        f.write(out_str)

if __name__ == "__main__":
    run_user_simulation()
