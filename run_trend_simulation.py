import sys
import os
import json
import time
from datetime import datetime

sys.path.append(os.getcwd())

from backtest.trend_backtest import TrendBacktest
from live_trader import get_all_usdt_pairs

class UserTrendBacktest(TrendBacktest):
    FIXED_MARGIN = 200.0

    def _open_trade(self, symbol: str, direction: str, entry_price: float,
                    sl_price: float, tp_price: float,
                    bar_idx: int, timestamp: str) -> bool:
        margin = self.FIXED_MARGIN
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
            'qty': qty,
            'margin': margin,
            'entry_bar': bar_idx,
            'entry_time': timestamp,
            'max_profit_pct': 0.0,
            'peak_price': entry_price,
            'trail_active': False,
        }
        return True

    def _build_results(self, symbol: str, days: int) -> dict:
        total = len(self.closed_trades)
        if total == 0:
            return {'success': True, 'symbol': symbol, 'timeframe': '1h', 'total_trades': 0, 'trades': []}
        
        wins = [t for t in self.closed_trades if t['pnl'] > 0]
        losses = [t for t in self.closed_trades if t['pnl'] <= 0]
        
        return {
            'success': True, 'symbol': symbol, 'timeframe': '1h',
            'days': days, 'total_trades': total, 'wins': len(wins), 'losses': len(losses),
            'trades': [
                {
                    'id': t['id'], 'direction': t['direction'],
                    'entry_price': round(t['entry_price'], 6),
                    'exit_price': round(t['exit_price'], 6),
                    'margin': round(t.get('margin', 0), 6),
                    'sl_price': round(t['sl_price'], 6),
                    'tp_price': round(t['tp_price'], 6),
                    'entry_time': t['entry_time'],
                    'exit_time': t['exit_time'],
                    'pnl': round(t['pnl'], 2),
                    'pnl_pct': round(t['pnl_pct'], 2),
                    'max_profit_pct': round(t.get('max_profit_pct', 0), 2),
                    'exit_reason': t['exit_reason']
                }
                for t in self.closed_trades
            ]
        }

def run_user_simulation():
    symbols = get_all_usdt_pairs()[:15]
    print(f"Running Trend Simulation Baseline for {len(symbols)} coins, 30 days, $1000 balance, 10x lev, $200 per trade.")
    
    results = []
    for idx, s in enumerate(symbols, 1):
        try:
            print(f"[{idx}/{len(symbols)}] Backtesting {s} for 30 days...")
            bt = UserTrendBacktest(initial_balance=1000.0, leverage=10)
            r = bt.run_backtest(s, days=30)
            r['symbol'] = s
            results.append(r)
        except Exception as e:
            results.append({'symbol': s, 'success': False, 'error': str(e)})

    oks = [x for x in results if x.get('success')]
    
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
            if m <= 0: continue
            trades.append({'symbol': sym, 'entry_time': et, 'exit_time': xt, 'margin': m, 'pnl': p, 'dir': t.get('direction', '')})

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

        balance -= 200.0
        open_pos.append({
            'exit_time': tr['exit_time'],
            'margin': 200.0, 
            'pnl': (tr['pnl'] / tr['margin']) * 200.0
        })
        accepted += 1

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
    report.append("         SIMULATION REPORT (30 DAYS)          ")
    report.append("==================================================")
    report.append(f"Sonuçlar:")
    report.append(f"- Toplam İşlem Sayısı: {accepted} (Atlanan: {skipped})")
    report.append(f"- Başarılı İşlem: {total_wins}")
    report.append(f"- Başarısız İşlem: {total_losses}")
    report.append(f"- Kazanma Oranı: %{win_rate:.2f}")
    report.append(f"- Maksimum Düşüş: %{max_dd*100:.2f}")
    report.append("--------------------------------------------------")
    report.append(f"- Başlangıç Bakiyesi: $1000.00")
    report.append(f"- Bitiş Bakiyesi: ${balance:.2f}")
    report.append(f"- Net Kar/Zarar (PnL): ${total_pnl:.2f} (%{(total_pnl/1000)*100:.2f})")
    report.append("==================================================")

    out_str = "\n".join(report)
    print(out_str)
    with open("trend_sim_report.txt", "w", encoding="utf-8") as f:
        f.write(out_str)

if __name__ == "__main__":
    run_user_simulation()
