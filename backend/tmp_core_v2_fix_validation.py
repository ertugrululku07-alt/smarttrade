import json
import time
import traceback
from datetime import datetime
from live_trader import get_all_usdt_pairs
from backtest.ict_full_backtest import ICTFullBacktest, full_backtest_ict

# Keep same portfolio preset for apples-to-apples comparison
ICTFullBacktest.MIN_RR = 1.45
ICTFullBacktest.MAX_LOSS_DOLLAR = 5.8
ICTFullBacktest.NOTIONAL_CAP = 300.0
ICTFullBacktest.BALANCE_PCT = 0.17
ICTFullBacktest.POI_PROXIMITY_ATR = 1.8
ICTFullBacktest.COOLDOWN_BARS = 3
ICTFullBacktest.TRAIL_START_PCT = 0.020
ICTFullBacktest.TRAIL_DIST_PCT = 0.006

symbols = get_all_usdt_pairs()[:50]
results = []
t0 = time.time()
print(f'START ict_fix_validation symbols={len(symbols)} days=120')

for i, s in enumerate(symbols, 1):
    try:
        r = full_backtest_ict(s, days=120, initial_balance=1000.0, leverage=10)
    except Exception as e:
        r = {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
    r['symbol'] = s
    results.append(r)
    print(f"[{i}/{len(symbols)}] {s} ok={r.get('success', False)} pnl={r.get('total_pnl', None)} trades={r.get('total_trades', None)}")

oks = [x for x in results if x.get('success')]
trades = []
for r in oks:
    sym = r.get('symbol')
    for t in r.get('trades', []):
        t2 = dict(t)
        t2['symbol'] = sym
        trades.append(t2)

# Core failure attribution metrics
loss = [t for t in trades if float(t.get('pnl', 0) or 0) <= 0]
wins = [t for t in trades if float(t.get('pnl', 0) or 0) > 0]
sl_losses = [t for t in loss if str(t.get('exit_reason', '')).upper() in ('SL', 'ICT_SL', 'ICT_MAXLOSS')]

long_trades = [t for t in trades if t.get('direction') == 'LONG']
short_trades = [t for t in trades if t.get('direction') == 'SHORT']
long_wins = [t for t in long_trades if float(t.get('pnl', 0) or 0) > 0]
short_wins = [t for t in short_trades if float(t.get('pnl', 0) or 0) > 0]

long_wr = (len(long_wins) / len(long_trades) * 100) if long_trades else 0
short_wr = (len(short_wins) / len(short_trades) * 100) if short_trades else 0

# Portfolio replay (shared balance, max 5 concurrent)
pt = []
for t in trades:
    try:
        et = datetime.fromisoformat(str(t.get('entry_time')).replace('Z', ''))
        xt = datetime.fromisoformat(str(t.get('exit_time')).replace('Z', ''))
    except Exception:
        continue
    m = float(t.get('margin', 0) or 0)
    p = float(t.get('pnl', 0) or 0)
    if m <= 0:
        continue
    pt.append({'entry_time': et, 'exit_time': xt, 'margin': m, 'pnl': p})

pt.sort(key=lambda x: (x['entry_time'], x['exit_time']))
balance = 1000.0
peak = balance
max_dd = 0.0
open_pos = []

for tr in pt:
    now = tr['entry_time']
    still = []
    for op in open_pos:
        if op['exit_time'] <= now:
            balance += op['margin_eff'] + op['pnl_eff']
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        else:
            still.append(op)
    open_pos = still

    if len(open_pos) >= 5:
        continue

    scale = max(0.0, balance / 1000.0)
    margin_eff = tr['margin'] * scale
    pnl_eff = tr['pnl'] * scale
    if margin_eff <= 0 or margin_eff > balance:
        continue

    balance -= margin_eff
    open_pos.append({'exit_time': tr['exit_time'], 'margin_eff': margin_eff, 'pnl_eff': pnl_eff})

for op in sorted(open_pos, key=lambda x: x['exit_time']):
    balance += op['margin_eff'] + op['pnl_eff']
    if balance > peak:
        peak = balance
    dd = (peak - balance) / peak if peak > 0 else 0
    if dd > max_dd:
        max_dd = dd

summary = {
    'symbols_total': len(symbols),
    'symbols_ok': len(oks),
    'total_trades': len(trades),
    'wins': len(wins),
    'losses': len(loss),
    'overall_wr': round((len(wins) / len(trades) * 100), 2) if trades else 0,
    'sl_loss_ratio_pct': round((len(sl_losses) / len(loss) * 100), 2) if loss else 0,
    'long_wr': round(long_wr, 2),
    'short_wr': round(short_wr, 2),
    'long_short_wr_gap': round(abs(long_wr - short_wr), 2),
    'portfolio_final_balance': round(balance, 2),
    'portfolio_return_pct': round((balance / 1000.0 - 1) * 100, 2),
    'max_dd_pct': round(max_dd * 100, 2),
    'elapsed_sec': round(time.time() - t0, 1),
}

out = {
    'meta': {'days': 120, 'symbols': 50, 'preset': 'A_balanced_aggr_with_entry_direction_fixes'},
    'summary': summary,
    'results': results,
}

path = 'logs/core_v2_fix_validation_120d_top50.json'
with open(path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print('DONE', path, summary)
