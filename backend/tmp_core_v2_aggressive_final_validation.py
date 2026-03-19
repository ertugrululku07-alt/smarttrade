import json
import time
import traceback
from datetime import datetime
from live_trader import get_all_usdt_pairs, CORE_V2_SYMBOL_LIMIT
from backtest.ict_full_backtest import ICTFullBacktest, full_backtest_ict

# Final validation preset = A_balanced_aggr
ICTFullBacktest.MIN_RR = 1.45
ICTFullBacktest.MAX_LOSS_DOLLAR = 5.8
ICTFullBacktest.NOTIONAL_CAP = 300.0
ICTFullBacktest.BALANCE_PCT = 0.17
ICTFullBacktest.POI_PROXIMITY_ATR = 1.8
ICTFullBacktest.COOLDOWN_BARS = 3
ICTFullBacktest.TRAIL_START_PCT = 0.020
ICTFullBacktest.TRAIL_DIST_PCT = 0.006

symbols = get_all_usdt_pairs()[:CORE_V2_SYMBOL_LIMIT]
results = []
t0 = time.time()
print(f'START final_balanced_aggr symbols={len(symbols)} days=180')

for i, s in enumerate(symbols, 1):
    try:
        r = full_backtest_ict(s, days=180, initial_balance=1000.0, leverage=10)
    except Exception as e:
        r = {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}

    r['symbol'] = s
    results.append(r)
    print(f"[{i}/{len(symbols)}] {s} ok={r.get('success', False)} pnl={r.get('total_pnl', None)} trades={r.get('total_trades', None)}")

oks = [x for x in results if x.get('success')]
pnls = [float(x.get('total_pnl', 0) or 0) for x in oks]
trs = [int(x.get('total_trades', 0) or 0) for x in oks]
prof = [x for x in oks if float(x.get('total_pnl', 0) or 0) > 0]

# Portfolio replay (shared balance, max 5 concurrent)
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
        trades.append({'symbol': sym, 'entry_time': et, 'exit_time': xt, 'margin': m, 'pnl': p})

trades.sort(key=lambda x: (x['entry_time'], x['exit_time']))
balance = 1000.0
open_pos = []
accepted = 0
skipped = 0
peak = balance
max_dd = 0.0
monthly = {}


def stamp_month(dt):
    return f"{dt.year:04d}-{dt.month:02d}"

for tr in trades:
    now = tr['entry_time']
    still = []
    for op in open_pos:
        if op['exit_time'] <= now:
            balance += op['margin_eff'] + op['pnl_eff']
            m = stamp_month(op['exit_time'])
            monthly.setdefault(m, {'start': None, 'end': None})
            monthly[m]['end'] = balance
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        else:
            still.append(op)
    open_pos = still

    if len(open_pos) >= 5:
        skipped += 1
        continue

    scale = max(0.0, balance / 1000.0)
    margin_eff = tr['margin'] * scale
    pnl_eff = tr['pnl'] * scale
    if margin_eff <= 0 or margin_eff > balance:
        skipped += 1
        continue

    em = stamp_month(tr['entry_time'])
    monthly.setdefault(em, {'start': None, 'end': None})
    if monthly[em]['start'] is None:
        monthly[em]['start'] = balance

    balance -= margin_eff
    open_pos.append({'exit_time': tr['exit_time'], 'margin_eff': margin_eff, 'pnl_eff': pnl_eff})
    accepted += 1

for op in sorted(open_pos, key=lambda x: x['exit_time']):
    balance += op['margin_eff'] + op['pnl_eff']
    m = stamp_month(op['exit_time'])
    monthly.setdefault(m, {'start': None, 'end': None})
    monthly[m]['end'] = balance
    if balance > peak:
        peak = balance
    dd = (peak - balance) / peak if peak > 0 else 0
    if dd > max_dd:
        max_dd = dd

monthly_returns = {}
for k, v in monthly.items():
    s = v.get('start')
    e = v.get('end')
    if s and e:
        monthly_returns[k] = round((e / s - 1) * 100, 2)

summary = {
    'symbols_total': len(symbols),
    'symbols_ok': len(oks),
    'symbols_profitable': len(prof),
    'total_pnl': round(sum(pnls), 2),
    'avg_pnl_per_symbol': round((sum(pnls) / len(oks)), 2) if oks else 0,
    'total_trades': int(sum(trs)),
    'avg_trades_per_symbol': round((sum(trs) / len(oks)), 2) if oks else 0,
    'elapsed_sec': round(time.time() - t0, 1),
}

portfolio = {
    'initial_balance': 1000.0,
    'final_balance': round(balance, 2),
    'total_pnl': round(balance - 1000.0, 2),
    'return_pct': round((balance / 1000.0 - 1) * 100, 2),
    'max_drawdown_pct': round(max_dd * 100, 2),
    'accepted_trades': accepted,
    'skipped_trades': skipped,
    'raw_trades': len(trades),
    'monthly_returns_pct': monthly_returns,
}

out = {
    'meta': {
        'preset': 'core_v2_aggressive_A_balanced_aggr_final',
        'days': 180,
        'initial_balance': 1000.0,
        'leverage': 10,
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'overrides': {
            'MIN_RR': ICTFullBacktest.MIN_RR,
            'MAX_LOSS_DOLLAR': ICTFullBacktest.MAX_LOSS_DOLLAR,
            'NOTIONAL_CAP': ICTFullBacktest.NOTIONAL_CAP,
            'BALANCE_PCT': ICTFullBacktest.BALANCE_PCT,
            'POI_PROXIMITY_ATR': ICTFullBacktest.POI_PROXIMITY_ATR,
            'COOLDOWN_BARS': ICTFullBacktest.COOLDOWN_BARS,
            'TRAIL_START_PCT': ICTFullBacktest.TRAIL_START_PCT,
            'TRAIL_DIST_PCT': ICTFullBacktest.TRAIL_DIST_PCT,
            'portfolio_max_open': 5,
        }
    },
    'summary': summary,
    'portfolio': portfolio,
    'results': results,
}

path = 'logs/core_v2_aggressive_final_validation_180d.json'
with open(path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print('DONE', path, {'summary': summary, 'portfolio': portfolio})
