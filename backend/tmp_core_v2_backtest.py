import json
import time
import traceback
from live_trader import get_all_usdt_pairs, CORE_V2_SYMBOL_LIMIT
from backtest.ict_full_backtest import full_backtest_ict

symbols = get_all_usdt_pairs()[:CORE_V2_SYMBOL_LIMIT]
results = []
t0 = time.time()
print(f'START symbols={len(symbols)} days=180')

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

out = {
    'meta': {
        'days': 180,
        'initial_balance': 1000.0,
        'leverage': 10,
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    },
    'summary': summary,
    'results': results,
}

path = 'logs/core_v2_backtest_180d.json'
with open(path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print('DONE', path, summary)
