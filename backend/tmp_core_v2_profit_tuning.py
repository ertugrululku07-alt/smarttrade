import json
import time
import traceback
from datetime import datetime
from live_trader import get_all_usdt_pairs
from backtest.ict_full_backtest import ICTFullBacktest, full_backtest_ict

symbols = get_all_usdt_pairs()[:30]

print("--- PRE-FETCHING DATA ---")
from backtest.data_fetcher import DataFetcher
import time
fetcher = DataFetcher('binance')
symbol_data = {}
for s in symbols:
    print(f"Fetching {s}...")
    time.sleep(1.0)
    try:
        df = fetcher.fetch_ohlcv(s, '1h', limit=max(120 * 24 + 100, 200))
        if df is not None and not df.empty:
            symbol_data[s] = df
    except Exception as e:
        print(f"Error fetching {s}: {e}")
try:
    fetcher.close()
except:
    pass

presets = {
    'A_baseline': {
        'SL_BUFFER_ATR': 1.0,
        'TRAIL_START_PCT': 0.020,
        'TRAIL_DIST_PCT': 0.006,
        'MIN_RR': 1.45,
        'MAX_LOSS_DOLLAR': 5.8,
    },
    'B_wide_sl_only': {
        'SL_BUFFER_ATR': 1.5,
        'TRAIL_START_PCT': 0.020,
        'TRAIL_DIST_PCT': 0.006,
        'MIN_RR': 1.45,
        'MAX_LOSS_DOLLAR': 5.8,
    },
    'C_loose_trail_only': {
        'SL_BUFFER_ATR': 1.0,
        'TRAIL_START_PCT': 0.030,
        'TRAIL_DIST_PCT': 0.015,
        'MIN_RR': 1.45,
        'MAX_LOSS_DOLLAR': 5.8,
    },
    'D_wide_sl_loose_trail': {
        'SL_BUFFER_ATR': 1.5,
        'TRAIL_START_PCT': 0.030,
        'TRAIL_DIST_PCT': 0.015,
        'MIN_RR': 1.45,
        'MAX_LOSS_DOLLAR': 5.8,
    },
    'E_aggressive_growth': {
        'SL_BUFFER_ATR': 1.2,
        'TRAIL_START_PCT': 0.040,
        'TRAIL_DIST_PCT': 0.020,
        'MIN_RR': 1.20,
        'MAX_LOSS_DOLLAR': 8.0,
    },
    'F_max_breathing': {
        'SL_BUFFER_ATR': 2.0,
        'TRAIL_START_PCT': 0.050,
        'TRAIL_DIST_PCT': 0.025,
        'MIN_RR': 1.50,
        'MAX_LOSS_DOLLAR': 5.0,
    },
    'G_baseline_boosted_size': {
        'SL_BUFFER_ATR': 1.0,
        'TRAIL_START_PCT': 0.020,
        'TRAIL_DIST_PCT': 0.006,
        'MIN_RR': 1.45,
        'MAX_LOSS_DOLLAR': 8.0, 
    },
    'H_baseline_loose_rr': {
        'SL_BUFFER_ATR': 1.2,
        'TRAIL_START_PCT': 0.025,
        'TRAIL_DIST_PCT': 0.008,
        'MIN_RR': 1.25,
        'MAX_LOSS_DOLLAR': 7.0,
    }
}

def replay_portfolio(results, max_open):
    oks = [x for x in results if x.get('success')]
    trades = []
    for r in oks:
        for t in r.get('trades', []):
            try:
                et = datetime.fromisoformat(str(t.get('entry_time')).replace('Z', ''))
                xt = datetime.fromisoformat(str(t.get('exit_time')).replace('Z', ''))
            except Exception:
                continue
            m = float(t.get('margin', 0) or 0)
            p = float(t.get('pnl', 0) or 0)
            if m <= 0: continue
            trades.append({'entry': et, 'exit': xt, 'margin': m, 'pnl': p})
    trades.sort(key=lambda x: (x['entry'], x['exit']))

    bal = 1000.0
    peak = bal
    max_dd = 0.0
    open_pos = []
    monthly = {}

    def ym(dt):
        return f"{dt.year:04d}-{dt.month:02d}"

    for tr in trades:
        now = tr['entry']
        still = []
        for op in open_pos:
            if op['exit'] <= now:
                bal += op['m'] + op['p']
                k = ym(op['exit'])
                monthly.setdefault(k, {'start': None, 'end': None})
                monthly[k]['end'] = bal
                peak = max(peak, bal)
                dd = (peak - bal) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            else:
                still.append(op)
        open_pos = still

        if len(open_pos) >= max_open:
            continue

        scale = max(0.0, bal / 1000.0)
        m = tr['margin'] * scale
        p = tr['pnl'] * scale
        if m <= 0 or m > bal:
            continue

        k = ym(tr['entry'])
        monthly.setdefault(k, {'start': None, 'end': None})
        if monthly[k]['start'] is None:
            monthly[k]['start'] = bal

        bal -= m
        open_pos.append({'exit': tr['exit'], 'm': m, 'p': p})

    for op in sorted(open_pos, key=lambda x: x['exit']):
        bal += op['m'] + op['p']
        k = ym(op['exit'])
        monthly.setdefault(k, {'start': None, 'end': None})
        monthly[k]['end'] = bal
        peak = max(peak, bal)
        dd = (peak - bal) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    mr = []
    for k in sorted(monthly.keys()):
        v = monthly[k]
        s = v.get('start')
        e = v.get('end')
        if s and e:
            mr.append((e / s - 1) * 100)

    avg_m = sum(mr) / len(mr) if mr else 0
    med_m = sorted(mr)[len(mr) // 2] if mr else 0
    neg_m = sum(1 for x in mr if x <= 0)
    
    return {
        'final_balance': round(bal, 2),
        'return_pct': round((bal / 1000.0 - 1) * 100, 2),
        'max_dd_pct': round(max_dd * 100, 2),
        'avg_monthly_pct': round(avg_m, 2),
        'median_monthly_pct': round(med_m, 2),
        'negative_months': int(neg_m),
        'month_count': len(mr),
    }

all_out = []
for name, p in presets.items():
    print(f'\\n--- RUNNING {name} ---')
    ICTFullBacktest.SL_BUFFER_ATR = p['SL_BUFFER_ATR']
    ICTFullBacktest.TRAIL_START_PCT = p['TRAIL_START_PCT']
    ICTFullBacktest.TRAIL_DIST_PCT = p['TRAIL_DIST_PCT']
    ICTFullBacktest.MIN_RR = p['MIN_RR']
    ICTFullBacktest.MAX_LOSS_DOLLAR = p['MAX_LOSS_DOLLAR']
    
    # Common settings
    ICTFullBacktest.NOTIONAL_CAP = 350.0  # Increased from 300 to allow larger trades
    ICTFullBacktest.BALANCE_PCT = 0.22    # Increased from 0.17 to push >15% monthly
    ICTFullBacktest.POI_PROXIMITY_ATR = 1.8
    ICTFullBacktest.COOLDOWN_BARS = 3

    results = []
    t0 = time.time()
    for i, s in enumerate(symbols, 1):
        if s not in symbol_data:
            results.append({'success': False, 'error': 'No data', 'symbol': s})
            continue
        try:
            r = full_backtest_ict(s, days=120, initial_balance=1000.0, leverage=10, pre_fetched_df=symbol_data[s])
        except Exception as e:
            r = {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
        r['symbol'] = s
        results.append(r)
        
    pr = replay_portfolio(results, max_open=5)
    elapsed = round(time.time() - t0, 1)
    
    row = {'preset': name, 'params': p, 'portfolio': pr, 'elapsed_sec': elapsed}
    all_out.append(row)
    print(f'DONE {name}', pr)

# rank: prioritize higher monthly median
all_out.sort(key=lambda x: (-x['portfolio']['median_monthly_pct'], x['portfolio']['max_dd_pct']))

path = 'logs/core_v2_profit_tuning_120d_top30.json'
with open(path, 'w', encoding='utf-8') as f:
    json.dump({'meta': {'symbols': len(symbols), 'days': 120}, 'results': all_out}, f, ensure_ascii=False, indent=2)

print('\\n=== FINAL RANKING ===')
for r in all_out:
    print(f"{r['preset']}: avg_mo={r['portfolio']['avg_monthly_pct']}%, dd={r['portfolio']['max_dd_pct']}% -> Total Return={r['portfolio']['return_pct']}%")
