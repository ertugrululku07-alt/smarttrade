import json
import time
import traceback
from datetime import datetime
from live_trader import get_all_usdt_pairs
from backtest.ict_full_backtest import ICTFullBacktest, full_backtest_ict

symbols = get_all_usdt_pairs()[:30]

presets = {
    'A_balanced_aggr': {
        'MIN_RR': 1.45,
        'MAX_LOSS_DOLLAR': 5.8,
        'NOTIONAL_CAP': 300.0,
        'BALANCE_PCT': 0.17,
        'POI_PROXIMITY_ATR': 1.8,
        'COOLDOWN_BARS': 3,
        'TRAIL_START_PCT': 0.020,
        'TRAIL_DIST_PCT': 0.006,
        'max_open': 5,
    },
    'B_consistent': {
        'MIN_RR': 1.50,
        'MAX_LOSS_DOLLAR': 5.2,
        'NOTIONAL_CAP': 280.0,
        'BALANCE_PCT': 0.16,
        'POI_PROXIMITY_ATR': 1.7,
        'COOLDOWN_BARS': 3,
        'TRAIL_START_PCT': 0.022,
        'TRAIL_DIST_PCT': 0.0055,
        'max_open': 4,
    },
    'C_growth': {
        'MIN_RR': 1.40,
        'MAX_LOSS_DOLLAR': 6.2,
        'NOTIONAL_CAP': 330.0,
        'BALANCE_PCT': 0.19,
        'POI_PROXIMITY_ATR': 2.0,
        'COOLDOWN_BARS': 2,
        'TRAIL_START_PCT': 0.018,
        'TRAIL_DIST_PCT': 0.007,
        'max_open': 6,
    },
    'D_tight_risk': {
        'MIN_RR': 1.55,
        'MAX_LOSS_DOLLAR': 4.8,
        'NOTIONAL_CAP': 260.0,
        'BALANCE_PCT': 0.15,
        'POI_PROXIMITY_ATR': 1.6,
        'COOLDOWN_BARS': 4,
        'TRAIL_START_PCT': 0.022,
        'TRAIL_DIST_PCT': 0.005,
        'max_open': 4,
    },
    'E_mid': {
        'MIN_RR': 1.48,
        'MAX_LOSS_DOLLAR': 5.5,
        'NOTIONAL_CAP': 290.0,
        'BALANCE_PCT': 0.17,
        'POI_PROXIMITY_ATR': 1.75,
        'COOLDOWN_BARS': 3,
        'TRAIL_START_PCT': 0.021,
        'TRAIL_DIST_PCT': 0.0058,
        'max_open': 5,
    }
}

def replay_portfolio(results, max_open):
    oks=[x for x in results if x.get('success')]
    trades=[]
    for r in oks:
        for t in r.get('trades', []):
            try:
                et=datetime.fromisoformat(str(t.get('entry_time')).replace('Z',''))
                xt=datetime.fromisoformat(str(t.get('exit_time')).replace('Z',''))
            except Exception:
                continue
            m=float(t.get('margin',0) or 0)
            p=float(t.get('pnl',0) or 0)
            if m<=0: continue
            trades.append({'entry':et,'exit':xt,'margin':m,'pnl':p})
    trades.sort(key=lambda x:(x['entry'],x['exit']))

    bal=1000.0
    peak=bal
    max_dd=0.0
    open_pos=[]
    monthly={}

    def ym(dt):
        return f"{dt.year:04d}-{dt.month:02d}"

    for tr in trades:
        now=tr['entry']
        still=[]
        for op in open_pos:
            if op['exit']<=now:
                bal += op['m'] + op['p']
                k=ym(op['exit'])
                monthly.setdefault(k, {'start':None,'end':None})
                monthly[k]['end']=bal
                peak=max(peak, bal)
                dd=(peak-bal)/peak if peak>0 else 0
                max_dd=max(max_dd, dd)
            else:
                still.append(op)
        open_pos=still

        if len(open_pos)>=max_open:
            continue

        scale=max(0.0, bal/1000.0)
        m=tr['margin']*scale
        p=tr['pnl']*scale
        if m<=0 or m>bal:
            continue

        k=ym(tr['entry'])
        monthly.setdefault(k, {'start':None,'end':None})
        if monthly[k]['start'] is None:
            monthly[k]['start']=bal

        bal -= m
        open_pos.append({'exit':tr['exit'],'m':m,'p':p})

    for op in sorted(open_pos, key=lambda x:x['exit']):
        bal += op['m'] + op['p']
        k=ym(op['exit'])
        monthly.setdefault(k, {'start':None,'end':None})
        monthly[k]['end']=bal
        peak=max(peak, bal)
        dd=(peak-bal)/peak if peak>0 else 0
        max_dd=max(max_dd, dd)

    mr=[]
    for v in monthly.values():
        s=v.get('start'); e=v.get('end')
        if s and e:
            mr.append((e/s-1)*100)

    avg_m = sum(mr)/len(mr) if mr else 0
    med_m = sorted(mr)[len(mr)//2] if mr else 0
    neg_m = sum(1 for x in mr if x<=0)
    return {
        'final_balance': round(bal,2),
        'return_pct': round((bal/1000.0-1)*100,2),
        'max_dd_pct': round(max_dd*100,2),
        'avg_monthly_pct': round(avg_m,2),
        'median_monthly_pct': round(med_m,2),
        'negative_months': int(neg_m),
        'month_count': len(mr),
    }

all_out=[]
for name, p in presets.items():
    print('RUN', name)
    ICTFullBacktest.MIN_RR=p['MIN_RR']
    ICTFullBacktest.MAX_LOSS_DOLLAR=p['MAX_LOSS_DOLLAR']
    ICTFullBacktest.NOTIONAL_CAP=p['NOTIONAL_CAP']
    ICTFullBacktest.BALANCE_PCT=p['BALANCE_PCT']
    ICTFullBacktest.POI_PROXIMITY_ATR=p['POI_PROXIMITY_ATR']
    ICTFullBacktest.COOLDOWN_BARS=p['COOLDOWN_BARS']
    ICTFullBacktest.TRAIL_START_PCT=p['TRAIL_START_PCT']
    ICTFullBacktest.TRAIL_DIST_PCT=p['TRAIL_DIST_PCT']

    results=[]
    t0=time.time()
    for i,s in enumerate(symbols,1):
        try:
            r=full_backtest_ict(s, days=120, initial_balance=1000.0, leverage=10)
        except Exception as e:
            r={'success':False,'error':str(e),'traceback':traceback.format_exc()}
        r['symbol']=s
        results.append(r)
    pr = replay_portfolio(results, max_open=p['max_open'])
    elapsed=round(time.time()-t0,1)
    row={'preset':name,'params':p,'portfolio':pr,'elapsed_sec':elapsed}
    all_out.append(row)
    print('DONE', name, pr)

# rank: prioritize lower DD and higher monthly median
all_out.sort(key=lambda x: (x['portfolio']['max_dd_pct']<=30, -x['portfolio']['median_monthly_pct'], -x['portfolio']['avg_monthly_pct'], -x['portfolio']['return_pct']), reverse=True)

path='logs/core_v2_aggressive_tuning_120d_top30.json'
with open(path,'w',encoding='utf-8') as f:
    json.dump({'meta':{'symbols':len(symbols),'days':120},'results':all_out}, f, ensure_ascii=False, indent=2)

print('FINAL', path)
for r in all_out:
    print(r['preset'], r['portfolio'])
