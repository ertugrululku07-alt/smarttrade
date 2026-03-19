import json,time
from datetime import datetime
from live_trader import get_all_usdt_pairs
from backtest.ict_full_backtest import ICTFullBacktest, full_backtest_ict

base={
 'MIN_RR':1.45,'MAX_LOSS_DOLLAR':5.8,'NOTIONAL_CAP':300.0,'BALANCE_PCT':0.17,'POI_PROXIMITY_ATR':1.8,
 'COOLDOWN_BARS':3,'TRAIL_START_PCT':0.020,'TRAIL_DIST_PCT':0.006,
}
# tuple(name, overrides)
presets=[
 ('p0_current', {}),
 ('p1_relax_entry', {'ENTRY_RANGE_TOP_CEIL':0.86,'ENTRY_RANGE_BOT_FLOOR':0.14,'ENTRY_MAX_EMA21_EXT':0.04}),
 ('p2_mid_relax_no_earlycut', {'ENTRY_RANGE_TOP_CEIL':0.85,'ENTRY_RANGE_BOT_FLOOR':0.15,'ENTRY_MAX_EMA21_EXT':0.045,'EARLY_CUT_R':0.75}),
 ('p3_keep_filters_tighten_trail', {'TRAIL_START_PCT':0.018,'TRAIL_DIST_PCT':0.007,'EARLY_CUT_R':0.70}),
 ('p4_restore_sl_style', {'MAX_SL_PCT':0.025,'ENTRY_RANGE_TOP_CEIL':0.84,'ENTRY_RANGE_BOT_FLOOR':0.16,'EARLY_CUT_BARS':14,'EARLY_CUT_R':0.65}),
]

symbols=get_all_usdt_pairs()[:20]


def replay(trades):
    pt=[]
    for t in trades:
        try:
            et=datetime.fromisoformat(str(t['entry_time']).replace('Z',''))
            xt=datetime.fromisoformat(str(t['exit_time']).replace('Z',''))
            m=float(t.get('margin',0) or 0); p=float(t.get('pnl',0) or 0)
            if m>0: pt.append((et,xt,m,p))
        except: pass
    pt.sort(key=lambda x:(x[0],x[1]))
    bal=1000.0; peak=bal; dd=0.0; openp=[]
    for et,xt,m,p in pt:
        still=[]
        for o in openp:
            if o[0]<=et:
                bal+=o[1]+o[2]
                peak=max(peak,bal)
                dd=max(dd,(peak-bal)/peak if peak>0 else 0)
            else: still.append(o)
        openp=still
        if len(openp)>=5: continue
        scale=max(0.0,bal/1000.0)
        me=m*scale; pe=p*scale
        if me<=0 or me>bal: continue
        bal-=me
        openp.append((xt,me,pe))
    for xt,me,pe in sorted(openp,key=lambda x:x[0]):
        bal+=me+pe
        peak=max(peak,bal)
        dd=max(dd,(peak-bal)/peak if peak>0 else 0)
    return bal,dd

rows=[]
for name,ovr in presets:
    for k,v in base.items(): setattr(ICTFullBacktest,k,v)
    for k,v in ovr.items(): setattr(ICTFullBacktest,k,v)
    trs=[]; total=0
    for s in symbols:
        r=full_backtest_ict(s,days=80,initial_balance=1000.0,leverage=10)
        if not r.get('success'): continue
        total+=r.get('total_trades',0)
        trs.extend(r.get('trades',[]))
    wins=sum(1 for t in trs if float(t.get('pnl',0) or 0)>0)
    losses=sum(1 for t in trs if float(t.get('pnl',0) or 0)<=0)
    sl=sum(1 for t in trs if str(t.get('exit_reason','')).upper() in ('SL','MAXLOSS','EARLY_CUT'))
    bal,dd=replay(trs)
    wr=(wins/len(trs)*100) if trs else 0
    slr=(sl/losses*100) if losses else 0
    score=(bal-1000)-dd*250
    row={'preset':name,'trades':len(trs),'wr':round(wr,2),'slr':round(slr,2),'ret':round((bal/1000-1)*100,2),'dd':round(dd*100,2),'score':round(score,2)}
    rows.append(row)
    print(row)

rows=sorted(rows,key=lambda x:x['score'],reverse=True)
print('BEST',rows[0])
open('logs/ict_tune_quick.json','w',encoding='utf-8').write(json.dumps(rows,indent=2))
