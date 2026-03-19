import json
from copy import deepcopy
from live_trader import LivePaperTrader, get_all_usdt_pairs
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators

fetcher=DataFetcher('binance')
symbols=get_all_usdt_pairs()[:12]

# preload data once
cache={}
for s in symbols:
    d1=fetcher.fetch_ohlcv(s,'1h',limit=360)
    d4=fetcher.fetch_ohlcv(s,'4h',limit=220)
    d5=fetcher.fetch_ohlcv(s,'15m',limit=900)
    if d1 is not None and len(d1)>=220: d1=add_all_indicators(d1)
    else: d1=None
    if d4 is not None and len(d4)>=60: d4=add_all_indicators(d4)
    else: d4=None
    if d5 is not None and len(d5)>=120: d5=add_all_indicators(d5)
    else: d5=None
    cache[s]=(d1,d4,d5)

tr=LivePaperTrader(initial_balance=1000.0, leverage=10)
tr.log=lambda *_args, **_kwargs: None
tr.apply_profile_core_v2()
base=deepcopy(tr._ICT_PARAMS)

profiles={
 'old_like': {
   'entry_pullback_min_pct':0.03,
   'entry_range_lookback':30,
   'entry_range_top_ceil':0.70,
   'entry_range_bot_floor':0.30,
   'entry_max_ema21_ext':0.0,
 },
 'new_live': {
   'entry_pullback_min_pct':0.015,
   'entry_range_lookback':20,
   'entry_range_top_ceil':0.85,
   'entry_range_bot_floor':0.15,
   'entry_max_ema21_ext':0.045,
 }
}

out={}
for name,ovr in profiles.items():
    tr._ICT_PARAMS=deepcopy(base)
    tr._ICT_PARAMS.update(ovr)
    tested=0; signals=0
    by_symbol=[]
    for s in symbols:
        d1,d4,d5=cache[s]
        if d1 is None:
            by_symbol.append({'symbol':s,'ok':False})
            continue
        ss=0; tt=0
        start=max(120,len(d1)-120)
        for i in range(start, len(d1)):
            d1i=d1.iloc[:i+1]
            ts=d1i.index[-1]
            d4i=d4[d4.index<=ts] if d4 is not None else None
            if d4i is not None and len(d4i)<40: d4i=None
            d5i=d5[d5.index<=ts].iloc[-220:] if d5 is not None else None
            if d5i is not None and len(d5i)<80: d5i=None
            sig=tr._ict_smc_signal(d1i, df_4h=d4i, df_5m=d5i, symbol=s)
            tt+=1
            if sig: ss+=1
        tested+=tt; signals+=ss
        by_symbol.append({'symbol':s,'ok':True,'signals':ss,'tested':tt})
    out[name]={'tested':tested,'signals':signals,'rate':round((signals/tested*100) if tested else 0,2),'by_symbol':by_symbol}

print(out)
open('logs/live_ict_signal_rate_compare.json','w',encoding='utf-8').write(json.dumps(out,indent=2))
