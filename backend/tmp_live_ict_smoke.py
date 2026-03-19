import json
from live_trader import LivePaperTrader, get_all_usdt_pairs
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators

tr=LivePaperTrader(initial_balance=1000.0, leverage=10)
tr.apply_profile_core_v2()
fetcher=DataFetcher('binance')

symbols=get_all_usdt_pairs()[:30]
rows=[]
for s in symbols:
    try:
        df1=fetcher.fetch_ohlcv(s,'1h',limit=220)
        df4=fetcher.fetch_ohlcv(s,'4h',limit=120)
        df5=fetcher.fetch_ohlcv(s,'5m',limit=220)
        if df1 is None or len(df1)<120:
            rows.append({'symbol':s,'ok':False,'err':'no_1h'})
            continue
        df1=add_all_indicators(df1)
        if df4 is not None and len(df4)>=40:
            df4=add_all_indicators(df4)
        if df5 is not None and len(df5)>=80:
            df5=add_all_indicators(df5)
        sig=tr._ict_smc_signal(df1, df_4h=df4, df_5m=df5, symbol=s)
        if sig:
            rows.append({'symbol':s,'ok':True,'signal':True,'dir':sig.get('direction'),'q':sig.get('quality_score'),'rr':sig.get('rr_ratio')})
        else:
            rows.append({'symbol':s,'ok':True,'signal':False})
    except Exception as e:
        rows.append({'symbol':s,'ok':False,'err':str(e)[:120]})

ok=sum(1 for r in rows if r.get('ok'))
signals=[r for r in rows if r.get('signal')]
print('symbols',len(rows),'ok',ok,'signals',len(signals))
if signals:
    avg_q=round(sum(float(r.get('q',0) or 0) for r in signals)/len(signals),2)
    avg_rr=round(sum(float(r.get('rr',0) or 0) for r in signals)/len(signals),2)
    print('avg_q',avg_q,'avg_rr',avg_rr)
print('signal_examples',signals[:8])
fails=[r for r in rows if not r.get('ok')]
print('fails',len(fails),fails[:8])
open('logs/live_ict_core_v2_smoke.json','w',encoding='utf-8').write(json.dumps({'summary':{'symbols':len(rows),'ok':ok,'signals':len(signals)},'rows':rows},indent=2))
