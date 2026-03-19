import json
from live_trader import LivePaperTrader, get_all_usdt_pairs
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators

tr=LivePaperTrader(initial_balance=1000.0, leverage=10)
tr.apply_profile_core_v2()
fetcher=DataFetcher('binance')

symbols=get_all_usdt_pairs()[:12]
rows=[]
for s in symbols:
    try:
        df1=fetcher.fetch_ohlcv(s,'1h',limit=360)
        df4=fetcher.fetch_ohlcv(s,'4h',limit=220)
        df5=fetcher.fetch_ohlcv(s,'15m',limit=900)
        if df1 is None or len(df1)<220:
            rows.append({'symbol':s,'ok':False,'err':'no_1h'})
            continue
        df1=add_all_indicators(df1)
        if df4 is not None and len(df4)>=60:
            df4=add_all_indicators(df4)
        else:
            df4=None
        if df5 is not None and len(df5)>=120:
            df5=add_all_indicators(df5)
        else:
            df5=None

        sig_count=0
        q_sum=0.0
        rr_sum=0.0
        tested=0
        start=max(120, len(df1)-120)
        for i in range(start, len(df1)):
            df1i=df1.iloc[:i+1]
            ts=df1i.index[-1]
            df4i=None
            if df4 is not None:
                df4i=df4[df4.index<=ts]
                if len(df4i)<40:
                    df4i=None
            df5i=None
            if df5 is not None:
                df5i=df5[df5.index<=ts].iloc[-220:]
                if len(df5i)<80:
                    df5i=None
            sig=tr._ict_smc_signal(df1i, df_4h=df4i, df_5m=df5i, symbol=s)
            tested += 1
            if sig:
                sig_count += 1
                q_sum += float(sig.get('quality_score',0) or 0)
                rr_sum += float(sig.get('rr_ratio',0) or 0)
        rows.append({
            'symbol':s,'ok':True,'tested':tested,'signals':sig_count,
            'signal_rate_pct': round((sig_count/tested*100) if tested else 0,2),
            'avg_q': round(q_sum/sig_count,2) if sig_count else 0.0,
            'avg_rr': round(rr_sum/sig_count,2) if sig_count else 0.0,
        })
    except Exception as e:
        rows.append({'symbol':s,'ok':False,'err':str(e)[:140]})

ok=[r for r in rows if r.get('ok')]
signals=sum(r.get('signals',0) for r in ok)
tested=sum(r.get('tested',0) for r in ok)
rate=(signals/tested*100) if tested else 0
print('symbols',len(rows),'ok',len(ok),'tested',tested,'signals',signals,'rate',round(rate,2))
if signals>0:
    avg_q=sum(r.get('avg_q',0)*r.get('signals',0) for r in ok)/signals
    avg_rr=sum(r.get('avg_rr',0)*r.get('signals',0) for r in ok)/signals
    print('avg_q',round(avg_q,2),'avg_rr',round(avg_rr,2))
print('rows',rows)
open('logs/live_ict_core_v2_rolling_test.json','w',encoding='utf-8').write(json.dumps({'summary':{'symbols':len(rows),'ok':len(ok),'tested':tested,'signals':signals,'signal_rate_pct':round(rate,2)},'rows':rows},indent=2))
