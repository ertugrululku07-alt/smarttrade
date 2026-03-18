import requests

r = requests.get('http://localhost:8000/live/quant/status')
d = r.json()
print(f"Open: {d['open_trades_count']}")
for t in d.get('open_trades', []):
    pair = t.get('pair', '?')
    side = t.get('side', '?')
    entry = t.get('entry', 0)
    pnl = t.get('pnl', 0)
    strat = t.get('strategy', '?')
    tid = t.get('id', '?')
    tp = t.get('tp_price', 0)
    sl = t.get('sl_price', 0)
    print(f"  {tid}: {pair} {side} entry={entry} pnl=${pnl:.2f} TP={tp} SL={sl} strat={strat}")
