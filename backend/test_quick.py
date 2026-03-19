import requests
r = requests.post('http://localhost:8000/backtest/ict-full-backtest', json={
    'symbol': 'ADA/USDT', 'days': 30, 'initial_balance': 1000, 'leverage': 10
}, timeout=120)
d = r.json()
print(f"PnL: ${d.get('total_pnl',0):.2f}, Trades: {d.get('total_trades',0)}, WR: {d.get('win_rate',0):.1f}%, PF: {d.get('profit_factor',0):.2f}")
if 'error' in d:
    print(f"ERROR: {d['error']}")
    if 'traceback' in d:
        print(d['traceback'])
