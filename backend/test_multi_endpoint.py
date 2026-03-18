"""Test multi-coin endpoint"""
import requests

r = requests.post('http://localhost:8000/backtest/run-multi', json={
    'symbols': ['BTC/USDT', 'ETH/USDT', 'ATOM/USDT', 'DOGE/USDT', 'RUNE/USDT'],
    'timeframe': '1h',
    'limit': 500,
    'initial_balance': 1000
})
d = r.json()
s = d.get('summary', {})
print(f"Total PnL: ${s.get('total_pnl', 0)}")
print(f"Profitable: {s.get('profitable_symbols', 0)}/{s.get('total_symbols', 0)}")
print(f"Best: {s.get('best_symbol', '')} ${s.get('best_pnl', 0)}")
print(f"Worst: {s.get('worst_symbol', '')} ${s.get('worst_pnl', 0)}")
print()
for x in d.get('results', []):
    print(f"  {x['symbol']:12s} {x['total_trades']:3d}t WR:{x['win_rate']:5.1f}% PnL:${x['total_pnl']:+8.2f}")
