import requests

r = requests.post(
    'http://localhost:8000/backtest/run-adaptive',
    json={'symbol': 'RUNE/USDT', 'timeframe': '1h', 'limit': 720}
)
print(f"Status: {r.status_code}")
d = r.json()
m = d.get('metrics', {})
print(f"  PnL: ${m.get('total_pnl', 0):.2f}")
print(f"  Trades: {m.get('total_trades', '?')}")
print(f"  WR: {m.get('win_rate', '?')}%")
print(f"  Win: {m.get('win_trades', '?')}, Loss: {m.get('loss_trades', '?')}")
print(f"  DD: {m.get('max_drawdown', '?')}%")
print(f"  Strategy: {m.get('strategy_usage', {})}")
