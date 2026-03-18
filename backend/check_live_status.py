import requests
import json

r = requests.get('http://localhost:8000/live/quant/status')
d = r.json()

print("=" * 60)
print("AUTO-TRADER DURUM RAPORU")
print("=" * 60)
print(f"  Status:    {d['status']}")
print(f"  Balance:   ${d['balance']:.2f}  (Başlangıç: $10,000)")
print(f"  PnL:       ${d['balance'] - 10000:.2f}")
print(f"  Open:      {d['open_trades_count']}")
print(f"  Closed:    {d['closed_trades_count']}")
print(f"  Pending:   {d['pending_orders_count']}")
print(f"  Scanning:  {d['scanned_markets_count']} markets")
print(f"  Max Open:  {d['max_open_trades_limit']}")

# Open trades
if d['open_trades']:
    print(f"\n{'=' * 60}")
    print("AÇIK POZİSYONLAR")
    print("=" * 60)
    for t in d['open_trades']:
        print(f"  {t['pair']} {t['side']} | Entry: {t['entry']:.4f} | "
              f"PnL: ${t.get('pnl', 0):.2f} ({t.get('pnl_pct', 0):.1f}%) | "
              f"SL: {t.get('sl_price', 0):.4f} | TP: {t.get('tp_price', 0):.4f} | "
              f"Strategy: {t.get('strategy', '?')}")

# Closed trades (last 10)
if d['closed_trades']:
    print(f"\n{'=' * 60}")
    print(f"SON {min(len(d['closed_trades']), 20)} KAPANMIŞ İŞLEM")
    print("=" * 60)
    wins = 0
    losses = 0
    total_pnl = 0
    for t in d['closed_trades']:
        pnl = t.get('pnl', 0)
        total_pnl += pnl
        marker = "+" if pnl > 0 else "-"
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        print(f"  {marker} {t['pair']:12s} {t['side']:5s} | "
              f"PnL: ${pnl:+.2f} ({t.get('pnl_pct', 0):+.1f}%) | "
              f"Reason: {t.get('reason', '?'):12s} | "
              f"Strategy: {t.get('strategy', '?'):15s} | "
              f"Regime: {t.get('regime', '?')}")
    print(f"\n  Toplam: W={wins} L={losses} WR={wins/(wins+losses)*100:.1f}% PnL=${total_pnl:.2f}")

# Recent logs
if d.get('recent_logs'):
    print(f"\n{'=' * 60}")
    print("SON LOGLAR")
    print("=" * 60)
    for log in d['recent_logs'][:10]:
        print(f"  {log['text'][:100]}")

# V3 Stats
r2 = requests.get('http://localhost:8000/live/v3/stats')
stats = r2.json()
print(f"\n{'=' * 60}")
print("V3 ENGINE İSTATİSTİKLERİ")
print("=" * 60)
print(f"  Total Trades: {stats.get('total_trades', 0)}")
print(f"  Wins: {stats.get('wins', 0)}")
print(f"  Losses: {stats.get('losses', 0)}")
print(f"  Win Rate: {stats.get('win_rate', 0):.1f}%")
print(f"  Total PnL: ${stats.get('total_pnl', 0):.2f}")
print(f"  Avg RR: ${stats.get('avg_rr', 0):.2f}")
