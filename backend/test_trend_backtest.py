"""Test Trend Following Backtest v1.0 — 7 coins x 30 days"""
import requests

API = "http://localhost:8000"

coins = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT",
    "LINK/USDT", "ADA/USDT", "BANANAS31/USDT"
]

print("=" * 80)
print("TREND FOLLOWING v1.0 — Supertrend + EMA + ADX + Smart Trail — 7 COINS x 30 DAYS")
print("=" * 80)

results = []
for coin in coins:
    try:
        r = requests.post(f"{API}/backtest/trend-backtest", json={
            "symbol": coin, "days": 30,
            "initial_balance": 1000, "leverage": 10
        }, timeout=300)
        d = r.json()

        if not d.get('success'):
            print(f"\n  {coin}: ERROR — {d.get('error', 'unknown')}")
            results.append({'coin': coin, 'pnl': 0, 'trades': 0, 'wr': 0, 'pf': 0})
            continue

        pnl = d.get('total_pnl', 0)
        pnl_pct = d.get('total_pnl_pct', 0)
        trades = d.get('total_trades', 0)
        wr = d.get('win_rate', 0)
        pf = d.get('profit_factor', 0)
        wins = d.get('wins', 0)
        losses = d.get('losses', 0)
        avg_w = d.get('avg_win', 0)
        avg_l = d.get('avg_loss', 0)
        longs = d.get('long_trades', 0)
        shorts = d.get('short_trades', 0)
        best = d.get('max_profit_trade', 0)
        worst = d.get('max_loss_trade', 0)

        status = "PROFIT" if pnl > 0 else ("BREAK" if abs(pnl) < 1 else "LOSS")
        icon = "+" if pnl > 0 else ""

        print(f"\n{'='*60}")
        print(f"  {coin}: {icon}${pnl:.2f} ({icon}{pnl_pct:.2f}%) — {status}")
        print(f"  Trades: {trades} (L:{longs} S:{shorts}) | WR: {wr:.1f}% | PF: {pf:.2f}")
        print(f"  Avg Win: ${avg_w:.2f} | Avg Loss: ${avg_l:.2f}")
        print(f"  Best: ${best:.2f} | Worst: ${worst:.2f}")
        print(f"  W/L: {wins}/{losses}")

        trade_list = d.get('trades', [])
        if trade_list:
            print(f"  --- Last 8 trades ---")
            for t in trade_list[-8:]:
                icon2 = "+" if t['pnl'] > 0 else ""
                print(f"    {t['direction']:5s} | {t['entry_time'][:16]} | E:{t['entry_price']:.4f} SL:{t['sl_price']:.4f} TP:{t['tp_price']:.4f} | PnL: {icon2}${t['pnl']:.2f} ({icon2}{t['pnl_pct']:.1f}%) | {t['exit_reason']}")

        results.append({
            'coin': coin, 'pnl': pnl, 'pnl_pct': pnl_pct,
            'trades': trades, 'wr': wr, 'pf': pf,
            'longs': longs, 'shorts': shorts
        })

    except Exception as e:
        print(f"\n  {coin}: ERROR — {e}")
        results.append({'coin': coin, 'pnl': 0, 'trades': 0, 'wr': 0, 'pf': 0})

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

total_pnl = sum(r['pnl'] for r in results)
profitable = sum(1 for r in results if r['pnl'] > 0)
total_coins = len(results)
total_trades = sum(r['trades'] for r in results)

print(f"\n  Total PnL:       ${total_pnl:.2f}")
print(f"  Profitable:      {profitable}/{total_coins} coins")
print(f"  Total Trades:    {total_trades}")
print(f"  Avg PnL/coin:    ${total_pnl/total_coins:.2f}")

print(f"\n  {'Coin':<18} {'PnL':>10} {'Trades':>8} {'WR':>8} {'PF':>8} {'L/S':>10}")
print(f"  {'-'*16} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
for r in sorted(results, key=lambda x: x['pnl'], reverse=True):
    icon = "+" if r['pnl'] > 0 else ""
    ls = f"{r.get('longs',0)}/{r.get('shorts',0)}"
    print(f"  {r['coin']:<18} {icon}${r['pnl']:>8.2f} {r['trades']:>8} {r['wr']:>7.1f}% {r['pf']:>7.2f} {ls:>10}")
