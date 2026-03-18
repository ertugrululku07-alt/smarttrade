"""Trade history analyzer — one-time script"""
import json
from datetime import datetime

data = json.load(open('live_trader_state.json', 'r'))

balance = data.get('balance', 0)
closed = data.get('closed_trades', [])
open_trades = data.get('open_trades', [])

print(f"{'='*60}")
print(f"  TRADE HISTORY ANALYSIS")
print(f"{'='*60}")
print(f"Balance: ${balance:,.2f}")
print(f"Closed trades: {len(closed)}")
print(f"Open trades: {len(open_trades)}")
print()

# Open trades detail
if open_trades:
    print(f"--- OPEN TRADES ---")
    for t in open_trades:
        sym = t.get('symbol', '?')
        side = t.get('side', '?')
        entry = t.get('entry_price', 0)
        tp = t.get('tp_price', 0)
        sl = t.get('sl_price', 0)
        strat = t.get('strategy', t.get('signal_result', {}).get('strategy', '?'))
        qty = t.get('qty', 0)
        print(f"  {sym} {side} @ {entry} | qty={qty:.4f} | TP={tp} SL={sl} | {strat}")
    print()

# Closed trades analysis
if closed:
    print(f"--- CLOSED TRADES ---")
    total_pnl = 0
    wins = 0
    losses = 0
    be_trades = 0
    reasons = {}
    strategies = {}
    directions = {'LONG': {'count': 0, 'pnl': 0}, 'SHORT': {'count': 0, 'pnl': 0}}
    
    for t in closed:
        sym = t.get('symbol', '?')
        side = t.get('side', '?')
        entry = t.get('entry_price', 0)
        exit_p = t.get('exit_price', 0)
        qty = t.get('qty', 0)
        reason = t.get('close_reason', '?')
        strat = t.get('strategy', t.get('signal_result', {}).get('strategy', '?'))
        pnl = t.get('pnl', 0)
        pnl_pct = t.get('pnl_pct', 0)
        entry_time = t.get('entry_time', '?')
        exit_time = t.get('exit_time', '?')
        max_pnl = t.get('max_pnl_pct', 0)
        
        # Calculate PnL if not stored
        if pnl == 0 and entry > 0 and exit_p > 0 and qty > 0:
            if side == 'LONG':
                pnl = (exit_p - entry) * qty
            else:
                pnl = (entry - exit_p) * qty
        
        if pnl == 0 and entry > 0 and exit_p > 0:
            if side == 'LONG':
                pnl_pct = ((exit_p - entry) / entry) * 100
            else:
                pnl_pct = ((entry - exit_p) / entry) * 100
        
        total_pnl += pnl
        if pnl > 0.01:
            wins += 1
        elif pnl < -0.01:
            losses += 1
        else:
            be_trades += 1
        
        reasons[reason] = reasons.get(reason, 0) + 1
        strategies[strat] = strategies.get(strat, {'count': 0, 'pnl': 0})
        strategies[strat]['count'] += 1
        strategies[strat]['pnl'] += pnl
        
        if side in directions:
            directions[side]['count'] += 1
            directions[side]['pnl'] += pnl
        
        status = "WIN" if pnl > 0.01 else ("LOSS" if pnl < -0.01 else "BE")
        print(f"  [{status:4s}] {sym:15s} {side:5s} | Entry: {entry:<12} Exit: {exit_p:<12} | PnL: ${pnl:>+8.2f} ({pnl_pct:>+6.2f}%) | MaxPnL: {max_pnl:.1f}% | {reason} | {entry_time}->{exit_time}")
    
    print()
    print(f"{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"Total PnL:     ${total_pnl:>+10.2f}")
    print(f"Wins:          {wins}")
    print(f"Losses:        {losses}")
    print(f"Breakeven:     {be_trades}")
    total = wins + losses + be_trades
    wr = (wins / total * 100) if total > 0 else 0
    print(f"Win Rate:      {wr:.1f}% ({wins}/{total})")
    if wins > 0:
        avg_win = sum(t.get('pnl', 0) for t in closed if t.get('pnl', 0) > 0.01) / wins
        print(f"Avg Win:       ${avg_win:>+.2f}")
    if losses > 0:
        avg_loss = sum(t.get('pnl', 0) for t in closed if t.get('pnl', 0) < -0.01) / losses
        print(f"Avg Loss:      ${avg_loss:>+.2f}")
    
    print()
    print(f"--- BY DIRECTION ---")
    for d, v in directions.items():
        if v['count'] > 0:
            print(f"  {d:5s}: {v['count']} trades, PnL: ${v['pnl']:>+.2f}")
    
    print()
    print(f"--- BY STRATEGY ---")
    for s, v in strategies.items():
        print(f"  {s}: {v['count']} trades, PnL: ${v['pnl']:>+.2f}")
    
    print()
    print(f"--- BY CLOSE REASON ---")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")

print()
print(f"Initial balance: $10,000.00")
print(f"Current balance: ${balance:,.2f}")
print(f"Total return:    ${balance - 10000:>+.2f} ({(balance/10000 - 1)*100:>+.2f}%)")
