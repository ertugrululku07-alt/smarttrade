"""Detailed trade analysis — separate old vs new (v5.2) trades"""
import json
from collections import defaultdict

data = json.load(open('live_trader_state.json', 'r'))
balance = data.get('balance', 0)
closed = data.get('closed_trades', [])
open_trades = data.get('open_trades', [])

# v5.2 fixes applied ~13:35 on Mar 15. Trades opened after that are "new"
# But we also need to check: old bb_mr trades had large notional (~$1800)
# new bb_mr trades have small notional (~$150)

def calc_pnl(t):
    pnl = t.get('pnl', 0)
    if pnl != 0:
        return pnl
    entry = t.get('entry_price', 0)
    exit_p = t.get('exit_price', 0)
    qty = t.get('qty', 0)
    side = t.get('side', '')
    if entry > 0 and exit_p > 0 and qty > 0:
        if side == 'LONG':
            return (exit_p - entry) * qty
        else:
            return (entry - exit_p) * qty
    return 0

def get_notional(t):
    return t.get('entry_price', 0) * t.get('qty', 0)

def get_strategy(t):
    s = t.get('strategy', '')
    if not s:
        sr = t.get('signal_result', {})
        s = sr.get('strategy', '?')
    return s

print(f"{'='*70}")
print(f"  FULL TRADE ANALYSIS REPORT")
print(f"{'='*70}")
print(f"  Balance: ${balance:,.2f} | From: $10,000.00 | Return: ${balance-10000:+.2f}")
print(f"  Closed: {len(closed)} | Open: {len(open_trades)}")
print(f"{'='*70}\n")

# Separate into categories
old_legacy = []   # non-bb_mr trades
old_bb_mr = []    # bb_mr with high notional (pre-v5.2)
new_bb_mr = []    # bb_mr with low notional (post-v5.2)

for t in closed:
    strat = get_strategy(t)
    notional = get_notional(t)
    if 'bb_mr' not in strat:
        old_legacy.append(t)
    elif notional > 500:  # Old system had ~$1800 notional
        old_bb_mr.append(t)
    else:
        new_bb_mr.append(t)

def print_group(title, trades):
    if not trades:
        print(f"\n--- {title}: 0 trades ---\n")
        return
    
    total_pnl = 0
    wins = 0
    losses = 0
    max_win = 0
    max_loss = 0
    pnls = []
    
    for t in trades:
        pnl = calc_pnl(t)
        pnls.append(pnl)
        total_pnl += pnl
        if pnl > 0.01:
            wins += 1
            max_win = max(max_win, pnl)
        elif pnl < -0.01:
            losses += 1
            max_loss = min(max_loss, pnl)
    
    be = len(trades) - wins - losses
    wr = (wins / len(trades) * 100) if trades else 0
    avg_win = sum(p for p in pnls if p > 0.01) / wins if wins > 0 else 0
    avg_loss = sum(p for p in pnls if p < -0.01) / losses if losses > 0 else 0
    
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    print(f"  Trades: {len(trades)} | W: {wins} L: {losses} BE: {be} | WR: {wr:.1f}%")
    print(f"  Total PnL:  ${total_pnl:>+10.2f}")
    print(f"  Avg Win:    ${avg_win:>+10.2f} | Max Win:  ${max_win:>+.2f}")
    print(f"  Avg Loss:   ${avg_loss:>+10.2f} | Max Loss: ${max_loss:>+.2f}")
    if avg_loss != 0:
        print(f"  Risk/Reward: {abs(avg_win/avg_loss):.2f}:1")
    
    # Per-trade detail
    print(f"\n  {'#':>3} {'Symbol':15s} {'Side':5s} {'Notional':>10s} {'PnL':>10s} {'MaxPnl%':>8s} {'Reason':20s}")
    print(f"  {'─'*75}")
    for i, t in enumerate(trades):
        sym = t.get('symbol', '?')
        side = t.get('side', '?')
        pnl = calc_pnl(t)
        notional = get_notional(t)
        max_pnl = t.get('max_pnl_pct', 0)
        reason = t.get('close_reason', '?')
        mark = "+" if pnl > 0.01 else ("-" if pnl < -0.01 else "~")
        print(f"  {i+1:>3} {sym:15s} {side:5s} ${notional:>9.2f} ${pnl:>+9.2f} {max_pnl:>7.1f}% {reason:20s} {mark}")

print_group("LEGACY STRATEGIES (pre-BB MR)", old_legacy)
print_group("OLD BB MR (pre-v5.2, HIGH notional ~$1800)", old_bb_mr)
print_group("NEW BB MR (v5.2, LOW notional ~$150)", new_bb_mr)

# Open trades
if open_trades:
    print(f"\n{'─'*70}")
    print(f"  OPEN TRADES")
    print(f"{'─'*70}")
    total_unrealized = 0
    for t in open_trades:
        sym = t.get('symbol', '?')
        side = t.get('side', '?')
        entry = t.get('entry_price', 0)
        notional = get_notional(t)
        strat = get_strategy(t)
        pnl = t.get('pnl', 0)
        total_unrealized += pnl
        print(f"  {sym:15s} {side:5s} @ {entry:<12} Notional: ${notional:.2f} | PnL: ${pnl:>+.2f} | {strat}")
    print(f"  Total unrealized: ${total_unrealized:>+.2f}")

# Direction analysis for new trades
if new_bb_mr:
    print(f"\n{'─'*70}")
    print(f"  NEW v5.2 — DIRECTION BREAKDOWN")
    print(f"{'─'*70}")
    for d in ['LONG', 'SHORT']:
        dt = [t for t in new_bb_mr if t.get('side') == d]
        if dt:
            dpnl = sum(calc_pnl(t) for t in dt)
            dw = sum(1 for t in dt if calc_pnl(t) > 0.01)
            print(f"  {d:5s}: {len(dt)} trades, {dw} wins, PnL: ${dpnl:>+.2f}")
