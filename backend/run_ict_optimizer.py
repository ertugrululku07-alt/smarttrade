"""
ICT Portfolio-Level Grid Search Optimizer
Tests: margin size, max concurrent trades, circuit breakers
Goal: 80%+ profitability with <30% max drawdown
"""
import sys, os
sys.path.append(os.getcwd())

from itertools import product
from datetime import datetime
from backtest.ict_full_backtest import ICTFullBacktest
from live_trader import get_all_usdt_pairs

LOG = []
def log(msg):
    print(msg)
    LOG.append(msg)


def run_portfolio_sim(all_trades, margin_per_trade, max_concurrent, circuit_breaker_sl, circuit_breaker_cooldown):
    """Chronological portfolio replay with strict risk controls."""
    balance = 1000.0
    open_pos = []
    accepted = 0
    skipped_balance = 0
    skipped_concurrent = 0
    skipped_circuit = 0
    peak = balance
    max_dd = 0.0
    total_wins = 0
    total_losses = 0
    consecutive_sl = 0
    cooldown_until = None

    for tr in all_trades:
        now = tr['entry_time']

        # Close positions that exited before 'now'
        still = []
        for op in open_pos:
            if op['exit_time'] <= now:
                balance += op['margin'] + op['pnl']
                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
                if op['pnl'] > 0:
                    total_wins += 1
                    consecutive_sl = 0
                else:
                    total_losses += 1
                    consecutive_sl += 1
                    if consecutive_sl >= circuit_breaker_sl:
                        cooldown_until = op['exit_time']
            else:
                still.append(op)
        open_pos = still

        # Circuit breaker: skip trades during cooldown
        if cooldown_until is not None:
            # Cool down for circuit_breaker_cooldown hours after the trigger
            import datetime as dt_mod
            cooldown_end = cooldown_until + dt_mod.timedelta(hours=circuit_breaker_cooldown)
            if now < cooldown_end:
                skipped_circuit += 1
                continue
            else:
                cooldown_until = None
                consecutive_sl = 0

        # Max concurrent check
        if len(open_pos) >= max_concurrent:
            skipped_concurrent += 1
            continue

        # Balance check
        if balance < margin_per_trade:
            skipped_balance += 1
            continue

        # Scale PnL to the margin we're using
        original_margin = tr['margin']
        if original_margin <= 0:
            continue
        scaled_pnl = (tr['pnl'] / original_margin) * margin_per_trade

        balance -= margin_per_trade
        open_pos.append({
            'exit_time': tr['exit_time'],
            'margin': margin_per_trade,
            'pnl': scaled_pnl
        })
        accepted += 1

    # Close remaining
    for op in sorted(open_pos, key=lambda x: x['exit_time']):
        balance += op['margin'] + op['pnl']
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        if op['pnl'] > 0:
            total_wins += 1
        else:
            total_losses += 1

    total_pnl = balance - 1000.0
    total_trades = total_wins + total_losses
    wr = (total_wins / total_trades * 100) if total_trades > 0 else 0

    return {
        'balance': balance,
        'pnl': total_pnl,
        'pnl_pct': (total_pnl / 1000.0) * 100,
        'win_rate': wr,
        'max_dd': max_dd * 100,
        'total_trades': total_trades,
        'wins': total_wins,
        'losses': total_losses,
        'accepted': accepted,
        'skipped_balance': skipped_balance,
        'skipped_concurrent': skipped_concurrent,
        'skipped_circuit': skipped_circuit,
    }


def main():
    symbols = get_all_usdt_pairs()[:10]
    log(f"Phase 1: Fetching trade data for {len(symbols)} coins over 180 days...")

    # Collect ALL raw trades from all symbols
    all_trades = []
    for idx, s in enumerate(symbols, 1):
        try:
            log(f"  [{idx}/{len(symbols)}] {s}...")
            bt = ICTFullBacktest(initial_balance=100000.0, leverage=10)
            res = bt.run_backtest(s, days=180)
            if not res.get('success'):
                continue
            for t in res.get('trades', []):
                try:
                    et = datetime.fromisoformat(str(t.get('entry_time')).replace('Z', ''))
                    xt = datetime.fromisoformat(str(t.get('exit_time')).replace('Z', ''))
                except Exception:
                    continue
                m = float(t.get('margin', 0) or 0)
                p = float(t.get('pnl', 0) or 0)
                if m <= 0:
                    continue
                all_trades.append({
                    'symbol': s, 'entry_time': et, 'exit_time': xt,
                    'margin': m, 'pnl': p, 'dir': t.get('direction', '')
                })
        except Exception as e:
            log(f"  ERROR on {s}: {e}")

    all_trades.sort(key=lambda x: (x['entry_time'], x['exit_time']))
    log(f"\nPhase 1 complete: {len(all_trades)} total trades collected.\n")

    # Grid search parameters
    margins = [30, 50, 75, 100]
    max_concurrents = [1, 2, 3]
    cb_sls = [3, 5]
    cb_cooldowns = [6, 12, 24]

    combos = list(product(margins, max_concurrents, cb_sls, cb_cooldowns))
    log(f"Phase 2: Testing {len(combos)} portfolio configurations...\n")

    best = None
    best_score = -999999

    for i, (margin, mc, cb_sl, cb_cd) in enumerate(combos, 1):
        result = run_portfolio_sim(all_trades, margin, mc, cb_sl, cb_cd)
        pnl_pct = result['pnl_pct']
        max_dd = result['max_dd']
        wr = result['win_rate']

        dd_penalty = max(0, max_dd - 25) * 5
        score = pnl_pct - dd_penalty

        tag = ""
        if pnl_pct >= 80 and max_dd < 30:
            tag = " *** TARGET HIT!"
        elif pnl_pct >= 60 and max_dd < 40:
            tag = " ** GOOD"
        elif pnl_pct >= 40:
            tag = " * OK"

        log(f"[{i:3d}/{len(combos)}] M=${margin:3d} C={mc} CB={cb_sl}/{cb_cd}h "
              f"-> PnL: {pnl_pct:+7.1f}% DD: {max_dd:5.1f}% WR: {wr:4.1f}% "
              f"Trades: {result['total_trades']:4d}{tag}")

        if score > best_score and pnl_pct > 0:
            best_score = score
            best = {**result, 'margin': margin, 'max_concurrent': mc,
                    'cb_sl': cb_sl, 'cb_cooldown': cb_cd}

    log("\n" + "=" * 60)
    log("  OPTIMAL PORTFOLIO CONFIGURATION FOUND")
    log("=" * 60)
    if best:
        log(f"Margin per trade: ${best['margin']}")
        log(f"Max concurrent:   {best['max_concurrent']}")
        log(f"Circuit breaker:  {best['cb_sl']} SL -> {best['cb_cooldown']}h pause")
        log(f"---------------------------------")
        log(f"Net PnL:          {best['pnl_pct']:+.2f}%")
        log(f"Max Drawdown:     {best['max_dd']:.2f}%")
        log(f"Win Rate:         {best['win_rate']:.2f}%")
        log(f"Total Trades:     {best['total_trades']}")
        log(f"Wins/Losses:      {best['wins']}/{best['losses']}")
        log(f"Score:            {best_score:.2f}")
    else:
        log("No profitable configuration found!")
    log("=" * 60)

    # Write to file
    with open("ict_optimizer_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))


if __name__ == "__main__":
    main()


    # Collect ALL raw trades from all symbols
    all_trades = []
    for idx, s in enumerate(symbols, 1):
        try:
            print(f"  [{idx}/{len(symbols)}] {s}...")
            bt = ICTFullBacktest(initial_balance=100000.0, leverage=10)  # Big balance so no skipping
            res = bt.run_backtest(s, days=180)
            if not res.get('success'):
                continue
            for t in res.get('trades', []):
                try:
                    et = datetime.fromisoformat(str(t.get('entry_time')).replace('Z', ''))
                    xt = datetime.fromisoformat(str(t.get('exit_time')).replace('Z', ''))
                except Exception:
                    continue
                m = float(t.get('margin', 0) or 0)
                p = float(t.get('pnl', 0) or 0)
                if m <= 0:
                    continue
                all_trades.append({
                    'symbol': s, 'entry_time': et, 'exit_time': xt,
                    'margin': m, 'pnl': p, 'dir': t.get('direction', '')
                })
        except Exception as e:
            print(f"  ERROR on {s}: {e}")

    all_trades.sort(key=lambda x: (x['entry_time'], x['exit_time']))
    print(f"\nPhase 1 complete: {len(all_trades)} total trades collected.\n")

    # Grid search parameters
    margins = [30, 50, 75, 100]
    max_concurrents = [1, 2, 3]
    cb_sls = [3, 5]
    cb_cooldowns = [6, 12, 24]

    combos = list(product(margins, max_concurrents, cb_sls, cb_cooldowns))
    print(f"Phase 2: Testing {len(combos)} portfolio configurations...\n")

    best = None
    best_score = -999999

    for i, (margin, mc, cb_sl, cb_cd) in enumerate(combos, 1):
        result = run_portfolio_sim(all_trades, margin, mc, cb_sl, cb_cd)
        pnl_pct = result['pnl_pct']
        max_dd = result['max_dd']
        wr = result['win_rate']

        # Score: prioritize profit but penalize high drawdown heavily
        # Bonus for low drawdown, penalty for high
        dd_penalty = max(0, max_dd - 25) * 5  # Heavy penalty above 25% DD
        score = pnl_pct - dd_penalty

        tag = ""
        if pnl_pct >= 80 and max_dd < 30:
            tag = " ★★★ TARGET HIT!"
        elif pnl_pct >= 60 and max_dd < 40:
            tag = " ★★ GOOD"
        elif pnl_pct >= 40:
            tag = " ★ OK"

        print(f"[{i:3d}/{len(combos)}] M=${margin:3d} C={mc} CB={cb_sl}/{cb_cd}h "
              f"→ PnL: {pnl_pct:+7.1f}% DD: {max_dd:5.1f}% WR: {wr:4.1f}% "
              f"Trades: {result['total_trades']:4d}{tag}")

        if score > best_score and pnl_pct > 0:
            best_score = score
            best = {**result, 'margin': margin, 'max_concurrent': mc,
                    'cb_sl': cb_sl, 'cb_cooldown': cb_cd}

    print("\n" + "=" * 60)
    print("  OPTIMAL PORTFOLIO CONFIGURATION FOUND")
    print("=" * 60)
    if best:
        print(f"Margin per trade: ${best['margin']}")
        print(f"Max concurrent:   {best['max_concurrent']}")
        print(f"Circuit breaker:  {best['cb_sl']} SL → {best['cb_cooldown']}h pause")
        print(f"─────────────────────────────────")
        print(f"Net PnL:          {best['pnl_pct']:+.2f}%")
        print(f"Max Drawdown:     {best['max_dd']:.2f}%")
        print(f"Win Rate:         {best['win_rate']:.2f}%")
        print(f"Total Trades:     {best['total_trades']}")
        print(f"Wins/Losses:      {best['wins']}/{best['losses']}")
        print(f"Score:            {best_score:.2f}")
    else:
        print("No profitable configuration found!")
    print("=" * 60)


if __name__ == "__main__":
    main()
