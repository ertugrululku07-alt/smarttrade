"""
Zero-Lag Momentum Hybrid — Grid Search Optimizer + Portfolio Replay
Tests: breakout period, volume threshold, entry filters + portfolio controls
Goal: 80%+ profitability with minimal drawdown over 180 days
"""
import sys, os
sys.path.append(os.getcwd())

from itertools import product
from datetime import datetime, timedelta
from backtest.hybrid_momentum_backtest import MomentumHybridBacktest
from live_trader import get_all_usdt_pairs

LOG = []
def log(msg):
    print(msg)
    LOG.append(str(msg))


def run_portfolio_sim(all_trades, margin, max_conc, cb_sl, cb_cd):
    """Chronological portfolio replay with risk controls."""
    balance = 1000.0
    open_pos = []
    accepted = 0
    skip_bal = 0
    skip_con = 0
    skip_cb = 0
    peak = balance
    max_dd = 0.0
    wins = 0
    losses = 0
    consec_sl = 0
    cd_until = None

    for tr in all_trades:
        now = tr['entry_time']

        # Close positions that exited
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
                    wins += 1
                    consec_sl = 0
                else:
                    losses += 1
                    consec_sl += 1
                    if consec_sl >= cb_sl:
                        cd_until = op['exit_time']
            else:
                still.append(op)
        open_pos = still

        # Circuit breaker
        if cd_until is not None:
            cd_end = cd_until + timedelta(hours=cb_cd)
            if now < cd_end:
                skip_cb += 1
                continue
            else:
                cd_until = None
                consec_sl = 0

        if len(open_pos) >= max_conc:
            skip_con += 1
            continue

        if balance < margin:
            skip_bal += 1
            continue

        orig_m = tr['margin']
        if orig_m <= 0:
            continue
        scaled_pnl = (tr['pnl'] / orig_m) * margin

        balance -= margin
        open_pos.append({
            'exit_time': tr['exit_time'],
            'margin': margin,
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
            wins += 1
        else:
            losses += 1

    total = wins + losses
    return {
        'balance': balance,
        'pnl': balance - 1000.0,
        'pnl_pct': (balance - 1000.0) / 10.0,  # percentage
        'win_rate': wins / total * 100 if total > 0 else 0,
        'max_dd': max_dd * 100,
        'total': total,
        'wins': wins,
        'losses': losses,
        'accepted': accepted,
        'skip_bal': skip_bal,
        'skip_con': skip_con,
        'skip_cb': skip_cb,
    }


def main():
    symbols = get_all_usdt_pairs()[:10]
    log(f"=== MOMENTUM HYBRID OPTIMIZER ===")
    log(f"Phase 1: Collecting trades from {len(symbols)} coins, 180 days")
    log(f"Breakout periods to test: [5, 8, 10, 15]")
    log(f"Volume thresholds: [0.8, 1.0, 1.2]")
    log(f"ADX minimums: [15, 20, 25]\n")

    # ── PHASE 1: Test multiple entry parameter sets ──
    breakout_periods = [5, 8, 10, 15]
    vol_mults = [0.8, 1.0, 1.2]
    adx_mins = [15, 20, 25]

    entry_combos = list(product(breakout_periods, vol_mults, adx_mins))
    log(f"Testing {len(entry_combos)} entry parameter sets...\n")

    best_overall = None
    best_overall_score = -999999

    for ec_idx, (bp, vm, adx_m) in enumerate(entry_combos, 1):
        log(f"--- Entry Set [{ec_idx}/{len(entry_combos)}]: BP={bp} VM={vm} ADX>={adx_m} ---")

        # Collect all trades with these params
        all_trades = []
        for idx, s in enumerate(symbols, 1):
            try:
                bt = MomentumHybridBacktest(initial_balance=100000.0, leverage=10)
                bt.BREAKOUT_PERIOD = bp
                bt.VOL_MULT = vm
                bt.MIN_ADX = adx_m
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
                        'entry_time': et, 'exit_time': xt,
                        'margin': m, 'pnl': p
                    })
            except Exception as e:
                pass

        all_trades.sort(key=lambda x: (x['entry_time'], x['exit_time']))
        log(f"  Collected {len(all_trades)} trades")

        if len(all_trades) < 10:
            log(f"  Too few trades, skipping\n")
            continue

        # ── Test portfolio configs ──
        margins = [50, 75, 100]
        max_concs = [2, 3]
        cb_sls = [3, 5]
        cb_cds = [12, 24]

        port_combos = list(product(margins, max_concs, cb_sls, cb_cds))
        local_best = None
        local_best_score = -999999

        for margin, mc, cb_sl, cb_cd in port_combos:
            r = run_portfolio_sim(all_trades, margin, mc, cb_sl, cb_cd)
            pnl_pct = r['pnl_pct']
            max_dd = r['max_dd']

            dd_penalty = max(0, max_dd - 25) * 5
            score = pnl_pct - dd_penalty

            if score > local_best_score and pnl_pct > 0:
                local_best_score = score
                local_best = {**r, 'margin': margin, 'max_conc': mc,
                             'cb_sl': cb_sl, 'cb_cd': cb_cd,
                             'bp': bp, 'vm': vm, 'adx_m': adx_m}

        if local_best:
            tag = ""
            if local_best['pnl_pct'] >= 80 and local_best['max_dd'] < 30:
                tag = " *** TARGET HIT!"
            elif local_best['pnl_pct'] >= 60:
                tag = " ** GOOD"
            elif local_best['pnl_pct'] >= 40:
                tag = " * OK"

            log(f"  BEST: M=${local_best['margin']} C={local_best['max_conc']} "
                f"CB={local_best['cb_sl']}/{local_best['cb_cd']}h "
                f"-> PnL: {local_best['pnl_pct']:+.1f}% DD: {local_best['max_dd']:.1f}% "
                f"WR: {local_best['win_rate']:.1f}% T: {local_best['total']}{tag}")

            if local_best_score > best_overall_score:
                best_overall_score = local_best_score
                best_overall = local_best
        else:
            log(f"  No profitable config found")
        log("")

    # ── FINAL RESULTS ──
    log("\n" + "=" * 60)
    log("  OPTIMAL MOMENTUM HYBRID CONFIGURATION")
    log("=" * 60)
    if best_overall:
        log(f"Entry: BP={best_overall['bp']} VM={best_overall['vm']} ADX>={best_overall['adx_m']}")
        log(f"Portfolio: M=${best_overall['margin']} C={best_overall['max_conc']} CB={best_overall['cb_sl']}/{best_overall['cb_cd']}h")
        log(f"-" * 40)
        log(f"Net PnL:       {best_overall['pnl_pct']:+.2f}%")
        log(f"Max Drawdown:  {best_overall['max_dd']:.2f}%")
        log(f"Win Rate:      {best_overall['win_rate']:.2f}%")
        log(f"Total Trades:  {best_overall['total']}")
        log(f"Wins/Losses:   {best_overall['wins']}/{best_overall['losses']}")
        log(f"Score:         {best_overall_score:.2f}")
    else:
        log("No profitable configuration found!")
    log("=" * 60)

    # Write results
    with open("momentum_optimizer_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))


if __name__ == "__main__":
    main()
