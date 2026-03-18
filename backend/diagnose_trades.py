"""
Diagnose losing trades — find root cause of poor R:R
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators
from ai.adaptive_backtest import AdaptiveBacktest
import numpy as np

fetcher = DataFetcher('binance')

coins = [
    'BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT','XRP/USDT',
    'LINK/USDT','AVAX/USDT','DOGE/USDT','DOT/USDT','LTC/USDT',
    'ATOM/USDT','UNI/USDT','NEAR/USDT','INJ/USDT','TIA/USDT',
    'RUNE/USDT','AAVE/USDT',
]

all_wins = []
all_losses = []
all_tp_atr = []  # TP trades in ATR units
all_sl_atr = []  # SL trades in ATR units
long_stats = {'trades': 0, 'wins': 0, 'pnl': 0}
short_stats = {'trades': 0, 'wins': 0, 'pnl': 0}
trend_vs_counter = {'with_trend': {'trades': 0, 'wins': 0}, 'against_trend': {'trades': 0, 'wins': 0}}
exit_types = {'TP': 0, 'SL': 0, 'TIMEOUT': 0}
bb_dist_at_entry = []  # How far from BB mid at entry (in ATR)
bars_held_winners = []
bars_held_losers = []

for sym in coins:
    try:
        df = fetcher.fetch_ohlcv(sym, '1h', limit=2160)  # ~3 months
        if df is None or df.empty or len(df) < 100:
            continue
        df = add_all_indicators(df)
    except Exception as e:
        print(f"SKIP {sym}: {e}")
        continue

    engine = AdaptiveBacktest(timeframe='1h', initial_capital=1000, use_simple_strategy=True)
    result = engine.run(df, symbol=sym)

    # Compute EMA50 for trend context
    ema50 = df['close'].ewm(span=50, adjust=False).mean()

    for t in result.trades:
        pnl_dollar = t.pnl_pct / 100 * 1000
        bars = t.bars_held

        # Exit type
        exit_types[t.outcome] = exit_types.get(t.outcome, 0) + 1

        if t.is_winner:
            all_wins.append(pnl_dollar)
            bars_held_winners.append(bars)
            if t.outcome == 'TP':
                all_tp_atr.append(t.pnl_atr)
        else:
            all_losses.append(pnl_dollar)
            bars_held_losers.append(bars)
            if t.outcome == 'SL':
                all_sl_atr.append(t.pnl_atr)

        # Direction stats
        if t.direction == 'LONG':
            long_stats['trades'] += 1
            long_stats['pnl'] += pnl_dollar
            if t.is_winner: long_stats['wins'] += 1
        else:
            short_stats['trades'] += 1
            short_stats['pnl'] += pnl_dollar
            if t.is_winner: short_stats['wins'] += 1

        # Trend context at entry
        try:
            entry_idx = t.bar_index
            if entry_idx >= 50:
                ema50_val = ema50.iloc[entry_idx]
                ema50_prev = ema50.iloc[entry_idx - 10]
                trend_up = ema50_val > ema50_prev  # EMA50 rising = uptrend

                with_trend = (t.direction == 'LONG' and trend_up) or (t.direction == 'SHORT' and not trend_up)
                key = 'with_trend' if with_trend else 'against_trend'
                trend_vs_counter[key]['trades'] += 1
                if t.is_winner:
                    trend_vs_counter[key]['wins'] += 1
        except:
            pass

print("\n" + "="*60)
print("TRADE DIAGNOSTIC REPORT (3-month, 17 coins)")
print("="*60)

total_trades = len(all_wins) + len(all_losses)
print(f"\nTotal trades: {total_trades}")
print(f"Winners: {len(all_wins)} ({100*len(all_wins)/total_trades:.1f}%)")
print(f"Losers: {len(all_losses)} ({100*len(all_losses)/total_trades:.1f}%)")

print(f"\n--- R:R ANALYSIS ---")
avg_win = np.mean(all_wins) if all_wins else 0
avg_loss = np.mean(all_losses) if all_losses else 0
print(f"Avg WIN:  ${avg_win:+.2f}")
print(f"Avg LOSS: ${avg_loss:+.2f}")
print(f"R:R ratio: {abs(avg_win/avg_loss):.2f}:1" if avg_loss != 0 else "N/A")
print(f"Avg TP hit (ATR): {np.mean(all_tp_atr):.2f}" if all_tp_atr else "No TP")
print(f"Avg SL hit (ATR): {np.mean(all_sl_atr):.2f}" if all_sl_atr else "No SL")
print(f"Median WIN: ${np.median(all_wins):+.2f}" if all_wins else "")
print(f"Median LOSS: ${np.median(all_losses):+.2f}" if all_losses else "")

print(f"\n--- EXIT TYPE ---")
for k, v in exit_types.items():
    print(f"  {k}: {v} ({100*v/total_trades:.1f}%)")

print(f"\n--- DIRECTION ---")
for name, s in [("LONG", long_stats), ("SHORT", short_stats)]:
    wr = 100*s['wins']/s['trades'] if s['trades'] > 0 else 0
    print(f"  {name}: {s['trades']}t WR:{wr:.1f}% PnL:${s['pnl']:+.2f}")

print(f"\n--- TREND CONTEXT ---")
for key, s in trend_vs_counter.items():
    wr = 100*s['wins']/s['trades'] if s['trades'] > 0 else 0
    print(f"  {key}: {s['trades']}t WR:{wr:.1f}%")

print(f"\n--- BARS HELD ---")
print(f"  Winners avg: {np.mean(bars_held_winners):.1f} bars" if bars_held_winners else "")
print(f"  Losers avg:  {np.mean(bars_held_losers):.1f} bars" if bars_held_losers else "")

# Distribution of win sizes
if all_wins:
    print(f"\n--- WIN SIZE DISTRIBUTION ---")
    for pct in [10, 25, 50, 75, 90]:
        print(f"  P{pct}: ${np.percentile(all_wins, pct):+.2f}")

if all_losses:
    print(f"\n--- LOSS SIZE DISTRIBUTION ---")
    for pct in [10, 25, 50, 75, 90]:
        print(f"  P{pct}: ${np.percentile(all_losses, pct):+.2f}")

# What we need for breakeven
print(f"\n--- BREAKEVEN ANALYSIS ---")
wr = len(all_wins) / total_trades
print(f"Current WR: {100*wr:.1f}%")
print(f"Current avg W/L: ${avg_win:.2f} / ${abs(avg_loss):.2f}")
needed_wr = abs(avg_loss) / (avg_win + abs(avg_loss)) if (avg_win + abs(avg_loss)) > 0 else 0.5
print(f"Needed WR for breakeven: {100*needed_wr:.1f}%")
print(f"Gap: {100*(needed_wr - wr):+.1f}%")
