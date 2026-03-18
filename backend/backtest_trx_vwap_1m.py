"""VWAP Scalping 1m Backtest — TRX/USDT 16.03.2026-bugün"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators

print("=" * 70)
print("  VWAP Scalping v1.0 — TRX/USDT 1m Backtest")
print("=" * 70)

# 1m data çek (son 2-3 gün ~ 3000-4000 bar)
fetcher = DataFetcher()
print("Fetching 1m data... (may take a moment)")
df = fetcher.fetch_ohlcv('TRX/USDT', '1m', limit=2000)

if df is None or len(df) == 0:
    print("ERROR: No 1m data available")
    sys.exit(1)

df = add_all_indicators(df)
print(f"Data: {len(df)} bars | {df.index[0]} → {df.index[-1]}")

# 16 Mart 2026 filtresi
start_ts = pd.Timestamp('2026-03-16 00:00:00')
df_test = df[df.index >= start_ts].copy()
print(f"Test period: {len(df_test)} 1m bars ({len(df_test)/60:.0f}h) from {df_test.index[0]} to {df_test.index[-1]}")

if len(df_test) < 100:
    print("WARNING: Not enough 1m data for 16 March+, using all available data")
    df_test = df.copy()
    start_ts = df.index[0]

# VWAP calculation (intraday session-based)
def calc_vwap(df):
    df = df.copy()
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_volume'] = df['tp'] * df['volume']
    # Reset VWAP at each day
    df['date'] = df.index.date
    df['cum_tp_volume'] = df.groupby('date')['tp_volume'].cumsum()
    df['cum_volume'] = df.groupby('date')['volume'].cumsum()
    df['vwap'] = df['cum_tp_volume'] / df['cum_volume']
    # SD bands
    df['price_diff_sq'] = (df['tp'] - df['vwap']) ** 2
    df['volume_price_diff_sq'] = df['volume'] * df['price_diff_sq']
    df['cum_volume_price_diff_sq'] = df.groupby('date')['volume_price_diff_sq'].cumsum()
    df['vwap_variance'] = df['cum_volume_price_diff_sq'] / df['cum_volume']
    df['vwap_sd'] = np.sqrt(df['vwap_variance'])
    return df

# Volume delta calculation
def calc_delta(df, window=5):
    df = df.copy()
    df['body'] = df['close'] - df['open']
    df['body_pct'] = df['body'] / df['open']
    df['buy_volume'] = np.where(df['body'] > 0, df['volume'] * abs(df['body_pct']), 0)
    df['sell_volume'] = np.where(df['body'] < 0, df['volume'] * abs(df['body_pct']), 0)
    df['buy_volume_sum'] = df['buy_volume'].rolling(window).sum()
    df['sell_volume_sum'] = df['sell_volume'].rolling(window).sum()
    df['delta_ratio'] = df['buy_volume_sum'] / (df['sell_volume_sum'] + 1e-9)
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-9)
    return df

# VWAP Scalping signal
params = {
    'min_pullback_pct': 0.08,   # Tighter for 1m (0.08%)
    'max_pullback_pct': 0.30,   # Max distance
    'min_delta_ratio': 1.2,     # 20% buy pressure
    'min_volume_surge': 1.3,    # 30% above average
    'sl_pct': 0.08,             # 0.08% SL (8 ticks)
    'tp_pct': 0.16,             # 0.16% TP (16 ticks, R:R 1:2)
    'min_atr_pct': 0.001,       # 0.1% min ATR
    'breakeven_pct': 0.05,      # BE at 0.05%
    'timeout_bars': 10,         # 10 minutes
    'cooldown_bars': 3,         # 3 min between trades
}

def vwap_signal(df, i, p):
    if i < 50:
        return None
    
    row = df.iloc[i]
    close = float(row['close'])
    high = float(row['high'])
    low = float(row['low'])
    vwap = float(row['vwap'])
    delta_ratio = float(row.get('delta_ratio', 1.0))
    volume_ratio = float(row.get('volume_ratio', 1.0))
    atr = float(row.get('atr', close * 0.01))
    
    if np.isnan(vwap) or vwap <= 0:
        return None
    
    # ATR filter
    atr_pct = atr / close
    if atr_pct < p['min_atr_pct']:
        return None
    
    # Distance from VWAP
    dist_pct = abs(close - vwap) / vwap * 100
    if dist_pct > p['max_pullback_pct']:
        return None
    
    # LONG setup: touch VWAP from below, bounce up
    if low <= vwap and close > vwap:
        if delta_ratio < p['min_delta_ratio']:
            return None
        if volume_ratio < p['min_volume_surge']:
            return None
        
        entry = close
        sl = entry * (1 - p['sl_pct'] / 100)
        tp = entry * (1 + p['tp_pct'] / 100)
        
        return {
            'direction': 'LONG', 'entry': entry, 'sl': sl, 'tp': tp,
            'vwap': vwap, 'delta_ratio': delta_ratio, 'volume_ratio': volume_ratio,
            'dist_pct': dist_pct, 'atr': atr
        }
    
    # SHORT setup: touch VWAP from above, reject down
    if high >= vwap and close < vwap:
        if delta_ratio > (1 / p['min_delta_ratio']):
            return None
        if volume_ratio < p['min_volume_surge']:
            return None
        
        entry = close
        sl = entry * (1 + p['sl_pct'] / 100)
        tp = entry * (1 - p['tp_pct'] / 100)
        
        return {
            'direction': 'SHORT', 'entry': entry, 'sl': sl, 'tp': tp,
            'vwap': vwap, 'delta_ratio': delta_ratio, 'volume_ratio': volume_ratio,
            'dist_pct': dist_pct, 'atr': atr
        }
    
    return None

# Calculate indicators
print("Calculating VWAP and delta...")
df = calc_vwap(df)
df = calc_delta(df)

# Backtest
print("Backtesting...")
trades = []
in_trade = False
trade = None
cooldown_until = 0

for i in range(50, len(df)):
    row = df.iloc[i]
    ts = row.name
    close = float(row['close'])
    high = float(row['high'])
    low = float(row['low'])
    
    # Exit check
    if in_trade:
        side = trade['side']
        entry = trade['entry']
        sl = trade['sl']
        tp = trade['tp']
        
        # TP check
        if side == 'LONG' and high >= tp:
            trade['exit'] = tp
            trade['exit_time'] = ts
            trade['pnl'] = (tp - entry)
            trade['pnl_pct'] = (tp - entry) / entry * 100
            trade['reason'] = 'TP'
            trades.append(trade)
            in_trade = False
            cooldown_until = i + params['cooldown_bars']
        elif side == 'SHORT' and low <= tp:
            trade['exit'] = tp
            trade['exit_time'] = ts
            trade['pnl'] = (entry - tp)
            trade['pnl_pct'] = (entry - tp) / entry * 100
            trade['reason'] = 'TP'
            trades.append(trade)
            in_trade = False
            cooldown_until = i + params['cooldown_bars']
        # SL check
        elif side == 'LONG' and low <= sl:
            trade['exit'] = sl
            trade['exit_time'] = ts
            trade['pnl'] = (sl - entry)
            trade['pnl_pct'] = (sl - entry) / entry * 100
            trade['reason'] = 'SL'
            trades.append(trade)
            in_trade = False
            cooldown_until = i + params['cooldown_bars']
        elif side == 'SHORT' and high >= sl:
            trade['exit'] = sl
            trade['exit_time'] = ts
            trade['pnl'] = (entry - sl)
            trade['pnl_pct'] = (entry - sl) / entry * 100
            trade['reason'] = 'SL'
            trades.append(trade)
            in_trade = False
            cooldown_until = i + params['cooldown_bars']
        # Timeout
        elif i - trade['entry_idx'] >= params['timeout_bars']:
            trade['exit'] = close
            trade['exit_time'] = ts
            trade['pnl'] = (close - entry) if side == 'LONG' else (entry - close)
            trade['pnl_pct'] = trade['pnl'] / entry * 100
            trade['reason'] = 'TIMEOUT'
            trades.append(trade)
            in_trade = False
            cooldown_until = i + params['cooldown_bars']
    
    # Entry check
    if not in_trade and i >= cooldown_until:
        sig = vwap_signal(df, i, params)
        if sig:
            trade = {
                'side': sig['direction'],
                'entry': sig['entry'],
                'sl': sig['sl'],
                'tp': sig['tp'],
                'entry_time': ts,
                'entry_idx': i,
                'vwap': sig['vwap'],
                'delta_ratio': sig['delta_ratio'],
                'volume_ratio': sig['volume_ratio'],
                'dist_pct': sig['dist_pct'],
                'after_march16': ts >= start_ts
            }
            in_trade = True

# Results
print(f"\n{'='*70}")
print(f"  BACKTEST RESULTS")
print(f"{'='*70}")
print(f"  Total Trades: {len(trades)}")

if trades:
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]
    total_pnl = sum(t['pnl'] for t in trades)
    avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(abs(t['pnl']) for t in losses) / len(losses) if losses else 0
    
    print(f"  Win/Loss: {len(wins)}/{len(losses)}")
    print(f"  Win Rate: {len(wins)/len(trades)*100:.1f}%")
    print(f"  Total PnL: ${total_pnl*10000:.2f} (10000 qty)")
    print(f"  Avg Win: ${avg_win*10000:.2f}")
    print(f"  Avg Loss: ${avg_loss*10000:.2f}")
    print(f"  R:R: {avg_win/avg_loss:.2f}:1" if avg_loss > 0 else "  R:R: N/A")
    
    # March 16+ breakdown
    march16_trades = [t for t in trades if t.get('after_march16', False)]
    print(f"\n  🆕 16 March+ Trades: {len(march16_trades)}")
    if march16_trades:
        m16_pnl = sum(t['pnl'] for t in march16_trades)
        m16_wins = sum(1 for t in march16_trades if t['pnl'] > 0)
        print(f"    Win/Loss: {m16_wins}/{len(march16_trades)-m16_wins}")
        print(f"    Total PnL: ${m16_pnl*10000:.2f} (10000 qty)")
    
    # Close reason
    reasons = {}
    for t in trades:
        r = t['reason']
        if r not in reasons:
            reasons[r] = {'count': 0, 'pnl': 0}
        reasons[r]['count'] += 1
        reasons[r]['pnl'] += t['pnl']
    
    print(f"\n  Close Reason:")
    for r, info in sorted(reasons.items(), key=lambda x: x[1]['pnl'], reverse=True):
        print(f"    {r:15s} {info['count']:2d} trades | PnL=${info['pnl']*10000:+.2f}")
    
    print(f"\n  Last 10 Trade Details:")
    for t in trades[-10:]:
        side = t['side']
        entry = t['entry']
        exit_p = t['exit']
        pnl = t['pnl']
        reason = t['reason']
        entry_t = t['entry_time'].strftime('%m/%d %H:%M')
        exit_t = t['exit_time'].strftime('%H:%M')
        m16_mark = '🆕' if t.get('after_march16') else '  '
        
        print(f"  {m16_mark} {entry_t} {side:5s} ${entry:.4f}→${exit_p:.4f} "
              f"PnL=${pnl*10000:+6.2f} {reason} → {exit_t}")
else:
    print("  No trades generated!")
    print(f"\n  Last 10 bars VWAP check:")
    for i in range(len(df)-10, len(df)):
        row = df.iloc[i]
        ts = row.name
        close = float(row['close'])
        vwap = float(row['vwap'])
        dist = abs(close - vwap) / vwap * 100
        print(f"    {ts.strftime('%m/%d %H:%M')}: close=${close:.4f} vwap=${vwap:.4f} dist={dist:.2f}%")
