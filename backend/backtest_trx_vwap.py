"""VWAP Scalping Backtest — TRX/USDT 16.03.2026-bugün"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators

print("=" * 70)
print("  VWAP Scalping v1.0 — TRX/USDT Backtest")
print("=" * 70)

# Fetch data
fetcher = DataFetcher()
df = fetcher.fetch_ohlcv('TRX/USDT', '1h', limit=200)
df = add_all_indicators(df)

print(f"Data: {len(df)} bars | {df.index[0]} → {df.index[-1]}")

# 16 Mart 2026 filtresi
start_ts = pd.Timestamp('2026-03-16 00:00:00')
df_test = df[df.index >= start_ts].copy()
print(f"Test period: {len(df_test)} bars from {df_test.index[0]} to {df_test.index[-1]}")

# VWAP calculation
def calc_vwap(df):
    df = df.copy()
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_volume'] = df['tp'] * df['volume']
    df['cum_tp_volume'] = df['tp_volume'].cumsum()
    df['cum_volume'] = df['volume'].cumsum()
    df['vwap'] = df['cum_tp_volume'] / df['cum_volume']
    df['price_diff_sq'] = (df['tp'] - df['vwap']) ** 2
    df['volume_price_diff_sq'] = df['volume'] * df['price_diff_sq']
    df['cum_volume_price_diff_sq'] = df['volume_price_diff_sq'].cumsum()
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
def vwap_signal(df, i, params=None):
    if params is None:
        params = {
            'min_pullback_pct': 0.15,
            'max_pullback_pct': 0.50,
            'min_delta_ratio': 1.1,  # 10% buy pressure
            'min_volume_surge': 1.2,  # 20% above average
            'sl_pct': 0.10,  # Tighter 0.10%
            'tp_pct': 0.20,  # R:R 1:2
            'min_atr_pct': 0.003,
            'breakeven_pct': 0.10,
            'timeout_hours': 4,  # 4 bars on 1h
        }
    
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
    
    p = params
    
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
    
    # LONG setup
    if low <= vwap and close > vwap:
        if delta_ratio < p['min_delta_ratio']:
            return None
        if volume_ratio < p['min_volume_surge']:
            return None
        
        entry = close
        sl = entry * (1 - p['sl_pct'] / 100)
        tp = entry * (1 + p['tp_pct'] / 100)
        
        return {
            'direction': 'LONG',
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'vwap': vwap,
            'delta_ratio': delta_ratio,
            'volume_ratio': volume_ratio,
            'dist_pct': dist_pct,
            'atr': atr
        }
    
    # SHORT setup
    if high >= vwap and close < vwap:
        if delta_ratio > (1 / p['min_delta_ratio']):
            return None
        if volume_ratio < p['min_volume_surge']:
            return None
        
        entry = close
        sl = entry * (1 + p['sl_pct'] / 100)
        tp = entry * (1 - p['tp_pct'] / 100)
        
        return {
            'direction': 'SHORT',
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'vwap': vwap,
            'delta_ratio': delta_ratio,
            'volume_ratio': volume_ratio,
            'dist_pct': dist_pct,
            'atr': atr
        }
    
    return None

# Backtest
print("\n🔄 Backtest running...")
df = calc_vwap(df)
df = calc_delta(df)

# Test parametreleri
params = {
    'min_pullback_pct': 0.15,
    'max_pullback_pct': 0.50,
    'min_delta_ratio': 1.3,
    'min_volume_surge': 1.5,
    'sl_pct': 0.15,
    'tp_pct': 0.30,
    'min_atr_pct': 0.003,
    'breakeven_pct': 0.10,
    'timeout_hours': 2,  # 1h data → 2 bar = 2 hours
}

# Full backtest (200 bars)
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
    
    # Check exit
    if in_trade:
        side = trade['side']
        entry = trade['entry']
        sl = trade['sl']
        tp = trade['tp']
        atr = trade['atr']
        
        pnl_pct = (close - entry) / entry * 100 if side == 'LONG' else (entry - close) / entry * 100
        
        # TP check
        if side == 'LONG' and high >= tp:
            trade['exit'] = tp
            trade['exit_time'] = ts
            trade['pnl_pct'] = (tp - entry) / entry * 100
            trade['pnl'] = (tp - entry)
            trade['reason'] = 'TP'
            trades.append(trade)
            in_trade = False
            cooldown_until = i + 2
        elif side == 'SHORT' and low <= tp:
            trade['exit'] = tp
            trade['exit_time'] = ts
            trade['pnl_pct'] = (entry - tp) / entry * 100
            trade['pnl'] = (entry - tp)
            trade['reason'] = 'TP'
            trades.append(trade)
            in_trade = False
            cooldown_until = i + 2
        # SL check
        elif side == 'LONG' and low <= sl:
            trade['exit'] = sl
            trade['exit_time'] = ts
            trade['pnl_pct'] = (sl - entry) / entry * 100
            trade['pnl'] = (sl - entry)
            trade['reason'] = 'SL'
            trades.append(trade)
            in_trade = False
            cooldown_until = i + 2
        elif side == 'SHORT' and high >= sl:
            trade['exit'] = sl
            trade['exit_time'] = ts
            trade['pnl_pct'] = (entry - sl) / entry * 100
            trade['pnl'] = (entry - sl)
            trade['reason'] = 'SL'
            trades.append(trade)
            in_trade = False
            cooldown_until = i + 2
        # Timeout
        elif i - trade['entry_idx'] >= params['timeout_hours']:
            trade['exit'] = close
            trade['exit_time'] = ts
            trade['pnl_pct'] = pnl_pct
            trade['pnl'] = (close - entry) if side == 'LONG' else (entry - close)
            trade['reason'] = 'TIMEOUT'
            trades.append(trade)
            in_trade = False
            cooldown_until = i + 2
    
    # Check entry
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
                'atr': sig['atr'],
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
    print(f"  Total PnL: ${total_pnl*1000:.2f} (1000 qty)")
    print(f"  Avg Win: ${avg_win*1000:.2f}")
    print(f"  Avg Loss: ${avg_loss*1000:.2f}")
    print(f"  R:R: {avg_win/avg_loss:.2f}:1" if avg_loss > 0 else "  R:R: N/A")
    
    # March 16+ breakdown
    march16_trades = [t for t in trades if t.get('after_march16', False)]
    print(f"\n  🆕 16 March+ Trades: {len(march16_trades)}")
    if march16_trades:
        m16_pnl = sum(t['pnl'] for t in march16_trades)
        m16_wins = sum(1 for t in march16_trades if t['pnl'] > 0)
        print(f"    Win/Loss: {m16_wins}/{len(march16_trades)-m16_wins}")
        print(f"    Total PnL: ${m16_pnl*1000:.2f} (1000 qty)")
    
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
        print(f"    {r:15s} {info['count']:2d} trades | PnL=${info['pnl']*1000:+.2f}")
    
    print(f"\n  Trade Details:")
    for t in trades:
        side = t['side']
        entry = t['entry']
        exit_p = t['exit']
        pnl = t['pnl']
        reason = t['reason']
        entry_t = t['entry_time'].strftime('%m/%d %H:%M')
        exit_t = t['exit_time'].strftime('%m/%d %H:%M')
        m16_mark = '🆕' if t.get('after_march16') else '  '
        
        print(f"  {m16_mark} {entry_t} {side:5s} ${entry:.4f}→${exit_p:.4f} "
              f"PnL=${pnl*1000:+.2f} {reason} → {exit_t}")
else:
    print("  No trades generated!")
    print("  Possible reasons:")
    print("    - Delta ratio threshold (1.3) too strict")
    print("    - Volume surge (1.5x) too strict")
    print("    - Not enough VWAP touches with right conditions")
