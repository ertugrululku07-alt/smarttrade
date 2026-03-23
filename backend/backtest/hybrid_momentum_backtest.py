"""
Zero-Lag Momentum Hybrid Backtester
====================================
Entry (FAST — only 2 conditions):
  1. Direction: EMA9 > EMA21 = LONG, EMA9 < EMA21 = SHORT
  2. Trigger: Price breaks N-bar high/low + Volume above average

Exit (PROVEN — Trend v4.4 system):
  - SL: Swing low/high (5 bar), max cap
  - Breakeven at +1.5%
  - Tiered Trail: +1.5%→40%, +2.5%→60%, +3.5%→70%, +5%→85%
  - Supertrend flip exit
  - Max dollar loss + timeout
"""
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators, ema
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def _calc_supertrend(df, period=10, multiplier=3.0):
    """Supertrend for exit signals only."""
    hl2 = (df['high'] + df['low']) / 2
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1/period, adjust=False).mean()

    upper = hl2 + multiplier * atr_val
    lower = hl2 - multiplier * atr_val
    st_dir = pd.Series(1, index=df.index, dtype=int)
    fu = upper.copy()
    fl = lower.copy()

    for i in range(1, len(df)):
        if lower.iloc[i] > fl.iloc[i-1] or df['close'].iloc[i-1] < fl.iloc[i-1]:
            fl.iloc[i] = lower.iloc[i]
        else:
            fl.iloc[i] = fl.iloc[i-1]
        if upper.iloc[i] < fu.iloc[i-1] or df['close'].iloc[i-1] > fu.iloc[i-1]:
            fu.iloc[i] = upper.iloc[i]
        else:
            fu.iloc[i] = fu.iloc[i-1]
        if st_dir.iloc[i-1] == 1:
            st_dir.iloc[i] = -1 if df['close'].iloc[i] < fl.iloc[i] else 1
        else:
            st_dir.iloc[i] = 1 if df['close'].iloc[i] > fu.iloc[i] else -1

    return st_dir


class MomentumHybridBacktest:
    """Fast momentum entry + proven trailing stop exit."""

    # ── ENTRY PARAMS (TUNABLE) ──
    BREAKOUT_PERIOD = 10       # N-bar high/low breakout
    VOL_MULT = 1.0             # Volume must be >= this * 20-bar avg
    MIN_ADX = 20               # Minimum trend strength

    # ── EXIT PARAMS (PROVEN from Trend v4.4) ──
    MAX_SL_PCT = 0.025         # Max SL distance
    MAX_LOSS_DOLLAR = 50.0     # Max dollar loss per trade
    TIMEOUT_BARS = 72          # Max trade duration
    TRAIL_START = 0.015        # Start trailing at +1.5%
    TRAIL_KEEP = 0.40          # Lock 40% of peak profit

    # ── POSITION SIZING ──
    NOTIONAL_CAP = 300.0
    BALANCE_PCT = 0.20
    COOLDOWN_BARS = 2

    def __init__(self, initial_balance=100000.0, leverage=10):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.closed_trades = []
        self.current_trade = None
        self.trade_counter = 0
        self.last_close_bar = -10

    def _swing_low(self, df, i, n=5):
        return float(df['low'].iloc[max(0, i-n):i+1].min())

    def _swing_high(self, df, i, n=5):
        return float(df['high'].iloc[max(0, i-n):i+1].max())

    # ══════════════════════════════════════════════
    # FAST ENTRY — Only 2 conditions + filters
    # ══════════════════════════════════════════════
    def _check_entry(self, df, i):
        if i < self.BREAKOUT_PERIOD + 2:
            return None

        close = float(df['close'].iloc[i])
        high = float(df['high'].iloc[i])
        low = float(df['low'].iloc[i])
        ema9 = float(df['ema9'].iloc[i])
        ema21 = float(df['ema21'].iloc[i])
        adx = float(df['adx'].iloc[i])
        vol_ratio = float(df['vol_ratio_20'].iloc[i])
        rsi_val = float(df['rsi'].iloc[i])

        # 1. EMA Direction (instant — no waiting for cross)
        ema_bull = ema9 > ema21
        ema_bear = ema9 < ema21

        # 2. Momentum Breakout — price breaks N-bar high/low
        n = self.BREAKOUT_PERIOD
        prev_high = float(df['high'].iloc[max(0, i-n):i].max())
        prev_low = float(df['low'].iloc[max(0, i-n):i].min())

        breakout_bull = close > prev_high
        breakout_bear = close < prev_low

        direction = None
        sl = None

        # LONG: EMA bullish + breakout above N-bar high
        if ema_bull and breakout_bull:
            direction = 'LONG'
            sl = self._swing_low(df, i, 5)
        # SHORT: EMA bearish + breakout below N-bar low
        elif ema_bear and breakout_bear:
            direction = 'SHORT'
            sl = self._swing_high(df, i, 5)

        if direction is None:
            return None

        # Quick filters (no lag)
        if adx < self.MIN_ADX:
            return None
        if vol_ratio < self.VOL_MULT:
            return None
        if direction == 'LONG' and rsi_val > 75:
            return None
        if direction == 'SHORT' and rsi_val < 25:
            return None

        # SL cap
        if direction == 'LONG':
            sl = max(sl, close * (1 - self.MAX_SL_PCT))
            if sl >= close * 0.997:
                return None
        else:
            sl = min(sl, close * (1 + self.MAX_SL_PCT))
            if sl <= close * 1.003:
                return None

        # Safety TP (far away — trail does the work)
        sl_dist = abs(close - sl)
        tp = close + sl_dist * 20 if direction == 'LONG' else close - sl_dist * 20

        return (direction, sl, tp)

    # ══════════════════════════════════════════════
    # OPEN TRADE
    # ══════════════════════════════════════════════
    def _open_trade(self, symbol, direction, entry, sl, tp, bar_idx, ts):
        sl_dist = abs(entry - sl)
        if sl_dist <= 0:
            return False

        qty_risk = self.MAX_LOSS_DOLLAR / sl_dist
        max_notional = min(self.balance * self.BALANCE_PCT, self.NOTIONAL_CAP)
        qty_notional = max_notional / entry
        qty = min(qty_risk, qty_notional)
        if qty <= 0:
            return False

        notional = qty * entry
        margin = notional / self.leverage
        if margin > self.balance * 0.5:
            return False

        self.balance -= margin
        self.trade_counter += 1

        self.current_trade = {
            'id': self.trade_counter,
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry,
            'sl_price': sl,
            'tp_price': tp,
            'qty': qty,
            'margin': margin,
            'entry_bar': bar_idx,
            'entry_time': ts,
            'max_profit_pct': 0.0,
            'peak_price': entry,
            'trail_active': False,
        }
        return True

    # ══════════════════════════════════════════════
    # EXIT — PROVEN Trend v4.4 trailing system
    # ══════════════════════════════════════════════
    def _check_exit(self, df, i):
        t = self.current_trade
        if t is None:
            return None

        entry = t['entry_price']
        sl = t['sl_price']
        direction = t['direction']
        qty = t['qty']
        high = float(df['high'].iloc[i])
        low = float(df['low'].iloc[i])
        close = float(df['close'].iloc[i])

        # Track peak
        if direction == 'LONG':
            if high > t['peak_price']:
                t['peak_price'] = high
            pnl_pct = (close - entry) / entry
            peak_pnl_pct = (t['peak_price'] - entry) / entry
        else:
            if low < t['peak_price']:
                t['peak_price'] = low
            pnl_pct = (entry - close) / entry
            peak_pnl_pct = (entry - t['peak_price']) / entry
        t['max_profit_pct'] = max(t['max_profit_pct'], pnl_pct * 100)

        # ── Tier 1: Breakeven ──
        if peak_pnl_pct >= 0.015 and not t.get('_be_active'):
            t['_be_active'] = True
            if direction == 'LONG':
                be = entry * 1.002
                if be > sl:
                    t['sl_price'] = be
                    sl = be
            else:
                be = entry * 0.998
                if be < sl:
                    t['sl_price'] = be
                    sl = be

        # ── Tier 2: Trailing Stop ──
        keep = 0.0
        if peak_pnl_pct >= self.TRAIL_START:
            keep = self.TRAIL_KEEP
            if peak_pnl_pct >= 0.050:
                keep = 0.85
            elif peak_pnl_pct >= 0.035:
                keep = 0.70
            elif peak_pnl_pct >= 0.025:
                keep = 0.60

        if keep > 0:
            t['trail_active'] = True
            keep_dist = abs(t['peak_price'] - entry) * keep
            new_sl = entry + keep_dist if direction == 'LONG' else entry - keep_dist
            if (direction == 'LONG' and new_sl > sl) or (direction == 'SHORT' and new_sl < sl):
                t['sl_price'] = new_sl
                sl = new_sl

        # ── SL Hit ──
        if direction == 'LONG' and low <= sl:
            return ('SL', sl)
        if direction == 'SHORT' and high >= sl:
            return ('SL', sl)

        # ── Supertrend flip → exit if in profit ──
        st_dir = int(df['st_direction'].iloc[i])
        if direction == 'LONG' and st_dir == -1 and pnl_pct > 0.003:
            return ('ST_FLIP', close)
        if direction == 'SHORT' and st_dir == 1 and pnl_pct > 0.003:
            return ('ST_FLIP', close)

        # ── Max dollar loss ──
        pnl_dollar = (close - entry) * qty if direction == 'LONG' else (entry - close) * qty
        if pnl_dollar < -self.MAX_LOSS_DOLLAR:
            return ('MAXLOSS', close)

        # ── Timeout ──
        if i - t['entry_bar'] >= self.TIMEOUT_BARS:
            return ('TIMEOUT', close)

        return None

    # ══════════════════════════════════════════════
    # CLOSE TRADE
    # ══════════════════════════════════════════════
    def _close_trade(self, exit_price, exit_reason, ts, bar_idx=0):
        t = self.current_trade
        if t is None:
            return
        self.last_close_bar = bar_idx

        entry = t['entry_price']
        qty = t['qty']
        margin = t['margin']

        pnl = (exit_price - entry) * qty if t['direction'] == 'LONG' else (entry - exit_price) * qty
        fees = (entry * qty + exit_price * qty) * 0.0002
        pnl -= fees

        self.balance += margin + pnl

        closed = {
            **t,
            'exit_price': exit_price,
            'exit_time': ts,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'pnl_pct': (pnl / margin) * 100 if margin > 0 else 0,
        }
        self.closed_trades.append(closed)
        self.current_trade = None

    # ══════════════════════════════════════════════
    # MAIN BACKTEST LOOP
    # ══════════════════════════════════════════════
    def run_backtest(self, symbol, days=180):
        fetcher = DataFetcher('binance')
        limit = min(days * 24 + 265, 4600)
        df = fetcher.fetch_ohlcv(symbol, '1h', limit=limit)

        if df is None or len(df) < 100:
            return {'success': False, 'error': 'Insufficient data'}

        df = add_all_indicators(df)
        df['st_direction'] = _calc_supertrend(df, 10, 3.0)

        start_bar = 65

        for i in range(start_bar, len(df)):
            ts = str(df.index[i]) if isinstance(df.index, pd.DatetimeIndex) else str(i)

            # Check exit first
            if self.current_trade:
                exit_result = self._check_exit(df, i)
                if exit_result:
                    self._close_trade(exit_result[1], exit_result[0], ts, i)

            # Check entry
            if self.current_trade is None:
                if i - self.last_close_bar < self.COOLDOWN_BARS:
                    continue
                entry_result = self._check_entry(df, i)
                if entry_result:
                    direction, sl, tp = entry_result
                    close = float(df['close'].iloc[i])
                    self._open_trade(symbol, direction, close, sl, tp, i, ts)

        # Force close remaining
        if self.current_trade:
            last_close = float(df['close'].iloc[-1])
            ts = str(df.index[-1]) if isinstance(df.index, pd.DatetimeIndex) else str(len(df)-1)
            self._close_trade(last_close, 'END', ts, len(df)-1)

        wins = [t for t in self.closed_trades if t['pnl'] > 0]
        losses = [t for t in self.closed_trades if t['pnl'] <= 0]
        total = len(self.closed_trades)
        wr = len(wins) / total * 100 if total > 0 else 0
        net_pnl = sum(t['pnl'] for t in self.closed_trades)

        return {
            'success': True,
            'symbol': symbol,
            'total_trades': total,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': wr,
            'net_pnl': net_pnl,
            'final_balance': self.balance,
            'trades': self.closed_trades,
        }
