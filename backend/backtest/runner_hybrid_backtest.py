"""
Runner Engine - Focus on Mega Trends
Allows massive 20-40% price moves by using an extremely wide trailing stop (8% from peak).
Entry: Fast Momentum (15 Bar breakout + Volume + ADX)
Exit: Breakeven at 2%, then lock in with 8% distance from peak.
"""
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators
import pandas as pd
import numpy as np

def _calc_supertrend(df, period=10, multiplier=3.0):
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

class RunnerBacktest:
    BREAKOUT_PERIOD = 15
    VOL_MULT = 1.0
    MIN_ADX = 20
    MAX_SL_PCT = 0.03
    MAX_LOSS_DOLLAR = 50.0

    def __init__(self, initial_balance=100000.0, leverage=10):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.closed_trades = []
        self.current_trade = None
        self.trade_counter = 0

    def _swing_low(self, df, i, n=5):
        return float(df['low'].iloc[max(0, i-n):i+1].min())

    def _swing_high(self, df, i, n=5):
        return float(df['high'].iloc[max(0, i-n):i+1].max())

    def _check_entry(self, df, i):
        if i < self.BREAKOUT_PERIOD + 2: return None
        close = float(df['close'].iloc[i])
        ema9 = float(df['ema9'].iloc[i])
        ema21 = float(df['ema21'].iloc[i])
        adx = float(df['adx'].iloc[i])
        vol_ratio = float(df['vol_ratio_20'].iloc[i])
        rsi_val = float(df['rsi'].iloc[i])

        n = self.BREAKOUT_PERIOD
        prev_high = float(df['high'].iloc[max(0, i-n):i].max())
        prev_low = float(df['low'].iloc[max(0, i-n):i].min())

        direction = None
        sl = None

        if ema9 > ema21 and close > prev_high:
            direction = 'LONG'
            sl = self._swing_low(df, i, 5)
        elif ema9 < ema21 and close < prev_low:
            direction = 'SHORT'
            sl = self._swing_high(df, i, 5)

        if not direction: return None
        if adx < self.MIN_ADX: return None
        if vol_ratio < self.VOL_MULT: return None
        
        # ── CHOP FILTERS (Simulating Order Book depth check) ──
        # 1. Efficiency Ratio: if < 0.35, market is chopping (noise)
        er = float(df.get('efficiency_ratio', pd.Series([1.0])).iloc[i])
        if not pd.isna(er) and er < 0.35:
            return None
            
        # 2. BB Squeeze: don't break out if bands are flat/dead
        bb_width = float(df.get('bb_width', pd.Series([1.0])).iloc[i])
        if not pd.isna(bb_width) and bb_width < 0.02: # Needs at least 2% band width
            return None

        if direction == 'LONG' and rsi_val > 70: return None
        if direction == 'SHORT' and rsi_val < 30: return None

        if direction == 'LONG':
            sl = max(sl, close * (1 - self.MAX_SL_PCT))
            if sl >= close * 0.997: return None
        else:
            sl = min(sl, close * (1 + self.MAX_SL_PCT))
            if sl <= close * 1.003: return None

        return (direction, sl, close*0.01) # dummy TP

    def _check_exit(self, df, i):
        t = self.current_trade
        if not t: return None

        entry = t['entry_price']
        sl = t['sl_price']
        direction = t['direction']
        high = float(df['high'].iloc[i])
        low = float(df['low'].iloc[i])
        close = float(df['close'].iloc[i])

        if direction == 'LONG':
            if high > t['peak_price']: t['peak_price'] = high
            peak_pnl_pct = (t['peak_price'] - entry) / entry
        else:
            if low < t['peak_price']: t['peak_price'] = low
            peak_pnl_pct = (entry - t['peak_price']) / entry

        # ── TIER 1: Breakeven at +2% ──
        if peak_pnl_pct >= 0.02 and not t.get('_be_active'):
            t['_be_active'] = True
            be = entry * 1.002 if direction == 'LONG' else entry * 0.998
            if (direction == 'LONG' and be > sl) or (direction == 'SHORT' and be < sl):
                t['sl_price'] = be
                sl = be

        # ── TIER 2: WIDE TRAIL (LET IT RIDE) ──
        # Gives 8% breathing room from the absolute peak!
        TRAIL_DIST = 0.08
        if peak_pnl_pct >= 0.05:
            if direction == 'LONG':
                new_sl = t['peak_price'] * (1 - TRAIL_DIST)
                if new_sl > sl: t['sl_price'] = new_sl; sl = new_sl
            else:
                new_sl = t['peak_price'] * (1 + TRAIL_DIST)
                if new_sl < sl: t['sl_price'] = new_sl; sl = new_sl

        # ── EXITS ──
        if direction == 'LONG' and low <= sl: return ('SL', sl)
        if direction == 'SHORT' and high >= sl: return ('SL', sl)

        # Timeout 144 bars (6 days)
        if i - t['entry_bar'] >= 144: return ('TIMEOUT', close)
        return None

    def run_backtest(self, symbol, days=180):
        fetcher = DataFetcher('binance')
        limit = min(days * 24 + 100, 4600)
        df = fetcher.fetch_ohlcv(symbol, '1h', limit=limit)
        if df is None or len(df) < 100: return {'success': False}

        df = add_all_indicators(df)
        df['st_direction'] = _calc_supertrend(df, 10, 3.0)

        for i in range(65, len(df)):
            ts = str(df.index[i]) if isinstance(df.index, pd.DatetimeIndex) else str(i)
            if self.current_trade:
                ext = self._check_exit(df, i)
                if ext:
                    t = self.current_trade
                    pnl = (ext[1] - t['entry_price']) * t['qty'] if t['direction'] == 'LONG' else (t['entry_price'] - ext[1]) * t['qty']
                    pnl -= (t['entry_price'] * t['qty'] + ext[1] * t['qty']) * 0.0002
                    t['pnl'] = pnl
                    t['exit_price'] = ext[1]
                    t['exit_time'] = ts
                    self.closed_trades.append(t)
                    self.current_trade = None
            if not self.current_trade:
                ent = self._check_entry(df, i)
                if ent:
                    qt = 50.0 / abs(ent[1] - float(df['close'].iloc[i]))
                    self.current_trade = {
                        'symbol': symbol, 'direction': ent[0], 'entry_price': float(df['close'].iloc[i]),
                        'sl_price': ent[1], 'qty': qt, 'margin': (qt*float(df['close'].iloc[i]))/10,
                        'entry_time': ts, 'entry_bar': i, 'peak_price': float(df['close'].iloc[i])
                    }
        
        if self.current_trade:
            t = self.current_trade
            ext_price = float(df['close'].iloc[-1])
            pnl = (ext_price - t['entry_price']) * t['qty'] if t['direction'] == 'LONG' else (t['entry_price'] - ext_price) * t['qty']
            t['pnl'] = pnl
            t['exit_time'] = str(df.index[-1])
            self.closed_trades.append(t)
            self.current_trade = None
            
        return {'success': True, 'trades': self.closed_trades}
