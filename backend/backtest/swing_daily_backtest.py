"""
Swing Daily Trend Engine (1D Timeframe)
Catches the massive 900% trends by operating on the Daily chart.
Leverage: 3x (Protects against daily volatility)
Hold time: Weeks to Months
Entry: EMA9/21 cross or Supertrend + 10-Day Breakout
Exit: Supertrend flip (No trailing stop choking)
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

class SwingDailyBacktest:
    def __init__(self, initial_balance=1000.0, leverage=3):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.closed_trades = []
        self.current_trade = None

    def _check_entry(self, df, i):
        if i < 22: return None
        close = float(df['close'].iloc[i])
        st_dir = df['st_direction'].iloc[i]
        # 20-Day Breakout (Donchian Style)
        high_20 = float(df['high'].iloc[max(0, i-20):i].max())
        low_20 = float(df['low'].iloc[max(0, i-20):i].min())
        
        direction = None
        sl = None

        if close > high_20 and st_dir == 1:
            direction = 'LONG'
            sl = float(df['low'].iloc[max(0, i-10):i+1].min())  # 10 Day Low
        elif close < low_20 and st_dir == -1:
            direction = 'SHORT'
            sl = float(df['high'].iloc[max(0, i-10):i+1].max()) # 10 Day High

        if not direction: return None
        return (direction, sl)

    def _check_exit(self, df, i):
        t = self.current_trade
        if not t: return None
        close = float(df['close'].iloc[i])
        st_dir = df['st_direction'].iloc[i]

        # EXIT: Supertrend Flip against our position!
        if t['direction'] == 'LONG' and st_dir == -1:
            return ('ST_FLIP', close)
        elif t['direction'] == 'SHORT' and st_dir == 1:
            return ('ST_FLIP', close)

        # EXIT: 10-Day trailing stop (Donchian Exit)
        low_10 = float(df['low'].iloc[max(0, i-10):i].min())
        high_10 = float(df['high'].iloc[max(0, i-10):i].max())
        
        if t['direction'] == 'LONG' and close < low_10:
            return ('SL', close)
        if t['direction'] == 'SHORT' and close > high_10:
            return ('SL', close)

        return None

    def run_backtest(self, symbol, days=365):
        fetcher = DataFetcher('binance')
        limit = days + 50
        df = fetcher.fetch_ohlcv(symbol, '1d', limit=limit)
        if df is None or len(df) < 50: return {'success': False}

        df['st_direction'] = _calc_supertrend(df, 14, 3.0)

        for i in range(25, len(df)):
            ts = str(df.index[i]) if isinstance(df.index, pd.DatetimeIndex) else str(i)
            if self.current_trade:
                ext = self._check_exit(df, i)
                if ext:
                    t = self.current_trade
                    pnl = (ext[1] - t['entry_price']) * t['qty'] if t['direction'] == 'LONG' else (t['entry_price'] - ext[1]) * t['qty']
                    pnl -= (t['entry_price'] * t['qty'] + ext[1] * t['qty']) * 0.001
                    t['pnl'] = pnl
                    t['exit_price'] = ext[1]
                    t['exit_time'] = ts
                    self.closed_trades.append(t)
                    self.current_trade = None
            if not self.current_trade:
                ent = self._check_entry(df, i)
                if ent:
                    qt = 200.0 / float(df['close'].iloc[i])  # $200 Notional per trade
                    self.current_trade = {
                        'symbol': symbol, 'direction': ent[0], 'entry_price': float(df['close'].iloc[i]),
                        'qty': qt, 'margin': qt * float(df['close'].iloc[i]) / self.leverage,
                        'entry_time': ts, 'entry_bar': i
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
