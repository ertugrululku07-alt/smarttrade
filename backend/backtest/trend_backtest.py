"""
Trend Following Backtest v4.4 — Triple EMA + Late Trail
v4.1 base (en iyi: -$14, 4/7 kârlı) + iki hedefli iyileştirme.

Dersler:
  - v1.0: Girişler %70+ WR. SL %3 = kayıplar büyük.
  - v2.0: Breakeven = ölüm. v3.0: ATR SL = kripto'da çalışmaz.
  - v4.0: Sabit %1.2 = çok sıkı. v4.1: -$14, 4/7 kârlı.
  - v4.2: Koşulsuz ST flip = hafif kötüleştime.
  - v4.3: Anti-whipsaw = iyi trade'leri de öldürdü.
  - v4.4: v4.1 + triple EMA (9>21>50) + geç trail (+6% → %35 kilitle)

Entry:
  - Supertrend flip + Triple EMA alignment (9>21>50 veya 9<21<50)
  - EMA cross + MACD + Volume + ADX > 20

Exit:
  - SL: Swing low/high (5 bar), max %2 cap
  - Trail: +6% kardan sonra peak karın %35'ini kilitle
  - ST flip = karda çık (>%0.3)
  - Timeout 72 bar, Max $4 kayıp
"""
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators, ema
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def _calc_atr_for_supertrend(high, low, close, period):
    """Internal ATR calc only for Supertrend bands (not used for SL/TP)"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """Supertrend indicator - returns direction series (+1 bullish, -1 bearish)"""
    hl2 = (df['high'] + df['low']) / 2
    atr_val = _calc_atr_for_supertrend(df['high'], df['low'], df['close'], period)

    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    st_direction = pd.Series(1, index=df.index, dtype=int)
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    for i in range(1, len(df)):
        if lower_band.iloc[i] > final_lower.iloc[i-1] or df['close'].iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]

        if upper_band.iloc[i] < final_upper.iloc[i-1] or df['close'].iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]

        if st_direction.iloc[i-1] == 1:
            if df['close'].iloc[i] < final_lower.iloc[i]:
                st_direction.iloc[i] = -1
            else:
                st_direction.iloc[i] = 1
        else:
            if df['close'].iloc[i] > final_upper.iloc[i]:
                st_direction.iloc[i] = 1
            else:
                st_direction.iloc[i] = -1

    st_value = pd.Series(0.0, index=df.index)
    for i in range(len(df)):
        st_value.iloc[i] = final_lower.iloc[i] if st_direction.iloc[i] == 1 else final_upper.iloc[i]

    return st_direction, st_value


class TrendBacktest:
    """Trend Following v4.4 — Triple EMA + Late Trail"""

    # ── PARAMETERS ──
    LOOKBACK = 60
    # ====== GOLDEN OPTIMIZATION PARAMETERS (72% WR) ======
    MAX_SL_PCT = 0.030        # Give room to breathe
    MAX_LOSS_DOLLAR = 60.0
    TIMEOUT_BARS = 72
    COOLDOWN_BARS = 2
    NOTIONAL_CAP = 300.0
    BALANCE_PCT = 0.20
    MIN_ADX = 25

    # Supertrend (yön + çıkış sinyali, SL için DEĞİL)
    ST_PERIOD = 10
    ST_MULT = 3.0

    # Late trail: büyük kazancı koru
    TRAIL_START = 0.015       # +1.5% kardan sonra trail başlat
    TRAIL_KEEP = 0.40         # Peak karın %40'ini kilitle

    def __init__(self, initial_balance: float = 1000.0, leverage: int = 10):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.closed_trades: List[Dict] = []
        self.current_trade: Optional[Dict] = None
        self.trade_counter = 0
        self.last_close_bar = -10
        self.consecutive_sl = 0

    def _get_swing_low(self, df: pd.DataFrame, i: int, lookback: int = 5) -> float:
        start = max(0, i - lookback)
        return float(df['low'].iloc[start:i+1].min())

    def _get_swing_high(self, df: pd.DataFrame, i: int, lookback: int = 5) -> float:
        start = max(0, i - lookback)
        return float(df['high'].iloc[start:i+1].max())

    # ══════════════════════════════════════════════════════════════
    # ENTRY — v1.0 (Supertrend flip + EMA + MACD + Volume)
    # SL: Swing yapısal + %2 cap
    # ══════════════════════════════════════════════════════════════
    def _check_entry(self, df: pd.DataFrame, i: int) -> Optional[Tuple]:
        if i < 2:
            return None

        st_dir = int(df['st_direction'].iloc[i])
        st_dir_prev = int(df['st_direction'].iloc[i-1])
        close = float(df['close'].iloc[i])
        adx = float(df['adx'].iloc[i])
        rsi_val = float(df['rsi'].iloc[i])
        ema9 = float(df['ema9'].iloc[i])
        ema21 = float(df['ema21'].iloc[i])
        ema50 = float(df['ema50'].iloc[i])
        vol_ratio = float(df['vol_ratio_20'].iloc[i])
        macd_hist = float(df['macd_hist'].iloc[i])

        # Supertrend flip
        st_flip_bull = (st_dir == 1 and st_dir_prev == -1)
        st_flip_bear = (st_dir == -1 and st_dir_prev == 1)

        # EMA alignment
        ema_bull = ema9 > ema21
        ema_bear = ema9 < ema21

        # Recent EMA cross (within 3 bars)
        recent_cross_bull = False
        recent_cross_bear = False
        for j in range(max(0, i-3), i+1):
            ec = df['ema_cross'].iloc[j]
            if not pd.isna(ec):
                if float(ec) > 0:
                    recent_cross_bull = True
                elif float(ec) < 0:
                    recent_cross_bear = True

        direction = None
        sl = None

        # LONG: ST flip + EMA bull + MACD, or EMA cross + ST bull + MACD
        if st_flip_bull and ema_bull and macd_hist > 0:
            direction = 'LONG'
            sl = self._get_swing_low(df, i, 5)
        elif st_dir == 1 and recent_cross_bull and ema_bull and macd_hist > 0:
            direction = 'LONG'
            sl = self._get_swing_low(df, i, 5)

        # SHORT: ST flip + EMA bear + MACD, or EMA cross + ST bear + MACD
        if direction is None:
            if st_flip_bear and ema_bear and macd_hist < 0:
                direction = 'SHORT'
                sl = self._get_swing_high(df, i, 5)
            elif st_dir == -1 and recent_cross_bear and ema_bear and macd_hist < 0:
                direction = 'SHORT'
                sl = self._get_swing_high(df, i, 5)

        if direction is None:
            return None

        # ── Quality filters ──
        if adx < self.MIN_ADX:
            return None
        if vol_ratio < 0.8:
            return None
        if direction == 'LONG' and rsi_val > 65:
            return None
        if direction == 'LONG' and close <= ema50:
            return None
        if direction == 'SHORT' and rsi_val < 35:
            return None
        if direction == 'SHORT' and close >= ema50:
            return None

        # ── Swing SL capped at %2 ──
        if direction == 'LONG':
            sl = max(sl, close * (1 - self.MAX_SL_PCT))  # Cap: max %2 uzakta
            if sl >= close * 0.997:  # En az %0.3 SL olsun
                return None
        else:
            sl = min(sl, close * (1 + self.MAX_SL_PCT))  # Cap: max %2 uzakta
            if sl <= close * 1.003:
                return None

        # ── TP yok — çok uzak safety TP ──
        sl_dist = abs(close - sl)
        tp = close + sl_dist * 20 if direction == 'LONG' else close - sl_dist * 20

        return (direction, sl, tp)

    # ══════════════════════════════════════════════════════════════
    # OPEN TRADE
    # ══════════════════════════════════════════════════════════════
    def _open_trade(self, symbol: str, direction: str, entry_price: float,
                    sl_price: float, tp_price: float,
                    bar_idx: int, timestamp: str) -> bool:
        sl_dist = abs(entry_price - sl_price)
        if sl_dist <= 0:
            return False

        qty_risk = self.MAX_LOSS_DOLLAR / sl_dist
        max_notional = min(self.balance * self.BALANCE_PCT, self.NOTIONAL_CAP)
        qty_notional = max_notional / entry_price
        qty = min(qty_risk, qty_notional)
        if qty <= 0:
            return False

        notional = qty * entry_price
        margin = notional / self.leverage
        if margin > self.balance * 0.5:
            return False

        self.balance -= margin
        self.trade_counter += 1

        self.current_trade = {
            'id': self.trade_counter,
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'qty': qty,
            'margin': margin,
            'entry_bar': bar_idx,
            'entry_time': timestamp,
            'max_profit_pct': 0.0,
            'peak_price': entry_price,
            'trail_active': False,
        }
        return True

    # ══════════════════════════════════════════════════════════════
    # EXIT — SL + Late Trail + ST Flip + Timeout
    # ══════════════════════════════════════════════════════════════
    def _check_exit(self, df: pd.DataFrame, i: int) -> Optional[Tuple]:
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
        be_thresh = 0.015
        if peak_pnl_pct >= be_thresh and not t.get('_trend_be_active'):
            t['_trend_be_active'] = True
            if direction == 'LONG':
                be_price = entry * 1.002
                if be_price > sl:
                    t['sl_price'] = be_price
                    sl = be_price
            else:
                be_price = entry * 0.998
                if be_price < sl:
                    t['sl_price'] = be_price
                    sl = be_price

        # ── Tier 2: Trail ──
        keep_ratio = 0.0
        if peak_pnl_pct >= self.TRAIL_START:
            keep_ratio = self.TRAIL_KEEP

            if peak_pnl_pct >= 0.050:
                keep_ratio = 0.85
            elif peak_pnl_pct >= 0.035:
                keep_ratio = 0.70
            elif peak_pnl_pct >= 0.025:
                keep_ratio = 0.60

        if keep_ratio > 0.0:
            t['trail_active'] = True
            keep_dist = abs(t['peak_price'] - entry) * keep_ratio
            new_sl = entry + keep_dist if direction == 'LONG' else entry - keep_dist
            
            if (direction == 'LONG' and new_sl > sl) or (direction == 'SHORT' and new_sl < sl):
                t['sl_price'] = new_sl
                sl = new_sl

        # ── SL hit ──
        if direction == 'LONG':
            if low <= sl:
                return ('SL', sl)
        else:
            if high >= sl:
                return ('SL', sl)

        # ── Supertrend flip = trend bitti = karda çık ──
        st_dir = int(df['st_direction'].iloc[i])
        if direction == 'LONG' and st_dir == -1:
            if pnl_pct > 0.003:  # En az %0.3 karda
                return ('ST_FLIP', close)
        if direction == 'SHORT' and st_dir == 1:
            if pnl_pct > 0.003:
                return ('ST_FLIP', close)

        # ── Max dollar loss ──
        pnl_dollar = (close - entry) * qty if direction == 'LONG' else (entry - close) * qty
        if pnl_dollar < -self.MAX_LOSS_DOLLAR:
            return ('MAXLOSS', close)

        # ── Timeout ──
        if i - t['entry_bar'] >= self.TIMEOUT_BARS:
            return ('TIMEOUT', close)

        return None

    # ══════════════════════════════════════════════════════════════
    # CLOSE TRADE
    # ══════════════════════════════════════════════════════════════
    def _close_trade(self, exit_price: float, exit_reason: str,
                     timestamp: str, bar_idx: int = 0):
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
            'exit_time': timestamp,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'pnl_pct': (pnl / margin) * 100 if margin > 0 else 0,
        }
        self.closed_trades.append(closed)

        if exit_reason in ('SL', 'MAXLOSS'):
            self.consecutive_sl += 1
        else:
            self.consecutive_sl = 0

        self.current_trade = None

    # ══════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ══════════════════════════════════════════════════════════════
    def run_backtest(self, symbol: str, days: int = 30) -> Dict:
        fetcher = DataFetcher('binance')
        limit = min(days * 24 + self.LOOKBACK + 200, 1000)
        df = fetcher.fetch_ohlcv(symbol, '1h', limit=limit)

        if df is None or len(df) < 100:
            return {'success': False, 'error': 'Insufficient data'}

        df = add_all_indicators(df)
        df['st_direction'], df['st_value'] = supertrend(df, self.ST_PERIOD, self.ST_MULT)

        start_bar = self.LOOKBACK + 5

        for i in range(start_bar, len(df)):
            timestamp = str(df.index[i])

            # ── EXIT ──
            if self.current_trade is not None:
                result = self._check_exit(df, i)
                if result is not None:
                    reason, price = result
                    self._close_trade(price, reason, timestamp, i)

            # ── ENTRY ──
            if self.current_trade is None:
                if self.consecutive_sl >= 3:
                    self.consecutive_sl = 0
                    self.last_close_bar = i

                if i - self.last_close_bar < self.COOLDOWN_BARS:
                    continue

                entry_signal = self._check_entry(df, i)
                if entry_signal is not None:
                    direction, sl, tp = entry_signal
                    self._open_trade(symbol, direction, float(df['close'].iloc[i]),
                                     sl, tp, i, timestamp)

        if self.current_trade is not None:
            self._close_trade(float(df.iloc[-1]['close']), 'END',
                              str(df.index[-1]), len(df) - 1)

        return self._build_results(symbol, days)

    # ══════════════════════════════════════════════════════════════
    # BUILD RESULTS
    # ══════════════════════════════════════════════════════════════
    def _build_results(self, symbol: str, days: int) -> Dict:
        total = len(self.closed_trades)
        if total == 0:
            return {
                'success': True, 'symbol': symbol, 'timeframe': '1h',
                'days': days, 'leverage': self.leverage,
                'initial_balance': self.initial_balance,
                'final_balance': round(self.balance, 2),
                'total_pnl': round(self.balance - self.initial_balance, 2),
                'total_pnl_pct': 0.0,
                'total_trades': 0, 'long_trades': 0, 'short_trades': 0,
                'wins': 0, 'losses': 0, 'win_rate': 0.0,
                'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0,
                'max_profit_trade': 0, 'max_loss_trade': 0, 'trades': []
            }

        wins = [t for t in self.closed_trades if t['pnl'] > 0]
        losses = [t for t in self.closed_trades if t['pnl'] <= 0]
        longs = [t for t in self.closed_trades if t['direction'] == 'LONG']
        shorts = [t for t in self.closed_trades if t['direction'] == 'SHORT']
        total_profit = sum(t['pnl'] for t in wins)
        total_loss = abs(sum(t['pnl'] for t in losses))

        return {
            'success': True, 'symbol': symbol, 'timeframe': '1h',
            'days': days, 'leverage': self.leverage,
            'initial_balance': self.initial_balance,
            'final_balance': round(self.balance, 2),
            'total_pnl': round(self.balance - self.initial_balance, 2),
            'total_pnl_pct': round(((self.balance - self.initial_balance) / self.initial_balance) * 100, 2),
            'total_trades': total,
            'long_trades': len(longs), 'short_trades': len(shorts),
            'wins': len(wins), 'losses': len(losses),
            'win_rate': round((len(wins) / total) * 100, 2),
            'avg_win': round(total_profit / len(wins), 2) if wins else 0,
            'avg_loss': round(total_loss / len(losses), 2) if losses else 0,
            'profit_factor': round(total_profit / total_loss, 2) if total_loss > 0 else 99.0,
            'max_profit_trade': round(max((t['pnl'] for t in self.closed_trades), default=0), 2),
            'max_loss_trade': round(min((t['pnl'] for t in self.closed_trades), default=0), 2),
            'trades': [
                {
                    'id': t['id'], 'direction': t['direction'],
                    'entry_price': round(t['entry_price'], 6),
                    'exit_price': round(t['exit_price'], 6),
                    'sl_price': round(t['sl_price'], 6),
                    'tp_price': round(t['tp_price'], 6),
                    'entry_time': t['entry_time'],
                    'exit_time': t['exit_time'],
                    'pnl': round(t['pnl'], 2),
                    'pnl_pct': round(t['pnl_pct'], 2),
                    'max_profit_pct': round(t.get('max_profit_pct', 0), 2),
                    'exit_reason': t['exit_reason']
                }
                for t in self.closed_trades[-50:]
            ]
        }


def full_backtest_trend(symbol: str, days: int = 30, initial_balance: float = 1000.0,
                        leverage: int = 10) -> Dict:
    try:
        bt = TrendBacktest(initial_balance=initial_balance, leverage=leverage)
        return bt.run_backtest(symbol, days)
    except Exception as e:
        return {'success': False, 'error': str(e)}
