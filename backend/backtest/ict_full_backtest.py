"""
ICT/SMC Full Backtest v5.7 — Real ICT + CHoCH Reversal Quality
Uses ict_core.analyze() for proper Smart Money Concepts:
  - Market Structure (BOS/CHoCH)
  - Order Blocks (OB) + FVG proximity check
  - Liquidity Sweeps (SSL/BSL) — strong standalone signal
  - Structural SL via get_sweep_sl()
  - Liquidity TP via get_liquidity_tp()
  - CHoCH reversals require sweep or displacement confirmation
"""
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators
from ai import ict_core
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class ICTFullBacktest:
    """ICT/SMC Backtest — Real Smart Money Analysis via ict_core"""

    # ── PARAMETERS ──
    LOOKBACK = 80             # Bars for ICT analysis window
    MIN_RR = 1.0              # Lowered to allow more entries
    MAX_SL_PCT = 0.030        # 3% max SL for structural room
    MAX_LOSS_DOLLAR = 5.8     # Max loss per trade $
    TIMEOUT_BARS = 48         # Force close after 48h
    COOLDOWN_BARS = 3         # Min bars between trades
    TRAIL_START_PCT = 0.020   # Trail after 2.0% profit
    TRAIL_DIST_PCT = 0.006    # Trail distance 0.6%
    CIRCUIT_BREAKER_SL = 3
    CIRCUIT_BREAKER_BARS = 8
    NOTIONAL_CAP = 300.0
    BALANCE_PCT = 0.17
    POI_PROXIMITY_ATR = 1.8   # How close price must be to OB/FVG (in ATR)
    ENTRY_RANGE_LOOKBACK = 20
    ENTRY_RANGE_TOP_CEIL = 0.85
    ENTRY_RANGE_BOT_FLOOR = 0.15
    ENTRY_MAX_EMA21_EXT = 0.045
    EARLY_CUT_BARS = 10
    EARLY_CUT_R = 0.75
    SL_BUFFER_ATR = 1.0

    def __init__(self, initial_balance: float = 1000.0, leverage: int = 10):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.closed_trades: List[Dict] = []
        self.current_trade: Optional[Dict] = None
        self.trade_counter = 0
        self.last_close_bar = -999
        self.consecutive_sl = 0
        self.cooldown_until = -1

    # ══════════════════════════════════════════════════════════════
    # ATR HELPER
    # ══════════════════════════════════════════════════════════════
    def _get_atr(self, df: pd.DataFrame, i: int) -> float:
        if 'atr' in df.columns:
            val = float(df.iloc[i]['atr'])
            if not np.isnan(val) and val > 0:
                return val
        cp = float(df.iloc[i]['close'])
        return cp * 0.01

    # ══════════════════════════════════════════════════════════════
    # ENTRY — v5.4: BOS/CHoCH + OB/FVG + Sweep + POI quality
    # ══════════════════════════════════════════════════════════════
    def _check_entry(self, df: pd.DataFrame, i: int) -> Optional[Tuple]:
        """
        ICT entry:
          1. Market structure (bullish/bearish/ranging)
          2. BOS or CHoCH confirming direction
          3. Active OB or FVG near price, or liquidity sweep
          4. Candle confirmation
          5. Structural SL + Liquidity TP
          6. R:R >= 1.5 (1.2 for high confluence)
        Returns: (direction, sl, tp) or None
        """
        if i < self.LOOKBACK:
            return None

        # ── Prepare data slice (no look-ahead) ──
        start = max(0, i - self.LOOKBACK + 1)
        df_slice = df.iloc[start:i + 1].copy()
        if len(df_slice) < 40:
            return None

        # ── Run full ICT analysis ──
        analysis = ict_core.analyze(df_slice, '', swing_left=3, swing_right=2)
        ms = analysis.market_structure

        # ── Detect structure breaks ──
        has_bull_bos = bool(analysis.last_bos and analysis.last_bos.direction == 'bullish')
        has_bear_bos = bool(analysis.last_bos and analysis.last_bos.direction == 'bearish')
        has_bull_choch = bool(analysis.last_choch and analysis.last_choch.direction == 'bullish')
        has_bear_choch = bool(analysis.last_choch and analysis.last_choch.direction == 'bearish')
        has_ssl_sweep = (analysis.sweep_detected and analysis.sweep_type == 'ssl_sweep')
        has_bsl_sweep = (analysis.sweep_detected and analysis.sweep_type == 'bsl_sweep')

        # ── Build candidate directions with signal type ──
        # signal_type: 'trend' | 'choch' | 'sweep' | 'ranging'
        candidates = []  # list of (direction, signal_type)

        # A) Trend continuation (BOS in trend) — standard quality
        if ms == 'bullish' and has_bull_bos:
            candidates.append(('LONG', 'trend'))
        if ms == 'bearish' and has_bear_bos:
            candidates.append(('SHORT', 'trend'))

        # B) Trend-aligned CHoCH — standard quality
        if ms == 'bullish' and has_bull_choch:
            if not any(d == 'LONG' for d, _ in candidates):
                candidates.append(('LONG', 'trend'))
        if ms == 'bearish' and has_bear_choch:
            if not any(d == 'SHORT' for d, _ in candidates):
                candidates.append(('SHORT', 'trend'))

        # C) CHoCH reversal (counter-trend) — needs extra confirmation
        if has_bull_choch and not any(d == 'LONG' for d, _ in candidates):
            candidates.append(('LONG', 'choch'))
        if has_bear_choch and not any(d == 'SHORT' for d, _ in candidates):
            candidates.append(('SHORT', 'choch'))

        # D) Sweep reversal (strong standalone)
        if has_ssl_sweep and not any(d == 'LONG' for d, _ in candidates):
            candidates.append(('LONG', 'sweep'))
        if has_bsl_sweep and not any(d == 'SHORT' for d, _ in candidates):
            candidates.append(('SHORT', 'sweep'))

        # E) BOS in ranging
        if ms == 'ranging':
            if has_bull_bos and not any(d == 'LONG' for d, _ in candidates):
                candidates.append(('LONG', 'ranging'))
            if has_bear_bos and not any(d == 'SHORT' for d, _ in candidates):
                candidates.append(('SHORT', 'ranging'))

        if not candidates:
            return None

        # ── Current bar data ──
        cp = float(df_slice['close'].iloc[-1])
        opn = float(df_slice['open'].iloc[-1])
        atr = self._get_atr(df, i)
        proximity = atr * self.POI_PROXIMITY_ATR

        # ── Try each candidate ──
        for direction, signal_type in candidates:
            result = self._try_entry_direction(
                analysis, direction, signal_type, cp, opn, atr, proximity, df_slice)
            if result is not None:
                return result

        return None

    def _try_entry_direction(self, analysis, direction: str,
                             signal_type: str,
                             cp: float, opn: float, atr: float,
                             proximity: float,
                             df_slice: pd.DataFrame) -> Optional[Tuple]:
        """Try to build a valid entry for a given direction and signal type."""

        # ── 0. Entry location / chasing filter ──
        lb = min(self.ENTRY_RANGE_LOOKBACK, len(df_slice))
        loc = df_slice.iloc[-lb:]
        range_high = float(loc['high'].max())
        range_low = float(loc['low'].min())
        if range_high > range_low:
            pos = (cp - range_low) / (range_high - range_low)
            if direction == 'LONG' and pos > self.ENTRY_RANGE_TOP_CEIL:
                return None
            if direction == 'SHORT' and pos < self.ENTRY_RANGE_BOT_FLOOR:
                return None

        ema21 = float(df_slice['close'].astype(float).ewm(span=21, adjust=False).mean().iloc[-1])
        if ema21 > 0:
            if direction == 'LONG' and cp > ema21 * (1 + self.ENTRY_MAX_EMA21_EXT):
                return None
            if direction == 'SHORT' and cp < ema21 * (1 - self.ENTRY_MAX_EMA21_EXT):
                return None

        # ── 1. POI check: OB or FVG near price ──
        ob_type = 'bullish' if direction == 'LONG' else 'bearish'
        active_obs = [ob for ob in analysis.order_blocks
                      if not ob.mitigated and ob.type == ob_type]
        active_fvgs = [f for f in analysis.fvg_zones
                       if not f.filled and f.type == ob_type]

        prox_mult = 1.0
        if signal_type == 'choch':
            prox_mult = 0.75
        elif signal_type == 'sweep':
            prox_mult = 0.90
        eff_proximity = proximity * prox_mult

        near_poi = False
        for ob in active_obs:
            if cp >= ob.bottom - eff_proximity and cp <= ob.top + eff_proximity:
                near_poi = True
                break
        if not near_poi:
            for fvg in active_fvgs:
                if cp >= fvg.bottom - eff_proximity and cp <= fvg.top + eff_proximity:
                    near_poi = True
                    break

        # Sweep alone is strong enough
        has_sweep = False
        if direction == 'LONG' and analysis.sweep_detected and analysis.sweep_type == 'ssl_sweep':
            has_sweep = True
        if direction == 'SHORT' and analysis.sweep_detected and analysis.sweep_type == 'bsl_sweep':
            has_sweep = True

        if not near_poi and not has_sweep:
            return None

        # ── 2. CHoCH reversal quality gate ──
        # Counter-trend CHoCH entries need extra confirmation:
        # either a sweep or displacement in the entry direction
        if signal_type == 'choch' and not has_sweep:
            exp_disp = 'bullish' if direction == 'LONG' else 'bearish'
            if not (analysis.displacement and analysis.displacement_direction == exp_disp):
                return None

        # ── 3. Candle confirmation ──
        if direction == 'LONG':
            if cp < opn:
                if not has_sweep:
                    p_close = float(df_slice['close'].iloc[-2]) if len(df_slice) >= 2 else cp
                    if cp <= p_close:
                        return None
        else:
            if cp > opn:
                if not has_sweep:
                    p_close = float(df_slice['close'].iloc[-2]) if len(df_slice) >= 2 else cp
                    if cp >= p_close:
                        return None

        # ── 4. Structural SL ──
        sl = ict_core.get_sweep_sl(
            direction, analysis.sweep_level, cp, atr,
            analysis.swing_highs, analysis.swing_lows,
            sl_buffer_atr=self.SL_BUFFER_ATR
        )

        max_sl_pct = self.MAX_SL_PCT
        if signal_type == 'trend':
            max_sl_pct = min(0.032, self.MAX_SL_PCT * 1.25)
        elif signal_type == 'choch':
            max_sl_pct = max(0.018, self.MAX_SL_PCT * 0.9)

        if direction == 'LONG':
            sl = max(sl, cp * (1 - max_sl_pct))
            if sl >= cp:
                return None
        else:
            sl = min(sl, cp * (1 + max_sl_pct))
            if sl <= cp:
                return None

        # ── 5. Liquidity TP ──
        tp = ict_core.get_liquidity_tp(
            direction, cp,
            analysis.equal_highs, analysis.equal_lows,
            analysis.swing_highs, analysis.swing_lows
        )

        # ── 6. R:R check ──
        risk = abs(cp - sl)
        reward = abs(tp - cp)
        if risk <= 0:
            return None
        rr = reward / risk
        if rr < self.MIN_RR:
            return None

        return (direction, sl, tp)

    # ══════════════════════════════════════════════════════════════
    # OPEN TRADE — position sizing based on risk
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
            'initial_risk': abs(entry_price - sl_price),
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
    # EXIT — SL/TP/Trail/MaxLoss/Timeout
    # ══════════════════════════════════════════════════════════════
    def _check_exit(self, high: float, low: float, close: float,
                    bar_idx: int) -> Optional[Tuple]:
        t = self.current_trade
        if t is None:
            return None

        entry = t['entry_price']
        sl = t['sl_price']
        tp = t['tp_price']
        direction = t['direction']
        qty = t['qty']

        # Update peak price
        if direction == 'LONG':
            if high > t['peak_price']:
                t['peak_price'] = high
            pnl_pct = (close - entry) / entry
        else:
            if low < t['peak_price']:
                t['peak_price'] = low
            pnl_pct = (entry - close) / entry

        peak_pct = abs(t['peak_price'] - entry) / entry
        t['max_profit_pct'] = max(t['max_profit_pct'], pnl_pct * 100)

        # ── Trailing: after 2.0% profit → SL = peak - 0.6% ──
        if peak_pct >= self.TRAIL_START_PCT:
            t['trail_active'] = True
            if direction == 'LONG':
                trail_sl = t['peak_price'] * (1 - self.TRAIL_DIST_PCT)
                if trail_sl > sl:
                    t['sl_price'] = trail_sl
                    sl = trail_sl
            else:
                trail_sl = t['peak_price'] * (1 + self.TRAIL_DIST_PCT)
                if trail_sl < sl:
                    t['sl_price'] = trail_sl
                    sl = trail_sl

        # ── SL/TP via HIGH/LOW ──
        if direction == 'LONG':
            if low <= sl:
                return ('SL', sl)
            if high >= tp:
                return ('TP', tp)
        else:
            if high >= sl:
                return ('SL', sl)
            if low <= tp:
                return ('TP', tp)

        # ── Early cut: prolonged adverse move without SL touch ──
        held = bar_idx - t.get('entry_bar', bar_idx)
        initial_r = float(t.get('initial_risk', 0.0) or 0.0)
        if held >= self.EARLY_CUT_BARS and initial_r > 0:
            r_now = ((close - entry) / initial_r) if direction == 'LONG' else ((entry - close) / initial_r)
            if r_now <= -abs(self.EARLY_CUT_R):
                return ('EARLY_CUT', close)

        # ── Max dollar loss failsafe ──
        pnl_dollar = (close - entry) * qty if direction == 'LONG' else (entry - close) * qty
        if pnl_dollar < -self.MAX_LOSS_DOLLAR:
            return ('MAXLOSS', close)

        # ── Timeout ──
        if bar_idx - t['entry_bar'] >= self.TIMEOUT_BARS:
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

        if exit_reason in ('SL', 'MAXLOSS', 'EARLY_CUT'):
            self.consecutive_sl += 1
        else:
            self.consecutive_sl = 0

        self.current_trade = None

    # ══════════════════════════════════════════════════════════════
    # MAIN BACKTEST LOOP
    # ══════════════════════════════════════════════════════════════
    def run_backtest(self, symbol: str, days: int = 30, pre_fetched_df: Optional[pd.DataFrame] = None) -> Dict:
        if pre_fetched_df is not None:
            df = pre_fetched_df.copy()
        else:
            fetcher = DataFetcher('binance')
            limit = max(days * 24 + self.LOOKBACK, 200)
            df = fetcher.fetch_ohlcv(symbol, '1h', limit=limit)

        if df is None or len(df) < 100:
            return {'success': False, 'error': 'Insufficient data'}

        df = add_all_indicators(df)
        start_bar = self.LOOKBACK + 5

        for i in range(start_bar, len(df)):
            timestamp = str(df.index[i])
            close = float(df.iloc[i]['close'])
            high = float(df.iloc[i]['high'])
            low = float(df.iloc[i]['low'])

            # ── EXIT ──
            if self.current_trade is not None:
                result = self._check_exit(high, low, close, i)
                if result is not None:
                    reason, price = result
                    self._close_trade(price, reason, timestamp, i)

            # ── ENTRY ──
            if self.current_trade is None:
                if self.consecutive_sl >= self.CIRCUIT_BREAKER_SL:
                    if i < self.cooldown_until:
                        continue
                    else:
                        self.consecutive_sl = 0

                if i - self.last_close_bar < self.COOLDOWN_BARS:
                    continue

                entry_signal = self._check_entry(df, i)
                if entry_signal is not None:
                    direction, sl, tp = entry_signal
                    self._open_trade(symbol, direction, close, sl, tp, i, timestamp)

        # Close remaining
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
                    'margin': round(t.get('margin', 0), 6),
                    'sl_price': round(t['sl_price'], 6),
                    'tp_price': round(t['tp_price'], 6),
                    'entry_time': t['entry_time'],
                    'exit_time': t['exit_time'],
                    'pnl': round(t['pnl'], 2),
                    'pnl_pct': round(t['pnl_pct'], 2),
                    'max_profit_pct': round(t.get('max_profit_pct', 0), 2),
                    'exit_reason': t['exit_reason']
                }
                for t in self.closed_trades
            ]
        }


def full_backtest_ict(symbol: str, days: int = 30, initial_balance: float = 1000.0,
                      leverage: int = 10, pre_fetched_df: Optional[pd.DataFrame] = None) -> Dict:
    try:
        bt = ICTFullBacktest(initial_balance=initial_balance, leverage=leverage)
        return bt.run_backtest(symbol, days, pre_fetched_df=pre_fetched_df)
    except Exception as e:
        import traceback
        return {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
