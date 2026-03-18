"""
VWAP Scalping Strategy v1.0 — Order Flow + VWAP Mean Reversion

Crypto adaptation of ES/NQ order flow scalping:
- VWAP + 1-2 SD bands (intraday anchor)
- Volume delta (buy vs sell pressure)
- Price pullback to VWAP
- Tight stops (0.1-0.2% or 2-3 ticks)
- Quick targets (0.2-0.4% or 4-8 ticks)
- High frequency, low hold time (<30 min)

Original ES/NQ concept:
- 512-tick/1-min chart + footprint/DOM
- VWAP pullback + positive delta + absorption
- SL: 1-2 tick below VWAP
- TP: 4-8 tick (R:R 1:2+)
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime, timedelta


class VWAPScalpingMixin:
    """VWAP Scalping Strategy — Order Flow + Mean Reversion"""

    # ══════════════════════════════════════════════════════════════════════════
    # PARAMETERS
    # ══════════════════════════════════════════════════════════════════════════

    _VWAP_PARAMS = {
        # VWAP calculation
        'vwap_anchor': 'session',  # 'session' or 'rolling'
        'vwap_rolling_bars': 100,  # if rolling
        'sd_multiplier_1': 1.0,    # First SD band
        'sd_multiplier_2': 2.0,    # Second SD band
        
        # Entry conditions
        'min_pullback_pct': 0.15,  # Min % pullback to VWAP (crypto: 0.15%)
        'max_pullback_pct': 0.50,  # Max % pullback (reject if too far)
        'min_delta_ratio': 1.3,    # Buy/sell volume ratio (1.3 = 30% more buys)
        'min_volume_surge': 1.5,   # Volume vs 20-bar avg
        
        # Position sizing & risk
        'max_notional': 200.0,     # Max position size
        'sl_pct': 0.15,            # Stop loss % (tight: 0.15%)
        'tp_pct': 0.30,            # Take profit % (R:R 1:2)
        'max_loss_cap': 5.0,       # Max loss per trade
        
        # Exit management
        'breakeven_pct': 0.10,     # Move SL to BE at 0.10% profit
        'timeout_minutes': 30,     # Max hold time
        'cooldown_bars': 2,        # Bars between trades
        
        # Filters
        'min_atr_pct': 0.003,      # Min volatility (0.3%)
        'max_spread_pct': 0.05,    # Max bid-ask spread (0.05%)
    }

    # ──────────────────────────── State helpers ─────────────────────────────

    def _vwap_init_state(self):
        """Lazy init for VWAP state variables."""
        if not hasattr(self, '_vwap_last_trade_time'):
            self._vwap_last_trade_time = 0
        if not hasattr(self, '_vwap_consecutive_sl'):
            self._vwap_consecutive_sl = 0
        if not hasattr(self, '_vwap_recent_outcomes'):
            self._vwap_recent_outcomes = []

    def _vwap_record_outcome(self, outcome: str, symbol: str):
        """Record trade outcome for stats."""
        self._vwap_recent_outcomes.append({
            'outcome': outcome,
            'symbol': symbol,
            'time': time.time()
        })
        # Keep last 20
        if len(self._vwap_recent_outcomes) > 20:
            self._vwap_recent_outcomes.pop(0)
        
        if outcome == 'SL':
            self._vwap_consecutive_sl += 1
        else:
            self._vwap_consecutive_sl = 0

    # ══════════════════════════════════════════════════════════════════════════
    # VWAP CALCULATION
    # ══════════════════════════════════════════════════════════════════════════

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP + SD bands.
        
        VWAP = Σ(Price × Volume) / Σ(Volume)
        SD = sqrt(Σ(Volume × (Price - VWAP)²) / Σ(Volume))
        """
        df = df.copy()
        
        # Typical price
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        
        # VWAP calculation (session-based = cumulative from start)
        df['tp_volume'] = df['tp'] * df['volume']
        df['cum_tp_volume'] = df['tp_volume'].cumsum()
        df['cum_volume'] = df['volume'].cumsum()
        df['vwap'] = df['cum_tp_volume'] / df['cum_volume']
        
        # Standard deviation
        df['price_diff_sq'] = (df['tp'] - df['vwap']) ** 2
        df['volume_price_diff_sq'] = df['volume'] * df['price_diff_sq']
        df['cum_volume_price_diff_sq'] = df['volume_price_diff_sq'].cumsum()
        df['vwap_variance'] = df['cum_volume_price_diff_sq'] / df['cum_volume']
        df['vwap_sd'] = np.sqrt(df['vwap_variance'])
        
        # SD bands
        p = self._VWAP_PARAMS
        df['vwap_upper_1'] = df['vwap'] + df['vwap_sd'] * p['sd_multiplier_1']
        df['vwap_lower_1'] = df['vwap'] - df['vwap_sd'] * p['sd_multiplier_1']
        df['vwap_upper_2'] = df['vwap'] + df['vwap_sd'] * p['sd_multiplier_2']
        df['vwap_lower_2'] = df['vwap'] - df['vwap_sd'] * p['sd_multiplier_2']
        
        return df

    def _calculate_volume_delta(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Estimate buy/sell volume delta (crypto approximation).
        
        Since we don't have footprint/DOM data:
        - Up candle (close > open) → assume buy pressure
        - Down candle (close < open) → assume sell pressure
        - Weight by candle body size and volume
        """
        df = df.copy()
        
        # Candle direction
        df['body'] = df['close'] - df['open']
        df['body_pct'] = df['body'] / df['open']
        
        # Estimate buy/sell volume
        # Positive body = buy volume, negative = sell volume
        df['buy_volume'] = np.where(df['body'] > 0, 
                                     df['volume'] * abs(df['body_pct']), 
                                     0)
        df['sell_volume'] = np.where(df['body'] < 0, 
                                      df['volume'] * abs(df['body_pct']), 
                                      0)
        
        # Rolling delta
        df['buy_volume_sum'] = df['buy_volume'].rolling(window).sum()
        df['sell_volume_sum'] = df['sell_volume'].rolling(window).sum()
        df['delta_ratio'] = df['buy_volume_sum'] / (df['sell_volume_sum'] + 1e-9)
        
        # Volume surge
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-9)
        
        return df

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL GENERATION
    # ══════════════════════════════════════════════════════════════════════════

    def _vwap_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        VWAP Scalping signal logic.
        
        LONG setup:
        1. Price pulled back to VWAP (within 0.15-0.50%)
        2. Positive delta (buy pressure > sell pressure)
        3. Volume surge (>1.5x average)
        4. Price bouncing off VWAP (close > VWAP)
        
        SHORT setup: mirror
        """
        self._vwap_init_state()
        
        if len(df) < 50:
            return None
        
        # Calculate VWAP + delta
        df = self._calculate_vwap(df)
        df = self._calculate_volume_delta(df)
        
        i = len(df) - 1
        row = df.iloc[i]
        
        close = float(row['close'])
        high = float(row['high'])
        low = float(row['low'])
        vwap = float(row['vwap'])
        vwap_sd = float(row['vwap_sd'])
        delta_ratio = float(row.get('delta_ratio', 1.0))
        volume_ratio = float(row.get('volume_ratio', 1.0))
        
        p = self._VWAP_PARAMS
        
        # Basic checks
        if np.isnan(vwap) or vwap <= 0:
            return None
        
        # ATR filter
        atr = float(row.get('atr', close * 0.01))
        atr_pct = atr / close
        if atr_pct < p['min_atr_pct']:
            return None
        
        # Distance from VWAP
        dist_pct = abs(close - vwap) / vwap * 100
        
        # Check if price near VWAP
        if dist_pct > p['max_pullback_pct']:
            return None
        
        # LONG setup
        if low <= vwap and close > vwap:  # Touched VWAP and bounced
            # Check delta (buy pressure)
            if delta_ratio < p['min_delta_ratio']:
                return None
            
            # Check volume surge
            if volume_ratio < p['min_volume_surge']:
                return None
            
            # Entry, SL, TP
            entry = close
            sl = entry * (1 - p['sl_pct'] / 100)
            tp = entry * (1 + p['tp_pct'] / 100)
            
            return {
                'direction': 'LONG',
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'vwap': vwap,
                'vwap_sd': vwap_sd,
                'delta_ratio': delta_ratio,
                'volume_ratio': volume_ratio,
                'dist_pct': dist_pct,
                'atr': atr
            }
        
        # SHORT setup
        if high >= vwap and close < vwap:  # Touched VWAP and rejected
            # Check delta (sell pressure)
            if delta_ratio > (1 / p['min_delta_ratio']):  # Inverted for SHORT
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
                'vwap_sd': vwap_sd,
                'delta_ratio': delta_ratio,
                'volume_ratio': volume_ratio,
                'dist_pct': dist_pct,
                'atr': atr
            }
        
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # SCANNING & EXECUTION
    # ══════════════════════════════════════════════════════════════════════════

    def _vwap_scan(self, fetcher):
        """Scan all symbols for VWAP scalping opportunities."""
        self._vwap_init_state()
        
        # Cooldown check
        p = self._VWAP_PARAMS
        if time.time() - self._vwap_last_trade_time < p['cooldown_bars'] * 60:
            return
        
        # Circuit breaker: 3 consecutive SL → pause
        if self._vwap_consecutive_sl >= 3:
            self.log("[VWAP] Circuit breaker: 3 consecutive SL, pausing")
            return
        
        # Scan symbols
        candidates = []
        for symbol in getattr(self, 'scanned_symbols', getattr(self, 'symbols', [])):
            try:
                df = fetcher.fetch_ohlcv(symbol, '1m', limit=200)
                if df is None or len(df) < 50:
                    continue
                
                from backtest.signals import add_all_indicators
                df = add_all_indicators(df)
                
                sig = self._vwap_signal(df)
                if sig:
                    sig['symbol'] = symbol
                    sig['close'] = float(df['close'].iloc[-1])
                    candidates.append(sig)
            except Exception as e:
                self.log(f"[VWAP] Error scanning {symbol}: {e}")
        
        if not candidates:
            return
        
        # Select best candidate (highest delta_ratio)
        candidates.sort(key=lambda x: x['delta_ratio'], reverse=True)
        best = candidates[0]
        
        self.log(f"[VWAP] Signal: {best['symbol']} {best['direction']} "
                 f"delta={best['delta_ratio']:.2f} vol={best['volume_ratio']:.2f}")
        
        # Position sizing
        symbol = best['symbol']
        direction = best['direction']
        entry = best['entry']
        sl = best['sl']
        tp = best['tp']
        
        max_notional = min(p['max_notional'], self.balance * 0.05)
        qty = max_notional / entry
        
        # Risk clamp
        sl_dist = abs(entry - sl)
        if sl_dist > 0:
            max_qty_by_risk = p['max_loss_cap'] / sl_dist
            qty = min(qty, max_qty_by_risk)
        
        # Open trade
        with self.trades_lock:
            tid = self._open_locked(
                symbol=symbol,
                side=direction,
                price=entry,
                multiplier=1.0,
                tp_price=tp,
                sl_price=sl,
                signal_result={
                    'strategy': 'vwap_scalp_v1.0',
                    'regime': 'scalping',
                    'entry_type': 'market',
                    'delta_ratio': best['delta_ratio'],
                    'volume_ratio': best['volume_ratio'],
                    'dist_pct': best['dist_pct'],
                },
                absolute_qty=qty,
                atr=best['atr'],
                logger_id=f"{symbol}_{direction}_{int(time.time())}"
            )
            
            if tid:
                # Store VWAP data
                for t in self.open_trades:
                    if t.get('id') == tid:
                        t['_vwap'] = best['vwap']
                        t['_vwap_sd'] = best['vwap_sd']
                        break
                
                self._vwap_last_trade_time = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # EXIT MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════════

    def _check_vwap_exit(self, t, current_price):
        """
        VWAP Scalping exit logic:
        1. Max loss cap
        2. TP hit
        3. SL hit
        4. Breakeven: profit > 0.10% → SL to BE
        5. Timeout: 30 min
        """
        try:
            strat = t.get('strategy', '')
            if 'vwap_scalp' not in strat:
                return
            
            side = t['side']
            entry = t.get('entry_price', 0)
            tp = t.get('tp_price', 0)
            sl = t.get('sl_price', 0)
            symbol = t.get('symbol', '')
            p = self._VWAP_PARAMS
            
            if tp <= 0 or sl <= 0 or entry <= 0:
                return
            
            pnl_dollar, pnl_pct = self._compute_pnl(t, current_price)
            
            # Max loss cap
            if pnl_dollar < -p['max_loss_cap']:
                self._close_all_locked([t], symbol, current_price, "VWAP_MAXLOSS")
                self._vwap_record_outcome('SL', symbol)
                return
            
            # TP check
            hit_tp = (side == 'LONG' and current_price >= tp) or \
                     (side == 'SHORT' and current_price <= tp)
            if hit_tp:
                self._close_all_locked([t], symbol, current_price, "VWAP_TP")
                self._vwap_record_outcome('TP', symbol)
                return
            
            # SL check
            hit_sl = (side == 'LONG' and current_price <= sl) or \
                     (side == 'SHORT' and current_price >= sl)
            if hit_sl:
                reason = "VWAP_SL" if pnl_pct < 0 else "VWAP_TRAIL_SL"
                self._close_all_locked([t], symbol, current_price, reason)
                if pnl_pct < 0:
                    self._vwap_record_outcome('SL', symbol)
                else:
                    self._vwap_record_outcome('TP', symbol)
                return
            
            # Breakeven
            if pnl_pct >= p['breakeven_pct'] and not t.get('_vwap_be_set'):
                if side == 'LONG':
                    new_sl = entry + (entry * 0.001)
                    if new_sl > t['sl_price']:
                        t['sl_price'] = new_sl
                        t['_vwap_be_set'] = True
                        self.log(f"[VWAP BE] {symbol} SL→BE {new_sl:.6f}")
                else:
                    new_sl = entry - (entry * 0.001)
                    if new_sl < t['sl_price']:
                        t['sl_price'] = new_sl
                        t['_vwap_be_set'] = True
                        self.log(f"[VWAP BE] {symbol} SL→BE {new_sl:.6f}")
            
            # Timeout
            entry_ts = t.get('entry_timestamp', 0)
            if entry_ts > 0:
                minutes_held = (time.time() - entry_ts) / 60
                if minutes_held > p['timeout_minutes']:
                    self._close_all_locked([t], symbol, current_price, "VWAP_TIMEOUT")
                    self._vwap_record_outcome('TIMEOUT', symbol)
                    return
        
        except Exception as e:
            self.log(f"[ERR] _check_vwap_exit {t.get('symbol','?')}: {e}")
