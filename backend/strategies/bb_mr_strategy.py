"""
BB Mean Reversion v7.1 Strategy — Final Build
Calibrated quality thresholds, enhanced rejection scoring, refined exit logic.
Multi-factor quality: divergence, stoch RSI, OBV, ADX penalties, session weighting.

Mixin class — mix into LivePaperTrader to add BB MR signal, scan, and exit logic.

Requires from host class:
    self.balance, self.leverage
    self.open_trades, self.pending_orders, self.trades_lock
    self.scanned_symbols, self.max_open_trades_limit
    self.is_running
    self.log(), self._compute_pnl(), self._close_all_locked()
    self._open_locked(), self._force_save()
    self._check_loss_limits(), self._calc_partial_pnl()
    self._check_correlation()  # from ICTSMCStrategyMixin (always co-loaded)
    self._bb_consecutive_sl, self._bb_cooldown_until, etc.
    self._user_max_notional, self._user_max_loss_cap
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone

from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators


class BBMRStrategyMixin:
    """BB Mean Reversion v7.1 Final — Calibrated thresholds."""

    # ══════════════════════════════════════════════════════════════════════════
    # v7.1 BB MEAN REVERSION — Enhanced rejection + multi-factor quality
    # ══════════════════════════════════════════════════════════════════════════

    _BB_MR_PARAMS = {
        'tp_min': 0.5,
        'cooldown_bars': 3,
        'sl_cooldown': 18,
        'rsi_long_flat': 42,
        'rsi_long_counter': 35,
        'rsi_short_flat': 58,
        'rsi_short_counter': 65,
        'min_rej_score': 4.0,
        'sl_min_atr': 1.0,
        'sl_max_atr': 1.8,
        'min_rr': 1.5,
        'min_quality_pre': 25,
        'min_quality_base': 35,
        'min_quality_counter': 48,
        'timeout_hours': 18,
        'max_candle_range_atr': 1.8,
        'max_adx': 35,
        'adx_div_required': 30,
    }

    # ──────────────────────────── State helpers ─────────────────────────────

    def _bb_mr_init_state(self):
        """Lazy init for all BB MR state variables."""
        if not hasattr(self, '_bb_last_trade_time'):
            self._bb_last_trade_time = 0
        if not hasattr(self, '_bb_consecutive_sl'):
            self._bb_consecutive_sl = 0
        if not hasattr(self, '_bb_cooldown_until'):
            self._bb_cooldown_until = 0
        if not hasattr(self, '_bb_recent_outcomes'):
            self._bb_recent_outcomes = []
        if not hasattr(self, '_bb_symbol_cooldown'):
            self._bb_symbol_cooldown = {}
        if not hasattr(self, '_bb_symbol_stats'):
            self._bb_symbol_stats = {}

    def _bb_mr_get_symbol_stats(self, symbol: str) -> Dict:
        self._bb_mr_init_state()
        return self._bb_symbol_stats.get(symbol, {
            'total': 0, 'wins': 0, 'losses': 0,
            'avg_r': 0.0, 'last_outcomes': []
        })

    def _bb_mr_update_symbol_stats(self, symbol: str, outcome: str,
                                    realized_r: float = 0):
        self._bb_mr_init_state()
        stats = self._bb_symbol_stats.get(symbol, {
            'total': 0, 'wins': 0, 'losses': 0,
            'avg_r': 0.0, 'last_outcomes': []
        })
        stats['total'] += 1
        if outcome == 'TP':
            stats['wins'] += 1
        elif outcome == 'SL':
            stats['losses'] += 1
        stats['last_outcomes'].append(outcome)
        if len(stats['last_outcomes']) > 20:
            stats['last_outcomes'].pop(0)
        if stats['total'] > 0:
            stats['avg_r'] = ((stats['avg_r'] * (stats['total'] - 1)
                               + realized_r) / stats['total'])
        self._bb_symbol_stats[symbol] = stats

    def _bb_mr_is_symbol_blacklisted(self, symbol: str) -> bool:
        stats = self._bb_mr_get_symbol_stats(symbol)
        if stats['total'] < 12:
            return False
        recent = stats['last_outcomes'][-12:]
        if len(recent) < 12:
            return False
        wr = sum(1 for o in recent if o == 'TP') / len(recent)
        return wr < 0.25 and stats['avg_r'] < 0

    def _bb_mr_check_session(self) -> Tuple[bool, str]:
        hour = datetime.now(timezone.utc).hour
        if 0 <= hour < 7:
            return True, 'asia'
        elif 7 <= hour < 13:
            return True, 'london'
        elif 13 <= hour < 17:
            return True, 'ny_overlap'
        elif 17 <= hour < 21:
            return True, 'ny'
        return True, 'late_ny'

    # ──────────────────────────── Exit Logic ─────────────────────────────────

    def _calc_realized_r(self, side: str, entry: float,
                         current_price: float, initial_r: float) -> float:
        """Tutarlı R hesapla: pozitif = kâr, negatif = zarar."""
        if initial_r <= 0:
            return 0.0
        if side == 'LONG':
            return (current_price - entry) / initial_r
        else:
            return (entry - current_price) / initial_r

    def _check_bb_mr_exit(self, t, current_price, current_rsi=None):
        """
        BB MR v7.1 çıkış mantığı:
          1. Max loss cap → kapat
          2. TP hit → kapat
          3. SL hit → kapat
          4. BB Mid Partial: quality-based %25-40 kapat + SL → BE
          5. Post-mid trailing: R-bazlı trailing
          6. RSI momentum SL sıkıştırma (live RSI)
          7. Time-based SL tightening (12h+)
          8. Breakeven: PnL > %2 → SL → entry
          9. Trailing lock: PnL > %4 → SL kârın %50 kilitle
         10. Timeout: 18 saat → kapat
        """
        try:
            side = t['side']
            entry = t.get('entry_price', 0)
            tp = t.get('tp_price', 0)
            sl = t.get('sl_price', 0)
            symbol = t.get('symbol', '')
            p = self._BB_MR_PARAMS

            if tp <= 0 or sl <= 0 or entry <= 0:
                return

            pnl_dollar, pnl_pct = self._compute_pnl(t, current_price)
            initial_r = t.get('_initial_r', abs(entry - sl))

            # ── [FIX] max_pnl_pct güncelle ──
            if pnl_pct > t.get('max_pnl_pct', 0):
                t['max_pnl_pct'] = pnl_pct

            # ── [FIX] current_rsi güncelle ──
            if current_rsi is not None:
                t['_current_rsi'] = current_rsi

            # ── Max loss cap ──
            _max_loss = getattr(self, '_user_max_loss_cap', 5.0)
            if pnl_dollar < -_max_loss:
                realized_r = self._calc_realized_r(side, entry, current_price, initial_r)
                self._close_all_locked([t], symbol, current_price, "BB_MR_MAXLOSS")
                self._bb_mr_record_outcome('SL', symbol, realized_r)
                self.log(f"[CAP] {symbol} max loss cap: ${pnl_dollar:.2f}")
                return

            # ── TP check ──
            hit_tp = (side == 'LONG' and current_price >= tp) or \
                     (side == 'SHORT' and current_price <= tp)
            if hit_tp:
                realized_r = self._calc_realized_r(side, entry, current_price, initial_r)
                self._close_all_locked([t], symbol, current_price, "BB_MR_TP")
                self._bb_mr_record_outcome('TP', symbol, realized_r)
                return

            # ── SL check ──
            hit_sl = (side == 'LONG' and current_price <= sl) or \
                     (side == 'SHORT' and current_price >= sl)
            if hit_sl:
                reason = "BB_MR_SL" if pnl_pct < 0 else "BB_MR_TRAIL_SL"
                realized_r = self._calc_realized_r(side, entry, current_price, initial_r)
                self._close_all_locked([t], symbol, current_price, reason)
                if pnl_pct < 0:
                    self._bb_mr_record_outcome('SL', symbol, realized_r)
                else:
                    self._bb_mr_record_outcome('TP', symbol, realized_r)
                return

            # ── BB Mid Partial (quality-based %25-40 + RSI adjust) ──
            bb_mid = t.get('_bb_mid_target', 0)
            if bb_mid > 0 and not t.get('_bb_mid_partial'):
                hit_mid = (side == 'LONG' and current_price >= bb_mid) or \
                          (side == 'SHORT' and current_price <= bb_mid)
                if hit_mid:
                    quality = t.get('_quality_score', 50)
                    base_pct = 0.25 if quality >= 75 else 0.35 if quality >= 60 else 0.40
                    live_rsi = current_rsi if current_rsi is not None else t.get('_cached_rsi')
                    if live_rsi is not None:
                        if side == 'LONG' and live_rsi > 60:
                            base_pct = min(base_pct + 0.10, 0.50)
                        elif side == 'SHORT' and live_rsi < 40:
                            base_pct = min(base_pct + 0.10, 0.50)

                    close_qty = t['qty'] * base_pct
                    partial_pnl = self._calc_partial_pnl(t, current_price, close_qty)
                    margin_released = (close_qty * entry) / self.leverage
                    t['margin'] = max(0.0, t.get('margin', 0.0) - margin_released)
                    self.balance += margin_released + partial_pnl
                    t['qty'] -= close_qty
                    t['_bb_mid_partial'] = True
                    t['partial_profit'] = t.get('partial_profit', 0) + partial_pnl

                    buffer = entry * 0.001
                    if side == 'LONG':
                        t['sl_price'] = max(t['sl_price'], entry + buffer)
                    else:
                        t['sl_price'] = min(t['sl_price'], entry - buffer)
                    t['_breakeven_set'] = True

                    self.log(
                        f"[BB_MID] {symbol}: BB mid hit → %{int(base_pct*100)} kapatıldı "
                        f"(${partial_pnl:+.2f}), SL → BE"
                    )

                    # Dust check: kalan çok küçükse tamamını kapat
                    remaining_notional = t['qty'] * current_price
                    if remaining_notional < 5.0:
                        realized_r = self._calc_realized_r(side, entry, current_price, initial_r)
                        self._close_all_locked([t], symbol, current_price, "BB_MR_DUST_CLOSE")
                        self._bb_mr_record_outcome('TP', symbol, realized_r)
                        return

                    self._force_save()
                    return

            # ── v7.2: ATR-bazlı Breakeven + Trailing ──
            # Eski: margin PnL %2/%4 → leverage 10x ile %0.2/%0.4 fiyat = çok erken
            # Yeni: ATR-bazlı → leverage'den bağımsız, mean reversion'a uygun
            atr_val = t.get('atr', entry * 0.005)
            if atr_val <= 0:
                atr_val = entry * 0.005

            if side == 'LONG':
                peak_price = max(current_price, t.get('_bb_peak_price', entry))
                t['_bb_peak_price'] = peak_price
                peak_profit_atr = (peak_price - entry) / atr_val

                # Breakeven: 1.5 ATR kârda → SL = entry + 0.3 ATR
                if peak_profit_atr >= 1.5 and not t.get('_breakeven_set', False):
                    be_sl = entry + 0.3 * atr_val
                    if be_sl > t['sl_price']:
                        t['sl_price'] = be_sl
                        t['_breakeven_set'] = True
                        self.log(f"[BE] {symbol}: SL→BE {be_sl:.6f} (peak {peak_profit_atr:.1f} ATR)")

                # Trailing: 3.0 ATR kârda → SL = peak - 1.5 ATR
                if peak_profit_atr >= 3.0:
                    trail_sl = peak_price - 1.5 * atr_val
                    if trail_sl > t['sl_price']:
                        t['sl_price'] = trail_sl

            else:  # SHORT
                peak_price = min(current_price, t.get('_bb_peak_price', entry))
                t['_bb_peak_price'] = peak_price
                peak_profit_atr = (entry - peak_price) / atr_val

                if peak_profit_atr >= 1.5 and not t.get('_breakeven_set', False):
                    be_sl = entry - 0.3 * atr_val
                    if be_sl < t['sl_price']:
                        t['sl_price'] = be_sl
                        t['_breakeven_set'] = True
                        self.log(f"[BE] {symbol}: SL→BE {be_sl:.6f} (peak {peak_profit_atr:.1f} ATR)")

                if peak_profit_atr >= 3.0:
                    trail_sl = peak_price + 1.5 * atr_val
                    if trail_sl < t['sl_price']:
                        t['sl_price'] = trail_sl

            # ── Post-mid trailing (R-based) ──
            if t.get('_bb_mid_partial') and initial_r > 0:
                curr_r = abs(current_price - entry) / initial_r if entry > 0 else 0
                if curr_r >= 0.8:
                    lock = entry + initial_r * 0.5 if side == 'LONG' else entry - initial_r * 0.5
                    if (side == 'LONG' and lock > t['sl_price']) or \
                       (side == 'SHORT' and lock < t['sl_price']):
                        t['sl_price'] = lock
                elif curr_r >= 0.3:
                    lock = entry + initial_r * 0.15 if side == 'LONG' else entry - initial_r * 0.15
                    if (side == 'LONG' and lock > t['sl_price']) or \
                       (side == 'SHORT' and lock < t['sl_price']):
                        t['sl_price'] = lock

            # ── RSI momentum SL sıkıştırma (live RSI, sadece 2+ ATR kârda) ──
            live_rsi = current_rsi if current_rsi is not None else t.get('_cached_rsi')
            if live_rsi is not None and peak_profit_atr >= 2.0:
                if side == 'LONG' and live_rsi < 45:
                    momentum_sl = current_price - (current_price - entry) * 0.3
                    if momentum_sl > t['sl_price']:
                        t['sl_price'] = momentum_sl
                elif side == 'SHORT' and live_rsi > 55:
                    momentum_sl = current_price + (entry - current_price) * 0.3
                    if momentum_sl < t['sl_price']:
                        t['sl_price'] = momentum_sl

            # ── Time-based SL tightening (12h+) ──
            entry_ts = t.get('entry_timestamp', 0)
            if entry_ts > 0 and initial_r > 0:
                hours_held = (time.time() - entry_ts) / 3600
                if hours_held > 12:
                    shrink = min((hours_held - 12) / 6 * 0.3, 0.5)
                    if side == 'LONG':
                        time_sl = (entry - initial_r) + initial_r * shrink
                        if time_sl > t['sl_price']:
                            t['sl_price'] = time_sl
                    else:
                        time_sl = (entry + initial_r) - initial_r * shrink
                        if time_sl < t['sl_price']:
                            t['sl_price'] = time_sl

            # ── Timeout ──
            if entry_ts > 0 and time.time() - entry_ts > p['timeout_hours'] * 3600:
                realized_r = self._calc_realized_r(side, entry, current_price, initial_r)
                self._close_all_locked([t], symbol, current_price, "BB_MR_TIMEOUT")
                self._bb_mr_record_outcome('TIMEOUT', symbol, realized_r)
                return

        except Exception as e:
            self.log(f"[ERR] _check_bb_mr_exit {t.get('symbol','?')}: {e}")

    def _calc_rejection_score(self, df: pd.DataFrame, i: int,
                              direction: str, bb_lower: float,
                              bb_upper: float,
                              bb_mid: float) -> Tuple[float, dict]:
        close = float(df['close'].iloc[i])
        open_p = float(df['open'].iloc[i])
        high = float(df['high'].iloc[i])
        low = float(df['low'].iloc[i])
        full_range = high - low
        if full_range <= 0:
            return 0.0, {}

        score = 0.0
        details = {}
        body = abs(close - open_p)

        if direction == 'LONG':
            lower_wick = min(open_p, close) - low
            wick_ratio = lower_wick / full_range
            clv = (close - low) / full_range

            if wick_ratio >= 0.55:
                score += 3.0; details['wick'] = 'excellent'
            elif wick_ratio >= 0.40:
                score += 2.0; details['wick'] = 'good'
            elif wick_ratio >= 0.25:
                score += 1.0; details['wick'] = 'moderate'

            if clv >= 0.70:
                score += 2.0; details['clv'] = 'strong'
            elif clv >= 0.55:
                score += 1.0; details['clv'] = 'ok'

            if close > bb_lower:
                score += 0.5
                if close > open_p:
                    score += 1.0; details['candle'] = 'bullish'

            upper_wick = high - max(open_p, close)
            if wick_ratio > 0.5 and upper_wick < body * 0.3:
                score += 2.0; details['pattern'] = 'hammer'
            elif close > open_p and body > full_range * 0.3:
                score += 1.0; details['pattern'] = 'bullish_body'

            if i >= 2:
                c1 = float(df['close'].iloc[i - 1])
                c2 = float(df['close'].iloc[i - 2])
                if close > c1 > c2:
                    score += 1.5; details['follow'] = 'rising'
                elif close > c1:
                    score += 0.5; details['follow'] = 'turning'

        else:
            upper_wick = high - max(open_p, close)
            wick_ratio = upper_wick / full_range
            clv = (high - close) / full_range

            if wick_ratio >= 0.55:
                score += 3.0; details['wick'] = 'excellent'
            elif wick_ratio >= 0.40:
                score += 2.0; details['wick'] = 'good'
            elif wick_ratio >= 0.25:
                score += 1.0; details['wick'] = 'moderate'

            if clv >= 0.70:
                score += 2.0; details['clv'] = 'strong'
            elif clv >= 0.55:
                score += 1.0; details['clv'] = 'ok'

            if close < bb_upper:
                score += 0.5
                if close < open_p:
                    score += 1.0; details['candle'] = 'bearish'

            lower_wick_s = min(open_p, close) - low
            if wick_ratio > 0.5 and lower_wick_s < body * 0.3:
                score += 2.0; details['pattern'] = 'shooting_star'
            elif close < open_p and body > full_range * 0.3:
                score += 1.0; details['pattern'] = 'bearish_body'

            if i >= 2:
                c1 = float(df['close'].iloc[i - 1])
                c2 = float(df['close'].iloc[i - 2])
                if close < c1 < c2:
                    score += 1.5; details['follow'] = 'falling'
                elif close < c1:
                    score += 0.5; details['follow'] = 'turning'

        return round(min(score, 10), 1), details

    def _find_touch(self, df: pd.DataFrame, i: int,
                    direction: str) -> Tuple[bool, int]:
        for j in range(0, 3):
            idx = i - j
            if idx < 0:
                break
            if direction == 'LONG':
                bl = float(df['low'].iloc[idx])
                bb = float(df['bb_lower'].iloc[idx])
                if np.isnan(bb) or np.isnan(bl):
                    continue
                if bl <= bb:
                    return True, j
            else:
                bh = float(df['high'].iloc[idx])
                bb = float(df['bb_upper'].iloc[idx])
                if np.isnan(bb) or np.isnan(bh):
                    continue
                if bh >= bb:
                    return True, j
        return False, -1

    def _check_confirmation_candle(self, df: pd.DataFrame, i: int,
                                   direction: str) -> Tuple[bool, float]:
        if i < 2:
            return False, 0.0
        cc = float(df['close'].iloc[i])
        co = float(df['open'].iloc[i])
        ph = float(df['high'].iloc[i - 1])
        pl = float(df['low'].iloc[i - 1])
        po = float(df['open'].iloc[i - 1])
        pc = float(df['close'].iloc[i - 1])
        pr = ph - pl
        if pr <= 0:
            return False, 0.0

        if direction == 'LONG':
            if cc <= co:
                return False, 0.0
            if cc > ph and co <= pl:
                return True, 1.0
            elif cc > po - (po - pc) * 0.5:
                return True, 0.7
            elif cc > pl + pr * 0.5:
                return True, 0.4
            return False, 0.0
        else:
            if cc >= co:
                return False, 0.0
            if cc < pl and co >= ph:
                return True, 1.0
            elif cc < po + (pc - po) * 0.5:
                return True, 0.7
            elif cc < ph - pr * 0.5:
                return True, 0.4
            return False, 0.0

    def _is_band_walk(self, df: pd.DataFrame, i: int,
                      direction: str, band_price: float) -> bool:
        near = 0
        for j in range(1, 6):
            idx = i - j
            if idx < 0:
                break
            c = float(df['close'].iloc[idx])
            if direction == 'LONG' and c <= band_price * 1.015:
                near += 1
            elif direction == 'SHORT' and c >= band_price * 0.985:
                near += 1
        if near >= 3:
            return True

        if 'rsi' in df.columns and i >= 4:
            rsi_now = float(df['rsi'].iloc[i])
            rsi_4ago = float(df['rsi'].iloc[i - 4])
            if not np.isnan(rsi_now) and not np.isnan(rsi_4ago):
                if direction == 'LONG' and rsi_now < rsi_4ago - 5 and near >= 2:
                    return True
                if direction == 'SHORT' and rsi_now > rsi_4ago + 5 and near >= 2:
                    return True
        return False

    def _check_price_flow_confirmation(self, df: pd.DataFrame, i: int,
                                direction: str) -> Tuple[bool, float]:
        if 'volume' not in df.columns or i < 10:
            return True, 0.0
        close = df['close'].astype(float)
        up_bars = down_bars = 0
        for j in range(max(1, i - 4), i + 1):
            cv = float(close.iloc[j])
            pv = float(close.iloc[j - 1])
            if np.isnan(cv) or np.isnan(pv):
                continue
            if cv > pv:
                up_bars += 1
            elif cv < pv:
                down_bars += 1
        if direction == 'LONG':
            if up_bars >= 3:
                return True, 1.0
            if up_bars >= 2:
                return True, 0.5
            if down_bars >= 4:
                return False, 0.0
            return True, 0.2
        else:
            if down_bars >= 3:
                return True, 1.0
            if down_bars >= 2:
                return True, 0.5
            if up_bars >= 4:
                return False, 0.0
            return True, 0.2

    def _check_volume_exhaustion(self, df: pd.DataFrame, i: int,
                                 direction: str) -> Tuple[bool, float]:
        if 'volume' not in df.columns or i < 5:
            return True, 0.0
        vol = df['volume'].astype(float)
        vsma = vol.rolling(20, min_periods=5).mean()
        avg = float(vsma.iloc[i])
        if avg <= 0 or np.isnan(avg):
            return True, 0.0
        v0 = float(vol.iloc[i])
        v1 = float(vol.iloc[i - 1])
        v2 = float(vol.iloc[i - 2])
        v3 = float(vol.iloc[i - 3])
        if any(np.isnan(v) or v <= 0 for v in [v0, v1, v2, v3]):
            return True, 0.0
        sc = 0.0
        if v1 < v2 < v3 and v0 > v1 * 1.3:
            sc = 1.0
        elif v1 < avg * 0.8 and v0 > v1 * 1.3:
            sc = 0.7
        elif v0 > avg * 1.2:
            sc = 0.3
        if v1 > v2 > v3 and v1 > avg * 1.5:
            return False, 0.0
        return True, sc

    def _check_stoch_rsi_cross(self, df: pd.DataFrame, i: int,
                                direction: str) -> Tuple[bool, float]:
        if 'rsi' not in df.columns or i < 16:
            return False, 0.0
        rsi = df['rsi'].astype(float)
        vals = []
        for j in range(max(0, i - 4), i + 1):
            w = rsi.iloc[max(0, j - 13):j + 1].dropna()
            if len(w) < 5:
                vals.append(50.0); continue
            mn, mx = float(w.min()), float(w.max())
            r = mx - mn
            vals.append((float(rsi.iloc[j]) - mn) / r * 100 if r > 0 else 50.0)
        if len(vals) < 3:
            return False, 0.0
        k_now, k_prev = vals[-1], vals[-2]
        d_now = float(np.mean(vals[-3:]))
        d_prev = float(np.mean(vals[-4:-1])) if len(vals) >= 4 else d_now
        if direction == 'LONG':
            if k_prev <= d_prev and k_now > d_now and (k_prev < 25 or k_now < 30):
                return True, max((30 - min(k_prev, k_now)) / 30, 0.3)
            if k_now < 20 and k_now > k_prev:
                return True, 0.2
        else:
            if k_prev >= d_prev and k_now < d_now and (k_prev > 75 or k_now > 70):
                return True, max((min(k_prev, k_now) - 70) / 30, 0.3)
            if k_now > 80 and k_now < k_prev:
                return True, 0.2
        return False, 0.0

    def _check_rsi_divergence(self, df: pd.DataFrame, i: int,
                               direction: str) -> Tuple[bool, float]:
        if 'rsi' not in df.columns or i < 15:
            return False, 0.0
        rsi = df['rsi'].astype(float)
        lows = df['low'].astype(float)
        highs = df['high'].astype(float)

        if direction == 'LONG':
            swings = []
            for j in range(i - 15, i - 1):
                if j < 2 or j + 2 >= len(df):
                    continue
                l = float(lows.iloc[j])
                try:
                    if l <= float(lows.iloc[j - 1]) and l <= float(lows.iloc[j + 1]):
                        swings.append((j, l, float(rsi.iloc[j])))
                except (IndexError, KeyError):
                    continue
            cl = float(lows.iloc[i])
            if i >= 2 and cl <= float(lows.iloc[i - 1]):
                swings.append((i, cl, float(rsi.iloc[i])))
            if len(swings) >= 2:
                ps, cs = swings[-2], swings[-1]
                if cs[1] < ps[1] and cs[2] > ps[2]:
                    return True, max(min((cs[2] - ps[2]) / 15, 1.0), 0.3)
        else:
            swings = []
            for j in range(i - 15, i - 1):
                if j < 2 or j + 2 >= len(df):
                    continue
                h = float(highs.iloc[j])
                try:
                    if h >= float(highs.iloc[j - 1]) and h >= float(highs.iloc[j + 1]):
                        swings.append((j, h, float(rsi.iloc[j])))
                except (IndexError, KeyError):
                    continue
            ch = float(highs.iloc[i])
            if i >= 2 and ch >= float(highs.iloc[i - 1]):
                swings.append((i, ch, float(rsi.iloc[i])))
            if len(swings) >= 2:
                ps, cs = swings[-2], swings[-1]
                if cs[1] > ps[1] and cs[2] < ps[2]:
                    return True, max(min((ps[2] - cs[2]) / 15, 1.0), 0.3)
        return False, 0.0

    def _check_structural_confluence(self, df: pd.DataFrame, i: int,
                                     direction: str, touch_price: float,
                                     atr: float) -> Tuple[bool, float]:
        tol = atr * 0.5
        if i < 50 or atr <= 0:
            return False, 0.0
        levels = []
        for j in range(i - 50, i - 2):
            if j < 2:
                continue
            h, l = float(df['high'].iloc[j]), float(df['low'].iloc[j])
            try:
                if all(h >= float(df['high'].iloc[j + k]) and
                       h >= float(df['high'].iloc[j - k])
                       for k in range(1, 3) if 0 <= j - k and j + k < len(df)):
                    levels.append(('resistance', h))
            except (IndexError, KeyError):
                pass
            try:
                if all(l <= float(df['low'].iloc[j + k]) and
                       l <= float(df['low'].iloc[j - k])
                       for k in range(1, 3) if 0 <= j - k and j + k < len(df)):
                    levels.append(('support', l))
            except (IndexError, KeyError):
                pass
        cnt, closest = 0, float('inf')
        for lt, lp in levels:
            d = abs(touch_price - lp)
            if d <= tol:
                if (direction == 'LONG' and lt == 'support') or \
                   (direction == 'SHORT' and lt == 'resistance'):
                    cnt += 1; closest = min(closest, d)
        if cnt == 0:
            return False, 0.0
        s = min(cnt * 0.3, 1.0) + max(0, 1 - closest / tol) * 0.3
        return True, min(s, 1.0)

    def _check_path_clear(self, df: pd.DataFrame, i: int,
                          direction: str, entry: float,
                          target: float, atr: float) -> Tuple[bool, float]:
        if i < 20 or atr <= 0:
            return True, 0.0
        tol = atr * 0.3
        obs = 0
        for j in range(max(0, i - 20), i):
            if j < 1 or j + 1 >= len(df):
                continue
            h, l = float(df['high'].iloc[j]), float(df['low'].iloc[j])
            if direction == 'LONG':
                if (h >= float(df['high'].iloc[j - 1]) and
                        h >= float(df['high'].iloc[j + 1]) and
                        entry < h < target and abs(h - entry) > tol):
                    obs += 1
            else:
                if (l <= float(df['low'].iloc[j - 1]) and
                        l <= float(df['low'].iloc[j + 1]) and
                        target < l < entry and abs(l - entry) > tol):
                    obs += 1
        if obs <= 2:
            return True, obs * 0.3
        return False, 1.0

    def _check_adx_filter(self, df: pd.DataFrame, i: int,
                          direction: str) -> Tuple[bool, float, bool]:
        p = self._BB_MR_PARAMS
        adx_val = None
        for col in ('adx', 'ADX'):
            if col in df.columns:
                v = float(df[col].iloc[i])
                if not np.isnan(v):
                    adx_val = v
                    break
        if adx_val is None:
            adx_val = self._calc_adx_simple(df, i)

        if adx_val > p['max_adx']:
            return False, adx_val, True
        req = adx_val > p['adx_div_required']

        pdi_col = next((c for c in ('plus_di', '+DI', 'DI+')
                        if c in df.columns), None)
        mdi_col = next((c for c in ('minus_di', '-DI', 'DI-')
                        if c in df.columns), None)
        if pdi_col and mdi_col and i >= 3:
            pdi = float(df[pdi_col].iloc[i])
            mdi = float(df[mdi_col].iloc[i])
            if not np.isnan(pdi) and not np.isnan(mdi):
                if direction == 'LONG':
                    pm = float(df[mdi_col].iloc[i - 2])
                    if not np.isnan(pm) and mdi > pdi * 1.5 and mdi > pm:
                        req = True
                else:
                    pp = float(df[pdi_col].iloc[i - 2])
                    if not np.isnan(pp) and pdi > mdi * 1.5 and pdi > pp:
                        req = True
        return True, adx_val, req

    def _calc_adx_simple(self, df: pd.DataFrame, i: int,
                         period: int = 14) -> float:
        if i < period * 2:
            return 20.0
        try:
            h, l, c = (df['high'].astype(float), df['low'].astype(float),
                        df['close'].astype(float))
            st = max(0, i - period * 2)
            tr_l, dmp, dmm = [], [], []
            for j in range(st + 1, i + 1):
                hv, lv, pc = float(h.iloc[j]), float(l.iloc[j]), float(c.iloc[j - 1])
                ph, pl = float(h.iloc[j - 1]), float(l.iloc[j - 1])
                tr_l.append(max(hv - lv, abs(hv - pc), abs(lv - pc)))
                up, dn = hv - ph, pl - lv
                dmp.append(max(up, 0) if up > dn else 0)
                dmm.append(max(dn, 0) if dn > up else 0)
            if len(tr_l) < period:
                return 20.0
            atr_s = np.mean(tr_l[-period:])
            if atr_s <= 0:
                return 20.0
            pdi = np.mean(dmp[-period:]) / atr_s * 100
            mdi = np.mean(dmm[-period:]) / atr_s * 100
            ds = pdi + mdi
            return abs(pdi - mdi) / ds * 100 if ds > 0 else 20.0
        except Exception:
            return 20.0

    def _is_post_squeeze_breakout(self, df: pd.DataFrame, i: int) -> bool:
        if i < 25 or 'atr' not in df.columns or 'bb_upper' not in df.columns:
            return False
        ema20 = df['close'].astype(float).ewm(span=20, adjust=False).mean()
        atr_v = df['atr'].astype(float)
        sq = 0
        for j in range(max(0, i - 10), i):
            try:
                ku = float(ema20.iloc[j]) + float(atr_v.iloc[j]) * 1.5
                kl = float(ema20.iloc[j]) - float(atr_v.iloc[j]) * 1.5
                bu = float(df['bb_upper'].iloc[j])
                bl = float(df['bb_lower'].iloc[j])
                if any(np.isnan(v) for v in [ku, kl, bu, bl]):
                    continue
                if bu < ku and bl > kl:
                    sq += 1
            except (IndexError, KeyError):
                continue
        if sq < 3:
            return False
        try:
            ku_n = float(ema20.iloc[i]) + float(atr_v.iloc[i]) * 1.5
            kl_n = float(ema20.iloc[i]) - float(atr_v.iloc[i]) * 1.5
            bu_n = float(df['bb_upper'].iloc[i])
            bl_n = float(df['bb_lower'].iloc[i])
            if any(np.isnan(v) for v in [ku_n, kl_n, bu_n, bl_n]):
                return False
            rel = bu_n >= ku_n or bl_n <= kl_n
            w1 = bu_n - bl_n
            w2 = float(df['bb_upper'].iloc[i - 1]) - float(df['bb_lower'].iloc[i - 1])
            w3 = float(df['bb_upper'].iloc[i - 2]) - float(df['bb_lower'].iloc[i - 2])
            return rel and w1 > w2 > w3
        except (IndexError, KeyError):
            return False

    def _is_bands_expanding_aggressively(self, df: pd.DataFrame, i: int) -> bool:
        if i < 5 or 'bb_upper' not in df.columns:
            return False
        try:
            ws = [float(df['bb_upper'].iloc[j]) - float(df['bb_lower'].iloc[j])
                  for j in range(i - 4, i + 1)]
            if any(w <= 0 for w in ws):
                return False
            mono = all(ws[k] > ws[k - 1] for k in range(1, len(ws)))
            fast = (ws[-1] / ws[0]) > 1.5 if ws[0] > 0 else False
            ab = sum(abs(float(df['close'].iloc[j]) - float(df['open'].iloc[j]))
                     for j in range(i - 2, i + 1)) / 3
            atr = float(df['atr'].iloc[i]) if 'atr' in df.columns else 0
            big = ab > atr * 0.6 if atr > 0 else False
            c = float(df['close'].iloc[i])
            br = ws[-1]
            pos = (c - float(df['bb_lower'].iloc[i])) / br if br > 0 else 0.5
            near = pos > 0.85 or pos < 0.15
            return mono and fast and (big or near)
        except Exception:
            return False

    def _check_bb_squeeze(self, df, i):
        if 'bb_upper' not in df.columns:
            return False, 0.0
        bw = float(df['bb_upper'].iloc[i]) - float(df['bb_lower'].iloc[i])
        cv = float(df['close'].iloc[i])
        if cv <= 0 or bw <= 0:
            return False, 0.0
        bp = bw / cv
        try:
            mw = min((float(df['bb_upper'].iloc[j]) - float(df['bb_lower'].iloc[j])) /
                     float(df['close'].iloc[j])
                     for j in range(max(0, i - 20), i + 1)
                     if float(df['close'].iloc[j]) > 0 and
                     (float(df['bb_upper'].iloc[j]) - float(df['bb_lower'].iloc[j])) > 0)
        except ValueError:
            return False, 0.0
        sq = bp <= mw * 1.3
        exp = False
        if i >= 3:
            w1 = float(df['bb_upper'].iloc[i]) - float(df['bb_lower'].iloc[i])
            w2 = float(df['bb_upper'].iloc[i - 1]) - float(df['bb_lower'].iloc[i - 1])
            w3 = float(df['bb_upper'].iloc[i - 2]) - float(df['bb_lower'].iloc[i - 2])
            exp = w1 > w2 > w3
        sc = 0.5 if sq else 0.0
        if sq and exp:
            sc = 1.0
        return sq or exp, sc

    def _calc_bb_mr_sl_tp(self, df, direction, close, atr, bb_mid,
                           bb_upper, bb_lower):
        p = self._BB_MR_PARAMS
        i = len(df) - 1

        if direction == 'LONG':
            sl_bb = bb_lower - atr * 0.3
            rl = float(df['low'].iloc[max(0, i - 10):i + 1].min())
            sl_sw = rl - atr * 0.2
            sl_price = min(sl_bb, sl_sw)

            sl_dist = close - sl_price
            if sl_dist < atr * p['sl_min_atr']:
                sl_price = close - atr * p['sl_min_atr']
                sl_dist = atr * p['sl_min_atr']
            if sl_dist > atr * p['sl_max_atr']:
                sl_price = close - atr * p['sl_max_atr']
                sl_dist = atr * p['sl_max_atr']

            tp1 = bb_mid
            tp2 = bb_mid + (bb_mid - bb_lower) * 0.5
            t1d = tp1 - close
            t2d = tp2 - close

            candidates = []
            if t1d > 0 and t1d / sl_dist >= p['min_rr']:
                candidates.append((tp1, t1d / sl_dist))
            if t2d > 0 and t2d / sl_dist >= p['min_rr']:
                candidates.append((tp2, t2d / sl_dist))

            if not candidates:
                return None, None, 0

            best = min(candidates, key=lambda x: abs(x[1] - 2.8))
            tp_price = best[0]

            if tp_price - close < p['tp_min'] * atr:
                return None, None, 0

        else:
            sl_bb = bb_upper + atr * 0.3
            rh = float(df['high'].iloc[max(0, i - 10):i + 1].max())
            sl_sw = rh + atr * 0.2
            sl_price = max(sl_bb, sl_sw)

            sl_dist = sl_price - close
            if sl_dist < atr * p['sl_min_atr']:
                sl_price = close + atr * p['sl_min_atr']
                sl_dist = atr * p['sl_min_atr']
            if sl_dist > atr * p['sl_max_atr']:
                sl_price = close + atr * p['sl_max_atr']
                sl_dist = atr * p['sl_max_atr']

            tp1 = bb_mid
            tp2 = bb_mid - (bb_upper - bb_mid) * 0.5
            t1d = close - tp1
            t2d = close - tp2

            candidates = []
            if t1d > 0 and t1d / sl_dist >= p['min_rr']:
                candidates.append((tp1, t1d / sl_dist))
            if t2d > 0 and t2d / sl_dist >= p['min_rr']:
                candidates.append((tp2, t2d / sl_dist))

            if not candidates:
                return None, None, 0

            best = min(candidates, key=lambda x: abs(x[1] - 2.8))
            tp_price = best[0]

            if close - tp_price < p['tp_min'] * atr:
                return None, None, 0

        return sl_price, tp_price, sl_dist

    def _calc_quality_score(self, direction, rsi, rej_score, rr_ratio,
                            vol_score, squeeze_score, trend_bias,
                            conf_strength=0.0, vol_exhaustion=0.0,
                            structural_conf=False,
                            structural_conf_strength=0.0,
                            stoch_cross=False, stoch_strength=0.0,
                            has_div=False, div_strength=0.0,
                            adx_val=0.0, path_penalty=0.0,
                            price_flow_strength=0.0, session='london'):
        score = 0.0
        p = self._BB_MR_PARAMS

        if direction == 'LONG':
            depth = max(0, (p['rsi_long_flat'] - rsi) / p['rsi_long_flat'])
        else:
            depth = max(0, (rsi - p['rsi_short_flat']) / (100 - p['rsi_short_flat']))
        score += depth * 15

        score += min((rej_score - 3) * 4, 22)

        score += min(rr_ratio * 5, 15)

        score += conf_strength * 12

        score += vol_exhaustion * 8

        if has_div:
            score += 7 + div_strength * 5

        if structural_conf:
            score += structural_conf_strength * 8

        if stoch_cross:
            score += stoch_strength * 5

        score += price_flow_strength * 5

        score += squeeze_score * 3

        if ((direction == 'LONG' and trend_bias == 'up') or
                (direction == 'SHORT' and trend_bias == 'down')):
            score += 5
        elif ((direction == 'LONG' and trend_bias == 'down') or
              (direction == 'SHORT' and trend_bias == 'up')):
            score -= 3

        if adx_val > 25:
            score -= (adx_val - 25) * 0.5

        score -= path_penalty * 8

        score += min(vol_score * 2, 6)

        if session == 'ny_overlap':
            score += 3
        elif session == 'london':
            score += 2
        elif session == 'asia':
            score += 1
        elif session == 'late_ny':
            score -= 2

        return round(max(min(score, 100), 0), 1)

    def _evaluate_direction(self, df, i, direction, bb_lower, bb_upper,
                            bb_mid, rsi_val, trend_bias, atr_pct):
        p = self._BB_MR_PARAMS
        close = float(df['close'].iloc[i])

        if direction == 'LONG':
            base = (p['rsi_long_counter'] if trend_bias == 'down'
                    else p['rsi_long_flat'])
        else:
            base = (p['rsi_short_counter'] if trend_bias == 'up'
                    else p['rsi_short_flat'])

        if atr_pct > 0.03:
            adj = -5 if direction == 'LONG' else 5
        elif atr_pct < 0.01:
            adj = 3 if direction == 'LONG' else -3
        else:
            adj = 0
        thresh = base + adj

        if direction == 'LONG' and rsi_val >= thresh:
            return None
        if direction == 'SHORT' and rsi_val <= thresh:
            return None

        is_counter = ((direction == 'LONG' and trend_bias == 'down') or
                      (direction == 'SHORT' and trend_bias == 'up'))

        found, t_off = self._find_touch(df, i, direction)
        if not found:
            return None

        if direction == 'LONG' and close <= bb_lower:
            return None
        if direction == 'SHORT' and close >= bb_upper:
            return None

        rej, det = self._calc_rejection_score(
            df, i, direction, bb_lower, bb_upper, bb_mid)
        if rej < p['min_rej_score']:
            return None

        return rej, det, is_counter, t_off

    def _bb_mr_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        self._bb_mr_init_state()
        if len(df) < 55:
            return None

        i = len(df) - 1
        close = float(df['close'].iloc[i])
        high = float(df['high'].iloc[i])
        low = float(df['low'].iloc[i])
        full_range = high - low

        atr = float(df['atr'].iloc[i]) if 'atr' in df.columns else None
        rsi_val = float(df['rsi'].iloc[i]) if 'rsi' in df.columns else None
        bb_mid = float(df['bb_mid'].iloc[i]) if 'bb_mid' in df.columns else None
        bb_upper = float(df['bb_upper'].iloc[i]) if 'bb_upper' in df.columns else None
        bb_lower = float(df['bb_lower'].iloc[i]) if 'bb_lower' in df.columns else None

        vals = [atr, rsi_val, bb_mid, bb_upper, bb_lower]
        if any(v is None or np.isnan(v) for v in vals):
            return None
        if atr <= 0 or close <= 0:
            return None
        atr_pct = atr / close
        # v7.2: 0.003 → 0.0015. TRX gibi düşük-vol coinler %92 engelleniyordu
        # (TRX median ATR %0.24, eşik %0.30). %0.15 hâlâ ölü piyasayı filtreler.
        if atr_pct < 0.0015:
            return None

        p = self._BB_MR_PARAMS

        ema50 = None
        for col in ('ema_50', 'ema50'):
            if col in df.columns:
                ema50 = float(df[col].iloc[i])
                break

        trend_bias = 'flat'
        if ema50 is not None and not np.isnan(ema50) and len(df) > 20:
            e_col = 'ema_50' if 'ema_50' in df.columns else 'ema50'
            for lb in [20, 10]:
                ip = i - lb
                if ip >= 0:
                    v = float(df[e_col].iloc[ip])
                    if not np.isnan(v) and v > 0:
                        slope = (ema50 - v) / v
                        if slope >= 0.015:
                            trend_bias = 'up'
                        elif slope <= -0.015:
                            trend_bias = 'down'
                        break

        if self._is_post_squeeze_breakout(df, i):
            return None
        if self._is_bands_expanding_aggressively(df, i):
            return None

        direction = None
        # ── [FIX] Her iki yönü değerlendir, en iyi rej_score'u seç ──
        candidates_dir = {}
        for d in ['LONG', 'SHORT']:
            result = self._evaluate_direction(
                df, i, d, bb_lower, bb_upper, bb_mid,
                rsi_val, trend_bias, atr_pct)
            if result is not None:
                candidates_dir[d] = result

        if not candidates_dir:
            return None

        if len(candidates_dir) == 2:
            direction = max(candidates_dir, key=lambda d: candidates_dir[d][0])
        else:
            direction = list(candidates_dir.keys())[0]

        rej_score, rej_details, is_counter, touch_offset = candidates_dir[direction]

        adx_ok, adx_val, adx_req_div = self._check_adx_filter(df, i, direction)
        if not adx_ok:
            return None

        band_ref = bb_lower if direction == 'LONG' else bb_upper
        if self._is_band_walk(df, i, direction, band_ref):
            return None

        if atr > 0 and full_range > atr * p['max_candle_range_atr']:
            return None

        conf_strength = 0.0
        if touch_offset == 0:
            conf_strength = 0.7
        elif touch_offset == 1:
            ok, conf_strength = self._check_confirmation_candle(df, i, direction)
            if not ok:
                conf_strength = 0.0  # Soft: skip etme, quality düşük olacak
        elif touch_offset == 2:
            # 2 bar önceki touch → son 2 barda confirmation aranır
            ok, conf_strength = self._check_confirmation_candle(df, i, direction)
            if not ok:
                conf_strength = 0.0
            else:
                conf_strength *= 0.7  # Stale touch penalty
        else:
            return None

        ve_ok, ve_score = self._check_volume_exhaustion(df, i, direction)
        if not ve_ok:
            ve_score = 0.0  # Soft: momentum devam ediyor ama quality düşecek

        obv_ok, obv_str = self._check_price_flow_confirmation(df, i, direction)
        if not obv_ok:
            obv_str = 0.0  # Soft: quality'de 0 puan, hard skip yok

        has_div, div_str = self._check_rsi_divergence(df, i, direction)
        div_required = adx_req_div or is_counter
        if div_required and not has_div:
            return None

        sl_price, tp_price, sl_dist = self._calc_bb_mr_sl_tp(
            df, direction, close, atr, bb_mid, bb_upper, bb_lower)
        if sl_price is None or tp_price is None:
            return None

        tp_dist = abs(tp_price - close)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if rr < p['min_rr']:
            return None

        stoch_cross, stoch_str = self._check_stoch_rsi_cross(df, i, direction)
        touch_px = low if direction == 'LONG' else high
        s_conf, s_conf_str = self._check_structural_confluence(
            df, i, direction, touch_px, atr)
        path_ok, path_pen = self._check_path_clear(
            df, i, direction, close, tp_price, atr)
        if not path_ok:
            return None

        vol_score = 1
        vol_confirmed = False
        if 'volume' in df.columns:
            vol = df['volume'].astype(float)
            vsma = vol.rolling(20, min_periods=5).mean()
            mvr = 0.0
            for j in range(0, 3):
                if i - j >= 0:
                    av = float(vsma.iloc[i - j])
                    if not np.isnan(av) and av > 0:
                        mvr = max(mvr, float(vol.iloc[i - j]) / av)
            if mvr > 2.0:
                vol_confirmed = True; vol_score = 3
            elif mvr > 1.5:
                vol_confirmed = True; vol_score = 2
            elif mvr > 1.0:
                vol_score = 1
            else:
                vol_score = 0

        is_sq, sq_sc = self._check_bb_squeeze(df, i)
        _, session = self._bb_mr_check_session()

        quality = self._calc_quality_score(
            direction=direction, rsi=rsi_val, rej_score=rej_score,
            rr_ratio=rr, vol_score=vol_score, squeeze_score=sq_sc,
            trend_bias=trend_bias, conf_strength=conf_strength,
            vol_exhaustion=ve_score, structural_conf=s_conf,
            structural_conf_strength=s_conf_str,
            stoch_cross=stoch_cross, stoch_strength=stoch_str,
            has_div=has_div, div_strength=div_str,
            adx_val=adx_val, path_penalty=path_pen,
            price_flow_strength=obv_str, session=session)

        if quality < p['min_quality_pre']:
            return None

        return {
            'direction': direction,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'bb_mid': bb_mid,
            'tp_mult': tp_dist / atr if atr > 0 else 0,
            'sl_mult': sl_dist / atr if atr > 0 else 0,
            'atr': atr,
            'rsi': rsi_val,
            'strategy': 'bb_mr_v7.1',
            'quality_score': quality,
            'rr_ratio': round(rr, 2),
            'rej_score': rej_score,
            'rej_details': rej_details,
            'conf_strength': round(conf_strength, 2),
            'vol_score': vol_score,
            'vol_confirmed': vol_confirmed,
            'vol_exhaustion': round(ve_score, 2),
            'price_flow_strength': round(obv_str, 2),
            'squeeze': is_sq,
            'squeeze_score': sq_sc,
            'stoch_cross': stoch_cross,
            'stoch_strength': round(stoch_str, 2),
            'structural_conf': s_conf,
            'structural_conf_strength': round(s_conf_str, 2),
            'has_divergence': has_div,
            'div_strength': round(div_str, 2),
            'trend_bias': trend_bias,
            'is_counter_trend': is_counter,
            'adx_val': round(adx_val, 1),
            'path_penalty': round(path_pen, 2),
            'session': session,
            'signal_high': high,
            'signal_low': low,
            '_initial_r': sl_dist,
            'touch_offset': touch_offset,
        }


    # ──────────────────────────── HTF Check ─────────────────────────────────

    def _bb_mr_htf_check(self, symbol, direction, fetcher):
        """4H trend check — returns (multiplier 0.3-1.3, reason str)."""
        try:
            df_4h = fetcher.fetch_ohlcv(symbol, '4h', limit=50)
            if df_4h is None or len(df_4h) < 20:
                return 1.0, 'no_data'

            c = df_4h['close'].astype(float)
            ema20 = c.ewm(span=20, adjust=False).mean()
            cp = float(c.iloc[-1])
            ema20_val = float(ema20.iloc[-1])

            if len(c) >= 50:
                ema50_s = c.ewm(span=50, adjust=False).mean()
                ema50_val = float(ema50_s.iloc[-1])
                htf_bullish = cp > ema20_val and ema20_val > ema50_val
                htf_bearish = cp < ema20_val and ema20_val < ema50_val
            else:
                htf_bullish = cp > ema20_val
                htf_bearish = cp < ema20_val

            if direction == 'LONG':
                if htf_bearish:
                    return 0.3, 'htf_conflict'
                if htf_bullish:
                    return 1.3, 'htf_aligned'
                return 1.0, 'htf_neutral'
            elif direction == 'SHORT':
                if htf_bullish:
                    return 0.3, 'htf_conflict'
                if htf_bearish:
                    return 1.3, 'htf_aligned'
                return 1.0, 'htf_neutral'

            return 1.0, 'neutral'
        except Exception:
            return 1.0, 'error'

    # ──────────────────────────── Cooldown ────────────────────────────────────

    def _bb_mr_check_cooldown(self) -> bool:
        """Returns True = trade a\u00e7\u0131labilir, False = cooldown aktif."""
        self._bb_mr_init_state()
        now = time.time()
        if now - self._bb_last_trade_time < 3 * 3600:
            return False
        if now < self._bb_cooldown_until:
            return False
        return True

    def _bb_mr_record_outcome(self, outcome_type: str, symbol: str = '',
                               realized_r: float = 0):
        """Trade sonucunu kaydet, cooldown tetikle, symbol stats g\u00fcncelle."""
        self._bb_mr_init_state()
        now = time.time()

        if outcome_type == 'SL':
            self._bb_consecutive_sl += 1
            if self._bb_consecutive_sl >= 3:
                self._bb_cooldown_until = now + 18 * 3600
                self.log(f"[COOL] BB MR 3 ard\u0131\u015f\u0131k SL \u2192 18 saat mola")
                self._bb_consecutive_sl = 0
            if symbol:
                self._bb_symbol_cooldown[symbol] = now + 3 * 3600
        elif outcome_type == 'TP':
            self._bb_consecutive_sl = 0

        self._bb_recent_outcomes.append(outcome_type)
        if len(self._bb_recent_outcomes) > 10:
            self._bb_recent_outcomes.pop(0)
        if len(self._bb_recent_outcomes) >= 10:
            sl_count = sum(1 for o in self._bb_recent_outcomes if o == 'SL')
            if sl_count >= 7:
                new_cooldown = now + 15 * 3600
                self._bb_cooldown_until = max(self._bb_cooldown_until, new_cooldown)
                self.log(f"[COOL] 7/10 SL \u2192 15 saat ek mola")
                self._bb_recent_outcomes.clear()

        if symbol:
            self._bb_mr_update_symbol_stats(symbol, outcome_type, realized_r)

    # ──────────────────────────── Scan Loop ───────────────────────────────────

    def _bb_mr_scan(self, fetcher: DataFetcher):
        """
        v7.1 BB Mean Reversion \u2014 2-pass scan:
          PASS 1: T\u00fcm coinleri tara, sinyal olanlar\u0131 topla + skorla
          PASS 1.5: HTF (4H) do\u011frulama + quality bonus
          PASS 2: Composite score ile s\u0131rala, en iyisini a\u00e7
        """
        self._bb_mr_init_state()

        if not self._check_loss_limits():
            return

        if not self._bb_mr_check_cooldown():
            remaining = max(0, self._bb_cooldown_until - time.time())
            if remaining > 0:
                self.log(f"[COOL] BB MR cooldown aktif: {int(remaining/3600)}h {int((remaining%3600)/60)}m kaldi")
            return

        candidates = []
        skipped = 0

        with self.trades_lock:
            active_symbols = set(t['symbol'] for t in self.open_trades)
            pending_symbols = set(p['symbol'] for p in self.pending_orders)
            active_count = len(self.open_trades) + len(self.pending_orders)
            can_open = active_count < self.max_open_trades_limit

        if not can_open:
            self.log("[PERF] BB MR scan: limit dolu, yeni trade acilamaz")
            return

        for i, symbol in enumerate(self.scanned_symbols[:]):
            if not self.is_running:
                break

            try:
                if i % 30 == 0:
                    self.log(f"  [BB MR] Scanning {symbol} ({i}/{len(self.scanned_symbols)})")

                if symbol in active_symbols or symbol in pending_symbols:
                    continue

                sym_cd = self._bb_symbol_cooldown.get(symbol, 0)
                if time.time() < sym_cd:
                    continue

                if self._bb_mr_is_symbol_blacklisted(symbol):
                    continue

                if not self._check_correlation(symbol):
                    continue

                df_1h = fetcher.fetch_ohlcv(symbol, '1h', limit=100)
                if df_1h is None or df_1h.empty or len(df_1h) < 55:
                    continue

                df_1h = add_all_indicators(df_1h)

                sig = self._bb_mr_signal(df_1h)
                if sig is None:
                    skipped += 1
                    continue

                sig['symbol'] = symbol
                sig['close'] = float(df_1h['close'].iloc[-1])
                candidates.append(sig)

            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str:
                    time.sleep(1.0)
                else:
                    self.log(f"[ERR] BB MR scan {symbol}: {e}")
                    time.sleep(0.3)

        if not candidates:
            self.log(f"[PERF] BB MR scan: 0 signal, {skipped} skipped")
            return

        # HTF (4H) dogrulama
        p = self._BB_MR_PARAMS
        filtered = []
        for sig in candidates:
            htf_mult, htf_reason = self._bb_mr_htf_check(
                sig['symbol'], sig['direction'], fetcher)

            if htf_mult <= 0.3:
                self.log(
                    f"  [HTF] {sig['symbol']} {sig['direction']} REJECTED: {htf_reason}"
                )
                continue

            htf_bonus = 0
            if htf_reason == 'htf_aligned':
                htf_bonus = 8
            elif htf_reason == 'htf_neutral':
                htf_bonus = 5
            elif htf_reason == 'htf_conflict':
                htf_bonus = -5

            sig['quality_score'] = round(sig['quality_score'] + htf_bonus, 1)
            sig['htf_mult'] = htf_mult
            sig['htf_reason'] = htf_reason

            min_q = p['min_quality_counter'] if sig.get('is_counter_trend') else p['min_quality_base']
            if sig['quality_score'] < min_q:
                continue

            filtered.append(sig)

        if not filtered:
            self.log(f"[PERF] BB MR scan: {len(candidates)} raw -> 0 after HTF/quality filter")
            return

        # En iyi sinyali sec ve ac
        filtered.sort(key=lambda x: x['quality_score'], reverse=True)

        top_n = min(len(filtered), 5)
        self.log(f"[RANK] BB MR v7.1: {len(filtered)} sinyal - Top {top_n}:")
        for j, c in enumerate(filtered[:top_n]):
            marker = ">>>" if j == 0 else "   "
            self.log(
                f"  {marker} #{j+1} {c['symbol']} {c['direction']} | "
                f"Q:{c['quality_score']} R:R:{c['rr_ratio']} "
                f"Rej:{c.get('rej_score',0)} Div:{c.get('has_divergence',False)} "
                f"ADX:{c.get('adx_val',0)} HTF:{c.get('htf_reason','?')}"
            )

        best = filtered[0]
        symbol = best['symbol']
        direction = best['direction']
        cp = best['close']
        atr_val = best['atr']

        # Position sizing — risk cap clamps qty
        user_max_notional = getattr(self, '_user_max_notional', 150.0)
        max_notional = min(user_max_notional, self.balance * 0.10)
        qty = max_notional / cp
        sl_dist = abs(cp - best['sl_price'])
        max_loss_limit = getattr(self, '_user_max_loss_cap', 5.0)
        if sl_dist > 0:
            max_qty_by_risk = max_loss_limit / sl_dist
            qty = min(qty, max_qty_by_risk)

        logger_id = f"{symbol}_{direction}_{int(time.time())}"

        with self.trades_lock:
            tid = self._open_locked(
                symbol=symbol,
                side=direction,
                price=cp,
                multiplier=1.0,
                tp_price=best['tp_price'],
                sl_price=best['sl_price'],
                signal_result={
                    'strategy': 'bb_mr_v7.1',
                    'regime': 'ranging',
                    'entry_type': 'market',
                    'soft_score': 5,
                    'signal': direction,
                    'quality_score': best['quality_score'],
                    'rej_score': best.get('rej_score', 0),
                    'vol_score': best.get('vol_score', 0),
                    'squeeze_score': best.get('squeeze_score', 0),
                    'htf_reason': best.get('htf_reason', 'unknown'),
                    'has_divergence': best.get('has_divergence', False),
                    'adx_val': best.get('adx_val', 0),
                    'session': best.get('session', 'unknown'),
                },
                absolute_qty=qty,
                atr=atr_val,
                logger_id=logger_id,
            )
            if tid:
                for t in self.open_trades:
                    if t.get('id') == tid:
                        t['_bb_mid_target'] = best.get('bb_mid', 0)
                        t['_cached_rsi'] = best.get('rsi', None)
                        t['_quality_score'] = best.get('quality_score', 50)
                        t['_initial_r'] = best.get('_initial_r', sl_dist)
                        break
                self._bb_last_trade_time = time.time()

        self.log(
            f"[PERF] BB MR v7.1: {len(filtered)} signal(s), {skipped} skipped, "
            f"BEST: {symbol} Q={best['quality_score']} HTF={best.get('htf_reason','?')}"
        )

