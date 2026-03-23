"""
ICT/SMC v2.3 Strategy — Gate mandatory + RSI + Zone + POI hard + Quality 8
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Optional

from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators
from ai.entry_gate import five_layer_gate


class ICTSMCStrategyMixin:
    """ICT/SMC v2.3 — Smart Money Concepts Strategy Mixin."""

    # ── State Initialization ──
    def _ict_init_state(self):
        if not hasattr(self, '_ict_last_trade_time'):
            self._ict_last_trade_time = 0
        if not hasattr(self, '_ict_consecutive_sl'):
            self._ict_consecutive_sl = 0
        if not hasattr(self, '_ict_cooldown_until'):
            self._ict_cooldown_until = 0
        if not hasattr(self, '_ict_recent_outcomes'):
            self._ict_recent_outcomes = []
        if not hasattr(self, '_ict_symbol_cooldown'):
            self._ict_symbol_cooldown = {}
        if not hasattr(self, '_ict_symbol_sl_streak'):
            self._ict_symbol_sl_streak = {}
        if not hasattr(self, '_ict_side_outcomes'):
            self._ict_side_outcomes = []
        if not hasattr(self, '_ict_symbol_outcomes'):
            self._ict_symbol_outcomes = {}
        if not hasattr(self, '_trail_df_cache'):
            self._trail_df_cache = {}

    def _ict_side_win_rate(self, side: str, window: int = 20) -> Optional[float]:
        self._ict_init_state()
        recent = [x for x in self._ict_side_outcomes[-window:] if x.get('side') == side]
        if len(recent) < max(6, window // 3):
            return None
        wins = sum(1 for x in recent if x.get('outcome') == 'TP')
        return wins / len(recent) if recent else None

    def _ict_global_win_rate(self, window: int = 12) -> Optional[float]:
        self._ict_init_state()
        recent = self._ict_recent_outcomes[-window:]
        if len(recent) < max(6, window // 2):
            return None
        wins = sum(1 for x in recent if x == 'TP')
        return wins / len(recent) if recent else None

    def _ict_symbol_health(self, symbol: str, window: int = 30) -> Optional[Dict]:
        self._ict_init_state()
        items = list(self._ict_symbol_outcomes.get(symbol, []))
        if not items:
            return None
        recent = items[-max(6, window):]
        count = len(recent)
        if count < 6:
            return None

        wins = 0
        gross_profit_r = 0.0
        gross_loss_r = 0.0
        for it in recent:
            r_val = float(it.get('realized_r', 0.0) or 0.0)
            out = str(it.get('outcome', ''))
            if out == 'TP' or r_val > 0:
                wins += 1
                gross_profit_r += max(r_val, 0.0)
            else:
                gross_loss_r += abs(min(r_val, 0.0))

        wr = wins / count
        pf = 9.99 if gross_loss_r <= 1e-9 else (gross_profit_r / gross_loss_r)
        avg_r = sum(float(it.get('realized_r', 0.0) or 0.0) for it in recent) / count
        return {
            'count': count,
            'wr': wr,
            'pf': pf,
            'avg_r': avg_r,
        }

    def _calc_realized_r(self, side, entry, current_price, initial_r):
        if initial_r <= 0:
            return 0.0
        if side == 'LONG':
            return (current_price - entry) / initial_r
        return (entry - current_price) / initial_r

    def _get_fetcher(self):
        if not hasattr(self, '_cached_fetcher'):
            from backtest.data_fetcher import DataFetcher
            self._cached_fetcher = DataFetcher('binance')
        return self._cached_fetcher

    def _get_swing_params(self, timeframe: str = '1h'):
        defaults = {
            '15m': {'left': 3, 'right': 2},
            '1h': {'left': 5, 'right': 3},
            '4h': {'left': 5, 'right': 5},
            '1d': {'left': 7, 'right': 5},
        }
        params = getattr(self, '_ICT_PARAMS', {}) or {}
        cfg_all = params.get('swing_params', {})
        tf_cfg = cfg_all.get(timeframe, defaults.get(timeframe, defaults['1h']))

        left = int(tf_cfg.get('left', defaults.get(timeframe, defaults['1h'])['left']))
        right = int(tf_cfg.get('right', defaults.get(timeframe, defaults['1h'])['right']))
        return max(1, left), max(1, right)

    def _zone_distance_atr(self, cp: float, zone, atr: float) -> float:
        if atr <= 0:
            return float('inf')
        try:
            top = float(zone.top)
            bottom = float(zone.bottom)
        except Exception:
            return float('inf')

        if bottom <= cp <= top:
            return 0.0

        dist = min(abs(cp - top), abs(cp - bottom))
        return dist / atr

    def _zone_bars_ago(self, zone, df_len: int) -> Optional[int]:
        if df_len <= 0:
            return None
        idx = None
        for attr in ('bar_index', 'index', 'idx'):
            v = getattr(zone, attr, None)
            if isinstance(v, (int, np.integer)):
                idx = int(v)
                break

        if idx is None or idx < 0 or idx >= df_len:
            return None
        return max(0, df_len - 1 - idx)

    def _filter_active_zones(self, zones, cp: float, atr: float,
                             max_distance_atr: float,
                             df_len: int = 0,
                             max_age_bars: int = 0):
        filtered = []
        for z in zones:
            d_atr = self._zone_distance_atr(cp, z, atr)
            if d_atr > max_distance_atr:
                continue

            if max_age_bars > 0:
                bars_ago = self._zone_bars_ago(z, df_len)
                if bars_ago is not None and bars_ago > max_age_bars:
                    continue

            filtered.append(z)
        return filtered

    _CORRELATED_GROUPS = {
        'BTC_GROUP': {'BTC/USDT', 'ETH/USDT'},
        'ALT_L1': {'SOL/USDT', 'AVAX/USDT', 'DOT/USDT', 'ADA/USDT', 'NEAR/USDT', 'SUI/USDT'},
        'ALT_L2': {'LINK/USDT', 'AAVE/USDT', 'UNI/USDT', 'MKR/USDT'},
        'MEME': {'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'WIF/USDT', 'FLOKI/USDT'},
    }

    def _get_correlation_group(self, symbol: str) -> str:
        for group, syms in self._CORRELATED_GROUPS.items():
            if symbol in syms:
                return group
        return symbol

    def _check_correlation(self, symbol: str) -> bool:
        group = self._get_correlation_group(symbol)
        with self.trades_lock:
            for t in self.open_trades:
                if self._get_correlation_group(t['symbol']) == group:
                    return False
        return True

    # ══════════════════════════════════════════════════════════════════════
    # QUICK FILTER — v2.2: Sweep zorunlu değil, sadece trend+yapı+OB
    # ══════════════════════════════════════════════════════════════════════
    def _ict_quick_filter(self, df_1h: pd.DataFrame) -> Optional[Dict]:
        from ai import ict_core

        if df_1h is None or len(df_1h) < 50:
            return None

        sw_left, sw_right = self._get_swing_params('1h')
        ict_1h = ict_core.analyze(df_1h, '', swing_left=sw_left, swing_right=sw_right)

        htf_ms = ict_1h.market_structure
        if htf_ms == 'ranging':
            return None

        direction = 'LONG' if htf_ms == 'bullish' else 'SHORT'

        exp_dir = 'bullish' if direction == 'LONG' else 'bearish'
        has_struct = (ict_1h.last_bos and ict_1h.last_bos.direction == exp_dir) or \
                     (ict_1h.last_choch and ict_1h.last_choch.direction == exp_dir)
        if not has_struct:
            return None

        ob_type = 'bullish' if direction == 'LONG' else 'bearish'
        has_ob = any(ob for ob in ict_1h.order_blocks if not ob.mitigated and ob.type == ob_type)
        has_fvg = any(f for f in ict_1h.fvg_zones if not f.filled and f.type == ob_type)
        if not has_ob and not has_fvg:
            return None

        return {'direction': direction, 'htf_ms': htf_ms, 'df_1h': df_1h, 'ict_1h': ict_1h}

    # ══════════════════════════════════════════════════════════════════════
    # SIGNAL — v2.3: Gate mandatory + RSI + Zone + POI hard + Quality 8
    # ══════════════════════════════════════════════════════════════════════
    def _ict_smc_signal(self, df_1h: pd.DataFrame,
                        df_4h: Optional[pd.DataFrame] = None,
                        df_5m: Optional[pd.DataFrame] = None,
                        cached_ict_1h=None,
                        symbol: str = '') -> Optional[Dict]:
        """
        v2.3 — Gate mandatory + RSI overbought + Premium/Discount zone.
        """
        from ai import ict_core

        if df_1h is None or len(df_1h) < 50:
            return None

        params = self._ICT_PARAMS

        # ── HTF Analiz ──
        ict_htf = None
        sw_4h_left, sw_4h_right = self._get_swing_params('4h')
        sw_1h_left, sw_1h_right = self._get_swing_params('1h')

        if df_4h is not None and len(df_4h) >= 30:
            ict_htf = ict_core.analyze(df_4h, '', swing_left=sw_4h_left, swing_right=sw_4h_right)

        ict_1h = cached_ict_1h if cached_ict_1h is not None else \
            ict_core.analyze(df_1h, '', swing_left=sw_1h_left, swing_right=sw_1h_right)

        if ict_htf is None:
            ict_htf = ict_1h

        htf_ms = ict_htf.market_structure
        if htf_ms == 'ranging':
            self.log(f"  [DIAG] {symbol}: REJECT → HTF ranging")
            return None

        direction = 'LONG' if htf_ms == 'bullish' else 'SHORT'
        ict_1h_dir = ict_core.analyze(df_1h, direction, swing_left=sw_1h_left, swing_right=sw_1h_right)

        # ── 4h yön uyumu guard (counter-trend trade'leri sıkılaştır) ──
        counter_trend_4h = False
        ema200_dist_4h = 0.0
        ema50_slope_4h = 0.0
        if df_4h is not None and len(df_4h) >= 30:
            c4 = df_4h['close'].astype(float)
            cp4 = float(c4.iloc[-1])
            ema200_4h = float(c4.ewm(span=200, adjust=False).mean().iloc[-1])
            ema50_4h = c4.ewm(span=50, adjust=False).mean()
            ema50_now = float(ema50_4h.iloc[-1])
            ema50_prev = float(ema50_4h.iloc[-6]) if len(ema50_4h) >= 6 else ema50_now
            if ema200_4h > 0:
                ema200_dist_4h = (cp4 - ema200_4h) / ema200_4h
            if abs(ema50_prev) > 1e-12:
                ema50_slope_4h = (ema50_now - ema50_prev) / abs(ema50_prev)

            if direction == 'LONG' and ema200_dist_4h < -0.01 and ema50_slope_4h <= -0.003:
                counter_trend_4h = True
            if direction == 'SHORT' and ema200_dist_4h > 0.01 and ema50_slope_4h >= 0.003:
                counter_trend_4h = True

        # ── Sweep + Displacement ──
        expected_sweep = 'ssl_sweep' if direction == 'LONG' else 'bsl_sweep'
        has_sweep = ict_1h_dir.sweep_detected and ict_1h_dir.sweep_type == expected_sweep

        exp_disp = 'bullish' if direction == 'LONG' else 'bearish'
        has_disp = ict_1h_dir.displacement and ict_1h_dir.displacement_direction == exp_disp

        # ── BOS/CHoCH ──
        has_bos = False
        has_choch = False
        if direction == 'LONG':
            has_bos = bool(ict_1h_dir.last_bos and ict_1h_dir.last_bos.direction == 'bullish')
            has_choch = bool(ict_1h_dir.last_choch and ict_1h_dir.last_choch.direction == 'bullish')
        else:
            has_bos = bool(ict_1h_dir.last_bos and ict_1h_dir.last_bos.direction == 'bearish')
            has_choch = bool(ict_1h_dir.last_choch and ict_1h_dir.last_choch.direction == 'bearish')

        if not has_bos and not has_choch:
            self.log(f"  [DIAG] {symbol}: REJECT → No BOS/CHoCH for {direction}")
            return None

        # Setup türü (diagnostic + dinamik kalite kuralı)
        setup_type = 'trend'
        if has_choch and not has_bos:
            setup_type = 'choch'
        elif has_sweep:
            setup_type = 'sweep'

        if counter_trend_4h and not (has_sweep and has_disp):
            self.log(
                f"  [DIAG] {symbol}: REJECT → 4h counter-trend guard "
                f"(dist={ema200_dist_4h*100:+.1f}%, slope={ema50_slope_4h*100:+.2f}%)"
            )
            return None

        if setup_type == 'choch' and not (has_sweep or has_disp):
            self.log(f"  [DIAG] {symbol}: REJECT → CHoCH weak trigger (need sweep/disp)")
            return None

        # ── OB/FVG ──
        ob_type = 'bullish' if direction == 'LONG' else 'bearish'
        active_obs = [ob for ob in ict_1h_dir.order_blocks if not ob.mitigated and ob.type == ob_type]
        active_fvgs = [f for f in ict_1h_dir.fvg_zones if not f.filled and f.type == ob_type]

        # ── Killzone (bilgi amaçlı — BLOKLAMAZ) ──
        sess = ict_1h_dir.session
        killzone_name = sess.killzone if sess else 'off'

        # ── Weekend (koruyoruz) ──
        if sess and sess.weekday >= 5:
            self.log(f"  [DIAG] {symbol}: REJECT → Weekend")
            return None

        # ── Fiyat + ATR ──
        cp = float(df_1h['close'].iloc[-1])
        if 'atr' in df_1h.columns:
            atr_val = float(df_1h['atr'].iloc[-1])
            if np.isnan(atr_val) or atr_val <= 0:
                atr_val = cp * 0.01
        else:
            atr_val = cp * 0.01

        max_poi_distance_atr = float(params.get('max_poi_distance_atr', 5.0))
        max_zone_age_bars = int(params.get('max_zone_age_bars', 72))

        active_obs = self._filter_active_zones(
            active_obs,
            cp=cp,
            atr=atr_val,
            max_distance_atr=max_poi_distance_atr,
            df_len=len(df_1h),
            max_age_bars=max_zone_age_bars,
        )
        active_fvgs = self._filter_active_zones(
            active_fvgs,
            cp=cp,
            atr=atr_val,
            max_distance_atr=max_poi_distance_atr,
            df_len=len(df_1h),
            max_age_bars=max_zone_age_bars,
        )

        if not active_obs and not active_fvgs:
            self.log(
                f"  [DIAG] {symbol}: REJECT → No active OB/FVG near price "
                f"(>{max_poi_distance_atr:.1f} ATR or too old)"
            )
            return None

        # ── RSI: hard veto yerine extreme guard + quality penalty ──
        rsi_val = None
        rsi_stretched = False
        if 'rsi' in df_1h.columns:
            rsi_val = float(df_1h['rsi'].iloc[-1])
            if not np.isnan(rsi_val):
                rsi_overbought = float(params.get('rsi_overbought', 72.0))
                rsi_oversold = float(params.get('rsi_oversold', 28.0))
                rsi_extreme_overbought = float(params.get('rsi_extreme_overbought', 82.0))
                rsi_extreme_oversold = float(params.get('rsi_extreme_oversold', 18.0))

                if direction == 'LONG' and rsi_val >= rsi_extreme_overbought:
                    self.log(f"  [DIAG] {symbol}: REJECT → RSI {rsi_val:.1f} >= {rsi_extreme_overbought:.0f} (extreme overbought)")
                    return None
                if direction == 'SHORT' and rsi_val <= rsi_extreme_oversold:
                    self.log(f"  [DIAG] {symbol}: REJECT → RSI {rsi_val:.1f} <= {rsi_extreme_oversold:.0f} (extreme oversold)")
                    return None

                if direction == 'LONG' and rsi_val >= rsi_overbought:
                    rsi_stretched = True
                if direction == 'SHORT' and rsi_val <= rsi_oversold:
                    rsi_stretched = True

        # ── v2.5: HTF Resistance Proximity Check (CRITICAL FIX) ──
        # Prevent buying near recent highs (DEXE $5.579 was 93% of $6.000 high)
        htf_lookback = min(100, len(df_1h))  # 100 bars = ~4 days on 1h
        htf_slice = df_1h.iloc[-htf_lookback:]
        htf_high = float(htf_slice['high'].max())
        htf_low = float(htf_slice['low'].min())
        
        entry_pullback_min_pct = float(params.get('entry_pullback_min_pct', 0.015))

        if direction == 'LONG':
            # Reject if price is within 5% of recent high (resistance zone)
            distance_from_high = (htf_high - cp) / htf_high
            sr_proximity_limit = float(params.get('sr_proximity_limit', 0.05))
            if distance_from_high < sr_proximity_limit:
                self.log(f"  [DIAG] {symbol}: REJECT → Too close to resistance (${cp:.3f} vs ${htf_high:.3f}, {distance_from_high*100:.1f}% away)")
                return None
            
            # Require pullback: price must have retraced at least 3% from recent high
            recent_high_10bars = float(df_1h.iloc[-10:]['high'].max())
            pullback_pct = (recent_high_10bars - cp) / recent_high_10bars
            if entry_pullback_min_pct > 0 and pullback_pct < entry_pullback_min_pct:
                self.log(f"  [DIAG] {symbol}: REJECT → No pullback (only {pullback_pct*100:.1f}% from recent high ${recent_high_10bars:.3f})")
                return None
        
        elif direction == 'SHORT':
            # Reject if price is within 5% of recent low (support zone)
            distance_from_low = (cp - htf_low) / htf_low
            sr_proximity_limit = float(params.get('sr_proximity_limit', 0.05))
            if distance_from_low < sr_proximity_limit:
                self.log(f"  [DIAG] {symbol}: REJECT → Too close to support (${cp:.3f} vs ${htf_low:.3f}, {distance_from_low*100:.1f}% away)")
                return None
            
            # Require pullback for SHORT
            recent_low_10bars = float(df_1h.iloc[-10:]['low'].min())
            pullback_pct = (cp - recent_low_10bars) / recent_low_10bars
            if entry_pullback_min_pct > 0 and pullback_pct < entry_pullback_min_pct:
                self.log(f"  [DIAG] {symbol}: REJECT → No pullback (only {pullback_pct*100:.1f}% from recent low ${recent_low_10bars:.3f})")
                return None
        
        # ── Entry location / chasing filters (backtest-aligned) ──
        zone_lookback = min(int(params.get('entry_range_lookback', 20)), len(df_1h))
        recent_slice = df_1h.iloc[-zone_lookback:]
        range_high = float(recent_slice['high'].max())
        range_low = float(recent_slice['low'].min())
        entry_range_top_ceil = float(params.get('entry_range_top_ceil', 0.85))
        entry_range_bot_floor = float(params.get('entry_range_bot_floor', 0.15))
        if range_high > range_low:
            price_position = (cp - range_low) / (range_high - range_low)
            if direction == 'LONG' and price_position > entry_range_top_ceil:
                self.log(
                    f"  [DIAG] {symbol}: REJECT → LONG premium zone "
                    f"(pos={price_position:.0%}, need <{entry_range_top_ceil:.0%})"
                )
                return None
            if direction == 'SHORT' and price_position < entry_range_bot_floor:
                self.log(
                    f"  [DIAG] {symbol}: REJECT → SHORT discount zone "
                    f"(pos={price_position:.0%}, need >{entry_range_bot_floor:.0%})"
                )
                return None

        ema21 = float(df_1h['close'].astype(float).ewm(span=21, adjust=False).mean().iloc[-1])
        entry_max_ema21_ext = float(params.get('entry_max_ema21_ext', 0.045))
        if ema21 > 0 and entry_max_ema21_ext > 0:
            if direction == 'LONG' and cp > ema21 * (1 + entry_max_ema21_ext):
                ext_pct = ((cp / ema21) - 1.0) * 100
                self.log(f"  [DIAG] {symbol}: REJECT → LONG chasing EMA21 (ext={ext_pct:.2f}%)")
                return None
            if direction == 'SHORT' and cp < ema21 * (1 - entry_max_ema21_ext):
                ext_pct = (1.0 - (cp / ema21)) * 100
                self.log(f"  [DIAG] {symbol}: REJECT → SHORT chasing EMA21 (ext={ext_pct:.2f}%)")
                return None

        # ── v2.5: POI Confluence (INCREASED: 2 → 3) ──
        cc = ict_1h_dir.poi_details.get('confluence_count', 0)
        min_confluence = int(params.get('min_confluence', 3))
        if cc < min_confluence:
            self.log(f"  [DIAG] {symbol}: REJECT → Low confluence ({cc} < {min_confluence})")
            return None

        # ── v2.3: POI Proximity → HARD filtre (3 ATR max) ──
        poi_ok, poi_dist, poi_reason = self._check_poi_proximity(
            direction, cp, ict_1h_dir, atr_val)
        if not poi_ok:
            self.log(f"  [DIAG] {symbol}: REJECT → POI too far ({poi_dist/atr_val:.1f} ATR)")
            return None

        poi_dist_atr = (poi_dist / atr_val) if atr_val > 0 else 999.0
        base_chase_limit = float(params.get('entry_max_poi_dist_atr', 1.6))
        if setup_type == 'choch':
            base_chase_limit = min(base_chase_limit, float(params.get('choch_max_poi_dist_atr', 1.25)))
        if counter_trend_4h:
            base_chase_limit = min(base_chase_limit, float(params.get('counter_trend_max_poi_dist_atr', 1.0)))
        if poi_reason == 'near_poi' and poi_dist_atr > base_chase_limit:
            self.log(
                f"  [DIAG] {symbol}: REJECT → Chasing entry ({poi_dist_atr:.2f} ATR > {base_chase_limit:.2f})"
            )
            return None

        # ── Trend Strength (çok düşük eşik) ──
        trend_recent_bars = int(params.get('trend_recent_bars_4h', 48))
        trend_str, trend_reason = self._check_trend_strength(
            ict_htf,
            direction,
            recent_bars=trend_recent_bars,
        )
        if trend_str < 0.10:
            self.log(f"  [DIAG] {symbol}: REJECT → Trend çok zayıf ({trend_str:.0%})")
            return None

        # ── 5M Rejection (bonus) ──
        has_rejection = False
        rejection_reason = ''
        if df_5m is not None and len(df_5m) >= 3:
            poi_price = cp
            for ob in active_obs:
                poi_price = ob.top if direction == 'LONG' else ob.bottom
                break
            for fvg in active_fvgs:
                poi_price = fvg.ce
                break
            atr_5m = cp * 0.003
            if len(df_5m) >= 14:
                atr_5m_calc = float((df_5m['high'] - df_5m['low']).astype(float).rolling(14).mean().iloc[-1])
                if not np.isnan(atr_5m_calc) and atr_5m_calc > 0:
                    atr_5m = atr_5m_calc
            has_rejection, rejection_reason = self._check_5m_rejection(
                df_5m, direction, poi_price, atr_5m)

        if counter_trend_4h and not has_rejection:
            self.log(f"  [DIAG] {symbol}: REJECT → 4h counter-trend needs 5m rejection")
            return None

        if setup_type == 'choch' and not has_rejection:
            self.log(f"  [DIAG] {symbol}: REJECT → CHoCH needs 5m rejection")
            return None

        # ── SL hesaplama ──
        sweep_sl = ict_core.get_sweep_sl(
            direction, ict_1h_dir.sweep_level, cp, atr_val,
            ict_1h_dir.swing_highs, ict_1h_dir.swing_lows
        )

        max_sl_pct = float(params.get('max_sl_pct', 3.0)) / 100
        min_stop_pct = float(params.get('min_stop_pct', 0.7)) / 100
        base_sl_atr = float(params.get('base_sl_atr', 1.25))
        if setup_type == 'choch':
            base_sl_atr = min(base_sl_atr, float(params.get('choch_sl_atr', 1.10)))
        if counter_trend_4h:
            base_sl_atr = min(base_sl_atr, float(params.get('counter_trend_sl_atr', 0.95)))
        elif trend_str >= 0.70:
            base_sl_atr = max(base_sl_atr, float(params.get('strong_trend_sl_atr', 1.35)))

        if direction == 'LONG':
            atr_sl = cp - atr_val * base_sl_atr
            structural_sl = sweep_sl if sweep_sl < cp else cp * (1 - max_sl_pct)
            sl_price = min(structural_sl, atr_sl)
            sl_price = max(sl_price, cp * (1 - max_sl_pct))
            sl_price = min(sl_price, cp * (1 - min_stop_pct))
            if sl_price >= cp:
                sl_price = cp * (1 - max_sl_pct)
        else:
            atr_sl = cp + atr_val * base_sl_atr
            structural_sl = sweep_sl if sweep_sl > cp else cp * (1 + max_sl_pct)
            sl_price = max(structural_sl, atr_sl)
            sl_price = min(sl_price, cp * (1 + max_sl_pct))
            sl_price = max(sl_price, cp * (1 + min_stop_pct))
            if sl_price <= cp:
                sl_price = cp * (1 + max_sl_pct)

        # ── TP hesaplama ──
        liq_tp = ict_core.get_liquidity_tp(
            direction, cp,
            ict_1h_dir.equal_highs, ict_1h_dir.equal_lows,
            ict_1h_dir.swing_highs, ict_1h_dir.swing_lows
        )

        risk = abs(cp - sl_price)
        min_rr = params.get('min_rr', 1.8)  # v2.2: 2.0 → 1.8

        if direction == 'LONG':
            tp_price = max(liq_tp, cp + risk * min_rr)
        else:
            tp_price = min(liq_tp, cp - risk * min_rr)

        reward = abs(tp_price - cp)
        rr = reward / risk if risk > 0 else 0
        if rr < min_rr:
            self.log(f"  [DIAG] {symbol}: REJECT → R:R too low ({rr:.2f} < {min_rr})")
            return None

        # ── v2.3: Entry Gate → MANDATORY ──
        gate_passed = False
        gate_score = 0.0
        gate_summary = ''

        try:
            gate_result = five_layer_gate(
                direction=direction,
                df_1h=df_1h,
                df_5m=df_5m,
                df_4h=df_4h,
                entry_price=cp,
                sl_price=sl_price,
                tp_price=tp_price,
                meta_confidence=min(1.0, cc / 4.0),
                regime=htf_ms,
                strategy='ict_smc_v2',
            )
            gate_passed = gate_result.passed
            gate_score = gate_result.total_score
            gate_summary = gate_result.summary() if hasattr(gate_result, 'summary') else ''

            if gate_passed:
                if gate_result.sl_price > 0:
                    sl_price = gate_result.sl_price
                if gate_result.tp_price > 0:
                    tp_price = gate_result.tp_price
                if gate_result.rr_ratio > 0:
                    rr = gate_result.rr_ratio
            else:
                self.log(f"  [DIAG] {symbol}: REJECT → Gate FAIL (score={gate_score:.0%}) {gate_summary}")
                return None
        except Exception as e:
            self.log(f"  [DIAG] {symbol}: REJECT → Gate ERROR: {e}")
            return None

        # ── Volume spike ──
        vol_spike = False
        if 'volume' in df_1h.columns and len(df_1h) >= 20:
            vol = df_1h['volume'].astype(float)
            vma = vol.rolling(20, min_periods=5).mean()
            max_rvol = 0
            for vidx in range(max(0, len(df_1h) - 5), len(df_1h)):
                v_avg = float(vma.iloc[vidx]) if not np.isnan(vma.iloc[vidx]) else 0
                if v_avg > 0:
                    rvol = float(vol.iloc[vidx]) / v_avg
                    max_rvol = max(max_rvol, rvol)
            vol_spike = max_rvol > 1.5

        # ── v2.3: Quality Score ──
        quality = 0

        # Temel yapısal puanlar (structure-first)
        if has_sweep: quality += 2
        if has_disp: quality += 3
        if has_bos: quality += 3
        if has_choch: quality += 3
        if active_obs: quality += 1
        if active_fvgs: quality += 1

        # Confluence bonus
        if cc >= 3: quality += 2
        elif cc >= 2: quality += 1
        elif cc >= 1: quality += 1  # v2.2: 1 confluence da +1

        # OTE zone
        in_ote = False
        if ict_1h_dir.ote:
            ote = ict_1h_dir.ote
            if ote.bottom <= cp <= ote.top:
                quality += 2
                in_ote = True

        # POI proximity bonus (artık zorunlu değil, sadece bonus)
        if poi_reason in ('inside_ob', 'inside_fvg', 'inside_ote'):
            quality += 2
        elif poi_reason == 'near_poi':
            quality += 1
        # 'far_but_allowed' → 0 bonus

        # Killzone bonus
        if killzone_name == 'ny_am': quality += 2
        elif killzone_name in ('london', 'ny_pm'): quality += 1

        # Extras
        if vol_spike: quality += 2
        if rr >= 3.0: quality += 1
        if has_rejection: quality += 2
        if rsi_stretched: quality -= 1
        if poi_reason == 'near_poi' and poi_dist_atr > 1.2: quality -= 1

        # Gate bonus
        if gate_passed:
            quality += 2
            quality += int(gate_score * 2)
        elif gate_score >= 0.5:
            quality += 1

        # Trend bonus/penalty
        if trend_str >= 0.7: quality += 1
        elif trend_str < 0.30: quality -= 1
        if counter_trend_4h: quality -= 2

        # ══════════════════════════════════════════════════════════
        min_quality = int(params.get('min_quality', 6))
        global_wr = self._ict_global_win_rate(int(params.get('quality_wr_window', 12)) or 12)
        side_wr = self._ict_side_win_rate(direction, int(params.get('side_wr_window', 20)) or 20)

        if global_wr is not None and global_wr < float(params.get('quality_wr_floor', 0.40)):
            min_quality += 1
        if side_wr is not None and side_wr < float(params.get('side_wr_floor', 0.34)):
            min_quality += 1

        opposite_side = 'SHORT' if direction == 'LONG' else 'LONG'
        opposite_wr = self._ict_side_win_rate(opposite_side, int(params.get('side_wr_window', 20)) or 20)
        wr_gap_tighten = float(params.get('side_wr_gap_tighten', 0.06))
        side_underperforming = False
        if side_wr is not None and opposite_wr is not None and (side_wr + wr_gap_tighten) < opposite_wr:
            min_quality += 1
            side_underperforming = True

        if setup_type == 'choch' and not has_sweep:
            min_quality += 1
        if counter_trend_4h:
            min_quality += 1

        if side_underperforming and not (has_disp or has_rejection):
            self.log(
                f"  [DIAG] {symbol}: REJECT → {direction} underperforming side needs disp/rejection "
                f"(side_wr={side_wr:.1%}, opp_wr={opposite_wr:.1%})"
            )
            return None

        if quality < min_quality:
            self.log(
                f"  [DIAG] {symbol}: REJECT → Quality {quality} < {min_quality} | "
                f"sweep={has_sweep} disp={has_disp} bos={has_bos} choch={has_choch} "
                f"obs={len(active_obs)} fvgs={len(active_fvgs)} cc={cc} "
                f"ote={in_ote} kz={killzone_name} poi={poi_reason} "
                f"gate={'✓' if gate_passed else '✗'}({gate_score:.0%}) "
                f"rej={has_rejection} vol={vol_spike} rsi_stretched={rsi_stretched} "
                f"setup={setup_type} ctrend={counter_trend_4h} side_under={side_underperforming} rr={rr:.1f}"
            )
            return None

        # ═══ PASSED! Log detay ═══
        self.log(
            f"  [PASS] {symbol} {direction}: Q={quality} RR={rr:.1f} | "
            f"sweep={has_sweep} disp={has_disp} bos={has_bos} choch={has_choch} "
            f"obs={len(active_obs)} fvgs={len(active_fvgs)} cc={cc} "
            f"ote={in_ote} kz={killzone_name} poi={poi_reason} "
            f"gate={'✓' if gate_passed else '✗'}({gate_score:.0%}) "
            f"vol={vol_spike} rsi_stretched={rsi_stretched} setup={setup_type} "
            f"ctrend={counter_trend_4h} side_under={side_underperforming}"
        )

        return {
            'direction': direction,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'sl_atr_mult': round(base_sl_atr, 3),
            'quality_score': quality,
            'rr_ratio': round(rr, 2),
            'atr': atr_val,
            'confluence': cc,
            'has_sweep': has_sweep,
            'has_displacement': has_disp,
            'htf_trend': htf_ms,
            'obs': len(active_obs),
            'fvgs': len(active_fvgs),
            'killzone': killzone_name,
            'vol_spike': vol_spike,
            'has_bos': has_bos,
            'has_choch': has_choch,
            'in_ote': in_ote,
            'gate_passed': gate_passed,
            'gate_score': round(gate_score, 3),
            'gate_summary': gate_summary,
            'poi_proximity': poi_reason,
            'poi_dist_atr': round(poi_dist_atr, 2) if atr_val > 0 else 0,
            'trend_strength': round(trend_str, 2),
            'trend_detail': trend_reason,
            'has_rejection': has_rejection,
            'rejection_detail': rejection_reason,
            'rsi': round(rsi_val, 1) if rsi_val is not None else None,
            'rsi_stretched': rsi_stretched,
            'setup_type': setup_type,
            'counter_trend_4h': counter_trend_4h,
            'ema200_dist_4h': round(ema200_dist_4h * 100, 2),
            'ema50_slope_4h': round(ema50_slope_4h * 100, 3),
            'min_quality_used': min_quality,
            'side_wr': round(side_wr * 100, 2) if side_wr is not None else None,
            'opposite_wr': round(opposite_wr * 100, 2) if opposite_wr is not None else None,
            'side_underperforming': side_underperforming,
        }

    # ── POI Proximity (v2.2: 3 ATR, artık bloklama yok) ──
    def _check_poi_proximity(self, direction, cp, ict_1h, atr):
        ob_type = 'bullish' if direction == 'LONG' else 'bearish'
        max_distance = atr * 3.0

        best_ob_dist = float('inf')
        for ob in ict_1h.order_blocks:
            if ob.type == ob_type and not ob.mitigated:
                if direction == 'LONG':
                    if ob.bottom <= cp <= ob.top:
                        return True, 0.0, 'inside_ob'
                    dist = cp - ob.top
                    if 0 <= dist <= max_distance:
                        best_ob_dist = min(best_ob_dist, dist)
                else:
                    if ob.bottom <= cp <= ob.top:
                        return True, 0.0, 'inside_ob'
                    dist = ob.bottom - cp
                    if 0 <= dist <= max_distance:
                        best_ob_dist = min(best_ob_dist, dist)

        best_fvg_dist = float('inf')
        for fvg in ict_1h.fvg_zones:
            if fvg.type == ob_type and not fvg.filled:
                if fvg.bottom <= cp <= fvg.top:
                    return True, 0.0, 'inside_fvg'
                dist = abs(cp - fvg.ce)
                if dist <= max_distance:
                    best_fvg_dist = min(best_fvg_dist, dist)

        if ict_1h.ote:
            ote = ict_1h.ote
            if ote.bottom <= cp <= ote.top:
                return True, 0.0, 'inside_ote'

        min_dist = min(best_ob_dist, best_fvg_dist)
        if min_dist <= max_distance:
            return True, min_dist, 'near_poi'

        return False, min_dist, 'too_far'

    # ── 5M Rejection ──
    def _check_5m_rejection(self, df_5m, direction, poi_price, atr_5m):
        if df_5m is None or len(df_5m) < 3:
            return False, 'no_data'

        for i in range(-3, 0):
            o_v = float(df_5m['open'].iloc[i])
            h_v = float(df_5m['high'].iloc[i])
            l_v = float(df_5m['low'].iloc[i])
            c_v = float(df_5m['close'].iloc[i])
            body = abs(c_v - o_v)
            full_range = h_v - l_v

            if full_range <= 0:
                continue

            if direction == 'LONG':
                lower_wick = min(o_v, c_v) - l_v
                wick_ratio = lower_wick / full_range
                if (wick_ratio > 0.5 and body < full_range * 0.4
                        and l_v <= poi_price + atr_5m * 0.5 and c_v > o_v):
                    return True, f'rejection@{l_v:.0f}'
                if i > -3:
                    prev_c = float(df_5m['close'].iloc[i - 1])
                    prev_o = float(df_5m['open'].iloc[i - 1])
                    if (prev_c < prev_o and c_v > o_v
                            and body > abs(prev_c - prev_o)
                            and l_v <= poi_price + atr_5m * 0.5):
                        return True, f'engulfing@{l_v:.0f}'
            elif direction == 'SHORT':
                upper_wick = h_v - max(o_v, c_v)
                wick_ratio = upper_wick / full_range
                if (wick_ratio > 0.5 and body < full_range * 0.4
                        and h_v >= poi_price - atr_5m * 0.5 and c_v < o_v):
                    return True, f'rejection@{h_v:.0f}'
                if i > -3:
                    prev_c = float(df_5m['close'].iloc[i - 1])
                    prev_o = float(df_5m['open'].iloc[i - 1])
                    if (prev_c > prev_o and c_v < o_v
                            and body > abs(prev_c - prev_o)
                            and h_v >= poi_price - atr_5m * 0.5):
                        return True, f'engulfing@{h_v:.0f}'

        return False, 'no_rejection'

    # ── Trend Strength ──
    def _check_trend_strength(self, ict_htf, direction, recent_bars: int = 48):
        all_labels = list(getattr(ict_htf, 'structure_labels', []) or [])
        if not all_labels:
            return 0.5, 'no_labels'

        params = getattr(self, '_ICT_PARAMS', {}) or {}
        min_labels = int(params.get('trend_min_labels', 8))
        max_labels = int(params.get('trend_max_labels', 20))

        indexed = []
        for lb in all_labels:
            idx = None
            for attr in ('bar_index', 'index', 'idx'):
                v = getattr(lb, attr, None)
                if isinstance(v, (int, np.integer)):
                    idx = int(v)
                    break
            indexed.append((lb, idx))

        with_index = [(lb, idx) for lb, idx in indexed if idx is not None]
        if with_index and recent_bars > 0:
            latest_idx = max(idx for _, idx in with_index)
            cutoff = latest_idx - recent_bars + 1
            recent = [lb for lb, idx in with_index if idx >= cutoff]
            label_pool = recent if len(recent) >= min_labels else [lb for lb, _ in indexed]
        else:
            label_pool = [lb for lb, _ in indexed]

        selected = label_pool[-max_labels:]
        if len(selected) < min_labels:
            return 0.4, f'low_conf(labels={len(selected)})'

        good_labels = ('HH', 'HL') if direction == 'LONG' else ('LL', 'LH')
        bad_labels = ('LL', 'LH') if direction == 'LONG' else ('HH', 'HL')

        raw = []
        for lb in selected:
            name = str(getattr(lb, 'label', '')).upper()
            if name in good_labels:
                raw.append(1)
            elif name in bad_labels:
                raw.append(-1)

        if not raw:
            return 0.5, f'neutral(labels={len(selected)})'

        positive = sum(1 for v in raw if v > 0)
        base_strength = positive / len(raw)

        weighted_score = 0.0
        weight_sum = 0.0
        for i, v in enumerate(raw):
            w = (i + 1) / len(raw)
            weighted_score += w * v
            weight_sum += w

        weighted_norm = (weighted_score / weight_sum + 1.0) / 2.0 if weight_sum > 0 else base_strength
        strength = (base_strength * 0.55) + (weighted_norm * 0.45)

        bos_count = sum(1 for sb in getattr(ict_htf, 'structure_breaks', [])
                        if sb.type == 'BOS' and
                        sb.direction == ('bullish' if direction == 'LONG' else 'bearish'))

        if bos_count >= 5:
            strength = max(0.0, strength - 0.15)
        has_choch = getattr(ict_htf, 'last_choch', None) and \
            ict_htf.last_choch.direction == ('bullish' if direction == 'LONG' else 'bearish')
        if has_choch and bos_count <= 2:
            strength = min(1.0, strength + 0.15)

        confidence = 'high' if len(selected) >= 12 else 'medium'
        return strength, f'hybrid(str={strength:.0%},labels={len(selected)},bos={bos_count},conf={confidence})'

    # ── Structural SL ──
    def _update_structural_sl(self, t, current_price, fetcher=None):
        symbol = t['symbol']
        side = t['side']
        entry = t['entry_price']
        current_sl = t['sl_price']

        pnl, _ = self._compute_pnl(t, current_price)
        if pnl <= 0:
            return
        if not t.get('_ict_tp1_hit'):
            return

        last_struct = t.get('_last_structural_update', 0)
        if time.time() - last_struct < 300:
            return

        try:
            from ai import ict_core
            if fetcher is None:
                fetcher = self._get_fetcher()

            df_1h = fetcher.fetch_ohlcv(symbol, '1h', limit=30)
            if df_1h is None or len(df_1h) < 10:
                return
            df_1h = add_all_indicators(df_1h)
            sw_left, sw_right = self._get_swing_params('1h')
            ict = ict_core.analyze(df_1h, side, swing_left=sw_left, swing_right=sw_right)
            atr = float(df_1h['atr'].iloc[-1]) if 'atr' in df_1h.columns else current_price * 0.01
            buffer = atr * 0.2

            if side == 'LONG':
                new_sl = current_sl
                for sw in reversed(ict.swing_lows):
                    candidate = sw.price - buffer
                    if candidate > entry and candidate > current_sl:
                        new_sl = max(new_sl, candidate)
                        break
                for ob in reversed(ict.order_blocks):
                    if ob.type == 'bullish' and not ob.mitigated:
                        candidate = ob.bottom - buffer
                        if candidate > entry and candidate > current_sl:
                            new_sl = max(new_sl, candidate)
                            break
                if ict.last_choch and ict.last_choch.direction == 'bearish':
                    tight_sl = current_price - atr * 0.5
                    new_sl = max(new_sl, tight_sl)
                    t['_choch_warning'] = True
                if new_sl > current_sl:
                    t['sl_price'] = new_sl
                    self.log(f"[STRUCT] {symbol} SL trail: {current_sl:.4f} → {new_sl:.4f}")
            elif side == 'SHORT':
                new_sl = current_sl
                for sw in reversed(ict.swing_highs):
                    candidate = sw.price + buffer
                    if candidate < entry and candidate < current_sl:
                        new_sl = min(new_sl, candidate)
                        break
                for ob in reversed(ict.order_blocks):
                    if ob.type == 'bearish' and not ob.mitigated:
                        candidate = ob.top + buffer
                        if candidate < entry and candidate < current_sl:
                            new_sl = min(new_sl, candidate)
                            break
                if ict.last_choch and ict.last_choch.direction == 'bullish':
                    tight_sl = current_price + atr * 0.5
                    new_sl = min(new_sl, tight_sl)
                    t['_choch_warning'] = True
                if new_sl < current_sl:
                    t['sl_price'] = new_sl
                    self.log(f"[STRUCT] {symbol} SL trail: {current_sl:.4f} → {new_sl:.4f}")

            t['_last_structural_update'] = time.time()
        except Exception as e:
            self.log(f"[WARN] Structural SL error {symbol}: {e}")

    # ── Cooldown — v2.2: Çok gevşek ──
    def _ict_check_cooldown(self) -> bool:
        self._ict_init_state()
        now = time.time()
        if now - self._ict_last_trade_time < 20:  # v2.2: 30s → 20s
            return False
        if now < self._ict_cooldown_until:
            return False
        return True

    def _ict_record_outcome(self, outcome_type: str, symbol: str = '',
                            realized_r: float = 0.0, side: str = ''):
        self._ict_init_state()
        now = time.time()
        base_symbol_cd = int(getattr(self, '_ICT_PARAMS', {}).get('symbol_sl_cooldown_sec', 2 * 3600))

        if outcome_type == 'SL':
            self._ict_consecutive_sl += 1
            if self._ict_consecutive_sl >= 3:
                self._ict_cooldown_until = now + 1800  # 30 dk
                self.log(f"[COOL] ICT 3 ardışık SL → 30 dk mola")
                self._ict_consecutive_sl = 0
            if symbol:
                streak = int(self._ict_symbol_sl_streak.get(symbol, 0)) + 1
                self._ict_symbol_sl_streak[symbol] = streak
                mult = 1
                if streak >= 3:
                    mult = 4
                elif streak >= 2:
                    mult = 2
                self._ict_symbol_cooldown[symbol] = now + base_symbol_cd * mult
        elif outcome_type == 'TP':
            self._ict_consecutive_sl = 0
            if symbol:
                self._ict_symbol_sl_streak[symbol] = 0

        self._ict_recent_outcomes.append(outcome_type)
        if len(self._ict_recent_outcomes) > 10:
            self._ict_recent_outcomes.pop(0)

        if side in ('LONG', 'SHORT'):
            self._ict_side_outcomes.append({'side': side, 'outcome': outcome_type, 'ts': now})
            if len(self._ict_side_outcomes) > 60:
                self._ict_side_outcomes.pop(0)

        if symbol:
            arr = self._ict_symbol_outcomes.setdefault(symbol, [])
            arr.append({
                'outcome': outcome_type,
                'realized_r': float(realized_r or 0.0),
                'side': side,
                'ts': now,
            })
            if len(arr) > 80:
                del arr[:-80]

    # ══════════════════════════════════════════════════════════════════════
    # SCAN — v2.2: symbol'ü signal fonksiyonuna geçir (diagnostic log için)
    # ══════════════════════════════════════════════════════════════════════
    def _ict_smc_scan(self, fetcher: DataFetcher):
        if not self._check_loss_limits():
            return
        if not self._ict_check_cooldown():
            remaining = max(0, self._ict_cooldown_until - time.time())
            if remaining > 0:
                self.log(f"[COOL] ICT cooldown: {int(remaining/60)}m kaldı")
            return

        with self.trades_lock:
            active_symbols = set(t['symbol'] for t in self.open_trades)
            pending_symbols = set(p['symbol'] for p in self.pending_orders)
            active_count = len(self.open_trades) + len(self.pending_orders)
            can_open = active_count < self.max_open_trades_limit

        if not can_open:
            return

        # ── Rotating Scan — dinamik grup × 50 coin ──
        self._ict_init_state()
        if not hasattr(self, '_ict_scan_group'):
            self._ict_scan_group = 0

        chunk_size = 50
        group_count = max(1, (len(self.scanned_symbols) + chunk_size - 1) // chunk_size)
        if self._ict_scan_group >= group_count:
            self._ict_scan_group = 0

        group = self._ict_scan_group
        start_idx = group * chunk_size
        end_idx = min(start_idx + chunk_size, len(self.scanned_symbols))
        scan_list = self.scanned_symbols[start_idx:end_idx]

        # Sonraki scan için grup değiştir
        self._ict_scan_group = (group + 1) % group_count

        self.log(f"[SCAN] ICT Grup {group+1}/{group_count}: {len(scan_list)} coin ({start_idx}-{max(start_idx, end_idx-1)})")

        # ── AŞAMA 1 ──
        pre_candidates = []
        skipped_pre = 0

        for symbol in scan_list:
            if not self.is_running:
                break
            try:
                if symbol in active_symbols or symbol in pending_symbols:
                    continue
                sym_cd = self._ict_symbol_cooldown.get(symbol, 0)
                if time.time() < sym_cd:
                    continue

                # Sembol bazlı SL streak cezası: 3+ streak varsa taramayı geçici atla
                if int(self._ict_symbol_sl_streak.get(symbol, 0)) >= 3:
                    continue

                health = self._ict_symbol_health(
                    symbol,
                    window=int(self._ICT_PARAMS.get('symbol_health_window', 30))
                )
                if health and health['count'] >= int(self._ICT_PARAMS.get('symbol_health_min_count', 14)):
                    low_wr = health['wr'] < float(self._ICT_PARAMS.get('symbol_health_wr_floor', 0.30))
                    low_pf = health['pf'] < float(self._ICT_PARAMS.get('symbol_health_pf_floor', 0.90))
                    if low_wr and low_pf:
                        penalty_sec = int(self._ICT_PARAMS.get('symbol_health_penalty_sec', 24 * 3600))
                        self._ict_symbol_cooldown[symbol] = max(
                            self._ict_symbol_cooldown.get(symbol, 0),
                            time.time() + penalty_sec,
                        )
                        self.log(
                            f"[COOL] ICT weak symbol {symbol}: WR={health['wr']:.1%} "
                            f"PF={health['pf']:.2f} → {int(penalty_sec/3600)}h pasif"
                        )
                        continue

                if not self._check_correlation(symbol):
                    continue

                df_1h = fetcher.fetch_ohlcv(symbol, '1h', limit=100)
                if df_1h is None or df_1h.empty or len(df_1h) < 55:
                    continue
                df_1h = add_all_indicators(df_1h)

                quick = self._ict_quick_filter(df_1h)
                if quick is None:
                    skipped_pre += 1
                    continue

                pre_candidates.append({
                    'symbol': symbol,
                    'df_1h': df_1h,
                    'ict_1h': quick.get('ict_1h'),
                })
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str:
                    time.sleep(1.0)
                else:
                    self.log(f"[ERR] ICT scan-1 {symbol}: {e}")
                    time.sleep(0.3)

        if not pre_candidates:
            self.log(f"[PERF] ICT Aşama-1: 0 aday, {skipped_pre} elendi")
            return

        self.log(f"[PERF] ICT Aşama-1: {len(pre_candidates)} aday (/{len(scan_list)}, {skipped_pre} elendi)")

        # ── AŞAMA 2: Detaylı (diagnostic loglarla) ──
        candidates = []

        for item in pre_candidates:
            if not self.is_running:
                break
            symbol = item['symbol']
            df_1h = item['df_1h']
            cached_ict_1h = item.get('ict_1h')

            try:
                df_4h = None
                try:
                    df_4h = fetcher.fetch_ohlcv(symbol, '4h', limit=100)
                    if df_4h is not None and not df_4h.empty:
                        df_4h = add_all_indicators(df_4h)
                    else:
                        df_4h = None
                except Exception:
                    df_4h = None

                entry_trigger_tf = self._ICT_PARAMS.get('entry_trigger_tf', '15m')
                df_trigger = None
                try:
                    df_trigger = fetcher.fetch_ohlcv(symbol, entry_trigger_tf, limit=80)
                    if df_trigger is not None and not df_trigger.empty:
                        df_trigger = add_all_indicators(df_trigger)
                    else:
                        df_trigger = None
                except Exception:
                    df_trigger = None

                # ★ v2.2: symbol geçiriyoruz → diagnostic log
                sig = self._ict_smc_signal(
                    df_1h, df_4h, df_trigger,
                    cached_ict_1h=cached_ict_1h,
                    symbol=symbol
                )
                if sig is None:
                    continue

                sig['symbol'] = symbol
                sig['close'] = float(df_1h['close'].iloc[-1])
                candidates.append(sig)

            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str:
                    time.sleep(1.0)
                else:
                    self.log(f"[ERR] ICT scan-2 {symbol}: {e}")
                    time.sleep(0.3)

        if not candidates:
            self.log(f"[PERF] ICT Aşama-2: 0 sinyal ({len(pre_candidates)} aday — detay yukarıda)")
            return

        candidates.sort(key=lambda x: x['quality_score'], reverse=True)

        top_n = min(len(candidates), 3)
        self.log(f"[RANK] ICT {len(candidates)} sinyal:")
        for j, c in enumerate(candidates[:top_n]):
            marker = ">>>" if j == 0 else "   "
            self.log(
                f"  {marker} #{j+1} {c['symbol']} {c['direction']} | "
                f"Q:{c['quality_score']} RR:{c['rr_ratio']} "
                f"Gate:{'✓' if c.get('gate_passed') else '✗'}"
            )

        best = candidates[0]
        symbol = best['symbol']
        direction = best['direction']
        cp = best['close']
        atr_val = best['atr']

        ict_max_notional = self._ICT_PARAMS.get('max_notional', 200.0)
        # v2.4: balance*0.02 çok kısıtlayıcıydı ($198). Kullanıcı $1000 ayarladığında
        # $19 margin açılıyordu. Şimdi %15 cap (leverage 10x → %1.5 margin cap).
        # Risk max_loss_cap ile kontrol ediliyor.
        max_notional = min(self.balance * 0.15, ict_max_notional)
        qty = max_notional / cp
        sl_dist = abs(cp - best['sl_price'])
        ict_max_loss = self._ICT_PARAMS.get('max_loss_cap', 8.0)
        if sl_dist > 0:
            max_qty_by_risk = ict_max_loss / sl_dist
            qty = min(qty, max_qty_by_risk)

        if best.get('counter_trend_4h'):
            risk_mult = float(self._ICT_PARAMS.get('counter_trend_risk_mult', 0.70))
            qty *= max(0.25, min(1.0, risk_mult))

        trend_strength = float(best.get('trend_strength', 0.5) or 0.5)
        weak_trend_th = float(self._ICT_PARAMS.get('weak_trend_risk_threshold', 0.45))
        if trend_strength < weak_trend_th:
            weak_mult = float(self._ICT_PARAMS.get('weak_trend_risk_mult', 0.75))
            qty *= max(0.35, min(1.0, weak_mult))

        logger_id = f"ICT_{symbol}_{direction}_{int(time.time())}"

        with self.trades_lock:
            tid = self._open_locked(
                symbol=symbol,
                side=direction,
                price=cp,
                multiplier=1.0,
                tp_price=best['tp_price'],
                sl_price=best['sl_price'],
                signal_result={
                    'strategy': 'ict_smc_v3',
                    'regime': best['htf_trend'],
                    'entry_type': 'market',
                    'soft_score': min(best['quality_score'], 5),
                    'signal': direction,
                    'quality_score': best['quality_score'],
                    'setup_type': best.get('setup_type', 'ict_confluence'),
                    'gate_passed': best.get('gate_passed', False),
                    'gate_score': best.get('gate_score', 0),
                    'killzone': best.get('killzone', ''),
                    'confluence': best.get('confluence', 0),
                    'has_sweep': best.get('has_sweep', False),
                    'has_displacement': best.get('has_displacement', False),
                    'vol_spike': best.get('vol_spike', False),
                    'in_ote': best.get('in_ote', False),
                    'rr_ratio': best.get('rr_ratio', 0),
                    'trend_strength': best.get('trend_strength', 0),
                    'counter_trend_4h': best.get('counter_trend_4h', False),
                    'sl_atr_mult': best.get('sl_atr_mult', 0),
                },
                absolute_qty=qty,
                atr=atr_val,
                logger_id=logger_id,
            )
            if tid:
                self._ict_last_trade_time = time.time()
                self.log(
                    f"[OPEN] ICT {symbol} {direction} Q={best['quality_score']} "
                    f"RR={best['rr_ratio']} Gate={'✓' if best.get('gate_passed') else '✗'}"
                )

    # ══════════════════════════════════════════════════════════════════════
    # EXIT LOGIC
    # ══════════════════════════════════════════════════════════════════════
    def _check_ict_exit(self, t, current_price):
        strat = t.get('strategy', '') or t.get('signal_result', {}).get('strategy', '')
        if strat not in ('ict_smc_v1', 'ict_smc_v2', 'ict_smc_v3'):
            return

        side = t.get('side', '')
        entry = t.get('entry_price', 0)
        tp = t.get('tp_price', 0)
        sl = t.get('sl_price', 0)
        symbol = t.get('symbol', '')

        if tp <= 0 or sl <= 0 or entry <= 0:
            return

        pnl_dollar, pnl_pct = self._compute_pnl(t, current_price)
        initial_r = abs(entry - sl) if sl > 0 else 0
        ict_max_loss = self._ICT_PARAMS.get('max_loss_cap', 8.0)

        # Core V2 hard-exit: 4h yapı tersine döndüyse pozisyonu kapat
        if strat == 'ict_smc_v3':
            hard_exit_tf = self._ICT_PARAMS.get('hard_exit_tf', '4h')
            if hard_exit_tf:
                last_hard_check = t.get('_ict_last_hard_check', 0)
                if time.time() - last_hard_check > 900:
                    t['_ict_last_hard_check'] = time.time()
                    try:
                        from ai import ict_core as _ic
                        _f = self._get_fetcher()
                        _df_hard = _f.fetch_ohlcv(symbol, hard_exit_tf, limit=80)
                        if _df_hard is not None and len(_df_hard) >= 30:
                            _df_hard = add_all_indicators(_df_hard)
                            sw_left, sw_right = self._get_swing_params(hard_exit_tf)
                            _ict_hard = _ic.analyze(_df_hard, side, swing_left=sw_left, swing_right=sw_right)
                            ms = (_ict_hard.market_structure or '').lower()
                            hard_broken = (side == 'LONG' and ms == 'bearish') or (side == 'SHORT' and ms == 'bullish')
                            if hard_broken:
                                r_val = self._calc_realized_r(side, entry, current_price, initial_r)
                                self._close_all_locked([t], symbol, current_price, "ICT_4H_HARD_EXIT")
                                outcome = 'TP' if pnl_dollar > 0 else 'SL'
                                self._ict_record_outcome(outcome, symbol, r_val, side)
                                self.log(f"[HARD] ICT {symbol}: {hard_exit_tf} structure flipped → ${pnl_dollar:+.2f}")
                                return
                    except Exception:
                        pass

        if pnl_dollar < -ict_max_loss:
            r_val = self._calc_realized_r(side, entry, current_price, initial_r)
            self._close_all_locked([t], symbol, current_price, "ICT_MAXLOSS")
            self._ict_record_outcome('SL', symbol, r_val, side)
            self.log(f"[CAP] ICT {symbol} max loss: ${pnl_dollar:.2f}")
            return

        # Erken invalidation: yapı hızlı bozulduysa tam SL bekleme
        if strat == 'ict_smc_v3' and not t.get('_ict_tp1_hit') and initial_r > 0:
            last_inv_check = t.get('_ict_last_inv_check', 0)
            if time.time() - last_inv_check > 300:
                t['_ict_last_inv_check'] = time.time()
                try:
                    entry_ts = float(t.get('entry_timestamp', 0) or 0)
                    age_min = (time.time() - entry_ts) / 60 if entry_ts > 0 else 0
                    min_age = float(self._ICT_PARAMS.get('invalidation_min_age_min', 90))
                    max_age = float(self._ICT_PARAMS.get('invalidation_max_age_min', 480))
                    r_now = self._calc_realized_r(side, entry, current_price, initial_r)
                    inv_r = -abs(float(self._ICT_PARAMS.get('invalidation_r_threshold', 0.45)))

                    if age_min >= min_age and age_min <= max_age and r_now <= inv_r:
                        _f = self._get_fetcher()
                        tf = self._ICT_PARAMS.get('entry_trigger_tf', '15m')
                        _df_fast = _f.fetch_ohlcv(symbol, tf, limit=70)
                        if _df_fast is not None and len(_df_fast) >= 30:
                            _df_fast = add_all_indicators(_df_fast)
                            from ai import ict_core as _ic
                            sw_l, sw_r = self._get_swing_params(tf)
                            _ict_fast = _ic.analyze(_df_fast, side, swing_left=sw_l, swing_right=sw_r)
                            ms_fast = (_ict_fast.market_structure or '').lower()
                            bad_choch = bool(
                                _ict_fast.last_choch and
                                ((side == 'LONG' and _ict_fast.last_choch.direction == 'bearish') or
                                 (side == 'SHORT' and _ict_fast.last_choch.direction == 'bullish'))
                            )
                            bad_ms = (side == 'LONG' and ms_fast == 'bearish') or (side == 'SHORT' and ms_fast == 'bullish')
                            if bad_choch or bad_ms:
                                self._close_all_locked([t], symbol, current_price, "ICT_EARLY_INVALIDATION")
                                self._ict_record_outcome('SL', symbol, r_now, side)
                                self.log(
                                    f"[INVAL] ICT {symbol}: early invalidation "
                                    f"(r={r_now:.2f}, age={age_min:.0f}m, ms={ms_fast}, choch={bad_choch})"
                                )
                                return
                except Exception:
                    pass

        # ── v2.5: EMERGENCY ROI-based protection (failsafe) ──
        # If profit > 5% ROI, protect at least breakeven
        if pnl_pct >= 5.0 and not t.get('_ict_emergency_be_set'):
            emergency_sl = entry * (1.001 if side == 'LONG' else 0.999)
            if (side == 'LONG' and emergency_sl > sl) or (side == 'SHORT' and emergency_sl < sl):
                t['sl_price'] = emergency_sl
                sl = emergency_sl
                t['_ict_emergency_be_set'] = True
                self.log(f"[EMERGENCY-BE] ICT {symbol}: SL→{emergency_sl:.6f} @ +{pnl_pct:.1f}% ROI")
                self._force_save()
        
        # --- Advanced Tiered ROI Profit Protection ---
        # Continuously ratchet SL based on peak price instead of single 8% lock
        peak_price = max(current_price, t.get('_ict_peak_price', entry)) if side == 'LONG' else min(current_price, t.get('_ict_peak_price', entry))
        t['_ict_peak_price'] = peak_price
        peak_pnl_pct = abs(peak_price - entry) / entry
        
        keep_ratio = 0.0
        if peak_pnl_pct >= 0.12:    # +12% price move -> lock 90%
            keep_ratio = 0.90
        elif peak_pnl_pct >= 0.08:  # +8% price move -> lock 85%
            keep_ratio = 0.85
        elif peak_pnl_pct >= 0.05:  # +5% price move -> lock 70%
            keep_ratio = 0.70
        elif peak_pnl_pct >= 0.025: # +2.5% price move -> lock 50%
            keep_ratio = 0.40
        elif peak_pnl_pct >= 0.012: # +1.2% price move -> lock 25%
            keep_ratio = 0.25
        
        if keep_ratio > 0.0:
            protected_dist = abs(peak_price - entry) * keep_ratio
            protected_price = entry + protected_dist if side == 'LONG' else entry - protected_dist
            
            if (side == 'LONG' and protected_price > sl) or (side == 'SHORT' and protected_price < sl):
                t['sl_price'] = protected_price
                sl = protected_price
                t['_ict_emergency_trail_set'] = True
                self.log(f"[ROI-TRAIL] ICT {symbol}: Peak {peak_pnl_pct*100:.1f}%, SL→{protected_price:.6f} (Locked {keep_ratio*100:.0f}%)")
                self._force_save()
        
        # ── v2.4: ATR-based Breakeven + Trailing (PM yoksa da koruma) ──
        atr_val = t.get('atr', entry * 0.005)
        if atr_val <= 0:
            atr_val = entry * 0.005
        
        # DEBUG: Log profit protection check every 10% profit milestone
        if pnl_pct > 0 and int(pnl_pct / 10) > t.get('_last_debug_milestone', -1):
            t['_last_debug_milestone'] = int(pnl_pct / 10)
            self.log(
                f"[DEBUG] ICT {symbol} @ +{pnl_pct:.1f}%: "
                f"ATR=${atr_val:.6f} ({atr_val/entry*100:.2f}%), "
                f"Peak={t.get('_ict_peak_price', 'NOT_SET')}, "
                f"BE_set={t.get('_ict_be_set', False)}, "
                f"Emergency_BE={t.get('_ict_emergency_be_set', False)}, "
                f"Emergency_Trail={t.get('_ict_emergency_trail_set', False)}"
            )

        if side == 'LONG':
            price_profit_atr = (current_price - entry) / atr_val
            peak_price = max(current_price, t.get('_ict_peak_price', entry))
            t['_ict_peak_price'] = peak_price
            peak_profit_atr = (peak_price - entry) / atr_val

            # Breakeven: 1.5 ATR kârda → SL = entry + 0.3 ATR
            if peak_profit_atr >= 1.5 and not t.get('_ict_be_set'):
                be_sl = entry + 0.3 * atr_val
                if be_sl > sl:
                    t['sl_price'] = be_sl
                    sl = be_sl
                    t['_ict_be_set'] = True
                    self.log(f"[BE] ICT {symbol}: SL→BE {be_sl:.6f} (peak {peak_profit_atr:.1f} ATR)")
                    self._force_save()
                else:
                    self.log(f"[BE-SKIP] ICT {symbol}: be_sl {be_sl:.6f} <= current_sl {sl:.6f}")

            # Trailing: 3.0 ATR kârda → SL = peak - 1.5 ATR
            if peak_profit_atr >= 3.0:
                trail_sl = peak_price - 1.5 * atr_val
                if trail_sl > sl:
                    t['sl_price'] = trail_sl
                    sl = trail_sl
                    self.log(f"[TRAIL] ICT {symbol}: SL→{trail_sl:.6f} (peak {peak_price:.6f})")
                else:
                    # Only log once per trade to avoid spam
                    if not t.get('_trail_skip_logged'):
                        t['_trail_skip_logged'] = True
                        self.log(f"[TRAIL-SKIP] ICT {symbol}: trail_sl {trail_sl:.6f} <= current_sl {sl:.6f}")

        else:  # SHORT
            price_profit_atr = (entry - current_price) / atr_val
            peak_price = min(current_price, t.get('_ict_peak_price', entry))
            t['_ict_peak_price'] = peak_price
            peak_profit_atr = (entry - peak_price) / atr_val

            peak_pct = abs(t['_ict_peak_price'] - entry) / entry
            
            # ── Tier 1: Breakeven ──
            be_thresh = 0.015
            if peak_pct >= be_thresh and not t.get('_ict_be_active'):
                t['_ict_be_active'] = True
                if side == 'LONG':
                    be_sl = max(sl, entry * 1.002)
                    t['sl_price'] = be_sl
                    sl = be_sl
                else:
                    be_sl = min(sl, entry * 0.998)
                    t['sl_price'] = be_sl
                    sl = be_sl
                self.log(f"[BE] ICT {symbol}: SL→BE {be_sl:.6f} (peak +{peak_pct*100:.1f}%)")
                self._force_save()

            # ── Tier 2: Dynamic Trailing Stop ──
            keep_ratio = 0.0
            if peak_pct >= 0.015:
                keep_ratio = 0.40
                if peak_pct >= 0.050: keep_ratio = 0.85
                elif peak_pct >= 0.035: keep_ratio = 0.70
                elif peak_pct >= 0.025: keep_ratio = 0.60
                    
            if keep_ratio > 0.0:
                keep_dist = abs(t['_ict_peak_price'] - entry) * keep_ratio
                new_sl = entry + keep_dist if side == 'LONG' else entry - keep_dist
                
                if (side == 'LONG' and new_sl > sl) or (side == 'SHORT' and new_sl < sl):
                    t['sl_price'] = new_sl
                    sl = new_sl
                    self.log(f"[TRAIL] ICT {symbol}: SL→{new_sl:.6f} (lock {keep_ratio*100:.0f}%)")
                    self._force_save()

        hit_sl = (side == 'LONG' and current_price <= sl) or \
                 (side == 'SHORT' and current_price >= sl)
        if hit_sl:
            reason = "ICT_SL" if pnl_dollar < 0 else "ICT_TRAIL_SL"
            r_val = self._calc_realized_r(side, entry, current_price, initial_r)
            self._close_all_locked([t], symbol, current_price, reason)
            if pnl_dollar < 0:
                self._ict_record_outcome('SL', symbol, r_val, side)
            else:
                self._ict_record_outcome('TP', symbol, r_val, side)
            self.log(f"[STOP] ICT {reason}: {symbol} ${pnl_dollar:+.2f}")
            return

        if t.get('_ict_tp1_hit') and pnl_dollar > 0:
            try:
                self._update_structural_sl(t, current_price, fetcher=self._get_fetcher())
            except Exception:
                pass

        dynamic_partials = bool(self._ICT_PARAMS.get('use_dynamic_partials', False)) or strat == 'ict_smc_v3'

        tp_dist = abs(tp - entry)
        if tp_dist > 0 and not dynamic_partials:
            if side == 'LONG':
                progress = (current_price - entry) / tp_dist if current_price > entry else 0
            else:
                progress = (entry - current_price) / tp_dist if current_price < entry else 0

            original_qty = t.get('original_qty', t['qty'])
            if 'original_qty' not in t:
                t['original_qty'] = t['qty']

            if progress >= 0.33 and not t.get('_ict_tp1_hit'):
                close_pct = 0.30
                remaining_qty = t['qty']
                close_qty = original_qty * close_pct
                close_qty = min(close_qty, remaining_qty * 0.95)

                if close_qty > 0 and remaining_qty > close_qty:
                    partial_pnl = self._calc_partial_pnl(t, current_price, close_qty)
                    margin_released = (close_qty * t['entry_price']) / self.leverage
                    t['margin'] = max(0.0, t.get('margin', 0.0) - margin_released)
                    self.balance += margin_released + partial_pnl
                    t['qty'] -= close_qty
                    t['_ict_tp1_hit'] = True
                    t['partial_profit'] = t.get('partial_profit', 0) + partial_pnl

                    buffer = entry * 0.001
                    if side == 'LONG':
                        t['sl_price'] = max(t['sl_price'], entry + buffer)
                    else:
                        t['sl_price'] = min(t['sl_price'], entry - buffer)

                    self.log(f"[TP1] ICT {symbol}: %30 kapatıldı (${partial_pnl:+.2f}), SL→BE")

                    if t['qty'] * current_price < 5.0:
                        r_val = self._calc_realized_r(side, entry, current_price, initial_r)
                        self._close_all_locked([t], symbol, current_price, "ICT_DUST_CLOSE")
                        self._ict_record_outcome('TP', symbol, r_val, side)
                        return
                    self._force_save()
                    return

            if progress >= 0.65 and not t.get('_ict_tp2_hit'):
                close_pct = 0.40
                remaining_qty = t['qty']
                close_qty = original_qty * close_pct
                close_qty = min(close_qty, remaining_qty * 0.95)

                if close_qty > 0 and remaining_qty > close_qty:
                    partial_pnl = self._calc_partial_pnl(t, current_price, close_qty)
                    margin_released = (close_qty * t['entry_price']) / self.leverage
                    t['margin'] = max(0.0, t.get('margin', 0.0) - margin_released)
                    self.balance += margin_released + partial_pnl
                    t['qty'] -= close_qty
                    t['_ict_tp2_hit'] = True
                    t['partial_profit'] = t.get('partial_profit', 0) + partial_pnl

                    if side == 'LONG':
                        tp1_sl = entry + tp_dist * 0.33
                        t['sl_price'] = max(t['sl_price'], tp1_sl)
                    else:
                        tp1_sl = entry - tp_dist * 0.33
                        t['sl_price'] = min(t['sl_price'], tp1_sl)

                    self.log(f"[TP2] ICT {symbol}: %40 kapatıldı (${partial_pnl:+.2f}), SL→TP1")

                    if t['qty'] * current_price < 5.0:
                        r_val = self._calc_realized_r(side, entry, current_price, initial_r)
                        self._close_all_locked([t], symbol, current_price, "ICT_DUST_CLOSE")
                        self._ict_record_outcome('TP', symbol, r_val, side)
                        return
                    self._force_save()
                    return

            if progress >= 1.0 and not t.get('_ict_tp3_hit'):
                t['_ict_tp3_hit'] = True
                t['_ict_trail_mode'] = True
                t['_ict_tp3_time'] = time.time()

                if side == 'LONG':
                    tp3_sl = entry + tp_dist * 0.75
                    t['sl_price'] = max(t['sl_price'], tp3_sl)
                else:
                    tp3_sl = entry - tp_dist * 0.75
                    t['sl_price'] = min(t['sl_price'], tp3_sl)

                self.log(f"[TRAIL] ICT {symbol}: TP3→TRAIL, SL→{t['sl_price']:.4f}")
                self._force_save()
                return

        if dynamic_partials and tp_dist > 0 and not t.get('_ict_dynamic_lock'):
            if side == 'LONG':
                tp_progress = (current_price - entry) / tp_dist if current_price > entry else 0
            else:
                tp_progress = (entry - current_price) / tp_dist if current_price < entry else 0

            if tp_progress >= 0.90:
                close_qty = t['qty'] * 0.20
                if close_qty > 0 and t['qty'] > close_qty:
                    partial_pnl = self._calc_partial_pnl(t, current_price, close_qty)
                    margin_released = (close_qty * t['entry_price']) / self.leverage
                    t['margin'] = max(0.0, t.get('margin', 0.0) - margin_released)
                    self.balance += margin_released + partial_pnl
                    t['qty'] -= close_qty
                    t['partial_profit'] = t.get('partial_profit', 0) + partial_pnl
                    t['_ict_dynamic_lock'] = True
                    self.log(f"[DYN] ICT {symbol}: near-liquidity partial ${partial_pnl:+.2f}, runner active")
                    self._force_save()
                    return

        if t.get('_ict_trail_mode'):
            trail_start = t.get('_ict_tp3_time', time.time())
            if time.time() - trail_start > 24 * 3600:
                r_val = self._calc_realized_r(side, entry, current_price, initial_r)
                self._close_all_locked([t], symbol, current_price, "ICT_TRAIL_TIMEOUT")
                self._ict_record_outcome('TP', symbol, r_val, side)
                self.log(f"[TRAIL] ICT {symbol}: timeout → ${pnl_dollar:+.2f}")
                return

            try:
                from ai import ict_core as _ic
                last_choch_check = t.get('_last_choch_check', 0)
                if time.time() - last_choch_check > 300:
                    t['_last_choch_check'] = time.time()
                    self._ict_init_state()
                    cache_key = f'{symbol}_ts'
                    if time.time() - self._trail_df_cache.get(cache_key, 0) > 300:
                        try:
                            _f = self._get_fetcher()
                            _df = _f.fetch_ohlcv(symbol, '1h', limit=30)
                            if _df is not None and len(_df) >= 10:
                                self._trail_df_cache[cache_key] = time.time()
                                sw_left, sw_right = self._get_swing_params('1h')
                                _ict = _ic.analyze(_df, side, swing_left=sw_left, swing_right=sw_right)
                                bad_dir = 'bearish' if side == 'LONG' else 'bullish'
                                if _ict.last_choch and _ict.last_choch.direction == bad_dir:
                                    r_val = self._calc_realized_r(side, entry, current_price, initial_r)
                                    self._close_all_locked([t], symbol, current_price, "ICT_CHOCH_EXIT")
                                    self._ict_record_outcome('TP', symbol, r_val, side)
                                    self.log(f"[CHOCH] ICT {symbol}: yapı bozuldu → ${pnl_dollar:+.2f}")
                                    return
                        except Exception:
                            pass
            except ImportError:
                pass
            return

        entry_ts = t.get('entry_timestamp', 0)
        max_timeout = 72 * 3600 if t.get('_ict_tp3_hit') else 48 * 3600
        if entry_ts > 0 and time.time() - entry_ts > max_timeout:
            reason = "ICT_TRAIL_TIMEOUT" if t.get('_ict_tp3_hit') else "ICT_TIMEOUT"
            r_val = self._calc_realized_r(side, entry, current_price, initial_r)
            self._close_all_locked([t], symbol, current_price, reason)
            outcome = 'TP' if pnl_dollar > 0 else 'SL'
            self._ict_record_outcome(outcome, symbol, r_val, side)
            self.log(f"[TIME] ICT timeout: {symbol} ${pnl_dollar:+.2f}")
            return