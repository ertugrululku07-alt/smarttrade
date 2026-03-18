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
        if not hasattr(self, '_trail_df_cache'):
            self._trail_df_cache = {}

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

        ict_1h = ict_core.analyze(df_1h, '', swing_left=3, swing_right=2)

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
        if df_4h is not None and len(df_4h) >= 30:
            ict_htf = ict_core.analyze(df_4h, '', swing_left=3, swing_right=2)

        ict_1h = cached_ict_1h if cached_ict_1h is not None else \
            ict_core.analyze(df_1h, '', swing_left=3, swing_right=2)

        if ict_htf is None:
            ict_htf = ict_1h

        htf_ms = ict_htf.market_structure
        if htf_ms == 'ranging':
            self.log(f"  [DIAG] {symbol}: REJECT → HTF ranging")
            return None

        direction = 'LONG' if htf_ms == 'bullish' else 'SHORT'
        ict_1h_dir = ict_core.analyze(df_1h, direction, swing_left=3, swing_right=2)

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

        # ── OB/FVG ──
        ob_type = 'bullish' if direction == 'LONG' else 'bearish'
        active_obs = [ob for ob in ict_1h_dir.order_blocks if not ob.mitigated and ob.type == ob_type]
        active_fvgs = [f for f in ict_1h_dir.fvg_zones if not f.filled and f.type == ob_type]

        if not active_obs and not active_fvgs:
            self.log(f"  [DIAG] {symbol}: REJECT → No active OB/FVG (dir re-analysis)")
            return None

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

        # ── v2.3: RSI Overbought/Oversold filtresi ──
        rsi_val = None
        if 'rsi' in df_1h.columns:
            rsi_val = float(df_1h['rsi'].iloc[-1])
            if not np.isnan(rsi_val):
                if direction == 'LONG' and rsi_val > 70:
                    self.log(f"  [DIAG] {symbol}: REJECT → RSI {rsi_val:.1f} > 70 (overbought, LONG yasak)")
                    return None
                if direction == 'SHORT' and rsi_val < 30:
                    self.log(f"  [DIAG] {symbol}: REJECT → RSI {rsi_val:.1f} < 30 (oversold, SHORT yasak)")
                    return None

        # ── v2.3: Premium/Discount Zone kontrolü ──
        # Son 30 bar'ın gerçek high/low'u kullanılır (eski swing low'lar trending
        # market'te range'ı bozar — her yer premium görünür).
        zone_lookback = min(30, len(df_1h))
        recent_slice = df_1h.iloc[-zone_lookback:]
        range_high = float(recent_slice['high'].max())
        range_low = float(recent_slice['low'].min())
        if range_high > range_low:
            price_position = (cp - range_low) / (range_high - range_low)
            if direction == 'LONG' and price_position > 0.85:
                self.log(f"  [DIAG] {symbol}: REJECT → LONG premium zone (pos={price_position:.0%})")
                return None
            if direction == 'SHORT' and price_position < 0.15:
                self.log(f"  [DIAG] {symbol}: REJECT → SHORT discount zone (pos={price_position:.0%})")
                return None

        # ── POI Confluence ──
        cc = ict_1h_dir.poi_details.get('confluence_count', 0)

        # ── v2.3: POI Proximity → HARD filtre (3 ATR max) ──
        poi_ok, poi_dist, poi_reason = self._check_poi_proximity(
            direction, cp, ict_1h_dir, atr_val)
        if not poi_ok:
            self.log(f"  [DIAG] {symbol}: REJECT → POI too far ({poi_dist/atr_val:.1f} ATR)")
            return None

        # ── Trend Strength (çok düşük eşik) ──
        trend_str, trend_reason = self._check_trend_strength(ict_htf, direction)
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

        # ── SL hesaplama ──
        sweep_sl = ict_core.get_sweep_sl(
            direction, ict_1h_dir.sweep_level, cp, atr_val,
            ict_1h_dir.swing_highs, ict_1h_dir.swing_lows
        )

        max_sl_pct = params.get('max_sl_pct', 3.0) / 100  # v2.2: 2.5% → 3.0%
        if direction == 'LONG':
            sl_price = min(sweep_sl, cp * (1 - max_sl_pct))
            if sl_price >= cp:
                sl_price = cp * (1 - max_sl_pct)
        else:
            sl_price = max(sweep_sl, cp * (1 + max_sl_pct))
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

        # Temel yapısal puanlar
        if has_sweep: quality += 2
        if has_disp: quality += 2
        if has_bos: quality += 2
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
        if vol_spike: quality += 1
        if rr >= 3.0: quality += 1
        if has_rejection: quality += 2

        # Gate bonus
        if gate_passed:
            quality += 2
            quality += int(gate_score * 2)
        elif gate_score >= 0.5:
            quality += 1

        # Trend bonus/penalty
        if trend_str >= 0.7: quality += 1
        elif trend_str < 0.30: quality -= 1

        # ══════════════════════════════════════════════════════════
        min_quality = 6  # v2.4: 8 → 6 (tested: more trades, higher WR, better profit)

        if quality < min_quality:
            self.log(
                f"  [DIAG] {symbol}: REJECT → Quality {quality} < {min_quality} | "
                f"sweep={has_sweep} disp={has_disp} bos={has_bos} choch={has_choch} "
                f"obs={len(active_obs)} fvgs={len(active_fvgs)} cc={cc} "
                f"ote={in_ote} kz={killzone_name} poi={poi_reason} "
                f"gate={'✓' if gate_passed else '✗'}({gate_score:.0%}) "
                f"rej={has_rejection} vol={vol_spike} rr={rr:.1f}"
            )
            return None

        # ═══ PASSED! Log detay ═══
        self.log(
            f"  [PASS] {symbol} {direction}: Q={quality} RR={rr:.1f} | "
            f"sweep={has_sweep} disp={has_disp} bos={has_bos} choch={has_choch} "
            f"obs={len(active_obs)} fvgs={len(active_fvgs)} cc={cc} "
            f"ote={in_ote} kz={killzone_name} poi={poi_reason} "
            f"gate={'✓' if gate_passed else '✗'}({gate_score:.0%})"
        )

        return {
            'direction': direction,
            'tp_price': tp_price,
            'sl_price': sl_price,
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
            'poi_dist_atr': round(poi_dist / atr_val, 2) if atr_val > 0 else 0,
            'trend_strength': round(trend_str, 2),
            'trend_detail': trend_reason,
            'has_rejection': has_rejection,
            'rejection_detail': rejection_reason,
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
    def _check_trend_strength(self, ict_htf, direction):
        labels = [lb.label for lb in ict_htf.structure_labels[-8:]]
        if not labels:
            return 0.5, 'no_labels'

        if direction == 'LONG':
            trend_labels = sum(1 for lb in labels if lb in ('HH', 'HL'))
        else:
            trend_labels = sum(1 for lb in labels if lb in ('LL', 'LH'))

        total = len(labels)
        if total == 0:
            return 0.5, 'empty'

        strength = trend_labels / total
        bos_count = sum(1 for sb in ict_htf.structure_breaks
                        if sb.type == 'BOS' and
                        sb.direction == ('bullish' if direction == 'LONG' else 'bearish'))

        if bos_count >= 5:
            return max(0.0, strength - 0.2), f'exhausted(bos={bos_count})'
        has_choch = ict_htf.last_choch and \
            ict_htf.last_choch.direction == ('bullish' if direction == 'LONG' else 'bearish')
        if has_choch and bos_count <= 2:
            return min(1.0, strength + 0.2), f'fresh(choch+bos={bos_count})'

        return strength, f'normal(str={strength:.0%},bos={bos_count})'

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
            ict = ict_core.analyze(df_1h, side, swing_left=3, swing_right=2)
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
                            realized_r: float = 0.0):
        self._ict_init_state()
        now = time.time()
        if outcome_type == 'SL':
            self._ict_consecutive_sl += 1
            if self._ict_consecutive_sl >= 3:
                self._ict_cooldown_until = now + 1800  # 30 dk
                self.log(f"[COOL] ICT 3 ardışık SL → 30 dk mola")
                self._ict_consecutive_sl = 0
            if symbol:
                self._ict_symbol_cooldown[symbol] = now + 2 * 3600
        elif outcome_type == 'TP':
            self._ict_consecutive_sl = 0

        self._ict_recent_outcomes.append(outcome_type)
        if len(self._ict_recent_outcomes) > 10:
            self._ict_recent_outcomes.pop(0)

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

        # ── v2.5: Rotating Scan — 3 grup × 50 coin (toplam 150) ──
        self._ict_init_state()
        if not hasattr(self, '_ict_scan_group'):
            self._ict_scan_group = 0
        
        group = self._ict_scan_group
        start_idx = group * 50
        end_idx = min(start_idx + 50, len(self.scanned_symbols))
        scan_list = self.scanned_symbols[start_idx:end_idx]
        
        # Sonraki scan için grup değiştir (0→1→2→0)
        self._ict_scan_group = (group + 1) % 3
        
        self.log(f"[SCAN] ICT Grup {group+1}/3: {len(scan_list)} coin ({start_idx}-{end_idx-1})")

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

                df_5m = None
                try:
                    df_5m = fetcher.fetch_ohlcv(symbol, '5m', limit=60)
                    if df_5m is not None and not df_5m.empty:
                        df_5m = add_all_indicators(df_5m)
                    else:
                        df_5m = None
                except Exception:
                    df_5m = None

                # ★ v2.2: symbol geçiriyoruz → diagnostic log
                sig = self._ict_smc_signal(
                    df_1h, df_4h, df_5m,
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
                    'strategy': 'ict_smc_v2',
                    'regime': best['htf_trend'],
                    'entry_type': 'market',
                    'soft_score': min(best['quality_score'], 5),
                    'signal': direction,
                    'quality_score': best['quality_score'],
                    'gate_passed': best.get('gate_passed', False),
                    'gate_score': best.get('gate_score', 0),
                    'killzone': best.get('killzone', ''),
                    'confluence': best.get('confluence', 0),
                    'has_sweep': best.get('has_sweep', False),
                    'has_displacement': best.get('has_displacement', False),
                    'vol_spike': best.get('vol_spike', False),
                    'in_ote': best.get('in_ote', False),
                    'rr_ratio': best.get('rr_ratio', 0),
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
    # EXIT LOGIC (değişiklik yok)
    # ══════════════════════════════════════════════════════════════════════
    def _check_ict_exit(self, t, current_price):
        strat = t.get('strategy', '') or t.get('signal_result', {}).get('strategy', '')
        if strat not in ('ict_smc_v1', 'ict_smc_v2'):
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

        if pnl_dollar < -ict_max_loss:
            r_val = self._calc_realized_r(side, entry, current_price, initial_r)
            self._close_all_locked([t], symbol, current_price, "ICT_MAXLOSS")
            self._ict_record_outcome('SL', symbol, r_val)
            self.log(f"[CAP] ICT {symbol} max loss: ${pnl_dollar:.2f}")
            return

        # ── v2.4: ATR-based Breakeven + Trailing (PM yoksa da koruma) ──
        atr_val = t.get('atr', entry * 0.005)
        if atr_val <= 0:
            atr_val = entry * 0.005

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

            # Trailing: 3.0 ATR kârda → SL = peak - 1.5 ATR
            if peak_profit_atr >= 3.0:
                trail_sl = peak_price - 1.5 * atr_val
                if trail_sl > sl:
                    t['sl_price'] = trail_sl
                    sl = trail_sl
                    self.log(f"[TRAIL] ICT {symbol}: SL→{trail_sl:.6f} (peak {peak_price:.6f})")

        else:  # SHORT
            price_profit_atr = (entry - current_price) / atr_val
            peak_price = min(current_price, t.get('_ict_peak_price', entry))
            t['_ict_peak_price'] = peak_price
            peak_profit_atr = (entry - peak_price) / atr_val

            if peak_profit_atr >= 1.5 and not t.get('_ict_be_set'):
                be_sl = entry - 0.3 * atr_val
                if be_sl < sl:
                    t['sl_price'] = be_sl
                    sl = be_sl
                    t['_ict_be_set'] = True
                    self.log(f"[BE] ICT {symbol}: SL→BE {be_sl:.6f} (peak {peak_profit_atr:.1f} ATR)")
                    self._force_save()

            if peak_profit_atr >= 3.0:
                trail_sl = peak_price + 1.5 * atr_val
                if trail_sl < sl:
                    t['sl_price'] = trail_sl
                    sl = trail_sl
                    self.log(f"[TRAIL] ICT {symbol}: SL→{trail_sl:.6f} (peak {peak_price:.6f})")

        hit_sl = (side == 'LONG' and current_price <= sl) or \
                 (side == 'SHORT' and current_price >= sl)
        if hit_sl:
            reason = "ICT_SL" if pnl_dollar < 0 else "ICT_TRAIL_SL"
            r_val = self._calc_realized_r(side, entry, current_price, initial_r)
            self._close_all_locked([t], symbol, current_price, reason)
            if pnl_dollar < 0:
                self._ict_record_outcome('SL', symbol, r_val)
            else:
                self._ict_record_outcome('TP', symbol, r_val)
            self.log(f"[STOP] ICT {reason}: {symbol} ${pnl_dollar:+.2f}")
            return

        if t.get('_ict_tp1_hit') and pnl_dollar > 0:
            try:
                self._update_structural_sl(t, current_price, fetcher=self._get_fetcher())
            except Exception:
                pass

        tp_dist = abs(tp - entry)
        if tp_dist > 0:
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
                        self._ict_record_outcome('TP', symbol, r_val)
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
                        self._ict_record_outcome('TP', symbol, r_val)
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

        if t.get('_ict_trail_mode'):
            trail_start = t.get('_ict_tp3_time', time.time())
            if time.time() - trail_start > 24 * 3600:
                r_val = self._calc_realized_r(side, entry, current_price, initial_r)
                self._close_all_locked([t], symbol, current_price, "ICT_TRAIL_TIMEOUT")
                self._ict_record_outcome('TP', symbol, r_val)
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
                                _ict = _ic.analyze(_df, side, swing_left=3, swing_right=2)
                                bad_dir = 'bearish' if side == 'LONG' else 'bullish'
                                if _ict.last_choch and _ict.last_choch.direction == bad_dir:
                                    r_val = self._calc_realized_r(side, entry, current_price, initial_r)
                                    self._close_all_locked([t], symbol, current_price, "ICT_CHOCH_EXIT")
                                    self._ict_record_outcome('TP', symbol, r_val)
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
            self._ict_record_outcome(outcome, symbol, r_val)
            self.log(f"[TIME] ICT timeout: {symbol} ${pnl_dollar:+.2f}")
            return