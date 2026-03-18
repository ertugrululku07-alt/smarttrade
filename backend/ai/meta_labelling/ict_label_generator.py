"""
ICT/SMC Meta-Label Generator v1.0

ICT setup'larını tespit eder, outcome simüle eder, meta_label üretir.
Eski AdaptiveEngine.decide() yerine ict_core.analyze() kullanır.

Pipeline:
  1. Her bar'da ict_core.analyze() çalıştır
  2. ICT setup var mı? (sweep + CHoCH/BOS + FVG/OB)
  3. Varsa → yön belirle, SL/TP hesapla, outcome simüle et
  4. ICT feature'ları + meta_label → meta_df satırı
  5. Tüm satırları birleştir → meta_df
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, List, Tuple, Optional

from ai import ict_core
from ai.ict_features import extract_ict_features, ICT_FEATURES


# ─────────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────────

REGIME_VALUES = ['trending', 'mean_reverting', 'high_volatile', 'low_volatile']

# ICT Setup minimum gereksinimleri
MIN_STRUCTS_FOR_SIGNAL = 2   # 1h'de min 2 yapı (sweep+CHoCH veya FVG+OB vb.)
LOOKBACK_WINDOW = 60         # Her bar için analiz penceresi
LOOKAHEAD = 16               # Outcome simülasyonu ileri bakış

# Timeout threshold (ATR cinsinden PnL)
TIMEOUT_THRESHOLD = 0.2

# Trail stop
TRAIL_ACTIVATION = 0.6


class ICTLabelGenerator:
    """
    ICT/SMC tabanlı meta-label üretici.
    """

    def __init__(
        self,
        lookback: int = LOOKBACK_WINDOW,
        lookahead: int = LOOKAHEAD,
        min_structs: int = MIN_STRUCTS_FOR_SIGNAL,
        swing_left: int = 3,
        swing_right: int = 2,
        progress_interval: int = 1000,
    ):
        self.lookback = lookback
        self.lookahead = lookahead
        self.min_structs = min_structs
        self.swing_left = swing_left
        self.swing_right = swing_right
        self.progress_interval = progress_interval

    def generate(
        self,
        df: pd.DataFrame,
        df_4h: Optional[pd.DataFrame] = None,
        symbol: str = "UNKNOWN",
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        ICT setup tarama + meta-label üretimi.

        Args:
            df:     1h OHLCV + indicators DataFrame
            df_4h:  4h OHLCV DataFrame (HTF context)
            symbol: Coin sembolü
            verbose: Detaylı çıktı

        Returns:
            (meta_df, stats)
        """
        N = len(df)
        min_required = self.lookback + self.lookahead

        if N < min_required:
            if verbose:
                print(f"  [WARN] {symbol}: Yetersiz veri ({N} < {min_required})")
            return pd.DataFrame(), {}

        closes = df['close'].values.astype(float)
        highs = df['high'].values.astype(float)
        lows = df['low'].values.astype(float)
        atrs = self._safe_get_atr(df, closes)

        stats = {
            'bars_evaluated': 0,
            'signals_generated': 0,
            'meta_label_1': 0,
            'meta_label_0': 0,
            'long_signals': 0,
            'short_signals': 0,
        }

        # 4h HTF analiz (bir kez, her ~4 saatte güncellenir)
        htf_cache = {}

        start_time = time.time()
        scan_start = self.lookback
        scan_end = N - self.lookahead
        results = []

        for i in range(scan_start, scan_end):
            stats['bars_evaluated'] += 1

            # Progress
            if verbose and self.progress_interval > 0:
                bars_done = i - scan_start + 1
                total = scan_end - scan_start
                if bars_done % self.progress_interval == 0:
                    pct = bars_done / max(total, 1) * 100
                    elapsed = time.time() - start_time
                    print(f"    {symbol}: {pct:.0f}% ({bars_done}/{total}, "
                          f"{elapsed:.1f}s, sigs={stats['signals_generated']})")

            atr_val = atrs[i]
            if np.isnan(atr_val) or atr_val < 1e-10:
                continue

            # ── 1h ICT Analiz (pencere) ──
            w_start = max(0, i - self.lookback)
            df_slice = df.iloc[w_start:i + 1]

            analysis = ict_core.analyze(
                df_slice, '', self.swing_left, self.swing_right
            )

            # ── 4h HTF Analiz (cache'li) ──
            htf_analysis = self._get_htf_analysis(df_4h, df, i, htf_cache)

            # ── ICT Setup kontrolü ──
            direction, structs, setup_info = self._check_ict_setup(
                analysis, htf_analysis, closes[i], atr_val
            )

            if direction is None:
                continue

            # ── Regime tespiti (basit) ──
            regime = self._detect_regime(df_slice)

            # ── SL/TP hesapla ──
            sl_price, tp_price = self._calculate_sl_tp(
                direction, analysis, closes[i], atr_val
            )

            # ── Outcome simüle ──
            outcome = self._simulate_outcome(
                direction=direction,
                entry=closes[i],
                atr_val=atr_val,
                sl_price=sl_price,
                tp_price=tp_price,
                highs=highs,
                lows=lows,
                closes=closes,
                start_idx=i,
                N=N,
            )

            meta_label = outcome['meta_label']
            stats['signals_generated'] += 1
            if direction == 'LONG':
                stats['long_signals'] += 1
            else:
                stats['short_signals'] += 1

            if meta_label == 1:
                stats['meta_label_1'] += 1
            else:
                stats['meta_label_0'] += 1

            # ── ICT Features ──
            ict_feats = extract_ict_features(
                analysis, closes[i], atr_val, direction
            )

            # HTF features
            htf_feats = self._extract_htf_features(htf_analysis, closes[i], atr_val)

            # ── Row oluştur ──
            row = {
                '_bar_index': i,
                '_regime': regime,
                '_outcome': outcome['type'],
                '_pnl_atr': outcome['pnl_atr'],
                '_bars_to_outcome': outcome['bars'],
                '_setup_info': setup_info,
                'signal_is_long': int(direction == 'LONG'),
                'signal_confidence': min(1.0, structs * 0.25),
                'meta_label': meta_label,
                **{f"regime_{rv}": int(regime == rv) for rv in REGIME_VALUES},
                **ict_feats,
                **htf_feats,
            }

            results.append(row)

        # ── Assemble ──
        if not results:
            if verbose:
                print(f"  [WARN] {symbol}: Sinyal üretilemedi")
            return pd.DataFrame(), stats

        meta_df = pd.DataFrame(results)

        # Context features varsa ekle (signals.py'den)
        meta_df = self._add_context_features(df, meta_df)

        if verbose:
            elapsed = time.time() - start_time
            wr = stats['meta_label_1'] / max(stats['signals_generated'], 1) * 100
            print(f"  {symbol}: {stats['signals_generated']} sinyals "
                  f"(L:{stats['long_signals']} S:{stats['short_signals']}) "
                  f"WR={wr:.1f}% ({elapsed:.1f}s)")

        return meta_df, stats

    # ══════════════════════════════════════════════
    # ICT SETUP TESPİTİ
    # ══════════════════════════════════════════════

    def _check_ict_setup(
        self,
        analysis: ict_core.ICTAnalysis,
        htf_analysis: Optional[ict_core.ICTAnalysis],
        current_price: float,
        atr: float,
    ) -> Tuple[Optional[str], int, str]:
        """
        ICT setup var mı kontrol et.

        Setup gereksinimi:
          - HTF yapı uyumlu
          - 1h'de min 2 yapı elemanı (sweep, CHoCH/BOS, FVG, OB)

        Returns: (direction, struct_count, setup_info) veya (None, 0, '')
        """
        # ── Yön belirleme ──
        # HTF varsa ona göre, yoksa 1h yapısına göre
        htf_ms = htf_analysis.market_structure if htf_analysis else analysis.market_structure

        if htf_ms == 'bullish':
            direction = 'LONG'
        elif htf_ms == 'bearish':
            direction = 'SHORT'
        else:
            # Ranging → CHoCH varsa ona göre yön belirle
            if analysis.last_choch:
                direction = 'LONG' if analysis.last_choch.direction == 'bullish' else 'SHORT'
            else:
                return None, 0, ''

        # ── Yapı sayacı ──
        structs = 0
        info_parts = []
        exp_dir = 'bullish' if direction == 'LONG' else 'bearish'
        exp_sweep = 'ssl_sweep' if direction == 'LONG' else 'bsl_sweep'

        # Sweep
        if analysis.sweep_detected and analysis.sweep_type == exp_sweep:
            structs += 1
            info_parts.append('Sweep')

        # CHoCH (en güçlü)
        if analysis.last_choch and analysis.last_choch.direction == exp_dir:
            structs += 1
            info_parts.append('CHoCH')
        elif analysis.last_bos and analysis.last_bos.direction == exp_dir:
            structs += 1
            info_parts.append('BOS')

        # FVG (unfilled, doğru yön)
        fvg_type = 'bullish' if direction == 'LONG' else 'bearish'
        active_fvgs = [f for f in analysis.fvg_zones
                       if f.type == fvg_type and not f.filled]
        if active_fvgs:
            structs += 1
            info_parts.append(f'FVG({len(active_fvgs)})')

        # OB (unmitigated, doğru yön)
        ob_type = 'bullish' if direction == 'LONG' else 'bearish'
        active_obs = [ob for ob in analysis.order_blocks
                      if ob.type == ob_type and not ob.mitigated]
        if active_obs:
            structs += 1
            info_parts.append(f'OB({len(active_obs)})')

        # Displacement
        if analysis.displacement and analysis.displacement_direction == exp_dir:
            structs += 1
            info_parts.append('Disp')

        # Premium/Discount zone check
        sh_p = [sp.price for sp in analysis.swing_highs[-5:]] if analysis.swing_highs else []
        sl_p = [sp.price for sp in analysis.swing_lows[-5:]] if analysis.swing_lows else []
        sh = max(sh_p) if sh_p else current_price * 1.02
        sl = min(sl_p) if sl_p else current_price * 0.98
        rng = sh - sl
        pos = (current_price - sl) / rng if rng > 0 else 0.5

        # Zone filtre: LONG premium'da (>0.6) veya SHORT discount'ta (<0.4) → reject
        if direction == 'LONG' and pos > 0.65:
            return None, 0, ''
        if direction == 'SHORT' and pos < 0.35:
            return None, 0, ''

        if structs < self.min_structs:
            return None, 0, ''

        # Yapısal kırılma ZORUNLU — sadece FVG+OB yetmez
        has_break = ('CHoCH' in info_parts or 'BOS' in info_parts)
        if not has_break:
            return None, 0, ''

        # POI confluence bonus
        if analysis.poi_confluence >= 0.5:
            structs += 1
            info_parts.append(f'POI({analysis.poi_confluence:.1f})')

        setup_info = f"{direction}:{'+'.join(info_parts)}"
        return direction, structs, setup_info

    # ══════════════════════════════════════════════
    # SL / TP
    # ══════════════════════════════════════════════

    def _calculate_sl_tp(
        self,
        direction: str,
        analysis: ict_core.ICTAnalysis,
        entry: float,
        atr: float,
    ) -> Tuple[float, float]:
        """ICT bazlı SL/TP hesapla."""
        # SL: sweep bazlı
        sl = ict_core.get_sweep_sl(
            direction, analysis.sweep_level, entry, atr,
            analysis.swing_highs, analysis.swing_lows
        )

        # TP: likidite bazlı
        tp = ict_core.get_liquidity_tp(
            direction, entry,
            analysis.equal_highs, analysis.equal_lows,
            analysis.swing_highs, analysis.swing_lows
        )

        # Safety checks
        if direction == 'LONG':
            if sl >= entry:
                sl = entry - atr * 1.5
            if tp <= entry:
                tp = entry + atr * 3.0
            # Max SL %2.5
            if (entry - sl) / entry > 0.025:
                sl = entry * 0.975
        else:
            if sl <= entry:
                sl = entry + atr * 1.5
            if tp >= entry:
                tp = entry - atr * 3.0
            if (sl - entry) / entry > 0.025:
                sl = entry * 1.025

        return sl, tp

    # ══════════════════════════════════════════════
    # OUTCOME SİMÜLASYONU
    # ══════════════════════════════════════════════

    def _simulate_outcome(
        self,
        direction: str,
        entry: float,
        atr_val: float,
        sl_price: float,
        tp_price: float,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        start_idx: int,
        N: int,
    ) -> Dict:
        """Conservative outcome simulation with trailing stop."""
        trail_activated = False
        trail_sl = sl_price
        best_price = entry

        for j in range(1, self.lookahead + 1):
            idx = start_idx + j
            if idx >= N:
                break

            h = highs[idx]
            lo = lows[idx]

            if direction == 'LONG':
                best_price = max(best_price, h)
                trail_level = entry + (tp_price - entry) * TRAIL_ACTIVATION
                if not trail_activated and best_price >= trail_level:
                    trail_activated = True
                if trail_activated:
                    new_trail = best_price - atr_val * 1.5
                    if new_trail > trail_sl:
                        trail_sl = new_trail

                active_sl = trail_sl if trail_activated else sl_price
                sl_hit = lo <= active_sl
                tp_hit = h >= tp_price

                if sl_hit and tp_hit:
                    pnl = (active_sl - entry) / atr_val
                    return {'meta_label': 1 if pnl > 0 else 0,
                            'type': 'AMBIGUOUS', 'pnl_atr': round(pnl, 4), 'bars': j}
                if sl_hit:
                    pnl = (active_sl - entry) / atr_val
                    return {'meta_label': 1 if pnl > 0 else 0,
                            'type': 'TRAIL_STOP' if trail_activated else 'SL',
                            'pnl_atr': round(pnl, 4), 'bars': j}
                if tp_hit:
                    tp_mult = (tp_price - entry) / atr_val
                    return {'meta_label': 1, 'type': 'TP',
                            'pnl_atr': round(tp_mult, 4), 'bars': j}

            else:  # SHORT
                best_price = min(best_price, lo)
                trail_level = entry - (entry - tp_price) * TRAIL_ACTIVATION
                if not trail_activated and best_price <= trail_level:
                    trail_activated = True
                if trail_activated:
                    new_trail = best_price + atr_val * 1.5
                    if new_trail < trail_sl:
                        trail_sl = new_trail

                active_sl = trail_sl if trail_activated else sl_price
                sl_hit = h >= active_sl
                tp_hit = lo <= tp_price

                if sl_hit and tp_hit:
                    pnl = (entry - active_sl) / atr_val
                    return {'meta_label': 1 if pnl > 0 else 0,
                            'type': 'AMBIGUOUS', 'pnl_atr': round(pnl, 4), 'bars': j}
                if sl_hit:
                    pnl = (entry - active_sl) / atr_val
                    return {'meta_label': 1 if pnl > 0 else 0,
                            'type': 'TRAIL_STOP' if trail_activated else 'SL',
                            'pnl_atr': round(pnl, 4), 'bars': j}
                if tp_hit:
                    tp_mult = (entry - tp_price) / atr_val
                    return {'meta_label': 1, 'type': 'TP',
                            'pnl_atr': round(tp_mult, 4), 'bars': j}

        # Timeout
        final_idx = min(start_idx + self.lookahead, N - 1)
        final_close = closes[final_idx]
        if direction == 'LONG':
            pnl = (final_close - entry) / atr_val
        else:
            pnl = (entry - final_close) / atr_val

        return {
            'meta_label': 1 if pnl > TIMEOUT_THRESHOLD else 0,
            'type': 'TIMEOUT', 'pnl_atr': round(pnl, 4), 'bars': self.lookahead,
        }

    # ══════════════════════════════════════════════
    # YARDIMCI METODLAR
    # ══════════════════════════════════════════════

    def _safe_get_atr(self, df: pd.DataFrame, closes: np.ndarray) -> np.ndarray:
        """ATR array al."""
        if 'atr' in df.columns:
            atrs = df['atr'].values.astype(float)
            # NaN olanları hesapla
            mask = np.isnan(atrs) | (atrs <= 0)
            if mask.any():
                atrs[mask] = closes[mask] * 0.01
            return atrs
        # Manuel ATR
        highs = df['high'].values.astype(float)
        lows = df['low'].values.astype(float)
        tr = np.maximum(highs - lows,
                        np.maximum(np.abs(highs - np.roll(closes, 1)),
                                   np.abs(lows - np.roll(closes, 1))))
        tr[0] = highs[0] - lows[0]
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
        return atr

    def _detect_regime(self, df_slice: pd.DataFrame) -> str:
        """ADX + Hurst + volatilite bazlı regime tespiti."""
        if len(df_slice) < 30:
            return 'trending'

        # ADX varsa kullan
        adx_val = None
        if 'adx' in df_slice.columns:
            adx_val = float(df_slice['adx'].iloc[-1])
            if np.isnan(adx_val):
                adx_val = None

        # Hurst varsa kullan
        hurst_val = None
        if 'hurst' in df_slice.columns:
            hurst_val = float(df_slice['hurst'].iloc[-1])
            if np.isnan(hurst_val):
                hurst_val = None

        # Volatilite
        c = df_slice['close'].values.astype(float)
        returns = np.diff(c[-30:]) / c[-30:-1]
        vol = np.std(returns) if len(returns) > 5 else 0.01

        # Karar ağacı
        if vol > 0.025:
            return 'high_volatile'
        if vol < 0.004:
            return 'low_volatile'

        # ADX bazlı trend/MR
        if adx_val is not None:
            if adx_val > 25:
                return 'trending'
            elif adx_val < 18:
                return 'mean_reverting'

        # Hurst bazlı
        if hurst_val is not None:
            if hurst_val > 0.55:
                return 'trending'
            elif hurst_val < 0.45:
                return 'mean_reverting'

        # Fallback: momentum
        mom = (c[-1] - c[-20]) / c[-20] if len(c) >= 20 else 0
        if abs(mom) > 0.03:
            return 'trending'
        return 'mean_reverting'

    def _get_htf_analysis(
        self,
        df_4h: Optional[pd.DataFrame],
        df_1h: pd.DataFrame,
        current_idx: int,
        cache: Dict,
    ) -> Optional[ict_core.ICTAnalysis]:
        """4h analiz (cache'li, look-ahead korumalı)."""
        if df_4h is None or len(df_4h) < 30:
            return None

        # 1h index'ten 4h zamanına map
        if isinstance(df_1h.index, pd.DatetimeIndex) and isinstance(df_4h.index, pd.DatetimeIndex):
            current_time = df_1h.index[current_idx]
            safe_4h = df_4h[df_4h.index <= current_time]
            if len(safe_4h) < 30:
                return None

            cache_key = str(safe_4h.index[-1])
            if cache_key in cache:
                return cache[cache_key]

            analysis = ict_core.analyze(safe_4h, '', 3, 2)
            cache[cache_key] = analysis
            return analysis
        else:
            # Timestamp index yok → ratio ile approximation
            ratio = 4  # 1h → 4h
            safe_4h_len = min(len(df_4h), current_idx // ratio + 1)
            if safe_4h_len < 30:
                return None

            cache_key = f"idx_{safe_4h_len}"
            if cache_key in cache:
                return cache[cache_key]

            analysis = ict_core.analyze(df_4h.iloc[:safe_4h_len], '', 3, 2)
            cache[cache_key] = analysis
            return analysis

    def _extract_htf_features(
        self,
        htf_analysis: Optional[ict_core.ICTAnalysis],
        current_price: float,
        atr: float,
    ) -> Dict[str, float]:
        """HTF feature extraction (4h → 6 feature)."""
        defaults = {
            'htf_bias_numeric': 0.0,
            'htf_trend_strength': 0.0,
            'htf_rsi': 50.0,
            'htf_price_vs_ema200': 0.0,
            'htf_ema_alignment': 0.0,
            'htf_structure_numeric': 0.0,
        }

        if htf_analysis is None:
            return defaults

        ms = htf_analysis.market_structure
        # bias
        bias = 0.0
        if ms == 'bullish': bias = 1.5
        elif ms == 'bearish': bias = -1.5
        if htf_analysis.last_bos:
            bias += 0.5 if htf_analysis.last_bos.direction == 'bullish' else -0.5

        defaults['htf_bias_numeric'] = max(-2.0, min(2.0, bias))
        defaults['htf_structure_numeric'] = {'bullish': 1.0, 'bearish': -1.0}.get(ms, 0.0)

        # Trend strength
        n_labels = len(htf_analysis.structure_labels)
        if n_labels >= 4:
            recent = [lb.label for lb in htf_analysis.structure_labels[-4:]]
            bull = recent.count('HH') + recent.count('HL')
            bear = recent.count('LL') + recent.count('LH')
            defaults['htf_trend_strength'] = abs(bull - bear) / 4.0

        return defaults

    def _add_context_features(
        self,
        df: pd.DataFrame,
        meta_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Context feature'ları meta_df'e ekle (bar_index ile join)."""
        context_cols = [
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'vol_relative_50', 'vol_trend', 'vol_climax',
            'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
            'consec_direction', 'consec_magnitude',
            'bar_range_vs_atr', 'dist_ema200_atr',
            'price_position_50', 'range_position',
            'bb_squeeze', 'rsi_slope_5', 'sr_proximity',
        ]

        available = [c for c in context_cols if c in df.columns]
        if not available or '_bar_index' not in meta_df.columns:
            return meta_df

        for col in available:
            vals = df[col].values
            meta_df[col] = meta_df['_bar_index'].apply(
                lambda idx: float(vals[idx]) if idx < len(vals) else 0.0
            )

        return meta_df


# ═══════════════════════════════════════════════════════════════
# BULK GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_ict_labels_bulk(
    all_dfs: Dict[str, pd.DataFrame],
    all_dfs_4h: Optional[Dict[str, pd.DataFrame]] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Birden fazla coin için ICT meta-label üret.
    """
    verbose = kwargs.pop('verbose', True)
    if all_dfs_4h is None:
        all_dfs_4h = {}

    all_results = []
    combined_stats = {
        'total_signals': 0, 'total_correct': 0, 'total_wrong': 0,
        'coins_processed': 0, 'coins_skipped': 0,
        'per_coin': {},
    }

    total_coins = len(all_dfs)
    total_start = time.time()

    for idx, (symbol, df) in enumerate(all_dfs.items(), 1):
        if verbose:
            print(f"\n  [{idx}/{total_coins}] {symbol} ({len(df)} bars)")

        try:
            gen = ICTLabelGenerator(**kwargs)
            df_4h = all_dfs_4h.get(symbol)

            meta_df, stats = gen.generate(df, df_4h=df_4h, symbol=symbol, verbose=verbose)

            if meta_df.empty:
                combined_stats['coins_skipped'] += 1
                continue

            meta_df['_symbol'] = symbol
            all_results.append(meta_df)

            combined_stats['coins_processed'] += 1
            combined_stats['total_signals'] += stats.get('signals_generated', 0)
            combined_stats['total_correct'] += stats.get('meta_label_1', 0)
            combined_stats['total_wrong'] += stats.get('meta_label_0', 0)
            combined_stats['per_coin'][symbol] = stats

        except Exception as e:
            print(f"  [ERROR] {symbol}: {e}")
            import traceback
            traceback.print_exc()
            combined_stats['coins_skipped'] += 1

    combined_stats['total_time'] = round(time.time() - total_start, 1)

    if not all_results:
        if verbose:
            print("\n  [WARN] Hiç meta-label üretilemedi!")
        return pd.DataFrame(), combined_stats

    total_df = pd.concat(all_results, axis=0, ignore_index=True)

    if verbose:
        wr = combined_stats['total_correct'] / max(combined_stats['total_signals'], 1) * 100
        print(f"\n  [BULK] {combined_stats['coins_processed']} coin, "
              f"{combined_stats['total_signals']} sinyal, WR={wr:.1f}%, "
              f"{combined_stats['total_time']}s")

    return total_df, combined_stats
