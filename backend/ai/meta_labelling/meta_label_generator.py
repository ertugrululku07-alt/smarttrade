"""
Meta-Label Generator v1.3 (Production-Hardened + MTF Ready)

Approach (Marcos Lopez de Prado):
  1. Primary model (rule-based) generates signal
  2. Simulate if the signal turns out to be CORRECT
  3. meta_label = 1 (correct) / 0 (wrong)
  4. Inject Cross-TF features into training data (MTF Phase 1.5)

v1.3 Improvements:
  - 3-tier engine.decide() interface fallback
  - 4h look-ahead: completed-bar-only protection
  - MTF cross-TF features added to meta_df bar-by-bar
  - signal dict validation + default fallback
  - Regime-aware timeout thresholds
  - Bulk: fresh generator per coin (stale data protection)
  - Progress reporting for large datasets
  - Trailing stop edge case protection
  - Clean ASCII logs and comments
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
import warnings


# =====================================================================
# Constants
# =====================================================================

REGIME_VALUES = [
    'trending',
    'mean_reverting',
    'high_volatile',
    'low_volatile',
]

# Signal info: defaults if missing
SIGNAL_DEFAULTS = {
    'direction': None,
    'confidence': 0.5,
    'tp_atr_mult': 3.0,
    'sl_atr_mult': 1.5,
}

# Timeout thresholds: regime-based
TIMEOUT_THRESHOLDS = {
    'trending': 0.3,            # Tolerant in Trend
    'mean_reverting': 0.15,     # Strict in MR
    'high_volatile': 0.25,      # Medium in Volatile
    'low_volatile': 0.1,        # Strict in Low Vol
}
TIMEOUT_THRESHOLD_DEFAULT = 0.2


# =====================================================================
# Data Structures
# =====================================================================

@dataclass
class MetaLabelRow:
    """Structure for a single meta-label record."""
    bar_index: int
    regime: str
    signal_direction: str
    meta_label: int
    outcome_type: str
    pnl_atr: float
    bars_to_outcome: int


# =====================================================================
# Main Class
# =====================================================================

class MetaLabelGenerator:
    """
    Simulates strategy signal success to produce labeled training data
    for Meta-Model (XGBoost).

    MTF Integration (Phase 1.5):
      - Injects cross-TF features into meta_df if 4h data exists
      - Look-ahead protection: uses only completed 4h bars
      - Graceful degradation if MTFAnalyzer is missing
    """

    def __init__(
        self,
        lookahead: int = 16,
        trail_activation: float = 0.6,
        regime_lookback: int = 50,
        timeframe: str = "1h",
        enable_mtf: bool = True,
        progress_interval: int = 1000,
    ):
        self.lookahead = lookahead
        self.trail_activation = trail_activation
        self.regime_lookback = regime_lookback
        self.timeframe = timeframe
        self.enable_mtf = enable_mtf
        self.progress_interval = progress_interval

        # Engine - lazy init for import cycle protection
        self._engine = None
        self._mtf_analyzer = None
        self._mtf_import_attempted = False

    # -- Lazy Properties -------------------------------

    @property
    def engine(self):
        """Engine lazy initialization."""
        if self._engine is None:
            from ai.adaptive_engine import AdaptiveEngine
            self._engine = AdaptiveEngine(
                timeframe=self.timeframe,
                use_meta_filter=False,
            )
        return self._engine

    @property
    def mtf_analyzer(self):
        """MTF Analyzer lazy initialization (optional)."""
        if not self.enable_mtf:
            return None

        if self._mtf_analyzer is None and not self._mtf_import_attempted:
            self._mtf_import_attempted = True
            try:
                from ai.mtf_analyzer import MTFAnalyzer
                self._mtf_analyzer = MTFAnalyzer()
            except ImportError:
                self._mtf_analyzer = None

        return self._mtf_analyzer

    def reset_engine(self):
        """Reset engine state (between coins)."""
        self._engine = None

    # ==================================================
    # MAIN GENERATION
    # ==================================================

    def generate(
        self,
        df: pd.DataFrame,
        df_4h: Optional[pd.DataFrame] = None,
        symbol: str = "UNKNOWN",
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Scan dataset and generate meta-labels.

        Args:
            df:      Primary TF OHLCV + indicator DataFrame
            df_4h:   4h OHLCV DataFrame (optional, for MTF)
            symbol:  Coin symbol
            verbose: Reporting

        Returns:
            (meta_df, stats_dict)
        """
        N = len(df)
        min_required = self.regime_lookback + self.lookahead

        if N < min_required:
            if verbose:
                print(
                    f"  [WARN] {symbol}: Insufficient data "
                    f"({N} < {min_required})"
                )
            return pd.DataFrame(), {}

        # -- Data Arrays --
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        atrs = self._safe_get_atr(df)

        # -- Statistics --
        stats = {
            'bars_evaluated': 0,
            'signals_generated': 0,
            'meta_label_1': 0,
            'meta_label_0': 0,
            'regime_counts': {},
            'by_regime': {},
        }

        # -- Engine Precompute --
        self._safe_precompute(df)

        # -- MTF Context Cache --
        mtf_cache = {}

        # -- Window Size --
        engine_lookback = max(300, self.regime_lookback)

        # -- Progress Tracking --
        start_time = time.time()
        scan_start = self.regime_lookback
        scan_end = N - self.lookahead
        total_bars = scan_end - scan_start
        bars_processed = 0

        results = []

        for i in range(scan_start, scan_end):
            bars_processed += 1
            stats['bars_evaluated'] += 1

            # Progress reporting
            if (verbose
                    and self.progress_interval > 0
                    and bars_processed % self.progress_interval == 0):
                elapsed = time.time() - start_time
                pct = bars_processed / max(total_bars, 1) * 100
                print(
                    f"    {symbol}: {pct:.0f}% "
                    f"({bars_processed}/{total_bars} bars, "
                    f"{elapsed:.1f}s, "
                    f"signals={stats['signals_generated']})"
                )

            # ATR check
            atr_val = atrs[i]
            if np.isnan(atr_val) or atr_val < 1e-10:
                continue

            # -- Engine Window --
            window_start = max(0, i - engine_lookback)
            df_slice = df.iloc[window_start:i + 1]

            # -- 4h Look-Ahead Protection --
            active_df_4h = self._get_safe_4h_slice(df_4h, df, i)

            # -- MTF Context (Cached) --
            htf_context = self._get_htf_context(
                active_df_4h, symbol, mtf_cache
            )

            # -- Engine Decision --
            decision = self._safe_decide(
                df_slice, active_df_4h, symbol
            )

            if decision is None:
                continue

            regime_str = self._extract_regime(decision)

            # Regime count
            stats['regime_counts'][regime_str] = (
                stats['regime_counts'].get(regime_str, 0) + 1
            )

            # HOLD -> continue
            if self._is_hold(decision):
                continue

            # -- Signal Validation --
            signal = self._get_validated_signal()
            if signal is None:
                continue

            # -- Signal Stats --
            stats['signals_generated'] += 1
            if regime_str not in stats['by_regime']:
                stats['by_regime'][regime_str] = {
                    'signals': 0, 'correct': 0, 'wrong': 0,
                }
            stats['by_regime'][regime_str]['signals'] += 1

            # -- Outcome Simulation --
            timeout_thresh = TIMEOUT_THRESHOLDS.get(
                regime_str, TIMEOUT_THRESHOLD_DEFAULT
            )

            outcome = self._simulate_outcome(
                direction=signal['direction'],
                entry=closes[i],
                atr_val=atr_val,
                tp_mult=signal['tp_atr_mult'],
                sl_mult=signal['sl_atr_mult'],
                highs=highs,
                lows=lows,
                closes=closes,
                start_idx=i,
                N=N,
                timeout_threshold=timeout_thresh,
                actual_tp=signal.get('tp_price', 0.0),
                actual_sl=signal.get('sl_price', 0.0),
            )

            meta_label = outcome['meta_label']
            if meta_label == 1:
                stats['meta_label_1'] += 1
                stats['by_regime'][regime_str]['correct'] += 1
            else:
                stats['meta_label_0'] += 1
                stats['by_regime'][regime_str]['wrong'] += 1

            # -- Result Row --
            row = {
                '_bar_index': i,
                '_regime': regime_str,
                '_outcome': outcome['type'],
                '_pnl_atr': outcome['pnl_atr'],
                '_bars_to_outcome': outcome['bars'],
                'signal_is_long': int(signal['direction'] == 'LONG'),
                'signal_confidence': signal['confidence'],
                'meta_label': meta_label,
                **{
                    f"regime_{rv}": int(regime_str == rv)
                    for rv in REGIME_VALUES
                },
            }

            # -- MTF Features --
            mtf_features = self._generate_mtf_features(
                htf_context, closes[i], atr_val
            )
            row.update(mtf_features)

            results.append(row)

        # -- Assemble DataFrame --
        if not results:
            if verbose:
                print(f"  [WARN] {symbol}: No signals generated")
            return pd.DataFrame(), stats

        meta_df = self._assemble_dataframe(df, results)

        if verbose:
            elapsed = time.time() - start_time
            self._print_report(symbol, stats, elapsed)

        return meta_df, stats

    # ==========================================================
    # ENGINE INTERFACE
    # ==========================================================

    def _safe_decide(
        self,
        df_slice: pd.DataFrame,
        df_4h: Optional[pd.DataFrame],
        symbol: str,
    ):
        """
        Engine.decide() - 3-tier interface fallback.
        """
        # Tier 1: MTF-aware
        try:
            return self.engine.decide(
                df_slice,
                df_4h=df_4h,
                symbol=symbol,
            )
        except TypeError:
            pass

        # Tier 2: Symbol-aware
        try:
            return self.engine.decide(df_slice, symbol=symbol)
        except TypeError:
            pass

        # Tier 3: Basic
        try:
            return self.engine.decide(df_slice)
        except Exception as e:
            warnings.warn(
                f"Engine.decide() error: {e}",
                RuntimeWarning,
            )
            return None

    def _extract_regime(self, decision) -> str:
        """Extract regime string from decision object."""
        if hasattr(decision, 'regime'):
            regime = str(decision.regime).lower().strip()
        elif isinstance(decision, dict):
            regime = str(decision.get('regime', 'unknown')).lower().strip()
        else:
            regime = 'unknown'

        if regime not in REGIME_VALUES:
            regime = 'unknown'

        return regime

    def _is_hold(self, decision) -> bool:
        """Is decision HOLD?"""
        if hasattr(decision, 'action'):
            return decision.action == 'HOLD'
        if isinstance(decision, dict):
            return decision.get('action') == 'HOLD'
        return True

    # ==========================================================
    # SIGNAL VALIDATION
    # ==========================================================

    def _get_validated_signal(self) -> Optional[Dict]:
        """
        Get and validate last signal info from engine.
        """
        try:
            signal = self.engine.get_last_signal_info()
        except (AttributeError, Exception):
            return None

        if not signal or not isinstance(signal, dict):
            return None

        # Direction is required
        direction = signal.get('direction')
        if direction not in ('LONG', 'SHORT'):
            return None

        # Fill missing with defaults
        validated = {
            'direction': direction,
            'confidence': self._safe_float(
                signal.get('confidence'),
                SIGNAL_DEFAULTS['confidence'],
                min_val=0.0,
                max_val=1.0,
            ),
            'tp_atr_mult': self._safe_float(
                signal.get('tp_atr_mult'),
                SIGNAL_DEFAULTS['tp_atr_mult'],
                min_val=0.1,
            ),
            'sl_atr_mult': self._safe_float(
                signal.get('sl_atr_mult'),
                SIGNAL_DEFAULTS['sl_atr_mult'],
                min_val=0.1,
            ),
            'tp_price': self._safe_float(
                signal.get('tp_price'), 0.0,
            ),
            'sl_price': self._safe_float(
                signal.get('sl_price'), 0.0,
            ),
        }

        return validated

    @staticmethod
    def _safe_float(
        value,
        default: float,
        min_val: float = None,
        max_val: float = None,
    ) -> float:
        """Safe float conversion + bounds check."""
        try:
            result = float(value)
            if np.isnan(result) or np.isinf(result):
                return default
        except (TypeError, ValueError):
            return default

        if min_val is not None and result < min_val:
            return default
        if max_val is not None and result > max_val:
            return default

        return result

    # ==========================================================
    # 4h LOOK-AHEAD PROTECTION
    # ==========================================================

    def _get_safe_4h_slice(
        self,
        df_4h: Optional[pd.DataFrame],
        df_primary: pd.DataFrame,
        current_idx: int,
    ) -> Optional[pd.DataFrame]:
        """
        Return only COMPLETED 4h bars.
        """
        if df_4h is None or df_4h.empty:
            return None

        try:
            current_time = df_primary.index[current_idx]
        except (IndexError, KeyError):
            return None

        if not isinstance(current_time, pd.Timestamp):
            try:
                return df_4h[df_4h.index <= current_time]
            except Exception:
                return None

        try:
            # Bar is completed only after 4 hours from its open time
            completed_cutoff = current_time - pd.Timedelta(hours=4)
            safe_4h = df_4h[df_4h.index <= completed_cutoff]

            if safe_4h.empty:
                return None

            return safe_4h

        except Exception:
            return None

    # ==========================================================
    # MTF FEATURE GENERATION
    # ==========================================================

    def _get_htf_context(
        self,
        df_4h: Optional[pd.DataFrame],
        symbol: str,
        cache: Dict,
    ):
        """Manage 4h context with cache."""
        if self.mtf_analyzer is None or df_4h is None or df_4h.empty:
            return None

        # Cache key: time of last 4h bar
        cache_key = str(df_4h.index[-1])
        if cache_key in cache:
            return cache[cache_key]

        try:
            context = self.mtf_analyzer.analyze_htf(df_4h, symbol)
            cache[cache_key] = context
            return context
        except Exception:
            cache[cache_key] = None
            return None

    def _generate_mtf_features(
        self,
        htf_context,
        current_price: float,
        current_atr: float,
    ) -> Dict[str, float]:
        """
        Generate cross-TF features.
        """
        defaults = {
            'htf_bias_numeric': 0.0,
            'htf_trend_strength': 0.0,
            'htf_rsi': 50.0,
            'htf_price_vs_ema200': 0.0,
            'htf_ema_alignment': 0.0,
            'htf_structure_numeric': 0.0,
        }

        if self.mtf_analyzer is None or htf_context is None:
            return defaults

        try:
            features = self.mtf_analyzer.generate_cross_tf_features(
                htf_context, current_price, current_atr
            )
            for k, v in defaults.items():
                if k not in features:
                    features[k] = v
            return features
        except Exception:
            return defaults

    # ==========================================================
    # OUTCOME SIMULATION
    # ==========================================================

    def _simulate_outcome(
        self,
        direction: str,
        entry: float,
        atr_val: float,
        tp_mult: float,
        sl_mult: float,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        start_idx: int,
        N: int,
        timeout_threshold: float = 0.2,
        actual_tp: float = 0.0,
        actual_sl: float = 0.0,
    ) -> Dict:
        """
        Conservative outcome simulation with trailing stop.

        Uses actual TP/SL prices from the strategy when available,
        falls back to ATR multiplier calculation otherwise.
        """
        if direction == 'LONG':
            tp_price = entry + atr_val * tp_mult
            sl_price = entry - atr_val * sl_mult
            # Use actual strategy prices if valid
            if actual_tp > entry:
                tp_price = actual_tp
            if 0 < actual_sl < entry:
                sl_price = actual_sl
        else:
            tp_price = entry - atr_val * tp_mult
            sl_price = entry + atr_val * sl_mult
            # Use actual strategy prices if valid
            if 0 < actual_tp < entry:
                tp_price = actual_tp
            if actual_sl > entry:
                sl_price = actual_sl

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

                trail_level = (
                    entry + (tp_price - entry) * self.trail_activation
                )
                if not trail_activated and best_price >= trail_level:
                    trail_activated = True

                if trail_activated:
                    new_trail = best_price - atr_val * sl_mult
                    if new_trail > trail_sl:
                        trail_sl = new_trail

                active_sl = trail_sl if trail_activated else sl_price

                sl_hit = lo <= active_sl
                tp_hit = h >= tp_price

                if sl_hit and tp_hit:
                    actual_pnl = (active_sl - entry) / atr_val
                    return {
                        'meta_label': 1 if actual_pnl > 0 else 0,
                        'type': 'AMBIGUOUS',
                        'pnl_atr': round(actual_pnl, 4),
                        'bars': j,
                    }

                if sl_hit:
                    actual_pnl = (active_sl - entry) / atr_val
                    return {
                        'meta_label': 1 if actual_pnl > 0 else 0,
                        'type': (
                            'TRAIL_STOP' if trail_activated else 'SL'
                        ),
                        'pnl_atr': round(actual_pnl, 4),
                        'bars': j,
                    }

                if tp_hit:
                    return {
                        'meta_label': 1,
                        'type': 'TP',
                        'pnl_atr': round(tp_mult, 4),
                        'bars': j,
                    }

            else:  # SHORT
                best_price = min(best_price, lo)

                trail_level = (
                    entry - (entry - tp_price) * self.trail_activation
                )
                if not trail_activated and best_price <= trail_level:
                    trail_activated = True

                if trail_activated:
                    new_trail = best_price + atr_val * sl_mult
                    if new_trail < trail_sl:
                        trail_sl = new_trail

                active_sl = trail_sl if trail_activated else sl_price

                sl_hit = h >= active_sl
                tp_hit = lo <= tp_price

                if sl_hit and tp_hit:
                    actual_pnl = (entry - active_sl) / atr_val
                    return {
                        'meta_label': 1 if actual_pnl > 0 else 0,
                        'type': 'AMBIGUOUS',
                        'pnl_atr': round(actual_pnl, 4),
                        'bars': j,
                    }

                if sl_hit:
                    actual_pnl = (entry - active_sl) / atr_val
                    return {
                        'meta_label': 1 if actual_pnl > 0 else 0,
                        'type': (
                            'TRAIL_STOP' if trail_activated else 'SL'
                        ),
                        'pnl_atr': round(actual_pnl, 4),
                        'bars': j,
                    }

                if tp_hit:
                    return {
                        'meta_label': 1,
                        'type': 'TP',
                        'pnl_atr': round(tp_mult, 4),
                        'bars': j,
                    }

        # -- Timeout --
        final_idx = min(start_idx + self.lookahead, N - 1)
        final_close = closes[final_idx]

        if direction == 'LONG':
            pnl_atr = (final_close - entry) / atr_val
        else:
            pnl_atr = (entry - final_close) / atr_val

        return {
            'meta_label': 1 if pnl_atr > timeout_threshold else 0,
            'type': 'TIMEOUT',
            'pnl_atr': round(pnl_atr, 4),
            'bars': self.lookahead,
        }

    # ==========================================================
    # DATAFRAME ASSEMBLY
    # ==========================================================

    def _assemble_dataframe(
        self,
        df: pd.DataFrame,
        results: List[Dict],
    ) -> pd.DataFrame:
        """
        Merge feature DataFrame with meta-label results.
        """
        results_df = pd.DataFrame(results)
        indices = results_df['_bar_index'].values

        # Feature rows
        features_df = df.iloc[indices].copy()
        original_dt_index = features_df.index

        # Column collision control
        result_cols = set(results_df.columns)
        feature_cols = set(features_df.columns)
        collisions = result_cols & feature_cols

        if collisions:
            features_df = features_df.drop(
                columns=list(collisions),
                errors='ignore',
            )

        # DatetimeIndex assignment
        if original_dt_index.duplicated().any():
            results_df = results_df.reset_index(drop=True)
            features_df = features_df.reset_index(drop=True)
            meta_df = pd.concat(
                [features_df, results_df], axis=1
            )
            meta_df['_timestamp'] = original_dt_index.values
        else:
            results_df.index = original_dt_index
            meta_df = pd.concat(
                [features_df, results_df], axis=1
            )

        return meta_df

    # ==========================================================
    # HELPERS
    # ==========================================================

    def _safe_get_atr(self, df: pd.DataFrame) -> np.ndarray:
        """Safe ATR retrieval. Compute if missing."""
        if 'atr' in df.columns:
            atrs = df['atr'].values.copy()
            return np.nan_to_num(atrs, nan=0.0)

        # Fallback calculation
        try:
            h = df['high'].values
            lo = df['low'].values
            c = df['close'].values

            tr = np.maximum(
                h[1:] - lo[1:],
                np.maximum(
                    np.abs(h[1:] - c[:-1]),
                    np.abs(lo[1:] - c[:-1]),
                ),
            )
            tr = np.concatenate([[0.0], tr])
            atr = pd.Series(tr).ewm(
                span=14, adjust=False
            ).mean().values
            return atr
        except Exception:
            return np.zeros(len(df))

    def _safe_precompute(self, df: pd.DataFrame):
        """Safe indicators precompute."""
        try:
            if hasattr(self.engine, 'precompute_indicators'):
                self.engine.precompute_indicators(df)
        except Exception as e:
            warnings.warn(
                f"precompute_indicators error: {e}",
                RuntimeWarning,
            )

    def _print_report(
        self,
        symbol: str,
        stats: Dict,
        elapsed: float,
    ):
        """Reporting."""
        total = stats['signals_generated']
        correct = stats['meta_label_1']
        wrong = stats['meta_label_0']
        wr = (correct / total * 100) if total > 0 else 0
        bars = stats['bars_evaluated']

        print(
            f"  [INFO] {symbol}: "
            f"Bars={bars} | Signals={total} | "
            f"WR={wr:.1f}% ({correct}W {wrong}L) | "
            f"{elapsed:.1f}s"
        )

        for regime, r_stats in stats['by_regime'].items():
            r_total = r_stats['signals']
            r_correct = r_stats['correct']
            r_wr = (r_correct / r_total * 100) if r_total > 0 else 0
            print(
                f"         {regime}: "
                f"{r_total} sigs, WR={r_wr:.0f}%"
            )


# =====================================================================
# Bulk Generation
# =====================================================================

def generate_meta_labels_bulk(
    all_dfs: Dict[str, pd.DataFrame],
    all_dfs_4h: Optional[Dict[str, pd.DataFrame]] = None,
    timeframe: str = "1h",
    **kwargs,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate meta-labels for multiple coins.
    """
    verbose = kwargs.pop('verbose', True)

    if all_dfs_4h is None:
        all_dfs_4h = {}

    all_results = []
    combined_stats = {
        'total_signals': 0,
        'total_correct': 0,
        'total_wrong': 0,
        'total_bars': 0,
        'coins_processed': 0,
        'coins_skipped': 0,
        'per_coin': {},
        'total_time': 0.0,
    }

    total_coins = len(all_dfs)
    total_start = time.time()

    for idx, (symbol, df) in enumerate(all_dfs.items(), 1):
        if verbose:
            print(
                f"\n  [{idx}/{total_coins}] {symbol} "
                f"({len(df)} bars)"
            )

        try:
            # Fresh generator per coin
            generator = MetaLabelGenerator(
                timeframe=timeframe,
                **kwargs,
            )

            df_4h = all_dfs_4h.get(symbol)

            meta_df, stats = generator.generate(
                df,
                df_4h=df_4h,
                symbol=symbol,
                verbose=verbose,
            )

            if meta_df.empty:
                combined_stats['coins_skipped'] += 1
                continue

            meta_df['_symbol'] = symbol
            all_results.append(meta_df)

            combined_stats['coins_processed'] += 1
            combined_stats['total_signals'] += stats.get(
                'signals_generated', 0
            )
            combined_stats['total_correct'] += stats.get(
                'meta_label_1', 0
            )
            combined_stats['total_wrong'] += stats.get(
                'meta_label_0', 0
            )
            combined_stats['total_bars'] += stats.get(
                'bars_evaluated', 0
            )
            combined_stats['per_coin'][symbol] = stats

        except Exception as e:
            print(f"  [WARN] {symbol} meta-label error: {e}")
            combined_stats['coins_skipped'] += 1
            import traceback
            traceback.print_exc()

    combined_stats['total_time'] = round(
        time.time() - total_start, 1
    )

    if not all_results:
        if verbose:
            print("\n  [WARN] No meta-labels generated!")
        return pd.DataFrame(), combined_stats

    combined_df = pd.concat(all_results, ignore_index=False)

    if verbose:
        _print_bulk_summary(combined_stats)

    return combined_df, combined_stats


def _print_bulk_summary(stats: Dict):
    """Bulk summary report."""
    total = stats['total_signals']
    correct = stats['total_correct']
    wr = (correct / total * 100) if total > 0 else 0

    print(f"\n  {'='*55}")
    print(f"  META-LABEL BULK REPORT")
    print(f"  {'='*55}")
    print(f"  Coins processed : {stats['coins_processed']}")
    print(f"  Coins skipped   : {stats['coins_skipped']}")
    print(f"  Total bars      : {stats['total_bars']:,}")
    print(f"  Total signals   : {total:,}")
    print(f"  Overall WR      : {wr:.1f}%")
    print(f"  Correct (1)     : {correct:,}")
    print(f"  Wrong (0)       : {stats['total_wrong']:,}")
    print(f"  Time            : {stats['total_time']:.1f}s")
    print(f"  {'='*55}")
