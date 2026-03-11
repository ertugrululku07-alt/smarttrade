"""
Meta-Label Generator v1.0

Marcos López de Prado yaklaşımı:
  1. Primary model (rule-based) sinyal üretir
  2. Sinyalin sonradan DOĞRU çıkıp çıkmadığını belirle
  3. meta_label = 1 (sinyal doğru, TP vurdu) / 0 (sinyal yanlış, SL vurdu)

Bu dosya TÜM veri üzerinde çalışarak eğitim verisi hazırlar.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.regime_detector import Regime, detect_regime
from ai.strategies import Signal
from ai.adaptive_engine import AdaptiveEngine


@dataclass
class MetaLabelRow:
    """Tek bir meta-label kaydı"""
    bar_index: int
    regime: str
    signal_direction: str     # 'LONG' or 'SHORT'
    signal_confidence: float
    strategy_name: str
    signal_reason: str
    meta_label: int           # 1=doğru çıktı, 0=yanlış çıktı
    outcome_type: str         # 'TP', 'SL', 'TRAIL', 'TIMEOUT'
    pnl_atr: float            # ATR cinsinden PnL
    bars_to_outcome: int      # Sonuca kaç bar sürdü


class MetaLabelGenerator:
    """
    Tüm veri üzerinde:
      1. Her bar'da rejim tespit et
      2. Rejime uygun strateji ile sinyal üret
      3. Sinyalin sonucunu simüle et (TP/SL)
      4. meta_label oluştur
    """

    def __init__(
        self,
        lookahead: int = 16,
        trail_activation: float = 0.6,
        regime_lookback: int = 50,
        timeframe: str = "1h",
    ):
        self.lookahead = lookahead
        self.trail_activation = trail_activation
        self.regime_lookback = regime_lookback
        self.timeframe = timeframe

        # ── v1.4: Engine-v1.4 ─────────────────────────────
        self.engine = AdaptiveEngine(timeframe=timeframe)
        self.engine.meta_predictor = None  # Filtresiz sinyal istiyoruz

    def generate(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Tüm veri üzerinde meta-label üret.

        Returns:
            (meta_df, stats_dict)

            meta_df kolonları:
              - Tüm orijinal feature'lar
              - signal_is_long: int
              - signal_confidence: float
              - regime_*: one-hot encoded rejim
              - meta_label: 0/1 (hedef değişken)
              - _regime: str (debug)
              - _signal_dir: str (debug)
              - _outcome: str (debug)
              - _symbol: str
        """
        N = len(df)
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        atrs = df['atr'].values

        results = []
        stats = {
            'total_bars': N,
            'signals_generated': 0,
            'meta_label_1': 0,    # Doğru sinyaller
            'meta_label_0': 0,    # Yanlış sinyaller
            'no_signal': 0,
            'regime_counts': {r.value: 0 for r in Regime},
            'strategy_counts': {},
            'by_regime': {
                r.value: {'signals': 0, 'correct': 0, 'wrong': 0}
                for r in Regime
            },
        }

        # ── Her bar için sinyal üret ve sonucunu kontrol et ──
        for i in range(self.regime_lookback, N - self.lookahead):
            atr_val = atrs[i]
            if np.isnan(atr_val) or atr_val < 1e-10:
                continue

            entry = closes[i]

            # ── Katman 1: Rejim Tespiti ──────────────────────
            window = df.iloc[max(0, i - self.regime_lookback):i + 1]
            regime, _ = detect_regime(window, lookback=self.regime_lookback)
            stats['regime_counts'][regime.value] += 1

            # ── Katman 2: v1.4 Engine Kararı ──────────────────
            decision = self.engine.decide(df.iloc[:i + 1])
            
            # Rejim istatistiği güncelle
            regime_str = decision.regime
            stats['regime_counts'][regime_str] += 1
            
            if decision.action == 'HOLD':
                stats['no_signal'] += 1
                continue

            # Engine içindeki ham sinyali al
            signal = getattr(self.engine, '_last_signal', None)
            if not signal:
                stats['no_signal'] += 1
                continue

            stats['signals_generated'] += 1
            stats['by_regime'][regime_str]['signals'] += 1

            sname = signal.strategy_name
            stats['strategy_counts'][sname] = stats['strategy_counts'].get(sname, 0) + 1

            # ── Katman 3: Sinyal Sonucunu Simüle Et ──────────
            tp_mult = signal.tp_atr_mult
            sl_mult = signal.sl_atr_mult

            outcome = self._simulate_outcome(
                direction=signal.direction,
                entry=entry,
                atr_val=atr_val,
                tp_mult=tp_mult,
                sl_mult=sl_mult,
                highs=highs,
                lows=lows,
                closes=closes,
                start_idx=i,
                N=N,
            )

            meta_label = outcome['meta_label']

            if meta_label == 1:
                stats['meta_label_1'] += 1
                stats['by_regime'][regime.value]['correct'] += 1
            else:
                stats['meta_label_0'] += 1
                stats['by_regime'][regime.value]['wrong'] += 1

            # ── Sonuç kaydı ──────────────────────────────────
            results.append({
                '_bar_index': i,
                '_regime': regime_str,
                '_signal_dir': signal.direction,
                '_signal_conf': signal.confidence,
                '_strategy': signal.strategy_name,
                '_reason': signal.reason,
                '_outcome': outcome['type'],
                '_pnl_atr': outcome['pnl_atr'],
                '_bars_to_outcome': outcome['bars'],
                '_symbol': symbol,
                # Meta features
                'signal_is_long': int(signal.direction == 'LONG'),
                'signal_confidence': signal.confidence,
                'regime_trending': int(regime_str == Regime.TRENDING.value),
                'regime_mean_rev': int(regime_str == Regime.MEAN_REVERTING.value),
                'regime_high_vol': int(regime_str == Regime.HIGH_VOLATILE.value),
                'regime_low_vol': int(regime_str == Regime.LOW_VOLATILE.value),
                # Hedef
                'meta_label': meta_label,
            })

        if not results:
            if verbose:
                print(f"   ⚠️ {symbol}: Hiç sinyal üretilemedi")
            return pd.DataFrame(), stats

        # ── Results'ı DataFrame'e çevir ──────────────────────
        results_df = pd.DataFrame(results)

        # Orijinal feature'ları ekle
        feature_rows = []
        for _, row in results_df.iterrows():
            idx = int(row['_bar_index'])
            feat_row = df.iloc[idx].to_dict()
            feature_rows.append(feat_row)

        features_df = pd.DataFrame(feature_rows, index=results_df.index)

        # Birleştir
        meta_df = pd.concat([features_df, results_df], axis=1)

        # ── Debug Log ────────────────────────────────────────
        if verbose:
            total_sig = stats['signals_generated']
            correct = stats['meta_label_1']
            wrong = stats['meta_label_0']
            wr = correct / total_sig * 100 if total_sig > 0 else 0

            print(f"   📊 {symbol}: Signals={total_sig} | "
                  f"Correct={correct} Wrong={wrong} | "
                  f"Raw WR=%{wr:.1f} | "
                  f"Regimes: {stats['regime_counts']}")

        return meta_df, stats

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
    ) -> Dict:
        """
        Sinyalin TP/SL sonucunu simüle et.

        Returns:
            {
                'meta_label': 0 or 1,
                'type': 'TP', 'SL', 'TRAIL', 'TIMEOUT',
                'pnl_atr': float,
                'bars': int,
            }
        """
        if direction == 'LONG':
            tp_price = entry + atr_val * tp_mult
            sl_price = entry - atr_val * sl_mult
        else:  # SHORT
            tp_price = entry - atr_val * tp_mult
            sl_price = entry + atr_val * sl_mult

        # Trailing stop state
        best_price = entry
        trail_sl = sl_price
        trail_activated = False

        for j in range(1, self.lookahead + 1):
            idx = start_idx + j
            if idx >= N:
                break

            h = highs[idx]
            lo = lows[idx]
            c = closes[idx]

            if direction == 'LONG':
                # Track best price
                if h > best_price:
                    best_price = h

                # Trailing stop activation
                trail_level = entry + (tp_price - entry) * self.trail_activation
                if best_price >= trail_level and not trail_activated:
                    trail_activated = True

                if trail_activated:
                    new_trail = best_price - atr_val * sl_mult
                    trail_sl = max(trail_sl, new_trail)
                    if lo <= trail_sl:
                        # Trail stop hit — kârlı çıkış
                        pnl = (trail_sl - entry) / atr_val
                        return {
                            'meta_label': 1 if pnl > 0 else 0,
                            'type': 'TRAIL',
                            'pnl_atr': round(pnl, 4),
                            'bars': j,
                        }

                # Normal TP
                if h >= tp_price:
                    return {
                        'meta_label': 1,
                        'type': 'TP',
                        'pnl_atr': round(tp_mult, 4),
                        'bars': j,
                    }

                # Normal SL
                if lo <= sl_price:
                    return {
                        'meta_label': 0,
                        'type': 'SL',
                        'pnl_atr': round(-sl_mult, 4),
                        'bars': j,
                    }

            else:  # SHORT
                if lo < best_price:
                    best_price = lo

                trail_level = entry - (entry - tp_price) * self.trail_activation
                if best_price <= trail_level and not trail_activated:
                    trail_activated = True

                if trail_activated:
                    new_trail = best_price + atr_val * sl_mult
                    trail_sl = min(trail_sl, new_trail)
                    if h >= trail_sl:
                        pnl = (entry - trail_sl) / atr_val
                        return {
                            'meta_label': 1 if pnl > 0 else 0,
                            'type': 'TRAIL',
                            'pnl_atr': round(pnl, 4),
                            'bars': j,
                        }

                if lo <= tp_price:
                    return {
                        'meta_label': 1,
                        'type': 'TP',
                        'pnl_atr': round(tp_mult, 4),
                        'bars': j,
                    }

                if h >= sl_price:
                    return {
                        'meta_label': 0,
                        'type': 'SL',
                        'pnl_atr': round(-sl_mult, 4),
                        'bars': j,
                    }

        # Timeout — son kapanışa göre
        final_close = closes[min(start_idx + self.lookahead, N - 1)]
        if direction == 'LONG':
            pnl = (final_close - entry) / atr_val
        else:
            pnl = (entry - final_close) / atr_val

        return {
            'meta_label': 1 if pnl > 0 else 0,
            'type': 'TIMEOUT',
            'pnl_atr': round(pnl, 4),
            'bars': self.lookahead,
        }


# ═══════════════════════════════════════════════════════════════════
# Toplu Meta-Label Üretimi (Çoklu Coin)
# ═══════════════════════════════════════════════════════════════════

def generate_meta_labels_bulk(
    all_dfs: Dict[str, pd.DataFrame],
    lookahead: int = 16,
    trail_activation: float = 0.6,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Birden fazla coin için meta-label üret.

    Args:
        all_dfs: {symbol: df} dict'i (feature'ları hesaplanmış)

    Returns:
        (combined_meta_df, combined_stats)
    """
    generator = MetaLabelGenerator(
        lookahead=lookahead,
        trail_activation=trail_activation,
    )

    all_meta_dfs = []
    combined_stats = {
        'total_signals': 0,
        'total_correct': 0,
        'total_wrong': 0,
        'coins_processed': 0,
        'coins_skipped': 0,
        'by_regime': {r.value: {'signals': 0, 'correct': 0, 'wrong': 0}
                      for r in Regime},
    }

    for symbol, df in all_dfs.items():
        try:
            meta_df, stats = generator.generate(df, symbol=symbol, verbose=verbose)

            if meta_df.empty:
                combined_stats['coins_skipped'] += 1
                continue

            all_meta_dfs.append(meta_df)
            combined_stats['coins_processed'] += 1
            combined_stats['total_signals'] += stats['signals_generated']
            combined_stats['total_correct'] += stats['meta_label_1']
            combined_stats['total_wrong'] += stats['meta_label_0']

            for regime_val in [r.value for r in Regime]:
                for key in ['signals', 'correct', 'wrong']:
                    combined_stats['by_regime'][regime_val][key] += \
                        stats['by_regime'][regime_val][key]

        except Exception as e:
            print(f"   ⚠️ {symbol} meta-label hatası: {e}")
            combined_stats['coins_skipped'] += 1

    if not all_meta_dfs:
        return pd.DataFrame(), combined_stats

    combined_df = pd.concat(all_meta_dfs, ignore_index=True)

    # ── Özet Rapor ───────────────────────────────────────────
    if verbose:
        total = combined_stats['total_signals']
        correct = combined_stats['total_correct']
        wrong = combined_stats['total_wrong']
        wr = correct / total * 100 if total > 0 else 0

        print(f"\n{'─' * 55}")
        print(f"  📊 META-LABEL ÖZETİ")
        print(f"{'─' * 55}")
        print(f"  Coins     : {combined_stats['coins_processed']} "
              f"(skipped: {combined_stats['coins_skipped']})")
        print(f"  Signals   : {total:,}")
        print(f"  Correct   : {correct:,} (%{wr:.1f})")
        print(f"  Wrong     : {wrong:,} (%{100 - wr:.1f})")
        print(f"  Raw WR    : %{wr:.1f}")
        print(f"{'─' * 55}")

        for regime_val in [r.value for r in Regime]:
            rd = combined_stats['by_regime'][regime_val]
            rs = rd['signals']
            rc = rd['correct']
            rw = rd['wrong']
            rwr = rc / rs * 100 if rs > 0 else 0
            print(f"  {regime_val:16}: Sig={rs:,} "
                  f"WR=%{rwr:.1f} (C={rc} W={rw})")

        print(f"{'─' * 55}\n")

    return combined_df, combined_stats
