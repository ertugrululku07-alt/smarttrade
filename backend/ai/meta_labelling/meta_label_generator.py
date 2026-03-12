"""
Meta-Label Generator v1.2 (Hardened)

Marcos López de Prado yaklaşımı:
  1. Primary model (rule-based) sinyal üretir
  2. Sinyalin sonradan DOĞRU çıkıp çıkmadığını belirle
  3. meta_label = 1 (sinyal doğru) / 0 (sinyal yanlış)

v1.2 Düzeltmeleri:
  - use_meta_filter=False (hack yerine explicit API)
  - Regime one-hot encoding case mismatch düzeltildi
  - _last_signal stale data koruması (engine tarafında)
  - generate_meta_labels_bulk verbose hatası giderildi
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import datetime
import time


@dataclass
class MetaLabelRow:
    """Tek bir meta-label kaydı için veri yapısı"""
    bar_index: int
    regime: str
    signal_direction: str
    meta_label: int
    outcome_type: str
    pnl_atr: float
    bars_to_outcome: int


class MetaLabelGenerator:
    """
    Strateji sinyallerinin başarısını simüle ederek Meta-Model (XGBoost) için
    etiketli eğitim verisi üretir.
    """

    # Regime enum değerleri (lowercase) — detect_regime ve engine bu formatta döndürür
    REGIME_VALUES = ['trending', 'mean_reverting', 'high_volatile', 'low_volatile']

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

        # ✅ Fix: use_meta_filter=False ile explicit API kullanımı
        from ai.adaptive_engine import AdaptiveEngine
        self.engine = AdaptiveEngine(
            timeframe=timeframe,
            use_meta_filter=False,
        )

    def generate(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Tüm veri seti üzerinde gezinerek meta-label üretir.

        Returns:
            (meta_df, stats_dict)
            meta_df: DatetimeIndex korunmuş, feature + meta-label DataFrame
        """
        N = len(df)
        if N < self.regime_lookback + self.lookahead:
            return pd.DataFrame(), {}

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        atrs = df['atr'].values

        results = []
        stats = {
            'signals_generated': 0,
            'meta_label_1': 0,
            'meta_label_0': 0,
            'regime_counts': {},
            'by_regime': {},
        }

        # Performans: Engine indikatörlerini bir kez ön hesapla
        if hasattr(self.engine, 'precompute_indicators'):
            self.engine.precompute_indicators(df)

        # Sabit pencereli slice (O(N) toplam)
        engine_lookback = max(300, self.regime_lookback)

        for i in range(self.regime_lookback, N - self.lookahead):
            atr_val = atrs[i]
            if np.isnan(atr_val) or atr_val < 1e-10:
                continue

            # Sabit pencereli engine çağrısı
            window_start = max(0, i - engine_lookback)
            decision = self.engine.decide(df.iloc[window_start:i + 1])
            regime_str = decision.regime  # "trending", "mean_reverting", vb.

            # İstatistik güncelleme (tek kaynak, duplike yok)
            stats['regime_counts'][regime_str] = (
                stats['regime_counts'].get(regime_str, 0) + 1
            )
            if regime_str not in stats['by_regime']:
                stats['by_regime'][regime_str] = {
                    'signals': 0, 'correct': 0, 'wrong': 0
                }

            if decision.action == 'HOLD':
                continue

            # Explicit Signal API
            signal = self.engine.get_last_signal_info()
            if not signal:
                continue

            stats['signals_generated'] += 1
            stats['by_regime'][regime_str]['signals'] += 1

            # Conservative Outcome Simulation
            outcome = self._simulate_outcome_conservative(
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
            )

            meta_label = outcome['meta_label']
            if meta_label == 1:
                stats['meta_label_1'] += 1
                stats['by_regime'][regime_str]['correct'] += 1
            else:
                stats['meta_label_0'] += 1
                stats['by_regime'][regime_str]['wrong'] += 1

            # ✅ Fix: One-hot encoding — REGIME_VALUES ile tutarlı (lowercase)
            results.append({
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
                    for rv in self.REGIME_VALUES
                },
            })

        if not results:
            return pd.DataFrame(), stats

        # DatetimeIndex'i koru (WFV Trainer uyumluluğu)
        results_df = pd.DataFrame(results)
        indices = results_df['_bar_index'].values

        # Orijinal DatetimeIndex'i taşıyan feature satırları
        features_df = df.iloc[indices].copy()
        original_dt_index = features_df.index

        # results_df'e aynı DatetimeIndex'i ata
        results_df.index = original_dt_index

        # Birleştir
        meta_df = pd.concat([features_df, results_df], axis=1)

        if verbose:
            self._print_report(symbol, stats)

        return meta_df, stats

    def _simulate_outcome_conservative(
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
        Conservative simülasyon:
        - Trailing stop desteği
        - Aynı bar'da hem TP hem SL varsa gerçek PnL'ye göre karar verir
        """
        if direction == 'LONG':
            tp_price = entry + atr_val * tp_mult
            sl_price = entry - atr_val * sl_mult
        else:
            tp_price = entry - atr_val * tp_mult
            sl_price = entry + atr_val * sl_mult

        trail_activated = False
        trail_sl = sl_price
        best_price = entry

        for j in range(1, self.lookahead + 1):
            idx = start_idx + j
            if idx >= N:
                break

            h, lo = highs[idx], lows[idx]

            if direction == 'LONG':
                best_price = max(best_price, h)

                # Trailing stop activation
                trail_level = (
                    entry + (tp_price - entry) * self.trail_activation
                )
                if not trail_activated and best_price >= trail_level:
                    trail_activated = True

                if trail_activated:
                    new_trail = best_price - atr_val * sl_mult
                    trail_sl = max(trail_sl, new_trail)

                # Aktif stop seviyesi
                active_sl = trail_sl if trail_activated else sl_price

                # Hit detection
                sl_hit = lo <= active_sl
                tp_hit = h >= tp_price

                # Ambiguous: Gerçek PnL'ye göre karar
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
                        'type': 'TRAIL_STOP' if trail_activated else 'SL',
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
                    trail_sl = min(trail_sl, new_trail)

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
                        'type': 'TRAIL_STOP' if trail_activated else 'SL',
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

        # Timeout
        final_close = closes[min(start_idx + self.lookahead, N - 1)]
        if direction == 'LONG':
            pnl_atr = (final_close - entry) / atr_val
        else:
            pnl_atr = (entry - final_close) / atr_val

        return {
            'meta_label': 1 if pnl_atr > 0.2 else 0,
            'type': 'TIMEOUT',
            'pnl_atr': round(pnl_atr, 4),
            'bars': self.lookahead,
        }

    def _print_report(self, symbol: str, stats: Dict):
        total = stats['signals_generated']
        correct = stats['meta_label_1']
        wr = (correct / total * 100) if total > 0 else 0
        print(
            f" 📊 {symbol}: Signals={total} | "
            f"WR=%{wr:.1f} | Regimes={stats['regime_counts']}"
        )


# ═══════════════════════════════════════════════════════════════════
# Toplu Meta-Label Üretimi (Çoklu Coin)
# ═══════════════════════════════════════════════════════════════════

def generate_meta_labels_bulk(
    all_dfs: Dict[str, pd.DataFrame],
    timeframe: str = "1h",
    **kwargs,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Birden fazla coin için meta-label üret.

    Args:
        all_dfs: {symbol: df} dict'i (feature'ları hesaplanmış)
        timeframe: Zaman dilimi
        **kwargs: MetaLabelGenerator'a iletilecek ek parametreler + verbose flag'i

    Returns:
        (combined_meta_df, combined_stats)
    """
    # ✅ Fix: verbose parametresini MetaLabelGenerator'a gitmeden ayıklıyoruz
    verbose = kwargs.pop('verbose', True)
    generator = MetaLabelGenerator(timeframe=timeframe, **kwargs)

    all_results = []
    combined_stats = {
        'total_signals': 0,
        'total_correct': 0,
        'total_wrong': 0,
        'coins_processed': 0,
        'coins_skipped': 0,
    }

    for symbol, df in all_dfs.items():
        try:
            # ✅ Fix: verbose flag'ini generate metoduna güvenli bir şekilde gönderiyoruz
            meta_df, stats = generator.generate(df, symbol=symbol, verbose=verbose)

            if meta_df.empty:
                combined_stats['coins_skipped'] += 1
                continue

            meta_df['_symbol'] = symbol
            all_results.append(meta_df)

            combined_stats['coins_processed'] += 1
            combined_stats['total_signals'] += stats['signals_generated']
            combined_stats['total_correct'] += stats['meta_label_1']
            combined_stats['total_wrong'] += stats['meta_label_0']

        except Exception as e:
            print(f" ⚠️ {symbol} meta-label error: {e}")
            combined_stats['coins_skipped'] += 1

    if not all_results:
        return pd.DataFrame(), combined_stats

    combined_df = pd.concat(all_results, ignore_index=False)

    return combined_df, combined_stats
