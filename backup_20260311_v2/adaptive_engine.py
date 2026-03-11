"""
Adaptive Engine v1.3

v1.3 Değişiklikleri:
  - Debug sayaçları ve istatistikleri eklendi
  - Minimum sinyal confidence gevşetildi (0.60 -> 0.50)
  - Meta-confidence çift kontrolü kaldırıldı (threshold yeterli)
  - Position sizing mantığı güncellendi
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from ai.regime_detector import detect_regime, Regime
from ai.strategies import (
    MomentumStrategy, MeanReversionStrategy,
    VolatilityStrategy, ScalpingStrategy, Signal
)

@dataclass
class TradeDecision:
    action: str             # 'LONG', 'SHORT', 'HOLD'
    confidence: float       # 0.0 - 1.0
    position_size: float    # % of capital (0.0 - 1.0)
    regime: str
    primary_signal: Optional[Signal] = None
    meta_confidence: float = 0.0
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.0
    reason: str = ""

class AdaptiveEngine:
    """
    Ana Karar Mekanizması
    """

    def __init__(self, primary_tf: str = "1h", secondary_tf: str = "15m", timeframe: Optional[str] = None):
        if timeframe:
            primary_tf = timeframe
        self.timeframe = primary_tf
        self.secondary_tf = secondary_tf

        self.strategies = {
            Regime.TRENDING: MomentumStrategy(),
            Regime.MEAN_REVERTING: MeanReversionStrategy(),
            Regime.HIGH_VOLATILE: VolatilityStrategy(),
            Regime.LOW_VOLATILE: ScalpingStrategy(),
        }

        self.meta_predictor = None
        try:
            from ai.meta_labelling.meta_predictor import MetaPredictor
            self.meta_predictor = MetaPredictor(timeframes=[primary_tf, secondary_tf])
            if not self.meta_predictor.is_ready:
                print("  WARNING: Meta-model bulunamadi, filtresiz calisilacak")
                self.meta_predictor = None
        except Exception as e:
            print(f"  ERROR: MetaPredictor yuklenemedi: {e}")
            self.meta_predictor = None

        # ── v1.3: Debug sayaçları ────────────────────────────
        self._debug_counts = {
            'total_calls': 0,
            'no_strategy': 0,
            'no_signal': 0,
            'low_confidence': 0,
            'meta_rejected': 0,
            'meta_low': 0,
            'passed': 0,
            'regime_counts': {},
            'reject_reasons': {},
        }

    def decide(self, df: pd.DataFrame, df_secondary: Optional[pd.DataFrame] = None) -> TradeDecision:
        """
        Karar Mekanizması - Hybrid v1.5
        df: Ana timeframe (orn 1h)
        df_secondary: Ikincil timeframe (orn 15m) - Trending rejiminde kullanilir
        """
        self._debug_counts['total_calls'] += 1

        # Katman 1: Rejim Tespiti (Stable 1H clock)
        regime, regime_details = detect_regime(df)
        rv = regime.value
        self._debug_counts['regime_counts'][rv] = \
            self._debug_counts['regime_counts'].get(rv, 0) + 1

        # ── v1.5: Hybrid Timeframe Switching ────────────────
        active_df = df
        active_tf = self.timeframe

        # Rejim TRENDING ise ve ikincil veri varsa 15m'ye gec (AUC avantajı)
        if regime == Regime.TRENDING and df_secondary is not None:
            active_df = df_secondary
            active_tf = self.secondary_tf
            # print(f"  [HYBRID] Trending rejiminde {active_tf} timeframe'e gecildi.")

        # Katman 2: Primary Strateji Sinyali (Active TF uzerinde)
        strategy = self.strategies.get(regime)
        if strategy is None:
            self._debug_counts['no_strategy'] += 1
            return self._hold_decision(regime, "Strateji bulunamadı")

        primary_signal = strategy.generate_signal(active_df)

        # ── v1.4: Birincil strateji sinyal vermezse ─────────
        if not primary_signal.is_valid:
            primary_signal = self._try_secondary_strategy(active_df, regime)

        if not primary_signal.is_valid:
            self._debug_counts['no_signal'] += 1
            self._track_reason(f"no_signal_{rv}")
            return self._hold_decision(
                regime, f"Hicbir strateji sinyal uretmedi ({rv})")

        # ── v1.4: Backtest uyumluluğu için sinyali sakla ─────
        self._last_signal = primary_signal

        # ── v1.5: Meta-Filter (Timeframe Aware) ─────────────
        meta_conf = primary_signal.confidence * 0.85

        if self.meta_predictor is not None:
            meta_conf, should_trade, threshold, meta_reason = self.meta_predictor.predict(
                df=active_df,
                regime=regime,
                signal_direction=primary_signal.direction,
                signal_confidence=primary_signal.confidence,
                timeframe=active_tf
            )

            if not should_trade:
                self._debug_counts['meta_rejected'] += 1
                self._track_reason(f"meta_reject_{rv}_{meta_conf:.2f}<{threshold:.2f}")
                return self._hold_decision(
                    regime,
                    f"Meta-filter rejected ({meta_reason}): {meta_conf:.2f} < {threshold:.2f}")

        # ── v1.3: Meta conf çift kontrol KALDIRILDI ─────────
        # (meta_predictor zaten threshold kontrolü yapıyor)

        # Risk Yönetimi (Position Sizing)
        position_size = self._calculate_position_size(
            primary_signal.confidence, meta_conf, regime
        )

        # Final Güven Skoru (Ağırlıklı Ortalama)
        final_confidence = (primary_signal.confidence * 0.3 + meta_conf * 0.7)

        self._debug_counts['passed'] += 1

        return TradeDecision(
            action=primary_signal.direction,
            confidence=final_confidence,
            position_size=position_size,
            regime=regime.value,
            primary_signal=primary_signal,
            meta_confidence=meta_conf,
            tp_atr_mult=primary_signal.tp_atr_mult,
            sl_atr_mult=primary_signal.sl_atr_mult,
            reason=(f"{regime.value} | {primary_signal.reason} | "
                    f"meta={meta_conf:.2f} size={position_size:.0%}"),
        )

    def _try_secondary_strategy(self, df: pd.DataFrame, primary_regime: Regime) -> Signal:
        """
        Birincil strateji sinyal vermezse diğer stratejileri dene.
        Ama confidence'ı düşür (birincil rejim değil).
        """
        # Denenecek strateji sırası (birincil hariç)
        secondary_order = {
            Regime.MEAN_REVERTING: [Regime.TRENDING, Regime.LOW_VOLATILE],
            Regime.TRENDING: [Regime.MEAN_REVERTING, Regime.HIGH_VOLATILE],
            Regime.HIGH_VOLATILE: [Regime.TRENDING, Regime.MEAN_REVERTING],
            Regime.LOW_VOLATILE: [Regime.MEAN_REVERTING, Regime.TRENDING],
        }

        for alt_regime in secondary_order.get(primary_regime, []):
            alt_strategy = self.strategies.get(alt_regime)
            if alt_strategy is None:
                continue

            signal = alt_strategy.generate_signal(df)
            if signal.is_valid:
                # Confidence'ı %20 düşür (yanlış rejimde çalışıyor)
                adjusted_conf = signal.confidence * 0.80
                return Signal(
                    direction=signal.direction,
                    confidence=adjusted_conf,
                    strategy_name=f"{signal.strategy_name}(secondary)",
                    regime=primary_regime.value,
                    reason=f"[2nd] {signal.reason}",
                    entry_price=signal.entry_price,
                    tp_atr_mult=signal.tp_atr_mult,
                    sl_atr_mult=signal.sl_atr_mult,
                )

        return Signal(
            direction=None,
            confidence=0.0,
            strategy_name="none",
            regime=primary_regime.value,
            reason="Ikincil stratejiler de sinyal uretmedi",
        )

    def _track_reason(self, reason: str):
        key = reason[:50]
        self._debug_counts['reject_reasons'][key] = \
            self._debug_counts['reject_reasons'].get(key, 0) + 1

    def _calculate_position_size(
        self, signal_conf: float, meta_conf: float, regime: Regime
    ) -> float:
        """v1.3: Position sizing mantığı"""
        combined = signal_conf * 0.3 + meta_conf * 0.7
        regime_max = {
            Regime.TRENDING: 1.0,
            Regime.MEAN_REVERTING: 0.80,
            Regime.HIGH_VOLATILE: 0.50,
            Regime.LOW_VOLATILE: 0.60,
        }
        max_size = regime_max.get(regime, 0.50)

        if combined >= 0.75:
            size = 1.0
        elif combined >= 0.65:
            size = 0.75
        elif combined >= 0.55:
            size = 0.50
        elif combined >= 0.45:
            size = 0.25
        else:
            size = 0.10  # Minimum pozisyon

        return min(size, max_size)

    def _hold_decision(self, regime, reason: str) -> TradeDecision:
        regime_val = regime.value if isinstance(regime, Regime) else str(regime)
        return TradeDecision(
            action='HOLD',
            confidence=0.0,
            position_size=0.0,
            regime=regime_val,
            primary_signal=None,
            meta_confidence=0.0,
            tp_atr_mult=0.0,
            sl_atr_mult=0.0,
            reason=f"HOLD [{reason}]",
        )

    def get_status(self) -> dict:
        return {
            "timeframe": self.timeframe,
            "meta_models_loaded": {
                r.value: (r.value in self.meta_predictor.models) if self.meta_predictor else False
                for r in Regime
            },
            "strategies": {
                r.value: self.strategies[r].name for r in self.strategies
            },
        }

    def print_debug_stats(self):
        """Debug istatistiklerini yazdır"""
        d = self._debug_counts
        total = d['total_calls']
        if total == 0:
            print("  📊 Debug: Henüz çağrı yok")
            return

        print(f"\n  {'-' * 50}")
        print(f"  ADAPTIVE ENGINE DEBUG STATS")
        print(f"  {'-' * 50}")
        print(f"  Total calls    : {total:,}")
        print(f"  No strategy    : {d['no_strategy']:,} "
              f"(%{d['no_strategy']/total*100:.1f})")
        print(f"  No signal      : {d['no_signal']:,} "
              f"(%{d['no_signal']/total*100:.1f})")
        print(f"  Low confidence : {d['low_confidence']:,} "
              f"(%{d['low_confidence']/total*100:.1f})")
        print(f"  Meta rejected  : {d['meta_rejected']:,} "
              f"(%{d['meta_rejected']/total*100:.1f})")
        print(f"  Meta low       : {d['meta_low']:,} "
              f"(%{d['meta_low']/total*100:.1f})")
        print(f"  ✅ PASSED      : {d['passed']:,} "
              f"(%{d['passed']/total*100:.1f})")

        print(f"\n  Rejim dağılımı:")
        for r, c in sorted(d['regime_counts'].items(), key=lambda x: -x[1]):
            print(f"    {r:16}: {c:,} (%{c/total*100:.1f})")

        if d['reject_reasons']:
            print(f"\n  Top red nedenleri:")
            sorted_reasons = sorted(d['reject_reasons'].items(),
                                    key=lambda x: -x[1])[:10]
            for reason, count in sorted_reasons:
                print(f"    {count:5,}x | {reason}")

        print(f"  {'-' * 50}\n")
