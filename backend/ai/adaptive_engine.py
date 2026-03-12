"""
Adaptive Engine v1.4

v1.4 Değişiklikleri (v1.3 üzeri):
  - use_meta_filter parametrersi eklendi (MetaLabelGenerator uyumu)
  - _last_signal her decide() başında temizleniyor (stale data önlemi)
  - Kullanılmayan _ict_alt kaldırıldı
  - Signal field'larına güvenli erişim (getattr fallback)
  - get_status meta_predictor.models güvenli erişim
  - precompute_indicators stub eklendi
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from ai.regime_detector import detect_regime, Regime
from ai.strategies import (
    MomentumStrategy, MeanReversionStrategy,
    ScalpingStrategy, Signal,
    ICTStrategy,
)


@dataclass
class TradeDecision:
    action: str              # 'LONG', 'SHORT', 'HOLD'
    confidence: float        # 0.0 - 1.0
    position_size: float     # % of capital (0.0 - 1.0)
    regime: str
    primary_signal: Optional[Signal] = None
    meta_confidence: float = 0.0
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.0
    reason: str = ""
    soft_score: int = 0
    entry_type: str = "none"
    tp_price: float = 0.0
    sl_price: float = 0.0


class AdaptiveEngine:
    """
    Ana Karar Mekanizması — Rejim tespiti + Strateji seçimi + Meta filtre
    """

    def __init__(
        self,
        primary_tf: str = "1h",
        secondary_tf: str = "15m",
        timeframe: Optional[str] = None,
        use_meta_filter: bool = True,
    ):
        if timeframe:
            primary_tf = timeframe
        self.timeframe = primary_tf
        self.secondary_tf = secondary_tf

        self.strategies = {
            Regime.TRENDING: MomentumStrategy(),
            Regime.MEAN_REVERTING: MeanReversionStrategy(),
            Regime.HIGH_VOLATILE: ICTStrategy(),
            Regime.LOW_VOLATILE: ScalpingStrategy(),
        }

        # Son sinyal referansı (get_last_signal_info için)
        self._last_signal: Optional[Signal] = None

        # ── Meta Predictor (opsiyonel filtre) ────────────────
        self.meta_predictor = None

        if use_meta_filter:
            try:
                from ai.meta_labelling.meta_predictor import MetaPredictor
                self.meta_predictor = MetaPredictor(
                    timeframes=[primary_tf, secondary_tf]
                )
                if not self.meta_predictor.is_ready:
                    print(
                        "  WARNING: Meta-model bulunamadi, "
                        "filtresiz calisilacak"
                    )
                    self.meta_predictor = None
            except Exception as e:
                print(f"  ERROR: MetaPredictor yuklenemedi: {e}")
                self.meta_predictor = None

        # ── Debug sayaçları ──────────────────────────────────
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

    # ──────────────────────────────────────────────────────────
    # Precompute Hook (MetaLabelGenerator uyumu)
    # ──────────────────────────────────────────────────────────

    def precompute_indicators(self, df: pd.DataFrame):
        """
        MetaLabelGenerator tarafından çağrılabilir.
        Stratejilerin ön hesaplama yapması gerekiyorsa burada tetiklenir.
        Şu an stratejiler stateless olduğu için no-op.
        """
        pass

    # ──────────────────────────────────────────────────────────
    # Ana Karar Mekanizması
    # ──────────────────────────────────────────────────────────

    def decide(
        self,
        df: pd.DataFrame,
        df_secondary: Optional[pd.DataFrame] = None,
    ) -> TradeDecision:
        """
        Karar Mekanizması - Hybrid v1.5

        Args:
            df: Ana timeframe verisi (ör. 1h)
            df_secondary: İkincil timeframe (ör. 15m) — Trending'de kullanılır

        Returns:
            TradeDecision
        """
        self._debug_counts['total_calls'] += 1

        # ✅ Fix: Her karar döngüsünde eski sinyali temizle
        # (stale _last_signal verisi MetaLabelGenerator'ı yanıltmasın)
        self._last_signal = None

        # Katman 1: Rejim Tespiti
        regime, regime_details = detect_regime(df)
        rv = regime.value
        self._debug_counts['regime_counts'][rv] = (
            self._debug_counts['regime_counts'].get(rv, 0) + 1
        )

        # ── Hybrid Timeframe Switching ───────────────────────
        active_df = df
        active_tf = self.timeframe

        if regime == Regime.TRENDING and df_secondary is not None:
            active_df = df_secondary
            active_tf = self.secondary_tf

        # Katman 2: Primary Strateji Sinyali
        strategy = self.strategies.get(regime)
        if strategy is None:
            self._debug_counts['no_strategy'] += 1
            return self._hold_decision(regime, "Strateji bulunamadı")

        primary_signal = strategy.generate_signal(active_df)

        # Birincil strateji sinyal vermezse alternatif dene
        if not primary_signal.is_valid:
            primary_signal = self._try_secondary_strategy(
                active_df, regime
            )

        if not primary_signal.is_valid:
            self._debug_counts['no_signal'] += 1
            self._track_reason(f"{primary_signal.reason}_{rv}")
            return self._hold_decision(
                regime,
                f"Signal rejected ({primary_signal.reason})",
            )

        # ✅ Fix: Geçerli sinyali sakla (get_last_signal_info için)
        self._last_signal = primary_signal

        # Katman 3: Meta-Filter (opsiyonel)
        meta_conf = primary_signal.confidence * 0.85

        if self.meta_predictor is not None:
            meta_conf, should_trade, threshold, meta_reason = (
                self.meta_predictor.predict(
                    df=active_df,
                    regime=regime,
                    signal_direction=primary_signal.direction,
                    signal_confidence=primary_signal.confidence,
                    timeframe=active_tf,
                )
            )

            if not should_trade:
                self._debug_counts['meta_rejected'] += 1
                self._track_reason(
                    f"meta_reject_{rv}_{meta_conf:.2f}<{threshold:.2f}"
                )
                # ✅ Fix: Meta-reject olduğunda _last_signal'ı temizle
                self._last_signal = None
                return self._hold_decision(
                    regime,
                    f"Meta-filter rejected ({meta_reason}): "
                    f"{meta_conf:.2f} < {threshold:.2f}",
                )

        # Katman 4: Position Sizing
        sig_score = _safe_signal_attr(primary_signal, 'soft_score', 3)
        position_size = self._calculate_position_size(
            sig_score, meta_conf, regime
        )

        # Final Güven Skoru
        final_confidence = (
            primary_signal.confidence * 0.3 + meta_conf * 0.7
        )

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
            reason=(
                f"{regime.value} | {primary_signal.reason} | "
                f"meta={meta_conf:.2f} size={position_size:.0%}"
            ),
            soft_score=sig_score,
            entry_type=_safe_signal_attr(
                primary_signal, 'entry_type', 'none'
            ),
            tp_price=_safe_signal_attr(
                primary_signal, 'tp_price', 0.0
            ),
            sl_price=_safe_signal_attr(
                primary_signal, 'sl_price', 0.0
            ),
        )

    # ──────────────────────────────────────────────────────────
    # Secondary Strategy Fallback
    # ──────────────────────────────────────────────────────────

    def _try_secondary_strategy(
        self,
        df: pd.DataFrame,
        primary_regime: Regime,
    ) -> Signal:
        """
        Birincil strateji sinyal vermezse diğer stratejileri dene.
        Confidence %20 düşürülür (yanlış rejimde çalışıyor olabilir).
        """
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

    # ──────────────────────────────────────────────────────────
    # Position Sizing
    # ──────────────────────────────────────────────────────────

    def _calculate_position_size(
        self,
        soft_score: int,
        meta_conf: float,
        regime: Regime,
    ) -> float:
        """
        Scored Position Sizing (v2.0)
        Score 5: 100%, Score 4: 75%, Score 3: 50%, <3: filtrelenir
        """
        score_mult = {5: 1.0, 4: 0.75, 3: 0.50}.get(soft_score, 0.0)

        # Meta confidence da boyutu etkiler (0.75+ → 1.0x)
        meta_mult = min(1.0, meta_conf / 0.75)

        regime_max = {
            Regime.TRENDING: 1.0,
            Regime.MEAN_REVERTING: 0.80,
            Regime.HIGH_VOLATILE: 0.50,
            Regime.LOW_VOLATILE: 0.60,
        }
        max_size = regime_max.get(regime, 0.50)

        size = score_mult * meta_mult
        return min(size, max_size)

    # ──────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────

    def _hold_decision(self, regime, reason: str) -> TradeDecision:
        regime_val = (
            regime.value if isinstance(regime, Regime) else str(regime)
        )
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

    def _track_reason(self, reason: str):
        key = reason[:50]
        self._debug_counts['reject_reasons'][key] = (
            self._debug_counts['reject_reasons'].get(key, 0) + 1
        )

    def get_last_signal_info(self) -> Optional[Dict]:
        """
        Son geçerli sinyalin bilgilerini döndürür.
        MetaLabelGenerator ve diğer modüller tarafından kullanılır.

        Returns:
            None: Sinyal yoksa veya HOLD kararı verildiyse
            Dict: Sinyal detayları
        """
        sig = self._last_signal
        if sig is None:
            return None

        return {
            'direction': sig.direction,
            'confidence': sig.confidence,
            'strategy_name': sig.strategy_name,
            'tp_atr_mult': sig.tp_atr_mult,
            'sl_atr_mult': sig.sl_atr_mult,
            'tp_price': _safe_signal_attr(sig, 'tp_price', 0.0),
            'sl_price': _safe_signal_attr(sig, 'sl_price', 0.0),
            'soft_score': _safe_signal_attr(sig, 'soft_score', 0),
            'entry_type': _safe_signal_attr(sig, 'entry_type', 'none'),
            'reason': sig.reason,
        }

    def get_status(self) -> dict:
        """Engine durumunu döndürür (API/UI için)."""
        meta_status = {}
        for r in Regime:
            if self.meta_predictor is not None:
                models = getattr(self.meta_predictor, 'models', {})
                meta_status[r.value] = r.value in models
            else:
                meta_status[r.value] = False

        return {
            "timeframe": self.timeframe,
            "secondary_tf": self.secondary_tf,
            "meta_filter_active": self.meta_predictor is not None,
            "meta_models_loaded": meta_status,
            "strategies": {
                r.value: type(self.strategies[r]).__name__
                for r in self.strategies
            },
        }

    def print_debug_stats(self):
        """Debug istatistiklerini yazdır."""
        d = self._debug_counts
        total = d['total_calls']
        if total == 0:
            print("  📊 Debug: Henüz çağrı yok")
            return

        def _pct(val):
            return f"{val / total * 100:.1f}"

        print(f"\n  {'-' * 50}")
        print(f"  ADAPTIVE ENGINE DEBUG STATS")
        print(f"  {'-' * 50}")
        print(f"  Total calls    : {total:,}")
        print(f"  No strategy    : {d['no_strategy']:,} "
              f"(%{_pct(d['no_strategy'])})")
        print(f"  No signal      : {d['no_signal']:,} "
              f"(%{_pct(d['no_signal'])})")
        print(f"  Low confidence : {d['low_confidence']:,} "
              f"(%{_pct(d['low_confidence'])})")
        print(f"  Meta rejected  : {d['meta_rejected']:,} "
              f"(%{_pct(d['meta_rejected'])})")
        print(f"  Meta low       : {d['meta_low']:,} "
              f"(%{_pct(d['meta_low'])})")
        print(f"  ✅ PASSED      : {d['passed']:,} "
              f"(%{_pct(d['passed'])})")

        print(f"\n  Rejim dağılımı:")
        for r, c in sorted(
            d['regime_counts'].items(), key=lambda x: -x[1]
        ):
            print(f"    {r:16}: {c:,} (%{_pct(c)})")

        if d['reject_reasons']:
            print(f"\n  Top red nedenleri:")
            sorted_reasons = sorted(
                d['reject_reasons'].items(), key=lambda x: -x[1]
            )[:10]
            for reason, count in sorted_reasons:
                print(f"    {count:5,}x | {reason}")

        print(f"  {'-' * 50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Signal Field Güvenli Erişim
# ─────────────────────────────────────────────────────────────────────────────

def _safe_signal_attr(signal: Signal, attr: str, default=None):
    """
    Signal dataclass'ında olmayan opsiyonel alanlara güvenli erişim.
    Farklı strateji versiyonları farklı field'lar ekleyebilir;
    bu fonksiyon AttributeError'ı önler.
    """
    return getattr(signal, attr, default)
