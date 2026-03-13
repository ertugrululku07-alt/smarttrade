"""
Adaptive Engine v1.5 (MTF Phase 1)

v1.5 Değişiklikleri:
  - MTFAnalyzer entegrasyonu (4h Trend Filter)
  - decide() metodu df_4h parametresi alır
  - _process_signal_with_mtf ile hiyerarşik onay sistemi
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from ai.regime_detector import detect_regime, Regime
from ai.mtf_analyzer import MTFAnalyzer
from ai.strategies import (
    MomentumStrategy, MeanReversionStrategy,
    ScalpingStrategy, Signal,
    ICTStrategy, SmartMoneyOrderFlowStrategy,
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
    Ana Karar Mekanizması — Rejim tespiti + Strateji seçimi + Meta filtre + MTF Filtre
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

        self._last_signal: Optional[Signal] = None
        self.meta_predictor = None
        self.mtf_analyzer = MTFAnalyzer()

        if use_meta_filter:
            try:
                from ai.meta_labelling.meta_predictor import MetaPredictor
                self.meta_predictor = MetaPredictor(
                    timeframes=[primary_tf, secondary_tf]
                )
                if not self.meta_predictor.is_ready:
                    print("  WARNING: Meta-model bulunamadi, filtresiz calisilacak")
                    self.meta_predictor = None
            except Exception as e:
                print(f"  ERROR: MetaPredictor yuklenemedi: {e}")
                self.meta_predictor = None

        self._debug_counts = {
            'total_calls': 0, 'no_strategy': 0, 'no_signal': 0,
            'low_confidence': 0, 'meta_rejected': 0, 'meta_low': 0,
            'passed': 0, 'regime_counts': {}, 'reject_reasons': {},
        }

    def _process_signal_with_mtf(self, signal: Signal, df_4h: pd.DataFrame, symbol: str) -> Signal:
        """
        4h trend filtresini sinyale uygula (Phase 1).
        """
        if signal is None or signal.direction is None or df_4h is None:
            return signal

        ctx = self.mtf_analyzer.analyze_htf(df_4h, symbol)
        adj = self.mtf_analyzer.get_signal_adjustment(ctx, signal.direction)

        if not adj.allowed:
            return Signal(None, 0.0, signal.strategy_name, "", f"MTF RED: {adj.reason}", hard_pass=False)

        signal.soft_score += adj.score_modifier
        if signal.soft_score < 3:
            return Signal(None, 0.0, signal.strategy_name, "", f"MTF Score Low: {signal.soft_score} ({adj.reason})", hard_pass=False)

        signal.reason += f" | MTF: {adj.reason}"
        
        if signal.tp_price > 0 and signal.sl_price > 0:
            risk = abs(signal.entry_price - signal.sl_price)
            reward = abs(signal.tp_price - signal.entry_price)
            if signal.direction == 'LONG':
                signal.sl_price = signal.entry_price - (risk * adj.sl_multiplier)
                signal.tp_price = signal.entry_price + (reward * adj.tp_multiplier)
            else:
                signal.sl_price = signal.entry_price + (risk * adj.sl_multiplier)
                signal.tp_price = signal.entry_price - (reward * adj.tp_multiplier)
        return signal

    def decide(
        self,
        df: pd.DataFrame,
        df_secondary: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
        symbol: str = "UNKNOWN",
    ) -> TradeDecision:
        """
        Karar Mekanizması v2.0 — Multi-TF Hierarchy

        Akış:
          4h → Trend yönü (MTF filter)
          1h → Rejim tespiti + strateji sinyali + meta filter
          15m → Giriş zamanlaması (entry timing)
        """
        self._debug_counts['total_calls'] += 1
        self._last_signal = None

        # 1. Rejim Tespiti (HER ZAMAN primary TF = 1h)
        regime, _ = detect_regime(df)
        rv = regime.value
        self._debug_counts['regime_counts'][rv] = self._debug_counts['regime_counts'].get(rv, 0) + 1

        # 2. Strateji Sinyali (HER ZAMAN primary TF = 1h)
        strategy = self.strategies.get(regime)
        if not strategy:
            return self._hold_decision(regime, "Strateji bulunamadı")

        primary_signal = strategy.generate_signal(df)
        if not primary_signal.is_valid:
            primary_signal = self._try_secondary_strategy(df, regime)

        # 3. MTF Filtrelemesi (4h trend yönü)
        if primary_signal.is_valid and df_4h is not None:
            primary_signal = self._process_signal_with_mtf(primary_signal, df_4h, symbol)

        if not primary_signal.is_valid:
            self._debug_counts['no_signal'] += 1
            self._track_reason(f"{primary_signal.reason}_{rv}")
            return self._hold_decision(regime, f"Signal rejected ({primary_signal.reason})")

        # 4. Giriş Zamanlaması (15m/5m entry timing)
        if df_secondary is not None and len(df_secondary) >= 20:
            entry_ok, entry_reason = self._check_entry_timing(
                df_secondary, primary_signal.direction
            )
            if not entry_ok:
                self._debug_counts['no_signal'] += 1
                self._track_reason(f"entry_timing_{rv}")
                return self._hold_decision(
                    regime, f"Entry timing: {entry_reason}"
                )
            primary_signal.reason += f" | Entry: {entry_reason}"

        self._last_signal = primary_signal

        # 5. Meta-Filter
        meta_conf = primary_signal.confidence * 0.85
        if self.meta_predictor is not None:
            meta_conf, should_trade, threshold, meta_reason = self.meta_predictor.predict(
                df=df, regime=regime, signal_direction=primary_signal.direction,
                signal_confidence=primary_signal.confidence, timeframe=self.timeframe
            )
            if not should_trade:
                self._debug_counts['meta_rejected'] += 1
                self._last_signal = None
                return self._hold_decision(regime, f"Meta rejected: {meta_conf:.2f}<{threshold:.2f} ({meta_reason})")

        # 6. Position Sizing & Confidence
        sig_score = _safe_signal_attr(primary_signal, 'soft_score', 3)
        position_size = self._calculate_position_size(sig_score, meta_conf, regime)
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
            reason=f"{regime.value} | {primary_signal.reason} | meta={meta_conf:.2f}",
            soft_score=sig_score,
            entry_type=_safe_signal_attr(primary_signal, 'entry_type', 'none'),
            tp_price=_safe_signal_attr(primary_signal, 'tp_price', 0.0),
            sl_price=_safe_signal_attr(primary_signal, 'sl_price', 0.0),
        )

    def _try_secondary_strategy(self, df: pd.DataFrame, primary_regime: Regime) -> Signal:
        secondary_order = {
            Regime.MEAN_REVERTING: [Regime.TRENDING, Regime.LOW_VOLATILE],
            Regime.TRENDING: [Regime.HIGH_VOLATILE, Regime.MEAN_REVERTING],
            Regime.HIGH_VOLATILE: [Regime.TRENDING, Regime.MEAN_REVERTING],
            Regime.LOW_VOLATILE: [Regime.MEAN_REVERTING, Regime.TRENDING],
        }
        # Smart Money as additional secondary for trending/high_vol
        smo = SmartMoneyOrderFlowStrategy()
        if primary_regime in (Regime.TRENDING, Regime.HIGH_VOLATILE):
            signal = smo.generate_signal(df)
            if signal.is_valid:
                signal.confidence *= 0.85
                signal.strategy_name += "(secondary)"
                return signal
        for alt_regime in secondary_order.get(primary_regime, []):
            alt_strategy = self.strategies.get(alt_regime)
            if not alt_strategy: continue
            signal = alt_strategy.generate_signal(df)
            if signal.is_valid:
                signal.confidence *= 0.8
                signal.strategy_name += "(secondary)"
                return signal
        return Signal(None, 0.0, "none", primary_regime.value, "Tüm stratejiler reddedildi")

    def _check_entry_timing(
        self,
        df_ltf: pd.DataFrame,
        direction: str,
    ) -> Tuple[bool, str]:
        """
        Alt TF (15m/5m) giriş zamanlaması kontrolü.

        1h sinyali onaylandıktan sonra, 15m/5m'de uygun giriş koşulu arar:
          LONG  → RSI aşırı satılmış değil, son mum yapısı uygun
          SHORT → RSI aşırı alınmış değil, son mum yapısı uygun
        """
        try:
            close = df_ltf['close']
            high = df_ltf['high']
            low = df_ltf['low']
            last_close = float(close.iloc[-1])

            # RSI hesapla (yoksa)
            if 'rsi' in df_ltf.columns:
                rsi_val = float(df_ltf['rsi'].iloc[-1])
            else:
                delta = close.diff()
                gain = delta.where(delta > 0, 0.0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
                rs = gain / loss.replace(0, 1e-10)
                rsi_s = 100 - (100 / (1 + rs))
                rsi_val = float(rsi_s.iloc[-1]) if not rsi_s.empty else 50.0

            if np.isnan(rsi_val):
                rsi_val = 50.0

            # Son 3 mum momentum
            if len(close) >= 4:
                mom_3 = (float(close.iloc[-1]) - float(close.iloc[-4])) / float(close.iloc[-4]) * 100
            else:
                mom_3 = 0.0

            # Son mum body oranı
            body = abs(float(close.iloc[-1]) - float(df_ltf['open'].iloc[-1]))
            full_range = float(high.iloc[-1]) - float(low.iloc[-1])
            body_ratio = body / full_range if full_range > 0 else 0.5

            if direction == 'LONG':
                # LONG giriş: RSI 80+ = aşırı alınmış → kötü giriş
                if rsi_val > 80:
                    return False, f"RSI aşırı alınmış ({rsi_val:.0f})"
                # Son 3 mum çok sert yükselmiş → geç kalınmış
                if mom_3 > 2.0:
                    return False, f"Hareket kaçırılmış (mom={mom_3:.1f}%)"
                # Doji/wick candle = belirsizlik
                if body_ratio < 0.15 and rsi_val > 65:
                    return False, f"Zayıf mum + yüksek RSI"
                return True, f"OK (rsi={rsi_val:.0f} mom={mom_3:.1f}%)"

            elif direction == 'SHORT':
                if rsi_val < 20:
                    return False, f"RSI aşırı satılmış ({rsi_val:.0f})"
                if mom_3 < -2.0:
                    return False, f"Hareket kaçırılmış (mom={mom_3:.1f}%)"
                if body_ratio < 0.15 and rsi_val < 35:
                    return False, f"Zayıf mum + düşük RSI"
                return True, f"OK (rsi={rsi_val:.0f} mom={mom_3:.1f}%)"

            return True, "neutral"

        except Exception as e:
            # Entry timing hatası sinyali engellemez
            return True, f"timing_error ({e})"

    def _calculate_position_size(self, soft_score: int, meta_conf: float, regime: Regime) -> float:
        score_mult = {5: 1.0, 4: 0.75, 3: 0.50}.get(soft_score, 0.0)
        meta_mult = min(1.0, meta_conf / 0.75)
        regime_max = {Regime.TRENDING: 1.0, Regime.MEAN_REVERTING: 0.8, Regime.HIGH_VOLATILE: 0.5, Regime.LOW_VOLATILE: 0.6}
        return min(score_mult * meta_mult, regime_max.get(regime, 0.5))

    def _hold_decision(self, regime, reason: str) -> TradeDecision:
        return TradeDecision('HOLD', 0.0, 0.0, regime.value if hasattr(regime, 'value') else str(regime), reason=f"HOLD [{reason}]")

    def _track_reason(self, reason: str):
        key = reason[:50]
        self._debug_counts['reject_reasons'][key] = self._debug_counts['reject_reasons'].get(key, 0) + 1

    def get_last_signal_info(self) -> Optional[Dict]:
        sig = self._last_signal
        if not sig: return None
        return {
            'direction': sig.direction, 'confidence': sig.confidence, 'strategy_name': sig.strategy_name,
            'tp_atr_mult': sig.tp_atr_mult, 'sl_atr_mult': sig.sl_atr_mult,
            'tp_price': _safe_signal_attr(sig, 'tp_price', 0.0), 'sl_price': _safe_signal_attr(sig, 'sl_price', 0.0),
            'soft_score': _safe_signal_attr(sig, 'soft_score', 0), 'entry_type': _safe_signal_attr(sig, 'entry_type', 'none'),
            'reason': sig.reason,
        }

    def get_status(self) -> dict:
        meta_status = {r.value: (r.value in self.meta_predictor.models if self.meta_predictor else False) for r in Regime}
        return {
            "timeframe": self.timeframe, "secondary_tf": self.secondary_tf,
            "meta_filter_active": self.meta_predictor is not None, "meta_models_loaded": meta_status,
            "strategies": {r.value: type(self.strategies[r]).__name__ for r in self.strategies},
        }

    def print_debug_stats(self):
        d = self._debug_counts
        total = d['total_calls']
        if total == 0: return
        print(f"\n  ADAPTIVE ENGINE DEBUG (MTF v1.5)\n  Passed: {d['passed']}/{total} ({d['passed']/total:.1%})")


def _safe_signal_attr(signal: Signal, attr: str, default=None):
    return getattr(signal, attr, default)
