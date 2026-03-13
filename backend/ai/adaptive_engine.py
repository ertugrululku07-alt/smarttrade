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
                meta_tfs = list(dict.fromkeys([primary_tf, secondary_tf, "4h"]))
                self.meta_predictor = MetaPredictor(
                    timeframes=meta_tfs
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
            sig_score_now = _safe_signal_attr(primary_signal, 'soft_score', 3)
            entry_ok, entry_reason = self._check_entry_timing(
                df_secondary, primary_signal.direction,
                regime=rv, mtf_score=sig_score_now,
            )
            if not entry_ok:
                self._debug_counts['no_signal'] += 1
                self._track_reason(f"entry_timing_{rv}")
                return self._hold_decision(
                    regime, f"Entry timing: {entry_reason}"
                )
            primary_signal.reason += f" | Entry: {entry_reason}"

        self._last_signal = primary_signal

        # 4b. HTF Feature Injection — Meta model için 6 HTF feature'ı df'e ekle
        df = self._inject_htf_features(df, df_4h)

        # 5. Meta-Filter (Soft Scaling)
        # Hard gate SADECE çok düşük confidence'da → geri kalanı position size ile ölçekle
        MINIMUM_META = 0.15  # Bu seviyenin altı = "neredeyse kesinlikle yanlış"
        meta_conf = primary_signal.confidence * 0.85
        if self.meta_predictor is not None:
            meta_conf, should_trade, threshold, meta_reason = self.meta_predictor.predict(
                df=df, regime=regime, signal_direction=primary_signal.direction,
                signal_confidence=primary_signal.confidence, timeframe=self.timeframe
            )
            if meta_conf < MINIMUM_META:
                self._debug_counts['meta_rejected'] += 1
                self._last_signal = None
                return self._hold_decision(regime, f"Meta hard-reject: {meta_conf:.2f}<{MINIMUM_META}")
            if not should_trade:
                self._debug_counts['meta_low'] += 1

        # 6. Position Sizing & Confidence (meta_conf scales position)
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
        regime: str = "unknown",
        mtf_score: int = 0,
    ) -> Tuple[bool, str]:
        """
        Alt TF (15m/5m) giriş zamanlaması kontrolü — Regime-Aware v2.0

        Rejime göre RSI limitleri:
          TRENDING   → RSI 88/12 (güçlü trendde yüksek RSI normal)
          MEAN_REV   → RSI 75/25 (dönüş bekleniyor, hassas)
          HIGH_VOL   → RSI 80/20
          LOW_VOL    → RSI 80/20

        MTF score ≥ 4 → Limitler daha da gevşer (92/8)
        """
        try:
            close = df_ltf['close']
            high = df_ltf['high']
            low = df_ltf['low']

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

            # ── Regime-Aware RSI Limitleri ──
            regime_lower = regime.lower() if isinstance(regime, str) else str(regime).lower()
            if regime_lower == 'trending':
                rsi_high, rsi_low = 88, 12
                mom_limit = 3.5
                body_check = False        # Trending'de zayıf mum kontrolü kapalı
            elif regime_lower == 'mean_reverting':
                rsi_high, rsi_low = 75, 25
                mom_limit = 2.0
                body_check = True
            else:
                rsi_high, rsi_low = 80, 20
                mom_limit = 2.5
                body_check = True

            # ── MTF Override: Güçlü 4h trend → daha da gevşet ──
            if mtf_score >= 4:
                rsi_high = min(rsi_high + 4, 92)
                rsi_low = max(rsi_low - 4, 8)
                mom_limit *= 1.5

            if direction == 'LONG':
                if rsi_val > rsi_high:
                    return False, f"RSI aşırı alınmış ({rsi_val:.0f}, limit={rsi_high})"
                if mom_3 > mom_limit:
                    return False, f"Hareket kaçırılmış (mom={mom_3:.1f}%, limit={mom_limit:.1f}%)"
                if body_check and body_ratio < 0.15 and rsi_val > 65:
                    return False, f"Zayıf mum + yüksek RSI"
                return True, f"OK (rsi={rsi_val:.0f} mom={mom_3:.1f}% regime={regime_lower})"

            elif direction == 'SHORT':
                if rsi_val < rsi_low:
                    return False, f"RSI aşırı satılmış ({rsi_val:.0f}, limit={rsi_low})"
                if mom_3 < -mom_limit:
                    return False, f"Hareket kaçırılmış (mom={mom_3:.1f}%, limit={-mom_limit:.1f}%)"
                if body_check and body_ratio < 0.15 and rsi_val < 35:
                    return False, f"Zayıf mum + düşük RSI"
                return True, f"OK (rsi={rsi_val:.0f} mom={mom_3:.1f}% regime={regime_lower})"

            return True, "neutral"

        except Exception as e:
            return True, f"timing_error ({e})"

    def _inject_htf_features(self, df: pd.DataFrame, df_4h: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        4h verisinden 6 HTF feature hesapla ve df'in son satırına ekle.
        Meta model bu feature'lara ihtiyaç duyar.

        Features: htf_bias_numeric, htf_trend_strength, htf_rsi,
                  htf_price_vs_ema200, htf_ema_alignment, htf_structure_numeric
        """
        try:
            if df_4h is not None and len(df_4h) >= 30:
                cp = float(df['close'].iloc[-1])
                atr_val = float(df['atr'].iloc[-1]) if 'atr' in df.columns else cp * 0.01
                ctx = self.mtf_analyzer.analyze_htf(df_4h, "")
                htf_feats = self.mtf_analyzer.generate_cross_tf_features(ctx, cp, atr_val)
            else:
                htf_feats = {
                    'htf_bias_numeric': 0.0, 'htf_trend_strength': 0.0,
                    'htf_rsi': 50.0, 'htf_price_vs_ema200': 0.0,
                    'htf_ema_alignment': 0.0, 'htf_structure_numeric': 0.0,
                }

            for col, val in htf_feats.items():
                if col not in df.columns:
                    df[col] = 0.0
                df.at[df.index[-1], col] = val
        except Exception as e:
            pass  # HTF feature injection failure is non-critical
        return df

    def _calculate_position_size(self, soft_score: int, meta_conf: float, regime: Regime) -> float:
        """
        Soft-scaling position size:
          meta_conf >= 0.55 → full multiplier (1.0)
          meta_conf >= 0.40 → moderate (0.70)
          meta_conf >= 0.25 → small (0.40)
          meta_conf >= 0.15 → minimum (0.20)
          meta_conf <  0.15 → zero (hard reject)
        """
        score_mult = {5: 1.0, 4: 0.75, 3: 0.50}.get(soft_score, 0.0)

        if meta_conf >= 0.55:
            meta_mult = 1.0
        elif meta_conf >= 0.40:
            meta_mult = 0.70
        elif meta_conf >= 0.25:
            meta_mult = 0.40
        elif meta_conf >= 0.15:
            meta_mult = 0.20
        else:
            meta_mult = 0.0

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

    def diagnose(
        self,
        df: pd.DataFrame,
        df_secondary: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
        symbol: str = "UNKNOWN",
    ) -> Dict:
        """
        Pipeline Diagnostic — Karar sürecinin her adımını trace eder.

        Döndürür: Adım adım sonuç raporu (dict).
        """
        trace = {
            "symbol": symbol,
            "timeframe": self.timeframe,
            "secondary_tf": self.secondary_tf,
            "data": {
                "1h_bars": len(df) if df is not None else 0,
                "15m_bars": len(df_secondary) if df_secondary is not None else 0,
                "4h_bars": len(df_4h) if df_4h is not None else 0,
            },
            "steps": [],
            "final_action": "HOLD",
            "passed_all": False,
        }

        def step(name, status, detail):
            trace["steps"].append({"step": name, "status": status, **detail})

        # ── Step 1: Rejim Tespiti ──
        regime, regime_info = detect_regime(df)
        step("1_regime_detection", "✅", {
            "regime": regime.value,
            "adx": round(float(df['adx'].iloc[-1]), 1) if 'adx' in df.columns else None,
            "hurst": round(float(df['hurst'].iloc[-1]), 3) if 'hurst' in df.columns else None,
        })

        # ── Step 2: Primary Strategy ──
        strategy = self.strategies.get(regime)
        strategy_name = type(strategy).__name__ if strategy else "None"
        primary_signal = strategy.generate_signal(df) if strategy else None

        if primary_signal and primary_signal.is_valid:
            step("2_primary_strategy", "✅", {
                "strategy": strategy_name,
                "direction": primary_signal.direction,
                "confidence": round(primary_signal.confidence, 3),
                "soft_score": _safe_signal_attr(primary_signal, 'soft_score', 0),
                "entry_type": _safe_signal_attr(primary_signal, 'entry_type', 'none'),
                "tp_price": _safe_signal_attr(primary_signal, 'tp_price', 0),
                "sl_price": _safe_signal_attr(primary_signal, 'sl_price', 0),
                "reason": primary_signal.reason[:120],
            })
        else:
            step("2_primary_strategy", "⚠️ NO SIGNAL", {
                "strategy": strategy_name,
                "reason": primary_signal.reason[:120] if primary_signal else "no strategy",
            })
            # Try secondary
            secondary_signal = self._try_secondary_strategy(df, regime)
            if secondary_signal.is_valid:
                primary_signal = secondary_signal
                step("2b_secondary_strategy", "✅", {
                    "strategy": primary_signal.strategy_name,
                    "direction": primary_signal.direction,
                    "confidence": round(primary_signal.confidence, 3),
                    "soft_score": _safe_signal_attr(primary_signal, 'soft_score', 0),
                    "reason": primary_signal.reason[:120],
                })
            else:
                step("2b_secondary_strategy", "❌ REJECTED", {
                    "reason": "Tüm stratejiler reddedildi"
                })
                trace["final_action"] = "HOLD"
                trace["hold_reason"] = "No strategy signal"
                return trace

        # ── Step 3: MTF Filter (4h Trend) ──
        if df_4h is not None:
            ctx = self.mtf_analyzer.analyze_htf(df_4h, symbol)
            adj = self.mtf_analyzer.get_signal_adjustment(ctx, primary_signal.direction)
            mtf_score_before = _safe_signal_attr(primary_signal, 'soft_score', 3)
            mtf_score_after = mtf_score_before + adj.score_modifier

            if not adj.allowed:
                step("3_mtf_filter", "❌ BLOCKED", {
                    "trend_bias": ctx.bias.value,
                    "trend_strength": round(ctx.trend_strength, 2) if ctx.trend_strength else 0,
                    "signal_direction": primary_signal.direction,
                    "reason": adj.reason,
                })
                trace["final_action"] = "HOLD"
                trace["hold_reason"] = f"MTF blocked: {adj.reason}"
                return trace
            elif mtf_score_after < 3:
                step("3_mtf_filter", "❌ LOW SCORE", {
                    "trend_bias": ctx.bias.value,
                    "score_before": mtf_score_before,
                    "score_modifier": adj.score_modifier,
                    "score_after": mtf_score_after,
                    "reason": adj.reason,
                })
                trace["final_action"] = "HOLD"
                trace["hold_reason"] = f"MTF score low: {mtf_score_after}"
                return trace
            else:
                # Apply MTF to signal for subsequent steps
                primary_signal = self._process_signal_with_mtf(primary_signal, df_4h, symbol)
                step("3_mtf_filter", "✅", {
                    "trend_bias": ctx.bias.value,
                    "trend_strength": round(ctx.trend_strength, 2) if ctx.trend_strength else 0,
                    "score_before": mtf_score_before,
                    "score_modifier": adj.score_modifier,
                    "score_after": mtf_score_after,
                    "aligned": adj.reason,
                })
                if not primary_signal.is_valid:
                    step("3_mtf_filter_post", "❌ SIGNAL INVALIDATED", {
                        "reason": primary_signal.reason[:120],
                    })
                    trace["final_action"] = "HOLD"
                    trace["hold_reason"] = f"MTF invalidated signal"
                    return trace
        else:
            step("3_mtf_filter", "⚠️ SKIPPED", {"reason": "4h verisi yok"})

        # ── Step 4: Entry Timing (15m) ──
        diag_score = _safe_signal_attr(primary_signal, 'soft_score', 3)
        if df_secondary is not None and len(df_secondary) >= 20:
            entry_ok, entry_reason = self._check_entry_timing(
                df_secondary, primary_signal.direction,
                regime=regime.value, mtf_score=diag_score,
            )
            if entry_ok:
                step("4_entry_timing", "✅", {
                    "timeframe": self.secondary_tf,
                    "result": entry_reason,
                })
            else:
                step("4_entry_timing", "❌ REJECTED", {
                    "timeframe": self.secondary_tf,
                    "result": entry_reason,
                })
                trace["final_action"] = "HOLD"
                trace["hold_reason"] = f"Entry timing: {entry_reason}"
                return trace
        else:
            step("4_entry_timing", "⚠️ SKIPPED", {
                "reason": f"15m verisi yok veya yetersiz ({len(df_secondary) if df_secondary is not None else 0} bars)"
            })

        # ── Step 4c: HTF Feature Injection ──
        df = self._inject_htf_features(df, df_4h)

        # ── Step 5: Meta Filter ──
        meta_conf = primary_signal.confidence * 0.85
        meta_active = self.meta_predictor is not None
        if meta_active:
            meta_conf, should_trade, threshold, meta_reason = self.meta_predictor.predict(
                df=df, regime=regime, signal_direction=primary_signal.direction,
                signal_confidence=primary_signal.confidence, timeframe=self.timeframe,
                debug=True,
            )
            # Feature debug from last prediction
            feat_debug = self.meta_predictor.last_debug or {}

            # Find which model key was used
            regime_val = regime.value
            used_key = feat_debug.get("model_key", f"{regime_val}_{self.timeframe}")

            model_meta = self.meta_predictor.models.get(used_key, {}).get('meta', {})

            # Feature summary for trace
            feat_summary = {
                "total_features": feat_debug.get("total_features", 0),
                "ok_nonzero": feat_debug.get("ok_nonzero", 0),
                "signal_features": feat_debug.get("signal_features", 0),
                "zero_or_nan": feat_debug.get("zero_or_nan", 0),
                "missing_from_df": feat_debug.get("missing_from_df", 0),
                "missing_list": feat_debug.get("missing_list", []),
                "zero_list": feat_debug.get("zero_list", []),
            }

            # Soft-scaling tiers
            MINIMUM_META = 0.15
            if meta_conf >= 0.55:
                meta_tier = "FULL (1.0x)"
            elif meta_conf >= 0.40:
                meta_tier = "MODERATE (0.70x)"
            elif meta_conf >= 0.25:
                meta_tier = "SMALL (0.40x)"
            elif meta_conf >= MINIMUM_META:
                meta_tier = "MINIMUM (0.20x)"
            else:
                meta_tier = "HARD REJECT"

            if meta_conf < MINIMUM_META:
                step("5_meta_filter", "❌ HARD REJECT", {
                    "model_key": used_key,
                    "meta_confidence": round(meta_conf, 4),
                    "minimum_threshold": MINIMUM_META,
                    "reason": meta_reason,
                    "features": feat_summary,
                })
                trace["final_action"] = "HOLD"
                trace["hold_reason"] = f"Meta hard-reject: {meta_conf:.3f} < {MINIMUM_META}"
                return trace

            status = "✅ PASSED" if should_trade else f"⚠️ LOW → {meta_tier}"
            step("5_meta_filter", status, {
                "model_key": used_key,
                "meta_confidence": round(meta_conf, 4),
                "threshold": round(threshold, 3),
                "position_tier": meta_tier,
                "model_auc": model_meta.get('auc', '?'),
                "reason": meta_reason,
                "features": feat_summary,
            })
        else:
            step("5_meta_filter", "⚠️ DISABLED", {
                "reason": "MetaPredictor yüklenmedi",
                "fallback_conf": round(meta_conf, 3),
            })

        # ── Step 6: Final Decision ──
        sig_score = _safe_signal_attr(primary_signal, 'soft_score', 3)
        position_size = self._calculate_position_size(sig_score, meta_conf, regime)
        final_confidence = (primary_signal.confidence * 0.3 + meta_conf * 0.7)

        step("6_final_decision", "✅ TRADE SIGNAL", {
            "action": primary_signal.direction,
            "strategy": primary_signal.strategy_name,
            "confidence": round(final_confidence, 3),
            "meta_confidence": round(meta_conf, 3),
            "soft_score": sig_score,
            "position_size": round(position_size, 3),
            "entry_type": _safe_signal_attr(primary_signal, 'entry_type', 'none'),
            "tp_price": _safe_signal_attr(primary_signal, 'tp_price', 0),
            "sl_price": _safe_signal_attr(primary_signal, 'sl_price', 0),
        })

        # ── Quality Gate Check ──
        quality_issues = []
        if sig_score < 3:
            quality_issues.append(f"soft_score={sig_score} < 3")
        if meta_conf <= 0:
            quality_issues.append(f"meta_confidence={meta_conf:.3f} <= 0")
        if final_confidence < 0.60:
            quality_issues.append(f"confidence={final_confidence:.3f} < 0.60")

        if quality_issues:
            step("7_quality_gate", "❌ WOULD BE BLOCKED", {
                "issues": quality_issues,
                "note": "should_open_position() bu trade'i engellerdi"
            })
            trace["final_action"] = "HOLD"
            trace["hold_reason"] = f"Quality gate: {', '.join(quality_issues)}"
        else:
            step("7_quality_gate", "✅ PASSED", {
                "soft_score": f"{sig_score}/5",
                "meta_confidence": round(meta_conf, 3),
                "final_confidence": round(final_confidence, 3),
            })
            trace["final_action"] = primary_signal.direction
            trace["passed_all"] = True

        return trace

    def get_health_check(self) -> Dict:
        """Sistem sağlık kontrolü — model, strateji, pipeline durumu."""
        health = {
            "engine_version": "v2.0 MTF",
            "timeframe": self.timeframe,
            "secondary_tf": self.secondary_tf,
            "strategies": {},
            "meta_filter": {},
            "mtf_analyzer": True,
            "pipeline_stats": dict(self._debug_counts),
        }

        # Strategies
        for regime, strat in self.strategies.items():
            health["strategies"][regime.value] = {
                "class": type(strat).__name__,
                "loaded": True,
            }

        # Meta filter
        if self.meta_predictor:
            health["meta_filter"]["active"] = True
            health["meta_filter"]["models"] = {}
            for key, info in self.meta_predictor.models.items():
                meta = info.get('meta', {})
                health["meta_filter"]["models"][key] = {
                    "loaded": True,
                    "features": len(info.get('features', [])),
                    "threshold": info.get('threshold', '?'),
                    "auc": meta.get('auc', '?'),
                    "precision": meta.get('precision', '?'),
                    "verdict": meta.get('verdict', '?'),
                }
            # Missing models
            for tf in self.meta_predictor.timeframes:
                for regime in Regime:
                    key = f"{regime.value}_{tf}"
                    if key not in health["meta_filter"]["models"]:
                        health["meta_filter"]["models"][key] = {"loaded": False}
        else:
            health["meta_filter"]["active"] = False

        # Pass rate
        total = self._debug_counts.get('total_calls', 0)
        passed = self._debug_counts.get('passed', 0)
        health["pass_rate"] = f"{passed}/{total} ({passed/total:.1%})" if total > 0 else "0/0"

        return health

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
