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
from ai.strategies.volatility_strategy import VolatilityStrategy


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

        # Multi-Strategy Ensemble: her rejim için birden fazla strateji
        self._all_strategies = {
            'momentum': MomentumStrategy(),
            'ict': ICTStrategy(),
            'smart_money': SmartMoneyOrderFlowStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'scalping': ScalpingStrategy(),
            'volatility': VolatilityStrategy(),
        }

        # Rejim başına strateji öncelik sırası (hepsi denenir, en iyi seçilir)
        # v2.1: smart_money kaldırıldı (backtest'te -11% PnL, %40 WR)
        self._regime_strategies = {
            Regime.TRENDING:       ['momentum', 'ict'],
            Regime.MEAN_REVERTING: ['mean_reversion', 'scalping'],
            Regime.HIGH_VOLATILE:  ['ict', 'volatility'],
            Regime.LOW_VOLATILE:   ['scalping', 'mean_reversion'],
        }

        # Backward compat: strategies dict (primary per regime)
        self.strategies = {
            Regime.TRENDING: self._all_strategies['momentum'],
            Regime.MEAN_REVERTING: self._all_strategies['mean_reversion'],
            Regime.HIGH_VOLATILE: self._all_strategies['ict'],
            Regime.LOW_VOLATILE: self._all_strategies['scalping'],
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
        Counter-trend sinyalleri bloklanır (allowed=False).
        Bu filtre BTC/BNB/SOL gibi coinlerde kanıtlanmış şekilde çalışıyor.
        """
        if signal is None or signal.direction is None or df_4h is None:
            return signal

        ctx = self.mtf_analyzer.analyze_htf(df_4h, symbol)
        adj = self.mtf_analyzer.get_signal_adjustment(ctx, signal.direction)

        if not adj.allowed:
            return Signal(None, 0.0, signal.strategy_name, "", f"MTF RED: {adj.reason}", hard_pass=False)

        signal.soft_score = min(5, signal.soft_score + adj.score_modifier)
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

        # Multi-strategy ensemble: tüm uygun stratejileri dene, en iyisini seç
        primary_signal = self._best_signal_ensemble(df, regime)

        # 3. MTF Filtrelemesi (4h trend yönü)
        if primary_signal.is_valid and df_4h is not None:
            primary_signal = self._process_signal_with_mtf(primary_signal, df_4h, symbol)

        if not primary_signal.is_valid:
            self._debug_counts['no_signal'] += 1
            self._track_reason(f"{primary_signal.reason}_{rv}")
            return self._hold_decision(regime, f"Signal rejected ({primary_signal.reason})")

        # 3b. 1h RSI Overbought/Oversold Guard
        if 'rsi' in df.columns:
            rsi_1h = float(df['rsi'].iloc[-1])
            if not np.isnan(rsi_1h):
                if primary_signal.direction == 'LONG' and rsi_1h > 70:
                    self._debug_counts['no_signal'] += 1
                    self._track_reason(f"rsi_overbought_{rv}")
                    return self._hold_decision(
                        regime, f"1h RSI overbought: {rsi_1h:.0f}>70 — LONG blocked"
                    )
                if primary_signal.direction == 'SHORT' and rsi_1h < 30:
                    self._debug_counts['no_signal'] += 1
                    self._track_reason(f"rsi_oversold_{rv}")
                    return self._hold_decision(
                        regime, f"1h RSI oversold: {rsi_1h:.0f}<30 — SHORT blocked"
                    )

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

        # 4a. Giriş Lokasyonu Kalitesi (1h swing range + EMA proximity)
        entry_type = _safe_signal_attr(primary_signal, 'entry_type', 'none')
        loc_ok, loc_reason = self._check_entry_location(df, primary_signal.direction, entry_type)
        if not loc_ok:
            self._debug_counts['no_signal'] += 1
            self._track_reason(f"entry_location_{rv}")
            return self._hold_decision(regime, f"Entry location: {loc_reason}")
        primary_signal.reason += f" | Loc: {loc_reason}"

        self._last_signal = primary_signal

        # 4b. HTF Feature Injection — Meta model için 6 HTF feature'ı df'e ekle
        df = self._inject_htf_features(df, df_4h, symbol)

        # 5. Meta-Filter (Soft Scaling)
        # Hard gate SADECE çok düşük confidence'da → geri kalanı position size ile ölçekle
        # Gate 2 (should_trade) kaldırıldı: threshold 0.64-0.65 her şeyi blokluyor,
        # soft-scaling zaten düşük meta'lı sinyallere küçük pozisyon veriyor.
        MINIMUM_META = 0.30
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

    def _best_signal_ensemble(self, df: pd.DataFrame, regime: Regime) -> Signal:
        """
        Multi-Strategy Ensemble: Rejime uygun TÜM stratejileri dene,
        en yüksek (soft_score * confidence) olan sinyali seç.
        Eğer hiçbiri sinyal üretmezse, cross-regime stratejileri dene.
        """
        candidates = []

        # 1. Rejime uygun stratejileri dene
        strategy_names = self._regime_strategies.get(regime, [])
        for sname in strategy_names:
            strat = self._all_strategies.get(sname)
            if not strat:
                continue
            try:
                signal = strat.generate_signal(df)
                if signal.is_valid:
                    candidates.append(signal)
            except Exception:
                pass

        # 2. v2.1: Cross-regime secondary kaldırıldı (backtest: %25 WR, -8.65% PnL)
        #    Rejime uygun strateji sinyal vermezse → HOLD (daha iyi)

        # 3. En iyi sinyali seç: soft_score * confidence
        if candidates:
            best = max(candidates, key=lambda s: (s.soft_score or 0) * s.confidence)
            return best

        return Signal(None, 0.0, "none", regime.value, "Tüm stratejiler reddedildi")

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
                rsi_high, rsi_low = 78, 22
                mom_limit = 2.0
                body_check = True
            elif regime_lower == 'mean_reverting':
                rsi_high, rsi_low = 72, 28
                mom_limit = 1.5
                body_check = True
            else:
                rsi_high, rsi_low = 75, 25
                mom_limit = 2.0
                body_check = True

            # ── MTF Override: Güçlü 4h trend → daha da gevşet ──
            if mtf_score >= 4:
                rsi_high = min(rsi_high + 4, 92)
                rsi_low = max(rsi_low - 4, 8)
                mom_limit *= 1.5

            # ── Ardışık mum kontrolü: son 5+ mum aynı yönde → tükenme riski ──
            consec_bear = 0
            consec_bull = 0
            if len(close) >= 6:
                for i in range(-1, -6, -1):
                    if float(close.iloc[i]) < float(df_ltf['open'].iloc[i]):
                        consec_bear += 1
                    elif float(close.iloc[i]) > float(df_ltf['open'].iloc[i]):
                        consec_bull += 1
                    else:
                        break

            # ── 15m Yapısal Onay: HL/LH + EMA alignment ──
            ema9 = close.ewm(span=9, adjust=False).mean()
            ema21 = close.ewm(span=21, adjust=False).mean()
            ema9_val = float(ema9.iloc[-1])
            ema21_val = float(ema21.iloc[-1])
            ema_bull = ema9_val > ema21_val
            ema_bear = ema9_val < ema21_val

            # Son 10 bar'da Higher Low / Lower High arama
            has_hl = False  # Higher Low (bullish structure)
            has_lh = False  # Lower High (bearish structure)
            if len(low) >= 10:
                lows_10 = [float(low.iloc[i]) for i in range(-10, 0)]
                highs_10 = [float(high.iloc[i]) for i in range(-10, 0)]
                # Son low > önceki minimum low = Higher Low
                min_low_first_half = min(lows_10[:5]) if len(lows_10) >= 5 else lows_10[0]
                min_low_second_half = min(lows_10[5:]) if len(lows_10) >= 5 else lows_10[-1]
                has_hl = min_low_second_half > min_low_first_half
                # Son high < önceki maximum high = Lower High
                max_high_first_half = max(highs_10[:5]) if len(highs_10) >= 5 else highs_10[0]
                max_high_second_half = max(highs_10[5:]) if len(highs_10) >= 5 else highs_10[-1]
                has_lh = max_high_second_half < max_high_first_half

            if direction == 'LONG':
                if rsi_val > rsi_high:
                    return False, f"RSI aşırı alınmış ({rsi_val:.0f}, limit={rsi_high})"
                if mom_3 > mom_limit:
                    return False, f"Hareket kaçırılmış (mom={mom_3:.1f}%, limit={mom_limit:.1f}%)"
                if body_check and body_ratio < 0.15 and rsi_val > 65:
                    return False, f"Zayıf mum + yüksek RSI"
                if consec_bull >= 5:
                    return False, f"5+ ardışık bull mum — tükenme (consec={consec_bull})"
                # Yapısal onay: EMA bull veya Higher Low olmalı
                struct_ok = ema_bull or has_hl
                struct_tag = f"ema{'✓' if ema_bull else '✗'} hl{'✓' if has_hl else '✗'}"
                if not struct_ok and regime_lower != 'mean_reverting':
                    return False, f"15m yapı yok ({struct_tag} rsi={rsi_val:.0f})"
                return True, f"OK (rsi={rsi_val:.0f} mom={mom_3:.1f}% {struct_tag})"

            elif direction == 'SHORT':
                if rsi_val < rsi_low:
                    return False, f"RSI aşırı satılmış ({rsi_val:.0f}, limit={rsi_low})"
                if mom_3 < -mom_limit:
                    return False, f"Hareket kaçırılmış (mom={mom_3:.1f}%, limit={-mom_limit:.1f}%)"
                if body_check and body_ratio < 0.15 and rsi_val < 35:
                    return False, f"Zayıf mum + düşük RSI"
                if consec_bear >= 5:
                    return False, f"5+ ardışık bear mum — tükenme (consec={consec_bear})"
                # Yapısal onay: EMA bear veya Lower High olmalı
                struct_ok = ema_bear or has_lh
                struct_tag = f"ema{'✓' if ema_bear else '✗'} lh{'✓' if has_lh else '✗'}"
                if not struct_ok and regime_lower != 'mean_reverting':
                    return False, f"15m yapı yok ({struct_tag} rsi={rsi_val:.0f})"
                return True, f"OK (rsi={rsi_val:.0f} mom={mom_3:.1f}% {struct_tag})"

            return True, "neutral"

        except Exception as e:
            return True, f"timing_error ({e})"

    def _check_entry_location(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_type: str = "pullback",
    ) -> Tuple[bool, str]:
        """
        Giriş Lokasyonu Kalite Kontrolü — Premium/Discount Zone

        Pullback SHORT → fiyat range'in üst kısmında olmalı (premium zone)
        Pullback LONG  → fiyat range'in alt kısmında olmalı (discount zone)

        Kontroller:
          1. Swing range pozisyonu (son 20 bar high/low)
          2. EMA21 mesafesi (pullback kalitesi)
          3. Swing low/high'a ATR-bazlı yakınlık
        """
        try:
            lookback = min(20, len(df) - 1)
            if lookback < 10:
                return True, "insufficient_data"

            recent = df.iloc[-lookback:]
            swing_high = float(recent['high'].max())
            swing_low = float(recent['low'].min())
            cp = float(df['close'].iloc[-1])

            swing_range = swing_high - swing_low
            if swing_range <= 0 or swing_range / cp < 0.005:
                return True, "flat_range"

            # 0.0 = dip, 1.0 = tepe
            pos = (cp - swing_low) / swing_range

            atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else cp * 0.01

            # EMA21 mesafesi
            ema21 = float(df['ema21'].iloc[-1]) if 'ema21' in df.columns else cp
            ema_dist_pct = (cp - ema21) / ema21 * 100 if ema21 > 0 else 0.0

            info = f"pos={pos:.0%} ema21={ema_dist_pct:+.1f}%"

            is_pullback = entry_type in ('pullback', 'none')

            if direction == 'SHORT':
                if is_pullback:
                    # Destek dibine short yasak — alt %35
                    if pos < 0.35:
                        return False, f"Discount zone SHORT ({info})"
                    # EMA21'in çok altında = hareket kaçırılmış
                    if ema_dist_pct < -1.5:
                        return False, f"EMA21 altı uzama ({info})"
                    # Swing low'a çok yakın
                    if (cp - swing_low) < 0.7 * atr:
                        return False, f"Swing low'a çok yakın ({info})"
                else:  # breakout, ict_setup, smart_money
                    # Aşırı uzama kontrolü (gevşek)
                    if ema_dist_pct < -3.0:
                        return False, f"Breakout aşırı uzama ({info})"
                    # Range'in en dibinde bile olmamalı
                    if pos < 0.20:
                        return False, f"Dip bölgesinde SHORT ({info})"
                    # Yükselen tepe koruması: range tepesi + EMA üstü
                    if pos > 0.80 and ema_dist_pct > 0.8:
                        return False, f"Yükselen tepe SHORT ({info})"

            elif direction == 'LONG':
                if is_pullback:
                    # Direnç tepesine long yasak — üst %35
                    if pos > 0.65:
                        return False, f"Premium zone LONG ({info})"
                    # EMA21'in çok üstünde = hareket kaçırılmış
                    if ema_dist_pct > 1.5:
                        return False, f"EMA21 üstü uzama ({info})"
                    # Swing high'a çok yakın
                    if (swing_high - cp) < 0.7 * atr:
                        return False, f"Swing high'a çok yakın ({info})"
                else:  # breakout, ict_setup, smart_money
                    if ema_dist_pct > 3.0:
                        return False, f"Breakout aşırı uzama ({info})"
                    if pos > 0.80:
                        return False, f"Tepe bölgesinde LONG ({info})"
                    # Düşen bıçak koruması: range dibi + EMA altı
                    if pos < 0.20 and ema_dist_pct < -0.8:
                        return False, f"Düşen bıçak LONG ({info})"

            return True, f"OK ({info})"

        except Exception as e:
            return True, f"location_error ({e})"

    def _inject_htf_features(self, df: pd.DataFrame, df_4h: Optional[pd.DataFrame] = None, symbol: str = "UNKNOWN") -> pd.DataFrame:
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
                ctx = self.mtf_analyzer.analyze_htf(df_4h, symbol)
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
        if soft_score >= 5:
            score_mult = 1.0
        elif soft_score == 4:
            score_mult = 0.75
        elif soft_score == 3:
            score_mult = 0.50
        else:
            score_mult = 0.0

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

        # ── Step 2: Multi-Strategy Ensemble ──
        primary_signal = self._best_signal_ensemble(df, regime)

        # Show which strategies were tried
        tried_strategies = self._regime_strategies.get(regime, [])
        if primary_signal and primary_signal.is_valid:
            step("2_strategy_ensemble", "✅", {
                "tried": tried_strategies,
                "selected": primary_signal.strategy_name,
                "direction": primary_signal.direction,
                "confidence": round(primary_signal.confidence, 3),
                "soft_score": _safe_signal_attr(primary_signal, 'soft_score', 0),
                "entry_type": _safe_signal_attr(primary_signal, 'entry_type', 'none'),
                "tp_price": _safe_signal_attr(primary_signal, 'tp_price', 0),
                "sl_price": _safe_signal_attr(primary_signal, 'sl_price', 0),
                "reason": primary_signal.reason[:120],
            })
        else:
            step("2_strategy_ensemble", "❌ NO SIGNAL", {
                "tried": tried_strategies,
                "reason": primary_signal.reason[:120] if primary_signal else "no strategy",
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

        # ── Step 4a: Entry Location Quality ──
        entry_type = _safe_signal_attr(primary_signal, 'entry_type', 'none')
        loc_ok, loc_reason = self._check_entry_location(df, primary_signal.direction, entry_type)
        if loc_ok:
            step("4a_entry_location", "✅", {
                "entry_type": entry_type,
                "result": loc_reason,
            })
        else:
            step("4a_entry_location", "❌ REJECTED", {
                "entry_type": entry_type,
                "result": loc_reason,
            })
            trace["final_action"] = "HOLD"
            trace["hold_reason"] = f"Entry location: {loc_reason}"
            return trace

        # ── Step 4c: HTF Feature Injection ──
        df = self._inject_htf_features(df, df_4h, symbol)

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
        if final_confidence < 0.40:
            quality_issues.append(f"confidence={final_confidence:.3f} < 0.40")

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
