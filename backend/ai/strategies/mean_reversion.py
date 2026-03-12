"""
Mean Reversion Strategy v1.1

Rejim: MEAN_REVERTING

v1.1 Değişiklikler:
  - ✅ Kolon güvenliği eklendi (_validate_columns)
  - ✅ NaN koruması eklendi
  - ✅ entry_type set edildi
  - ✅ Reason mesajları standardize
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, Signal


class MeanReversionStrategy(BaseStrategy):

    name = "mean_reversion"
    regime = "mean_reverting"
    default_tp_mult = 2.0
    default_sl_mult = 1.0

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 50:
            return self._no_signal("Yetersiz veri")

        # ── Kolon Güvenliği ──────────────────────────────
        required = ['close', 'rsi', 'stoch_k', 'stoch_d', 'bb_pos']
        col_err = self._validate_columns(df, required)
        if col_err:
            return self._no_signal(col_err)

        # ── Data Extraction ─────────────────────────────
        c = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        rsi_prev = df['rsi'].iloc[-2]

        # NaN koruması
        if pd.isna(rsi) or pd.isna(rsi_prev):
            return self._no_signal("RSI NaN")

        stoch_k = df['stoch_k'].iloc[-1]
        stoch_d = df['stoch_d'].iloc[-1]
        bb_pos = df['bb_pos'].iloc[-1]

        if any(pd.isna(v) for v in [stoch_k, stoch_d, bb_pos]):
            return self._no_signal("Osilatör değerleri NaN")

        # ── Opsiyonel Kolonlar (güvenli) ─────────────────
        adx = self._safe_get(df, 'adx', default=20.0)
        hurst = self._safe_get(df, 'hurst', default=0.5)
        candle_dir = self._safe_get(df, 'candle_dir', default=0)
        lower_wick = self._safe_get(df, 'lower_wick', default=0.0)
        upper_wick = self._safe_get(df, 'upper_wick', default=0.0)

        # ── Trend Filtresi ───────────────────────────────
        if adx > 25:
            return self._no_signal(
                f"ADX çok yüksek ({adx:.0f}) — Trend var"
            )

        if hurst > 0.48:
            return self._no_signal(
                f"Hurst çok yüksek ({hurst:.2f}) — Trend riski"
            )

        # ── RSI Momentum (son 3 bar) ────────────────────
        rsi_3_ago = df['rsi'].iloc[-3] if len(df) > 3 else rsi
        if pd.isna(rsi_3_ago):
            rsi_3_ago = rsi
        rsi_momentum = rsi - rsi_3_ago

        # ═══════════════════════════════════════════════════
        # LONG Sinyalleri
        # ═══════════════════════════════════════════════════

        # L1: Extreme oversold
        if rsi < 25 and stoch_k < 15 and bb_pos < 0.05:
            reversal_candle = (lower_wick > 0.003) or (candle_dir > 0)
            score = 5 if reversal_candle else 4
            return self._long_signal(
                soft_score=score,
                reason=f"Extreme oversold: RSI={rsi:.0f} "
                       f"Stoch={stoch_k:.0f} BB={bb_pos:.2f}",
                entry_price=c,
                entry_type="pullback",
            )

        # L2: Güçlü oversold
        if rsi < 30 and stoch_k < 25 and bb_pos < 0.15:
            rsi_turning = rsi > rsi_prev
            score = 4 if rsi_turning else 3
            return self._long_signal(
                soft_score=score,
                reason=f"Oversold: RSI={rsi:.0f} "
                       f"{'dönüyor^' if rsi_turning else 'düşüyor'}",
                entry_price=c,
                entry_type="pullback",
            )

        # L3: Orta oversold + dönüş onayı
        if rsi < 35 and stoch_k < 30 and bb_pos < 0.20:
            stoch_cross = stoch_k > stoch_d
            rsi_turning = rsi > rsi_prev and rsi_momentum > 0
            if stoch_cross and rsi_turning:
                return self._long_signal(
                    soft_score=3,
                    reason=f"Oversold dönüş: RSI={rsi:.0f}^ StochX^",
                    entry_price=c,
                    entry_type="pullback",
                )

        # L4: Hafif oversold + güçlü dönüş
        if rsi < 38 and bb_pos < 0.25:
            stoch_cross = stoch_k > stoch_d
            rsi_turning = rsi > rsi_prev and rsi_momentum > 2
            lower_wick_reversal = lower_wick > 0.005
            if stoch_cross and rsi_turning and lower_wick_reversal:
                return self._long_signal(
                    soft_score=3,
                    reason=f"Dönüş sinyali: RSI={rsi:.0f}^ wick "
                           f"mom={rsi_momentum:.1f}",
                    entry_price=c,
                    entry_type="pullback",
                )

        # L5: BB alt bandı + stoch cross
        if bb_pos < 0.15 and stoch_k > stoch_d and stoch_k < 35:
            if rsi < 42 and rsi > rsi_prev:
                return self._long_signal(
                    soft_score=3,
                    reason=f"BB alt+stoch: BB={bb_pos:.2f} "
                           f"RSI={rsi:.0f}^",
                    entry_price=c,
                    entry_type="pullback",
                )

        # ═══════════════════════════════════════════════════
        # SHORT Sinyalleri
        # ═══════════════════════════════════════════════════

        # S1: Extreme overbought
        if rsi > 75 and stoch_k > 85 and bb_pos > 0.95:
            reversal_candle = (upper_wick > 0.003) or (candle_dir < 0)
            score = 5 if reversal_candle else 4
            return self._short_signal(
                soft_score=score,
                reason=f"Extreme overbought: RSI={rsi:.0f} "
                       f"Stoch={stoch_k:.0f} BB={bb_pos:.2f}",
                entry_price=c,
                entry_type="pullback",
            )

        # S2: Güçlü overbought
        if rsi > 70 and stoch_k > 75 and bb_pos > 0.85:
            rsi_turning = rsi < rsi_prev
            score = 4 if rsi_turning else 3
            return self._short_signal(
                soft_score=score,
                reason=f"Overbought: RSI={rsi:.0f} "
                       f"{'dönüyorv' if rsi_turning else 'yükseliyor'}",
                entry_price=c,
                entry_type="pullback",
            )

        # S3: Orta overbought + dönüş onayı
        if rsi > 65 and stoch_k > 70 and bb_pos > 0.80:
            stoch_cross = stoch_k < stoch_d
            rsi_turning = rsi < rsi_prev and rsi_momentum < 0
            if stoch_cross and rsi_turning:
                return self._short_signal(
                    soft_score=3,
                    reason=f"Overbought dönüş: RSI={rsi:.0f}v StochXv",
                    entry_price=c,
                    entry_type="pullback",
                )

        # S4: Hafif overbought + güçlü dönüş
        if rsi > 62 and bb_pos > 0.75:
            stoch_cross = stoch_k < stoch_d
            rsi_turning = rsi < rsi_prev and rsi_momentum < -2
            upper_wick_reversal = upper_wick > 0.005
            if stoch_cross and rsi_turning and upper_wick_reversal:
                return self._short_signal(
                    soft_score=3,
                    reason=f"Dönüş sinyali: RSI={rsi:.0f}v wick "
                           f"mom={rsi_momentum:.1f}",
                    entry_price=c,
                    entry_type="pullback",
                )

        # S5: BB üst bandı + stoch cross
        if bb_pos > 0.85 and stoch_k < stoch_d and stoch_k > 65:
            if rsi > 58 and rsi < rsi_prev:
                return self._short_signal(
                    soft_score=3,
                    reason=f"BB üst+stoch: BB={bb_pos:.2f} "
                           f"RSI={rsi:.0f}v",
                    entry_price=c,
                    entry_type="pullback",
                )

        return self._no_signal(
            f"RSI={rsi:.0f} nötr bölgede",
            hard_pass=True,
            soft_score=0
        )
