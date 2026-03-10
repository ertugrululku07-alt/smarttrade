"""
Mean Reversion Strategy v1.0 (Original) + Trend Filter

Rejim: MEAN_REVERTING
v1.0 + ADX trend filtresi
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, Signal


class MeanReversionStrategy(BaseStrategy):

    name = "mean_reversion"
    regime = "mean_reverting"
    default_tp_mult = 2.0  # 1.5 -> 2.0 (Daha yüksek R:R)
    default_sl_mult = 1.0

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 50:
            return self._no_signal("Yetersiz veri")

        c = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        rsi_prev = df['rsi'].iloc[-2]
        stoch_k = df['stoch_k'].iloc[-1]
        stoch_d = df['stoch_d'].iloc[-1]
        bb_pos = df['bb_pos'].iloc[-1]
        adx = df['adx'].iloc[-1] if 'adx' in df.columns else 20.0
        candle_dir = df['candle_dir'].iloc[-1] if 'candle_dir' in df.columns else 0
        lower_wick = df['lower_wick'].iloc[-1] if 'lower_wick' in df.columns else 0
        upper_wick = df['upper_wick'].iloc[-1] if 'upper_wick' in df.columns else 0

        # Trend kontrolü — güçlü trend varsa mean reversion yapma
        if adx > 30:
            return self._no_signal(f"ADX cok yuksek ({adx:.0f})")

        # ── v1.3: RSI momentum (son 3 bar'ın yönü) ──────────
        rsi_3_ago = df['rsi'].iloc[-3] if len(df) > 3 else rsi
        rsi_momentum = rsi - rsi_3_ago  # Pozitif = yükseliyor

        # ═══════════════════════════════════════════════════════
        # LONG Sinyalleri — v1.3: GENİŞLETİLMİŞ
        # ═══════════════════════════════════════════════════════

        # Seviye 1: Çok güçlü oversold (orijinal)
        if rsi < 25 and stoch_k < 15 and bb_pos < 0.05:
            reversal_candle = (lower_wick > 0.003) or (candle_dir > 0)
            conf = 0.85 if reversal_candle else 0.75
            return self._long_signal(conf,
                f"Extreme oversold: RSI={rsi:.0f} Stoch={stoch_k:.0f}", c)

        # Seviye 2: Güçlü oversold (orijinal)
        if rsi < 30 and stoch_k < 25 and bb_pos < 0.15:
            rsi_turning = rsi > rsi_prev
            conf = 0.70 if rsi_turning else 0.58
            return self._long_signal(conf,
                f"Oversold: RSI={rsi:.0f}", c)

        # Seviye 3: Orta oversold + dönüş onayı (YENİ)
        if rsi < 35 and stoch_k < 30 and bb_pos < 0.20:
            stoch_cross = stoch_k > stoch_d
            rsi_turning = rsi > rsi_prev and rsi_momentum > 0
            if stoch_cross and rsi_turning:
                return self._long_signal(0.62,
                    f"Oversold donus: RSI={rsi:.0f}↑ StochX↑", c)

        # Seviye 4: Hafif oversold + güçlü dönüş (YENİ)
        if rsi < 38 and bb_pos < 0.25:
            stoch_cross = stoch_k > stoch_d
            rsi_turning = rsi > rsi_prev and rsi_momentum > 2
            lower_wick_reversal = lower_wick > 0.005
            if stoch_cross and rsi_turning and lower_wick_reversal:
                return self._long_signal(0.55,
                    f"Donus sinyali: RSI={rsi:.0f}↑ wick", c)

        # Seviye 5: BB alt bandı yakını + stoch cross (YENİ)
        if bb_pos < 0.15 and stoch_k > stoch_d and stoch_k < 35:
            if rsi < 42 and rsi > rsi_prev:
                return self._long_signal(0.52,
                    f"BB alt+stoch: BB={bb_pos:.2f} RSI={rsi:.0f}↑", c)

        # ═══════════════════════════════════════════════════════
        # SHORT Sinyalleri — v1.3: GENİŞLETİLMİŞ
        # ═══════════════════════════════════════════════════════

        # Seviye 1: Çok güçlü overbought
        if rsi > 75 and stoch_k > 85 and bb_pos > 0.95:
            reversal_candle = (upper_wick > 0.003) or (candle_dir < 0)
            conf = 0.85 if reversal_candle else 0.75
            return self._short_signal(conf,
                f"Extreme overbought: RSI={rsi:.0f} Stoch={stoch_k:.0f}", c)

        # Seviye 2: Güçlü overbought
        if rsi > 70 and stoch_k > 75 and bb_pos > 0.85:
            rsi_turning = rsi < rsi_prev
            conf = 0.70 if rsi_turning else 0.58
            return self._short_signal(conf,
                f"Overbought: RSI={rsi:.0f}", c)

        # Seviye 3: Orta overbought + dönüş onayı (YENİ)
        if rsi > 65 and stoch_k > 70 and bb_pos > 0.80:
            stoch_cross = stoch_k < stoch_d
            rsi_turning = rsi < rsi_prev and rsi_momentum < 0
            if stoch_cross and rsi_turning:
                return self._short_signal(0.62,
                    f"Overbought donus: RSI={rsi:.0f}↓ StochX↓", c)

        # Seviye 4: Hafif overbought + güçlü dönüş (YENİ)
        if rsi > 62 and bb_pos > 0.75:
            stoch_cross = stoch_k < stoch_d
            rsi_turning = rsi < rsi_prev and rsi_momentum < -2
            upper_wick_reversal = upper_wick > 0.005
            if stoch_cross and rsi_turning and upper_wick_reversal:
                return self._short_signal(0.55,
                    f"Donus sinyali: RSI={rsi:.0f}↓ wick", c)

        # Seviye 5: BB üst bandı yakını + stoch cross (YENİ)
        if bb_pos > 0.85 and stoch_k < stoch_d and stoch_k > 65:
            if rsi > 58 and rsi < rsi_prev:
                return self._short_signal(0.52,
                    f"BB ust+stoch: BB={bb_pos:.2f} RSI={rsi:.0f}↓", c)

        return self._no_signal(f"RSI={rsi:.0f} notr bolgede")
