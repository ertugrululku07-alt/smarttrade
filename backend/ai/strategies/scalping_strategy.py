"""
Scalping / Low Volatility Strategy

Rejim: LOW_VOLATILE
Mantık: Düşük volatilitede küçük hareketlerden kazan
WR: %55-60 (küçük TP/SL)

NOT: Düşük volatilite genelde "fırtına öncesi sessizlik"
     Bu yüzden çok agresif olma, sinyaller seçici olmalı
"""

import pandas as pd
from .base_strategy import BaseStrategy, Signal


class ScalpingStrategy(BaseStrategy):

    name = "scalping"
    regime = "low_volatile"
    default_tp_mult = 1.0   # Çok kısa TP
    default_sl_mult = 0.8   # Çok dar SL

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 50:
            return self._no_signal("Yetersiz veri")

        c = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        bb_pos = df['bb_pos'].iloc[-1]
        stoch_k = df['stoch_k'].iloc[-1]
        stoch_d = df['stoch_d'].iloc[-1]

        vwap_dist = df['vwap_dist'].iloc[-1] if 'vwap_dist' in df.columns else 0
        vol_ratio = df['vol_ratio_20'].iloc[-1] if 'vol_ratio_20' in df.columns else 1.0

        # Düşük volatilitede sadece ÇOK NET sinyaller
        # BB bandının uçlarında + stoch cross

        # ═══════════════════════════════════════════════════════
        # LONG: BB alt bandı + RSI düşük + Stoch dönüyor
        # ═══════════════════════════════════════════════════════

        if (bb_pos < 0.10 and rsi < 35 and stoch_k > stoch_d
                and stoch_k < 25):
            return self._long_signal(
                confidence=0.65,
                reason=f"Scalp LONG: BB={bb_pos:.2f} RSI={rsi:.0f} StochCross↑",
                entry_price=c,
            )

        # VWAP altında + momentum dönüyor
        if (vwap_dist < -0.003 and rsi < 40 and stoch_k > stoch_d):
            return self._long_signal(
                confidence=0.55,
                reason=f"VWAP scalp↑: VWAP_dist={vwap_dist:.4f} RSI={rsi:.0f}",
                entry_price=c,
            )

        # ═══════════════════════════════════════════════════════
        # SHORT: BB üst bandı + RSI yüksek + Stoch dönüyor
        # ═══════════════════════════════════════════════════════

        if (bb_pos > 0.90 and rsi > 65 and stoch_k < stoch_d
                and stoch_k > 75):
            return self._short_signal(
                confidence=0.65,
                reason=f"Scalp SHORT: BB={bb_pos:.2f} RSI={rsi:.0f} StochCross↓",
                entry_price=c,
            )

        if (vwap_dist > 0.003 and rsi > 60 and stoch_k < stoch_d):
            return self._short_signal(
                confidence=0.55,
                reason=f"VWAP scalp↓: VWAP_dist={vwap_dist:.4f} RSI={rsi:.0f}",
                entry_price=c,
            )

        # Düşük volatilitede sinyal yoksa → BEKLE
        # Genellikle büyük hareket öncesi sessizlik
        return self._no_signal("Düşük vol, net sinyal yok → Bekle")
