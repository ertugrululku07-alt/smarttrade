"""
Volatility Expansion Strategy

Rejim: HIGH_VOLATILE
Mantık: Yüksek volatilitede breakout'ları yakala
WR: %45-50 (ama R:R yüksek)
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, Signal


class VolatilityStrategy(BaseStrategy):

    name = "volatility"
    regime = "high_volatile"
    default_tp_mult = 3.0   # Geniş TP (büyük hareketler)
    default_sl_mult = 1.5   # Geniş SL (volatilite yüksek)

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 50:
            return self._no_signal("Yetersiz veri")

        c = df['close'].iloc[-1]
        h = df['high'].iloc[-1]
        lo = df['low'].iloc[-1]

        # BOS (Break of Structure)
        bos_bull = df['bos_bullish'].iloc[-1] if 'bos_bullish' in df.columns else 0
        bos_bear = df['bos_bearish'].iloc[-1] if 'bos_bearish' in df.columns else 0

        # Volume spike
        vol_ratio = df['vol_ratio_20'].iloc[-1] if 'vol_ratio_20' in df.columns else 1.0

        # Son N bar'ın range'i
        high_10 = df['high'].tail(10).max()
        low_10 = df['low'].tail(10).min()
        range_10 = high_10 - low_10
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else range_10 * 0.1

        # Breakout seviyesi
        high_20 = df['high'].tail(20).max()
        low_20 = df['low'].tail(20).min()

        macd_hist = df['macd_hist'].iloc[-1]
        rsi = df['rsi'].iloc[-1]

        # ═══════════════════════════════════════════════════════
        # Bullish Breakout
        # ═══════════════════════════════════════════════════════

        # Range üstü kırılım + volume spike + BOS
        if (c > high_20 * 0.998 and vol_ratio > 2.0 and bos_bull):
            return self._long_signal(
                soft_score=4,
                reason=f"Breakout^: Close>{high_20:.2f} Vol={vol_ratio:.1f}x BOSv",
                entry_price=c,
            )

        # Volume spike + yukarı momentum
        if (vol_ratio > 2.5 and macd_hist > 0 and rsi > 55
                and c > df['ema9'].iloc[-1]):
            return self._long_signal(
                soft_score=3,
                reason=f"Volume spike^: Vol={vol_ratio:.1f}x MACD+ RSI={rsi:.0f}",
                entry_price=c,
            )

        # ═══════════════════════════════════════════════════════
        # Bearish Breakout
        # ═══════════════════════════════════════════════════════

        if (c < low_20 * 1.002 and vol_ratio > 2.0 and bos_bear):
            return self._short_signal(
                soft_score=4,
                reason=f"Breakoutv: Close<{low_20:.2f} Vol={vol_ratio:.1f}x BOSv",
                entry_price=c,
            )

        if (vol_ratio > 2.5 and macd_hist < 0 and rsi < 45
                and c < df['ema9'].iloc[-1]):
            return self._short_signal(
                soft_score=3,
                reason=f"Volume spikev: Vol={vol_ratio:.1f}x MACD- RSI={rsi:.0f}",
                entry_price=c,
            )

        return self._no_signal("Breakout koşulları sağlanmadı")
