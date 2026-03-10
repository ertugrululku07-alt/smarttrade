"""
Momentum Strategy v1.0 (Original)

Rejim: TRENDING
Mantık: Trend yönünde güçlü kırılımları yakala
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):

    name = "momentum"
    regime = "trending"
    default_tp_mult = 2.5
    default_sl_mult = 1.3

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 50:
            return self._no_signal("Yetersiz veri")

        c = df['close'].iloc[-1]
        ema9 = df['ema9'].iloc[-1]
        ema21 = df['ema21'].iloc[-1]
        ema50 = df['ema50'].iloc[-1]
        macd_hist = df['macd_hist'].iloc[-1]
        macd_hist_prev = df['macd_hist'].iloc[-2]
        adx = df['adx'].iloc[-1]
        di_plus = df['di_plus'].iloc[-1]
        di_minus = df['di_minus'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        vol_ratio = df['vol_ratio_20'].iloc[-1] if 'vol_ratio_20' in df.columns else 1.0

        bullish_ema = ema9 > ema21 > ema50
        bearish_ema = ema9 < ema21 < ema50
        macd_bullish = macd_hist > 0 and macd_hist > macd_hist_prev
        macd_bearish = macd_hist < 0 and macd_hist < macd_hist_prev
        strong_trend = adx > 25
        very_strong_trend = adx > 35
        vol_confirm = vol_ratio > 1.2

        # LONG
        if (bullish_ema and macd_bullish and very_strong_trend
                and di_plus > di_minus and vol_confirm and 40 < rsi < 70):
            return self._long_signal(0.85,
                f"Güçlü trend↑: ADX={adx:.0f} Vol={vol_ratio:.1f}", c)

        if (bullish_ema and macd_bullish and strong_trend
                and di_plus > di_minus and 35 < rsi < 75):
            return self._long_signal(0.70,
                f"Trend devam↑: ADX={adx:.0f}", c)

        if (ema9 > ema21 and macd_hist > 0 and adx > 22
                and di_plus > di_minus):
            return self._long_signal(0.55,
                f"Zayıf trend↑: ADX={adx:.0f}", c)

        # SHORT
        if (bearish_ema and macd_bearish and very_strong_trend
                and di_minus > di_plus and vol_confirm and 30 < rsi < 60):
            return self._short_signal(0.85,
                f"Güçlü trend↓: ADX={adx:.0f} Vol={vol_ratio:.1f}", c)

        if (bearish_ema and macd_bearish and strong_trend
                and di_minus > di_plus and 25 < rsi < 65):
            return self._short_signal(0.70,
                f"Trend devam↓: ADX={adx:.0f}", c)

        if (ema9 < ema21 and macd_hist < 0 and adx > 22
                and di_minus > di_plus):
            return self._short_signal(0.55,
                f"Zayıf trend↓: ADX={adx:.0f}", c)

        return self._no_signal("Trend koşulları sağlanmadı")
