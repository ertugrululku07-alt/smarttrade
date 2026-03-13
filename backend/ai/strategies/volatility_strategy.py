"""
Volatility Expansion Strategy v2.0

Rejim: HIGH_VOLATILE
Mantik: Yuksek volatilitede breakout + BOS + volume spike yakala
Hedef WR: %45-50 (yuksek R:R ile pozitif beklenen deger)

v2.0 Degisiklikler:
  - Hard/soft filter framework eklendi
  - Explicit TP/SL hesaplama (swing bazli)
  - Entry type atandi
  - NaN korumasi eklendi
  - Kolon guvenligi eklendi
  - Minimum RR kontrolu (2.0:1)
  - BOS + FVG + Volume confluence sistemi
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, Signal


class VolatilityStrategy(BaseStrategy):

    name = "volatility"
    regime = "high_volatile"
    default_tp_mult = 3.0    # Genis TP (buyuk hareketler)
    default_sl_mult = 1.2    # Kontrollü SL
    MIN_RR = 2.0

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 60:
            return self._no_signal("Yetersiz veri")

        # -- Kolon Guvenligi ------------------------------------
        required = ['close', 'high', 'low', 'open', 'volume', 'atr']
        col_err = self._validate_columns(df, required)
        if col_err:
            return self._no_signal(col_err)

        # -- Data Extraction ------------------------------------
        last = df.iloc[-1]
        prev = df.iloc[-2]
        c = float(last['close'])
        h = float(last['high'])
        lo = float(last['low'])
        o = float(last['open'])
        atr = float(last['atr'])
        vol = float(last['volume'])

        # ATR korumasi
        if pd.isna(atr) or atr < 1e-10:
            return self._no_signal("ATR gecersiz")

        vol_ma = df['volume'].rolling(20, min_periods=5).mean().iloc[-1]
        if pd.isna(vol_ma) or vol_ma < 1e-10:
            vol_ma = vol

        # Opsiyonel kolonlar
        bos_bull = self._safe_get(df, 'bos_bullish', default=0)
        bos_bear = self._safe_get(df, 'bos_bearish', default=0)
        vol_ratio = self._safe_get(df, 'vol_ratio_20', default=1.0)
        rsi = self._safe_get(df, 'rsi', default=50.0)
        macd_hist = self._safe_get(df, 'macd_hist', default=0.0)
        ema9 = self._safe_get(df, 'ema9', default=c)
        adx = self._safe_get(df, 'adx', default=20.0)
        fvg_bull = self._safe_get(df, 'fvg_bull', default=0)
        fvg_bear = self._safe_get(df, 'fvg_bear', default=0)

        # ====================================================
        # HARD FILTERS
        # ====================================================

        # H1: No Nuke -- asiri buyuk mum korumasi
        candle_range = h - lo
        hard1_noNuke = candle_range < 3.5 * atr

        # H2: Volume Alive -- olu piyasa korumasi
        hard2_volAlive = vol > vol_ma * 0.4

        # H3: Volatilite yeterli (ATR rank veya mutlak)
        atr_rank = self._safe_get(df, 'atr_rank_50', default=0.5)
        hard3_volExpansion = atr_rank > 0.5 or vol_ratio > 1.5

        hard_pass = hard1_noNuke and hard2_volAlive and hard3_volExpansion

        if not hard_pass:
            reasons = []
            if not hard1_noNuke:
                reasons.append("Nuke")
            if not hard2_volAlive:
                reasons.append("VolDead")
            if not hard3_volExpansion:
                reasons.append("LowVol")
            return self._no_signal(
                f"HardFail: {';'.join(reasons)}",
                hard_pass=False
            )

        # Breakout seviyeleri
        high_20 = df['high'].tail(20).max()
        low_20 = df['low'].tail(20).min()

        # Setup tespiti
        is_bull_breakout = (c > high_20 * 0.998) and (vol_ratio > 1.5)
        is_bear_breakout = (c < low_20 * 1.002) and (vol_ratio > 1.5)
        is_bull_vol_spike = (vol_ratio > 2.0) and (macd_hist > 0) and (c > ema9)
        is_bear_vol_spike = (vol_ratio > 2.0) and (macd_hist < 0) and (c < ema9)

        has_bull = is_bull_breakout or is_bull_vol_spike
        has_bear = is_bear_breakout or is_bear_vol_spike

        if not has_bull and not has_bear:
            return self._no_signal(
                "Breakout kosullari saglanmadi",
                hard_pass=True,
                soft_score=0
            )

        # ====================================================
        # SOFT SCORING (yon bagimsiz puanlama)
        # ====================================================
        def _calc_soft(direction):
            score = 0
            # S1: Volume spike guclu
            if vol_ratio > 2.5:
                score += 1
            # S2: BOS onayi
            if direction == 'LONG' and bos_bull:
                score += 1
            elif direction == 'SHORT' and bos_bear:
                score += 1
            # S3: FVG onayi
            if direction == 'LONG' and fvg_bull:
                score += 1
            elif direction == 'SHORT' and fvg_bear:
                score += 1
            # S4: ADX guclu trend
            if adx > 22:
                score += 1
            # S5: Candle body orani (guclu mum)
            body = abs(c - o)
            if body > candle_range * 0.5:
                score += 1
            return score

        # ====================================================
        # TP / SL Hesaplama
        # ====================================================
        def _compute_tp_sl(direction, entry):
            lb = min(50, len(df) - 1)
            if direction == 'LONG':
                tp_raw = float(df['high'].shift(1).iloc[-lb:].max())
                sl_raw = float(df['low'].shift(1).iloc[-lb:].min())
                if pd.isna(tp_raw) or tp_raw <= entry + atr:
                    tp_raw = entry + self.default_tp_mult * atr
                if pd.isna(sl_raw) or sl_raw >= entry:
                    sl_raw = entry - self.default_sl_mult * atr
                # Minimum mesafeler
                if (tp_raw - entry) < 2 * atr:
                    tp_raw = entry + self.default_tp_mult * atr
                if (entry - sl_raw) < 0.5 * atr:
                    sl_raw = entry - self.default_sl_mult * atr
            else:
                tp_raw = float(df['low'].shift(1).iloc[-lb:].min())
                sl_raw = float(df['high'].shift(1).iloc[-lb:].max())
                if pd.isna(tp_raw) or tp_raw >= entry - atr:
                    tp_raw = entry - self.default_tp_mult * atr
                if pd.isna(sl_raw) or sl_raw <= entry:
                    sl_raw = entry + self.default_sl_mult * atr
                if (entry - tp_raw) < 2 * atr:
                    tp_raw = entry - self.default_tp_mult * atr
                if (sl_raw - entry) < 0.5 * atr:
                    sl_raw = entry + self.default_sl_mult * atr
            return tp_raw, sl_raw

        # ====================================================
        # BULLISH BREAKOUT SIGNAL
        # ====================================================
        if has_bull:
            soft_score = _calc_soft('LONG')
            tp_price, sl_price = _compute_tp_sl('LONG', c)

            # RR kontrolu
            risk = abs(c - sl_price)
            reward = abs(tp_price - c)
            if risk < 1e-10:
                return self._no_signal("Risk sifira cok yakin")
            rr = reward / risk
            if rr < self.MIN_RR:
                return self._no_signal(f"RR yetersiz ({rr:.2f} < {self.MIN_RR})")

            entry_type = "breakout" if is_bull_breakout else "vol_spike"
            return self._long_signal(
                soft_score=soft_score,
                reason=f"Breakout^: Vol={vol_ratio:.1f}x ADX={adx:.0f} "
                       f"RR={rr:.2f} | {entry_type}",
                entry_price=c,
                entry_type=entry_type,
                tp_price=tp_price,
                sl_price=sl_price,
            )

        # ====================================================
        # BEARISH BREAKOUT SIGNAL
        # ====================================================
        if has_bear:
            soft_score = _calc_soft('SHORT')
            tp_price, sl_price = _compute_tp_sl('SHORT', c)

            risk = abs(sl_price - c)
            reward = abs(c - tp_price)
            if risk < 1e-10:
                return self._no_signal("Risk sifira cok yakin")
            rr = reward / risk
            if rr < self.MIN_RR:
                return self._no_signal(f"RR yetersiz ({rr:.2f} < {self.MIN_RR})")

            entry_type = "breakout" if is_bear_breakout else "vol_spike"
            return self._short_signal(
                soft_score=soft_score,
                reason=f"Breakoutv: Vol={vol_ratio:.1f}x ADX={adx:.0f} "
                       f"RR={rr:.2f} | {entry_type}",
                entry_price=c,
                entry_type=entry_type,
                tp_price=tp_price,
                sl_price=sl_price,
            )

        return self._no_signal("Breakout kosullari saglanmadi")
