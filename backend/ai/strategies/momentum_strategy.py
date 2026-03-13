"""
Momentum Strategy v1.1

Rejim: TRENDING

v1.1 Değişiklikler:
  - [OK] SHORT sinyal eklendi (kritik bug fix)
  - [OK] Pullback TP/SL look-ahead bias düzeltildi (.shift(1))
  - [OK] Soft score yön-bağımlı hale getirildi
  - [OK] Kolon güvenliği eklendi
  - [OK] Hard filter SHORT breakout/pullback eklendi
  - [OK] NaN koruması eklendi
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

        # ── Kolon Güvenliği ──────────────────────────────
        required = ['close', 'high', 'low', 'open', 'volume',
                     'atr', 'rsi', 'adx', 'ema9', 'ema21', 'ema50',
                     'di_plus', 'di_minus']
        col_err = self._validate_columns(df, required)
        if col_err:
            return self._no_signal(col_err)

        # ── Data Extraction ─────────────────────────────
        last = df.iloc[-1]
        prev = df.iloc[-2]
        c = last['close']
        h = last['high']
        lo = last['low']
        o = last['open']
        atr = last['atr']
        rsi = last['rsi']
        adx = last['adx']
        ema9 = last['ema9']
        ema21 = last['ema21']
        ema50 = last['ema50']
        vol = last['volume']
        di_plus = last['di_plus']
        di_minus = last['di_minus']

        # ATR koruması
        if pd.isna(atr) or atr < 1e-10:
            return self._no_signal("ATR geçersiz")

        vol_ma = df['volume'].rolling(20, min_periods=5).mean().iloc[-1]
        if pd.isna(vol_ma) or vol_ma < 1e-10:
            vol_ma = vol  # Fallback

        # ════════════════════════════════════════════════
        # HARD FILTERS
        # ════════════════════════════════════════════════

        # H1: No Nuke — aşırı büyük mum koruması
        hard1_noNuke = (h - lo) < 3.5 * atr

        # H2: Volume Alive — ölü piyasa koruması
        hard2_volAlive = vol > vol_ma * 0.4

        # H3: Breakout veya Pullback tespit (çift yönlü)
        is_breakout_bull = (c > prev['high']) and (vol > vol_ma * 1.2)
        is_breakout_bear = (c < prev['low']) and (vol > vol_ma * 1.2)
        is_pullback_bull = (c > ema50) and (lo <= ema9 * 1.01)
        is_pullback_bear = (c < ema50) and (h >= ema9 * 0.99)

        has_bull_setup = is_breakout_bull or is_pullback_bull
        has_bear_setup = is_breakout_bear or is_pullback_bear

        hard4_trendDir = has_bull_setup or has_bear_setup

        hard_pass = hard1_noNuke and hard2_volAlive and hard4_trendDir

        if not hard_pass:
            reasons = []
            if not hard1_noNuke:
                reasons.append("Nuke")
            if not hard2_volAlive:
                reasons.append("VolDead")
            if not hard4_trendDir:
                reasons.append("NoBO/PB")
            return self._no_signal(
                f"HardFail: {';'.join(reasons)}",
                hard_pass=False
            )

        # ════════════════════════════════════════════════
        # YÖN BELİRLEME
        # ════════════════════════════════════════════════
        bull_bias = (c > ema50) and (di_plus > di_minus)
        bear_bias = (c < ema50) and (di_minus > di_plus)

        # ════════════════════════════════════════════════
        # SOFT FILTERS (yön-bağımlı)
        # ════════════════════════════════════════════════
        soft_score = 0
        ema_dist_pct = abs(c - ema21) / ema21 * 100 if ema21 != 0 else 0

        if bull_bias:
            # S1: EMA Stretch — aşırı uzama yok
            if ema_dist_pct < 2.0:
                soft_score += 1
            # S2: RSI ideal LONG bölgesi
            if 42 <= rsi <= 68:
                soft_score += 1
            # S3: Candle range makul
            if (h - lo) < 2.0 * atr:
                soft_score += 1
            # S4: ADX güçlü trend
            if adx > 22:
                soft_score += 1
            # S5: Artan hacim
            if vol > prev['volume']:
                soft_score += 1

        elif bear_bias:
            # S1: EMA Stretch
            if ema_dist_pct < 2.0:
                soft_score += 1
            # S2: RSI ideal SHORT bölgesi
            if 32 <= rsi <= 58:
                soft_score += 1
            # S3: Candle range makul
            if (h - lo) < 2.0 * atr:
                soft_score += 1
            # S4: ADX güçlü trend
            if adx > 22:
                soft_score += 1
            # S5: Artan hacim
            if vol > prev['volume']:
                soft_score += 1

        # ════════════════════════════════════════════════
        # LONG SİNYAL
        # ════════════════════════════════════════════════
        if bull_bias and has_bull_setup:
            entry_type = "breakout" if is_breakout_bull else "pullback"
            tp_price = 0.0
            sl_price = 0.0

            if entry_type == "pullback":
                # Look-ahead bias fix: .shift(1)
                tp_raw = df['high'].shift(1).rolling(
                    10, min_periods=3
                ).max().iloc[-1]
                sl_raw = df['low'].shift(1).rolling(
                    10, min_periods=3
                ).min().iloc[-1]

                tp_price = tp_raw if (
                    not pd.isna(tp_raw) and tp_raw > c
                ) else c + self.default_tp_mult * atr

                sl_price = sl_raw if (
                    not pd.isna(sl_raw) and sl_raw < c
                ) else c - self.default_sl_mult * atr

            else:  # breakout
                # Swing-based TP, ATR-based SL
                tp_raw = df['high'].shift(1).rolling(
                    20, min_periods=5
                ).max().iloc[-1]
                tp_price = tp_raw if (
                    not pd.isna(tp_raw) and tp_raw > c + atr
                ) else c + self.default_tp_mult * atr

                sl_price = c - self.default_sl_mult * atr
                # Use prev candle low as SL if tighter
                if prev['low'] < c and (c - prev['low']) > 0.5 * atr:
                    sl_price = prev['low'] - 0.1 * atr

            # Minimum TP/SL mesafe korumasi
            if (tp_price - c) < 2 * atr:
                tp_price = c + self.default_tp_mult * atr
            if (c - sl_price) < 0.5 * atr:
                sl_price = c - self.default_sl_mult * atr

            return self._long_signal(
                soft_score=soft_score,
                reason=f"LONG {entry_type} | Score={soft_score}/5 | "
                       f"RSI={rsi:.0f} ADX={adx:.0f}",
                entry_price=c,
                entry_type=entry_type,
                tp_price=tp_price,
                sl_price=sl_price
            )

        # ════════════════════════════════════════════════
        # SHORT SİNYAL (v1.1 — YENİ)
        # ════════════════════════════════════════════════
        if bear_bias and has_bear_setup:
            entry_type = "breakout" if is_breakout_bear else "pullback"
            tp_price = 0.0
            sl_price = 0.0

            if entry_type == "pullback":
                tp_raw = df['low'].shift(1).rolling(
                    10, min_periods=3
                ).min().iloc[-1]
                sl_raw = df['high'].shift(1).rolling(
                    10, min_periods=3
                ).max().iloc[-1]

                tp_price = tp_raw if (
                    not pd.isna(tp_raw) and tp_raw < c
                ) else c - self.default_tp_mult * atr

                sl_price = sl_raw if (
                    not pd.isna(sl_raw) and sl_raw > c
                ) else c + self.default_sl_mult * atr

            else:  # breakout
                tp_raw = df['low'].shift(1).rolling(
                    20, min_periods=5
                ).min().iloc[-1]
                tp_price = tp_raw if (
                    not pd.isna(tp_raw) and tp_raw < c - atr
                ) else c - self.default_tp_mult * atr

                sl_price = c + self.default_sl_mult * atr
                if prev['high'] > c and (prev['high'] - c) > 0.5 * atr:
                    sl_price = prev['high'] + 0.1 * atr

            # Minimum TP/SL mesafe korumasi
            if (c - tp_price) < 2 * atr:
                tp_price = c - self.default_tp_mult * atr
            if (sl_price - c) < 0.5 * atr:
                sl_price = c + self.default_sl_mult * atr

            return self._short_signal(
                soft_score=soft_score,
                reason=f"SHORT {entry_type} | Score={soft_score}/5 | "
                       f"RSI={rsi:.0f} ADX={adx:.0f}",
                entry_price=c,
                entry_type=entry_type,
                tp_price=tp_price,
                sl_price=sl_price
            )

        return self._no_signal(
            f"Trend direction mismatch | RSI={rsi:.0f} ADX={adx:.0f}",
            hard_pass=True,
            soft_score=soft_score
        )
