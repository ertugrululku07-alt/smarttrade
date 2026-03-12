"""
ICT/SMC Strategy v1.2

Rejimler: TRENDING, HIGH_VOLATILE
Mantık:
  1. MSS (Market Structure Shift) ile yönü belirle.
  2. FVG (Fair Value Gap) içine çekilmeyi (pullback) bekle.
  3. OB (Order Block) üzerinde veya altında stop-loss tut.

v1.2 Düzeltmeleri:
  - ✅ Python 3.9 uyumu (Optional[float])
  - ✅ Kolon güvenliği _safe_get ile
  - ✅ Entry type standardize
  - ✅ Reason mesajları iyileştirildi
  - ✅ ATR NaN/sıfır koruması güçlendirildi
"""

import pandas as pd
import numpy as np
from typing import Optional
from .base_strategy import BaseStrategy, Signal


class ICTStrategy(BaseStrategy):

    name = "ict_smc"
    regime = "trending"
    default_tp_mult = 3.0
    default_sl_mult = 1.0
    MIN_RR = 1.5
    OB_MAX_AGE = 20

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 50:
            return self._no_signal("Yetersiz veri")

        # ── Kolon Güvenliği ──────────────────────────────
        required = ['close', 'high', 'low']
        col_err = self._validate_columns(df, required)
        if col_err:
            return self._no_signal(col_err)

        last_row = df.iloc[-1]
        c = last_row['close']

        # ── ATR Validasyonu ──────────────────────────────
        atr = self._safe_get(df, 'atr', default=None)
        if atr is None or atr < 1e-10:
            return self._no_signal("ATR geçersiz")

        # ── 1. Market Structure Shift (MSS) ──────────────
        mss_up = self._safe_get(df, 'mss_up', default=0) == 1
        mss_down = self._safe_get(df, 'mss_down', default=0) == 1

        # ── 2. Fair Value Gaps (FVG) — Son 3 bar ────────
        fvg_bull = False
        if 'fvg_bull' in df.columns:
            fvg_bull = df['fvg_bull'].iloc[-3:].any()

        fvg_bear = False
        if 'fvg_bear' in df.columns:
            fvg_bear = df['fvg_bear'].iloc[-3:].any()

        # ── 3. Order Blocks — Staleness kontrolü ────────
        ob_bull_price = self._get_recent_ob(df, 'ob_bull')
        ob_bear_price = self._get_recent_ob(df, 'ob_bear')

        # ── 4. EMA50 ────────────────────────────────────
        ema50 = self._safe_get(df, 'ema50', default=c)

        # ── 5. Yön ve Skor Hesaplama ─────────────────────
        score = 0
        direction = None
        reason_parts = []

        if mss_up or (c > ema50 and fvg_bull):
            direction = 'LONG'
            if mss_up:
                score += 2
                reason_parts.append("MSS^")
            if fvg_bull:
                score += 2
                reason_parts.append("FVG^")
            if ob_bull_price is not None and c > ob_bull_price:
                score += 1
                reason_parts.append(f"OB({ob_bull_price:.2f})")

        elif mss_down or (c < ema50 and fvg_bear):
            direction = 'SHORT'
            if mss_down:
                score += 2
                reason_parts.append("MSSv")
            if fvg_bear:
                score += 2
                reason_parts.append("FVGv")
            if ob_bear_price is not None and c < ob_bear_price:
                score += 1
                reason_parts.append(f"OB({ob_bear_price:.2f})")

        if direction is None or score < 3:
            return self._no_signal(
                f"ICT Filter: score={score} "
                f"{'|'.join(reason_parts) if reason_parts else 'NoSetup'}"
            )

        # ── 6. SL Hesaplama ──────────────────────────────
        sl_price = self._calculate_sl(
            direction, c, atr, ob_bull_price, ob_bear_price
        )

        # ── 7. TP Hesaplama ──────────────────────────────
        tp_price = self._calculate_tp(direction, c, atr, df)

        # ── 8. SL Yön Doğrulaması ────────────────────────
        if direction == 'LONG' and sl_price >= c:
            sl_price = c - 1.2 * atr
        elif direction == 'SHORT' and sl_price <= c:
            sl_price = c + 1.2 * atr

        # ── 9. TP Yön Doğrulaması ────────────────────────
        if direction == 'LONG' and tp_price <= c:
            tp_price = c + 3.0 * atr
        elif direction == 'SHORT' and tp_price >= c:
            tp_price = c - 3.0 * atr

        # ── 10. Minimum R:R Kontrolü ─────────────────────
        risk = abs(c - sl_price)
        reward = abs(tp_price - c)

        if risk < 1e-10:
            return self._no_signal("Risk sıfıra çok yakın")

        rr_ratio = reward / risk
        if rr_ratio < self.MIN_RR:
            return self._no_signal(
                f"R:R yetersiz ({rr_ratio:.1f} < {self.MIN_RR})"
            )

        # ── 11. Sinyal ───────────────────────────────────
        signal_reason = (
            f"ICT {'|'.join(reason_parts)} | R:R={rr_ratio:.1f}"
        )

        if direction == 'LONG':
            return self._long_signal(
                soft_score=score,
                reason=signal_reason,
                entry_price=c,
                entry_type="ict_setup",
                tp_price=tp_price,
                sl_price=sl_price,
            )
        else:
            return self._short_signal(
                soft_score=score,
                reason=signal_reason,
                entry_price=c,
                entry_type="ict_setup",
                tp_price=tp_price,
                sl_price=sl_price,
            )

    # ──────────────────────────────────────────────────
    # Helper Methods
    # ──────────────────────────────────────────────────

    def _get_recent_ob(self, df: pd.DataFrame,
                       col: str) -> Optional[float]:
        """
        Son OB_MAX_AGE bar içindeki en son Order Block fiyatı.
        """
        if col not in df.columns:
            return None

        recent = df[col].iloc[-self.OB_MAX_AGE:]
        valid = recent[recent > 0]

        if valid.empty:
            return None

        return float(valid.iloc[-1])

    def _calculate_sl(
        self,
        direction: str,
        entry: float,
        atr: float,
        ob_bull_price: Optional[float],
        ob_bear_price: Optional[float],
    ) -> float:
        """
        ICT SL: OB varsa ve doğru taraftaysa OB bazlı,
        yoksa ATR bazlı.
        """
        if direction == 'LONG':
            if (ob_bull_price is not None
                    and ob_bull_price < entry):
                return ob_bull_price - atr * 0.3
            else:
                return entry - 1.5 * atr
        else:
            if (ob_bear_price is not None
                    and ob_bear_price > entry):
                return ob_bear_price + atr * 0.3
            else:
                return entry + 1.5 * atr

    def _calculate_tp(
        self,
        direction: str,
        entry: float,
        atr: float,
        df: pd.DataFrame,
    ) -> float:
        """
        ICT TP: Geçmiş likidite havuzu (swing high/low).
        Mevcut bar hariç (.shift(1)).
        """
        if direction == 'LONG':
            past_high = df['high'].shift(1).rolling(
                30, min_periods=5
            ).max()
            tp = past_high.iloc[-1]

            if pd.isna(tp) or tp <= entry:
                tp = entry + 3.0 * atr

        else:
            past_low = df['low'].shift(1).rolling(
                30, min_periods=5
            ).min()
            tp = past_low.iloc[-1]

            if pd.isna(tp) or tp >= entry:
                tp = entry - 3.0 * atr

        return float(tp)
