"""
Scalping / Low Volatility Strategy v1.1

Rejim: LOW_VOLATILE

v1.1 Değişiklikler:
  - [OK] Graduated scoring (kalite bazlı skor)
  - [OK] Kolon güvenliği eklendi
  - [OK] entry_type set edildi
  - [OK] Volume filtresi eklendi (çok düşük hacimde işlem yapma)
  - [OK] NaN koruması eklendi
  - [OK] Ek sinyal katmanları (daha fazla fırsat)
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, Signal


class ScalpingStrategy(BaseStrategy):

    name = "scalping"
    regime = "low_volatile"
    default_tp_mult = 1.8
    default_sl_mult = 0.7

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 50:
            return self._no_signal("Yetersiz veri")

        # ── Kolon Güvenliği ──────────────────────────────
        required = ['close', 'rsi', 'bb_pos', 'stoch_k', 'stoch_d']
        col_err = self._validate_columns(df, required)
        if col_err:
            return self._no_signal(col_err)

        # ── Data Extraction ─────────────────────────────
        c = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        bb_pos = df['bb_pos'].iloc[-1]
        stoch_k = df['stoch_k'].iloc[-1]
        stoch_d = df['stoch_d'].iloc[-1]

        # NaN koruması
        if any(pd.isna(v) for v in [rsi, bb_pos, stoch_k, stoch_d]):
            return self._no_signal("Osilatör NaN")

        # Opsiyonel
        vwap_dist = self._safe_get(df, 'vwap_dist', default=0.0)
        vol_ratio = self._safe_get(df, 'vol_ratio_20', default=1.0)
        rsi_prev = df['rsi'].iloc[-2] if len(df) > 2 else rsi

        # ── Volume Ölü mü? ──────────────────────────────
        # Çok düşük hacimde scalp bile tehlikeli
        if vol_ratio < 0.3:
            return self._no_signal(
                f"Hacim çok düşük ({vol_ratio:.2f}x) — likidite riski"
            )

        # ═══════════════════════════════════════════════════
        # LONG Sinyalleri (Graduated Scoring)
        # ═══════════════════════════════════════════════════

        # L1: BB extreme + RSI extreme + Stoch cross (EN GÜÇLÜ)
        if (bb_pos < 0.10 and rsi < 35 and stoch_k > stoch_d
                and stoch_k < 25):
            score = 4
            # Bonus: RSI dönüyor + hacim desteği
            if rsi > rsi_prev and vol_ratio > 0.8:
                score = 5

            return self._long_signal(
                soft_score=score,
                reason=f"ScalpL1^: BB={bb_pos:.2f} RSI={rsi:.0f} "
                       f"StochX^ score={score}",
                entry_price=c,
                entry_type="pullback",
            )

        # L2: VWAP altında + momentum dönüyor (ORTA)
        if (vwap_dist < -0.003 and rsi < 40
                and stoch_k > stoch_d and rsi > rsi_prev):
            score = 4 if bb_pos < 0.20 else 3

            return self._long_signal(
                soft_score=score,
                reason=f"ScalpL2^: VWAP={vwap_dist:.4f} RSI={rsi:.0f}^ "
                       f"score={score}",
                entry_price=c,
                entry_type="pullback",
            )

        # L3: BB alt bölge + stoch cross (ZAYIF ama geçerli)
        if (bb_pos < 0.20 and stoch_k > stoch_d and stoch_k < 30
                and rsi < 45 and rsi > rsi_prev):
            return self._long_signal(
                soft_score=3,
                reason=f"ScalpL3^: BB={bb_pos:.2f} Stoch={stoch_k:.0f}X^",
                entry_price=c,
                entry_type="pullback",
            )

        # ═══════════════════════════════════════════════════
        # SHORT Sinyalleri (Graduated Scoring)
        # ═══════════════════════════════════════════════════

        # S1: BB extreme + RSI extreme + Stoch cross (EN GÜÇLÜ)
        if (bb_pos > 0.90 and rsi > 65 and stoch_k < stoch_d
                and stoch_k > 75):
            score = 4
            if rsi < rsi_prev and vol_ratio > 0.8:
                score = 5

            return self._short_signal(
                soft_score=score,
                reason=f"ScalpS1v: BB={bb_pos:.2f} RSI={rsi:.0f} "
                       f"StochXv score={score}",
                entry_price=c,
                entry_type="pullback",
            )

        # S2: VWAP üstünde + momentum dönüyor (ORTA)
        if (vwap_dist > 0.003 and rsi > 60
                and stoch_k < stoch_d and rsi < rsi_prev):
            score = 4 if bb_pos > 0.80 else 3

            return self._short_signal(
                soft_score=score,
                reason=f"ScalpS2v: VWAP={vwap_dist:.4f} RSI={rsi:.0f}v "
                       f"score={score}",
                entry_price=c,
                entry_type="pullback",
            )

        # S3: BB üst bölge + stoch cross (ZAYIF ama geçerli)
        if (bb_pos > 0.80 and stoch_k < stoch_d and stoch_k > 70
                and rsi > 55 and rsi < rsi_prev):
            return self._short_signal(
                soft_score=3,
                reason=f"ScalpS3v: BB={bb_pos:.2f} Stoch={stoch_k:.0f}Xv",
                entry_price=c,
                entry_type="pullback",
            )

        return self._no_signal(
            f"Düşük vol, net sinyal yok | "
            f"RSI={rsi:.0f} BB={bb_pos:.2f} → Bekle"
        )
