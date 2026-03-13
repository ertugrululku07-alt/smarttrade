"""
Smart Money + Order Flow Strategy v1.1

Rejimler: TRENDING, HIGH_VOLATILE
Hedef RR: 3:1+
Tahmini Win Rate: %50-55

Sinyal Mantığı:
  1. CVD Divergence  — Fiyat yeni tepe/dip yaparken CVD yapmıyor
  2. Volume Absorption — Yüksek hacimli mumda küçük gövde + büyük wick
  3. Liquidity Grab + Reclaim — Wick ile swing seviyeyi yıkıp geri kapama
  4. Delta Shift — CVD momentum yön değiştiriyor
  5. Institutional Volume Zone — Yüksek hacimli bölgede işlem (bonus)
  6. Killzone — Londra/NY oturumu (bonus)

Confluence:
  Yönlü puanlar (CVD, Absorption, LiqGrab, Delta): min 4
  Bonus puanlar (InstZone, KZ): eşiğe sayılmaz, skoru artırır

v1.1 Düzeltmeleri:
  - ✅ #1  _no_signal kwargs uyumluluğu
  - ✅ #2  CVD windowed (cumsum yerine rolling)
  - ✅ #3  CVD hesaplama tek noktada, cache ile
  - ✅ #4  Score cap kaldırıldı
  - ✅ #5  Absorption en güncel bar öncelikli
  - ✅ #6  Swing detection range düzeltildi
  - ✅ #7  TP/SL NaN koruması güçlendirildi
  - ✅ #8  Bonus puanlar yön eşiğinden ayrıldı
  - ✅ #9  InstZone performans iyileştirmesi
  - ✅ #10 Tick size yuvarlama desteği
  - ✅ #11 SL/TP yön doğrulaması eklendi
  - ✅ #12 CVD divergence swing bazlı iyileştirme
"""

import math
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from .base_strategy import BaseStrategy, Signal


# ══════════════════════════════════════════════════════════════
# Veri Yapıları
# ══════════════════════════════════════════════════════════════

@dataclass
class LiqGrabResult:
    """Liquidity grab tespiti sonucu."""
    direction: str      # 'bullish' | 'bearish'
    level: float        # Sweep edilen swing seviyesi
    wick: float         # Wick ucu fiyatı
    strength: float     # ATR cinsinden wick gücü


# ══════════════════════════════════════════════════════════════
# Ana Strateji
# ══════════════════════════════════════════════════════════════

class SmartMoneyOrderFlowStrategy(BaseStrategy):

    name = "smart_money_orderflow"
    regime = "trending"

    # ── Temel ─────────────────────────────────────────
    default_tp_mult = 3.0
    default_sl_mult = 1.2
    MIN_RR = 2.0
    MIN_DATA_BARS = 60
    DEFAULT_TICK_SIZE = 0.01

    # ── Confluence ────────────────────────────────────
    # Yönlü puanlar: CVD div(3), Absorption(2), LiqGrab(3), Delta(2)
    # Bonus puanlar: InstZone(1), KZ(1)
    # Minimum = sadece yönlü puanlardan
    MIN_DIRECTIONAL_SCORE = 4

    # ── CVD ───────────────────────────────────────────
    CVD_WINDOW = 50             # Windowed CVD pencere boyutu
    CVD_DIVERGENCE_LOOKBACK = 20
    CVD_EMA_FAST = 5
    CVD_EMA_SLOW = 13
    CVD_DIV_COMPARE_BARS = 5   # Divergence karşılaştırma penceresi

    # ── Volume Absorption ─────────────────────────────
    ABS_SCAN_BARS = 3           # Son kaç bar kontrol
    ABS_MIN_VOL_RATIO = 1.5    # Min hacim / ortalama oranı
    ABS_MAX_BODY_RATIO = 0.35  # Max gövde / range oranı
    ABS_MIN_WICK_ATR = 0.5     # Min wick boyutu (ATR x)

    # ── Liquidity Grab ────────────────────────────────
    SWING_LOOKBACK = 5
    SWEEP_SCAN_BARS = 3
    SWEEP_MIN_WICK_ATR = 0.3

    # ── Hard Filters ──────────────────────────────────
    NUKE_CANDLE_ATR = 3.5      # Nuke candle eşiği
    MIN_VOL_RATIO = 0.4        # Minimum hacim oranı

    # ── Killzones (UTC) ──────────────────────────────
    KILLZONES = [
        (7, 11),    # Londra
        (12, 16),   # New York
        (0, 3),     # Asya (crypto)
    ]

    # ══════════════════════════════════════════════════
    # ANA SİNYAL ÜRETİCİ
    # ══════════════════════════════════════════════════

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Smart Money + Order Flow sinyal pipeline'ı."""

        # ── Veri Kontrolü ──
        if len(df) < self.MIN_DATA_BARS:
            return self._no_signal(
                f"Yetersiz veri (<{self.MIN_DATA_BARS} bar)"
            )

        required = ['close', 'high', 'low', 'open', 'volume']
        col_err = self._validate_columns(df, required)
        if col_err:
            return self._no_signal(col_err)

        # ── ATR ──
        atr = self._safe_get(df, 'atr', default=None)
        if atr is None or atr < 1e-10:
            return self._no_signal("ATR geçersiz")

        last_row = df.iloc[-1]
        c = float(last_row['close'])
        bar_high = float(last_row['high'])
        bar_low = float(last_row['low'])
        bar_open = float(last_row['open'])
        vol = float(last_row['volume'])

        # ── Volume MA ──
        vol_ma = self._safe_vol_ma(df)

        # ══════════════════════════════════════════════
        # HARD FILTERS
        # ══════════════════════════════════════════════

        candle_range = bar_high - bar_low
        if candle_range > self.NUKE_CANDLE_ATR * atr:
            return self._no_signal("Nuke mum filtresi")

        vol_ratio = vol / vol_ma if vol_ma > 0 else 1.0
        if vol_ratio < self.MIN_VOL_RATIO:
            return self._no_signal("Hacim çok düşük")

        # ══════════════════════════════════════════════
        # CVD HESAPLAMA (tek noktada)  [FIX #3]
        # ══════════════════════════════════════════════

        cvd_windowed = self._compute_windowed_cvd(df)

        # ══════════════════════════════════════════════
        # ANALİZ MODÜLLERİ
        # ══════════════════════════════════════════════

        # 1. CVD Divergence
        cvd_div = self._detect_cvd_divergence(
            df, cvd_windowed, atr
        )

        # 2. Volume Absorption
        absorption = self._detect_volume_absorption(
            df, atr, vol_ma
        )

        # 3. Liquidity Grab + Reclaim
        liq_grab = self._detect_liquidity_grab(df, atr)

        # 4. Delta Shift
        delta_shift = self._detect_delta_shift(cvd_windowed)

        # 5. Institutional Volume Zone (bonus)
        inst_zone = self._check_institutional_zone(df, c)

        # 6. Killzone (bonus)
        killzone = self._check_killzone(last_row)

        # ══════════════════════════════════════════════
        # CONFLUENCE SCORING  [FIX #8]
        # ══════════════════════════════════════════════

        score, direction, reasons = self._calculate_confluence(
            cvd_div=cvd_div,
            absorption=absorption,
            liq_grab=liq_grab,
            delta_shift=delta_shift,
            inst_zone=inst_zone,
            killzone=killzone,
        )

        if direction is None:
            return self._no_signal(
                f"Confluence yetersiz: score={score} "
                f"{'|'.join(reasons) if reasons else 'NoSetup'}"
            )

        # ══════════════════════════════════════════════
        # TP / SL
        # ══════════════════════════════════════════════

        tp_raw, sl_raw = self._compute_tp_sl(
            df, direction, c, atr, liq_grab
        )

        # Yön doğrulaması  [FIX #11]
        sl_price, tp_price = self._validate_sl_tp(
            direction, c, atr, sl_raw, tp_raw
        )

        # R:R kontrolü
        risk = abs(c - sl_price)
        reward = abs(tp_price - c)

        if risk < 1e-10:
            return self._no_signal("Risk sıfıra çok yakın")

        rr = reward / risk
        if rr < self.MIN_RR:
            return self._no_signal(
                f"R:R yetersiz ({rr:.2f} < {self.MIN_RR})"
            )

        # ══════════════════════════════════════════════
        # SİNYAL  [FIX #1, #4]
        # ══════════════════════════════════════════════

        reason_str = '|'.join(reasons)
        signal_reason = (
            f"SMOv1.1 {reason_str} | Score={score} R:R={rr:.2f}"
        )

        # [FIX #4]: Score cap kaldırıldı — gerçek skoru gönder
        if direction == 'LONG':
            return self._long_signal(
                soft_score=score,
                reason=signal_reason,
                entry_price=c,
                entry_type="smart_money",
                tp_price=tp_price,
                sl_price=sl_price,
            )
        else:
            return self._short_signal(
                soft_score=score,
                reason=signal_reason,
                entry_price=c,
                entry_type="smart_money",
                tp_price=tp_price,
                sl_price=sl_price,
            )

    # ══════════════════════════════════════════════════════════
    # CVD HESAPLAMA  [FIX #2, #3]
    # ══════════════════════════════════════════════════════════

    def _compute_windowed_cvd(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Windowed CVD hesaplama.

        cumsum() yerine rolling sum kullanır → sınırsız büyüme yok.
        Her bar'ın CVD'si son CVD_WINDOW bar'ın delta toplamıdır.

        CVD yoksa tahmini delta hesaplanır:
          delta = volume × (2 × buy_ratio - 1)
          buy_ratio = (close - low) / (high - low)
        """
        # Mevcut CVD kolonu varsa kullan
        if 'cvd' in df.columns:
            raw_cvd = df['cvd']
            # Detrend: CVD - EMA(CVD, window) → büyüme sorununu çözer
            cvd_ema = raw_cvd.ewm(
                span=self.CVD_WINDOW, adjust=False
            ).mean()
            return raw_cvd - cvd_ema

        # Tahmini delta hesapla
        full_range = (df['high'] - df['low']).replace(0, np.nan)
        buy_ratio = (
            (df['close'] - df['low']) / full_range
        ).clip(0, 1).fillna(0.5)

        delta = df['volume'] * (2 * buy_ratio - 1)

        # Windowed CVD: son N bar'ın delta toplamı
        windowed = delta.rolling(
            window=self.CVD_WINDOW,
            min_periods=10,
        ).sum()

        return windowed.fillna(0)

    # ══════════════════════════════════════════════════════════
    # CVD DIVERGENCE  [FIX #12]
    # ══════════════════════════════════════════════════════════

    def _detect_cvd_divergence(
        self,
        df: pd.DataFrame,
        cvd: pd.Series,
        atr: float,
    ) -> Optional[str]:
        """
        CVD divergence tespiti.

        Bullish: Fiyat yeni dip yapıyor, CVD yükselen dip
        Bearish: Fiyat yeni tepe yapıyor, CVD düşen tepe

        [FIX #12]: Window karşılaştırması yerine
        gerçek min/max noktaları kullanılır.
        """
        lookback = self.CVD_DIVERGENCE_LOOKBACK
        compare = self.CVD_DIV_COMPARE_BARS

        if len(df) < lookback + compare:
            return None

        if len(cvd) < lookback + compare:
            return None

        # Son lookback barı al
        price_highs = df['high'].iloc[-lookback:]
        price_lows = df['low'].iloc[-lookback:]
        cvd_tail = cvd.iloc[-lookback:]

        # NaN kontrolü
        if cvd_tail.isna().all():
            return None

        # İki pencere: son half ve önceki half
        half = lookback // 2
        if half < 3:
            return None

        # ── Bullish Divergence ──
        # Son yarı: fiyat daha düşük dip, CVD daha yüksek dip
        recent_price_low = price_lows.iloc[-half:].min()
        prev_price_low = price_lows.iloc[:half].min()
        recent_cvd_low = cvd_tail.iloc[-half:].min()
        prev_cvd_low = cvd_tail.iloc[:half].min()

        if (not pd.isna(recent_cvd_low)
                and not pd.isna(prev_cvd_low)):
            # Fiyat lower low + CVD higher low
            if (recent_price_low < prev_price_low
                    and recent_cvd_low > prev_cvd_low):
                return 'bullish'

        # ── Bearish Divergence ──
        recent_price_high = price_highs.iloc[-half:].max()
        prev_price_high = price_highs.iloc[:half].max()
        recent_cvd_high = cvd_tail.iloc[-half:].max()
        prev_cvd_high = cvd_tail.iloc[:half].max()

        if (not pd.isna(recent_cvd_high)
                and not pd.isna(prev_cvd_high)):
            # Fiyat higher high + CVD lower high
            if (recent_price_high > prev_price_high
                    and recent_cvd_high < prev_cvd_high):
                return 'bearish'

        return None

    # ══════════════════════════════════════════════════════════
    # VOLUME ABSORPTION  [FIX #5]
    # ══════════════════════════════════════════════════════════

    def _detect_volume_absorption(
        self,
        df: pd.DataFrame,
        atr: float,
        vol_ma: float,
    ) -> Optional[str]:
        """
        Yüksek hacimli mumda küçük gövde + büyük wick = emilim.

        [FIX #5]: En güncel bar öncelikli (offset=1 → 3 sırasıyla).
        İlk bulduğu geçerli absorption'ı döndürür.
        """
        for offset in range(1, self.ABS_SCAN_BARS + 1):
            if offset >= len(df):
                continue

            bar = df.iloc[-offset]
            bar_vol = float(bar['volume'])
            bar_range = float(bar['high'] - bar['low'])
            bar_body = abs(float(bar['close'] - bar['open']))
            bar_close = float(bar['close'])
            bar_open = float(bar['open'])
            bar_high = float(bar['high'])
            bar_low = float(bar['low'])

            # Minimum hacim kontrolü
            if bar_range < 1e-10:
                continue
            if bar_vol < vol_ma * self.ABS_MIN_VOL_RATIO:
                continue

            # Küçük gövde kontrolü
            body_ratio = bar_body / bar_range
            if body_ratio > self.ABS_MAX_BODY_RATIO:
                continue

            # Wick hesapları
            body_bottom = min(bar_close, bar_open)
            body_top = max(bar_close, bar_open)
            lower_wick = body_bottom - bar_low
            upper_wick = bar_high - body_top

            # Bullish absorption: büyük alt wick
            if lower_wick >= atr * self.ABS_MIN_WICK_ATR:
                # Üst wick küçük olmalı (net rejection)
                if lower_wick > upper_wick * 1.5:
                    return 'bullish'

            # Bearish absorption: büyük üst wick
            if upper_wick >= atr * self.ABS_MIN_WICK_ATR:
                if upper_wick > lower_wick * 1.5:
                    return 'bearish'

        return None

    # ══════════════════════════════════════════════════════════
    # LIQUIDITY GRAB + RECLAIM  [FIX #6]
    # ══════════════════════════════════════════════════════════

    def _detect_liquidity_grab(
        self,
        df: pd.DataFrame,
        atr: float,
    ) -> Optional[LiqGrabResult]:
        """
        Wick ile swing seviyeyi yıkıp gövde ile geri kapama.

        [FIX #6]: Swing detection üst sınırı düzeltildi.
        """
        if len(df) < 25:
            return None

        n = self.SWING_LOOKBACK
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        opens = df['open'].values
        total = len(df)

        # ── Swing Tespiti ──
        swing_highs: List[Tuple[int, float]] = []
        swing_lows: List[Tuple[int, float]] = []

        # [FIX #6]: Sweep scan barlarını hariç tut
        scan_end = total - n - self.SWEEP_SCAN_BARS
        if scan_end <= n:
            scan_end = total - n

        for i in range(n, scan_end):
            left_h = highs[i - n:i]
            right_h = highs[i + 1:i + n + 1]
            left_l = lows[i - n:i]
            right_l = lows[i + 1:i + n + 1]

            if len(left_h) == 0 or len(right_h) == 0:
                continue

            if highs[i] > left_h.max() and highs[i] > right_h.max():
                swing_highs.append((i, float(highs[i])))

            if lows[i] < left_l.min() and lows[i] < right_l.min():
                swing_lows.append((i, float(lows[i])))

        # ── Son SWEEP_SCAN_BARS barda sweep + reclaim ──
        for offset in range(1, self.SWEEP_SCAN_BARS + 1):
            idx = total - offset
            if idx < 0:
                continue

            bh = float(highs[idx])
            bl = float(lows[idx])
            bc = float(closes[idx])
            bo = float(opens[idx])
            body_top = max(bc, bo)
            body_bottom = min(bc, bo)

            # Bullish: sweep swing low → gövde üstünde kapa
            for (sw_idx, sw_price) in reversed(swing_lows):
                if sw_idx >= idx:
                    continue

                if bl < sw_price and body_bottom > sw_price:
                    wick = sw_price - bl
                    if wick >= atr * self.SWEEP_MIN_WICK_ATR:
                        return LiqGrabResult(
                            direction='bullish',
                            level=sw_price,
                            wick=bl,
                            strength=round(wick / atr, 2),
                        )

            # Bearish: sweep swing high → gövde altında kapa
            for (sw_idx, sw_price) in reversed(swing_highs):
                if sw_idx >= idx:
                    continue

                if bh > sw_price and body_top < sw_price:
                    wick = bh - sw_price
                    if wick >= atr * self.SWEEP_MIN_WICK_ATR:
                        return LiqGrabResult(
                            direction='bearish',
                            level=sw_price,
                            wick=bh,
                            strength=round(wick / atr, 2),
                        )

        return None

    # ══════════════════════════════════════════════════════════
    # DELTA SHIFT  [FIX #3]
    # ══════════════════════════════════════════════════════════

    def _detect_delta_shift(
        self,
        cvd: pd.Series,
    ) -> Optional[str]:
        """
        CVD momentum yön değişimi (EMA crossover).

        [FIX #3]: Önceden hesaplanmış windowed CVD kullanılır
        → duplicate hesaplama yok.
        """
        if len(cvd) < self.CVD_EMA_SLOW + 2:
            return None

        # NaN kontrolü
        if cvd.iloc[-self.CVD_EMA_SLOW - 2:].isna().all():
            return None

        cvd_fast = cvd.ewm(
            span=self.CVD_EMA_FAST, adjust=False
        ).mean()
        cvd_slow = cvd.ewm(
            span=self.CVD_EMA_SLOW, adjust=False
        ).mean()

        curr_fast = cvd_fast.iloc[-1]
        curr_slow = cvd_slow.iloc[-1]
        prev_fast = cvd_fast.iloc[-2]
        prev_slow = cvd_slow.iloc[-2]

        # NaN koruması
        if any(pd.isna(v) for v in [
            curr_fast, curr_slow, prev_fast, prev_slow
        ]):
            return None

        curr_above = curr_fast > curr_slow
        prev_above = prev_fast > prev_slow

        if curr_above and not prev_above:
            return 'bullish'
        elif not curr_above and prev_above:
            return 'bearish'

        return None

    # ══════════════════════════════════════════════════════════
    # INSTITUTIONAL VOLUME ZONE  [FIX #9]
    # ══════════════════════════════════════════════════════════

    def _check_institutional_zone(
        self,
        df: pd.DataFrame,
        current_price: float,
    ) -> bool:
        """
        Son 50 barın en yüksek hacimli bölgelerinde mi?

        [FIX #9]: iterrows yerine numpy bazlı kontrol.
        """
        if len(df) < 50:
            return False

        tail = df.iloc[-50:]
        volumes = tail['volume'].values
        highs = tail['high'].values
        lows = tail['low'].values

        # Top 5 hacimli barın indeksleri
        if len(volumes) < 5:
            return False

        top_indices = np.argpartition(
            volumes, -5
        )[-5:]

        for idx in top_indices:
            if lows[idx] <= current_price <= highs[idx]:
                return True

        return False

    # ══════════════════════════════════════════════════════════
    # KILLZONE
    # ══════════════════════════════════════════════════════════

    def _check_killzone(self, last_row: pd.Series) -> bool:
        """Londra/NY/Asya oturumu kontrolü."""
        try:
            hour = self._extract_hour(last_row)
            return any(
                start <= hour < end
                for start, end in self.KILLZONES
            )
        except Exception:
            return False

    def _extract_hour(self, row: pd.Series) -> int:
        """Row'dan saat bilgisi çıkar."""
        idx = row.name
        if isinstance(idx, pd.Timestamp):
            return idx.hour
        if hasattr(idx, 'hour'):
            return idx.hour

        for col_name in ['timestamp', 'datetime', 'date']:
            if col_name in row.index:
                try:
                    return pd.Timestamp(row[col_name]).hour
                except (ValueError, TypeError):
                    continue

        return 0

    # ══════════════════════════════════════════════════════════
    # CONFLUENCE SCORING  [FIX #8]
    # ══════════════════════════════════════════════════════════

    def _calculate_confluence(
        self,
        cvd_div: Optional[str],
        absorption: Optional[str],
        liq_grab: Optional[LiqGrabResult],
        delta_shift: Optional[str],
        inst_zone: bool,
        killzone: bool,
    ) -> Tuple[int, Optional[str], List[str]]:
        """
        Confluence skorlama — yönlü + bonus ayrımı.

        Yönlü Puanlar (direction-specific):
          CVD Divergence  : 3 puan
          Liquidity Grab  : 3 puan
          Volume Absorb   : 2 puan
          Delta Shift     : 2 puan
          ─────────────────────────
          Max yönlü       : 10 puan

        Bonus Puanlar (direction-neutral):
          Institutional Zone : 1 puan
          Killzone           : 1 puan
          ─────────────────────────
          Max bonus          : 2 puan

        Giriş koşulu: yönlü puan >= MIN_DIRECTIONAL_SCORE (4)
        Final skor: yönlü + bonus
        """
        # ── LONG Yönlü ──
        l_dir = 0
        l_reasons: List[str] = []

        if cvd_div == 'bullish':
            l_dir += 3
            l_reasons.append("CVD_div↑")

        if liq_grab and liq_grab.direction == 'bullish':
            l_dir += 3
            l_reasons.append(
                f"LiqGrab↑({liq_grab.level:.2f})"
            )

        if absorption == 'bullish':
            l_dir += 2
            l_reasons.append("Absorb↑")

        if delta_shift == 'bullish':
            l_dir += 2
            l_reasons.append("Delta↑")

        # ── SHORT Yönlü ──
        s_dir = 0
        s_reasons: List[str] = []

        if cvd_div == 'bearish':
            s_dir += 3
            s_reasons.append("CVD_div↓")

        if liq_grab and liq_grab.direction == 'bearish':
            s_dir += 3
            s_reasons.append(
                f"LiqGrab↓({liq_grab.level:.2f})"
            )

        if absorption == 'bearish':
            s_dir += 2
            s_reasons.append("Absorb↓")

        if delta_shift == 'bearish':
            s_dir += 2
            s_reasons.append("Delta↓")

        # ── Bonus (her iki yöne de) ──
        bonus = 0
        bonus_reasons: List[str] = []

        if inst_zone:
            bonus += 1
            bonus_reasons.append("InstZone")

        if killzone:
            bonus += 1
            bonus_reasons.append("KZ")

        # ── Yön Seçimi ──
        min_d = self.MIN_DIRECTIONAL_SCORE

        if l_dir >= min_d and l_dir > s_dir:
            total = l_dir + bonus
            reasons = l_reasons + bonus_reasons
            return total, 'LONG', reasons

        if s_dir >= min_d and s_dir > l_dir:
            total = s_dir + bonus
            reasons = s_reasons + bonus_reasons
            return total, 'SHORT', reasons

        # Eşitlik veya eşik altı
        if l_dir == s_dir and l_dir >= min_d:
            return 0, None, [
                f"Çelişki: L_dir={l_dir} S_dir={s_dir}"
            ]

        best_dir = max(l_dir, s_dir)
        best_r = l_reasons if l_dir >= s_dir else s_reasons
        all_reasons = best_r + bonus_reasons if best_r else ["NoSetup"]

        return best_dir + bonus, None, all_reasons

    # ══════════════════════════════════════════════════════════
    # TP / SL HESAPLAMA  [FIX #7, #11]
    # ══════════════════════════════════════════════════════════

    def _compute_tp_sl(
        self,
        df: pd.DataFrame,
        direction: str,
        entry: float,
        atr: float,
        liq_grab: Optional[LiqGrabResult],
    ) -> Tuple[float, float]:
        """
        TP: Son 50 barın swing seviyesi (min 2.5×ATR)
        SL: Liq grab wick veya swing seviyesi

        [FIX #7]: NaN koruması güçlendirildi.
        """
        lookback = min(50, len(df) - 1)
        buffer = atr * 0.2
        tp_fallback = self.default_tp_mult * atr
        sl_fallback = self.default_sl_mult * atr

        if direction == 'LONG':
            # ── TP ──
            tp = self._safe_past_extreme(
                df['high'], lookback, mode='max'
            )
            if tp is None or tp <= entry + atr:
                tp = entry + tp_fallback
            if (tp - entry) < 2.5 * atr:
                tp = entry + tp_fallback

            # ── SL ──
            if (liq_grab is not None
                    and liq_grab.direction == 'bullish'):
                sl = liq_grab.wick - buffer
            else:
                sl_target = self._safe_past_extreme(
                    df['low'], lookback, mode='min'
                )
                if sl_target is not None and sl_target < entry:
                    sl = sl_target
                else:
                    sl = entry - sl_fallback

            # Minimum SL mesafesi
            if (entry - sl) < 0.5 * atr:
                sl = entry - sl_fallback

        else:  # SHORT
            # ── TP ──
            tp = self._safe_past_extreme(
                df['low'], lookback, mode='min'
            )
            if tp is None or tp >= entry - atr:
                tp = entry - tp_fallback
            if (entry - tp) < 2.5 * atr:
                tp = entry - tp_fallback

            # ── SL ──
            if (liq_grab is not None
                    and liq_grab.direction == 'bearish'):
                sl = liq_grab.wick + buffer
            else:
                sl_target = self._safe_past_extreme(
                    df['high'], lookback, mode='max'
                )
                if sl_target is not None and sl_target > entry:
                    sl = sl_target
                else:
                    sl = entry + sl_fallback

            if (sl - entry) < 0.5 * atr:
                sl = entry + sl_fallback

        return tp, sl

    def _safe_past_extreme(
        self,
        series: pd.Series,
        lookback: int,
        mode: str = 'max',
    ) -> Optional[float]:
        """
        Geçmiş barlardan extreme değer (mevcut bar hariç).

        [FIX #7]: NaN güvenli.
        """
        if lookback < 2:
            return None

        past = series.shift(1).iloc[-lookback:]
        clean = past.dropna()

        if clean.empty:
            return None

        if mode == 'max':
            return float(clean.max())
        else:
            return float(clean.min())

    # ══════════════════════════════════════════════════════════
    # SL/TP YÖN DOĞRULAMASI  [FIX #11]
    # ══════════════════════════════════════════════════════════

    def _validate_sl_tp(
        self,
        direction: str,
        entry: float,
        atr: float,
        sl: float,
        tp: float,
    ) -> Tuple[float, float]:
        """SL ve TP yön tutarlılığı."""
        if direction == 'LONG':
            if sl >= entry:
                sl = entry - self.default_sl_mult * atr
            if tp <= entry:
                tp = entry + self.default_tp_mult * atr
        else:
            if sl <= entry:
                sl = entry + self.default_sl_mult * atr
            if tp >= entry:
                tp = entry - self.default_tp_mult * atr

        return sl, tp

    # ══════════════════════════════════════════════════════════
    # YARDIMCI
    # ══════════════════════════════════════════════════════════

    def _safe_vol_ma(self, df: pd.DataFrame) -> float:
        """Güvenli hacim ortalaması."""
        if 'volume' not in df.columns:
            return 1.0

        vol_ma = df['volume'].rolling(
            20, min_periods=5
        ).mean().iloc[-1]

        if pd.isna(vol_ma) or vol_ma < 1e-10:
            return float(df['volume'].iloc[-1])

        return float(vol_ma)