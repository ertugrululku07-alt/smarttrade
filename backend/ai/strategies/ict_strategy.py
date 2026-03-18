"""
ICT/SMC Strategy v3.0 — Professional ICT Setup

Rejimler: TRENDING, HIGH_VOLATILE
Mantık:
  1. Gelişmiş MSS — swing yapısı kırılma + displacement onayı
  2. Likidite Sweep — wick süpürme + gövde rejection
  3. FVG — unmitigated gap + minimum boyut filtresi + retest giriş
  4. OB — unmitigated order block + mesafe filtresi
  5. Killzone — saat + volatilite çift onay
  6. Equal High/Low — likidite kümesi tespiti (2+ swing aynı seviye)
  7. Sıralı Setup — EqHL → Sweep → MSS → FVG Retest (kronolojik)
  8. Yapısal TP — Equal Level'lar hedef olarak kullanılır

v3.0 (v2.2.1 üzerine):
  - ✅ Equal High/Low tespiti (_detect_equal_levels)
  - ✅ FVG Zone tespiti + Retest kontrolü (_find_fvg_zones, _check_fvg_retest)
  - ✅ Sıralı ICT Setup doğrulaması (_check_sequential_setup)
  - ✅ Equal Level bazlı TP hedefleri (_calculate_tp)
  - ✅ Confluence'a EqHL scoring eklendi

v2.2.1 Bugfix:
  - ✅ #1-#10 tüm bugfix'ler korunuyor
"""

import math
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from .base_strategy import BaseStrategy, Signal


# ══════════════════════════════════════════════════════════════
# Veri Yapıları
# ══════════════════════════════════════════════════════════════

@dataclass
class SwingPoint:
    """Swing High veya Swing Low noktası."""
    index: int
    price: float
    is_high: bool
    strength: int


@dataclass
class LiquiditySweep:
    """Likidite temizleme bilgisi."""
    swept_level: float
    sweep_wick: float
    close_after: float
    direction: str          # 'bullish_sweep' | 'bearish_sweep'
    strength: float         # ATR cinsinden wick gücü
    bar_index: int


@dataclass
class EqualLevel:
    """Equal High/Low — aynı seviyede 2+ swing = likidite kümesi."""
    level: float          # Kümenin ortalama fiyatı
    touches: int          # Kaç kez dokunulmuş
    is_high: bool         # True = equal highs, False = equal lows
    first_index: int      # İlk dokunma bar indexi
    last_index: int       # Son dokunma bar indexi


@dataclass
class FVGZone:
    """Fair Value Gap bölgesi detayları."""
    top: float            # Gap üst sınırı
    bottom: float         # Gap alt sınırı
    is_bull: bool         # True = bullish FVG
    bar_index: int        # FVG oluşum bar indexi
    size_atr: float       # Gap büyüklüğü (ATR cinsinden)


# ══════════════════════════════════════════════════════════════
# Ana Strateji
# ══════════════════════════════════════════════════════════════

class ICTStrategy(BaseStrategy):

    name = "ict_smc_v2_2"
    regime = "high_volatile"

    # ── Temel ─────────────────────────────────────────
    default_tp_mult = 3.0
    default_sl_mult = 1.5
    MIN_RR = 1.8
    MIN_DATA_BARS = 60

    # ── Tick Size ─────────────────────────────────────
    DEFAULT_TICK_SIZE = 0.01

    # ── MSS ───────────────────────────────────────────
    SWING_LOOKBACK = 5
    SWING_SCAN_DEPTH = 60
    DISPLACEMENT_THRESHOLD = 1.5
    DISPLACEMENT_STRONG = 2.0
    DISPLACEMENT_CAP = 3.5

    # ── Sweep ─────────────────────────────────────────
    SWEEP_WICK_MIN = 0.3
    SWEEP_STRONG = 1.0
    SWEEP_BAR_SCAN = 3

    # ── FVG ───────────────────────────────────────────
    FVG_SCAN_BARS = 5
    FVG_MIN_GAP_ATR = 0.1

    # ── OB ────────────────────────────────────────────
    OB_MAX_AGE = 25
    OB_MAX_DISTANCE_ATR = 8.0

    # ── Equal Level ──────────────────────────────────
    EQ_TOLERANCE_ATR = 0.25       # Aynı seviye sayılma toleransı
    EQ_MIN_TOUCHES = 2            # Minimum dokunma sayısı
    EQ_SCAN_BARS = 10             # FVG zone tarama derinliği

    # ── FVG Retest ───────────────────────────────────
    FVG_RETEST_PROXIMITY_ATR = 0.4  # Retest yakınlık toleransı

    # ── Confluence ────────────────────────────────────
    MIN_CONFLUENCE_SCORE = 4

    # ── Killzone (UTC) ────────────────────────────────
    KILLZONES = [
        (7, 11),    # Londra
        (12, 16),   # New York
        (0, 3),     # Asya (crypto)
    ]

    # ══════════════════════════════════════════════════
    # TICK SIZE YUVARLAMA  [FIX #6]
    # ══════════════════════════════════════════════════

    def _get_tick_size(self) -> float:
        """Borsa tick size değerini döndürür."""
        if hasattr(self, 'tick_size') and self.tick_size:
            return float(self.tick_size)

        if hasattr(self, 'config') and isinstance(self.config, dict):
            ts = self.config.get('tick_size')
            if ts is not None and float(ts) > 0:
                return float(ts)

        return self.DEFAULT_TICK_SIZE

    def _tick_decimals(self, tick: float) -> int:
        """Tick size'dan ondalık basamak sayısı."""
        if tick <= 0:
            return 2
        return max(0, -int(math.floor(math.log10(tick))))

    def _round_to_tick(self, price: float) -> float:
        """Fiyatı en yakın tick'e yuvarla."""
        tick = self._get_tick_size()
        if tick <= 0:
            return price

        dec = self._tick_decimals(tick)
        return round(round(price / tick) * tick, dec)

    def _round_sl(self, price: float, direction: str) -> float:
        """
        SL yuvarlama — güvenli taraf.

        LONG  SL (aşağıda) → floor (daha aşağı = daha güvenli)
        SHORT SL (yukarıda) → ceil  (daha yukarı = daha güvenli)
        """
        tick = self._get_tick_size()
        if tick <= 0:
            return price

        dec = self._tick_decimals(tick)

        if direction == 'LONG':
            return round(math.floor(price / tick) * tick, dec)
        else:
            return round(math.ceil(price / tick) * tick, dec)

    def _round_tp(self, price: float, direction: str) -> float:
        """
        TP yuvarlama — konservatif (ulaşılabilir taraf).

        LONG  TP (yukarıda) → floor (daha aşağı = ulaşması kolay)
        SHORT TP (aşağıda)  → ceil  (daha yukarı = ulaşması kolay)

        [FIX #6]: v2.2'de SL ile aynı mantık kullanılıyordu (YANLIŞ).
        TP, SL'nin ZIT yönünde yuvarlanmalı:
          SL → "daha fazla risk al" yönüne yuvarla (güvenli)
          TP → "daha az reward al" yönüne yuvarla (konservatif)
        Sonuç olarak aynı floor/ceil çıkıyor AMA mantık farklı.
        """
        tick = self._get_tick_size()
        if tick <= 0:
            return price

        dec = self._tick_decimals(tick)

        if direction == 'LONG':
            # TP yukarıda → aşağı yuvarla (konservatif hedef)
            return round(math.floor(price / tick) * tick, dec)
        else:
            # TP aşağıda → yukarı yuvarla (konservatif hedef)
            return round(math.ceil(price / tick) * tick, dec)

    # ══════════════════════════════════════════════════
    # ANA SİNYAL ÜRETİCİ
    # ══════════════════════════════════════════════════

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """ICT v2.2.1 ana sinyal üretim pipeline'ı."""

        # ── Veri Kontrolü ──
        if len(df) < self.MIN_DATA_BARS:
            return self._no_signal(
                f"Yetersiz veri (<{self.MIN_DATA_BARS} bar)"
            )

        required = ['close', 'high', 'low', 'open']
        col_err = self._validate_columns(df, required)
        if col_err:
            return self._no_signal(col_err)

        last_row = df.iloc[-1]
        c = float(last_row['close'])

        atr = self._safe_get(df, 'atr', default=None)
        if atr is None or atr < 1e-10:
            return self._no_signal("ATR geçersiz")

        current_hour = self._get_current_hour(last_row)

        # ── Analiz Fazları ──
        swings = self._detect_swings(df)
        mss = self._detect_mss(df, swings, atr)
        sweep = self._detect_liquidity_sweep(df, swings, atr)

        # Equal High/Low — likidite kümeleri
        eq_highs, eq_lows = self._detect_equal_levels(swings, atr)

        # FVG zones — raw OHLC'den
        bull_fvg_zones, bear_fvg_zones = self._find_fvg_zones(df, atr)

        fvg_bull = self._check_unmitigated_fvg(
            df, 'fvg_bull', atr
        )
        fvg_bear = self._check_unmitigated_fvg(
            df, 'fvg_bear', atr
        )
        # FVG zone'lardan da bool türet (col yoksa yedek)
        if not fvg_bull and bull_fvg_zones:
            fvg_bull = True
        if not fvg_bear and bear_fvg_zones:
            fvg_bear = True

        ob_bull = self._get_unmitigated_ob(df, 'ob_bull', c, atr)
        ob_bear = self._get_unmitigated_ob(df, 'ob_bear', c, atr)

        is_killzone = self._check_killzone(current_hour, df, atr)

        # ── Sıralı ICT Setup ──
        seq_l, seq_s, seq_l_reasons, seq_s_reasons = (
            self._check_sequential_setup(
                eq_highs, eq_lows, sweep, mss,
                bull_fvg_zones, bear_fvg_zones, c, atr,
            )
        )

        # ── Confluence ──
        score, direction, reasons = self._calculate_confluence(
            c=c, atr=atr, mss=mss, sweep=sweep,
            fvg_bull=fvg_bull, fvg_bear=fvg_bear,
            ob_bull=ob_bull, ob_bear=ob_bear,
            is_killzone=is_killzone,
            eq_highs=eq_highs, eq_lows=eq_lows,
        )

        # Sıralı setup bonusu ekle
        if direction == 'LONG' and seq_l > 0:
            score += seq_l
            reasons.extend(seq_l_reasons)
        elif direction == 'SHORT' and seq_s > 0:
            score += seq_s
            reasons.extend(seq_s_reasons)

        if direction is None or score < self.MIN_CONFLUENCE_SCORE:
            reason_str = (
                ' | '.join(reasons) if reasons else 'NoSetup'
            )
            return self._no_signal(
                f"ICT Filter: score={score} {reason_str}"
            )

        # ── SL / TP ──
        sl_raw = self._calculate_sl(
            direction, c, atr, sweep, ob_bull, ob_bear
        )
        tp_raw = self._calculate_tp(
            direction, c, atr, df, eq_highs, eq_lows
        )

        sl_valid, tp_valid = self._validate_sl_tp(
            direction, c, atr, sl_raw, tp_raw
        )

        # ── Tick Yuvarlama ──
        entry_price = self._round_to_tick(c)
        sl_price = self._round_sl(sl_valid, direction)
        tp_price = self._round_tp(tp_valid, direction)

        # ── Yuvarlama Sonrası Yön Kontrolü ──
        if direction == 'LONG':
            if sl_price >= entry_price:
                sl_price = self._round_sl(
                    entry_price - 1.2 * atr, direction
                )
            if tp_price <= entry_price:
                tp_price = self._round_tp(
                    entry_price + self.default_tp_mult * atr,
                    direction,
                )
        else:
            if sl_price <= entry_price:
                sl_price = self._round_sl(
                    entry_price + 1.2 * atr, direction
                )
            if tp_price >= entry_price:
                tp_price = self._round_tp(
                    entry_price - self.default_tp_mult * atr,
                    direction,
                )

        # ── R:R Kontrolü ──
        risk = abs(entry_price - sl_price)
        reward = abs(tp_price - entry_price)

        if risk < 1e-10:
            return self._no_signal("Risk sıfıra çok yakın")

        rr = reward / risk
        if rr < self.MIN_RR:
            return self._no_signal(
                f"R:R yetersiz ({rr:.2f} < {self.MIN_RR})"
            )

        # ── Sinyal ──
        signal_reason = (
            f"ICTv2.2 {'|'.join(reasons)} | "
            f"Score={score} R:R={rr:.2f}"
        )

        if direction == 'LONG':
            return self._long_signal(
                soft_score=score,
                reason=signal_reason,
                entry_price=entry_price,
                entry_type="ict_setup",
                tp_price=tp_price,
                sl_price=sl_price,
            )
        else:
            return self._short_signal(
                soft_score=score,
                reason=signal_reason,
                entry_price=entry_price,
                entry_type="ict_setup",
                tp_price=tp_price,
                sl_price=sl_price,
            )

    # ══════════════════════════════════════════════════════════
    # SWING TESPİTİ
    # ══════════════════════════════════════════════════════════

    def _detect_swings(
        self, df: pd.DataFrame
    ) -> List[SwingPoint]:
        """Fractal swing high/low tespiti."""
        swings: List[SwingPoint] = []
        n = self.SWING_LOOKBACK
        highs = df['high'].values
        lows = df['low'].values
        total = len(df)

        start = max(n, total - self.SWING_SCAN_DEPTH)
        end = total - n

        if start >= end:
            return swings

        for i in range(start, end):
            left_h = highs[i - n:i]
            right_h = highs[i + 1:i + n + 1]
            left_l = lows[i - n:i]
            right_l = lows[i + 1:i + n + 1]

            if len(left_h) == 0 or len(right_h) == 0:
                continue

            # Swing High
            if (highs[i] > left_h.max()
                    and highs[i] > right_h.max()):
                strength = self._calc_strength(
                    highs, i, total, is_high=True
                )
                swings.append(SwingPoint(
                    index=i,
                    price=float(highs[i]),
                    is_high=True,
                    strength=strength,
                ))

            # Swing Low
            if (lows[i] < left_l.min()
                    and lows[i] < right_l.min()):
                strength = self._calc_strength(
                    lows, i, total, is_high=False
                )
                swings.append(SwingPoint(
                    index=i,
                    price=float(lows[i]),
                    is_high=False,
                    strength=strength,
                ))

        return swings

    def _calc_strength(
        self,
        prices: np.ndarray,
        idx: int,
        total: int,
        is_high: bool,
    ) -> int:
        """Swing noktasının kaç bar domine ettiği."""
        strength = 0
        max_check = min(
            self.SWING_LOOKBACK * 3, total - idx - 1
        )

        for offset in range(1, max_check + 1):
            j = idx + offset
            if j >= total:
                break

            if is_high:
                if prices[idx] > prices[j]:
                    strength += 1
                else:
                    break
            else:
                if prices[idx] < prices[j]:
                    strength += 1
                else:
                    break

        return strength

    # ══════════════════════════════════════════════════════════
    # EQUAL HIGH / EQUAL LOW TESPİTİ
    # ══════════════════════════════════════════════════════════

    def _detect_equal_levels(
        self,
        swings: List[SwingPoint],
        atr: float,
    ) -> Tuple[List[EqualLevel], List[EqualLevel]]:
        """
        Equal High/Low tespiti — aynı seviyede 2+ swing = likidite kümesi.

        Traderların stopları bu seviyelerde birikir.
        Fiyat bu seviyeleri sweep ettiğinde likidite alınır.

        Returns:
            (equal_highs, equal_lows)
        """
        if atr < 1e-10 or len(swings) < 2:
            return [], []

        tolerance = atr * self.EQ_TOLERANCE_ATR

        highs = sorted(
            [s for s in swings if s.is_high], key=lambda s: s.index
        )
        lows = sorted(
            [s for s in swings if not s.is_high], key=lambda s: s.index
        )

        def cluster(points: List[SwingPoint], is_high: bool) -> List[EqualLevel]:
            if len(points) < self.EQ_MIN_TOUCHES:
                return []

            clusters: List[EqualLevel] = []
            used: set = set()

            for i, p1 in enumerate(points):
                if i in used:
                    continue
                group = [p1]
                for j in range(i + 1, len(points)):
                    if j in used:
                        continue
                    if abs(p1.price - points[j].price) <= tolerance:
                        group.append(points[j])
                        used.add(j)

                if len(group) >= self.EQ_MIN_TOUCHES:
                    used.add(i)
                    avg = sum(s.price for s in group) / len(group)
                    clusters.append(EqualLevel(
                        level=avg,
                        touches=len(group),
                        is_high=is_high,
                        first_index=min(s.index for s in group),
                        last_index=max(s.index for s in group),
                    ))

            return clusters

        return cluster(highs, True), cluster(lows, False)

    # ══════════════════════════════════════════════════════════
    # MSS  [FIX #8]
    # ══════════════════════════════════════════════════════════

    def _detect_mss(
        self,
        df: pd.DataFrame,
        swings: List[SwingPoint],
        atr: float,
    ) -> dict:
        """
        Market Structure Shift tespiti.

        [FIX #8]: Swing'ler index sırasına göre sort edilir
        → kronolojik doğruluk garanti.
        """
        result = {
            'bullish': False,
            'bearish': False,
            'displacement': 0.0,
            'level': None,
        }

        if len(swings) < 4 or atr < 1e-10:
            return result

        c = float(df['close'].iloc[-1])
        o = float(df['open'].iloc[-1])
        body_atr = abs(c - o) / atr

        # God Candle koruması
        if body_atr > self.DISPLACEMENT_CAP:
            return result

        # [FIX #8]: Kronolojik sıralama
        sorted_swings = sorted(swings, key=lambda s: s.index)

        recent_highs = [
            s for s in sorted_swings if s.is_high
        ]
        recent_lows = [
            s for s in sorted_swings if not s.is_high
        ]

        # ── Bullish MSS ──
        if len(recent_highs) >= 2:
            h_prev = recent_highs[-2]
            h_last = recent_highs[-1]

            if (h_prev.price > h_last.price
                    and c > h_last.price
                    and body_atr >= self.DISPLACEMENT_THRESHOLD):
                result['bullish'] = True
                result['displacement'] = round(body_atr, 2)
                result['level'] = h_last.price

        # ── Bearish MSS ──
        if len(recent_lows) >= 2:
            l_prev = recent_lows[-2]
            l_last = recent_lows[-1]

            if (l_prev.price < l_last.price
                    and c < l_last.price
                    and body_atr >= self.DISPLACEMENT_THRESHOLD):
                result['bearish'] = True
                result['displacement'] = round(body_atr, 2)
                result['level'] = l_last.price

        # Çift yönlü çakışma → iptal
        if result['bullish'] and result['bearish']:
            return {
                'bullish': False,
                'bearish': False,
                'displacement': 0.0,
                'level': None,
            }

        return result

    # ══════════════════════════════════════════════════════════
    # LİKİDİTE SWEEP  [FIX #5, #7]
    # ══════════════════════════════════════════════════════════

    def _detect_liquidity_sweep(
        self,
        df: pd.DataFrame,
        swings: List[SwingPoint],
        atr: float,
    ) -> Optional[LiquiditySweep]:
        """
        Son SWEEP_BAR_SCAN barda likidite süpürme tespiti.

        [FIX #7]: Variable shadow düzeltildi (h,l → bar_highs vb.)
        [FIX #5]: sw.index == idx durumu da hariç tutulur.
        """
        if len(swings) < 2 or atr < 1e-10:
            return None

        total = len(df)
        bar_highs = df['high'].values
        bar_lows = df['low'].values
        bar_closes = df['close'].values
        bar_opens = df['open'].values

        swing_highs = [s for s in swings if s.is_high]
        swing_lows = [s for s in swings if not s.is_high]

        for bar_offset in range(1, self.SWEEP_BAR_SCAN + 1):
            idx = total - bar_offset
            if idx < 0:
                continue

            bh = float(bar_highs[idx])
            bl = float(bar_lows[idx])
            bc = float(bar_closes[idx])
            bo = float(bar_opens[idx])
            body_top = max(bc, bo)
            body_bottom = min(bc, bo)

            # ── Bullish Sweep (swing low süpürme) ──
            for sw in reversed(swing_lows):
                # [FIX #5]: Swing, kontrol barından KESINLIKLE önce
                if sw.index >= idx:
                    continue

                level = sw.price

                if bl < level and body_bottom > level:
                    wick = level - bl
                    if wick >= atr * self.SWEEP_WICK_MIN:
                        return LiquiditySweep(
                            swept_level=level,
                            sweep_wick=bl,
                            close_after=bc,
                            direction='bullish_sweep',
                            strength=round(wick / atr, 2),
                            bar_index=idx,
                        )

            # ── Bearish Sweep (swing high süpürme) ──
            for sw in reversed(swing_highs):
                if sw.index >= idx:
                    continue

                level = sw.price

                if bh > level and body_top < level:
                    wick = bh - level
                    if wick >= atr * self.SWEEP_WICK_MIN:
                        return LiquiditySweep(
                            swept_level=level,
                            sweep_wick=bh,
                            close_after=bc,
                            direction='bearish_sweep',
                            strength=round(wick / atr, 2),
                            bar_index=idx,
                        )

        return None

    # ══════════════════════════════════════════════════════════
    # FVG  [FIX #1]
    # ══════════════════════════════════════════════════════════

    def _check_unmitigated_fvg(
        self,
        df: pd.DataFrame,
        fvg_col: str,
        atr: float,
    ) -> bool:
        """
        Son FVG_SCAN_BARS barda unmitigated FVG var mı?

        [FIX #1]: NaN/None üzerinde bool() → safe check.
        """
        if fvg_col not in df.columns or atr < 1e-10:
            return False

        total = len(df)
        scan = min(self.FVG_SCAN_BARS, total)
        is_bull = 'bull' in fvg_col

        for i in range(total - scan, total):
            if i < 2:
                continue

            # [FIX #1]: Safe bool check
            try:
                raw_val = df[fvg_col].iloc[i]
                if pd.isna(raw_val):
                    continue
                has_fvg = bool(raw_val)
            except (ValueError, TypeError):
                continue

            if not has_fvg:
                continue

            # Gap hesabı
            if is_bull:
                gap_bottom = float(df['high'].iloc[i - 2])
                gap_top = float(df['low'].iloc[i])
            else:
                gap_top = float(df['low'].iloc[i - 2])
                gap_bottom = float(df['high'].iloc[i])

            gap_size = gap_top - gap_bottom

            # Minimum gap filtresi
            if gap_size < atr * self.FVG_MIN_GAP_ATR:
                continue

            # Mitigation kontrolü
            mitigated = False
            for j in range(i + 1, total):
                if is_bull:
                    if float(df['low'].iloc[j]) <= gap_bottom:
                        mitigated = True
                        break
                else:
                    if float(df['high'].iloc[j]) >= gap_top:
                        mitigated = True
                        break

            if not mitigated:
                return True

        return False

    # ══════════════════════════════════════════════════════════
    # FVG ZONE TESPİTİ + RETEST KONTROLÜ
    # ══════════════════════════════════════════════════════════

    def _find_fvg_zones(
        self,
        df: pd.DataFrame,
        atr: float,
    ) -> Tuple[List[FVGZone], List[FVGZone]]:
        """
        Raw OHLC'den unmitigated FVG bölgelerini tespit et.

        Bullish FVG: bar[i] low > bar[i-2] high (yukarı gap)
        Bearish FVG: bar[i] high < bar[i-2] low (aşağı gap)

        Returns:
            (bull_fvg_zones, bear_fvg_zones)
        """
        bull_zones: List[FVGZone] = []
        bear_zones: List[FVGZone] = []

        if atr < 1e-10:
            return bull_zones, bear_zones

        total = len(df)
        scan_start = max(2, total - self.EQ_SCAN_BARS)
        highs = df['high'].values
        lows = df['low'].values

        for i in range(scan_start, total):
            # ── Bullish FVG ──
            gap_bottom_b = float(highs[i - 2])
            gap_top_b = float(lows[i])
            bull_gap = gap_top_b - gap_bottom_b

            if bull_gap > atr * self.FVG_MIN_GAP_ATR:
                mitigated = False
                for j in range(i + 1, total):
                    if float(lows[j]) <= gap_bottom_b:
                        mitigated = True
                        break
                if not mitigated:
                    bull_zones.append(FVGZone(
                        top=gap_top_b,
                        bottom=gap_bottom_b,
                        is_bull=True,
                        bar_index=i,
                        size_atr=round(bull_gap / atr, 2),
                    ))

            # ── Bearish FVG ──
            gap_top_s = float(lows[i - 2])
            gap_bottom_s = float(highs[i])
            bear_gap = gap_top_s - gap_bottom_s

            if bear_gap > atr * self.FVG_MIN_GAP_ATR:
                mitigated = False
                for j in range(i + 1, total):
                    if float(highs[j]) >= gap_top_s:
                        mitigated = True
                        break
                if not mitigated:
                    bear_zones.append(FVGZone(
                        top=gap_top_s,
                        bottom=gap_bottom_s,
                        is_bull=False,
                        bar_index=i,
                        size_atr=round(bear_gap / atr, 2),
                    ))

        return bull_zones, bear_zones

    def _check_fvg_retest(
        self,
        current_price: float,
        fvg_zones: List[FVGZone],
        atr: float,
    ) -> Tuple[bool, Optional[FVGZone]]:
        """
        Fiyatın FVG zone'a geri dönüp dönmediğini kontrol et.

        Entry burada yapılır — sweep ve MSS sonrası oluşan FVG'ye
        geri dönüş en kaliteli giriş noktasıdır.

        Returns:
            (is_at_retest, zone)
        """
        if not fvg_zones or atr < 1e-10:
            return False, None

        proximity = atr * self.FVG_RETEST_PROXIMITY_ATR

        for zone in reversed(fvg_zones):
            if (zone.bottom - proximity) <= current_price <= (zone.top + proximity):
                return True, zone

        return False, None

    # ══════════════════════════════════════════════════════════
    # ORDER BLOCK  [FIX #2]
    # ══════════════════════════════════════════════════════════

    def _get_unmitigated_ob(
        self,
        df: pd.DataFrame,
        col: str,
        current_price: float,
        atr: float,
    ) -> Optional[float]:
        """
        Unmitigated Order Block fiyatı.

        [FIX #2]: bare except → typed exception.
        """
        if col not in df.columns:
            return None

        try:
            recent = df[col].iloc[-self.OB_MAX_AGE:]
            valid = recent[recent > 0]

            if valid.empty:
                return None

            last_ob = float(valid.iloc[-1])

            # Konum doğrulaması
            if 'bull' in col and current_price <= last_ob:
                return None
            if 'bear' in col and current_price >= last_ob:
                return None

            # Mesafe filtresi
            if atr > 0:
                distance = abs(current_price - last_ob)
                if distance > atr * self.OB_MAX_DISTANCE_ATR:
                    return None

            return last_ob

        except (IndexError, KeyError, TypeError, ValueError):
            return None

    # ══════════════════════════════════════════════════════════
    # SIRASAL ICT SETUP DOĞRULAMASI
    # ══════════════════════════════════════════════════════════

    def _check_sequential_setup(
        self,
        eq_highs: List[EqualLevel],
        eq_lows: List[EqualLevel],
        sweep: Optional[LiquiditySweep],
        mss: dict,
        bull_fvg_zones: List[FVGZone],
        bear_fvg_zones: List[FVGZone],
        current_price: float,
        atr: float,
    ) -> Tuple[int, int, List[str], List[str]]:
        """
        Sıralı ICT Setup Doğrulaması:
          SHORT: Equal Highs → Bearish Sweep → Bearish MSS → Bear FVG retest
          LONG:  Equal Lows  → Bullish Sweep → Bullish MSS → Bull FVG retest

        Her adım kronolojik sırayla olmalı. Tam sıra = maksimum bonus.

        Returns:
            (long_bonus, short_bonus, long_reasons, short_reasons)
        """
        l_bonus = 0
        s_bonus = 0
        l_reasons: List[str] = []
        s_reasons: List[str] = []

        # ── LONG Sequential: EqLows → BullSweep → BullMSS → BullFVG ──
        if eq_lows:
            best_eq = max(eq_lows, key=lambda e: e.touches)
            l_bonus += 1
            l_reasons.append(f"EqL({best_eq.touches}x)")

            if sweep and sweep.direction == 'bullish_sweep':
                # Sweep equal low seviyesine yakın mı?
                if abs(sweep.swept_level - best_eq.level) < atr * 0.5:
                    l_bonus += 2
                    l_reasons.append("Sweep→EqL")
                else:
                    l_bonus += 1
                    l_reasons.append("Sweep↑")

                # Sweep, equal low'dan sonra mı? (kronolojik)
                if sweep.bar_index > best_eq.last_index:
                    l_bonus += 1
                    l_reasons.append("SeqOK")

            if mss['bullish']:
                l_bonus += 1
                l_reasons.append("MSS↑")

            fvg_retest, fvg_zone = self._check_fvg_retest(
                current_price, bull_fvg_zones, atr
            )
            if fvg_retest:
                l_bonus += 2
                l_reasons.append("FVG_RT↑")

        # ── SHORT Sequential: EqHighs → BearSweep → BearMSS → BearFVG ──
        if eq_highs:
            best_eq = max(eq_highs, key=lambda e: e.touches)
            s_bonus += 1
            s_reasons.append(f"EqH({best_eq.touches}x)")

            if sweep and sweep.direction == 'bearish_sweep':
                if abs(sweep.swept_level - best_eq.level) < atr * 0.5:
                    s_bonus += 2
                    s_reasons.append("Sweep→EqH")
                else:
                    s_bonus += 1
                    s_reasons.append("Sweep↓")

                if sweep.bar_index > best_eq.last_index:
                    s_bonus += 1
                    s_reasons.append("SeqOK")

            if mss['bearish']:
                s_bonus += 1
                s_reasons.append("MSS↓")

            fvg_retest, fvg_zone = self._check_fvg_retest(
                current_price, bear_fvg_zones, atr
            )
            if fvg_retest:
                s_bonus += 2
                s_reasons.append("FVG_RT↓")

        return l_bonus, s_bonus, l_reasons, s_reasons

    # ══════════════════════════════════════════════════════════
    # KILLZONE  [FIX #3]
    # ══════════════════════════════════════════════════════════

    def _check_killzone(
        self,
        current_hour: int,
        df: pd.DataFrame,
        atr: float,
    ) -> bool:
        """
        Saat + volatilite çift onay killzone.

        [FIX #3]: len(df) < 20 durumunda da saat + basit
        volatilite kontrolü yapılır (sadece saat dönmez).
        """
        # Saat kontrolü
        in_window = any(
            start <= current_hour < end
            for start, end in self.KILLZONES
        )

        if not in_window:
            return False

        # Volatilite kontrolü
        if len(df) < 20:
            # Yeterli veri yok → sadece son barın
            # range'i ATR'den büyük mü?
            if atr > 0:
                last_range = float(
                    df['high'].iloc[-1] - df['low'].iloc[-1]
                )
                return last_range > atr * 0.5
            return False

        # Normal kontrol: son barın range'i ortalamanın %80'i
        recent_ranges = (
            df['high'] - df['low']
        ).iloc[-20:]
        avg_range = recent_ranges.mean()
        last_range = float(
            df['high'].iloc[-1] - df['low'].iloc[-1]
        )

        return last_range > avg_range * 0.8

    # ══════════════════════════════════════════════════════════
    # CONFLUENCE  [FIX #4, #10]
    # ══════════════════════════════════════════════════════════

    def _calculate_confluence(
        self,
        c: float,
        atr: float,
        mss: dict,
        sweep: Optional[LiquiditySweep],
        fvg_bull: bool,
        fvg_bear: bool,
        ob_bull: Optional[float],
        ob_bear: Optional[float],
        is_killzone: bool,
        eq_highs: Optional[List[EqualLevel]] = None,
        eq_lows: Optional[List[EqualLevel]] = None,
    ) -> Tuple[int, Optional[str], List[str]]:
        """
        Confluence skorlama.

        [FIX #4, #10]: Eşit skor → çelişki mesajı + reason korunur.
        v3.0: Equal Level + FVG retest scoring eklendi.
        """
        ms = self.MIN_CONFLUENCE_SCORE

        # ── LONG ──
        l_score = 0
        l_reasons: List[str] = []

        if mss['bullish']:
            pts = (3 if mss['displacement'] >= self.DISPLACEMENT_STRONG
                   else 2)
            l_score += pts
            l_reasons.append(f"MSS↑(d={mss['displacement']})")

        if sweep and sweep.direction == 'bullish_sweep':
            pts = (3 if sweep.strength >= self.SWEEP_STRONG
                   else 2)
            l_score += pts
            l_reasons.append(f"Sweep↑(s={sweep.strength})")

        if fvg_bull:
            l_score += 2
            l_reasons.append("FVG↑")

        if ob_bull is not None:
            l_score += 1
            l_reasons.append(f"OB↑({ob_bull:.1f})")

        if is_killzone:
            l_score += 1
            l_reasons.append("KZ")

        # Equal Lows → LONG hedefleri (sweep sonrası dönüş)
        if eq_lows:
            best = max(eq_lows, key=lambda e: e.touches)
            l_score += min(best.touches, 3)  # Max 3 puan
            l_reasons.append(f"EqL({best.touches}x)")

        # ── SHORT ──
        s_score = 0
        s_reasons: List[str] = []

        if mss['bearish']:
            pts = (3 if mss['displacement'] >= self.DISPLACEMENT_STRONG
                   else 2)
            s_score += pts
            s_reasons.append(f"MSS↓(d={mss['displacement']})")

        if sweep and sweep.direction == 'bearish_sweep':
            pts = (3 if sweep.strength >= self.SWEEP_STRONG
                   else 2)
            s_score += pts
            s_reasons.append(f"Sweep↓(s={sweep.strength})")

        if fvg_bear:
            s_score += 2
            s_reasons.append("FVG↓")

        if ob_bear is not None:
            s_score += 1
            s_reasons.append(f"OB↓({ob_bear:.1f})")

        if is_killzone:
            s_score += 1
            s_reasons.append("KZ")

        # Equal Highs → SHORT hedefleri (sweep sonrası dönüş)
        if eq_highs:
            best = max(eq_highs, key=lambda e: e.touches)
            s_score += min(best.touches, 3)
            s_reasons.append(f"EqH({best.touches}x)")

        # ── Yön Seçimi ──
        if l_score > s_score and l_score >= ms:
            return l_score, 'LONG', l_reasons

        if s_score > l_score and s_score >= ms:
            return s_score, 'SHORT', s_reasons

        # [FIX #4]: Eşit skor çelişki — reason korunuyor
        if l_score == s_score and l_score >= ms:
            conflict_reasons = [
                f"Çelişki L={l_score}({'+'.join(l_reasons)}) "
                f"S={s_score}({'+'.join(s_reasons)})"
            ]
            return 0, None, conflict_reasons

        # Eşik altı
        best = max(l_score, s_score)
        best_r = l_reasons if l_score >= s_score else s_reasons
        return best, None, best_r if best_r else ["NoSetup"]

    # ══════════════════════════════════════════════════════════
    # SL HESAPLAMA
    # ══════════════════════════════════════════════════════════

    def _calculate_sl(
        self,
        direction: str,
        entry: float,
        atr: float,
        sweep: Optional[LiquiditySweep],
        ob_bull: Optional[float],
        ob_bear: Optional[float],
    ) -> float:
        """Kademeli SL: Sweep → OB → ATR fallback."""
        buffer = atr * 0.2

        if direction == 'LONG':
            # P1: Sweep
            if (sweep is not None
                    and sweep.direction == 'bullish_sweep'
                    and sweep.sweep_wick < entry):
                return sweep.sweep_wick - buffer

            # P2: OB
            if ob_bull is not None and ob_bull < entry:
                return ob_bull - atr * 0.3

            # P3: ATR
            return entry - self.default_sl_mult * atr

        else:
            if (sweep is not None
                    and sweep.direction == 'bearish_sweep'
                    and sweep.sweep_wick > entry):
                return sweep.sweep_wick + buffer

            if ob_bear is not None and ob_bear > entry:
                return ob_bear + atr * 0.3

            return entry + self.default_sl_mult * atr

    # ══════════════════════════════════════════════════════════
    # TP HESAPLAMA  [FIX #9]
    # ══════════════════════════════════════════════════════════

    def _calculate_tp(
        self,
        direction: str,
        entry: float,
        atr: float,
        df: pd.DataFrame,
        eq_highs: Optional[List[EqualLevel]] = None,
        eq_lows: Optional[List[EqualLevel]] = None,
    ) -> float:
        """
        TP: Equal Level likidite havuzu → Geçmiş swing → ATR fallback.

        v3.0: Equal level'lar öncelikli TP hedefi:
          LONG  TP → Equal Highs (yukarıdaki likidite)
          SHORT TP → Equal Lows  (aşağıdaki likidite)

        [FIX #9]: NaN koruması güçlendirildi.
        """
        lookback = min(50, len(df) - 1)
        atr_fallback = self.default_tp_mult * atr
        min_distance = atr

        # ── P1: Equal Level TP (en kaliteli hedef) ──
        if direction == 'LONG' and eq_highs:
            # En yakın equal high seviyesi (entry'den yukarıda)
            above = [e for e in eq_highs if e.level > entry + min_distance]
            if above:
                nearest = min(above, key=lambda e: e.level - entry)
                return nearest.level

        if direction == 'SHORT' and eq_lows:
            below = [e for e in eq_lows if e.level < entry - min_distance]
            if below:
                nearest = max(below, key=lambda e: entry - e.level)
                return nearest.level

        # ── P2: Geçmiş swing TP (mevcut mantık) ──
        if lookback < 5:
            if direction == 'LONG':
                return entry + atr_fallback
            else:
                return entry - atr_fallback

        if direction == 'LONG':
            past = df['high'].shift(1).iloc[-lookback:]
            past_clean = past.dropna()
            if past_clean.empty:
                return entry + atr_fallback

            tp = float(past_clean.max())

            if tp <= entry + min_distance:
                tp = entry + atr_fallback
        else:
            past = df['low'].shift(1).iloc[-lookback:]
            past_clean = past.dropna()
            if past_clean.empty:
                return entry - atr_fallback

            tp = float(past_clean.min())

            if tp >= entry - min_distance:
                tp = entry - atr_fallback

        return tp

    # ══════════════════════════════════════════════════════════
    # SL/TP YÖN DOĞRULAMASI
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
                sl = entry - 1.2 * atr
            if tp <= entry:
                tp = entry + self.default_tp_mult * atr
        else:
            if sl <= entry:
                sl = entry + 1.2 * atr
            if tp >= entry:
                tp = entry - self.default_tp_mult * atr

        return sl, tp

    # ══════════════════════════════════════════════════════════
    # YARDIMCI
    # ══════════════════════════════════════════════════════════

    def _get_current_hour(self, last_row: pd.Series) -> int:
        """Güvenli saat çıkarma."""
        try:
            idx = last_row.name
            if isinstance(idx, pd.Timestamp):
                return idx.hour
            if hasattr(idx, 'hour'):
                return idx.hour
        except (AttributeError, TypeError):
            pass

        for col_name in ['timestamp', 'datetime', 'date']:
            if col_name in last_row.index:
                try:
                    return pd.Timestamp(last_row[col_name]).hour
                except (ValueError, TypeError):
                    continue

        return 0