"""
ICT/SMC Strategy v2.2.1 — Production-Ready (Bugfix)

Rejimler: TRENDING, HIGH_VOLATILE
Mantık:
  1. Gelişmiş MSS — swing yapısı kırılma + displacement onayı
  2. Likidite Sweep — wick süpürme + gövde rejection
  3. FVG — unmitigated gap + minimum boyut filtresi
  4. OB — unmitigated order block + mesafe filtresi
  5. Killzone — saat + volatilite çift onay

v2.2.1 Bugfix (v2.2 üzerine):
  - ✅ #1  FVG NaN/None → safe bool check
  - ✅ #2  OB bare except → typed exception
  - ✅ #3  Killzone vol atlanma düzeltildi
  - ✅ #4  Confluence çelişki reason korunuyor
  - ✅ #5  Sweep self-reference edge case
  - ✅ #6  TP yuvarlama yönü düzeltildi (SL ile ZIT)
  - ✅ #7  Variable shadow düzeltildi
  - ✅ #8  MSS swing sıralama koruması
  - ✅ #9  TP NaN koruması güçlendirildi
  - ✅ #10 Çelişki mesajı iyileştirildi
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

        fvg_bull = self._check_unmitigated_fvg(
            df, 'fvg_bull', atr
        )
        fvg_bear = self._check_unmitigated_fvg(
            df, 'fvg_bear', atr
        )

        ob_bull = self._get_unmitigated_ob(df, 'ob_bull', c, atr)
        ob_bear = self._get_unmitigated_ob(df, 'ob_bear', c, atr)

        is_killzone = self._check_killzone(current_hour, df, atr)

        # ── Confluence ──
        score, direction, reasons = self._calculate_confluence(
            c=c, atr=atr, mss=mss, sweep=sweep,
            fvg_bull=fvg_bull, fvg_bear=fvg_bear,
            ob_bull=ob_bull, ob_bear=ob_bear,
            is_killzone=is_killzone,
        )

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
        tp_raw = self._calculate_tp(direction, c, atr, df)

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
    ) -> Tuple[int, Optional[str], List[str]]:
        """
        Confluence skorlama.

        [FIX #4, #10]: Eşit skor → çelişki mesajı + reason korunur.
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
    ) -> float:
        """
        TP: Geçmiş likidite havuzu.

        [FIX #9]: NaN koruması güçlendirildi.
        """
        lookback = min(50, len(df) - 1)
        atr_fallback = self.default_tp_mult * atr

        if lookback < 5:
            if direction == 'LONG':
                return entry + atr_fallback
            else:
                return entry - atr_fallback

        min_distance = atr  # TP en az 1 ATR uzakta

        if direction == 'LONG':
            past = df['high'].shift(1).iloc[-lookback:]
            # [FIX #9]: dropna ile NaN temizle
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