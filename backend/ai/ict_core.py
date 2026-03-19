"""
═══════════════════════════════════════════════════════════════════
 ICT / SMC CORE ENGINE v1.1 (düzeltilmiş)
 Tüm Smart Money Concepts primitive tespitleri
═══════════════════════════════════════════════════════════════════

 Bölüm 1:  Swing Points (proper pivot)
 Bölüm 2:  Market Structure (HH/HL/LL/LH, BOS, CHoCH)
 Bölüm 3:  Equal Highs / Equal Lows (liquidity pools)
 Bölüm 4:  Order Blocks (OB) + Mitigation tracking
 Bölüm 5:  Fair Value Gap (FVG) + Consequent Encroachment (CE)
 Bölüm 6:  Displacement (impulsive move)
 Bölüm 7:  Breaker Blocks
 Bölüm 8:  OTE Zone (Fib 61.8–78.6)
 Bölüm 9:  Inducement
 Bölüm 10: Session / Killzones / Asian Range
 Bölüm 11: POI Confluence scorer
 Bölüm 12: Unified analyze() — tek çağrı, tam analiz

 v1.1 Düzeltmeler:
   - BOS/CHoCH çift kayıt sorunu giderildi + h/l dizileri kullanılıyor
   - analyze_market_structure → h/l parametre eklendi
   - OTE Zone → son ardışık impulse bazlı (yanlış swing seçimi düzeltildi)
   - Sweep target'lar fiyata göre sıralanıyor
   - OB tek mum minimum gövde kontrolü eklendi
   - FVG filled → close bazlı (wick değil)
   - Session default → gerçek UTC saati
   - ATR np.roll wrap-around düzeltildi
   - OB freshness POI confluence'a dahil edildi
   - FVG quality filter gevşetildi (0.3 → 0.2)
═══════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# ─────────────────── DATA CLASSES ───────────────────

@dataclass
class SwingPoint:
    index: int
    price: float
    type: str          # 'high' | 'low'
    strength: int = 2  # kaç mum onaylıyor (sol+sağ)


@dataclass
class StructureLabel:
    """Tek bir swing'e atanan yapı etiketi."""
    index: int
    price: float
    label: str   # 'HH' | 'HL' | 'LL' | 'LH'


@dataclass
class StructureBreak:
    """BOS veya CHoCH olayı."""
    index: int
    price: float
    type: str       # 'BOS' | 'CHoCH'
    direction: str  # 'bullish' | 'bearish'
    broken_level: float


@dataclass
class EqualLevel:
    """Equal Highs veya Equal Lows."""
    price: float
    type: str         # 'EQH' | 'EQL'
    touches: int
    indices: List[int]
    swept: bool = False
    sweep_index: int = -1


@dataclass
class OrderBlock:
    top: float
    bottom: float
    type: str           # 'bullish' | 'bearish'
    index: int
    mitigated: bool = False
    test_count: int = 0
    has_fvg: bool = False
    displacement_strength: float = 0.0
    freshness: float = 1.0  # 1.0=yeni, 0.0=eski (30+ bar)


@dataclass
class FVGZone:
    top: float
    bottom: float
    ce: float           # Consequent Encroachment (midpoint)
    type: str           # 'bullish' | 'bearish'
    index: int
    filled: bool = False
    quality: float = 1.0  # FVG size / ATR — < 0.2 = zayıf


@dataclass
class BreakerBlock:
    top: float
    bottom: float
    type: str           # 'bullish' | 'bearish'  (yeni yönü)
    index: int


@dataclass
class OTEZone:
    top: float          # Fib 61.8%
    bottom: float       # Fib 78.6%
    direction: str      # 'bullish' | 'bearish'
    fib_618: float
    fib_705: float      # ideal entry
    fib_786: float


@dataclass
class SessionInfo:
    killzone: str       # 'london' | 'ny_am' | 'ny_pm' | 'asia' | 'off'
    quality: float      # 0-1
    asian_high: float
    asian_low: float
    daily_bias: str     # 'bullish' | 'bearish' | 'neutral'
    weekday: int        # 0=Mon ... 6=Sun
    weekly_bias: str    # 'early' | 'mid' | 'late'


@dataclass
class ICTAnalysis:
    """Tek TF analiz sonucu."""
    # Structure
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)
    structure_labels: List[StructureLabel] = field(default_factory=list)
    market_structure: str = 'ranging'   # 'bullish' | 'bearish' | 'ranging'
    structure_breaks: List[StructureBreak] = field(default_factory=list)
    last_bos: Optional[StructureBreak] = None
    last_choch: Optional[StructureBreak] = None

    # Liquidity
    equal_highs: List[EqualLevel] = field(default_factory=list)
    equal_lows: List[EqualLevel] = field(default_factory=list)
    sweep_detected: bool = False
    sweep_type: str = ''       # 'ssl_sweep' | 'bsl_sweep'
    sweep_level: float = 0.0

    # POI
    order_blocks: List[OrderBlock] = field(default_factory=list)
    fvg_zones: List[FVGZone] = field(default_factory=list)
    breaker_blocks: List[BreakerBlock] = field(default_factory=list)

    # Displacement
    displacement: bool = False
    displacement_direction: str = ''

    # OTE
    ote: Optional[OTEZone] = None

    # Session
    session: Optional[SessionInfo] = None

    # Confluence
    poi_confluence: float = 0.0
    poi_details: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# BÖLÜM 1: SWING POINTS
# ═══════════════════════════════════════════════════════════════

def detect_swing_points(
    h: np.ndarray, l: np.ndarray,
    left: int = 3, right: int = 3,
) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    Proper pivot detection.
    Swing High: h[i] > h[i-left:i] AND h[i] > h[i+1:i+right+1]
    Swing Low:  l[i] < l[i-left:i] AND l[i] < l[i+1:i+right+1]
    """
    n = len(h)
    swing_highs = []
    swing_lows = []

    for i in range(left, n - right):
        # Swing High
        is_sh = True
        for j in range(1, left + 1):
            if h[i] <= h[i - j]:
                is_sh = False
                break
        if is_sh:
            for j in range(1, right + 1):
                if h[i] <= h[i + j]:
                    is_sh = False
                    break
        if is_sh:
            swing_highs.append(SwingPoint(i, float(h[i]), 'high', left + right))

        # Swing Low
        is_sl = True
        for j in range(1, left + 1):
            if l[i] >= l[i - j]:
                is_sl = False
                break
        if is_sl:
            for j in range(1, right + 1):
                if l[i] >= l[i + j]:
                    is_sl = False
                    break
        if is_sl:
            swing_lows.append(SwingPoint(i, float(l[i]), 'low', left + right))

    return swing_highs, swing_lows


# ═══════════════════════════════════════════════════════════════
# BÖLÜM 2: MARKET STRUCTURE (BOS / CHoCH)
# ═══════════════════════════════════════════════════════════════

def analyze_market_structure(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    c: np.ndarray,
    h: np.ndarray = None,
    l: np.ndarray = None,
) -> Tuple[str, List[StructureLabel], List[StructureBreak]]:
    """
    HH/HL/LL/LH etiketleme + BOS/CHoCH tespiti.

    BOS  = Mevcut trend DEVAM → trend yönünde yeni swing kırar
    CHoCH = Trend DEĞİŞİYOR → ters yöndeki kritik swing kırar

    v1.1: h/l dizileri eklendi, çift kayıt sorunu giderildi.

    Returns: (market_structure, labels, breaks)
    """
    labels: List[StructureLabel] = []
    breaks: List[StructureBreak] = []

    # h/l fallback
    if h is None:
        h = c
    if l is None:
        l = c

    # Tüm swing'leri index'e göre sırala
    all_swings = []
    for sh in swing_highs:
        all_swings.append(('high', sh))
    for sl in swing_lows:
        all_swings.append(('low', sl))
    all_swings.sort(key=lambda x: x[1].index)

    if len(all_swings) < 4:
        return 'ranging', labels, breaks

    # HH/HL/LL/LH etiketleme
    prev_high = None
    prev_low = None

    for stype, sp in all_swings:
        if stype == 'high':
            if prev_high is not None:
                if sp.price > prev_high.price:
                    labels.append(StructureLabel(sp.index, sp.price, 'HH'))
                else:
                    labels.append(StructureLabel(sp.index, sp.price, 'LH'))
            prev_high = sp
        else:  # low
            if prev_low is not None:
                if sp.price > prev_low.price:
                    labels.append(StructureLabel(sp.index, sp.price, 'HL'))
                else:
                    labels.append(StructureLabel(sp.index, sp.price, 'LL'))
            prev_low = sp

    # Son 6 label'dan trend analizi
    recent = labels[-6:] if len(labels) >= 6 else labels
    recent_labels = [lb.label for lb in recent]

    hh_count = recent_labels.count('HH')
    hl_count = recent_labels.count('HL')
    ll_count = recent_labels.count('LL')
    lh_count = recent_labels.count('LH')

    if hh_count + hl_count > ll_count + lh_count:
        current_trend = 'bullish'
    elif ll_count + lh_count > hh_count + hl_count:
        current_trend = 'bearish'
    else:
        current_trend = 'ranging'

    # ── BOS / CHoCH tespiti ──
    if len(labels) < 3:
        return current_trend, labels, breaks

    # Son yapı seviyelerini bul
    last_hl = None
    last_lh = None
    last_hh = None
    last_ll = None

    for lb in reversed(labels):
        if lb.label == 'HL' and last_hl is None:
            last_hl = lb
        elif lb.label == 'LH' and last_lh is None:
            last_lh = lb
        elif lb.label == 'HH' and last_hh is None:
            last_hh = lb
        elif lb.label == 'LL' and last_ll is None:
            last_ll = lb
        if all([last_hl, last_lh, last_hh, last_ll]):
            break

    # Son bar + son N bar kontrolü (birleşik — çift kayıt önlenir)
    # v1.1: 5 → 20 bar. 1H'de 5 bar = 5 saat (çok kısa — pompadaki BOS
    # pullback'e geldiğinde kayboluyordu). 20 bar = 20 saat → pullback
    # penceresi yakalanır.
    n_bars = min(20, len(c) - 1)
    found_bullish = False
    found_bearish = False
    found_bull_bos = False
    found_bear_bos = False

    for bi in range(0, n_bars + 1):
        idx = len(c) - 1 - bi
        if idx < 0:
            break

        bar_high = float(h[idx])
        bar_low = float(l[idx])
        bar_close = float(c[idx])

        # ── CHoCH: Trend DEĞİŞİYOR ──
        # Bearish→Bullish CHoCH: bar, son LH üstüne çıktı
        if not found_bullish and last_lh and current_trend in ('bearish', 'ranging'):
            if bar_high > last_lh.price and bar_close > last_lh.price:
                breaks.append(StructureBreak(
                    idx, bar_close, 'CHoCH', 'bullish', last_lh.price
                ))
                found_bullish = True

        # Bullish→Bearish CHoCH: bar, son HL altına indi
        if not found_bearish and last_hl and current_trend in ('bullish', 'ranging'):
            if bar_low < last_hl.price and bar_close < last_hl.price:
                breaks.append(StructureBreak(
                    idx, bar_close, 'CHoCH', 'bearish', last_hl.price
                ))
                found_bearish = True

        # ── BOS: Trend DEVAM ediyor ──
        # Bullish BOS: Son HH kırıldı
        if not found_bull_bos and last_hh and current_trend in ('bullish', 'ranging'):
            if bar_close > last_hh.price:
                breaks.append(StructureBreak(
                    idx, bar_close, 'BOS', 'bullish', last_hh.price
                ))
                found_bull_bos = True

        # Bearish BOS: Son LL kırıldı
        if not found_bear_bos and last_ll and current_trend in ('bearish', 'ranging'):
            if bar_close < last_ll.price:
                breaks.append(StructureBreak(
                    idx, bar_close, 'BOS', 'bearish', last_ll.price
                ))
                found_bear_bos = True

    return current_trend, labels, breaks


# ═══════════════════════════════════════════════════════════════
# BÖLÜM 3: EQUAL HIGHS / EQUAL LOWS
# ═══════════════════════════════════════════════════════════════

def detect_equal_levels(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    atr: float,
    tolerance_atr: float = 0.25,
    min_touches: int = 2,
) -> Tuple[List[EqualLevel], List[EqualLevel]]:
    """
    Equal Highs: 2+ swing high aynı seviyeye ±tolerance
    Equal Lows:  2+ swing low aynı seviyeye ±tolerance

    Bunlar likidite havuzları — büyük para buraya gelir.
    """
    tol = atr * tolerance_atr

    eq_highs = _cluster_levels(swing_highs, tol, min_touches, 'EQH')
    eq_lows = _cluster_levels(swing_lows, tol, min_touches, 'EQL')

    return eq_highs, eq_lows


def _cluster_levels(
    points: List[SwingPoint], tol: float, min_touches: int, eq_type: str,
) -> List[EqualLevel]:
    """Swing point'leri tolerans içinde grupla."""
    if not points:
        return []

    used = set()
    levels = []

    for i, p in enumerate(points):
        if i in used:
            continue
        cluster = [p]
        cluster_idx = [p.index]
        used.add(i)

        for j, q in enumerate(points):
            if j in used:
                continue
            if abs(p.price - q.price) <= tol:
                cluster.append(q)
                cluster_idx.append(q.index)
                used.add(j)

        if len(cluster) >= min_touches:
            avg_price = sum(sp.price for sp in cluster) / len(cluster)
            levels.append(EqualLevel(
                price=avg_price,
                type=eq_type,
                touches=len(cluster),
                indices=cluster_idx,
            ))

    return levels


def check_sweep(
    eq_highs: List[EqualLevel],
    eq_lows: List[EqualLevel],
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    h: np.ndarray, l: np.ndarray, c: np.ndarray,
    lookback: int = 8,
) -> Tuple[bool, str, float]:
    """
    Likidite sweep kontrolü:
      SSL Sweep (bullish): Son barlar EQL/swing low altına inip close üstünde
      BSL Sweep (bearish): Son barlar EQH/swing high üstüne çıkıp close altında

    v1.1: Target'lar fiyata göre sıralanıyor (en yakın önce).

    Returns: (sweep_detected, sweep_type, sweep_level)
    """
    n = len(c)
    if n < lookback + 1:
        return False, '', 0.0

    last_close = float(c[-1])

    # ── SSL Sweep (Buy signal) — alttan sweep ──
    ssl_targets = []
    for eq in eq_lows:
        if not eq.swept:
            ssl_targets.append(eq.price)
    for sl in swing_lows[-5:]:
        ssl_targets.append(sl.price)

    # En yakın (en yüksek) target önce — daha anlamlı sweep
    ssl_targets = sorted(set(ssl_targets), reverse=True)

    for target in ssl_targets:
        recent_low = float(np.min(l[-lookback:]))
        if recent_low < target and last_close > target:
            return True, 'ssl_sweep', target

    # ── BSL Sweep (Sell signal) — üstten sweep ──
    bsl_targets = []
    for eq in eq_highs:
        if not eq.swept:
            bsl_targets.append(eq.price)
    for sh in swing_highs[-5:]:
        bsl_targets.append(sh.price)

    # En yakın (en düşük) target önce
    bsl_targets = sorted(set(bsl_targets))

    for target in bsl_targets:
        recent_high = float(np.max(h[-lookback:]))
        if recent_high > target and last_close < target:
            return True, 'bsl_sweep', target

    return False, '', 0.0


# ═══════════════════════════════════════════════════════════════
# BÖLÜM 4: ORDER BLOCKS
# ═══════════════════════════════════════════════════════════════

def detect_order_blocks(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
    atr: float,
    lookback: int = 30,
    min_displacement_mult: float = 2.0,
) -> List[OrderBlock]:
    """
    Order Block: Büyük impuls öncesi son ters mum.

    Bullish OB: Son bearish candle → ardından 2+ bullish displacement
    Bearish OB: Son bullish candle → ardından 2+ bearish displacement

    v1.1: Tek mum minimum gövde kontrolü eklendi.

    Kalite kontrolleri:
      - Her displacement mumu > avg_body * 0.8
      - Toplam impuls > min_displacement_mult × ortalama gövde
      - FVG bıraktı mı?
      - Henüz mitigate edilmemiş
    """
    n = len(c)
    obs: List[OrderBlock] = []
    start = max(0, n - lookback)

    avg_body = float(np.mean(np.abs(c[start:] - o[start:])))
    if avg_body <= 0:
        avg_body = atr * 0.5

    # Tek mum minimum gövde (avg'nin %80'i)
    min_single_body = avg_body * 0.8

    for i in range(start + 3, n - 1):
        # ── Bullish OB: bearish candle → bullish displacement ──
        if c[i - 1] < o[i - 1]:  # i-1 bearish candle
            disp_count = 0
            disp_body_sum = 0.0
            has_fvg = False
            for k in range(i, min(i + 4, n)):
                body_k = float(c[k] - o[k])
                if body_k > min_single_body:  # ★ v1.1: tek mum minimum
                    disp_count += 1
                    disp_body_sum += body_k
                else:
                    break
                # FVG: k ve k-2 arasında boşluk
                if k >= i + 2 and float(l[k]) > float(h[k - 2]):
                    has_fvg = True

            if disp_count >= 2 and disp_body_sum > avg_body * min_displacement_mult:
                ob = OrderBlock(
                    top=float(h[i - 1]),
                    bottom=float(l[i - 1]),
                    type='bullish',
                    index=i - 1,
                    has_fvg=has_fvg,
                    displacement_strength=disp_body_sum / avg_body,
                )
                # Mitigation check
                for m in range(i + disp_count, n):
                    if float(l[m]) < ob.bottom:
                        ob.mitigated = True
                        ob.test_count += 1
                        break
                    elif float(l[m]) < ob.top:
                        ob.test_count += 1
                obs.append(ob)

        # ── Bearish OB: bullish candle → bearish displacement ──
        if c[i - 1] > o[i - 1]:
            disp_count = 0
            disp_body_sum = 0.0
            has_fvg = False
            for k in range(i, min(i + 4, n)):
                body_k = float(c[k] - o[k])
                if body_k < 0 and abs(body_k) > min_single_body:  # ★ v1.1
                    disp_count += 1
                    disp_body_sum += abs(body_k)
                else:
                    break
                if k >= i + 2 and float(h[k]) < float(l[k - 2]):
                    has_fvg = True

            if disp_count >= 2 and disp_body_sum > avg_body * min_displacement_mult:
                ob = OrderBlock(
                    top=float(h[i - 1]),
                    bottom=float(l[i - 1]),
                    type='bearish',
                    index=i - 1,
                    has_fvg=has_fvg,
                    displacement_strength=disp_body_sum / avg_body,
                )
                for m in range(i + disp_count, n):
                    if float(h[m]) > ob.top:
                        ob.mitigated = True
                        ob.test_count += 1
                        break
                    elif float(h[m]) > ob.bottom:
                        ob.test_count += 1
                obs.append(ob)

    # Freshness scoring: yeni OB'ler daha değerli
    n_total = n
    for ob in obs:
        bars_since = n_total - 1 - ob.index
        ob.freshness = max(0.0, 1.0 - (bars_since / 30.0))

    return obs


# ═══════════════════════════════════════════════════════════════
# BÖLÜM 5: FAIR VALUE GAP + CE
# ═══════════════════════════════════════════════════════════════

def detect_fvg_zones(
    h: np.ndarray, l: np.ndarray, c: np.ndarray,
    lookback: int = 20,
    atr: float = 0.0,
) -> List[FVGZone]:
    """
    FVG tespit + CE (Consequent Encroachment) hesaplama.

    Bullish FVG: candle[i].low > candle[i-2].high
    Bearish FVG: candle[i].high < candle[i-2].low

    CE = (top + bottom) / 2 → ideal entry noktası

    v1.1: quality filter 0.3→0.2, filled=close bazlı (wick değil).
    """
    n = len(h)
    zones: List[FVGZone] = []
    start = max(2, n - lookback)

    # ATR fallback
    if atr <= 0:
        atr = float(c[-1]) * 0.01 if n > 0 else 1.0

    for i in range(start, n):
        # Bullish FVG
        gap_bull = float(l[i]) - float(h[i - 2])
        if gap_bull > 0:
            top = float(l[i])
            bottom = float(h[i - 2])
            ce = (top + bottom) / 2
            fvg_quality = (top - bottom) / atr if atr > 0 else 1.0
            if fvg_quality < 0.2:  # ★ v1.1: 0.3→0.2 (daha az agresif)
                continue
            z = FVGZone(top, bottom, ce, 'bullish', i, quality=fvg_quality)
            # ★ v1.1: Fill = close bazlı (wick dokunması "test", close geçme "fill")
            for m in range(i + 1, n):
                if float(c[m]) < bottom:
                    z.filled = True
                    break
            zones.append(z)

        # Bearish FVG
        gap_bear = float(l[i - 2]) - float(h[i])
        if gap_bear > 0:
            top = float(l[i - 2])
            bottom = float(h[i])
            ce = (top + bottom) / 2
            fvg_quality = (top - bottom) / atr if atr > 0 else 1.0
            if fvg_quality < 0.2:  # ★ v1.1
                continue
            z = FVGZone(top, bottom, ce, 'bearish', i, quality=fvg_quality)
            # ★ v1.1: Fill = close bazlı
            for m in range(i + 1, n):
                if float(c[m]) > top:
                    z.filled = True
                    break
            zones.append(z)

    return zones


# ═══════════════════════════════════════════════════════════════
# BÖLÜM 6: DISPLACEMENT
# ═══════════════════════════════════════════════════════════════

def detect_displacement(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
    atr: float,
    lookback: int = 10,
    min_candles: int = 3,
    body_mult: float = 1.5,
) -> Tuple[bool, str, int]:
    """
    Displacement: 3+ ardışık güçlü mum + FVG bırakır.
    Smart Money'nin eli.

    Returns: (detected, direction, start_index)
    """
    n = len(c)
    avg_body = float(np.mean(np.abs(c - o))) if n > 0 else atr * 0.5
    threshold = avg_body * body_mult
    start = max(0, n - lookback)

    # Bullish displacement
    for i in range(start, n - min_candles + 1):
        count = 0
        for k in range(i, min(i + 6, n)):
            body = float(c[k] - o[k])
            if body > 0 and body > threshold:
                count += 1
            else:
                break
        if count >= min_candles:
            return True, 'bullish', i

    # Bearish displacement
    for i in range(start, n - min_candles + 1):
        count = 0
        for k in range(i, min(i + 6, n)):
            body = float(c[k] - o[k])
            if body < 0 and abs(body) > threshold:
                count += 1
            else:
                break
        if count >= min_candles:
            return True, 'bearish', i

    return False, '', -1


# ═══════════════════════════════════════════════════════════════
# BÖLÜM 7: BREAKER BLOCKS
# ═══════════════════════════════════════════════════════════════

def detect_breaker_blocks(
    order_blocks: List[OrderBlock],
) -> List[BreakerBlock]:
    """
    Kırılmış Order Block → Breaker Block.
    Eski destek → yeni direnç (ve tam tersi).
    """
    breakers = []
    for ob in order_blocks:
        if ob.mitigated:
            new_type = 'bearish' if ob.type == 'bullish' else 'bullish'
            breakers.append(BreakerBlock(ob.top, ob.bottom, new_type, ob.index))
    return breakers


# ═══════════════════════════════════════════════════════════════
# BÖLÜM 8: OTE ZONE (Fibonacci 61.8–78.6)
# ═══════════════════════════════════════════════════════════════

def calculate_ote_zone(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    direction: str,
) -> Optional[OTEZone]:
    """
    OTE = Optimal Trade Entry.
    Son impulsive hareketin %61.8-%78.6 retrace bölgesi.

    v1.1: Son ardışık impulse bazlı — doğru swing çifti seçimi.

    LONG (bullish): Son swing low → ondan sonraki swing high arası retrace
    SHORT (bearish): Son swing high → ondan sonraki swing low arası retrace
    """
    if not swing_highs or not swing_lows:
        return None

    if direction == 'bullish':
        # Son swing high bul
        sh = swing_highs[-1]

        # sh'den ÖNCE gelen en yakın swing low bul
        sl = None
        for sw in reversed(swing_lows):
            if sw.index < sh.index:
                sl = sw
                break

        if sl is None or sh.price <= sl.price:
            return None

        rng = sh.price - sl.price
        fib_618 = sh.price - rng * 0.618
        fib_705 = sh.price - rng * 0.705
        fib_786 = sh.price - rng * 0.786

        return OTEZone(
            top=fib_618, bottom=fib_786,
            direction='bullish',
            fib_618=fib_618, fib_705=fib_705, fib_786=fib_786,
        )

    elif direction == 'bearish':
        # Son swing low bul
        sl = swing_lows[-1]

        # sl'den ÖNCE gelen en yakın swing high bul
        sh = None
        for sw in reversed(swing_highs):
            if sw.index < sl.index:
                sh = sw
                break

        if sh is None or sh.price <= sl.price:
            return None

        rng = sh.price - sl.price
        fib_618 = sl.price + rng * 0.618
        fib_705 = sl.price + rng * 0.705
        fib_786 = sl.price + rng * 0.786

        return OTEZone(
            top=fib_786, bottom=fib_618,
            direction='bearish',
            fib_618=fib_618, fib_705=fib_705, fib_786=fib_786,
        )

    return None


# ═══════════════════════════════════════════════════════════════
# BÖLÜM 9: INDUCEMENT
# ═══════════════════════════════════════════════════════════════

def detect_inducement(
    labels: List[StructureLabel],
    c: np.ndarray,
) -> Tuple[bool, str]:
    """
    Inducement: Küçük yapı kırılması ile tuzak.
    Minor HL kırılır (perakende SHORT açar) → sonra gerçek yükseliş.

    Returns: (detected, type)  type='bullish_trap'|'bearish_trap'
    """
    if len(labels) < 4:
        return False, ''

    recent = labels[-4:]
    last_close = float(c[-1])

    # Bullish inducement: HL kırıldı (tuzak) ama HH üstüne döndü
    hl_levels = [lb for lb in recent if lb.label == 'HL']
    hh_levels = [lb for lb in recent if lb.label == 'HH']
    if hl_levels and hh_levels:
        last_hl = hl_levels[-1]
        last_hh = hh_levels[-1]
        if last_close > last_hl.price and last_close > (last_hl.price + last_hh.price) / 2:
            return True, 'bullish_trap'

    # Bearish inducement: LH kırıldı ama LL altına döndü
    lh_levels = [lb for lb in recent if lb.label == 'LH']
    ll_levels = [lb for lb in recent if lb.label == 'LL']
    if lh_levels and ll_levels:
        last_lh = lh_levels[-1]
        last_ll = ll_levels[-1]
        if last_close < last_lh.price and last_close < (last_lh.price + last_ll.price) / 2:
            return True, 'bearish_trap'

    return False, ''


# ═══════════════════════════════════════════════════════════════
# BÖLÜM 10: SESSION / KILLZONES / ASIAN RANGE
# ═══════════════════════════════════════════════════════════════

def analyze_session(
    df: pd.DataFrame,
) -> SessionInfo:
    """
    Session analizi + Asian Range + Daily/Weekly bias.

    v1.1: Default → gerçek UTC saati (yanlış killzone bonusu önlenir).
    """
    # ★ v1.1: Gerçek UTC saatini default olarak kullan
    try:
        now_utc = datetime.utcnow()
        hour = now_utc.hour
        weekday = now_utc.weekday()
    except Exception:
        hour = 3     # Dead zone (güvenli default)
        weekday = 2  # Çarşamba

    # DataFrame'den saat tespiti
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        hour = df.index[-1].hour  # UTC
        weekday = df.index[-1].weekday()
    elif 'timestamp' in df.columns:
        try:
            ts = pd.to_datetime(df['timestamp'].iloc[-1])
            hour = ts.hour
            weekday = ts.weekday()
        except Exception:
            pass

    # Killzone
    if 7 <= hour < 10:
        kz, quality = 'london', 0.9
    elif 12 <= hour < 15:
        kz, quality = 'ny_am', 1.0
    elif 15 <= hour < 17:
        kz, quality = 'ny_pm', 0.8
    elif 10 <= hour < 12:
        kz, quality = 'london_ny_transition', 0.7
    elif 0 <= hour < 7:
        kz, quality = 'asia', 0.3
    else:
        kz, quality = 'off', 0.2

    # Asian Range (UTC 00:00–07:00)
    asian_high = 0.0
    asian_low = 0.0
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 20:
        today = df.index[-1].date()
        mask = (df.index.date == today) & (df.index.hour >= 0) & (df.index.hour < 7)
        asian_bars = df[mask]
        if len(asian_bars) >= 3:
            asian_high = float(asian_bars['high'].max())
            asian_low = float(asian_bars['low'].min())

    # Daily bias
    daily_bias = 'neutral'
    if len(df) >= 5:
        open_today = float(df['open'].iloc[-1])
        close_now = float(df['close'].iloc[-1])
        if close_now > open_today * 1.001:
            daily_bias = 'bullish'
        elif close_now < open_today * 0.999:
            daily_bias = 'bearish'

    # Weekly bias
    if weekday <= 1:
        weekly_bias = 'early'    # Pzt-Sal: yön belirleme
    elif weekday <= 3:
        weekly_bias = 'mid'      # Çar-Per: devam/reversal
    else:
        weekly_bias = 'late'     # Cum+: riskli

    return SessionInfo(kz, quality, asian_high, asian_low, daily_bias, weekday, weekly_bias)


# ═══════════════════════════════════════════════════════════════
# BÖLÜM 11: POI CONFLUENCE SCORER
# ═══════════════════════════════════════════════════════════════

def score_poi_confluence(
    direction: str,
    current_price: float,
    order_blocks: List[OrderBlock],
    fvg_zones: List[FVGZone],
    breaker_blocks: List[BreakerBlock],
    ote: Optional[OTEZone],
    eq_highs: List[EqualLevel],
    eq_lows: List[EqualLevel],
    atr: float,
) -> Tuple[float, Dict]:
    """
    POI Confluence skoru: Fiyatın yakınındaki üst üste gelen yapılar.

    v1.1: OB freshness bonusu eklendi.

    1 POI  = 0.2 (zayıf)
    2 POI  = 0.5 (orta)
    3 POI  = 0.8 (güçlü)
    4+ POI = 1.0 (mükemmel)
    """
    proximity = atr * 1.5  # POI'ye yakınlık eşiği
    confluence = 0
    details = {}

    ob_type = 'bullish' if direction == 'LONG' else 'bearish'

    # 1. Order Block proximity (freshness bonusu ile)
    for ob in order_blocks:
        if ob.type == ob_type and not ob.mitigated:
            if ob.bottom - proximity <= current_price <= ob.top + proximity:
                confluence += 1
                details['ob'] = (
                    f"{ob.type} OB [{ob.bottom:.1f}-{ob.top:.1f}] "
                    f"disp={ob.displacement_strength:.1f}x fresh={ob.freshness:.0%}"
                )
                # OB+FVG overlap bonusu
                if ob.has_fvg:
                    confluence += 1
                    details['ob_fvg'] = "OB+FVG overlap"
                # ★ v1.1: Fresh OB bonusu (freshness > 0.7 → ek puan)
                if ob.freshness > 0.7:
                    confluence += 1
                    details['ob_fresh'] = f"Fresh OB ({ob.freshness:.0%})"
                break

    # 2. FVG zone proximity (unfilled)
    fvg_type = 'bullish' if direction == 'LONG' else 'bearish'
    for fvg in fvg_zones:
        if fvg.type == fvg_type and not fvg.filled:
            if fvg.bottom - proximity <= current_price <= fvg.top + proximity:
                confluence += 1
                details['fvg'] = f"{fvg.type} FVG [{fvg.bottom:.1f}-{fvg.top:.1f}] CE={fvg.ce:.1f}"
                break

    # 3. OTE zone
    if ote and ote.direction == ('bullish' if direction == 'LONG' else 'bearish'):
        if ote.bottom <= current_price <= ote.top:
            confluence += 1
            details['ote'] = f"OTE [{ote.bottom:.1f}-{ote.top:.1f}] ideal={ote.fib_705:.1f}"

    # 4. Breaker Block
    bb_type = 'bullish' if direction == 'LONG' else 'bearish'
    for bb in breaker_blocks:
        if bb.type == bb_type:
            if bb.bottom - proximity <= current_price <= bb.top + proximity:
                confluence += 1
                details['breaker'] = f"{bb.type} Breaker [{bb.bottom:.1f}-{bb.top:.1f}]"
                break

    # 5. Equal Level (likidite hedefi — TP için, confluence'a dahil değil)
    if direction == 'LONG':
        for eq in eq_highs:
            if eq.price > current_price:
                details['tp_liquidity'] = f"BSL target={eq.price:.1f} ({eq.touches} touches)"
                break
    else:
        for eq in eq_lows:
            if eq.price < current_price:
                details['tp_liquidity'] = f"SSL target={eq.price:.1f} ({eq.touches} touches)"
                break

    # Score
    if confluence >= 4:
        score = 1.0
    elif confluence == 3:
        score = 0.8
    elif confluence == 2:
        score = 0.5
    elif confluence == 1:
        score = 0.3
    else:
        score = 0.0

    details['confluence_count'] = confluence
    return score, details


# ═══════════════════════════════════════════════════════════════
# BÖLÜM 12: UNIFIED ANALYZE
# ═══════════════════════════════════════════════════════════════

def analyze(
    df: pd.DataFrame,
    direction: str = '',
    swing_left: int = 3,
    swing_right: int = 2,
) -> ICTAnalysis:
    """
    Tek çağrı ile tüm ICT/SMC analizi.

    df: OHLCV DataFrame (1h veya 4h veya 5m)
    direction: 'LONG' | 'SHORT' | '' (otomatik)

    Returns: ICTAnalysis with all detected structures
    """
    result = ICTAnalysis()

    if df is None or len(df) < 30:
        return result

    o = df['open'].astype(float).values
    h = df['high'].astype(float).values
    l = df['low'].astype(float).values
    c = df['close'].astype(float).values

    # ★ v1.1: ATR düzgün hesapla (np.roll wrap-around sorunu giderildi)
    if 'atr' in df.columns:
        atr = float(df['atr'].iloc[-1])
        if np.isnan(atr) or atr <= 0:
            atr = float(c[-1]) * 0.01
    else:
        # Düzgün TR hesaplama (ilk bar sorunu yok)
        c_prev = np.concatenate([[c[0]], c[:-1]])  # İlk bar → kendi close'u
        tr = np.maximum(h - l, np.maximum(
            np.abs(h - c_prev),
            np.abs(l - c_prev)
        ))
        atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(c[-1]) * 0.01

    # 1. Swing Points
    result.swing_highs, result.swing_lows = detect_swing_points(h, l, swing_left, swing_right)

    # 2. Market Structure (★ v1.1: h/l dizileri geçiriliyor)
    result.market_structure, result.structure_labels, result.structure_breaks = \
        analyze_market_structure(result.swing_highs, result.swing_lows, c, h, l)

    for sb in reversed(result.structure_breaks):
        if sb.type == 'BOS' and result.last_bos is None:
            result.last_bos = sb
        if sb.type == 'CHoCH' and result.last_choch is None:
            result.last_choch = sb

    # 3. Equal Levels
    result.equal_highs, result.equal_lows = detect_equal_levels(
        result.swing_highs, result.swing_lows, atr)

    # 4. Liquidity Sweep
    result.sweep_detected, result.sweep_type, result.sweep_level = check_sweep(
        result.equal_highs, result.equal_lows,
        result.swing_highs, result.swing_lows,
        h, l, c)

    # 5. Order Blocks
    result.order_blocks = detect_order_blocks(o, h, l, c, atr)

    # 6. FVG (with ATR-based quality filter)
    result.fvg_zones = detect_fvg_zones(h, l, c, atr=atr)

    # 7. Displacement
    result.displacement, result.displacement_direction, _ = detect_displacement(o, h, l, c, atr)

    # 8. Breaker Blocks
    result.breaker_blocks = detect_breaker_blocks(result.order_blocks)

    # 9. OTE
    ote_dir = ''
    if direction == 'LONG':
        ote_dir = 'bullish'
    elif direction == 'SHORT':
        ote_dir = 'bearish'
    elif result.market_structure == 'bullish':
        ote_dir = 'bullish'
    elif result.market_structure == 'bearish':
        ote_dir = 'bearish'
    if ote_dir:
        result.ote = calculate_ote_zone(result.swing_highs, result.swing_lows, ote_dir)

    # 10. Session
    result.session = analyze_session(df)

    # 11. Inducement (info only — hafif tutmak için ICTAnalysis'e eklenmedi)

    # 12. POI Confluence
    if direction:
        cp = float(c[-1])
        result.poi_confluence, result.poi_details = score_poi_confluence(
            direction, cp,
            result.order_blocks, result.fvg_zones,
            result.breaker_blocks, result.ote,
            result.equal_highs, result.equal_lows, atr,
        )

    return result


def get_liquidity_tp(
    direction: str,
    current_price: float,
    eq_highs: List[EqualLevel],
    eq_lows: List[EqualLevel],
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
) -> float:
    """
    Likidite bazlı TP hedefi.
    LONG → BSL (üstteki equal highs / swing highs)
    SHORT → SSL (alttaki equal lows / swing lows)
    """
    if direction == 'LONG':
        targets = [eq.price for eq in eq_highs if eq.price > current_price]
        targets += [sh.price for sh in swing_highs if sh.price > current_price * 1.005]
        targets.sort()
        return targets[0] if targets else current_price * 1.03

    elif direction == 'SHORT':
        targets = [eq.price for eq in eq_lows if eq.price < current_price]
        targets += [sl.price for sl in swing_lows if sl.price < current_price * 0.995]
        targets.sort(reverse=True)
        return targets[0] if targets else current_price * 0.97

    return 0.0


def get_sweep_sl(
    direction: str,
    sweep_level: float,
    current_price: float,
    atr: float,
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    sl_buffer_atr: float = 1.0,
) -> float:
    """
    Sweep bazlı SL: Sweep seviyesinin ötesi.
    LONG  → sweep low altı + buffer
    SHORT → sweep high üstü + buffer
    """
    buffer = atr * sl_buffer_atr

    if direction == 'LONG':
        if sweep_level > 0:
            return sweep_level - buffer
        # Fallback: son swing low
        if swing_lows:
            return swing_lows[-1].price - buffer
        return current_price * 0.98

    elif direction == 'SHORT':
        if sweep_level > 0:
            return sweep_level + buffer
        if swing_highs:
            return swing_highs[-1].price + buffer
        return current_price * 1.02

    return 0.0