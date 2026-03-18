"""
═══════════════════════════════════════════════════════════════
 5-LAYER ENTRY GATE v2.1  —  PURE ICT/SMC
 "Bir trade'e girmek için 5 katmanın HEPSİ uyumlu olmalı"
═══════════════════════════════════════════════════════════════

 Katman 1: YAPI    — HTF Market Structure (BOS/CHoCH) + Trend
 Katman 2: POI     — Likidite Haritası + Premium/Discount + OB/FVG/OTE Confluence
 Katman 3: TETİK   — Sweep → CHoCH → FVG/OB retest (1h yapı, 5m entry)
 Katman 4: TEYİT   — Displacement + Volume + Killzone + MTF
 Katman 5: RİSK    — Sweep bazlı SL + Likidite bazlı TP + min RR

 1 tanesi bile FAIL → HOLD

 v2.1 Düzeltmeler:
   - Exception bypass → FAIL (güvenlik)
   - 5m retest fiyat yakınlığı doğrulaması
   - Volume spike son 5 barda arama (sweep barı)
   - OTE zone → POI skorlamasına dahil
   - ATR gerçek hesaplama (df_1h parametre)
   - Summary'de reject reason görünür
   - Ağırlıklı toplam skor
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

from ai import ict_core


# ── Katman ağırlıkları (toplam 1.0) ──
LAYER_WEIGHTS = {
    'YAPI':  0.25,
    'POI':   0.15,
    'TETİK': 0.25,
    'TEYİT': 0.10,
    'RİSK':  0.25,
}


@dataclass
class LayerResult:
    """Tek katman sonucu."""
    name: str
    passed: bool
    reason: str
    score: float = 0.0
    details: Dict = field(default_factory=dict)


@dataclass
class GateResult:
    """5 Katman toplam sonucu."""
    passed: bool
    layers: List[LayerResult] = field(default_factory=list)
    reject_layer: str = ""
    reject_reason: str = ""
    total_score: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    rr_ratio: float = 0.0

    def summary(self) -> str:
        marks = []
        for lr in self.layers:
            marks.append(f"{'✅' if lr.passed else '❌'}{lr.name}({lr.score:.0%})")
        status = "PASS" if self.passed else f"FAIL@{self.reject_layer}"
        reason = f" [{self.reject_reason}]" if self.reject_reason and not self.passed else ""
        return f"[GATE {status}{reason}] {' | '.join(marks)} RR={self.rr_ratio:.1f}"


# ═══════════════════════════════════════════════════════════════
# ANA GATE FONKSİYONU
# ═══════════════════════════════════════════════════════════════

def five_layer_gate(
    direction: str,
    df_1h: pd.DataFrame,
    df_5m: Optional[pd.DataFrame],
    df_4h: Optional[pd.DataFrame],
    entry_price: float,
    sl_price: float,
    tp_price: float,
    meta_confidence: float = 0.5,
    regime: str = "trending",
    strategy: str = "unknown",
) -> GateResult:
    """Pure ICT/SMC 5-Layer Entry Gate."""
    result = GateResult(passed=False)

    # ── ICT analiz (her TF için bir kez) ──
    ict_4h = None
    if df_4h is not None and len(df_4h) >= 30:
        ict_4h = ict_core.analyze(df_4h, direction, swing_left=3, swing_right=2)
    ict_1h = ict_core.analyze(df_1h, direction, swing_left=3, swing_right=2)
    ict_5m = None
    if df_5m is not None and len(df_5m) >= 20:
        ict_5m = ict_core.analyze(df_5m, direction, swing_left=2, swing_right=1)

    ict_htf = ict_4h if ict_4h is not None else ict_1h

    # ── Gerçek ATR hesapla (Layer 5 için) ──
    atr_1h = _calc_atr(df_1h, entry_price)

    layers_fn = [
        lambda: _layer_1_structure(direction, ict_htf, df_4h if df_4h is not None else df_1h),
        lambda: _layer_2_poi(direction, ict_1h, ict_htf, entry_price, df_1h),
        lambda: _layer_3_trigger(direction, ict_1h, ict_5m, df_5m, entry_price),
        lambda: _layer_4_confirmation(direction, ict_1h, ict_5m, ict_htf, df_1h, df_5m),
        lambda: _layer_5_risk(direction, ict_1h, ict_5m, ict_htf,
                              entry_price, sl_price, tp_price, meta_confidence, atr_1h),
    ]

    for fn in layers_fn:
        lr = fn()
        result.layers.append(lr)
        if not lr.passed:
            result.reject_layer = lr.name
            result.reject_reason = lr.reason
            result.total_score = _weighted_score(result.layers)
            return result

    # ── HEPSİ GEÇTİ ──
    result.passed = True
    result.total_score = _weighted_score(result.layers)
    l5 = result.layers[-1]
    result.sl_price = l5.details.get('final_sl', sl_price)
    result.tp_price = l5.details.get('final_tp', tp_price)
    result.rr_ratio = l5.details.get('rr_ratio', 0.0)
    return result


def _weighted_score(layers: List[LayerResult]) -> float:
    """Ağırlıklı toplam skor hesapla."""
    total = 0.0
    for lr in layers:
        w = LAYER_WEIGHTS.get(lr.name, 0.20)
        total += lr.score * w
    return total


def _calc_atr(df: pd.DataFrame, fallback_price: float, period: int = 14) -> float:
    """Gerçek ATR hesapla, veri yoksa fallback."""
    try:
        if df is not None and len(df) >= period + 1:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            atr = float(tr.rolling(period).mean().iloc[-1])
            if not np.isnan(atr) and atr > 0:
                return atr
    except Exception:
        pass
    return fallback_price * 0.01  # %1 fallback


# ═══════════════════════════════════════════════════════════════
# KATMAN 1: YAPI — HTF Market Structure + Trend
# ═══════════════════════════════════════════════════════════════

def _layer_1_structure(
    direction: str,
    ict_htf: ict_core.ICTAnalysis,
    df_htf: pd.DataFrame,
) -> LayerResult:
    """
    HTF (4h/1h) market structure — BOS/CHoCH bazlı.
    LONG  → Bullish structure (HH+HL) VEYA bullish CHoCH
    SHORT → Bearish structure (LL+LH) VEYA bearish CHoCH
    EMA200 bonus skor verir.
    """
    name = "YAPI"
    try:
        ms = ict_htf.market_structure
        labels = [lb.label for lb in ict_htf.structure_labels[-6:]]

        has_bull_bos = ict_htf.last_bos and ict_htf.last_bos.direction == 'bullish'
        has_bear_bos = ict_htf.last_bos and ict_htf.last_bos.direction == 'bearish'
        has_bull_choch = ict_htf.last_choch and ict_htf.last_choch.direction == 'bullish'
        has_bear_choch = ict_htf.last_choch and ict_htf.last_choch.direction == 'bearish'

        cp = float(df_htf['close'].iloc[-1])
        ema200 = float(df_htf['close'].ewm(span=200, adjust=False).mean().iloc[-1]) \
            if len(df_htf) >= 200 else cp
        above_ema = cp > ema200
        ema_dist = (cp - ema200) / ema200 * 100 if ema200 > 0 else 0

        details = {
            'ms': ms, 'labels': labels,
            'bull_bos': has_bull_bos, 'bear_bos': has_bear_bos,
            'bull_choch': has_bull_choch, 'bear_choch': has_bear_choch,
            'above_ema200': above_ema, 'ema200_dist': round(ema_dist, 2),
        }

        if direction == 'LONG':
            if ms == 'bullish' and has_bull_bos:
                s = min(1.0, 0.9 + (0.1 if above_ema else 0))
                return LayerResult(name, True, f"Bull BOS ✅ EMA{ema_dist:+.1f}%", s, details)
            if has_bull_choch:
                s = min(1.0, 0.8 + (0.1 if above_ema else 0))
                return LayerResult(name, True, f"Bull CHoCH ✅ EMA{ema_dist:+.1f}%", s, details)
            if ms == 'bullish':
                s = 0.7 + (0.1 if above_ema else 0)
                return LayerResult(name, True, f"Bull yapı (HH+HL) EMA{ema_dist:+.1f}%", s, details)
            if ms == 'ranging' and above_ema:
                # M3: Ranging BOS/CHoCH yok — sadece CHoCH varsa geçir
                if has_bull_choch:
                    return LayerResult(name, True, "Ranging+CHoCH+EMA↑ LONG OK", 0.45, details)
                return LayerResult(name, False, f"LONG: ranging+EMA↑ ama BOS/CHoCH yok", 0.1, details)
            return LayerResult(name, False, f"LONG: bull yapı yok (ms={ms})", 0.0, details)

        elif direction == 'SHORT':
            if ms == 'bearish' and has_bear_bos:
                s = min(1.0, 0.9 + (0.1 if not above_ema else 0))
                return LayerResult(name, True, f"Bear BOS ✅ EMA{ema_dist:+.1f}%", s, details)
            if has_bear_choch:
                s = min(1.0, 0.8 + (0.1 if not above_ema else 0))
                return LayerResult(name, True, f"Bear CHoCH ✅ EMA{ema_dist:+.1f}%", s, details)
            if ms == 'bearish':
                s = 0.7 + (0.1 if not above_ema else 0)
                return LayerResult(name, True, f"Bear yapı (LL+LH) EMA{ema_dist:+.1f}%", s, details)
            if ms == 'ranging' and not above_ema:
                # M3: Ranging BOS/CHoCH yok — sadece CHoCH varsa geçir
                if has_bear_choch:
                    return LayerResult(name, True, "Ranging+CHoCH+EMA↓ SHORT OK", 0.45, details)
                return LayerResult(name, False, f"SHORT: ranging+EMA↓ ama BOS/CHoCH yok", 0.1, details)
            return LayerResult(name, False, f"SHORT: bear yapı yok (ms={ms})", 0.0, details)

        return LayerResult(name, False, "Bilinmeyen yön", 0.0, details)

    except Exception as e:
        # ★ FIX: Exception → FAIL (bypass değil)
        return LayerResult(name, False, f"Yapı analiz hatası ({e})", 0.0)


# ═══════════════════════════════════════════════════════════════
# KATMAN 2: POI — Likidite Haritası + Confluence
# ═══════════════════════════════════════════════════════════════

def _layer_2_poi(
    direction: str,
    ict_1h: ict_core.ICTAnalysis,
    ict_htf: ict_core.ICTAnalysis,
    entry_price: float,
    df_1h: Optional[pd.DataFrame] = None,
) -> LayerResult:
    """
    POI (Point of Interest) Confluence + Premium/Discount + OTE Zone.
    Equilibrium'da trade açmak yasak.
    """
    name = "POI"
    try:
        cp = entry_price

        # v2.1: Son 30 bar range kullan (swing point'ler trending
        # market'te range'ı bozar — her yer premium görünür)
        if df_1h is not None and len(df_1h) >= 10:
            lookback = min(30, len(df_1h))
            recent = df_1h.iloc[-lookback:]
            swing_hi = float(recent['high'].max())
            swing_lo = float(recent['low'].min())
        else:
            sh_p = [sp.price for sp in ict_1h.swing_highs[-5:]] if ict_1h.swing_highs else []
            sl_p = [sp.price for sp in ict_1h.swing_lows[-5:]] if ict_1h.swing_lows else []
            swing_hi = max(sh_p) if sh_p else cp * 1.03
            swing_lo = min(sl_p) if sl_p else cp * 0.97
        rng = swing_hi - swing_lo
        pos = (cp - swing_lo) / rng if rng > 0 else 0.5

        conf_score = ict_1h.poi_confluence
        conf_det = dict(ict_1h.poi_details)
        if ict_htf is not ict_1h and ict_htf.poi_confluence > 0:
            conf_score = min(1.0, conf_score + ict_htf.poi_confluence * 0.3)
            conf_det['htf_conf'] = round(ict_htf.poi_confluence, 2)

        cc_raw = conf_det.get('confluence_count', 0)
        cc = cc_raw  # OTE eklenmeden önceki değer
        active_obs = len([ob for ob in ict_1h.order_blocks
                          if not ob.mitigated and
                          ob.type == ('bullish' if direction == 'LONG' else 'bearish')])
        active_fvgs = len([f for f in ict_1h.fvg_zones
                           if not f.filled and
                           f.type == ('bullish' if direction == 'LONG' else 'bearish')])

        # ★ FIX: OTE zone → confluence bonusu
        ote_in_zone = False
        if ict_1h.ote is not None:
            ote = ict_1h.ote
            if ote.bottom <= cp <= ote.top:
                ote_in_zone = True
                cc += 1  # OTE = ek confluence point
                conf_score = min(1.0, conf_score + 0.2)

        details = {
            'pos': round(pos, 3), 'swing_hi': round(swing_hi, 2),
            'swing_lo': round(swing_lo, 2), 'poi_conf': round(conf_score, 2),
            'cc': cc, 'cc_raw': cc_raw, 'obs': active_obs, 'fvgs': active_fvgs,
            'ote': ict_1h.ote is not None, 'ote_in_zone': ote_in_zone,
            'poi_det': conf_det,
        }

        # Zone kontrolü
        if direction == 'LONG' and pos > 0.70 and cc < 1:
            return LayerResult(name, False,
                f"LONG Premium/Eq yasak pos={pos:.0%} cc={cc}", 0.0, details)
        if direction == 'SHORT' and pos < 0.30 and cc < 1:
            return LayerResult(name, False,
                f"SHORT Discount/Eq yasak pos={pos:.0%} cc={cc}", 0.0, details)

        # Confluence skorlaması
        if cc >= 3:
            return LayerResult(name, True,
                f"Mükemmel POI ✅ cc={cc} pos={pos:.0%}", 1.0, details)
        if cc >= 2:
            return LayerResult(name, True,
                f"Güçlü POI cc={cc} pos={pos:.0%}", 0.7, details)
        if cc >= 1:
            zone_ok = (direction == 'LONG' and pos < 0.45) or \
                      (direction == 'SHORT' and pos > 0.55)
            if zone_ok:
                return LayerResult(name, True,
                    f"Tek POI+zone OK pos={pos:.0%}", 0.5, details)
            return LayerResult(name, False,
                f"Tek POI+zone kötü pos={pos:.0%}", 0.1, details)

        # POI yok → deep zone ise geçir
        if direction == 'LONG' and pos <= 0.30:
            return LayerResult(name, True,
                f"Deep discount {pos:.0%} (POI yok)", 0.4, details)
        if direction == 'SHORT' and pos >= 0.70:
            return LayerResult(name, True,
                f"Deep premium {pos:.0%} (POI yok)", 0.4, details)

        return LayerResult(name, False,
            f"POI yok+zone zayıf pos={pos:.0%}", 0.0, details)

    except Exception as e:
        # ★ FIX: Exception → FAIL
        return LayerResult(name, False, f"POI analiz hatası ({e})", 0.0)


# ═══════════════════════════════════════════════════════════════
# KATMAN 3: TETİK — Sweep → CHoCH → FVG/OB Retest
# ═══════════════════════════════════════════════════════════════

def _layer_3_trigger(
    direction: str,
    ict_1h: ict_core.ICTAnalysis,
    ict_5m: Optional[ict_core.ICTAnalysis],
    df_5m: Optional[pd.DataFrame],
    entry_price: float,
) -> LayerResult:
    """
    ICT Sequential Trigger:
      1h: ①Sweep ②BOS/CHoCH ③FVG ④OB
      5m: ⑤FVG CE retest (yakınlık doğrulamalı) ⑥OB retest (yakınlık)
          ⑦CHoCH ⑧Displacement ⑨micro sweep
    Gerekli: 1h min 1 yapı + 5m min 1 entry
    """
    name = "TETİK"
    structs = 0  # try dışında tanımla (except'te kullanılabilir)
    try:
        s_reasons = []
        exp_dir = 'bullish' if direction == 'LONG' else 'bearish'
        exp_sweep = 'ssl_sweep' if direction == 'LONG' else 'bsl_sweep'

        # 1h Sweep
        if ict_1h.sweep_detected and ict_1h.sweep_type == exp_sweep:
            structs += 1
            s_reasons.append(f"1h:{ict_1h.sweep_type}")

        # 1h CHoCH / BOS
        if ict_1h.last_choch and ict_1h.last_choch.direction == exp_dir:
            structs += 1
            s_reasons.append("1h:CHoCH")
        elif ict_1h.last_bos and ict_1h.last_bos.direction == exp_dir:
            structs += 1
            s_reasons.append("1h:BOS")

        # 1h FVG (unfilled)
        fvg_t = 'bullish' if direction == 'LONG' else 'bearish'
        a_fvgs = [f for f in ict_1h.fvg_zones if f.type == fvg_t and not f.filled]
        if a_fvgs:
            structs += 1
            s_reasons.append(f"1h:FVG({len(a_fvgs)})")

        # 1h OB (unmitigated)
        ob_t = 'bullish' if direction == 'LONG' else 'bearish'
        a_obs = [ob for ob in ict_1h.order_blocks
                 if ob.type == ob_t and not ob.mitigated]
        if a_obs:
            structs += 1
            s_reasons.append(f"1h:OB({len(a_obs)})")

        if structs == 0:
            return LayerResult(name, False, "1h yapı=0", 0.0, {'structs': 0})

        # ── 5m Entry (fiyat yakınlığı doğrulamalı) ──
        entries = 0
        e_reasons = []

        if ict_5m is None:
            if structs >= 2:
                return LayerResult(name, True,
                    f"1h güçlü(5m yok): {'+'.join(s_reasons)}", 0.4,
                    {'structs': structs, 'entries': 0})
            return LayerResult(name, False,
                f"5m yok+1h zayıf({structs})", 0.1, {'structs': structs})

        # ★ FIX: 5m current price + ATR yakınlık eşiği
        cp_5m = entry_price
        if df_5m is not None and len(df_5m) > 0:
            cp_5m = float(df_5m['close'].iloc[-1])

        # 5m ATR yakınlık eşiği
        if df_5m is not None and len(df_5m) >= 14:
            atr_5m = float((df_5m['high'] - df_5m['low']).astype(float).rolling(14).mean().iloc[-1])
            if np.isnan(atr_5m) or atr_5m <= 0:
                atr_5m = cp_5m * 0.003
        else:
            atr_5m = cp_5m * 0.003

        proximity = atr_5m * 1.5

        # 5m FVG CE retest — ★ FIX: fiyat CE'ye YAKIN MI?
        for fvg in a_fvgs:
            if abs(cp_5m - fvg.ce) <= proximity:
                entries += 1
                e_reasons.append(f"FVG_CE@{fvg.ce:.0f}(d={abs(cp_5m-fvg.ce):.0f})")
                break

        # 5m OB retest — ★ FIX: fiyat OB zone İÇİNDE veya YAKIN MI?
        for ob in a_obs:
            in_ob = ob.bottom - proximity <= cp_5m <= ob.top + proximity
            if in_ob:
                entries += 1
                e_reasons.append(f"OB@{ob.bottom:.0f}-{ob.top:.0f}")
                break

        # 5m CHoCH
        if ict_5m.last_choch and ict_5m.last_choch.direction == exp_dir:
            entries += 1
            e_reasons.append("5m:CHoCH")

        # 5m Displacement
        if ict_5m.displacement and ict_5m.displacement_direction == exp_dir:
            entries += 1
            e_reasons.append("5m:Disp")

        # 5m sweep (mikro)
        micro_sweep = 'ssl_sweep' if direction == 'LONG' else 'bsl_sweep'
        if ict_5m.sweep_detected and ict_5m.sweep_type == micro_sweep:
            entries += 1
            e_reasons.append("5m:Sweep")

        det = {
            'structs': structs, 's_reasons': s_reasons,
            'entries': entries, 'e_reasons': e_reasons,
        }
        tag = f"[1h:{'+'.join(s_reasons)}] [5m:{'+'.join(e_reasons) if e_reasons else 'NONE'}]"

        if structs >= 1 and entries >= 1:
            sc = min(1.0, (structs + entries) * 0.15 + 0.2)
            return LayerResult(name, True, tag, sc, det)
        if structs >= 2:
            return LayerResult(name, True, f"1h güçlü({structs}): {tag}", 0.4, det)
        return LayerResult(name, False, f"5m entry yok: {tag}", 0.1, det)

    except (AttributeError, TypeError) as e:
        # ★ FIX: Bilinen hatalar — 5m verisi yoksa/format sorunuysa
        if structs >= 2:
            return LayerResult(name, True,
                f"5m hatası ama 1h güçlü ({e})", 0.3,
                {'structs': structs, 'entries': 0})
        return LayerResult(name, False, f"Trigger veri hatası ({e})", 0.0)
    except Exception as e:
        # ★ FIX: Beklenmeyen hatalar → KESİNLİKLE FAIL
        return LayerResult(name, False, f"Trigger kritik hata ({e})", 0.0)


# ═══════════════════════════════════════════════════════════════
# KATMAN 4: TEYİT — Displacement+Volume+Killzone+MTF
# ═══════════════════════════════════════════════════════════════

def _layer_4_confirmation(
    direction: str,
    ict_1h: ict_core.ICTAnalysis,
    ict_5m: Optional[ict_core.ICTAnalysis],
    ict_htf: ict_core.ICTAnalysis,
    df_1h: pd.DataFrame,
    df_5m: Optional[pd.DataFrame],
) -> LayerResult:
    """
    6 alt kontrol — 3 çekirdek + 3 bonus:
      Çekirdek: ①Displacement ②Volume(>1.5x son 5 bar) ③MTF
      Bonus:    ④Killzone ⑤WeekBias ⑥AsianSweep
    Min 2/6 gerekli (en az 1 çekirdek). Skor çekirdek ağırlıklı.
    """
    name = "TEYİT"
    try:
        confs = 0
        core_confs = 0
        reasons = []
        total = 6
        disp_dir = 'bullish' if direction == 'LONG' else 'bearish'

        # ── ÇEKIRDEK ──
        # ① Displacement (Smart Money eli)
        has_disp = (ict_1h.displacement and ict_1h.displacement_direction == disp_dir) or \
                   (ict_5m is not None and ict_5m.displacement and
                    ict_5m.displacement_direction == disp_dir)
        if has_disp:
            confs += 1; core_confs += 1; reasons.append("Disp")

        # ② Volume spike — ★ FIX: Son 5 barda EN YÜKSEK RVOL
        vol_ok = False
        vol_src = '5m'
        df_v = df_5m if (df_5m is not None and 'volume' in df_5m.columns and len(df_5m) >= 20) else df_1h
        if df_v is df_1h:
            vol_src = '1h'  # M5: farklı SMA penceresi (5m=100dk vs 1h=20saat)
        if 'volume' in df_v.columns and len(df_v) >= 20:
            vol = df_v['volume'].astype(float)
            vma = vol.rolling(20, min_periods=5).mean()
            max_rvol = 0
            for lookback in range(1, 6):  # Son 5 bar
                idx = len(vol) - lookback
                if idx >= 0:
                    v_avg = float(vma.iloc[idx])
                    if not np.isnan(v_avg) and v_avg > 0:
                        rvol = float(vol.iloc[idx]) / v_avg
                        max_rvol = max(max_rvol, rvol)
            if max_rvol > 1.5:
                vol_ok = True; confs += 1; core_confs += 1
                reasons.append(f"Vol({max_rvol:.1f}x)")

        # ③ MTF uyumu
        mtf_ok = False
        exp_ms = 'bullish' if direction == 'LONG' else 'bearish'
        htf_aligned = ict_htf.market_structure == exp_ms
        h1_aligned = ict_1h.market_structure == exp_ms
        aligned = (1 if htf_aligned else 0) + (1 if h1_aligned else 0)

        if aligned >= 1:
            mtf_ok = True; confs += 1; core_confs += 1
            reasons.append(f"MTF({aligned}/2)")

        # ── BONUS ──
        # ④ Killzone
        kz_ok = False
        sess = ict_1h.session
        if sess and sess.killzone in ('london', 'ny_am', 'ny_pm', 'london_ny_transition'):
            kz_ok = True; confs += 1; reasons.append(f"KZ({sess.killzone})")

        # ⑤ Weekly bias
        week_ok = False
        if sess and sess.weekly_bias in ('early', 'mid'):
            week_ok = True; confs += 1; reasons.append(f"Week({sess.weekly_bias})")

        # ⑥ Asian sweep — lookback 8 bar (full Asian session)
        asian_ok = False
        if sess and sess.asian_high > 0 and sess.asian_low > 0:
            cp = float(df_1h['close'].iloc[-1])
            lookback = min(8, len(df_1h))
            if direction == 'LONG':
                rl = float(df_1h['low'].iloc[-lookback:].min()) if lookback > 0 else cp
                if rl < sess.asian_low and cp > sess.asian_low:
                    asian_ok = True; confs += 1; reasons.append("AsianSweep")
            else:
                rh = float(df_1h['high'].iloc[-lookback:].max()) if lookback > 0 else cp
                if rh > sess.asian_high and cp < sess.asian_high:
                    asian_ok = True; confs += 1; reasons.append("AsianSweep")

        det = {
            'disp': has_disp, 'vol': vol_ok, 'vol_src': vol_src,
            'mtf': mtf_ok,
            'kz': kz_ok, 'week': week_ok, 'asian': asian_ok,
            'confs': confs, 'core': core_confs,
        }

        # Skor: çekirdek ağırlıklı
        sc = (core_confs * 0.25 + (confs - core_confs) * 0.08)
        sc = min(1.0, sc)

        # Karar: min 2 teyit VE en az 1 çekirdek
        if confs >= 2 and core_confs >= 1:
            return LayerResult(name, True,
                f"{confs}/{total} (core:{core_confs}): {' '.join(reasons)}", sc, det)

        # Sadece bonus varsa (session iyi ama core yok)
        if confs >= 3 and core_confs == 0:
            return LayerResult(name, True,
                f"Bonus-only {confs}/{total}: {' '.join(reasons)}", 0.3, det)

        missing = []
        if not has_disp: missing.append("Disp")
        if not vol_ok: missing.append("Vol")
        if not mtf_ok: missing.append("MTF")
        if not kz_ok: missing.append("KZ")
        if not week_ok: missing.append("Week")
        if not asian_ok: missing.append("Asian")
        return LayerResult(name, False,
            f"Yetersiz {confs}/{total} core={core_confs} (eksik:{','.join(missing)})", sc, det)

    except Exception as e:
        # ★ FIX: Exception → FAIL
        return LayerResult(name, False, f"Confirmation hatası ({e})", 0.0)


# ═══════════════════════════════════════════════════════════════
# KATMAN 5: RİSK — Sweep SL + Likidite TP + min RR
# ═══════════════════════════════════════════════════════════════

def _layer_5_risk(
    direction: str,
    ict_1h: ict_core.ICTAnalysis,
    ict_5m: Optional[ict_core.ICTAnalysis],
    ict_htf: ict_core.ICTAnalysis,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    meta_confidence: float,
    atr: float = 0.0,
) -> LayerResult:
    """
    ICT Risk:
      1. SL = Sweep seviyesi altı / son swing + buffer
      2. TP = Karşı taraf likiditesi (BSL/SSL)
      3. Min R:R = 2.0
      4. Max SL = %2.5
      5. Meta confidence ≥ 0.40
    """
    name = "RİSK"
    try:
        # ★ FIX: Gerçek ATR (dışarıdan geliyor), fallback
        if atr <= 0:
            atr = entry_price * 0.01

        # ── 1. Sweep bazlı SL ──
        sweep_sl = ict_core.get_sweep_sl(
            direction, ict_1h.sweep_level, entry_price, atr,
            ict_1h.swing_highs, ict_1h.swing_lows)

        if direction == 'LONG':
            final_sl = min(sl_price, sweep_sl) if sweep_sl > 0 else sl_price
            if final_sl >= entry_price:
                final_sl = entry_price * 0.985
        else:
            final_sl = max(sl_price, sweep_sl) if sweep_sl > 0 else sl_price
            if final_sl <= entry_price:
                final_sl = entry_price * 1.015

        # Max SL %2.5
        sl_pct = abs(entry_price - final_sl) / entry_price * 100
        if sl_pct > 2.5:
            final_sl = entry_price * (0.975 if direction == 'LONG' else 1.025)
            sl_pct = 2.5

        # ── 2. Likidite bazlı TP ──
        liq_tp = ict_core.get_liquidity_tp(
            direction, entry_price,
            ict_1h.equal_highs, ict_1h.equal_lows,
            ict_1h.swing_highs, ict_1h.swing_lows)

        if direction == 'LONG':
            final_tp = max(tp_price, liq_tp) if liq_tp > entry_price else tp_price
        else:
            final_tp = min(tp_price, liq_tp) if liq_tp < entry_price else tp_price

        # ── 3. R:R ──
        risk = abs(entry_price - final_sl)
        reward = abs(final_tp - entry_price)
        if risk <= 0:
            return LayerResult(name, False, "SL mesafesi=0", 0.0)
        rr = reward / risk

        # TP auto-adjust for min RR
        tp_adjusted = False
        if rr < 2.0:
            final_tp = entry_price + (risk * 2.0) if direction == 'LONG' \
                else entry_price - (risk * 2.0)
            rr = 2.0
            tp_adjusted = True

        # M4: TP mesafe limiti — 5 ATR'den uzak TP gerçekçi değil
        tp_dist = abs(final_tp - entry_price)
        max_tp_dist = atr * 5.0
        if tp_dist > max_tp_dist and max_tp_dist > 0:
            final_tp = entry_price + max_tp_dist if direction == 'LONG' \
                else entry_price - max_tp_dist
            reward = max_tp_dist
            rr = reward / risk if risk > 0 else 0
            tp_adjusted = True

        det = {
            'final_sl': round(final_sl, 4), 'final_tp': round(final_tp, 4),
            'sl_pct': round(sl_pct, 2), 'rr_ratio': round(rr, 2),
            'meta': round(meta_confidence, 3),
            'sweep_sl': round(sweep_sl, 4), 'liq_tp': round(liq_tp, 4),
            'tp_adjusted': tp_adjusted, 'atr': round(atr, 4),
        }

        # ── 4. Meta confidence ──
        if meta_confidence < 0.40:
            return LayerResult(name, False,
                f"Meta düşük {meta_confidence:.2f}<0.40", 0.1, det)

        # Skor
        if rr >= 3.0:
            sc = 1.0
        elif rr >= 2.0:
            sc = 0.8
        else:
            sc = 0.5
        if sl_pct <= 1.5:
            sc = min(sc + 0.1, 1.0)
        # ★ FIX: TP adjusted → skor cezası
        if tp_adjusted:
            sc *= 0.8

        return LayerResult(name, True,
            f"RR={rr:.1f} SL={sl_pct:.1f}% meta={meta_confidence:.2f} liqTP={liq_tp:.0f}",
            sc, det)

    except Exception as e:
        # ★ FIX: Exception → FAIL
        return LayerResult(name, False, f"Risk hesaplama hatası ({e})", 0.0)