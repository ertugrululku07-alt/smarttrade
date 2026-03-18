"""
ICT/SMC Feature Extractor — ict_core.analyze() → ML-ready numeric features

Meta-model'e verilecek ICT-specific feature'lar:
  - Yapı (market structure, BOS, CHoCH)
  - Likidite (EQH/EQL, sweep)
  - POI (OB, FVG, confluence, OTE)
  - Displacement
  - Session/Killzone
  - Pozisyon (premium/discount)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from ai import ict_core


# ── Feature isimleri ──
ICT_FEATURES = [
    # Yapı
    'ict_ms_bullish',
    'ict_ms_bearish',
    'ict_has_bull_bos',
    'ict_has_bear_bos',
    'ict_has_bull_choch',
    'ict_has_bear_choch',
    'ict_structure_score',       # -1 (bear) to +1 (bull)
    # Likidite
    'ict_eq_highs_count',
    'ict_eq_lows_count',
    'ict_sweep_ssl',
    'ict_sweep_bsl',
    # POI
    'ict_bull_ob_count',
    'ict_bear_ob_count',
    'ict_ob_best_strength',
    'ict_ob_has_fvg',
    'ict_bull_fvg_count',
    'ict_bear_fvg_count',
    'ict_fvg_ce_dist_atr',       # distance to nearest FVG CE / ATR
    'ict_poi_confluence',
    # Displacement
    'ict_displacement_bull',
    'ict_displacement_bear',
    # OTE
    'ict_in_ote',
    # Pozisyon
    'ict_premium_discount',      # 0=discount, 0.5=eq, 1=premium
    # Session
    'ict_session_quality',
    'ict_weekly_score',          # 0=late(bad), 0.5=early, 1=mid(best)
    # Breaker
    'ict_breaker_count',
]

# Toplam: 27 feature


def extract_ict_features(
    analysis: ict_core.ICTAnalysis,
    current_price: float,
    atr: float,
    direction: str = '',
) -> Dict[str, float]:
    """
    ICTAnalysis → flat numeric feature dict.

    Args:
        analysis: ict_core.analyze() sonucu
        current_price: mevcut fiyat
        atr: ATR değeri
        direction: 'LONG'|'SHORT'|'' (confluence skoru için)

    Returns:
        Dict[str, float] — tüm ICT_FEATURES key'lerini içerir
    """
    f: Dict[str, float] = {}

    # ── Yapı ──
    ms = analysis.market_structure
    f['ict_ms_bullish'] = 1.0 if ms == 'bullish' else 0.0
    f['ict_ms_bearish'] = 1.0 if ms == 'bearish' else 0.0

    f['ict_has_bull_bos'] = 1.0 if (analysis.last_bos and analysis.last_bos.direction == 'bullish') else 0.0
    f['ict_has_bear_bos'] = 1.0 if (analysis.last_bos and analysis.last_bos.direction == 'bearish') else 0.0
    f['ict_has_bull_choch'] = 1.0 if (analysis.last_choch and analysis.last_choch.direction == 'bullish') else 0.0
    f['ict_has_bear_choch'] = 1.0 if (analysis.last_choch and analysis.last_choch.direction == 'bearish') else 0.0

    # Structure score: -1 (full bear) to +1 (full bull)
    score = 0.0
    if ms == 'bullish': score += 0.5
    elif ms == 'bearish': score -= 0.5
    if f['ict_has_bull_bos']: score += 0.3
    if f['ict_has_bear_bos']: score -= 0.3
    if f['ict_has_bull_choch']: score += 0.2
    if f['ict_has_bear_choch']: score -= 0.2
    f['ict_structure_score'] = max(-1.0, min(1.0, score))

    # ── Likidite ──
    f['ict_eq_highs_count'] = float(len(analysis.equal_highs))
    f['ict_eq_lows_count'] = float(len(analysis.equal_lows))
    f['ict_sweep_ssl'] = 1.0 if (analysis.sweep_detected and analysis.sweep_type == 'ssl_sweep') else 0.0
    f['ict_sweep_bsl'] = 1.0 if (analysis.sweep_detected and analysis.sweep_type == 'bsl_sweep') else 0.0

    # ── POI ──
    bull_obs = [ob for ob in analysis.order_blocks if ob.type == 'bullish' and not ob.mitigated]
    bear_obs = [ob for ob in analysis.order_blocks if ob.type == 'bearish' and not ob.mitigated]
    f['ict_bull_ob_count'] = float(len(bull_obs))
    f['ict_bear_ob_count'] = float(len(bear_obs))

    # Best OB strength
    all_active_obs = bull_obs + bear_obs
    f['ict_ob_best_strength'] = max((ob.displacement_strength for ob in all_active_obs), default=0.0)
    f['ict_ob_has_fvg'] = 1.0 if any(ob.has_fvg for ob in all_active_obs) else 0.0

    # FVG
    bull_fvgs = [fvg for fvg in analysis.fvg_zones if fvg.type == 'bullish' and not fvg.filled]
    bear_fvgs = [fvg for fvg in analysis.fvg_zones if fvg.type == 'bearish' and not fvg.filled]
    f['ict_bull_fvg_count'] = float(len(bull_fvgs))
    f['ict_bear_fvg_count'] = float(len(bear_fvgs))

    # FVG CE distance (ATR normalized)
    all_fvgs = bull_fvgs + bear_fvgs
    if all_fvgs and atr > 0:
        min_ce_dist = min(abs(current_price - fvg.ce) for fvg in all_fvgs)
        f['ict_fvg_ce_dist_atr'] = min_ce_dist / atr
    else:
        f['ict_fvg_ce_dist_atr'] = 10.0  # far away

    # POI Confluence
    f['ict_poi_confluence'] = analysis.poi_confluence

    # ── Displacement ──
    f['ict_displacement_bull'] = 1.0 if (analysis.displacement and analysis.displacement_direction == 'bullish') else 0.0
    f['ict_displacement_bear'] = 1.0 if (analysis.displacement and analysis.displacement_direction == 'bearish') else 0.0

    # ── OTE ──
    in_ote = 0.0
    if analysis.ote:
        if analysis.ote.bottom <= current_price <= analysis.ote.top:
            in_ote = 1.0
    f['ict_in_ote'] = in_ote

    # ── Premium/Discount ──
    sh_p = [sp.price for sp in analysis.swing_highs[-5:]] if analysis.swing_highs else []
    sl_p = [sp.price for sp in analysis.swing_lows[-5:]] if analysis.swing_lows else []
    sh = max(sh_p) if sh_p else current_price * 1.02
    sl = min(sl_p) if sl_p else current_price * 0.98
    rng = sh - sl
    if rng > 0:
        f['ict_premium_discount'] = (current_price - sl) / rng
    else:
        f['ict_premium_discount'] = 0.5

    # ── Session ──
    if analysis.session:
        f['ict_session_quality'] = analysis.session.quality
        wb = analysis.session.weekly_bias
        f['ict_weekly_score'] = {'mid': 1.0, 'early': 0.5, 'late': 0.0}.get(wb, 0.3)
    else:
        f['ict_session_quality'] = 0.3
        f['ict_weekly_score'] = 0.3

    # ── Breaker ──
    f['ict_breaker_count'] = float(len(analysis.breaker_blocks))

    return f


def extract_ict_features_from_df(
    df: pd.DataFrame,
    direction: str = '',
    swing_left: int = 3,
    swing_right: int = 2,
) -> Dict[str, float]:
    """
    DataFrame → ICT feature extraction (convenience wrapper).
    Runs ict_core.analyze() + extract_ict_features().
    """
    if df is None or len(df) < 30:
        return {k: 0.0 for k in ICT_FEATURES}

    analysis = ict_core.analyze(df, direction, swing_left, swing_right)
    cp = float(df['close'].iloc[-1])

    # ATR
    if 'atr' in df.columns:
        atr = float(df['atr'].iloc[-1])
        if np.isnan(atr) or atr <= 0:
            atr = cp * 0.01
    else:
        atr = cp * 0.01

    return extract_ict_features(analysis, cp, atr, direction)


def add_ict_features_bulk(
    df: pd.DataFrame,
    window: int = 60,
    swing_left: int = 3,
    swing_right: int = 2,
) -> pd.DataFrame:
    """
    DataFrame'e ICT feature kolonları ekle (her bar için).
    Meta-label generator tarafından kullanılır.

    Her bar'da son `window` bar'lık pencere üzerinde ict_core.analyze() çalıştırır.
    Yavaş ama doğru. Caching ile optimize.
    """
    n = len(df)
    if n < window:
        for feat in ICT_FEATURES:
            df[feat] = 0.0
        return df

    # Pre-allocate
    results = {feat: np.full(n, np.nan) for feat in ICT_FEATURES}

    for i in range(window, n):
        w_start = max(0, i - window)
        df_slice = df.iloc[w_start:i + 1]

        feats = extract_ict_features_from_df(df_slice, '', swing_left, swing_right)
        for feat, val in feats.items():
            results[feat][i] = val

    for feat, arr in results.items():
        df[feat] = arr

    return df
