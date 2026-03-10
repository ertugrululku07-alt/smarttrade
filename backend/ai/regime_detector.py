"""
Adaptive Regime Detector v2.0

Rejimler:
  TRENDING:       ADX>25, güçlü yönsel hareket, Hurst>0.55
  MEAN_REVERTING: ADX<20, BB dar, RSI uçlarda, Hurst<0.45
  HIGH_VOLATILE:  ATR rank>0.75, ani spike'lar
  LOW_VOLATILE:   ATR rank<0.25, dar range, düşük hacim
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Tuple, Optional


class Regime(Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILE = "high_volatile"
    LOW_VOLATILE = "low_volatile"


# ═══════════════════════════════════════════════════════════════════
# Hurst Exponent — Trend vs Mean Reversion ayrımı
# ═══════════════════════════════════════════════════════════════════

def _hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """
    Hurst Exponent hesapla:
      H > 0.55 → Trending (momentum)
      H < 0.45 → Mean Reverting
      H ≈ 0.50 → Random Walk
    """
    if len(series) < max_lag * 2:
        return 0.50

    lags = range(2, max_lag + 1)
    tau = []
    for lag in lags:
        diffs = series[lag:] - series[:-lag]
        std = np.std(diffs)
        if std > 0:
            tau.append(std)
        else:
            tau.append(1e-10)

    if len(tau) < 3:
        return 0.50

    try:
        log_lags = np.log(list(lags[:len(tau)]))
        log_tau = np.log(tau)
        poly = np.polyfit(log_lags, log_tau, 1)
        return float(np.clip(poly[0], 0.0, 1.0))
    except Exception:
        return 0.50


# ═══════════════════════════════════════════════════════════════════
# Rejim Tespiti — Çoklu Metrik Bazlı
# ═══════════════════════════════════════════════════════════════════

def detect_regime(df: pd.DataFrame, lookback: int = 50) -> Tuple[Regime, dict]:
    """
    Mevcut rejimi tespit et.

    Returns:
        (Regime, details_dict)
    """
    if len(df) < lookback:
        return Regime.MEAN_REVERTING, {"reason": "yetersiz veri"}

    tail = df.tail(lookback)

    # ── Metrikler ─────────────────────────────────────────────
    adx = tail['adx'].iloc[-1] if 'adx' in tail.columns else 20.0
    adx_avg = tail['adx'].mean() if 'adx' in tail.columns else 20.0

    atr_rank = tail['atr_rank_50'].iloc[-1] if 'atr_rank_50' in tail.columns else 0.50
    atr_pct = tail['atr_pct'].iloc[-1] if 'atr_pct' in tail.columns else 0.01

    bb_width = tail['bb_width'].iloc[-1] if 'bb_width' in tail.columns else 0.05
    bb_width_avg = tail['bb_width'].mean() if 'bb_width' in tail.columns else 0.05

    vol_ratio = tail['vol_ratio_20'].iloc[-1] if 'vol_ratio_20' in tail.columns else 1.0

    di_diff = abs(tail['di_diff'].iloc[-1]) if 'di_diff' in tail.columns else 0.0

    # Hurst exponent
    close_vals = tail['close'].values.astype(float)
    hurst = _hurst_exponent(close_vals, max_lag=min(20, lookback // 3))

    # Yönsellik: Son N bar'daki kapanış yönü tutarlılığı
    returns = tail['close'].pct_change().dropna()
    if len(returns) > 5:
        pos_returns = (returns > 0).sum()
        neg_returns = (returns < 0).sum()
        directionality = abs(pos_returns - neg_returns) / len(returns)
    else:
        directionality = 0.0

    # ── Skor Hesaplama ────────────────────────────────────────
    scores = {
        Regime.TRENDING: 0.0,
        Regime.MEAN_REVERTING: 0.0,
        Regime.HIGH_VOLATILE: 0.0,
        Regime.LOW_VOLATILE: 0.0,
    }

    # --- TRENDING --- ORİJİNAL
    if adx > 30:
        scores[Regime.TRENDING] += 3.0
    elif adx > 25:
        scores[Regime.TRENDING] += 2.0
    elif adx > 20:
        scores[Regime.TRENDING] += 1.0

    if hurst > 0.55:
        scores[Regime.TRENDING] += 2.0
    elif hurst > 0.50:
        scores[Regime.TRENDING] += 1.0

    if di_diff > 10:
        scores[Regime.TRENDING] += 1.5
    if directionality > 0.3:
        scores[Regime.TRENDING] += 1.5

    # --- MEAN REVERTING --- ORİJİNAL
    if adx < 20:
        scores[Regime.MEAN_REVERTING] += 2.5
    elif adx < 25:
        scores[Regime.MEAN_REVERTING] += 1.5

    if hurst < 0.45:
        scores[Regime.MEAN_REVERTING] += 2.5
    elif hurst < 0.50:
        scores[Regime.MEAN_REVERTING] += 1.0

    if bb_width < bb_width_avg * 0.8:
        scores[Regime.MEAN_REVERTING] += 1.5
    if directionality < 0.15:
        scores[Regime.MEAN_REVERTING] += 1.0

    if bb_width < bb_width_avg * 0.8:
        scores[Regime.MEAN_REVERTING] += 1.5

    if directionality < 0.15:
        scores[Regime.MEAN_REVERTING] += 1.0

    # --- HIGH VOLATILE skorları ---
    if atr_rank > 0.80:
        scores[Regime.HIGH_VOLATILE] += 3.0
    elif atr_rank > 0.65:
        scores[Regime.HIGH_VOLATILE] += 1.5

    if vol_ratio > 2.0:
        scores[Regime.HIGH_VOLATILE] += 2.0
    elif vol_ratio > 1.5:
        scores[Regime.HIGH_VOLATILE] += 1.0

    if bb_width > bb_width_avg * 1.5:
        scores[Regime.HIGH_VOLATILE] += 1.5

    # --- LOW VOLATILE skorları ---
    if atr_rank < 0.20:
        scores[Regime.LOW_VOLATILE] += 3.0
    elif atr_rank < 0.30:
        scores[Regime.LOW_VOLATILE] += 1.5

    if vol_ratio < 0.6:
        scores[Regime.LOW_VOLATILE] += 2.0
    elif vol_ratio < 0.8:
        scores[Regime.LOW_VOLATILE] += 1.0

    if bb_width < bb_width_avg * 0.5:
        scores[Regime.LOW_VOLATILE] += 1.5

    # ── En yüksek skora sahip rejim ──────────────────────────
    best_regime = max(scores, key=scores.get)

    details = {
        "adx": round(adx, 2),
        "adx_avg": round(adx_avg, 2),
        "hurst": round(hurst, 3),
        "atr_rank": round(atr_rank, 3),
        "bb_width": round(bb_width, 5),
        "vol_ratio": round(vol_ratio, 2),
        "directionality": round(directionality, 3),
        "di_diff": round(di_diff, 2),
        "scores": {r.value: round(s, 2) for r, s in scores.items()},
        "best_score": round(scores[best_regime], 2),
    }

    return best_regime, details


# ═══════════════════════════════════════════════════════════════════
# Seri Rejim Tespiti (Backtest/Eğitim için)
# ═══════════════════════════════════════════════════════════════════

def detect_regime_series(
    df: pd.DataFrame,
    lookback: int = 50,
    smooth_window: int = 5,
) -> pd.Series:
    """
    Her bar için rejim tespit et.
    smooth_window: Rejim geçişlerini yumuşat (çok sık değişme)
    """
    regimes = []
    regime_values = {
        Regime.TRENDING: 0,
        Regime.MEAN_REVERTING: 1,
        Regime.HIGH_VOLATILE: 2,
        Regime.LOW_VOLATILE: 3,
    }

    for i in range(len(df)):
        if i < lookback:
            regimes.append(Regime.MEAN_REVERTING.value)
            continue

        window = df.iloc[max(0, i - lookback):i + 1]
        regime, _ = detect_regime(window, lookback=lookback)
        regimes.append(regime.value)

    regime_series = pd.Series(regimes, index=df.index, name='regime')

    # Rejim yumuşatma: Son N bar'ın en sık rejimi
    if smooth_window > 1:
        regime_numeric = pd.Series(
            [regime_values.get(Regime(r), 1) for r in regimes],
            index=df.index
        )
        smoothed = regime_numeric.rolling(smooth_window, center=False).apply(
            lambda x: pd.Series(x).mode().iloc[0] if len(x) > 0 else 1,
            raw=False
        )
        value_to_regime = {v: k.value for k, v in regime_values.items()}
        regime_series = smoothed.map(value_to_regime).fillna(Regime.MEAN_REVERTING.value)

    return regime_series


# ═══════════════════════════════════════════════════════════════════
# Rejim Dağılımı Raporu
# ═══════════════════════════════════════════════════════════════════

def regime_distribution_report(regime_series: pd.Series) -> dict:
    """Rejim dağılım istatistiklerini döndürür."""
    counts = regime_series.value_counts()
    total = len(regime_series)

    report = {}
    for regime_val in [r.value for r in Regime]:
        count = counts.get(regime_val, 0)
        report[regime_val] = {
            "count": int(count),
            "pct": round(count / total * 100, 1),
        }

    return report
