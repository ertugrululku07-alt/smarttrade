"""
XGBoost Feature Engineering v1.2

Feature üretim katmanı — tüm stratejiler ve ML modelleri için
ortak feature seti oluşturur.

Feature Kaynakları:
  1. Upstream (add_all_indicators):
     rsi, macd_hist, bb_width, stoch_k, ema_cross, bb_pos,
     adx, di_plus, di_minus, atr

  2. Bu modülde hesaplanan (generate_features):
     atr_pct, volatility_10, momentum_10, atr_rank_50,
     hurst, efficiency_ratio, zscore_20, adx_accel,
     di_diff, vol_ratio_20

  3. MTF Analyzer (prepare_bulk_mtf_features / meta pipeline):
     htf_bias_numeric, htf_trend_strength, htf_rsi,
     htf_price_vs_ema200, htf_ema_alignment, htf_structure_numeric

v1.2 Düzeltmeleri:
  - ✅ #1  HTF feature isimleri MTFAnalyzer ile senkronize
  - ✅ #2  Eksik HTF feature'lar eklendi
  - ✅ #3  di_diff hesaplaması eklendi
  - ✅ #4  vol_ratio_20 hesaplaması eklendi
  - ✅ #5  atr_pct hesaplaması eklendi
  - ✅ #6  Eksik feature'lar NaN (0 değil) — XGBoost native NaN
  - ✅ #7  Hurst NaN koruması güçlendirildi
  - ✅ #8  _safe_rank NaN array koruması
  - ✅ #9  Feature kaynakları belgelendi
  - ✅ #10 FEATURE_COLS doğru ve eksiksiz
  - ✅ #11 inf/NaN temizleme katmanı eklendi
  - ✅ #12 Feature dokümantasyonu eklendi
"""

import os
import warnings
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════
# Hurst Exponent
# ══════════════════════════════════════════════════════════════

try:
    from ai.regime_detector import _hurst_exponent as _hurst_imported
    _HURST_AVAILABLE = True
except ImportError:
    _HURST_AVAILABLE = False


def _hurst_exponent(x):
    """
    Hurst Exponent hesaplama (Numpy tabanlı).

    H > 0.5 → trending (persistent)
    H = 0.5 → random walk
    H < 0.5 → mean reverting (anti-persistent)

    [FIX #7]: NaN ve edge case koruması güçlendirildi.
    """
    # İmport edilmiş versiyon varsa onu kullan
    if _HURST_AVAILABLE:
        try:
            result = _hurst_imported(x)
            if np.isnan(result) or np.isinf(result):
                return 0.5
            return float(np.clip(result, 0.0, 1.0))
        except Exception:
            pass

    # Fallback: kendi hesaplamamız
    if len(x) < 20:
        return 0.5

    # NaN kontrolü
    if np.any(np.isnan(x)):
        x_clean = x[~np.isnan(x)]
        if len(x_clean) < 20:
            return 0.5
        x = x_clean

    try:
        max_lag = min(20, len(x) // 2)
        lags = range(2, max_lag)

        tau = []
        for lag in lags:
            diff = x[lag:] - x[:-lag]
            std = np.std(diff)
            if std <= 0 or np.isnan(std):
                return 0.5
            tau.append(std)

        if len(tau) < 2:
            return 0.5

        log_lags = np.log(list(lags))
        log_tau = np.log(tau)

        # Inf/NaN kontrolü
        if (np.any(np.isnan(log_lags))
                or np.any(np.isnan(log_tau))
                or np.any(np.isinf(log_lags))
                or np.any(np.isinf(log_tau))):
            return 0.5

        poly = np.polyfit(log_lags, log_tau, 1)
        result = float(poly[0])

        # Mantıklı aralıkta mı?
        if np.isnan(result) or np.isinf(result):
            return 0.5

        return float(np.clip(result, 0.0, 1.0))

    except Exception:
        return 0.5


# ══════════════════════════════════════════════════════════════
# Safe Rank
# ══════════════════════════════════════════════════════════════

def _safe_rank(arr):
    """
    Numpy tabanlı hızlı percentile rank.

    [FIX #8]: NaN array koruması eklendi.
    """
    if len(arr) < 2:
        return 0.5

    last_val = arr[-1]

    # NaN kontrolü
    if np.isnan(last_val):
        return 0.5

    # Tüm NaN array kontrolü
    valid = arr[~np.isnan(arr)]
    if len(valid) < 2:
        return 0.5

    rank = np.sum(valid <= last_val) / len(valid)
    return float(rank)


# ══════════════════════════════════════════════════════════════
# Feature Kolon Tanımları  [FIX #1, #2, #10]
# ══════════════════════════════════════════════════════════════

# Upstream indicator'lardan gelen feature'lar
# (add_all_indicators tarafından üretilmeli)
UPSTREAM_FEATURES = [
    'rsi',              # RSI (14)
    'macd_hist',        # MACD Histogram
    'bb_width',         # Bollinger Band genişliği
    'stoch_k',          # Stochastic %K
    'ema_cross',        # EMA cross sinyali
    'bb_pos',           # Bollinger Band içi pozisyon
    'adx',              # ADX (trend gücü)
    'di_plus',          # DI+
    'di_minus',         # DI-
]

# Bu modülde hesaplanan feature'lar
COMPUTED_FEATURES = [
    'atr_pct',          # ATR / Close (normalize)
    'volatility_10',    # 10-bar standard deviation
    'momentum_10',      # 10-bar price change
    'atr_rank_50',      # ATR percentile rank (50 bar)
    'hurst',            # Hurst exponent
    'efficiency_ratio', # Kaufman efficiency ratio
    'zscore_20',        # 20-bar z-score
    'adx_accel',        # ADX 3-bar değişim hızı
    'di_diff',          # DI+ - DI- (trend yönü)
    'vol_ratio_20',     # Volume / Volume MA20
]

# MTF Analyzer tarafından eklenen feature'lar  [FIX #1, #2]
# MTFAnalyzer.generate_cross_tf_features() ve
# MTFAnalyzer.prepare_bulk_mtf_features() ile tutarlı
HTF_FEATURES = [
    'htf_bias_numeric',         # [-2, +2] 4h trend yönü
    'htf_trend_strength',       # [0, 1] trend gücü  [FIX #1: was htf_atr]
    'htf_rsi',                  # 4h RSI
    'htf_price_vs_ema200',      # ATR-normalize fiyat/EMA200 mesafesi
    'htf_ema_alignment',        # {-1, 0, +1} EMA sıralaması  [FIX #2: was missing]
    'htf_structure_numeric',    # {-1, 0, +1} swing yapısı  [FIX #2: was missing]
]

# Tüm feature'lar — export edilen ana liste
FEATURE_COLS = UPSTREAM_FEATURES + COMPUTED_FEATURES + HTF_FEATURES


# ══════════════════════════════════════════════════════════════
# Feature Üretimi
# ══════════════════════════════════════════════════════════════

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ML modeli için feature üretimi.

    Bu fonksiyon:
      1. Upstream indicator'ların varlığını kontrol eder
      2. Türetilmiş feature'ları hesaplar
      3. Eksik feature'ları NaN ile işaretler (0 değil)
      4. inf değerleri NaN'e çevirir

    HTF feature'lar burada ÜRETİLMEZ — MTF Analyzer veya
    meta-label pipeline tarafından eklenir.

    Args:
        df: OHLCV + indicator DataFrame

    Returns:
        df + computed features
    """
    df = df.copy()

    # ══════════════════════════════════════════════
    # 1. TÜRETILMIŞ FEATURE'LAR
    # ══════════════════════════════════════════════

    # ── ATR Percentage ──  [FIX #5]
    if 'atr' in df.columns and 'close' in df.columns:
        close_safe = df['close'].replace(0, np.nan)
        df['atr_pct'] = df['atr'] / close_safe
    else:
        df['atr_pct'] = np.nan

    # ── ATR Rank (50-bar percentile) ──
    if 'atr' in df.columns:
        df['atr_rank_50'] = df['atr'].rolling(
            50, min_periods=10
        ).apply(_safe_rank, raw=True)
    else:
        df['atr_rank_50'] = np.nan

    # ── Volatility & Momentum ──
    if 'close' in df.columns:
        df['volatility_10'] = df['close'].rolling(
            10, min_periods=5
        ).std()

        df['momentum_10'] = df['close'].diff(10)

        # Hurst Exponent
        df['hurst'] = df['close'].rolling(
            50, min_periods=20
        ).apply(_hurst_exponent, raw=True)

        # Efficiency Ratio (Kaufman)
        direction = (
            df['close'] - df['close'].shift(10)
        ).abs()
        volatility_sum = df['close'].diff().abs().rolling(
            10, min_periods=5
        ).sum()
        df['efficiency_ratio'] = np.where(
            volatility_sum > 0,
            direction / volatility_sum,
            0.5,
        )

        # Z-Score (20-bar)
        mean_20 = df['close'].rolling(20, min_periods=10).mean()
        std_20 = df['close'].rolling(20, min_periods=10).std()
        df['zscore_20'] = np.where(
            std_20 > 1e-10,
            (df['close'] - mean_20) / std_20,
            0.0,
        )
    else:
        for col in ['volatility_10', 'momentum_10', 'hurst',
                     'efficiency_ratio', 'zscore_20']:
            df[col] = np.nan

    # ── ADX Acceleration ──
    if 'adx' in df.columns:
        df['adx_accel'] = df['adx'].diff(3)
    else:
        df['adx_accel'] = np.nan

    # ── DI Difference ──  [FIX #3]
    if 'di_plus' in df.columns and 'di_minus' in df.columns:
        df['di_diff'] = df['di_plus'] - df['di_minus']
    else:
        df['di_diff'] = np.nan

    # ── Volume Ratio ──  [FIX #4]
    if 'volume' in df.columns:
        vol_ma20 = df['volume'].rolling(
            20, min_periods=5
        ).mean()
        vol_ma_safe = vol_ma20.replace(0, np.nan)
        df['vol_ratio_20'] = df['volume'] / vol_ma_safe
    else:
        df['vol_ratio_20'] = np.nan

    # ══════════════════════════════════════════════
    # 2. UPSTREAM FEATURE VALIDATION  [FIX #6]
    # ══════════════════════════════════════════════

    # HTF feature'ları hariç tut (downstream'de eklenir)
    non_htf_features = [
        col for col in FEATURE_COLS
        if not col.startswith('htf_')
    ]

    missing = [
        col for col in non_htf_features
        if col not in df.columns
    ]

    if missing:
        # [FIX #6]: NaN ile doldur (0 değil)
        # XGBoost native NaN handling kullanır
        warnings.warn(
            f"[WARN] {len(missing)} feature eksik: {missing}. "
            f"NaN ile dolduruluyor (XGBoost native NaN handling).",
            UserWarning,
        )
        for col in missing:
            df[col] = np.nan

    # ══════════════════════════════════════════════
    # 3. INF / NaN TEMİZLEME  [FIX #11]
    # ══════════════════════════════════════════════

    # inf → NaN (XGBoost inf'i handle edemez)
    feature_cols_present = [
        col for col in non_htf_features
        if col in df.columns
    ]

    for col in feature_cols_present:
        if df[col].dtype in [np.float64, np.float32, float]:
            df[col] = df[col].replace(
                [np.inf, -np.inf], np.nan
            )

    return df


# ══════════════════════════════════════════════════════════════
# Feature Yardımcıları
# ══════════════════════════════════════════════════════════════

def get_base_feature_names() -> list:
    """
    HTF hariç base feature isimlerini döndürür.

    Kullanım: meta_trainer, adaptive_engine
    """
    return [
        col for col in FEATURE_COLS
        if not col.startswith('htf_')
    ]


def get_htf_feature_names() -> list:
    """
    HTF feature isimlerini döndürür.

    Kullanım: MTF-aware eğitim pipeline'ı
    """
    return HTF_FEATURES.copy()


def get_all_feature_names() -> list:
    """Tüm feature isimlerini döndürür."""
    return FEATURE_COLS.copy()


def validate_feature_coverage(
    df: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """
    DataFrame'deki feature coverage'ı raporla.

    Returns:
        {
            'total': int,
            'present': int,
            'missing': list,
            'null_pct': dict,  # kolon: NaN yüzdesi
            'coverage_pct': float,
        }
    """
    present = [col for col in FEATURE_COLS if col in df.columns]
    missing = [col for col in FEATURE_COLS if col not in df.columns]

    null_pct = {}
    for col in present:
        if len(df) > 0:
            pct = df[col].isna().mean() * 100
            if pct > 0:
                null_pct[col] = round(pct, 1)

    coverage = len(present) / len(FEATURE_COLS) * 100

    if verbose:
        print(f"  Feature Coverage: {coverage:.0f}% "
              f"({len(present)}/{len(FEATURE_COLS)})")
        if missing:
            print(f"  Missing: {missing}")
        if null_pct:
            high_null = {
                k: v for k, v in null_pct.items() if v > 50
            }
            if high_null:
                print(f"  High NaN (>50%): {high_null}")

    return {
        'total': len(FEATURE_COLS),
        'present': len(present),
        'missing': missing,
        'null_pct': null_pct,
        'coverage_pct': round(coverage, 1),
    }


# ══════════════════════════════════════════════════════════════
# Progress Tracker Stub
# ══════════════════════════════════════════════════════════════

def _update_progress(
    status: str,
    progress: int,
    message: str = "",
):
    """Progress tracker stub for UI sync."""
    pass