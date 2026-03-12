import pandas as pd
import numpy as np
import datetime
import warnings
import os

# Circular import riskini önlemek için modül başında güvenli import
try:
    from ai.regime_detector import _hurst_exponent
except ImportError:
    def _hurst_exponent(x):
        """Hurst Fallback (Numpy tabanlı hızlı versiyon)"""
        if len(x) < 20: return 0.5
        lags = range(2, min(20, len(x) // 2))
        tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
        if not tau or any(t == 0 for t in tau): return 0.5
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return float(poly[0])

FEATURE_COLS = [
    'rsi', 'macd_hist', 'bb_width', 
    'atr_pct', 'stoch_k', 'ema_cross', 'bb_pos', 'adx', 
    'di_plus', 'di_minus', 'volatility_10', 'momentum_10', 'atr_rank_50',
    'hurst', 'efficiency_ratio', 'zscore_20', 'adx_accel',
    'di_diff', 'vol_ratio_20'
]

def _safe_rank(arr):
    """
    Numpy tabanlı hızlı percentile rank (raw=True için optimize edildi).
    Pandas Series overhead'ini (pd.Series oluşturma) ortadan kaldırır.
    """
    if len(arr) < 2: return 0.5
    last_val = arr[-1]
    if np.isnan(last_val): return 0.5
    # Numpy vektörel karşılaştırma: ~10x-20x performans artışı
    rank = np.sum(arr <= last_val) / len(arr)
    return float(rank)

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize edilmiş ve doğrulanmış Feature Engineering katmanı.
    """
    df = df.copy()
    
    # --- 1. Temel Feature Üretimi (Numpy Vektörel) ---
    if 'atr' in df.columns:
        # raw=True: numpy array geçilir, Series objesi oluşturulmaz. Performans artışı!
        df['atr_rank_50'] = df['atr'].rolling(50, min_periods=10).apply(_safe_rank, raw=True)
    
    if 'close' in df.columns:
        df['volatility_10'] = df['close'].rolling(10).std()
        df['momentum_10'] = df['close'].diff(10)
        
        # Hurst Exponent (raw=True ile numpy hızında)
        df['hurst'] = df['close'].rolling(50).apply(_hurst_exponent, raw=True)
        
        # Phase 8: Efficiency Ratio (Kaufman)
        direction = (df['close'] - df['close'].shift(10)).abs()
        volatility = df['close'].diff().abs().rolling(10).sum()
        df['efficiency_ratio'] = np.where(volatility > 0, direction / volatility, 0.5)
        
        # Phase 8: Z-Score 20
        mean_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['zscore_20'] = np.where(std_20 > 0, (df['close'] - mean_20) / std_20, 0)

    # Phase 8: ADX Acceleration
    if 'adx' in df.columns:
        df['adx_accel'] = df['adx'].diff(3)
    else:
        # Prevent crash if ADX is missing during dynamic feature gen
        df['adx_accel'] = 0

    # --- 2. Kritik Doğrulama (Validation) Katmanı ---
    # Downstream kodun (XGBoost veya Karar Motoru) çökmemesi için garanti
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        # Önemli: Gerçek bir botta bu durum loglanmalı ve incelenmelidir.
        warnings.warn(f"⚠️ KRITIK: {len(missing)} feature eksik: {missing}. SIFIR ile dolduruluyor.")
        for col in missing:
            df[col] = 0
            
    return df

def _update_progress(status: str, progress: int, message: str = ""):
    """Progress tracker stub for UI sync"""
    pass
