import pandas as pd

FEATURE_COLS = [
    'rsi', 'macd_hist', 'bb_width', 
    'atr_pct', 'stoch_k', 'ema_cross', 'bb_pos', 'adx', 
    'di_plus', 'di_minus', 'volatility_10', 'momentum_10', 'atr_rank_50',
    'hurst', # Phase 7
    'efficiency_ratio', 'zscore_20', 'adx_accel' # Phase 8
]

def _safe_rank(series):
    """Son degerin rolling window icindeki percentile rank'i"""
    if len(series) < 2:
        return 0.5
    try:
        # series bir ndarray veya list olarak gelebilir (raw=False/True ayarına göre)
        s = pd.Series(series)
        if s.isna().all():
            return 0.5
        ranked = s.rank(pct=True)
        return float(ranked.iloc[-1])
    except:
        return 0.5

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulated generate_features to replace the legacy XGBoost trainer logic.
    Most features are already created in add_all_indicators in signals.py.
    """
    df = df.copy()
    
    # ATR bazlı rank (Regime Detection için kritik)
    if 'atr' in df.columns:
        df['atr_rank_50'] = df['atr'].rolling(50, min_periods=10).apply(
            _safe_rank, raw=False
        )
    else:
        df['atr_rank_50'] = 0.5

    if 'close' in df.columns:
        df['volatility_10'] = df['close'].rolling(10).std()
        df['momentum_10'] = df['close'].diff(10)
        
        # HURST Feature (Phase 7)
        from ai.regime_detector import _hurst_exponent
        df['hurst'] = df['close'].rolling(50).apply(
            lambda x: _hurst_exponent(x.values), raw=False
        )
        
    return df

def _update_progress(status: str, progress: int, message: str = ""):
    """Legacy UI progress updater stub"""
    pass
