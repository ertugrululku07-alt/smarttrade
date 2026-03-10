"""
SmartTrade Sinyal Kütüphanesi — Profesyonel Teknik Analiz İndikatörleri

Tüm hesaplamalar pandas + numpy ile yapılır, harici kütüphane gerekmez.
Her indikatör bir pd.Series döndürür.

İndikatörler:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - EMA (Exponential Moving Average)
    - SMA (Simple Moving Average)
    - ATR (Average True Range)  — dinamik SL/TP için
    - Stochastic Oscillator (%K, %D)
    - Volume Spike Detection
    - Confluence Score (çoklu sinyal birliği)
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Temel İndikatörler
# ─────────────────────────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=period, min_periods=1).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI — Wilder's smoothing metoduyla.
    Değerler: 0–100. <30 aşırı satım, >70 aşırı alım.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal_period)
    Histogram = MACD - Signal
    Dönen: (macd_line, signal_line, histogram)
    """
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0
                    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.
    Dönen: (upper, middle, lower)
    """
    middle = sma(close, period)
    std = close.rolling(window=period, min_periods=1).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
        ) -> pd.Series:
    """
    ATR — Average True Range.
    Volatilite göstergesi, dinamik SL/TP hesaplamada kullanılır.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3
               ) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator.
    %K: (close − lowest_low) / (highest_high − lowest_low) × 100
    %D: SMA(%K, d_period)
    """
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = ((close - lowest_low) / denom) * 100
    d = sma(k, d_period)
    return k, d


def volume_spike(volume: pd.Series, period: int = 20, multiplier: float = 2.0
                 ) -> pd.Series:
    """
    Volume Spike: volume, son N mumdaki ortalamanın X katından yüksekse True.
    """
    avg_volume = sma(volume, period)
    return volume > avg_volume * multiplier


def ema_cross(close: pd.Series, fast: int = 9, slow: int = 21
              ) -> pd.Series:
    """
    EMA crossover sinyali.
    +1: bullish cross (fast EMA geçti slow EMA'nın üzerine)
    -1: bearish cross
     0: cross yok
    """
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    above = (fast_ema > slow_ema).astype(int)
    cross = above.diff()
    return cross  # +1: bullish, -1: bearish


# ─────────────────────────────────────────────────────────────────────────────
# İndikatörleri DataFrame'e Hesapla
# ─────────────────────────────────────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham OHLCV DataFrame'ine tüm teknik indikatörleri ekler.
    Giriş kolonları: open, high, low, close, volume
    """
    df = df.copy()

    close = df['close']
    high  = df['high']
    low   = df['low']
    vol   = df['volume']

    # RSI
    df['rsi'] = rsi(close, 14)

    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = macd(close)

    # Bollinger Bands
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = bollinger_bands(close, 20, 2.0)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # EMA'lar
    df['ema5']  = ema(close, 5)
    df['ema10'] = ema(close, 10)
    df['ema9']  = ema(close, 9)
    df['ema21'] = ema(close, 21)
    df['ema50'] = ema(close, 50)
    df['ema200']= ema(close, 200)

    # ATR (dinamik SL/TP için)
    df['atr'] = atr(high, low, close, 14)
    df['atr_pct'] = df['atr'] / close * 100  # ATR yüzde olarak

    # Stochastic
    df['stoch_k'], df['stoch_d'] = stochastic(high, low, close)

    # Volume
    df['vol_spike'] = volume_spike(vol, 20, 2.0)

    # EMA Cross
    df['ema_cross'] = ema_cross(close, 9, 21)

    # Trend yönü (50 EMA üstü = uptrend)
    df['trend_up'] = close > df['ema50']
    df['trend_strong'] = close > df['ema200']

    # Bollinger pozisyonu
    df['bb_pos'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # ADX — Trend Gücü (Average Directional Index)
    # ADX > 20: trend var | ADX > 40: güçlü trend
    tr_raw = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    _dm_plus  = (high - high.shift(1)).clip(lower=0)
    _dm_minus = (low.shift(1) - low).clip(lower=0)
    _dm_plus  = _dm_plus.where(_dm_plus > _dm_minus, 0)
    _dm_minus = _dm_minus.where(_dm_minus > _dm_plus.where(_dm_plus > _dm_minus, 0).shift(0), 0)
    _atr14    = tr_raw.ewm(alpha=1/14, adjust=False).mean()
    _di_plus  = 100 * _dm_plus.ewm(alpha=1/14, adjust=False).mean() / (_atr14 + 1e-9)
    _di_minus = 100 * _dm_minus.ewm(alpha=1/14, adjust=False).mean() / (_atr14 + 1e-9)
    _dx       = 100 * (_di_plus - _di_minus).abs() / (_di_plus + _di_minus + 1e-9)
    df['adx']      = _dx.ewm(alpha=1/14, adjust=False).mean()
    df['di_plus']  = _di_plus
    df['di_minus'] = _di_minus

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Confluence Score — Çoklu sinyal birliği
# ─────────────────────────────────────────────────────────────────────────────

def confluence_score(df: pd.DataFrame) -> pd.Series:
    """
    Birden fazla indikatörün aynı yönü göstermesi durumunda +1/-1 ekleyen
    'Confluence Score'. Yüksek score = güçlü sinyal.

    Puanlama (BUY yönü için +1 her koşul):
      +1: RSI < 35 (aşırı satım)
      +1: MACD bullish crossover
      +1: Fiyat BB alt bandının altında
      +1: Stochastic %K < 25
      +1: EMA9 > EMA21 (kısa vadeli bullish)
      +1: Yüksek volume spike

    Tersine SELL yönü için negatif puan verilir.
    """
    score = pd.Series(0, index=df.index)

    # BUY puanları
    score += (df['rsi'] < 35).astype(int)
    score += ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    score += (df['close'] < df['bb_lower']).astype(int)
    score += (df['stoch_k'] < 25).astype(int)
    score += (df['ema9'] > df['ema21']).astype(int)
    score += df['vol_spike'].astype(int)

    # SELL puanları (çıkar)
    score -= (df['rsi'] > 65).astype(int)
    score -= ((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
    score -= (df['close'] > df['bb_upper']).astype(int)
    score -= (df['stoch_k'] > 75).astype(int)
    score -= (df['ema9'] < df['ema21']).astype(int)

    df = df.copy()
    return score


def generate_signals(df: pd.DataFrame, strategy: str = "confluence",
                     rsi_oversold: int = 30, rsi_overbought: int = 70,
                     confluence_min: int = 3) -> pd.Series:
    """
    Belirlenen strateji tipine göre BUY/SELL/HOLD sinyali üretir.

    Stratejiler:
      - rsi_only: Sadece RSI tabanlı (oversold/overbought)
      - macd_cross: Sadece MACD crossover
      - ema_cross: Sadece EMA 9/21 crossover
      - bb_bounce: Bollinger Band bouncing
      - confluence: Çoklu indikatör birliği (en güçlü)
      - scalping: EMA cross + RSI filtresi (hızlı trading)
      - swing: MACD + BB + trend (uzun vadeli)
    """
    signal = pd.Series("HOLD", index=df.index)
    score = confluence_score(df)

    if strategy == "rsi_only":
        signal[df['rsi'] < rsi_oversold] = "BUY"
        signal[df['rsi'] > rsi_overbought] = "SELL"

    elif strategy == "macd_cross":
        bullish_cross = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        bearish_cross = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        signal[bullish_cross] = "BUY"
        signal[bearish_cross] = "SELL"

    elif strategy == "ema_cross":
        signal[df['ema_cross'] == 1] = "BUY"
        signal[df['ema_cross'] == -1] = "SELL"

    elif strategy == "bb_bounce":
        signal[df['close'] < df['bb_lower']] = "BUY"
        signal[df['close'] > df['bb_upper']] = "SELL"

    elif strategy == "confluence":
        signal[score >= confluence_min] = "BUY"
        signal[score <= -confluence_min] = "SELL"

    elif strategy == "scalping":
        # EMA cross şartı + RSI < 65 filtresi
        bullish = (df['ema_cross'] == 1) & (df['rsi'] < 65)
        bearish = (df['ema_cross'] == -1) & (df['rsi'] > 35)
        signal[bullish] = "BUY"
        signal[bearish] = "SELL"

    elif strategy == "swing":
        # MACD + BB + trend filtresi
        buy_cond = (
            (df['macd'] > df['macd_signal']) &
            (df['close'] < df['bb_mid']) &
            (df['trend_up'])
        )
        sell_cond = (
            (df['macd'] < df['macd_signal']) &
            (df['close'] > df['bb_mid'])
        )
        signal[buy_cond] = "BUY"
        signal[sell_cond] = "SELL"

    return signal, score
