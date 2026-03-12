"""
SmartTrade Sinyal Kütüphanesi — Profesyonel Teknik Analiz İndikatörleri

Tüm hesaplamalar pandas + numpy ile yapılır, harici kütüphane gerekmez.
Her indikatör bir pd.Series döndürür.

İndikatörler:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - EMA / SMA
    - ATR (Average True Range)
    - Stochastic Oscillator (%K, %D)
    - ADX (Average Directional Index)
    - Volume Spike Detection
    - ICT/SMC: FVG, Order Blocks, MSS
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


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
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


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
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


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    ATR — Average True Range.
    Volatilite göstergesi, dinamik SL/TP hesaplamada kullanılır.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
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


def volume_spike(
    volume: pd.Series,
    period: int = 20,
    multiplier: float = 2.0,
) -> pd.Series:
    """
    Volume Spike: volume, son N mumdaki ortalamanın X katından yüksekse True.
    """
    avg_volume = sma(volume, period)
    return volume > avg_volume * multiplier


def ema_cross(
    close: pd.Series,
    fast: int = 9,
    slow: int = 21,
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
    return cross


# ─────────────────────────────────────────────────────────────────────────────
# ICT / SMC Concepts (Phase 12)
# ─────────────────────────────────────────────────────────────────────────────

def find_fvg(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Fair Value Gap (FVG) Tespiti.
    - Bullish FVG: Mum(i)'nin düşüğü > Mum(i-2)'nin yükseği
    - Bearish FVG: Mum(i)'nin yükseği < Mum(i-2)'nin düşüğü
    """
    high = df['high']
    low = df['low']

    # Bullish FVG: i-2 High < i Low (3-candle gap)
    bull_fvg = (low > high.shift(2))
    # Bearish FVG: i-2 Low > i High
    bear_fvg = (high < low.shift(2))

    return bull_fvg.astype(int), bear_fvg.astype(int)


def find_order_blocks(
    df: pd.DataFrame,
    lookback: int = 5,
) -> Tuple[pd.Series, pd.Series]:
    """
    Order Block (OB) Tespiti.
    - Bullish OB: Sert bir yükselişten önceki son düşüş mumunun seviyesi.
    - Bearish OB: Sert bir düşüşten önceki son yükseliş mumunun seviyesi.
    """
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']

    # Mum gövde büyüklüğü ve yönü
    body_size = (close - open_).abs()
    direction = np.sign(close - open_)

    # "Sert hareket" tespiti (Gövde, son 5 mumun ortalamasından 2 kat büyükse)
    avg_body = body_size.rolling(lookback, min_periods=1).mean()
    is_impulsive = body_size > (avg_body * 2.0)

    bull_ob = pd.Series(0.0, index=df.index)
    bear_ob = pd.Series(0.0, index=df.index)

    # Bullish OB: i, impulsive LONG ise i-1 OB olabilir
    bull_ob_mask = is_impulsive & (direction == 1) & (direction.shift(1) == -1)
    bull_ob[bull_ob_mask] = low.shift(1)

    # Bearish OB: i, impulsive SHORT ise i-1 OB olabilir
    bear_ob_mask = is_impulsive & (direction == -1) & (direction.shift(1) == 1)
    bear_ob[bear_ob_mask] = high.shift(1)

    return bull_ob, bear_ob


# ─────────────────────────────────────────────────────────────────────────────
# İndikatörleri DataFrame'e Hesapla
# ─────────────────────────────────────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham OHLCV DataFrame'ine tüm teknik indikatörleri ekler.
    Giriş kolonları: open, high, low, close, volume

    Eklenen kolonlar:
        rsi, macd, macd_signal, macd_hist,
        bb_upper, bb_mid, bb_lower, bb_width, bb_pos,
        ema5, ema9, ema10, ema21, ema50, ema200,
        atr, atr_pct,
        stoch_k, stoch_d,
        vol_spike, vol_ratio_20,
        ema_cross,
        trend_up, trend_strong,
        adx, di_plus, di_minus, di_diff,
        candle_dir, upper_wick, lower_wick,
        efficiency_ratio, zscore_20, wick_rejection, adx_accel,
        fvg_bull, fvg_bear, ob_bull, ob_bear, mss_up, mss_down
    """
    df = df.copy()

    close = df['close']
    high = df['high']
    low = df['low']
    vol = df['volume']

    # ── RSI ───────────────────────────────────────────────────
    df['rsi'] = rsi(close, 14)

    # ── MACD ──────────────────────────────────────────────────
    df['macd'], df['macd_signal'], df['macd_hist'] = macd(close)

    # ── Bollinger Bands ───────────────────────────────────────
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = bollinger_bands(
        close, 20, 2.0
    )
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_width'] = bb_range / df['bb_mid'].replace(0, np.nan)

    # ── EMA'lar ───────────────────────────────────────────────
    df['ema5'] = ema(close, 5)
    df['ema9'] = ema(close, 9)
    df['ema10'] = ema(close, 10)
    df['ema21'] = ema(close, 21)
    df['ema50'] = ema(close, 50)
    df['ema200'] = ema(close, 200)

    # ── ATR ───────────────────────────────────────────────────
    df['atr'] = atr(high, low, close, 14)
    df['atr_pct'] = df['atr'] / close.replace(0, np.nan) * 100

    # ── Stochastic ────────────────────────────────────────────
    df['stoch_k'], df['stoch_d'] = stochastic(high, low, close)

    # ── Volume ────────────────────────────────────────────────
    df['vol_spike'] = volume_spike(vol, 20, 2.0)
    vol_avg_20 = vol.rolling(20, min_periods=1).mean().replace(0, np.nan)
    df['vol_ratio_20'] = vol / vol_avg_20

    # ── EMA Cross ─────────────────────────────────────────────
    df['ema_cross'] = ema_cross(close, 9, 21)

    # ── Trend Yönü ────────────────────────────────────────────
    df['trend_up'] = close > df['ema50']
    df['trend_strong'] = close > df['ema200']

    # ── Bollinger Pozisyonu ───────────────────────────────────
    df['bb_pos'] = (close - df['bb_lower']) / bb_range.replace(0, np.nan)

    # ── ADX — Wilder's Directional Movement ──────────────────
    #
    # Wilder kuralı:
    #   DM+ geçerli = DM+ > DM- VE DM+ > 0   (aksi halde 0)
    #   DM- geçerli = DM- > DM+ VE DM- > 0   (aksi halde 0)
    #
    tr_raw = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Ham Directional Movement
    _dm_plus_raw = (high - high.shift(1)).clip(lower=0)
    _dm_minus_raw = (low.shift(1) - low).clip(lower=0)

    # ✅ Fix: Hangisi büyükse o geçerli, diğeri sıfırlanır
    _dm_plus = _dm_plus_raw.where(_dm_plus_raw > _dm_minus_raw, 0)
    _dm_minus = _dm_minus_raw.where(_dm_minus_raw > _dm_plus_raw, 0)

    # Wilder's Smoothing (EMA alpha=1/14)
    _atr14 = tr_raw.ewm(alpha=1 / 14, adjust=False).mean()
    _di_plus = (
        100 * _dm_plus.ewm(alpha=1 / 14, adjust=False).mean()
        / (_atr14 + 1e-9)
    )
    _di_minus = (
        100 * _dm_minus.ewm(alpha=1 / 14, adjust=False).mean()
        / (_atr14 + 1e-9)
    )
    _dx = (
        100 * (_di_plus - _di_minus).abs()
        / (_di_plus + _di_minus + 1e-9)
    )

    df['adx'] = _dx.ewm(alpha=1 / 14, adjust=False).mean()
    df['di_plus'] = _di_plus
    df['di_minus'] = _di_minus
    df['di_diff'] = _di_plus - _di_minus

    # ── Wick & Candle Indicators ─────────────────────────────
    df['candle_dir'] = np.sign(df['close'] - df['open'])
    body_top = df[['open', 'close']].max(axis=1)
    body_bottom = df[['open', 'close']].min(axis=1)
    df['upper_wick'] = (df['high'] - body_top) / close.replace(0, np.nan)
    df['lower_wick'] = (body_bottom - df['low']) / close.replace(0, np.nan)

    # ── Specialized Features (Phase 8 - Hybrid) ─────────────

    # Efficiency Ratio (Kaufman) — Trend kalitesi
    _direction = (close - close.shift(10)).abs()
    _volatility = close.diff().abs().rolling(10).sum()
    df['efficiency_ratio'] = _direction / (_volatility + 1e-9)

    # Z-Score 20 — Mean reversion mesafesi
    _sma20 = close.rolling(20).mean()
    _std20 = close.rolling(20).std()
    df['zscore_20'] = (close - _sma20) / (_std20 + 1e-9)

    # Wick Rejection Ratio
    df['wick_rejection'] = (
        (df['upper_wick'] - df['lower_wick'])
        / (df['bb_width'] + 1e-9)
    )

    # ADX Acceleration — Trend ivmesi
    df['adx_accel'] = df['adx'].diff(3)

    # ── ICT / SMC Indicators (Phase 12) ──────────────────────
    df['fvg_bull'], df['fvg_bear'] = find_fvg(df)
    df['ob_bull'], df['ob_bear'] = find_order_blocks(df)

    # MSS Detection (Son 10 bar zirve/dip kırılımı)
    df['mss_up'] = (
        close > df['high'].rolling(10).max().shift(1)
    ).astype(int)
    df['mss_down'] = (
        close < df['low'].rolling(10).min().shift(1)
    ).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Confluence Score — Çoklu sinyal birliği
# ─────────────────────────────────────────────────────────────────────────────

def confluence_score(df: pd.DataFrame) -> pd.Series:
    """
    Birden fazla indikatörün aynı yönü göstermesi durumunda puan veren skor.

    BUY yönü (her koşul +1):
      RSI < 35, MACD bullish cross, Fiyat < BB alt,
      Stoch %K < 25, EMA9 > EMA21, Volume spike

    SELL yönü (her koşul -1):
      RSI > 65, MACD bearish cross, Fiyat > BB üst,
      Stoch %K > 75, EMA9 < EMA21
    """
    score = pd.Series(0, index=df.index)

    # BUY puanları (+)
    score += (df['rsi'] < 35).astype(int)
    score += (
        (df['macd'] > df['macd_signal'])
        & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    ).astype(int)
    score += (df['close'] < df['bb_lower']).astype(int)
    score += (df['stoch_k'] < 25).astype(int)
    score += (df['ema9'] > df['ema21']).astype(int)
    score += df['vol_spike'].astype(int)

    # SELL puanları (-)
    score -= (df['rsi'] > 65).astype(int)
    score -= (
        (df['macd'] < df['macd_signal'])
        & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    ).astype(int)
    score -= (df['close'] > df['bb_upper']).astype(int)
    score -= (df['stoch_k'] > 75).astype(int)
    score -= (df['ema9'] < df['ema21']).astype(int)

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Sinyal Üretimi (Regime-Aware)
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(
    df: pd.DataFrame,
    strategy: str = "confluence",
    rsi_oversold: int = 30,
    rsi_overbought: int = 70,
    confluence_min: int = 3,
    regime: str = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Belirlenen strateji tipine göre BUY/SELL/HOLD sinyali üretir.
    Regime-Aware: Sinyal üretimini mevcut pazar rejimine göre filtreler.

    Args:
        df: İndikatörleri hesaplanmış DataFrame
        strategy: "confluence" | "rsi_only" | "macd_cross" | diğer (ema_cross)
        rsi_oversold: RSI aşırı satım seviyesi
        rsi_overbought: RSI aşırı alım seviyesi
        confluence_min: Minimum confluence skoru
        regime: Pazar rejimi ("trending" | "mean_reverting" | None)

    Returns:
        (signal_series, score_series)
        signal_series: "BUY" | "SELL" | "HOLD"
        score_series: Confluence puanı (int)
    """
    signal = pd.Series("HOLD", index=df.index)
    score = confluence_score(df)

    # --- STRATEJİ SEÇİMİ ---
    if strategy == "rsi_only":
        signal[df['rsi'] < rsi_oversold] = "BUY"
        signal[df['rsi'] > rsi_overbought] = "SELL"

    elif strategy == "macd_cross":
        bullish_cross = (
            (df['macd'] > df['macd_signal'])
            & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        bearish_cross = (
            (df['macd'] < df['macd_signal'])
            & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        )
        signal[bullish_cross] = "BUY"
        signal[bearish_cross] = "SELL"

    elif strategy == "confluence":
        signal[score >= confluence_min] = "BUY"
        signal[score <= -confluence_min] = "SELL"

    else:
        # Varsayılan: EMA cross
        signal[df['ema_cross'] == 1] = "BUY"
        signal[df['ema_cross'] == -1] = "SELL"

    # ── REGIME-AWARE FILTERING ───────────────────────────────
    if regime == "trending":
        # Trend rejiminde: Sadece trend yönündeki sinyalleri kabul et
        signal[(signal == "BUY") & (df['close'] < df['ema50'])] = "HOLD"
        signal[(signal == "SELL") & (df['close'] > df['ema50'])] = "HOLD"

    elif regime == "mean_reverting":
        # Yatay rejimde: RSI orta bölgedeyse sinyali zayıflat
        signal[(signal == "BUY") & (df['rsi'] > 45)] = "HOLD"
        signal[(signal == "SELL") & (df['rsi'] < 55)] = "HOLD"

    return signal, score
