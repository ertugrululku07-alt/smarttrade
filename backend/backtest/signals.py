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


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP — Volume Weighted Average Price.
    Intraday kurumsal referans fiyat. Fiyat VWAP ustundeyse bullish,
    altindaysa bearish baski var demektir.
    """
    typical = (df['high'] + df['low'] + df['close']) / 3
    cum_tp_vol = (typical * df['volume']).cumsum()
    cum_vol = df['volume'].cumsum().replace(0, np.nan)
    return cum_tp_vol / cum_vol


def cumulative_volume_delta(df: pd.DataFrame) -> pd.Series:
    """
    CVD — Cumulative Volume Delta (Tahmini).
    Her mumda alici / satici hacmini tahmin eder.
    Bullish mum -> volume pozitif, bearish -> negatif.
    Wick oranina gore agirliklandirilir.
    """
    body = (df['close'] - df['open']).abs()
    full_range = (df['high'] - df['low']).replace(0, np.nan)
    # Mumun ne kadarinin alici tarafinda oldugu
    if 'close' in df.columns and 'low' in df.columns:
        buy_ratio = (df['close'] - df['low']) / full_range
    else:
        buy_ratio = 0.5
    buy_ratio = buy_ratio.clip(0, 1).fillna(0.5)
    delta = df['volume'] * (2 * buy_ratio - 1)
    return delta.cumsum()


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    OBV — On Balance Volume.
    Fiyat yukseldigi barlarda hacim eklenir, dustugu barlarda cikarilir.
    Gizli talep/arz tespiti.
    """
    direction = np.sign(close.diff()).fillna(0)
    return (volume * direction).cumsum()


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

    # [OK] Fix: Hangisi büyükse o geçerli, diğeri sıfırlanır
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

    # ── ATR Rank (Regime Detector icin) ──────────────────────
    _atr_series = df['atr']
    df['atr_rank_50'] = _atr_series.rolling(50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # ── VWAP & Distance ──────────────────────────────────────
    df['vwap'] = vwap(df)
    df['vwap_dist'] = (close - df['vwap']) / df['vwap'].replace(0, np.nan)

    # ── CVD (Cumulative Volume Delta) ────────────────────────
    df['cvd'] = cumulative_volume_delta(df)

    # ── OBV (On Balance Volume) ──────────────────────────────
    df['obv'] = on_balance_volume(close, vol)

    # ── BOS Detection (Break of Structure) ───────────────────
    df['bos_bullish'] = (
        close > df['high'].rolling(20, min_periods=5).max().shift(1)
    ).astype(int)
    df['bos_bearish'] = (
        close < df['low'].rolling(20, min_periods=5).min().shift(1)
    ).astype(int)

    # ── ICT / SMC Indicators (Phase 12) ──────────────────────
    df['fvg_bull'], df['fvg_bear'] = find_fvg(df)
    df['ob_bull'], df['ob_bear'] = find_order_blocks(df)

    # MSS Detection (Son 10 bar zirve/dip kirilimi)
    df['mss_up'] = (
        close > df['high'].rolling(10).max().shift(1)
    ).astype(int)
    df['mss_down'] = (
        close < df['low'].rolling(10).min().shift(1)
    ).astype(int)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# v2.0 — META-CONTEXT FEATURES
#
# Bu feature'lar sinyal üretiminde KULLANILMAZ.
# Sadece meta-model'e "kontekst" bilgi sağlar.
# Döngüsel feature sorununu çözer.
# ═════════════════════════════════════════════════════════════════════════════

def add_meta_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Meta-labeling kontekst feature'ları.

    Kategoriler:
      1. Zaman        : hour_sin/cos, dow_sin/cos
      2. Rejim        : trend_duration, vol_regime_ratio, regime_age
      3. Diverjans    : price_vol_div, rsi_price_div, roc_div
      4. Kalite       : move_cleanliness, trend_consistency, momentum_quality
      5. Pozisyon     : price_position_50, range_position
      6. Volume       : vol_relative_50, vol_trend, vol_climax
      7. Candle       : body_ratio, is_doji, upper/lower_wick_ratio
      8. Ardışıklık   : consec_direction, consec_magnitude
      9. Mesafe       : dist_ema200_atr, dist_ema50_atr, dist_vwap_atr
     10. Mikro yapı   : bar_range_vs_atr, spread_estimate
     11. İnteraksiyon : vol_x_volatility, trend_x_clean
    """
    df = df.copy()

    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # ATR referans
    if 'atr' in df.columns:
        atr_ref = df['atr']
    else:
        atr_ref = atr(high, low, close, 14)

    # ═══════════════════════════════════════════════
    # 1. ZAMAN PATTERN'LERİ
    # ═══════════════════════════════════════════════
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    else:
        df['hour_sin'] = 0.0
        df['hour_cos'] = 1.0
        df['dow_sin'] = 0.0
        df['dow_cos'] = 1.0

    # ═══════════════════════════════════════════════
    # 2. REJİM KONTEKST
    # ═══════════════════════════════════════════════

    # Trend süresi (bar sayısı)
    if 'ema50' in df.columns:
        _above = (close > df['ema50']).astype(int)
    else:
        _above = (close > ema(close, 50)).astype(int)

    _groups = (_above != _above.shift()).cumsum()
    df['trend_duration'] = _above.groupby(_groups).cumcount() + 1

    # Volatilite rejim oranı
    atr_fast = atr_ref.rolling(5, min_periods=2).mean()
    atr_slow = atr_ref.rolling(50, min_periods=10).mean()
    df['vol_regime_ratio'] = atr_fast / atr_slow.replace(0, np.nan)

    # Rejim yaşı
    _vol_expanding = (atr_fast > atr_slow).astype(int)
    _vol_groups = (_vol_expanding != _vol_expanding.shift()).cumsum()
    df['regime_age'] = _vol_expanding.groupby(_vol_groups).cumcount() + 1

    # ═══════════════════════════════════════════════
    # 3. DİVERJANS
    # ═══════════════════════════════════════════════

    # Fiyat-hacim diverjansı
    price_chg_5 = close.pct_change(5)
    vol_chg_5 = volume.pct_change(5)
    df['price_vol_divergence'] = (
        np.sign(price_chg_5) - np.sign(vol_chg_5)
    ).fillna(0)

    # RSI-Fiyat diverjansı
    if 'rsi' in df.columns:
        price_up_10 = (close > close.shift(10)).astype(float)
        rsi_up_10 = (df['rsi'] > df['rsi'].shift(10)).astype(float)
        df['rsi_price_divergence'] = price_up_10 - rsi_up_10
    else:
        df['rsi_price_divergence'] = 0.0

    # ROC diverjansı
    roc_5 = close.pct_change(5)
    roc_20 = close.pct_change(20)
    df['roc_divergence'] = (
        np.sign(roc_5) - np.sign(roc_20)
    ).fillna(0)

    # ═══════════════════════════════════════════════
    # 4. HAREKET KALİTESİ
    # ═══════════════════════════════════════════════

    # Move cleanliness
    returns_20 = close.pct_change(20)
    vol_20 = close.pct_change().rolling(20).std()
    df['move_cleanliness'] = returns_20.abs() / (vol_20 * np.sqrt(20) + 1e-9)

    # Trend consistency
    bar_dir = np.sign(close.pct_change())
    trend_dir = np.sign(returns_20)
    same_dir = (bar_dir == trend_dir).astype(float)
    df['trend_consistency'] = same_dir.rolling(20, min_periods=5).mean()

    # Momentum quality
    mom_5 = close.pct_change(5)
    mom_5_prev = mom_5.shift(5)
    df['momentum_quality'] = (
        np.sign(mom_5) * (mom_5.abs() - mom_5_prev.abs())
    ).fillna(0)

    # ═══════════════════════════════════════════════
    # 5. FİYAT POZİSYONU
    # ═══════════════════════════════════════════════

    # 50-bar range pozisyonu
    rh_50 = high.rolling(50, min_periods=10).max()
    rl_50 = low.rolling(50, min_periods=10).min()
    range_50 = (rh_50 - rl_50).replace(0, np.nan)
    df['price_position_50'] = (close - rl_50) / range_50

    # 200-bar range pozisyonu
    rh_200 = high.rolling(200, min_periods=50).max()
    rl_200 = low.rolling(200, min_periods=50).min()
    range_200 = (rh_200 - rl_200).replace(0, np.nan)
    df['range_position'] = (close - rl_200) / range_200

    # ═══════════════════════════════════════════════
    # 6. VOLUME PROFİLİ
    # ═══════════════════════════════════════════════

    vol_med_50 = volume.rolling(50, min_periods=10).median()
    df['vol_relative_50'] = volume / vol_med_50.replace(0, np.nan)

    vol_sma5 = volume.rolling(5, min_periods=2).mean()
    vol_sma20 = volume.rolling(20, min_periods=5).mean()
    df['vol_trend'] = vol_sma5 / vol_sma20.replace(0, np.nan)

    vol_max50 = volume.rolling(50, min_periods=10).max()
    df['vol_climax'] = volume / vol_max50.replace(0, np.nan)

    # ═══════════════════════════════════════════════
    # 7. CANDLE YAPISI
    # ═══════════════════════════════════════════════

    body = (close - open_).abs()
    full_range = (high - low).replace(0, np.nan)
    df['body_ratio'] = body / full_range

    df['is_doji'] = (df['body_ratio'] < 0.1).astype(int)

    body_top = pd.concat([open_, close], axis=1).max(axis=1)
    body_bot = pd.concat([open_, close], axis=1).min(axis=1)
    df['upper_wick_ratio'] = (high - body_top) / atr_ref.replace(0, np.nan)
    df['lower_wick_ratio'] = (body_bot - low) / atr_ref.replace(0, np.nan)

    # ═══════════════════════════════════════════════
    # 8. ARDIŞIKLIK
    # ═══════════════════════════════════════════════

    candle_dir = np.sign(close - open_)
    cd_groups = (candle_dir != candle_dir.shift()).cumsum()
    consec_count = candle_dir.groupby(cd_groups).cumcount() + 1
    df['consec_direction'] = consec_count * candle_dir

    # Ardışık hareketin büyüklüğü
    _shift_n = consec_count.astype(int).clip(1, 20)
    df['consec_magnitude'] = 0.0
    for n in range(1, 21):
        mask = _shift_n == n
        if mask.any():
            df.loc[mask, 'consec_magnitude'] = (
                (close - close.shift(n)) / atr_ref.replace(0, np.nan)
            )[mask]
    df['consec_magnitude'] = df['consec_magnitude'].clip(-10, 10).fillna(0)

    # ═══════════════════════════════════════════════
    # 9. EMA MESAFELERİ (ATR normalize)
    # ═══════════════════════════════════════════════

    _ema200 = df['ema200'] if 'ema200' in df.columns else ema(close, 200)
    _ema50 = df['ema50'] if 'ema50' in df.columns else ema(close, 50)

    df['dist_ema200_atr'] = (close - _ema200) / atr_ref.replace(0, np.nan)
    df['dist_ema50_atr'] = (close - _ema50) / atr_ref.replace(0, np.nan)

    if 'vwap' in df.columns:
        df['dist_vwap_atr'] = (close - df['vwap']) / atr_ref.replace(0, np.nan)
    else:
        df['dist_vwap_atr'] = 0.0

    # ═══════════════════════════════════════════════
    # 10. MİKRO YAPI
    # ═══════════════════════════════════════════════

    df['bar_range_vs_atr'] = (high - low) / atr_ref.replace(0, np.nan)
    df['spread_estimate'] = (high - low) / close.replace(0, np.nan) * 10000

    # ═══════════════════════════════════════════════
    # 11. CROSS-FEATURE İNTERAKSİYONLAR
    # ═══════════════════════════════════════════════

    df['vol_x_volatility'] = (
        df['vol_relative_50'].fillna(1) *
        df['vol_regime_ratio'].fillna(1)
    )

    _max_td = df['trend_duration'].max()
    _td_norm = df['trend_duration'] / max(_max_td, 1)
    df['trend_x_clean'] = _td_norm * df['move_cleanliness'].fillna(0)

    # ═══════════════════════════════════════════════
    # 12. MEAN REVERSION FEATURES (v3.0)
    # ═══════════════════════════════════════════════

    # BB squeeze: düşük BB width = sıkışma, patlama öncesi
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        bb_width = (df['bb_upper'] - df['bb_lower']) / close.replace(0, np.nan)
        bb_width_ma = bb_width.rolling(50, min_periods=10).mean()
        df['bb_squeeze'] = bb_width / bb_width_ma.replace(0, np.nan)
    else:
        df['bb_squeeze'] = 1.0

    # BB position z-score: ne kadar extreme?
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        bb_mid = (df['bb_upper'] + df['bb_lower']) / 2
        bb_half = (df['bb_upper'] - df['bb_lower']) / 2
        df['bb_position_z'] = (close - bb_mid) / bb_half.replace(0, np.nan)
    else:
        df['bb_position_z'] = 0.0

    # RSI slope: son 5 bar RSI eğimi (reversal timing)
    if 'rsi' in df.columns:
        df['rsi_slope_5'] = (df['rsi'] - df['rsi'].shift(5)) / 5
    else:
        df['rsi_slope_5'] = 0.0

    # RSI divergence derinliği
    if 'rsi' in df.columns:
        price_chg_10 = close.pct_change(10)
        rsi_chg_10 = df['rsi'] - df['rsi'].shift(10)
        df['rsi_divergence_depth'] = (
            np.sign(price_chg_10) * (-rsi_chg_10 / 100)
        ).fillna(0).clip(-1, 1)
    else:
        df['rsi_divergence_depth'] = 0.0

    # Support/Resistance proximity (ATR normalize)
    swing_high_20 = high.rolling(20, min_periods=5).max()
    swing_low_20 = low.rolling(20, min_periods=5).min()
    dist_to_res = (swing_high_20 - close) / atr_ref.replace(0, np.nan)
    dist_to_sup = (close - swing_low_20) / atr_ref.replace(0, np.nan)
    df['sr_proximity'] = pd.concat([dist_to_res, dist_to_sup], axis=1).min(axis=1)

    # Swing range: son 20 bar range / fiyat
    df['swing_range_pct'] = (swing_high_20 - swing_low_20) / close.replace(0, np.nan) * 100

    # Mean reversion composite: BB extreme + RSI extreme + diverjans
    _bb_extreme = df['bb_position_z'].abs().clip(0, 2)
    _rsi_extreme = 0.0
    if 'rsi' in df.columns:
        _rsi_extreme = ((df['rsi'] - 50).abs() / 50).clip(0, 1)
    _div_signal = df['rsi_divergence_depth'].abs()
    df['mean_reversion_score'] = (_bb_extreme + _rsi_extreme + _div_signal) / 3

    # ═══════════════════════════════════════════════
    # 13. VOLATILITY REGIME FEATURES (v3.0)
    # ═══════════════════════════════════════════════

    # Volatilite genişleme hızı
    atr_chg_5 = atr_ref.pct_change(5)
    df['vol_expansion_rate'] = atr_chg_5.clip(-2, 2).fillna(0)

    # Vol contraction: sıkışma tespit
    atr_pctile = atr_ref.rolling(50, min_periods=10).rank(pct=True)
    df['vol_contraction'] = (1 - atr_pctile).fillna(0.5)

    # High vol sürekliliği (bar sayısı)
    _high_vol = (atr_ref > atr_ref.rolling(20, min_periods=5).mean()).astype(int)
    _hv_groups = (_high_vol != _high_vol.shift()).cumsum()
    df['high_vol_persistence'] = _high_vol.groupby(_hv_groups).cumcount() + 1

    # Regime transition: vol_regime_ratio'nun değişim hızı
    df['regime_transition_signal'] = df['vol_regime_ratio'].pct_change(3).clip(-2, 2).fillna(0)

    # Price acceleration (2. türev — inflection point detection)
    mom_5 = close.pct_change(5)
    mom_5_prev = mom_5.shift(5)
    df['price_acceleration'] = (mom_5 - mom_5_prev).clip(-0.1, 0.1).fillna(0)

    # ═══════════════════════════════════════════════
    # CLEANUP — inf temizliği
    # ═══════════════════════════════════════════════

    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

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
