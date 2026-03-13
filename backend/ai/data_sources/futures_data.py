"""
Futures Data Source — Binance Perpetual Futures Market Data
Funding Rate, Open Interest, Long/Short Ratio çeker.
Bu veriler kripto piyasasının en güçlü öncü sinyalleridir.
"""
import ccxt
import pandas as pd
import numpy as np
import time
from typing import Optional


_exchange_cache = None

def _get_exchange():
    global _exchange_cache
    if _exchange_cache is None:
        _exchange_cache = ccxt.binanceusdm({
            'enableRateLimit': True,
            'timeout': 15000,
            'options': {'defaultType': 'future'},
        })
    return _exchange_cache


def _ccxt_symbol(symbol: str) -> str:
    """'BTC/USDT' → 'BTC/USDT:USDT' (Binance perp format)"""
    if ':' not in symbol:
        base, quote = symbol.split('/')
        return f"{base}/{quote}:{quote}"
    return symbol


def fetch_funding_rate(symbol: str) -> Optional[float]:
    """
    Anlık funding rate çeker.
    Pozitif → longlar short'lara ödüyor (bearish eğilim)
    Negatif → short'lar longlara ödüyor (bullish eğilim)
    """
    try:
        ex  = _get_exchange()
        sym = _ccxt_symbol(symbol)
        fr  = ex.fetch_funding_rate(sym)
        return float(fr.get('fundingRate', 0.0))
    except Exception:
        return 0.0


def fetch_funding_rate_history(symbol: str, limit: int = 200) -> pd.Series:
    """
    Geçmiş funding rate'leri döndürür (8 saatlik).
    model eğitimi için zaman serisi feature üretiminde kullanılır.
    """
    try:
        ex  = _get_exchange()
        sym = _ccxt_symbol(symbol)
        history = ex.fetch_funding_rate_history(sym, limit=limit)
        if not history:
            return pd.Series(dtype=float)
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()
        return df['fundingRate'].astype(float)
    except Exception:
        return pd.Series(dtype=float)


def fetch_open_interest(symbol: str) -> float:
    """
    Anlık açık pozisyon büyüklüğü (USDT cinsinden).
    Artış + fiyat düşüşü → short baskısı.
    Artış + fiyat artışı → güçlü trend.
    """
    try:
        ex  = _get_exchange()
        sym = _ccxt_symbol(symbol)
        oi  = ex.fetch_open_interest(sym)
        return float(oi.get('openInterestValue', oi.get('openInterest', 0)))
    except Exception:
        return 0.0


def fetch_long_short_ratio(symbol: str) -> float:
    """
    Global long/short oranı.
    > 1.5 → aşırı long → potansiyel short squeeze
    < 0.7 → aşırı short → potansiyel long squeeze
    """
    try:
        ex  = _get_exchange()
        # Binance-spesifik endpoint
        base = symbol.split('/')[0]
        params = {'symbol': f'{base}USDT', 'period': '5m', 'limit': 1}
        data   = ex.fapiPublicGetGlobalLongShortAccountRatio(params)
        if data and len(data) > 0:
            return float(data[0].get('longShortRatio', 1.0))
        return 1.0
    except Exception:
        return 1.0


def enrich_ohlcv_with_futures(
    df: pd.DataFrame,
    symbol: str,
    silent: bool = True,
) -> pd.DataFrame:
    """
    OHLCV DataFrame'ini funding rate ve OI ile zenginleştirir.
    Eğitim verisi için bulk çekim (gerçek zaman serisi).
    Live tahmin için anlık spot değerler kullanılır.
    """
    df = df.copy()

    try:
        # Funding rate history — 8 saatlik → 15m'e resample et
        fr_series = fetch_funding_rate_history(symbol, limit=500)
        if not fr_series.empty:
            # 15m frame'e reindex + forward-fill
            fr_resampled = fr_series.reindex(df.index, method='ffill')
            df['funding_rate'] = fr_resampled.fillna(0.0)
            df['funding_rate_ma8'] = df['funding_rate'].rolling(8, min_periods=1).mean()
            df['funding_rate_trend'] = df['funding_rate'] - df['funding_rate'].shift(3)
        else:
            df['funding_rate']       = 0.0
            df['funding_rate_ma8']   = 0.0
            df['funding_rate_trend'] = 0.0

        # Anlık OI ve L/S ratio (son 1 değer)
        oi_val   = fetch_open_interest(symbol)
        ls_val   = fetch_long_short_ratio(symbol)
        df['open_interest_norm'] = oi_val / (oi_val + 1e-8)  # normalize
        df['long_short_ratio']   = ls_val

        if not silent:
            print(f"  [NET] Futures data: FR={df['funding_rate'].iloc[-1]:.5f} "
                  f"OI={oi_val:.0f} L/S={ls_val:.2f}")

    except Exception as e:
        if not silent:
            print(f"  [WARN] Futures data error ({symbol}): {e}")
        # Fallback: sıfır doldur
        for col in ['funding_rate', 'funding_rate_ma8', 'funding_rate_trend',
                    'open_interest_norm', 'long_short_ratio']:
            df[col] = 0.0

    return df
