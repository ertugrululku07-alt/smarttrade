"""
ICT/SMC v2.5 Quick Backtest Endpoint
Custom coin/timeframe backtesting with new filters
"""
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def simulate_v25_entry_filters(df: pd.DataFrame, idx: int, direction: str = 'LONG') -> tuple:
    """
    Simulate v2.5 entry filters at specific candle
    Returns: (allowed: bool, reject_reasons: List[str], metrics: Dict)
    """
    reject_reasons = []
    metrics = {}
    
    df_slice = df.iloc[:idx+1].copy()
    
    if len(df_slice) < 50:
        return False, ["Insufficient data"], {}
    
    cp = float(df_slice['close'].iloc[-1])
    
    # Filter 1: HTF Resistance Proximity (5%)
    htf_lookback = min(100, len(df_slice))
    htf_slice = df_slice.iloc[-htf_lookback:]
    htf_high = float(htf_slice['high'].max())
    htf_low = float(htf_slice['low'].min())
    
    metrics['htf_high'] = htf_high
    metrics['htf_low'] = htf_low
    
    if direction == 'LONG':
        distance_from_high = (htf_high - cp) / htf_high
        metrics['distance_from_high_pct'] = distance_from_high * 100
        
        if distance_from_high < 0.05:
            reject_reasons.append(f"Too close to resistance ({distance_from_high*100:.1f}% < 5%)")
    
    # Filter 2: Pullback Requirement (3%)
    if len(df_slice) >= 10:
        recent_high_10bars = float(df_slice.iloc[-10:]['high'].max())
        pullback_pct = (recent_high_10bars - cp) / recent_high_10bars
        
        metrics['recent_high_10bars'] = recent_high_10bars
        metrics['pullback_pct'] = pullback_pct * 100
        
        if direction == 'LONG' and pullback_pct < 0.03:
            reject_reasons.append(f"No pullback ({pullback_pct*100:.1f}% < 3%)")
    
    # Filter 3: Premium Zone (70%)
    zone_lookback = min(30, len(df_slice))
    recent_slice = df_slice.iloc[-zone_lookback:]
    range_high = float(recent_slice['high'].max())
    range_low = float(recent_slice['low'].min())
    
    if range_high > range_low:
        price_position = (cp - range_low) / (range_high - range_low)
        metrics['price_position_pct'] = price_position * 100
        
        if direction == 'LONG' and price_position > 0.70:
            reject_reasons.append(f"Premium zone ({price_position*100:.0f}% > 70%)")
    
    # Filter 4: RSI Overbought
    if 'rsi' in df_slice.columns:
        rsi_val = float(df_slice['rsi'].iloc[-1])
        metrics['rsi'] = rsi_val
        
        if not np.isnan(rsi_val):
            if direction == 'LONG' and rsi_val > 70:
                reject_reasons.append(f"RSI overbought ({rsi_val:.1f} > 70)")
    
    allowed = len(reject_reasons) == 0
    return allowed, reject_reasons, metrics

def quick_backtest_ict(
    symbol: str,
    timeframe: str = '1h',
    days: int = 7,
    direction: str = 'LONG'
) -> Dict:
    """
    Quick ICT/SMC v2.5 backtest for any coin
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT', 'BANANAS31/USDT')
        timeframe: Candle timeframe ('1h', '4h', '1d')
        days: Number of days to backtest
        direction: 'LONG' or 'SHORT'
    
    Returns:
        Dict with backtest results
    """
    fetcher = DataFetcher('binance')
    
    # Calculate limit based on timeframe and days
    candles_per_day = {
        '1h': 24,
        '4h': 6,
        '1d': 1,
        '15m': 96,
        '5m': 288
    }
    
    limit = candles_per_day.get(timeframe, 24) * days
    limit = min(limit, 1000)  # Max 1000 candles
    
    # Fetch data
    df = fetcher.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    if df is None or len(df) < 50:
        return {
            'success': False,
            'error': 'Failed to fetch data or insufficient candles',
            'symbol': symbol,
            'timeframe': timeframe
        }
    
    df = add_all_indicators(df)
    
    # Scan for valid entries
    valid_entries = []
    rejected_entries = []
    
    for i in range(100, len(df)):
        allowed, reasons, metrics = simulate_v25_entry_filters(df, i, direction)
        
        entry_data = {
            'index': i,
            'timestamp': str(df.index[i]),
            'price': float(df.iloc[i]['close']),
            'metrics': metrics,
            'reject_reasons': reasons
        }
        
        if allowed:
            valid_entries.append(entry_data)
        else:
            rejected_entries.append(entry_data)
    
    # Find most recent entry attempt (last 20 candles)
    recent_entry = None
    for i in range(len(df)-20, len(df)):
        if i < 100:
            continue
        allowed, reasons, metrics = simulate_v25_entry_filters(df, i, direction)
        recent_entry = {
            'timestamp': str(df.index[i]),
            'price': float(df.iloc[i]['close']),
            'allowed': allowed,
            'reject_reasons': reasons,
            'metrics': metrics
        }
        if allowed or len(reasons) > 0:
            break
    
    return {
        'success': True,
        'symbol': symbol,
        'timeframe': timeframe,
        'days': days,
        'direction': direction,
        'total_candles': len(df),
        'date_range': {
            'start': str(df.index[0]),
            'end': str(df.index[-1])
        },
        'valid_entries_count': len(valid_entries),
        'rejected_entries_count': len(rejected_entries),
        'valid_entries': valid_entries[-10:],  # Last 10 valid
        'rejected_entries': rejected_entries[-10:],  # Last 10 rejected
        'recent_entry': recent_entry,
        'current_price': float(df.iloc[-1]['close']),
        'current_rsi': float(df.iloc[-1]['rsi']) if 'rsi' in df.columns else None
    }
