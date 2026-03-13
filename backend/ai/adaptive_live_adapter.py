"""
Adaptive Live Adapter v1.1

v1.1: Feature hazırlama eklendi
  - Ham OHLCV gelirse otomatik indikatör + feature hesaplar
  - Debug logging eklendi
  - Hata yakalama güçlendirildi
"""

import os
import sys
import traceback
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.adaptive_engine import AdaptiveEngine, TradeDecision

# Engine cache
_engines: Dict[str, AdaptiveEngine] = {}

# Sinyal sayacı (debug)
_signal_stats = {
    'total_calls': 0,
    'feature_added': 0,
    'signals_generated': 0,
    'errors': 0,
    'holds': 0,
}


def _get_engine(primary_tf: str = "1h", secondary_tf: str = "15m") -> AdaptiveEngine:
    key = f"{primary_tf}_{secondary_tf}"
    if key not in _engines:
        _engines[key] = AdaptiveEngine(primary_tf=primary_tf, secondary_tf=secondary_tf)
    return _engines[key]


def _ensure_features(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """
    DataFrame'de gerekli feature'lar yoksa hesapla.
    
    Bot ham OHLCV gönderebilir — bu fonksiyon eksikleri tamamlar.
    """
    required_indicators = ['rsi', 'ema9', 'ema21', 'macd_hist', 'adx', 'atr', 'bb_upper']
    required_features = ['rsi_z', 'dist_ema9', 'atr_pct', 'bb_pos', 'vol_ratio_20']
    
    has_indicators = all(col in df.columns for col in required_indicators)
    has_features = all(col in df.columns for col in required_features)
    
    if has_indicators and has_features:
        return df
    
    # İndikatörler eksik → hesapla
    if not has_indicators:
        try:
            from backtest.signals import add_all_indicators
            df = add_all_indicators(df)
            _signal_stats['feature_added'] += 1
        except Exception as e:
            print(f"  ERROR: Indicator hesaplama hatasi ({symbol}): {e}")
            return df
    
    # Feature'lar eksik → hesapla
    if not has_features:
        try:
            from ai.xgboost_trainer import generate_features
            df = generate_features(df)
        except Exception as e:
            print(f"  ERROR: Feature hesaplama hatasi ({symbol}): {e}")
            # generate_features başarısız olsa bile indikatörler var
            # Stratejiler çalışabilir

    # Meta-context features (meta-predictor için)
    if 'trend_duration' not in df.columns:
        try:
            from backtest.signals import add_meta_context_features
            df = add_meta_context_features(df)
        except Exception:
            pass
    
    # Futures data (opsiyonel, hata verirse geç)
    if 'funding_rate' not in df.columns:
        try:
            from ai.data_sources.futures_data import enrich_ohlcv_with_futures
            df = enrich_ohlcv_with_futures(df, symbol, silent=True)
        except Exception:
            pass
    
    return df


def generate_signal(
    df: pd.DataFrame,
    df_secondary: Optional[pd.DataFrame] = None,
    df_4h: Optional[pd.DataFrame] = None,
    symbol: str = "UNKNOWN",
    timeframe: str = "1h",
    secondary_tf: str = "15m",
) -> Dict:
    """
    Mevcut bot ile uyumlu sinyal üretici.
    Hybrid Mod v1.5: 1H primary ve Opsiyonel 15m secondary destekler.
    """
    _signal_stats['total_calls'] += 1
    
    try:
        # ── v1.1: Feature kontrolü ve hazırlama ─────────────
        if len(df) < 50:
            _signal_stats['holds'] += 1
            return _hold_response(f"Yetersiz veri: {len(df)} bar < 50")
        
        df = _ensure_features(df, symbol)
        
        # ── Temel kolon kontrolü ─────────────────────────────
        critical_cols = ['close', 'high', 'low', 'open', 'volume']
        missing = [c for c in critical_cols if c not in df.columns]
        if missing:
            _signal_stats['errors'] += 1
            return _hold_response(f"Eksik kolonlar: {missing}")
        
        # ── Engine'e gönder (Hybrid) ──────────────────────────
        engine = _get_engine(timeframe, secondary_tf)
        decision = engine.decide(df, df_secondary=df_secondary, df_4h=df_4h, symbol=symbol)
        
        if decision.action in ('LONG', 'SHORT'):
            _signal_stats['signals_generated'] += 1
            
            # Debug log
            print(f"  TARGET: [{symbol}] {decision.action} | "
                  f"regime={decision.regime} | "
                  f"conf={decision.confidence:.2f} | "
                  f"meta={decision.meta_confidence:.2f} | "
                  f"size={decision.position_size:.0%} | "
                  f"{decision.reason}")
        else:
            _signal_stats['holds'] += 1
        
        return {
            'signal': decision.action,
            'confidence': round(decision.confidence, 4),
            'position_size': round(decision.position_size, 2),
            'tp_mult': decision.tp_atr_mult,
            'sl_mult': decision.sl_atr_mult,
            'regime': decision.regime,
            'strategy': (decision.primary_signal.strategy_name
                         if decision.primary_signal else 'none'),
            'meta_confidence': round(decision.meta_confidence, 4),
            'reason': decision.reason,
            'soft_score': decision.soft_score,
            'entry_type': decision.entry_type,
            'tp_price': decision.tp_price,
            'sl_price': decision.sl_price,
        }

    except Exception as e:
        _signal_stats['errors'] += 1
        error_detail = traceback.format_exc()
        # print(f"ERROR: Adaptive signal error ({symbol}): {e}\n{error_detail}")
        return _hold_decision_with_error(f"Signal Generation Error: {str(e)}")

def _hold_decision_with_error(error_msg: str) -> Dict:
    resp = _hold_response(error_msg)
    resp['strategy'] = 'error'
    return resp


def _hold_response(reason: str = "") -> Dict:
    """Standart HOLD response"""
    return {
        'signal': 'HOLD',
        'confidence': 0.0,
        'position_size': 0.0,
        'tp_mult': 2.0,
        'sl_mult': 1.3,
        'regime': 'unknown',
        'strategy': 'none',
        'meta_confidence': 0.0,
        'reason': reason,
    }


def get_system_status(timeframe: str = "15m") -> Dict:
    """Sistem durumunu döndür"""
    try:
        engine = _get_engine(timeframe)
        status = engine.get_status()
        
        meta_stats = {}
        if engine.meta_predictor:
            meta_stats = engine.meta_predictor.get_regime_stats()
        
        return {
            'status': 'ready',
            'timeframe': timeframe,
            'engine': status,
            'meta_models': meta_stats,
            'signal_stats': _signal_stats.copy(),
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def get_signal_stats() -> Dict:
    """Sinyal istatistiklerini döndür (debug)"""
    stats = _signal_stats.copy()
    total = stats['total_calls']
    if total > 0:
        stats['signal_rate'] = f"%{stats['signals_generated']/total*100:.1f}"
        stats['error_rate'] = f"%{stats['errors']/total*100:.1f}"
        stats['hold_rate'] = f"%{stats['holds']/total*100:.1f}"
    return stats


def print_debug():
    """Debug bilgilerini yazdır"""
    stats = get_signal_stats()
    print(f"\n  {'-' * 50}")
    print(f"  ADAPTIVE LIVE ADAPTER STATS")
    print(f"  {'-' * 50}")
    for k, v in stats.items():
        print(f"  {k:20}: {v}")
    print(f"  {'─' * 50}")
    
    # Engine debug stats
    for tf, engine in _engines.items():
        print(f"\n  Engine ({tf}):")
        engine.print_debug_stats()


# ═══════════════════════════════════════════════════════════════════
# Mevcut bot entegrasyon yardımcıları
# ═══════════════════════════════════════════════════════════════════

def should_open_position(signal_result: Dict, min_confidence: float = 0.60) -> bool:
    """Pozisyon açılmalı mı? v1.5 Alpha: threshold 0.60 (Yüksek Hassasiyet)"""
    return (
        signal_result['signal'] in ('LONG', 'SHORT')
        and signal_result['confidence'] >= min_confidence
        and signal_result['position_size'] > 0
    )


def get_tp_sl_prices(
    signal_result: Dict,
    entry_price: float,
    atr: float,
) -> Tuple[float, float]:
    """TP ve SL fiyatlarını hesapla"""
    tp_mult = signal_result['tp_mult']
    sl_mult = signal_result['sl_mult']
    
    if signal_result['signal'] == 'LONG':
        tp = entry_price + atr * tp_mult
        sl = entry_price - atr * sl_mult
    elif signal_result['signal'] == 'SHORT':
        tp = entry_price - atr * tp_mult
        sl = entry_price + atr * sl_mult
    else:
        tp = entry_price
        sl = entry_price
    
    return tp, sl


def calculate_position_amount(
    signal_result: Dict,
    total_capital: float,
    risk_per_trade: float = 0.02,
    entry_price: float = 0.0,
    atr: float = 0.0,
) -> float:
    """Position sizing: Risk-based + Kelly confidence"""
    if not should_open_position(signal_result):
        return 0.0
    
    sl_mult = signal_result['sl_mult']
    position_size_factor = signal_result['position_size']
    
    risk_amount = total_capital * risk_per_trade
    sl_distance = atr * sl_mult
    
    if sl_distance <= 0 or entry_price <= 0:
        return 0.0
    
    base_amount = risk_amount / sl_distance
    adjusted_amount = base_amount * position_size_factor
    max_amount = (total_capital * 0.10) / entry_price
    final_amount = min(adjusted_amount, max_amount)
    
    return round(final_amount, 6)
