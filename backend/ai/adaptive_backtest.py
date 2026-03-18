"""
Adaptive Regime + Meta-Label Backtest Engine v1.0

Tüm sistemi geçmiş veri üzerinde test eder:
  1. Her bar'da rejim tespit
  2. Rejime uygun sinyal üret
  3. Meta-model ile filtrele
  4. TP/SL simülasyonu
  5. PnL, WR, PF, Sharpe hesapla

Kullanım:
  python -m ai.adaptive_backtest --timeframe 15m --limit 3000
"""

import os
import sys
import gc
import json
import warnings
import numpy as np
import pandas as pd
import time as _time
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators, add_meta_context_features
from ai.data_sources.futures_data import enrich_ohlcv_with_futures
from ai.xgboost_trainer import generate_features
from ai.regime_detector import Regime, detect_regime
from ai.meta_labelling.meta_predictor import MetaPredictor
from ai.adaptive_engine import AdaptiveEngine, TradeDecision


MAKER_FEE = 0.0004

DEFAULT_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT',
    'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT',
    'UNIUSDT', 'LTCUSDT', 'ATOMUSDT', 'ETCUSDT', 'BCHUSDT',
    'FTMUSDT', 'NEARUSDT', 'RUNEUSDT', 'AAVEUSDT', 'SANDUSDT',
    'GRTUSDT', 'INJUSDT', 'TIAUSDT', 'WLDUSDT',
]


# ===================================================================
# Trade Kaydi
# ===================================================================

@dataclass
class TradeRecord:
    """Tek bir trade kaydı"""
    symbol: str
    bar_index: int
    timestamp: str
    direction: str          # 'LONG' / 'SHORT'
    entry_price: float
    exit_price: float
    tp_price: float
    sl_price: float
    atr_at_entry: float
    pnl_pct: float          # Yüzde PnL
    pnl_atr: float          # ATR cinsinden PnL
    outcome: str            # 'TP' / 'SL' / 'TRAIL' / 'TIMEOUT'
    regime: str
    strategy: str
    signal_confidence: float
    meta_confidence: float
    position_size: float
    bars_held: int
    is_winner: bool
    fee_paid: float


@dataclass
class BacktestResult:
    """Backtest sonucu"""
    trades: List[TradeRecord] = field(default_factory=list)
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl_pct: float = 0.0
    total_pnl_atr: float = 0.0
    avg_pnl_per_trade: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    avg_bars_held: float = 0.0
    signals_generated: int = 0
    signals_filtered: int = 0
    filter_rate: float = 0.0
    regime_stats: Dict = field(default_factory=dict)
    strategy_stats: Dict = field(default_factory=dict)
    by_direction: Dict = field(default_factory=dict)
    equity_curve: List[float] = field(default_factory=list)


# ===================================================================
# Backtest Engine
# ===================================================================

class AdaptiveBacktest:
    """
    Adaptive Regime + Meta-Label backtest motoru.

    Özellikler:
      - Bar-by-bar simülasyon
      - Trailing stop desteği
      - Rejim bazlı strateji seçimi
      - Meta-model filtreleme
      - Position sizing (Kelly)
      - Detaylı istatistikler
    """

    def __init__(
        self,
        timeframe: str = "15m",
        trail_activation: float = 0.75,
        regime_lookback: int = 50,
        initial_capital: float = 10000.0,
        max_concurrent_trades: int = 1,
        use_meta_filter: bool = True,
        use_simple_strategy: bool = True,
    ):
        self.timeframe = timeframe
        self.trail_activation = trail_activation
        self.regime_lookback = regime_lookback
        self.initial_capital = initial_capital
        self.max_concurrent = max_concurrent_trades
        self.use_meta_filter = use_meta_filter
        self.use_simple = use_simple_strategy

        # ── v1.5: Hybrid Engine Design ──────────────────────
        self.engine = AdaptiveEngine(primary_tf=timeframe, secondary_tf="15m")
        if not use_meta_filter:
            self.engine.meta_predictor = None
        
        # Hybrid modu (Backtest bazlı)
        self.is_hybrid = False

        # TF bazlı lookahead
        self.lookahead = self._get_lookahead(timeframe)

        # -- v1.3: Debug sayaclari ----------------------------
        self._debug_counts = {
            'total_steps': 0,
            'signals_generated': 0,
            'signals_filtered': 0,
            'no_strategy': 0,
            'no_signal': 0,
            'meta_rejected': 0,
            'regime_distribution': {},
        }

    # ==================================================================
    # v3.0: Simple Trend Momentum Strategy
    # ==================================================================
    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Backtest için gerekli tüm indikatörleri hesapla."""
        c = df['close']
        # EMA'lar
        df['ema9'] = c.ewm(span=9, adjust=False).mean()
        df['ema20'] = c.ewm(span=20, adjust=False).mean()
        df['ema50'] = c.ewm(span=50, adjust=False).mean()
        # RSI
        delta = c.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        # MACD
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        # ATR
        if 'atr' not in df.columns:
            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - c.shift()).abs(),
                (df['low'] - c.shift()).abs()
            ], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14).mean()
        # Volume MA
        if 'volume' in df.columns:
            df['vol_ma'] = df['volume'].rolling(20).mean()
        # Bollinger Bands (20-period, 2 std)
        df['bb_mid'] = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2.0 * bb_std
        df['bb_lower'] = df['bb_mid'] - 2.0 * bb_std
        # BB width (squeeze detection) — ATR normalize
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['atr'] + 1e-10)
        # EMA50 slope (trend direction) — 20-bar change normalized by ATR
        df['ema50_slope'] = (df['ema50'] - df['ema50'].shift(20)) / (df['atr'] * 20 + 1e-10)
        # ADX (trend strength) — 14 period
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        tr14 = df['atr'] * 14  # ATR * period = smoothed TR
        plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / (tr14 + 1e-10))
        minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / (tr14 + 1e-10))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
        df['adx'] = dx.ewm(span=14, adjust=False).mean()
        return df

    # ==================================================================
    # v5.0: Otonom AI Karar Sistemi
    # ==================================================================
    # Piyasa 3 hareket yapar: Yükselen, Düşen, Yatay.
    # Sistem bunu algılar ve stratejisini ona göre seçer.
    # ==================================================================

    def _detect_regime(self, df: pd.DataFrame, i: int) -> str:
        """
        Sticky rejim algılama — TF'ye uyarlanmış.

        Kısa TF (15m): Daha uzun bakış, daha yüksek eşikler (gürültü filtresi)
        Uzun TF (1h+): Standart bakış

        Returns: 'TRENDING_UP', 'TRENDING_DOWN', 'RANGING'
        """
        is_short_tf = self.timeframe in ('15m', '30m', '5m', '3m', '1m')

        if is_short_tf:
            lookback = 40       # 10 saat (15m), daha stabil
            adx_thresh = 28     # Sadece güçlü trendler
            slope_thresh = 0.025
        else:
            lookback = 20
            adx_thresh = 22
            slope_thresh = 0.015

        if i < lookback + 14:
            return 'RANGING'

        adx_vals = df['adx'].iloc[i-lookback:i+1]
        slope_vals = df['ema50_slope'].iloc[i-lookback:i+1]

        avg_adx = adx_vals.mean()
        avg_slope = slope_vals.mean()

        if np.isnan(avg_adx):
            avg_adx = 20
        if np.isnan(avg_slope):
            avg_slope = 0

        if avg_adx > adx_thresh and avg_slope > slope_thresh:
            return 'TRENDING_UP'
        elif avg_adx > adx_thresh and avg_slope < -slope_thresh:
            return 'TRENDING_DOWN'
        return 'RANGING'

    def _get_tf_params(self) -> Dict:
        """
        Timeframe-spesifik parametreler — sinyal kalitesi + risk yönetimi.

        15m: Çok seçici giriş (RSI extreme, BB genişlik, volume), sıkı SL
        1h:  Orijinal kanıtlanmış parametreler
        4h:  Geniş SL, rahat giriş
        """
        params = {
            '15m': {
                'sl_mult': 1.0, 'tp_min': 1.0, 'cooldown': 6, 'sl_cooldown': 12,
                'rsi_long': 40, 'rsi_short': 60, 'bb_width_min': 0,
                'require_volume': False,
                'min_rej_score': 3, 'sl_min_atr': 0.6, 'sl_max_atr': 1.5, 'min_rr': 1.5,
            },
            '30m': {
                'sl_mult': 1.2, 'tp_min': 0.5, 'cooldown': 5, 'sl_cooldown': 10,
                'rsi_long': 42, 'rsi_short': 58, 'bb_width_min': 0,
                'require_volume': False,
                'min_rej_score': 3, 'sl_min_atr': 0.7, 'sl_max_atr': 1.8, 'min_rr': 1.5,
            },
            '1h': {
                'sl_mult': 1.5, 'tp_min': 0.5, 'cooldown': 3, 'sl_cooldown': 18,
                'rsi_long': 45, 'rsi_short': 55, 'bb_width_min': 0,
                'require_volume': False,
                'min_rej_score': 3, 'sl_min_atr': 0.8, 'sl_max_atr': 2.0, 'min_rr': 1.5,
            },
            '4h': {
                'sl_mult': 2.0, 'tp_min': 0.8, 'cooldown': 2, 'sl_cooldown': 6,
                'rsi_long': 45, 'rsi_short': 55, 'bb_width_min': 0,
                'require_volume': False,
                'min_rej_score': 3, 'sl_min_atr': 1.0, 'sl_max_atr': 2.5, 'min_rr': 1.5,
            },
        }
        return params.get(self.timeframe, params['1h'])

    def _simple_signal(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        """
        v5.0 — Otonom AI Karar Sistemi.

        Piyasa 3 hareket yapar: Yükselen, Düşen, Yatay.
        Her timeframe farklı strateji gerektirir:

        ┌──────────┬─────────────────────┬───────────────────────┐
        │ TF       │ TRENDING            │ RANGING               │
        ├──────────┼─────────────────────┼───────────────────────┤
        │ 15m/30m  │ Momentum Pullback   │ BB MR (çok seçici)    │
        │ 1h+      │ Trend-yönlü MR      │ BB MR (orijinal)      │
        └──────────┴─────────────────────┴───────────────────────┘
        """
        if i < 55:
            return None

        close = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        rsi = df['rsi'].iloc[i]

        if np.isnan(atr) or atr < 1e-10:
            return None

        # ── 1) Rejim Algıla ──
        regime = self._detect_regime(df, i)

        # ── 2) Timeframe parametreleri ──
        tfp = self._get_tf_params()

        # ── 3) Timeframe + Rejim → Strateji Seç ──
        is_short_tf = self.timeframe in ('15m', '30m', '5m', '3m', '1m')

        if is_short_tf:
            # Kısa TF: SADECE trend modda momentum. Ranging = bekle.
            # BB MR kısa TF'de çalışmaz (R:R sorunu).
            if regime in ('TRENDING_UP', 'TRENDING_DOWN'):
                trend_dir = 'LONG' if regime == 'TRENDING_UP' else 'SHORT'
                return self._signal_momentum(df, i, close, atr, rsi, tfp, trend_dir)
            return None  # 15m ranging = trade yok
        else:
            # 1h+: DAİMA orijinal BB MR — kanlı test sonucu değiştirilmez.
            return self._signal_ranging(df, i, close, atr, rsi, tfp)

    # ══════════════════════════════════════════════
    # RANGING Strateji: BB Mean Reversion (iki yön)
    # ══════════════════════════════════════════════
    def _signal_ranging(self, df, i, close, atr, rsi, tfp):
        """
        v6.0 BB Mean Reversion — Rejection bazlı giriş.
        Yenilikler: rejection candle, yapısal SL, genişletilmiş TP, hacim, squeeze.
        """
        bb_mid = df['bb_mid'].iloc[i]
        bb_upper = df['bb_upper'].iloc[i]
        bb_lower = df['bb_lower'].iloc[i]

        if np.isnan(bb_mid) or np.isnan(bb_upper) or np.isnan(bb_lower):
            return None
        if atr <= 0 or close <= 0:
            return None

        atr_pct = atr / close
        if atr_pct < 0.005:
            return None

        open_p = df['open'].iloc[i]
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        full_range = high - low

        # EMA50 trend filtresi
        trend_bias = 'flat'
        if 'ema50' in df.columns and i >= 20:
            ema50_now = df['ema50'].iloc[i]
            ema50_prev = df['ema50'].iloc[i - 20]
            if not np.isnan(ema50_now) and not np.isnan(ema50_prev) and ema50_prev > 0:
                slope = (ema50_now - ema50_prev) / ema50_prev
                if slope >= 0.015:
                    trend_bias = 'up'
                elif slope <= -0.015:
                    trend_bias = 'down'

        rsi_long = tfp.get('rsi_long', 45)
        rsi_short = tfp.get('rsi_short', 55)
        min_rej = tfp.get('min_rej_score', 3)
        direction = None
        rej_score = 0

        # ── LONG — BB Lower Rejection ──
        if rsi < rsi_long and trend_bias != 'down':
            touched = False
            for j in range(1, 4):
                if i - j < 0:
                    break
                if not np.isnan(df['bb_lower'].iloc[i - j]) and df['low'].iloc[i - j] <= df['bb_lower'].iloc[i - j]:
                    touched = True
                    break
            if touched:
                r1 = close > bb_lower
                r2 = close < bb_mid
                r3 = close > open_p
                r4 = False
                if full_range > 0:
                    r4 = (min(open_p, close) - low) / full_range > 0.4
                r5 = False
                if i >= 2:
                    r5 = close > df['close'].iloc[i-1] and df['close'].iloc[i-1] > df['close'].iloc[i-2]
                rej_score = sum([r1, r2, r3, r4, r5])
                if rej_score >= min_rej:
                    direction = 'LONG'

        # ── SHORT — BB Upper Rejection ──
        if direction is None and rsi > rsi_short and trend_bias != 'up':
            touched = False
            for j in range(1, 4):
                if i - j < 0:
                    break
                if not np.isnan(df['bb_upper'].iloc[i - j]) and df['high'].iloc[i - j] >= df['bb_upper'].iloc[i - j]:
                    touched = True
                    break
            if touched:
                r1 = close < bb_upper
                r2 = close > bb_mid
                r3 = close < open_p
                r4 = False
                if full_range > 0:
                    r4 = (high - max(open_p, close)) / full_range > 0.4
                r5 = False
                if i >= 2:
                    r5 = close < df['close'].iloc[i-1] and df['close'].iloc[i-1] < df['close'].iloc[i-2]
                rej_score = sum([r1, r2, r3, r4, r5])
                if rej_score >= min_rej:
                    direction = 'SHORT'

        if direction is None:
            return None

        # ── Hacim doğrulaması ──
        if 'volume' in df.columns and 'vol_ma' in df.columns:
            max_vol_ratio = 0.0
            for j in range(0, 3):
                if i - j >= 0:
                    vm = df['vol_ma'].iloc[i - j]
                    if not np.isnan(vm) and vm > 0:
                        vr = df['volume'].iloc[i - j] / vm
                        max_vol_ratio = max(max_vol_ratio, vr)
            if max_vol_ratio < 1.0 and rej_score < 4:
                return None  # Düşük hacim + zayıf rejection → iptal

        # ── Yapısal SL + genişletilmiş TP ──
        sl_min_atr = tfp.get('sl_min_atr', 0.8)
        sl_max_atr = tfp.get('sl_max_atr', 2.0)
        min_rr = tfp.get('min_rr', 1.5)

        if direction == 'LONG':
            sl_bb = bb_lower - atr * 0.3
            recent_low = float(df['low'].iloc[max(0, i - 10):i + 1].min())
            sl_swing = recent_low - atr * 0.2
            sl_price = max(sl_bb, sl_swing)
            sl_dist = close - sl_price
            if sl_dist < atr * sl_min_atr:
                sl_price = close - atr * sl_min_atr
                sl_dist = atr * sl_min_atr
            if sl_dist > atr * sl_max_atr:
                sl_price = close - atr * sl_max_atr
                sl_dist = atr * sl_max_atr

            tp1 = bb_mid
            tp2 = bb_mid + (bb_mid - bb_lower) * 0.5
            tp_mid_d = tp1 - close
            tp_ext_d = tp2 - close
            if tp_mid_d > 0 and tp_mid_d / sl_dist >= min_rr:
                tp_price = tp1
            elif tp_ext_d > 0 and tp_ext_d / sl_dist >= min_rr:
                tp_price = tp2
            else:
                tp_price = close + sl_dist * min_rr
            if tp_price - close < tfp['tp_min'] * atr:
                tp_price = close + tfp['tp_min'] * atr
        else:
            sl_bb = bb_upper + atr * 0.3
            recent_high = float(df['high'].iloc[max(0, i - 10):i + 1].max())
            sl_swing = recent_high + atr * 0.2
            sl_price = min(sl_bb, sl_swing)
            sl_dist = sl_price - close
            if sl_dist < atr * sl_min_atr:
                sl_price = close + atr * sl_min_atr
                sl_dist = atr * sl_min_atr
            if sl_dist > atr * sl_max_atr:
                sl_price = close + atr * sl_max_atr
                sl_dist = atr * sl_max_atr

            tp1 = bb_mid
            tp2 = bb_mid - (bb_upper - bb_mid) * 0.5
            tp_mid_d = close - tp1
            tp_ext_d = close - tp2
            if tp_mid_d > 0 and tp_mid_d / sl_dist >= min_rr:
                tp_price = tp1
            elif tp_ext_d > 0 and tp_ext_d / sl_dist >= min_rr:
                tp_price = tp2
            else:
                tp_price = close - sl_dist * min_rr
            if close - tp_price < tfp['tp_min'] * atr:
                tp_price = close - tfp['tp_min'] * atr

        tp_dist = abs(tp_price - close)
        return {
            'direction': direction,
            'entry_price': close,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'tp_mult': tp_dist / atr if atr > 0 else 0,
            'sl_mult': sl_dist / atr if atr > 0 else 0,
            'atr': atr,
            'confidence': 0.80,
            'strategy': 'mr_ranging_v6',
        }

    # ══════════════════════════════════════════════
    # MOMENTUM Strateji: Kısa TF trend takibi
    # ══════════════════════════════════════════════
    def _signal_momentum(self, df, i, close, atr, rsi, tfp, trend_dir):
        """
        15m/30m momentum pullback — trend yönünde EMA geri çekilme.

        Koşullar (sıkı — az ama kaliteli trade):
          1. EMA tam hizalı: 9 > 20 > 50 (LONG) veya 9 < 20 < 50 (SHORT)
          2. Fiyat EMA9 veya EMA20'ye yakın (geri çekilmiş)
          3. Fiyat EMA9 üzerinde kapanmış (bounce teyidi)
          4. RSI aşırı değil (LONG 40-65, SHORT 35-60)
          5. MACD pozitif VE artıyor (momentum teyidi)

        R:R 3:1 → sadece %25 WR ile kârlı
        """
        ema9 = df['ema9'].iloc[i]
        ema20 = df['ema20'].iloc[i]
        ema50 = df['ema50'].iloc[i]

        if np.isnan(ema9) or np.isnan(ema20) or np.isnan(ema50):
            return None

        # MACD momentum
        macd_hist = df['macd_hist'].iloc[i] if 'macd_hist' in df.columns else 0
        prev_macd = df['macd_hist'].iloc[i-1] if i > 0 and 'macd_hist' in df.columns else 0
        if np.isnan(macd_hist): macd_hist = 0
        if np.isnan(prev_macd): prev_macd = 0

        direction = None

        if trend_dir == 'LONG':
            # EMA tam hizalama: 9 > 20 > 50
            if not (ema9 > ema20 > ema50):
                return None
            # Pullback: fiyat EMA9 veya EMA20'ye yakın
            at_ema9 = abs(close - ema9) < 0.3 * atr
            at_ema20 = abs(close - ema20) < 0.5 * atr
            if not (at_ema9 or at_ema20):
                return None
            # Fiyat EMA9 üzerinde (bounce teyidi)
            if close <= ema9:
                return None
            # RSI 40-65
            if rsi < 40 or rsi > 65:
                return None
            # MACD pozitif ve artıyor
            if macd_hist <= 0 or macd_hist <= prev_macd:
                return None
            direction = 'LONG'

        elif trend_dir == 'SHORT':
            if not (ema9 < ema20 < ema50):
                return None
            at_ema9 = abs(close - ema9) < 0.3 * atr
            at_ema20 = abs(close - ema20) < 0.5 * atr
            if not (at_ema9 or at_ema20):
                return None
            if close >= ema9:
                return None
            if rsi < 35 or rsi > 60:
                return None
            if macd_hist >= 0 or macd_hist >= prev_macd:
                return None
            direction = 'SHORT'

        if direction is None:
            return None

        # R:R 3:1 — sıkı SL ama yüksek R:R, %25 WR yeterli
        return self._build_signal(close, atr, direction, 1.5, 0.5, 'momentum_pullback')

    # ══════════════════════════════════════════════
    # EMA CROSS Strateji: Trend yakalama (1h+ ek gelir)
    # ══════════════════════════════════════════════
    def _signal_ema_cross(self, df, i, close, atr, rsi, tfp):
        """
        EMA9/EMA20 crossover — trending dönemlerde BB MR'ın kaybını telafi eder.

        BB MR sinyal üretmediğinde (trend varsa BB'ye dokunmuyor) bu strateji
        devreye girer ve trend yönünde trade açar.

        Koşullar:
          - EMA9, EMA20'yi yukarı/aşağı keser (crossover)
          - EMA50 trend yönünü teyit eder
          - RSI aşırı bölgede değil
        """
        ema9 = df['ema9'].iloc[i]
        ema20 = df['ema20'].iloc[i]
        ema50 = df['ema50'].iloc[i]

        if i < 1:
            return None
        prev_ema9 = df['ema9'].iloc[i-1]
        prev_ema20 = df['ema20'].iloc[i-1]

        if any(np.isnan(v) for v in [ema9, ema20, ema50, prev_ema9, prev_ema20]):
            return None

        # Golden cross: EMA9 yukarı keser EMA20'yi
        if prev_ema9 <= prev_ema20 and ema9 > ema20:
            if close > ema50 and rsi > 40 and rsi < 70:
                return self._build_signal(close, atr, 'LONG', 2.5, 1.2, 'ema_cross')

        # Death cross: EMA9 aşağı keser EMA20'yi
        if prev_ema9 >= prev_ema20 and ema9 < ema20:
            if close < ema50 and rsi > 30 and rsi < 60:
                return self._build_signal(close, atr, 'SHORT', 2.5, 1.2, 'ema_cross')

        return None

    # ══════════════════════════════════════════════
    # TRENDING Strateji: Trend yönünde trade (1h+)
    # ══════════════════════════════════════════════
    def _signal_trending(self, df, i, close, atr, rsi, tfp, trend_dir):
        """
        Trend piyasa — sadece trend yönünde trade.

        2 giriş noktası:
          A) BB bounce (trend yönünde) — dip/rally yakala
          B) EMA20 pullback + bounce — trend devam sinyali
        """
        # ── A) BB bounce (sadece trend yönünde) ──
        bb_mid = df['bb_mid'].iloc[i]
        bb_upper = df['bb_upper'].iloc[i]
        bb_lower = df['bb_lower'].iloc[i]

        bb_ok = not np.isnan(bb_mid)

        # BB genişlik filtresi
        if bb_ok and tfp['bb_width_min'] > 0:
            bb_w = df['bb_width'].iloc[i] if 'bb_width' in df.columns else 999
            if np.isnan(bb_w) or bb_w < tfp['bb_width_min']:
                bb_ok = False

        # Volume teyidi
        vol_ok = True
        if tfp.get('require_volume', False) and 'vol_ma' in df.columns:
            vol = df['volume'].iloc[i] if 'volume' in df.columns else 0
            vol_ma = df['vol_ma'].iloc[i]
            if not np.isnan(vol) and not np.isnan(vol_ma) and vol_ma > 0:
                vol_ok = vol > vol_ma * 1.2

        if bb_ok and vol_ok:
            touched_lower = False
            touched_upper = False
            for j in range(1, 4):
                if i - j < 0:
                    break
                pc = df['close'].iloc[i-j]
                pl = df['bb_lower'].iloc[i-j]
                pu = df['bb_upper'].iloc[i-j]
                if not np.isnan(pl) and pc <= pl:
                    touched_lower = True
                if not np.isnan(pu) and pc >= pu:
                    touched_upper = True

            rsi_long = tfp.get('rsi_long', 45)
            rsi_short = tfp.get('rsi_short', 55)

            if trend_dir == 'LONG' and touched_lower and close > bb_lower and close < bb_mid and rsi < rsi_long:
                tp_dist = abs(close - bb_mid)
                if tp_dist < tfp['tp_min'] * atr:
                    tp_dist = tfp['tp_min'] * atr
                return self._build_signal(close, atr, 'LONG', tp_dist / atr, tfp['sl_mult'], 'mr_trend')

            if trend_dir == 'SHORT' and touched_upper and close < bb_upper and close > bb_mid and rsi > rsi_short:
                tp_dist = abs(close - bb_mid)
                if tp_dist < tfp['tp_min'] * atr:
                    tp_dist = tfp['tp_min'] * atr
                return self._build_signal(close, atr, 'SHORT', tp_dist / atr, tfp['sl_mult'], 'mr_trend')

        # ── B) EMA20 pullback — trend devam ──
        ema20 = df['ema20'].iloc[i]
        ema50 = df['ema50'].iloc[i]
        if np.isnan(ema20) or np.isnan(ema50):
            return None

        dist_to_ema20 = abs(close - ema20) / atr
        macd_hist = df['macd_hist'].iloc[i] if 'macd_hist' in df.columns else 0
        prev_macd = df['macd_hist'].iloc[i-1] if i > 0 and 'macd_hist' in df.columns else 0
        if np.isnan(macd_hist): macd_hist = 0
        if np.isnan(prev_macd): prev_macd = 0

        if trend_dir == 'LONG':
            # Son 5 barda EMA20 altına düştü mü?
            had_pullback = any(
                df['low'].iloc[i-j] < ema20 for j in range(1, 6)
                if i - j >= 0 and not np.isnan(df['low'].iloc[i-j])
            )
            if (had_pullback and close > ema20 and close > ema50
                and dist_to_ema20 < 0.8
                and rsi > 40 and rsi < 60
                and macd_hist > prev_macd):
                return self._build_signal(close, atr, 'LONG', 1.5, tfp['sl_mult'] * 0.8, 'ema_pullback')

        elif trend_dir == 'SHORT':
            had_pullback = any(
                df['high'].iloc[i-j] > ema20 for j in range(1, 6)
                if i - j >= 0 and not np.isnan(df['high'].iloc[i-j])
            )
            if (had_pullback and close < ema20 and close < ema50
                and dist_to_ema20 < 0.8
                and rsi > 40 and rsi < 60
                and macd_hist < prev_macd):
                return self._build_signal(close, atr, 'SHORT', 1.5, tfp['sl_mult'] * 0.8, 'ema_pullback')

        return None

    def _build_signal(self, close, atr, direction, tp_mult, sl_mult, strategy):
        """Sinyal dict oluştur."""
        if direction == 'LONG':
            tp_price = close + atr * tp_mult
            sl_price = close - atr * sl_mult
        else:
            tp_price = close - atr * tp_mult
            sl_price = close + atr * sl_mult

        return {
            'direction': direction,
            'entry_price': close,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'tp_mult': tp_mult,
            'sl_mult': sl_mult,
            'atr': atr,
            'confidence': 0.80,
            'strategy': strategy,
        }

    def _get_lookahead(self, tf: str) -> int:
        mapping = {
            "1m": 30, "3m": 40, "5m": 36, "15m": 24,
            "30m": 20, "1h": 36, "4h": 18, "1d": 12,
        }
        return mapping.get(tf, 24)

    def run(
        self,
        df: pd.DataFrame,
        df_secondary: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
        symbol: str = "UNKNOWN",
        verbose: bool = False,
    ) -> BacktestResult:
        """Tek coin üzerinde backtest çalıştır."""
        if self.use_simple:
            return self._run_simple(df, symbol, verbose)
        return self._run_engine(df, df_secondary, df_4h, symbol, verbose)

    # ==================================================================
    # v3.0: Simple Strategy Run Loop
    # ==================================================================
    def _run_simple(self, df: pd.DataFrame, symbol: str, verbose: bool) -> BacktestResult:
        """Engine bypass — basit strateji ile backtest."""
        df = self._compute_indicators(df.copy())
        N = len(df)
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values

        result = BacktestResult()
        result.equity_curve = [self.initial_capital]
        capital = self.initial_capital

        open_trade = None
        signals_total = 0
        last_trade_bar = -10
        consecutive_sl = 0
        cooldown_until_bar = 0
        recent_outcomes = []  # Son 10 trade sonucu ('TP','SL','TIMEOUT')
        tfp = self._get_tf_params()
        trade_cooldown = tfp['cooldown']
        sl_cooldown = tfp['sl_cooldown']

        # Stats
        strategy_trades = {}
        regime_trades = {r.value: {'trades': 0, 'wins': 0, 'pnl': 0.0} for r in Regime}
        direction_stats = {'LONG': {'trades': 0, 'wins': 0, 'pnl': 0.0},
                           'SHORT': {'trades': 0, 'wins': 0, 'pnl': 0.0}}

        for i in range(55, N - 1):
            # -- Açık trade varsa çıkış kontrol --
            if open_trade is not None:
                # Update best price for breakeven tracking
                if open_trade['direction'] == 'LONG':
                    open_trade['best_price'] = max(open_trade['best_price'], highs[i])
                else:
                    open_trade['best_price'] = min(open_trade['best_price'], lows[i])
                outcome = self._check_trade_exit(
                    open_trade, highs[i], lows[i], closes[i], i
                )
                if outcome is not None:
                    rec = self._close_trade(open_trade, outcome, closes[i], i, symbol)
                    result.trades.append(rec)
                    trade_pnl = rec.pnl_pct / 100 * capital * rec.position_size
                    capital += trade_pnl
                    last_trade_bar = i

                    # Consecutive SL tracking + cooldown
                    if outcome['type'] == 'SL':
                        consecutive_sl += 1
                        if consecutive_sl >= 3:
                            cooldown_until_bar = i + sl_cooldown
                            consecutive_sl = 0
                    elif outcome['type'] == 'TP':
                        consecutive_sl = 0

                    # Rolling SL tracker — geniş kayıp dönemlerini yakala
                    recent_outcomes.append(outcome['type'])
                    if len(recent_outcomes) > 10:
                        recent_outcomes.pop(0)
                    if len(recent_outcomes) >= 10:
                        sl_count = sum(1 for o in recent_outcomes if o == 'SL')
                        if sl_count >= 7:
                            cooldown_until_bar = max(cooldown_until_bar, i + 15)
                            recent_outcomes.clear()

                    # Stats
                    d = open_trade['direction']
                    s = open_trade.get('strategy', 'unknown')
                    direction_stats[d]['trades'] += 1
                    direction_stats[d]['pnl'] += rec.pnl_atr
                    if s not in strategy_trades:
                        strategy_trades[s] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
                    strategy_trades[s]['trades'] += 1
                    if rec.is_winner:
                        direction_stats[d]['wins'] += 1
                        strategy_trades[s]['wins'] += 1
                    strategy_trades[s]['pnl'] += rec.pnl_atr

                    open_trade = None

            result.equity_curve.append(capital)

            # -- Yeni sinyal ara (sadece trade yoksa) --
            if open_trade is not None:
                continue
            if i - last_trade_bar < trade_cooldown:
                continue
            if i < cooldown_until_bar:
                continue

            sig = self._simple_signal(df, i)
            if sig is None:
                continue

            signals_total += 1

            # Trade aç
            open_trade = {
                'direction': sig['direction'],
                'entry_price': sig['entry_price'],
                'tp_price': sig['tp_price'],
                'sl_price': sig['sl_price'],
                'tp_mult': sig['tp_mult'],
                'sl_mult': sig['sl_mult'],
                'atr': sig['atr'],
                'entry_bar': i,
                'regime': 'simple',
                'strategy': sig['strategy'],
                'signal_conf': sig['confidence'],
                'meta_conf': 0.0,
                'position_size': 1.0,
                'best_price': sig['entry_price'],
                'trail_sl': sig['sl_price'],
                'trail_activated': False,
                'reason': 'trend_momentum_v3',
            }
            if verbose:
                print(f"  [{i}] {sig['direction']} entry={sig['entry_price']:.2f} "
                      f"TP={sig['tp_price']:.2f} SL={sig['sl_price']:.2f}")

        # Açık trade'i kapat
        if open_trade is not None:
            rec = self._close_trade(
                open_trade, {'type': 'TIMEOUT', 'exit_price': closes[-1]},
                closes[-1], N - 1, symbol
            )
            result.trades.append(rec)

        result = self._compute_stats(
            result, signals_total, 0,
            regime_trades, strategy_trades, direction_stats,
        )
        return result

    # ==================================================================
    # Legacy Engine Run (eski karmaşık engine)
    # ==================================================================
    def _run_engine(self, df, df_secondary, df_4h, symbol, verbose):
        """Eski AdaptiveEngine ile backtest (legacy)."""
        N = len(df)
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        atrs = df['atr'].values
        timestamps = df.index.values

        result = BacktestResult()
        result.equity_curve = [self.initial_capital]
        capital = self.initial_capital
        open_trades = []
        signals_total = 0
        signals_filtered = 0
        consecutive_sl = 0
        cooldown_until_bar = 0

        regime_trades = {r.value: {'trades': 0, 'wins': 0, 'pnl': 0.0} for r in Regime}
        strategy_trades = {}
        direction_stats = {'LONG': {'trades': 0, 'wins': 0, 'pnl': 0.0},
                           'SHORT': {'trades': 0, 'wins': 0, 'pnl': 0.0}}

        for i in range(self.regime_lookback, N - 1):
            atr_val = atrs[i]
            if np.isnan(atr_val) or atr_val < 1e-10:
                result.equity_curve.append(capital)
                continue

            closed_indices = []
            for t_idx, trade in enumerate(open_trades):
                outcome = self._check_trade_exit(trade, highs[i], lows[i], closes[i], i)
                if outcome is not None:
                    trade_record = self._close_trade(trade, outcome, closes[i], i, symbol)
                    result.trades.append(trade_record)
                    trade_pnl = trade_record.pnl_pct / 100 * capital * trade_record.position_size
                    capital += trade_pnl
                    regime_trades[trade['regime']]['trades'] += 1
                    regime_trades[trade['regime']]['pnl'] += trade_record.pnl_atr
                    direction_stats[trade['direction']]['trades'] += 1
                    direction_stats[trade['direction']]['pnl'] += trade_record.pnl_atr
                    sname = trade.get('strategy', 'unknown')
                    if sname not in strategy_trades:
                        strategy_trades[sname] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
                    strategy_trades[sname]['trades'] += 1
                    if trade_record.is_winner:
                        regime_trades[trade['regime']]['wins'] += 1
                        direction_stats[trade['direction']]['wins'] += 1
                        strategy_trades[sname]['wins'] += 1
                        consecutive_sl = 0
                    else:
                        if outcome['type'] == 'SL':
                            consecutive_sl += 1
                            if consecutive_sl >= 3:
                                cooldown_until_bar = i + 8
                                consecutive_sl = 0
                    strategy_trades[sname]['pnl'] += trade_record.pnl_atr
                    closed_indices.append(t_idx)

            for idx in sorted(closed_indices, reverse=True):
                open_trades.pop(idx)
            result.equity_curve.append(capital)

            if len(open_trades) >= self.max_concurrent:
                continue
            if i < cooldown_until_bar:
                continue

            df_4h_chunk = None
            if df_4h is not None:
                ts = pd.Timestamp(timestamps[i])
                df_4h_chunk = df_4h[df_4h.index <= ts]

            decision = self.engine.decide(
                df.iloc[:i + 1], df_secondary=None, df_4h=df_4h_chunk, symbol=symbol,
            )
            regime_val = decision.regime
            meta_conf = decision.meta_confidence
            position_size = decision.position_size

            if decision.action == 'HOLD':
                continue
            signals_total += 1
            signal = getattr(self.engine, '_last_signal', None)
            if not signal:
                continue
            if position_size < 0.30 or signal.confidence < 0.55:
                signals_filtered += 1
                continue

            same_dir_open = any(t['direction'] == signal.direction for t in open_trades)
            if same_dir_open:
                continue

            entry = closes[i]
            tp_mult = signal.tp_atr_mult
            sl_padded = signal.sl_atr_mult * 1.15
            if signal.direction == 'LONG':
                tp_price = entry + atrs[i] * tp_mult
                sl_price = entry - atrs[i] * sl_padded
            else:
                tp_price = entry - atrs[i] * tp_mult
                sl_price = entry + atrs[i] * sl_padded

            trade = {
                'direction': signal.direction, 'entry_price': entry,
                'tp_price': tp_price, 'sl_price': sl_price,
                'tp_mult': tp_mult, 'sl_mult': signal.sl_atr_mult,
                'atr': atrs[i], 'entry_bar': i, 'regime': regime_val,
                'strategy': signal.strategy_name, 'signal_conf': signal.confidence,
                'meta_conf': meta_conf, 'position_size': position_size,
                'best_price': entry, 'trail_sl': sl_price, 'trail_activated': False,
                'reason': signal.reason,
            }
            open_trades.append(trade)

        for trade in open_trades:
            rec = self._close_trade(trade, {'type': 'TIMEOUT', 'exit_price': closes[-1]}, closes[-1], N-1, symbol)
            result.trades.append(rec)

        result = self._compute_stats(result, signals_total, signals_filtered, regime_trades, strategy_trades, direction_stats)
        return result

    def _check_trade_exit(
        self, trade: Dict, high: float, low: float, close: float, bar: int
    ) -> Optional[Dict]:
        """
        v3.2 Çıkış mantığı — Sadelik:
          1. TP kontrol (önce — kârlı çıkışa öncelik)
          2. SL kontrol
          3. Timeout (close price ile çık)
        Trailing stop KALDIRILDI: kârlı trade'leri erken kesiyor,
        PF'yi düşürüyor (winner avg 0.3% vs loser avg 1.1%)
        """
        direction = trade['direction']
        entry = trade['entry_price']
        tp = trade['tp_price']
        sl = trade['sl_price']
        entry_bar = trade['entry_bar']
        bars_held = bar - entry_bar

        if direction == 'LONG':
            if high >= tp:
                return {'type': 'TP', 'exit_price': tp}
            if low <= sl:
                return {'type': 'SL', 'exit_price': sl}
            if bars_held >= self.lookahead:
                return {'type': 'TIMEOUT', 'exit_price': close}

        else:  # SHORT
            if low <= tp:
                return {'type': 'TP', 'exit_price': tp}
            if high >= sl:
                return {'type': 'SL', 'exit_price': sl}
            if bars_held >= self.lookahead:
                return {'type': 'TIMEOUT', 'exit_price': close}

        return None

    def _close_trade(
        self, trade: Dict, outcome: Dict, close: float, bar: int, symbol: str
    ) -> TradeRecord:
        """Trade'i kapat ve kayıt oluştur"""
        exit_price = outcome['exit_price']
        entry = trade['entry_price']
        direction = trade['direction']
        atr = trade['atr']

        fee = MAKER_FEE * 2  # Giriş + çıkış

        if direction == 'LONG':
            pnl_raw = (exit_price - entry) / entry
        else:
            pnl_raw = (entry - exit_price) / entry

        pnl_pct = (pnl_raw - fee) * 100
        pnl_atr = (exit_price - entry) / atr if direction == 'LONG' \
            else (entry - exit_price) / atr

        bars_held = bar - trade['entry_bar']
        is_winner = pnl_pct > 0

        return TradeRecord(
            symbol=symbol,
            bar_index=trade['entry_bar'],
            timestamp=str(bar),
            direction=direction,
            entry_price=entry,
            exit_price=exit_price,
            tp_price=trade['tp_price'],
            sl_price=trade['sl_price'],
            atr_at_entry=atr,
            pnl_pct=round(pnl_pct, 4),
            pnl_atr=round(pnl_atr, 4),
            outcome=outcome['type'],
            regime=trade['regime'],
            strategy=trade['strategy'],
            signal_confidence=trade['signal_conf'],
            meta_confidence=trade['meta_conf'],
            position_size=trade['position_size'],
            bars_held=bars_held,
            is_winner=is_winner,
            fee_paid=round(fee * 100, 4),
        )

    def _position_size(
        self, signal_conf: float, meta_conf: float, regime: Regime
    ) -> float:
        """Kelly-inspired position sizing"""
        combined = signal_conf * 0.3 + meta_conf * 0.7

        regime_max = {
            Regime.TRENDING: 1.0,
            Regime.MEAN_REVERTING: 0.80,
            Regime.HIGH_VOLATILE: 0.50,
            Regime.LOW_VOLATILE: 0.60,
        }
        max_size = regime_max.get(regime, 0.50)

        if combined >= 0.75:
            size = 1.0
        elif combined >= 0.65:
            size = 0.75
        elif combined >= 0.58:
            size = 0.50
        elif combined >= 0.52:
            size = 0.25
        else:
            size = 0.0

        return min(size, max_size)

    def _compute_stats(
        self, result: BacktestResult,
        signals_total: int, signals_filtered: int,
        regime_trades: Dict, strategy_trades: Dict,
        direction_stats: Dict,
    ) -> BacktestResult:
        """Backtest istatistiklerini hesapla"""
        trades = result.trades

        result.total_trades = len(trades)
        result.signals_generated = signals_total
        result.signals_filtered = signals_filtered
        result.filter_rate = (
            signals_filtered / signals_total * 100
            if signals_total > 0 else 0
        )

        if not trades:
            return result

        result.winners = sum(1 for t in trades if t.is_winner)
        result.losers = result.total_trades - result.winners
        result.win_rate = result.winners / result.total_trades * 100

        # PnL
        pnls = [t.pnl_pct for t in trades]
        pnl_atrs = [t.pnl_atr for t in trades]
        result.total_pnl_pct = sum(pnls)
        result.total_pnl_atr = sum(pnl_atrs)
        result.avg_pnl_per_trade = np.mean(pnls)

        # Profit Factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        result.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else 999.0
        )

        # Sharpe Ratio (günlük yaklaşım)
        if len(pnls) > 1:
            result.sharpe_ratio = (
                np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)
            )

        # Max Drawdown
        equity = result.equity_curve
        peak = equity[0]
        max_dd = 0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown_pct = round(max_dd, 2)

        # Avg bars held
        result.avg_bars_held = np.mean([t.bars_held for t in trades])

        # Regime stats
        for regime_val, data in regime_trades.items():
            t = data['trades']
            w = data['wins']
            pnl = data['pnl']
            result.regime_stats[regime_val] = {
                'trades': t,
                'wins': w,
                'wr': round(w / t * 100, 1) if t > 0 else 0,
                'pnl_atr': round(pnl, 2),
                'avg_pnl': round(pnl / t, 4) if t > 0 else 0,
            }

        # Strategy stats
        result.strategy_stats = {}
        for sname, data in strategy_trades.items():
            t = data['trades']
            w = data['wins']
            pnl = data['pnl']
            result.strategy_stats[sname] = {
                'trades': t,
                'wins': w,
                'wr': round(w / t * 100, 1) if t > 0 else 0,
                'pnl_atr': round(pnl, 2),
            }

        # Direction stats
        for d in ['LONG', 'SHORT']:
            data = direction_stats[d]
            t = data['trades']
            w = data['wins']
            pnl = data['pnl']
            result.by_direction[d] = {
                'trades': t,
                'wins': w,
                'wr': round(w / t * 100, 1) if t > 0 else 0,
                'pnl_atr': round(pnl, 2),
            }

        return result


# ===================================================================
# Coklu Coin Backtest
# ===================================================================

def run_full_backtest(
    symbols: Optional[List[str]] = None,
    timeframe: str = "15m",
    limit: int = 3000,
    use_meta_filter: bool = True,
    use_cache: bool = False,
    test_split: float = 0.30,
    verbose: bool = False,
) -> Dict:
    """
    Tüm coinler üzerinde backtest çalıştır.

    Args:
        test_split: Verinin son %X'ini test olarak kullan
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    print(f"\n{'=' * 65}")
    print(f"  ADAPTIVE BACKTEST v1.0")
    print(f"  TF: {timeframe} | Coins: {len(symbols)} | "
          f"Meta-filter: {'ON' if use_meta_filter else 'OFF'}")
    print(f"{'=' * 65}\n")

    engine = AdaptiveBacktest(
        timeframe=timeframe,
        use_meta_filter=use_meta_filter,
    )

    fetcher = DataFetcher('binance')
    all_results = []
    all_trades = []
    skipped = []

    for idx, sym in enumerate(symbols):
        try:
            if use_cache:
                try:
                    from ai.data_cache import get_cached_ohlcv
                    df = get_cached_ohlcv(sym, timeframe, limit=limit, fetcher=fetcher)
                except ImportError:
                    df = fetcher.fetch_ohlcv(sym, timeframe, limit=limit)
            else:
                df = fetcher.fetch_ohlcv(sym, timeframe, limit=limit)

            if df is None or df.empty or len(df) < 300:
                skipped.append(sym)
                continue

            df = add_all_indicators(df)
            df = enrich_ohlcv_with_futures(df, sym, silent=True)
            df = generate_features(df)
            df = add_meta_context_features(df)

            # -- v2.0: Multi-TF Data Fetching ----------------
            df_1h = df.copy()
            df_15m = None
            df_4h = None

            try:
                df_15m = fetcher.fetch_ohlcv(sym, "15m", limit=limit * 4)
                if df_15m is not None and not df_15m.empty:
                    df_15m = add_all_indicators(df_15m)
                    df_15m = generate_features(df_15m)
                    df_15m = add_meta_context_features(df_15m)
            except Exception as e:
                print(f"  [WARN] {sym} 15m hatasi: {e}")

            try:
                df_4h = fetcher.fetch_ohlcv(sym, "4h", limit=max(limit // 4, 500))
                if df_4h is not None and not df_4h.empty:
                    df_4h = add_all_indicators(df_4h)
            except Exception as e:
                print(f"  [WARN] {sym} 4h hatasi: {e}")

            # Son %30'u test olarak kullan
            test_start = int(len(df_1h) * (1 - test_split))
            test_df_1h = df_1h.iloc[test_start:].copy()

            # 15m verisini 1H test aralığına göre filtrele
            test_df_15m = None
            if df_15m is not None:
                start_ts = test_df_1h.index[0]
                test_df_15m = df_15m[df_15m.index >= (start_ts - pd.Timedelta(hours=24))].copy()

            # 4h verisini test aralığına göre filtrele
            test_df_4h = None
            if df_4h is not None and not df_4h.empty:
                start_ts = test_df_1h.index[0]
                test_df_4h = df_4h[df_4h.index <= test_df_1h.index[-1]].copy()

            if len(test_df_1h) < 100:
                skipped.append(sym)
                continue

            # Backtest çalıştır (Multi-TF: 4h→1h→15m)
            bt_result = engine.run(
                test_df_1h, df_secondary=test_df_15m,
                df_4h=test_df_4h, symbol=sym, verbose=verbose,
            )

            if bt_result.total_trades > 0:
                all_results.append({
                    'symbol': sym,
                    'result': bt_result,
                })
                all_trades.extend(bt_result.trades)

                if (idx + 1) % 5 == 0 or bt_result.total_trades > 20:
                    print(f"  [{idx + 1}/{len(symbols)}] {sym}: "
                          f"Trades={bt_result.total_trades} "
                          f"WR=%{bt_result.win_rate:.1f} "
                          f"PF={bt_result.profit_factor:.2f} "
                          f"PnL={bt_result.total_pnl_pct:.2f}%")
            else:
                if (idx + 1) % 10 == 0:
                    print(f"  [{idx + 1}/{len(symbols)}] {sym}: No trades")

            del df, df_1h, test_df_1h, df_15m, test_df_15m
            gc.collect()

            if not use_cache:
                _time.sleep(0.25)

        except Exception as e:
            print(f"  [ERROR] {sym}: {e}")
            skipped.append(sym)

    # -- v1.3: Engine debug stats -------------------------
    # Backtest engine'den strateji istatistiklerini al
    if hasattr(engine, 'strategies'):
        print(f"\n  Strateji Debug:")
        print(f"     Signals generated: {sum(1 for _ in all_trades)}")

    if not all_trades:
        print(f"\n  [EMPTY] 0 trade! Olasi sorunlar:")
        print(f"     1. Stratejiler hic sinyal uretmiyor")
        print(f"     2. Meta-filter her seyi reddediyor")
        print(f"     3. Confidence esigi cok yuksek")
        print(f"     -> --no-meta ile tekrar dene")
        return {"error": "No trades", "debug": "Check strategy thresholds"}

    summary = _compute_summary(all_trades, all_results, symbols, skipped)
    _print_summary(summary, use_meta_filter)

    return summary


def _compute_summary(
    all_trades: List[TradeRecord],
    all_results: List[Dict],
    symbols: List[str],
    skipped: List[str],
) -> Dict:
    """Toplam backtest istatistikleri"""
    total = len(all_trades)
    winners = sum(1 for t in all_trades if t.is_winner)
    losers = total - winners

    pnls = [t.pnl_pct for t in all_trades]
    pnl_atrs = [t.pnl_atr for t in all_trades]

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))

    # Rejim bazlı
    regime_agg = {}
    for t in all_trades:
        r = t.regime
        if r not in regime_agg:
            regime_agg[r] = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'pnl_atr': 0.0}
        regime_agg[r]['trades'] += 1
        regime_agg[r]['pnl'] += t.pnl_pct
        regime_agg[r]['pnl_atr'] += t.pnl_atr
        if t.is_winner:
            regime_agg[r]['wins'] += 1

    # Strateji bazlı
    strat_agg = {}
    for t in all_trades:
        s = t.strategy
        if s not in strat_agg:
            strat_agg[s] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
        strat_agg[s]['trades'] += 1
        strat_agg[s]['pnl'] += t.pnl_pct
        if t.is_winner:
            strat_agg[s]['wins'] += 1

    # Yön bazlı
    dir_agg = {'LONG': {'trades': 0, 'wins': 0, 'pnl': 0.0},
               'SHORT': {'trades': 0, 'wins': 0, 'pnl': 0.0}}
    for t in all_trades:
        dir_agg[t.direction]['trades'] += 1
        dir_agg[t.direction]['pnl'] += t.pnl_pct
        if t.is_winner:
            dir_agg[t.direction]['wins'] += 1

    # Outcome bazlı
    outcome_agg = {}
    for t in all_trades:
        o = t.outcome
        if o not in outcome_agg:
            outcome_agg[o] = {'count': 0, 'wins': 0}
        outcome_agg[o]['count'] += 1
        if t.is_winner:
            outcome_agg[o]['wins'] += 1

    sharpe = np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252) if len(pnls) > 1 else 0

    return {
        'total_trades': total,
        'winners': winners,
        'losers': losers,
        'win_rate': round(winners / total * 100, 2),
        'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999,
        'total_pnl_pct': round(sum(pnls), 2),
        'total_pnl_atr': round(sum(pnl_atrs), 2),
        'avg_pnl_per_trade': round(np.mean(pnls), 4),
        'median_pnl': round(np.median(pnls), 4),
        'sharpe_ratio': round(sharpe, 2),
        'avg_bars_held': round(np.mean([t.bars_held for t in all_trades]), 1),
        'coins_traded': len(set(t.symbol for t in all_trades)),
        'coins_total': len(symbols),
        'coins_skipped': len(skipped),
        'by_regime': {
            r: {
                'trades': d['trades'],
                'wr': round(d['wins'] / d['trades'] * 100, 1) if d['trades'] > 0 else 0,
                'pnl_pct': round(d['pnl'], 2),
                'pnl_atr': round(d['pnl_atr'], 2),
            }
            for r, d in regime_agg.items()
        },
        'by_strategy': {
            s: {
                'trades': d['trades'],
                'wr': round(d['wins'] / d['trades'] * 100, 1) if d['trades'] > 0 else 0,
                'pnl_pct': round(d['pnl'], 2),
            }
            for s, d in strat_agg.items()
        },
        'by_direction': {
            d: {
                'trades': data['trades'],
                'wr': round(data['wins'] / data['trades'] * 100, 1) if data['trades'] > 0 else 0,
                'pnl_pct': round(data['pnl'], 2),
            }
            for d, data in dir_agg.items()
        },
        'by_outcome': outcome_agg,
    }


def _print_summary(summary: Dict, use_meta: bool):
    """Sonuclari guzel formatta yazdir"""
    meta_str = "META-FILTER ON" if use_meta else "META-FILTER OFF"

    print(f"\n{'=' * 65}")
    print(f"  BACKTEST SONUCCLARI ({meta_str})")
    print(f"{'=' * 65}")
    print(f"  Total Trades : {summary['total_trades']:,}")
    print(f"  Winners      : {summary['winners']:,}")
    print(f"  Losers       : {summary['losers']:,}")
    print(f"  Win Rate     : %{summary['win_rate']}")
    print(f"  Profit Factor: {summary['profit_factor']}")
    print(f"  Total PnL    : %{summary['total_pnl_pct']} "
          f"({summary['total_pnl_atr']:.1f} ATR)")
    print(f"  Avg PnL/Trade: %{summary['avg_pnl_per_trade']:.4f}")
    print(f"  Sharpe Ratio : {summary['sharpe_ratio']}")
    print(f"  Avg Hold     : {summary['avg_bars_held']:.1f} bars")
    print(f"  Coins        : {summary['coins_traded']} / {summary['coins_total']}")

    print(f"\n{'-' * 55}")
    print(f"  REJIM BAZLI:")
    for regime, data in summary['by_regime'].items():
        print(f"    {regime:16}: T={data['trades']:4} "
              f"WR=%{data['wr']:5.1f} PnL=%{data['pnl_pct']:8.2f} "
              f"({data['pnl_atr']:.1f} ATR)")

    print(f"\n{'-' * 55}")
    print(f"  STRATEJI BAZLI:")
    for strat, data in summary['by_strategy'].items():
        print(f"    {strat:16}: T={data['trades']:4} "
              f"WR=%{data['wr']:5.1f} PnL=%{data['pnl_pct']:8.2f}")

    print(f"\n{'-' * 55}")
    print(f"  YON BAZLI:")
    for d, data in summary['by_direction'].items():
        print(f"    {d:16}: T={data['trades']:4} "
              f"WR=%{data['wr']:5.1f} PnL=%{data['pnl_pct']:8.2f}")

    print(f"\n{'-' * 55}")
    print(f"  SONUÇ BAZLI:")
    for outcome, data in summary.get('by_outcome', {}).items():
        ct = data['count']
        w = data['wins']
        wr = w / ct * 100 if ct > 0 else 0
        print(f"    {outcome:8}: {ct:4} trade | WR=%{wr:.1f}")

    print(f"{'=' * 65}\n")


# ===================================================================
# A/B Test: Meta-filter ON vs OFF
# ===================================================================

def run_ab_test(
    symbols: Optional[List[str]] = None,
    timeframe: str = "15m",
    limit: int = 3000,
    use_cache: bool = False,
) -> Dict:
    """Meta-filter ON vs OFF karşılaştırması"""
    print(f"\n{'=' * 65}")
    print(f"  [DEBUG] A/B TEST: Meta-Filter ON vs OFF")
    print(f"{'=' * 65}\n")

    # A: Meta-filter OFF
    print("--- A: META-FILTER OFF ---")
    result_off = run_full_backtest(
        symbols=symbols, timeframe=timeframe, limit=limit,
        use_meta_filter=False, use_cache=use_cache,
    )

    # B: Meta-filter ON
    print("\n--- B: META-FILTER ON ---")
    result_on = run_full_backtest(
        symbols=symbols, timeframe=timeframe, limit=limit,
        use_meta_filter=True, use_cache=use_cache,
    )

    # Karşılaştırma
    print(f"\n  {'-' * 50}")
    print(f"  BACKTEST DEBUG STATS")
    print(f"  {'-' * 50}")

    metrics = ['total_trades', 'win_rate', 'profit_factor',
               'total_pnl_pct', 'sharpe_ratio', 'avg_pnl_per_trade']

    for m in metrics:
        val_off = result_off.get(m, 0)
        val_on = result_on.get(m, 0)

        if isinstance(val_off, (int, float)) and isinstance(val_on, (int, float)):
            diff = val_on - val_off
            arrow = "[UP]" if diff > 0 else "[DOWN]" if diff < 0 else "[=]"
            print(f"  {m:20}: OFF={val_off:10} | ON={val_on:10} | "
                  f"Diff={diff:+.2f} {arrow}")

    print(f"{'=' * 65}\n")

    return {
        'without_meta': result_off,
        'with_meta': result_on,
    }


# ===================================================================
# CLI
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive Backtest v1.0")
    parser.add_argument('--timeframe', type=str, default='15m')
    parser.add_argument('--limit', type=int, default=3000)
    parser.add_argument('--symbols', type=str, default='')
    parser.add_argument('--no-meta', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--ab-test', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    syms = [s.strip() for s in args.symbols.split(',') if s.strip()] or None

    if args.ab_test:
        run_ab_test(
            symbols=syms, timeframe=args.timeframe,
            limit=args.limit, use_cache=args.cache,
        )
    else:
        run_full_backtest(
            symbols=syms, timeframe=args.timeframe, limit=args.limit,
            use_meta_filter=not args.no_meta,
            use_cache=args.cache, verbose=args.verbose,
        )
