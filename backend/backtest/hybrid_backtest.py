"""
Hybrid Backtest V2.1 — Rejim Bazlı Strateji Seçimi
====================================================

Mimari:
  1. Her bar'da detect_regime() → TRENDING / MEAN_REVERTING / HIGH_VOLATILE / LOW_VOLATILE
  2. TRENDING       → Trend Following v4.4 (Supertrend + Triple EMA)
  3. MEAN_REVERTING → BB Mean Reversion (BB touch + rejection + RSI)
  4. HIGH_VOLATILE  → BB MR (düşük risk, sıkı SL)
  5. LOW_VOLATILE   → Trade yok (bekle)

Amaç: Tek strateji yerine piyasa koşuluna göre uygun stratejiyi seç.
"""

from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators, ema
from backtest.trend_backtest import supertrend
from ai.regime_detector import detect_regime, Regime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════
#  TREND FOLLOWING ENTRY/EXIT (v4.4'ten alındı)
# ══════════════════════════════════════════════════════════════════

class TrendStrategy:
    """Trend Following v4.4 entry/exit logic (stateless)."""

    MIN_ADX = 20
    MAX_SL_PCT = 0.020
    BE_THRESHOLD = 0.03
    TRAIL_START = 0.06
    TRAIL_KEEP = 0.55

    @staticmethod
    def check_entry(df: pd.DataFrame, i: int) -> Optional[Tuple[str, float, float]]:
        if i < 2:
            return None

        st_dir = int(df['st_direction'].iloc[i])
        st_dir_prev = int(df['st_direction'].iloc[i - 1])
        close = float(df['close'].iloc[i])
        adx = float(df['adx'].iloc[i])
        rsi_val = float(df['rsi'].iloc[i])
        ema9 = float(df['ema9'].iloc[i])
        ema21 = float(df['ema21'].iloc[i])
        vol_ratio = float(df['vol_ratio_20'].iloc[i])
        macd_hist = float(df['macd_hist'].iloc[i])

        st_flip_bull = (st_dir == 1 and st_dir_prev == -1)
        st_flip_bear = (st_dir == -1 and st_dir_prev == 1)
        ema_bull = ema9 > ema21
        ema_bear = ema9 < ema21

        recent_cross_bull = False
        recent_cross_bear = False
        for j in range(max(0, i - 3), i + 1):
            ec = df['ema_cross'].iloc[j]
            if not pd.isna(ec):
                if float(ec) > 0:
                    recent_cross_bull = True
                elif float(ec) < 0:
                    recent_cross_bear = True

        direction = None
        if st_flip_bull and ema_bull:
            direction = 'LONG'
        elif st_dir == 1 and recent_cross_bull and ema_bull and macd_hist > 0:
            direction = 'LONG'
        if direction is None:
            if st_flip_bear and ema_bear:
                direction = 'SHORT'
            elif st_dir == -1 and recent_cross_bear and ema_bear and macd_hist < 0:
                direction = 'SHORT'

        if direction is None:
            return None

        if adx < TrendStrategy.MIN_ADX:
            return None
        if vol_ratio < 0.8:
            return None
        if direction == 'LONG' and rsi_val > 70:
            return None
        if direction == 'SHORT' and rsi_val < 30:
            return None

        # Swing SL
        lookback = 5
        start = max(0, i - lookback)
        if direction == 'LONG':
            sl = float(df['low'].iloc[start:i + 1].min())
            sl = max(sl, close * (1 - TrendStrategy.MAX_SL_PCT))
            if sl >= close * 0.997:
                return None
        else:
            sl = float(df['high'].iloc[start:i + 1].max())
            sl = min(sl, close * (1 + TrendStrategy.MAX_SL_PCT))
            if sl <= close * 1.003:
                return None

        sl_dist = abs(close - sl)
        tp = close + sl_dist * 20 if direction == 'LONG' else close - sl_dist * 20
        return (direction, sl, tp)

    @staticmethod
    def check_exit(trade: Dict, df: pd.DataFrame, i: int) -> Optional[Tuple[str, float]]:
        entry = trade['entry_price']
        sl = trade['sl_price']
        direction = trade['direction']
        qty = trade['qty']
        high = float(df['high'].iloc[i])
        low = float(df['low'].iloc[i])
        close = float(df['close'].iloc[i])

        if direction == 'LONG':
            if high > trade['peak_price']:
                trade['peak_price'] = high
            pnl_pct = (close - entry) / entry
            peak_pnl_pct = (trade['peak_price'] - entry) / entry
        else:
            if low < trade['peak_price']:
                trade['peak_price'] = low
            pnl_pct = (entry - close) / entry
            peak_pnl_pct = (entry - trade['peak_price']) / entry

        trade['max_profit_pct'] = max(trade.get('max_profit_pct', 0), pnl_pct * 100)

        # Tier 1: Breakeven at 3% peak
        if peak_pnl_pct >= TrendStrategy.BE_THRESHOLD and not trade.get('be_activated'):
            trade['be_activated'] = True
            if direction == 'LONG':
                be_price = entry * 1.001
                if be_price > sl:
                    trade['sl_price'] = be_price
                    sl = be_price
            else:
                be_price = entry * 0.999
                if be_price < sl:
                    trade['sl_price'] = be_price
                    sl = be_price

        # Tier 2: Trail at 6%/55%
        if peak_pnl_pct >= TrendStrategy.TRAIL_START:
            trade['trail_active'] = True
            keep = peak_pnl_pct * TrendStrategy.TRAIL_KEEP
            if direction == 'LONG':
                new_sl = entry * (1 + keep)
                if new_sl > sl:
                    trade['sl_price'] = new_sl
                    sl = new_sl
            else:
                new_sl = entry * (1 - keep)
                if new_sl < sl:
                    trade['sl_price'] = new_sl
                    sl = new_sl

        # SL hit
        if direction == 'LONG' and low <= sl:
            return ('SL', sl)
        if direction == 'SHORT' and high >= sl:
            return ('SL', sl)

        # ST flip exit (kârda)
        st_dir = int(df['st_direction'].iloc[i])
        if direction == 'LONG' and st_dir == -1 and pnl_pct > 0.003:
            return ('ST_FLIP', close)
        if direction == 'SHORT' and st_dir == 1 and pnl_pct > 0.003:
            return ('ST_FLIP', close)

        # Max dollar loss
        pnl_dollar = (close - entry) * qty if direction == 'LONG' else (entry - close) * qty
        if pnl_dollar < -4.0:
            return ('MAXLOSS', close)

        # Timeout
        if i - trade['entry_bar'] >= 72:
            return ('TIMEOUT', close)

        return None


# ══════════════════════════════════════════════════════════════════
#  BB MEAN REVERSION ENTRY/EXIT (bb_mr_strategy'den sadeleştirildi)
# ══════════════════════════════════════════════════════════════════

class MeanReversionStrategy:
    """BB Mean Reversion — backtest-uyumlu basitleştirilmiş versiyon."""

    MIN_REJ_SCORE = 3.5
    MAX_ADX = 30
    MIN_RR = 1.5
    SL_ATR_MULT = 1.5
    TIMEOUT_BARS = 18  # 18 saat (1h bar)

    @staticmethod
    def _rejection_score(df: pd.DataFrame, i: int, direction: str,
                         bb_lower: float, bb_upper: float) -> float:
        close = float(df['close'].iloc[i])
        open_p = float(df['open'].iloc[i])
        high = float(df['high'].iloc[i])
        low = float(df['low'].iloc[i])
        full_range = high - low
        if full_range <= 0:
            return 0.0

        score = 0.0
        body = abs(close - open_p)

        if direction == 'LONG':
            lower_wick = min(open_p, close) - low
            wick_ratio = lower_wick / full_range
            clv = (close - low) / full_range

            if wick_ratio >= 0.55:
                score += 3.0
            elif wick_ratio >= 0.40:
                score += 2.0
            elif wick_ratio >= 0.25:
                score += 1.0

            if clv >= 0.70:
                score += 2.0
            elif clv >= 0.55:
                score += 1.0

            if close > bb_lower and close > open_p:
                score += 1.5
            if wick_ratio > 0.5 and (high - max(open_p, close)) < body * 0.3:
                score += 2.0
        else:
            upper_wick = high - max(open_p, close)
            wick_ratio = upper_wick / full_range
            clv = (high - close) / full_range

            if wick_ratio >= 0.55:
                score += 3.0
            elif wick_ratio >= 0.40:
                score += 2.0
            elif wick_ratio >= 0.25:
                score += 1.0

            if clv >= 0.70:
                score += 2.0
            elif clv >= 0.55:
                score += 1.0

            if close < bb_upper and close < open_p:
                score += 1.5
            if wick_ratio > 0.5 and (min(open_p, close) - low) < body * 0.3:
                score += 2.0

        return min(score, 10)

    @staticmethod
    def _is_band_walk(df: pd.DataFrame, i: int, direction: str) -> bool:
        """Son 5 bar'da BB bandına yapışık mı (trend devam ediyor)."""
        near = 0
        for j in range(1, 6):
            idx = i - j
            if idx < 0:
                break
            c = float(df['close'].iloc[idx])
            if direction == 'LONG':
                bb = float(df['bb_lower'].iloc[idx])
                if not np.isnan(bb) and c <= bb * 1.015:
                    near += 1
            else:
                bb = float(df['bb_upper'].iloc[idx])
                if not np.isnan(bb) and c >= bb * 0.985:
                    near += 1
        return near >= 4

    @staticmethod
    def _has_rsi_divergence(df: pd.DataFrame, i: int, direction: str) -> bool:
        """RSI fiyattan ayrışıyor mu (mean reversion konfirmasyonu)."""
        if i < 10:
            return False
        close = df['close'].astype(float)
        rsi_s = df['rsi'].astype(float)
        if direction == 'LONG':
            # Fiyat düşük, RSI yükseliyor = bullish divergence
            price_lower = close.iloc[i] < close.iloc[i - 5]
            rsi_higher = rsi_s.iloc[i] > rsi_s.iloc[i - 5]
            return price_lower and rsi_higher
        else:
            price_higher = close.iloc[i] > close.iloc[i - 5]
            rsi_lower = rsi_s.iloc[i] < rsi_s.iloc[i - 5]
            return price_higher and rsi_lower

    @staticmethod
    def check_entry(df: pd.DataFrame, i: int,
                    risk_mult: float = 1.0) -> Optional[Tuple[str, float, float]]:
        if i < 10:
            return None

        close = float(df['close'].iloc[i])
        rsi_val = float(df['rsi'].iloc[i])
        adx_val = float(df['adx'].iloc[i])
        bb_upper = float(df['bb_upper'].iloc[i])
        bb_lower = float(df['bb_lower'].iloc[i])
        bb_mid = float(df['bb_mid'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        low = float(df['low'].iloc[i])
        high = float(df['high'].iloc[i])

        if any(np.isnan(v) for v in [rsi_val, adx_val, bb_upper, bb_lower, bb_mid, atr_val]):
            return None
        if atr_val <= 0 or close <= 0:
            return None
        if adx_val > MeanReversionStrategy.MAX_ADX:
            return None

        # Trend bias: EMA50 eğimi
        ema50 = float(df['ema50'].iloc[i]) if 'ema50' in df.columns else None
        trend_bias = 'flat'
        if ema50 is not None and not np.isnan(ema50) and i >= 20:
            ema50_prev = float(df['ema50'].iloc[i - 20])
            if not np.isnan(ema50_prev) and ema50_prev > 0:
                slope = (ema50 - ema50_prev) / ema50_prev
                if slope >= 0.015:
                    trend_bias = 'up'
                elif slope <= -0.015:
                    trend_bias = 'down'

        direction = None
        rej_score = 0.0

        # LONG: BB lower touch + RSI oversold
        if low <= bb_lower and close > bb_lower and rsi_val < 42:
            # Counter-trend LONG (düşüş trendinde) → daha sıkı RSI
            if trend_bias == 'down' and rsi_val > 35:
                pass  # RSI yeterince düşük değil
            else:
                if not MeanReversionStrategy._is_band_walk(df, i, 'LONG'):
                    rej = MeanReversionStrategy._rejection_score(df, i, 'LONG', bb_lower, bb_upper)
                    if rej >= MeanReversionStrategy.MIN_REJ_SCORE:
                        direction = 'LONG'
                        rej_score = rej

        # SHORT: BB upper touch + RSI overbought
        if direction is None and high >= bb_upper and close < bb_upper and rsi_val > 58:
            if trend_bias == 'up' and rsi_val < 65:
                pass  # RSI yeterince yüksek değil
            else:
                if not MeanReversionStrategy._is_band_walk(df, i, 'SHORT'):
                    rej = MeanReversionStrategy._rejection_score(df, i, 'SHORT', bb_lower, bb_upper)
                    if rej >= MeanReversionStrategy.MIN_REJ_SCORE:
                        direction = 'SHORT'
                        rej_score = rej

        if direction is None:
            return None

        # Counter-trend: divergence bonus, hard block değil
        is_counter = (direction == 'LONG' and trend_bias == 'down') or \
                     (direction == 'SHORT' and trend_bias == 'up')
        if is_counter:
            # Divergence yoksa rejection skoru daha yüksek olmalı
            if not MeanReversionStrategy._has_rsi_divergence(df, i, direction):
                if rej_score < 5.0:  # Divergence yoksa daha güçlü rejection gerekli
                    return None

        # SL/TP
        sl_mult = MeanReversionStrategy.SL_ATR_MULT * risk_mult
        if direction == 'LONG':
            sl = close - atr_val * sl_mult
            tp = bb_mid + (bb_mid - bb_lower) * 0.3
        else:
            sl = close + atr_val * sl_mult
            tp = bb_mid - (bb_upper - bb_mid) * 0.3

        # R:R check
        sl_dist = abs(close - sl)
        tp_dist = abs(tp - close)
        if sl_dist <= 0:
            return None
        rr = tp_dist / sl_dist
        if rr < MeanReversionStrategy.MIN_RR:
            return None

        return (direction, sl, tp)

    @staticmethod
    def check_exit(trade: Dict, df: pd.DataFrame, i: int) -> Optional[Tuple[str, float]]:
        entry = trade['entry_price']
        sl = trade['sl_price']
        tp = trade['tp_price']
        direction = trade['direction']
        qty = trade['qty']
        high = float(df['high'].iloc[i])
        low = float(df['low'].iloc[i])
        close = float(df['close'].iloc[i])

        # SL hit
        if direction == 'LONG' and low <= sl:
            return ('SL', sl)
        if direction == 'SHORT' and high >= sl:
            return ('SL', sl)

        # TP hit
        if direction == 'LONG' and high >= tp:
            return ('TP', tp)
        if direction == 'SHORT' and low <= tp:
            return ('TP', tp)

        # BB mid partial — close at bb_mid if in profit
        bb_mid = float(df['bb_mid'].iloc[i])
        pnl_pct = ((close - entry) / entry) if direction == 'LONG' else ((entry - close) / entry)
        if pnl_pct > 0.002:
            if direction == 'LONG' and close >= bb_mid:
                return ('BB_MID', close)
            if direction == 'SHORT' and close <= bb_mid:
                return ('BB_MID', close)

        # Max dollar loss
        pnl_dollar = (close - entry) * qty if direction == 'LONG' else (entry - close) * qty
        if pnl_dollar < -4.0:
            return ('MAXLOSS', close)

        # Timeout
        if i - trade['entry_bar'] >= MeanReversionStrategy.TIMEOUT_BARS:
            return ('TIMEOUT', close)

        return None


# ══════════════════════════════════════════════════════════════════
#  BREAKOUT STRATEGY (HIGH_VOLATILE rejimde opsiyonel)
# ══════════════════════════════════════════════════════════════════

class BreakoutStrategy:
    """Donchian breakout — yüksek volatilite rejimde kullanılır."""

    DONCHIAN_PERIOD = 20
    VOL_MULT = 1.5
    TRAIL_ATR_MULT = 2.0
    TIMEOUT_BARS = 48

    @staticmethod
    def check_entry(df: pd.DataFrame, i: int) -> Optional[Tuple[str, float, float]]:
        if i < BreakoutStrategy.DONCHIAN_PERIOD + 1:
            return None

        close = float(df['close'].iloc[i])
        high = float(df['high'].iloc[i])
        low = float(df['low'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        vol_ratio = float(df['vol_ratio_20'].iloc[i])

        if any(np.isnan(v) for v in [atr_val, vol_ratio]):
            return None
        if atr_val <= 0:
            return None

        # Volume confirmation
        if vol_ratio < BreakoutStrategy.VOL_MULT:
            return None

        # Donchian channel
        lookback = df.iloc[i - BreakoutStrategy.DONCHIAN_PERIOD:i]
        donch_high = float(lookback['high'].max())
        donch_low = float(lookback['low'].min())

        direction = None

        if high > donch_high:
            direction = 'LONG'
        elif low < donch_low:
            direction = 'SHORT'

        if direction is None:
            return None

        # SL/TP
        if direction == 'LONG':
            sl = close - atr_val * 1.2
            tp = close + atr_val * 3.0
        else:
            sl = close + atr_val * 1.2
            tp = close - atr_val * 3.0

        return (direction, sl, tp)

    @staticmethod
    def check_exit(trade: Dict, df: pd.DataFrame, i: int) -> Optional[Tuple[str, float]]:
        entry = trade['entry_price']
        sl = trade['sl_price']
        tp = trade['tp_price']
        direction = trade['direction']
        qty = trade['qty']
        high = float(df['high'].iloc[i])
        low = float(df['low'].iloc[i])
        close = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])

        # Track peak + trail
        if direction == 'LONG':
            if high > trade.get('peak_price', entry):
                trade['peak_price'] = high
            pnl_pct = (close - entry) / entry
        else:
            if low < trade.get('peak_price', entry):
                trade['peak_price'] = low
            pnl_pct = (entry - close) / entry

        # ATR trailing after +2%
        if pnl_pct > 0.02:
            if direction == 'LONG':
                trail_sl = trade['peak_price'] - atr_val * BreakoutStrategy.TRAIL_ATR_MULT
                if trail_sl > sl:
                    trade['sl_price'] = trail_sl
                    sl = trail_sl
            else:
                trail_sl = trade['peak_price'] + atr_val * BreakoutStrategy.TRAIL_ATR_MULT
                if trail_sl < sl:
                    trade['sl_price'] = trail_sl
                    sl = trail_sl

        # SL hit
        if direction == 'LONG' and low <= sl:
            return ('SL', sl)
        if direction == 'SHORT' and high >= sl:
            return ('SL', sl)

        # TP hit
        if direction == 'LONG' and high >= tp:
            return ('TP', tp)
        if direction == 'SHORT' and low <= tp:
            return ('TP', tp)

        # Max dollar loss
        pnl_dollar = (close - entry) * qty if direction == 'LONG' else (entry - close) * qty
        if pnl_dollar < -4.0:
            return ('MAXLOSS', close)

        # Timeout
        if i - trade['entry_bar'] >= BreakoutStrategy.TIMEOUT_BARS:
            return ('TIMEOUT', close)

        return None


# ══════════════════════════════════════════════════════════════════
#  HYBRID BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════

# V2.3: Paralel strateji — her ikisi de her zaman aktif
# Rejim sadece risk çarpanı olarak kullanılır
ALL_STRATEGIES = ['trend', 'mr']  # Her zaman ikisi de denenecek

# Rejim → Strateji bazında risk çarpanı
# Format: {rejim: {strateji: çarpan}}
REGIME_RISK_MAP = {
    Regime.TRENDING: {
        'trend': 1.0,    # Trend rejimde trend tam risk
        'mr':    0.6,    # Trend rejimde MR düşük risk
    },
    Regime.MEAN_REVERTING: {
        'trend': 1.0,    # Trend kendi filtrelerine güveniyor
        'mr':    1.0,    # MR rejimde MR tam risk
    },
    Regime.HIGH_VOLATILE: {
        'trend': 1.0,    # Trend her yerde çalışır
        'mr':    0.5,    # Yüksek vol'de MR dikkatli
    },
    Regime.LOW_VOLATILE: {
        'trend': 1.0,    # Trend kendi ADX filtresiyle korur
        'mr':    0.5,    # Düşük vol'de MR dikkatli
    },
}


class HybridBacktest:
    """
    Hibrit Backtest Engine V2.1
    - Rejim tespiti ile strateji seçimi
    - Multi-strateji: Trend + MR + Breakout
    - Trade yönetimi: SL/TP/Trail/Timeout
    """

    LOOKBACK = 65
    MAX_LOSS_DOLLAR = 4.0
    NOTIONAL_CAP = 300.0
    BALANCE_PCT = 0.20
    COOLDOWN_BARS = 2
    COOLDOWN_AFTER_3SL = 5
    MAX_SYMBOL_LOSS = 20.0      # Sembol başı max kayıp (toparlanma şansı ver)
    ST_PERIOD = 10
    ST_MULT = 3.0

    def __init__(self, initial_balance: float = 1000.0, leverage: int = 10):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.closed_trades: List[Dict] = []
        self.current_trade: Optional[Dict] = None
        self.trade_counter = 0
        self.last_close_bar = -10
        self.consecutive_sl = 0
        self.symbol_pnl: Dict[str, float] = {}  # Sembol başı toplam PnL

        # Rejim istatistikleri
        self.regime_stats: Dict[str, Dict] = {
            r.value: {'bars': 0, 'trades': 0, 'wins': 0, 'pnl': 0.0}
            for r in Regime
        }

    def _open_trade(self, symbol: str, direction: str, entry_price: float,
                    sl_price: float, tp_price: float, bar_idx: int,
                    timestamp: str, strategy: str, regime: str,
                    risk_mult: float = 1.0) -> bool:
        sl_dist = abs(entry_price - sl_price)
        if sl_dist <= 0:
            return False

        max_loss = self.MAX_LOSS_DOLLAR * risk_mult
        qty_risk = max_loss / sl_dist
        max_notional = min(self.balance * self.BALANCE_PCT, self.NOTIONAL_CAP)
        qty_notional = max_notional / entry_price
        qty = min(qty_risk, qty_notional)
        if qty <= 0:
            return False

        notional = qty * entry_price
        margin = notional / self.leverage
        if margin > self.balance * 0.5:
            return False

        self.balance -= margin
        self.trade_counter += 1

        self.current_trade = {
            'id': self.trade_counter,
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'qty': qty,
            'margin': margin,
            'entry_bar': bar_idx,
            'entry_time': timestamp,
            'strategy': strategy,
            'regime': regime,
            'max_profit_pct': 0.0,
            'peak_price': entry_price,
            'trail_active': False,
        }
        return True

    def _close_trade(self, exit_price: float, exit_reason: str,
                     timestamp: str, bar_idx: int = 0):
        t = self.current_trade
        if t is None:
            return
        self.last_close_bar = bar_idx

        entry = t['entry_price']
        qty = t['qty']
        margin = t['margin']
        pnl = (exit_price - entry) * qty if t['direction'] == 'LONG' else (entry - exit_price) * qty
        fees = (entry * qty + exit_price * qty) * 0.0002
        pnl -= fees
        self.balance += margin + pnl

        # Sembol PnL takibi
        sym = t.get('symbol', '')
        self.symbol_pnl[sym] = self.symbol_pnl.get(sym, 0.0) + pnl

        regime_val = t.get('regime', 'mean_reverting')
        if regime_val in self.regime_stats:
            self.regime_stats[regime_val]['trades'] += 1
            self.regime_stats[regime_val]['pnl'] += pnl
            if pnl > 0:
                self.regime_stats[regime_val]['wins'] += 1

        closed = {
            **t,
            'exit_price': exit_price,
            'exit_time': timestamp,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'pnl_pct': (pnl / margin) * 100 if margin > 0 else 0,
        }
        self.closed_trades.append(closed)

        if exit_reason in ('SL', 'MAXLOSS'):
            self.consecutive_sl += 1
        else:
            self.consecutive_sl = 0

        self.current_trade = None

    def run_backtest(self, symbol: str, days: int = 30) -> Dict:
        """Ana backtest döngüsü."""
        fetcher = DataFetcher('binance')
        limit = min(days * 24 + self.LOOKBACK + 200, 1000)
        df = fetcher.fetch_ohlcv(symbol, '1h', limit=limit)

        if df is None or len(df) < 100:
            return {'success': False, 'error': 'Insufficient data'}

        # İndikatörler + Supertrend
        df = add_all_indicators(df)
        df['st_direction'], df['st_value'] = supertrend(df, self.ST_PERIOD, self.ST_MULT)

        start_bar = self.LOOKBACK + 5

        for i in range(start_bar, len(df)):
            timestamp = str(df.index[i])

            # ── REJİM TESPİTİ ──
            window = df.iloc[max(0, i - 50):i + 1]
            regime, regime_details = detect_regime(window, lookback=50)
            regime_val = regime.value

            if regime_val in self.regime_stats:
                self.regime_stats[regime_val]['bars'] += 1

            # ── EXIT (aktif trade varsa) ──
            if self.current_trade is not None:
                strat = self.current_trade.get('strategy', '')
                result = None

                if strat == 'trend':
                    result = TrendStrategy.check_exit(self.current_trade, df, i)
                elif strat == 'mr':
                    result = MeanReversionStrategy.check_exit(self.current_trade, df, i)
                elif strat == 'breakout':
                    result = BreakoutStrategy.check_exit(self.current_trade, df, i)

                if result is not None:
                    reason, price = result
                    self._close_trade(price, reason, timestamp, i)

            # ── ENTRY (pozisyon yoksa) ──
            if self.current_trade is None:
                if self.consecutive_sl >= 3:
                    self.consecutive_sl = 0
                    self.last_close_bar = i

                if i - self.last_close_bar < self.COOLDOWN_BARS:
                    continue

                # Sembol bazlı kayıp limiti
                sym_loss = self.symbol_pnl.get(symbol, 0.0)
                if sym_loss < -self.MAX_SYMBOL_LOSS:
                    continue

                # V2.3: Tüm stratejileri dene, ilk geçerli sinyali al
                regime_risks = REGIME_RISK_MAP.get(regime, {})
                entry_signal = None
                chosen_strategy = None
                risk_mult = 1.0

                for strat_name in ALL_STRATEGIES:
                    strat_risk = regime_risks.get(strat_name, 0.5)
                    if strat_risk <= 0:
                        continue

                    if strat_name == 'trend':
                        entry_signal = TrendStrategy.check_entry(df, i)
                    elif strat_name == 'mr':
                        entry_signal = MeanReversionStrategy.check_entry(df, i, strat_risk)

                    if entry_signal is not None:
                        chosen_strategy = strat_name
                        risk_mult = strat_risk
                        break

                if entry_signal is not None and chosen_strategy is not None:
                    direction, sl, tp = entry_signal
                    self._open_trade(
                        symbol=symbol,
                        direction=direction,
                        entry_price=float(df['close'].iloc[i]),
                        sl_price=sl,
                        tp_price=tp,
                        bar_idx=i,
                        timestamp=timestamp,
                        strategy=chosen_strategy,
                        regime=regime_val,
                        risk_mult=risk_mult,
                    )

        # Son açık trade'i kapat
        if self.current_trade is not None:
            self._close_trade(
                float(df.iloc[-1]['close']), 'END',
                str(df.index[-1]), len(df) - 1
            )

        return self._build_results(symbol, days)

    def _build_results(self, symbol: str, days: int) -> Dict:
        total = len(self.closed_trades)
        if total == 0:
            return {
                'success': True, 'symbol': symbol, 'timeframe': '1h',
                'days': days, 'leverage': self.leverage,
                'initial_balance': self.initial_balance,
                'final_balance': round(self.balance, 2),
                'total_pnl': 0.0, 'total_pnl_pct': 0.0,
                'total_trades': 0, 'wins': 0, 'losses': 0,
                'win_rate': 0.0, 'avg_win': 0, 'avg_loss': 0,
                'profit_factor': 0, 'regime_stats': self.regime_stats,
                'strategy_breakdown': {}, 'trades': []
            }

        wins = [t for t in self.closed_trades if t['pnl'] > 0]
        losses = [t for t in self.closed_trades if t['pnl'] <= 0]
        total_profit = sum(t['pnl'] for t in wins)
        total_loss = abs(sum(t['pnl'] for t in losses))

        # Strateji kırılımı
        strategy_breakdown = {}
        for t in self.closed_trades:
            strat = t.get('strategy', 'unknown')
            if strat not in strategy_breakdown:
                strategy_breakdown[strat] = {
                    'trades': 0, 'wins': 0, 'pnl': 0.0
                }
            strategy_breakdown[strat]['trades'] += 1
            strategy_breakdown[strat]['pnl'] += t['pnl']
            if t['pnl'] > 0:
                strategy_breakdown[strat]['wins'] += 1

        for strat, data in strategy_breakdown.items():
            if data['trades'] > 0:
                data['win_rate'] = round(data['wins'] / data['trades'] * 100, 1)
                data['pnl'] = round(data['pnl'], 2)

        return {
            'success': True, 'symbol': symbol, 'timeframe': '1h',
            'days': days, 'leverage': self.leverage,
            'initial_balance': self.initial_balance,
            'final_balance': round(self.balance, 2),
            'total_pnl': round(self.balance - self.initial_balance, 2),
            'total_pnl_pct': round(((self.balance - self.initial_balance) / self.initial_balance) * 100, 2),
            'total_trades': total,
            'wins': len(wins), 'losses': len(losses),
            'win_rate': round((len(wins) / total) * 100, 2),
            'avg_win': round(total_profit / len(wins), 2) if wins else 0,
            'avg_loss': round(total_loss / len(losses), 2) if losses else 0,
            'profit_factor': round(total_profit / total_loss, 2) if total_loss > 0 else 99.0,
            'max_profit_trade': round(max((t['pnl'] for t in self.closed_trades), default=0), 2),
            'max_loss_trade': round(min((t['pnl'] for t in self.closed_trades), default=0), 2),
            'regime_stats': self.regime_stats,
            'strategy_breakdown': strategy_breakdown,
            'trades': [
                {
                    'id': t['id'], 'direction': t['direction'],
                    'strategy': t.get('strategy', '?'),
                    'regime': t.get('regime', '?'),
                    'entry_price': round(t['entry_price'], 6),
                    'exit_price': round(t['exit_price'], 6),
                    'sl_price': round(t['sl_price'], 6),
                    'entry_time': t['entry_time'],
                    'exit_time': t['exit_time'],
                    'pnl': round(t['pnl'], 2),
                    'pnl_pct': round(t['pnl_pct'], 2),
                    'exit_reason': t['exit_reason']
                }
                for t in self.closed_trades[-50:]
            ]
        }


# ══════════════════════════════════════════════════════════════════
#  MULTI-SYMBOL RUNNER
# ══════════════════════════════════════════════════════════════════

def run_hybrid_backtest(
    symbols: List[str] = None,
    days: int = 30,
    initial_balance: float = 1000.0,
    leverage: int = 10,
) -> Dict:
    """
    Birden fazla sembolde hibrit backtest çalıştır.
    Her sembol bağımsız — sonra toplam rapor.
    """
    if symbols is None:
        symbols = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
            'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT',
            'DOT/USDT', 'LINK/USDT',
        ]

    all_results = []
    total_pnl = 0.0
    total_trades = 0
    total_wins = 0

    # Toplam rejim istatistikleri
    combined_regime = {
        r.value: {'bars': 0, 'trades': 0, 'wins': 0, 'pnl': 0.0}
        for r in Regime
    }
    combined_strategy = {}

    print(f"\n{'='*70}")
    print(f"  HYBRID BACKTEST V2.1 — {len(symbols)} coins × {days} days")
    print(f"  Balance: ${initial_balance} | Leverage: {leverage}x")
    print(f"{'='*70}\n")

    for symbol in symbols:
        try:
            bt = HybridBacktest(initial_balance=initial_balance, leverage=leverage)
            result = bt.run_backtest(symbol, days)

            if not result.get('success'):
                print(f"  ✗ {symbol}: {result.get('error', 'unknown')}")
                continue

            pnl = result['total_pnl']
            trades = result['total_trades']
            wr = result['win_rate']
            wins = result['wins']

            total_pnl += pnl
            total_trades += trades
            total_wins += wins

            # Rejim istatistikleri birleştir
            for regime_val, stats in result.get('regime_stats', {}).items():
                if regime_val in combined_regime:
                    for k in ['bars', 'trades', 'wins']:
                        combined_regime[regime_val][k] += stats.get(k, 0)
                    combined_regime[regime_val]['pnl'] += stats.get('pnl', 0.0)

            # Strateji kırılımı birleştir
            for strat, data in result.get('strategy_breakdown', {}).items():
                if strat not in combined_strategy:
                    combined_strategy[strat] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
                combined_strategy[strat]['trades'] += data.get('trades', 0)
                combined_strategy[strat]['wins'] += data.get('wins', 0)
                combined_strategy[strat]['pnl'] += data.get('pnl', 0.0)

            icon = '✓' if pnl > 0 else '✗'
            # Trade detayları
            strat_detail = ', '.join(
                f"{s}:{d['trades']}t"
                for s, d in result.get('strategy_breakdown', {}).items()
            )
            print(
                f"  {icon} {symbol:12s} | PnL: ${pnl:+7.2f} | "
                f"Trades: {trades:2d} | WR: {wr:5.1f}% | {strat_detail}"
            )

            all_results.append(result)

        except Exception as e:
            print(f"  ✗ {symbol}: ERROR — {e}")

    # ── ÖZET ──
    wr_total = round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0

    print(f"\n{'='*70}")
    print(f"  TOPLAM: ${total_pnl:+.2f} | {total_trades} trades | WR: {wr_total}%")
    print(f"{'='*70}")

    print(f"\n  REJİM DAĞILIMI:")
    for regime_val, stats in combined_regime.items():
        bars = stats['bars']
        trades = stats['trades']
        wins = stats['wins']
        pnl = stats['pnl']
        wr = round(wins / trades * 100, 1) if trades > 0 else 0
        print(
            f"    {regime_val:18s} | {bars:5d} bars | "
            f"{trades:2d} trades | WR: {wr:5.1f}% | PnL: ${pnl:+.2f}"
        )

    print(f"\n  STRATEJİ KIRIMI:")
    for strat, data in combined_strategy.items():
        trades = data['trades']
        wins = data['wins']
        pnl = data['pnl']
        wr = round(wins / trades * 100, 1) if trades > 0 else 0
        print(
            f"    {strat:12s} | {trades:2d} trades | "
            f"WR: {wr:5.1f}% | PnL: ${pnl:+.2f}"
        )

    print(f"{'='*70}\n")

    return {
        'total_pnl': round(total_pnl, 2),
        'total_trades': total_trades,
        'total_wins': total_wins,
        'win_rate': wr_total,
        'regime_stats': combined_regime,
        'strategy_breakdown': combined_strategy,
        'per_symbol': all_results,
    }


# ══════════════════════════════════════════════════════════════════
#  COMPARISON: Hybrid vs Individual Strategies
# ══════════════════════════════════════════════════════════════════

def compare_strategies(
    symbols: List[str] = None,
    days: int = 30,
    initial_balance: float = 1000.0,
    leverage: int = 10,
) -> Dict:
    """
    Hibrit vs sadece Trend vs sadece MR karşılaştırması.
    """
    from backtest.trend_backtest import TrendBacktest

    if symbols is None:
        symbols = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
            'XRP/USDT', 'ADA/USDT', 'DOGE/USDT',
        ]

    results = {
        'hybrid': {'total_pnl': 0, 'trades': 0, 'wins': 0},
        'trend_only': {'total_pnl': 0, 'trades': 0, 'wins': 0},
    }

    for symbol in symbols:
        try:
            # Hybrid
            hbt = HybridBacktest(initial_balance=initial_balance, leverage=leverage)
            h_res = hbt.run_backtest(symbol, days)
            if h_res.get('success'):
                results['hybrid']['total_pnl'] += h_res['total_pnl']
                results['hybrid']['trades'] += h_res['total_trades']
                results['hybrid']['wins'] += h_res['wins']

            # Trend only
            tbt = TrendBacktest(initial_balance=initial_balance, leverage=leverage)
            t_res = tbt.run_backtest(symbol, days)
            if t_res.get('success'):
                results['trend_only']['total_pnl'] += t_res['total_pnl']
                results['trend_only']['trades'] += t_res['total_trades']
                results['trend_only']['wins'] += t_res['wins']

        except Exception as e:
            print(f"  Compare error {symbol}: {e}")

    print(f"\n{'='*70}")
    print(f"  KARŞILAŞTIRMA: Hybrid vs Trend Only ({len(symbols)} coins × {days} days)")
    print(f"{'='*70}")

    for name, data in results.items():
        pnl = data['total_pnl']
        trades = data['trades']
        wins = data['wins']
        wr = round(wins / trades * 100, 1) if trades > 0 else 0
        print(
            f"  {name:15s} | PnL: ${pnl:+8.2f} | "
            f"Trades: {trades:3d} | WR: {wr:5.1f}%"
        )

    print(f"{'='*70}\n")
    return results


# ══════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else 'hybrid'
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    if mode == 'compare':
        compare_strategies(days=days)
    else:
        run_hybrid_backtest(days=days)
