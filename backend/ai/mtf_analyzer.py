"""
Multi-Timeframe (MTF) Intelligence Module v1.0

Phase 1: 4h Trend Filter + Primary TF Execution

Architecture:
  4h Trend Filter -> Direction filter (EMA, Structure, RSI)
  Primary TF Execution -> Tactical signals (1h/15m)
  Result -> Filtered Signal
"""

import time
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from enum import Enum


# =====================================================================
# Constants & Enums
# =====================================================================

class TrendBias(Enum):
    """4h trend direction."""
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"


class TradeFilter(Enum):
    """Strategy filter decision."""
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    BOTH = "both"
    NO_TRADE = "no_trade"


@dataclass
class HTFContext:
    """Higher Timeframe (4h) analysis result."""
    bias: TrendBias
    trade_filter: TradeFilter
    ema_fast: float             # EMA 21
    ema_slow: float             # EMA 50
    ema_trend: float            # EMA 200
    rsi: float
    atr: float
    trend_strength: float       # 0.0 - 1.0
    structure: str              # 'HH_HL', 'LH_LL', 'RANGE'
    last_update: float          # timestamp
    reason: str


@dataclass
class MTFSignalAdjustment:
    """Effect of MTF filter on signal."""
    score_modifier: int         # -2 to +2
    sl_multiplier: float        # SL extension/contraction
    tp_multiplier: float        # TP extension/contraction
    allowed: bool               # Is signal valid?
    reason: str


# =====================================================================
# 4h Data Cache
# =====================================================================

@dataclass
class HTFCache:
    """4h data and analysis cache."""
    df: Optional[pd.DataFrame] = None
    context: Optional[HTFContext] = None
    last_fetch: float = 0.0
    fetch_interval: float = 900.0  # 15 minutes (4h bar)

    def is_stale(self) -> bool:
        """Does cache need refresh?"""
        return (
            self.df is None
            or self.context is None
            or (time.time() - self.last_fetch) > self.fetch_interval
        )


# =====================================================================
# MTF Analyzer
# =====================================================================

class MTFAnalyzer:
    """
    Multi-Timeframe analysis engine.
    Phase 1: 4h Trend Filtering.
    """

    # Parameters
    EMA_FAST = 21
    EMA_SLOW = 50
    EMA_TREND = 200

    RSI_BULL_THRESHOLD = 55
    RSI_BEAR_THRESHOLD = 45
    RSI_STRONG_BULL = 65
    RSI_STRONG_BEAR = 35

    def __init__(self):
        self._cache: Dict[str, HTFCache] = {}

    def analyze_htf(self, df_4h: pd.DataFrame, symbol: str = "default") -> Optional[HTFContext]:
        if symbol in self._cache:
            cache = self._cache[symbol]
            if not cache.is_stale() and cache.context is not None:
                return cache.context

        if df_4h is None or len(df_4h) < 50:
            return None

        # Conservative: remove last (unfinished) bar
        df = df_4h.iloc[:-1].copy()
        if len(df) < 50: return None

        ema_fast = self._ema(df['close'], self.EMA_FAST)
        ema_slow = self._ema(df['close'], self.EMA_SLOW)
        ema_trend = self._ema(df['close'], self.EMA_TREND)

        lc = float(df['close'].iloc[-1])
        lef, les, let = float(ema_fast.iloc[-1]), float(ema_slow.iloc[-1]), float(ema_trend.iloc[-1])

        rsi_s = self._rsi(df['close'], 14)
        lr = float(rsi_s.iloc[-1]) if not rsi_s.empty else 50.0

        atr_s = self._atr(df, 14)
        la = float(atr_s.iloc[-1]) if not atr_s.empty else 0.0

        structure = self._analyze_structure(df)
        strength = self._calculate_trend_strength(lc, lef, les, let, lr)
        bias = self._determine_bias(lc, lef, les, let, lr, structure)
        trade_filter = self._bias_to_filter(bias)
        reason = self._build_reason(bias, lc, let, lr, structure)

        context = HTFContext(
            bias=bias, trade_filter=trade_filter, ema_fast=lef, ema_slow=les,
            ema_trend=let, rsi=lr, atr=la, trend_strength=strength,
            structure=structure, last_update=time.time(), reason=reason
        )

        self._cache[symbol] = HTFCache(df=df_4h, context=context, last_fetch=time.time())
        return context

    def get_signal_adjustment(self, htf_context: Optional[HTFContext], direction: str) -> MTFSignalAdjustment:
        if htf_context is None:
            return MTFSignalAdjustment(0, 1.0, 1.0, True, "No 4h data, neutral")

        bias, strength = htf_context.bias, htf_context.trend_strength

        if direction == 'LONG':
            if bias == TrendBias.STRONG_BULL: return MTFSignalAdjustment(2, 1.0, 1.3, True, f"4h Strong Bull (str={strength:.2f})")
            if bias == TrendBias.BULL: return MTFSignalAdjustment(1, 1.0, 1.15, True, f"4h Bull (str={strength:.2f})")
            if bias == TrendBias.NEUTRAL: return MTFSignalAdjustment(0, 1.0, 1.0, True, "4h Neutral")
            if bias == TrendBias.BEAR: return MTFSignalAdjustment(-2, 0.7, 0.7, False, f"4h Bear -> LONG REJECTED (str={strength:.2f})")
            return MTFSignalAdjustment(-2, 0.7, 0.7, False, f"4h Strong Bear -> LONG REJECTED (str={strength:.2f})")
        
        elif direction == 'SHORT':
            if bias == TrendBias.STRONG_BEAR: return MTFSignalAdjustment(2, 1.0, 1.3, True, f"4h Strong Bear (str={strength:.2f})")
            if bias == TrendBias.BEAR: return MTFSignalAdjustment(1, 1.0, 1.15, True, f"4h Bear (str={strength:.2f})")
            if bias == TrendBias.NEUTRAL: return MTFSignalAdjustment(0, 1.0, 1.0, True, "4h Neutral")
            if bias == TrendBias.BULL: return MTFSignalAdjustment(-2, 0.7, 0.7, False, f"4h Bull -> SHORT REJECTED (str={strength:.2f})")
            return MTFSignalAdjustment(-2, 0.7, 0.7, False, f"4h Strong Bull -> SHORT REJECTED (str={strength:.2f})")

        return MTFSignalAdjustment(0, 1.0, 1.0, True, f"Unknown direction: {direction}")

    def generate_cross_tf_features(self, htf_context: Optional[HTFContext], cp: float, pa: float) -> Dict[str, float]:
        f = {
            'htf_bias_numeric': 0.0, 
            'htf_trend_strength': 0.0, 
            'htf_rsi': 50.0, 
            'htf_price_vs_ema200': 0.0, 
            'htf_ema_alignment': 0.0, 
            'htf_structure_numeric': 0.0
        }
        if htf_context is None: return f
        
        bmap = {
            TrendBias.STRONG_BULL: 2.0, 
            TrendBias.BULL: 1.0, 
            TrendBias.NEUTRAL: 0.0, 
            TrendBias.BEAR: -1.0, 
            TrendBias.STRONG_BEAR: -2.0
        }
        f['htf_bias_numeric'] = bmap.get(htf_context.bias, 0.0)
        f['htf_trend_strength'], f['htf_rsi'] = htf_context.trend_strength, htf_context.rsi
        if htf_context.atr > 0: f['htf_price_vs_ema200'] = (cp - htf_context.ema_trend) / htf_context.atr
        
        if htf_context.ema_fast > htf_context.ema_slow > htf_context.ema_trend: 
            f['htf_ema_alignment'] = 1.0
        elif htf_context.ema_fast < htf_context.ema_slow < htf_context.ema_trend: 
            f['htf_ema_alignment'] = -1.0
        
        smap = {'HH_HL': 1.0, 'LH_LL': -1.0, 'RANGE': 0.0}
        f['htf_structure_numeric'] = smap.get(htf_context.structure, 0.0)
        return f

    def prepare_bulk_mtf_features(self, df_tactical: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
        """
        Injects 4h context features into tactical (15m/1h) dataframe.
        Uses merge_asof to prevent look-ahead bias.
        Produces all 6 HTF features expected by MetaTrainer.
        """
        if df_tactical.empty or df_4h.empty:
            return df_tactical

        # Bulk HTF analysis
        df_4h = df_4h.copy()
        df_4h['ema21'] = self._ema(df_4h['close'], 21)
        df_4h['ema50'] = self._ema(df_4h['close'], 50)
        df_4h['ema200'] = self._ema(df_4h['close'], 200)
        df_4h['rsi'] = self._rsi(df_4h['close'], 14)
        df_4h['atr'] = self._atr(df_4h, 14)

        # 1. htf_bias_numeric (aggregate score)
        df_4h['htf_bias_numeric'] = 0.0
        df_4h['htf_bias_numeric'] += np.where(df_4h['close'] > df_4h['ema200'], 1, -1)
        df_4h['htf_bias_numeric'] += np.where(
            (df_4h['ema21'] > df_4h['ema50']) & (df_4h['ema50'] > df_4h['ema200']), 1, 0
        )
        df_4h['htf_bias_numeric'] -= np.where(
            (df_4h['ema21'] < df_4h['ema50']) & (df_4h['ema50'] < df_4h['ema200']), 1, 0
        )
        df_4h['htf_bias_numeric'] += np.where(df_4h['rsi'] > 65, 1.5, np.where(df_4h['rsi'] > 55, 1, 0))
        df_4h['htf_bias_numeric'] -= np.where(df_4h['rsi'] < 35, 1.5, np.where(df_4h['rsi'] < 45, 1, 0))

        # 2. htf_price_vs_ema200 (ATR-normalized distance)
        df_4h['htf_price_vs_ema200'] = (df_4h['close'] - df_4h['ema200']) / df_4h['atr'].replace(0, 1e-10)

        # 3. htf_trend_strength (0.0 - 1.0)
        def _calc_strength(row):
            c, e21, e50, e200, r = row['close'], row['ema21'], row['ema50'], row['ema200'], row['rsi']
            scores = []
            if e200 > 0:
                scores.append(min((abs(c - e200) / e200) * 20, 1.0))
            scores.append(abs(r - 50) / 50)
            if (e21 > e50 > e200) or (e21 < e50 < e200):
                scores.append(1.0)
            elif (e21 > e200) or (e21 < e200):
                scores.append(0.5)
            else:
                scores.append(0.0)
            return sum(scores) / len(scores) if scores else 0.0

        df_4h['htf_trend_strength'] = df_4h.apply(_calc_strength, axis=1)

        # 4. htf_ema_alignment (+1 bull, -1 bear, 0 mixed)
        df_4h['htf_ema_alignment'] = 0.0
        bull_aligned = (df_4h['ema21'] > df_4h['ema50']) & (df_4h['ema50'] > df_4h['ema200'])
        bear_aligned = (df_4h['ema21'] < df_4h['ema50']) & (df_4h['ema50'] < df_4h['ema200'])
        df_4h.loc[bull_aligned, 'htf_ema_alignment'] = 1.0
        df_4h.loc[bear_aligned, 'htf_ema_alignment'] = -1.0

        # 5. htf_structure_numeric (+1 HH_HL, -1 LH_LL, 0 RANGE)
        df_4h['htf_structure_numeric'] = 0.0
        h_vals = df_4h['high'].values
        l_vals = df_4h['low'].values
        n_struct = 3
        for i in range(n_struct, len(df_4h) - n_struct):
            # Find swing highs and lows in local window
            window_h = h_vals[max(0, i-10):i+1]
            window_l = l_vals[max(0, i-10):i+1]
            if len(window_h) >= 6:
                sh = []
                sl = []
                for j in range(n_struct, len(window_h) - n_struct):
                    if window_h[j] == max(window_h[j-n_struct:j+n_struct+1]):
                        sh.append(float(window_h[j]))
                    if window_l[j] == min(window_l[j-n_struct:j+n_struct+1]):
                        sl.append(float(window_l[j]))
                if len(sh) >= 2 and len(sl) >= 2:
                    hh = sh[-1] > sh[-2]
                    hl = sl[-1] > sl[-2]
                    lh = sh[-1] < sh[-2]
                    ll = sl[-1] < sl[-2]
                    if hh and hl:
                        df_4h.iloc[i, df_4h.columns.get_loc('htf_structure_numeric')] = 1.0
                    elif lh and ll:
                        df_4h.iloc[i, df_4h.columns.get_loc('htf_structure_numeric')] = -1.0

        # 6. htf_rsi and htf_atr (renamed)
        # Assemble all HTF columns
        htf_cols = [
            'htf_bias_numeric', 'htf_price_vs_ema200', 'htf_trend_strength',
            'htf_ema_alignment', 'htf_structure_numeric', 'rsi', 'atr'
        ]
        htf_features = df_4h[htf_cols].rename(columns={'rsi': 'htf_rsi', 'atr': 'htf_atr'})
        
        # Merge Asof (Look-ahead bias protection: tactical_time >= htf_time)
        df_tactical = df_tactical.sort_index()
        htf_features = htf_features.sort_index()
        
        merged = pd.merge_asof(
            df_tactical, 
            htf_features, 
            left_index=True, 
            right_index=True, 
            direction='backward'
        )
        
        return merged

    def _determine_bias(self, close, lef, les, let, rsi, structure) -> TrendBias:
        bull, bear = 0, 0
        if close > let: bull += 1
        elif close < let: bear += 1
        if lef > les > let: bull += 1
        elif lef < les < let: bear += 1
        if rsi > self.RSI_STRONG_BULL: bull += 1.5
        elif rsi > self.RSI_BULL_THRESHOLD: bull += 1
        elif rsi < self.RSI_STRONG_BEAR: bear += 1.5
        elif rsi < self.RSI_BEAR_THRESHOLD: bear += 1
        if structure == 'HH_HL': bull += 1
        elif structure == 'LH_LL': bear += 1

        if bull >= 3.0: return TrendBias.STRONG_BULL
        if bull >= 2.0: return TrendBias.BULL
        if bear >= 3.0: return TrendBias.STRONG_BEAR
        if bear >= 2.0: return TrendBias.BEAR
        return TrendBias.NEUTRAL

    def _bias_to_filter(self, bias: TrendBias) -> TradeFilter:
        return {
            TrendBias.STRONG_BULL: TradeFilter.LONG_ONLY, 
            TrendBias.BULL: TradeFilter.BOTH, 
            TrendBias.NEUTRAL: TradeFilter.BOTH, 
            TrendBias.BEAR: TradeFilter.BOTH, 
            TrendBias.STRONG_BEAR: TradeFilter.SHORT_ONLY
        }.get(bias, TradeFilter.BOTH)

    def _calculate_trend_strength(self, close, lef, les, let, rsi) -> float:
        scores = []
        if let > 0: scores.append(min((abs(close - let) / let) * 20, 1.0))
        scores.append(abs(rsi - 50) / 50)
        if (lef > les > let or lef < les < let): scores.append(1.0)
        elif (lef > let or lef < let): scores.append(0.5)
        else: scores.append(0.0)
        return round(sum(scores) / len(scores), 3) if scores else 0.0

    def _analyze_structure(self, df: pd.DataFrame) -> str:
        if len(df) < 30: return 'RANGE'
        h, l = df['high'].values[-30:], df['low'].values[-30:]
        n, sh, sl = 3, [], []
        for i in range(n, len(h) - n):
            if h[i] == max(h[i-n:i+n+1]): sh.append(float(h[i]))
            if l[i] == min(l[i-n:i+n+1]): sl.append(float(l[i]))
        if len(sh) < 2 or len(sl) < 2: return 'RANGE'
        hh, hl, lh, ll = sh[-1] > sh[-2], sl[-1] > sl[-2], sh[-1] < sh[-2], sl[-1] < sl[-2]
        if hh and hl: return 'HH_HL'
        if lh and ll: return 'LH_LL'
        return 'RANGE'

    def _ema(self, s, p): return s.ewm(span=p, adjust=False).mean()
    def _rsi(self, s, p=14):
        d = s.diff()
        g, l = d.where(d > 0, 0.0), (-d).where(d < 0, 0.0)
        ag, al = g.ewm(span=p, adjust=False).mean(), l.ewm(span=p, adjust=False).mean()
        rs = ag / al.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))
    def _atr(self, df, p=14):
        h, l, c = df['high'], df['low'], df['close'].shift(1)
        tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
        return tr.ewm(span=p, adjust=False).mean()

    def _build_reason(self, bias, close, let, rsi, structure) -> str:
        return f"4h {bias.value} | Price {'above' if close > let else 'below'} EMA200 | RSI={rsi:.1f} | Structure={structure}"
