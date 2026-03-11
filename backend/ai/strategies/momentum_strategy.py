"""
Momentum Strategy v1.0 (Original)

Rejim: TRENDING
Mantık: Trend yönünde güçlü kırılımları yakala
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):

    name = "momentum"
    regime = "trending"
    default_tp_mult = 2.5
    default_sl_mult = 1.3

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < 50:
            return self._no_signal("Yetersiz veri")

        # --- Data Extraction ---
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        c = last_row['close']
        h = last_row['high']
        l = last_row['low']
        o = last_row['open']
        atr = last_row['atr']
        rsi = last_row['rsi']
        adx = last_row['adx']
        ema9 = last_row['ema9']
        ema21 = last_row['ema21']
        ema50 = last_row['ema50']
        vol = last_row['volume']
        vol_ma = df['volume'].rolling(20).mean().iloc[-1]
        
        # --- HARD FILTERS (Mandatory) ---
        # 1. No Nuke: Son mum aşırı büyük mü? (Volatility Spike Protect)
        hard1_noNuke = (h - l) < 3.5 * atr
        
        # 2. Volume Alive: Hacim ölü mü?
        hard2_volAlive = vol > vol_ma * 0.4
        
        # 3. Market Structure / Trend Direction
        is_breakout = (c > prev_row['high']) and (vol > vol_ma * 1.2)
        is_pullback = (c > ema50) and (l <= ema9 * 1.01) # EMA9 civarına çekilme (Pullback)
        
        hard4_trendDir = is_breakout or is_pullback
        
        hard_pass = hard1_noNuke and hard2_volAlive and hard4_trendDir
        
        if not hard_pass:
            reason = "HardFail: "
            if not hard1_noNuke: reason += "Nuke;"
            if not hard2_volAlive: reason += "VolDead;"
            if not hard4_trendDir: reason += "NoBO/PB;"
            return self._no_signal(reason, hard_pass=False)

        # --- SOFT FILTERS (Scoring 0-5) ---
        soft_score = 0
        
        # S1: EMA Stretch (Aşırı uzama kontrolü)
        ema_dist_pct = abs(c - ema21) / ema21 * 100
        if ema_dist_pct < 2.0: soft_score += 1
        
        # S2: RSI Zone (Giriş için ideal bölge)
        if 42 <= rsi <= 68: soft_score += 1
        
        # S3: Candle Range (Sakin mumlar)
        if (h - l) < 2.0 * atr: soft_score += 1
        
        # S4: ADX Strength
        if adx > 22: soft_score += 1
        
        # S5: Volume Flow (Artan hacim)
        if vol > prev_row['volume']: soft_score += 1
        
        # --- Entry Type Logic ---
        entry_type = "breakout" if is_breakout else "pullback"
        
        # LONG Direction Check
        if c > ema50 and last_row['di_plus'] > last_row['di_minus']:
            # Pullback ise TP/SL daha sıkı (swing bazlı) set edilebilir
            tp_price = 0.0
            sl_price = 0.0
            if entry_type == "pullback":
                tp_price = df['high'].rolling(10).max().iloc[-1] # Son 10 bar zirvesi
                sl_price = df['low'].rolling(10).min().iloc[-1]  # Son 10 bar dibi
            
            return self._long_signal(
                soft_score=soft_score,
                reason=f"Score={soft_score}/5 | {entry_type}",
                entry_price=c,
                entry_type=entry_type,
                tp_price=tp_price,
                sl_price=sl_price
            )
            
        return self._no_signal(f"Trend direction mismatch", hard_pass=True, soft_score=soft_score)
