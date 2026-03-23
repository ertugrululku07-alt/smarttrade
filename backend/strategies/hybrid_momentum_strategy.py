import time
import math
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

class HybridMomentumStrategyMixin:
    """
    Zero-Lag Hybrid Momentum Engine
    Proven via 180-day optimization.
    15-bar Breakout + EMAs + Volume Filter + Tight Trailing Stop
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hybrid_enabled = True
        self._hybrid_margin_pct = 0.20  # 20% array
        self._hybrid_leverage = 10
        self.BREAKOUT_PERIOD = 15
        self.VOL_MULT = 1.2
        self.MIN_ADX = 20
        self.MAX_SL_PCT = 0.03

        # Trail parameters (Trend v4.4 core)
        self.TRAIL_START = 0.015
        self.TRAIL_KEEP = 0.40

    def _swing_low(self, df, i, n=5):
        return float(df['low'].iloc[max(0, i-n):i+1].min())

    def _swing_high(self, df, i, n=5):
        return float(df['high'].iloc[max(0, i-n):i+1].max())

    def check_hybrid_momentum_entry(self, symbol: str, df: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """Check for Fast Breakout + EMA direction."""
        if not self._hybrid_enabled or len(df) < self.BREAKOUT_PERIOD + 2:
            return None

        i = len(df) - 1
        
        # Check active trades for the strategy
        active_for_strat = [t for t in self.open_trades if t.get('strategy') == 'hybrid_momentum']
        if len(active_for_strat) >= 3: # Max 3 concurrent
            return None

        # Check cooldown
        if time.time() < getattr(self, '_cooldown_until', 0):
            return None

        close = float(df['close'].iloc[i])
        ema9 = float(df['ema9'].iloc[i])
        ema21 = float(df['ema21'].iloc[i])
        adx = float(df['adx'].iloc[i])
        vol_ratio = float(df['vol_ratio_20'].iloc[i])
        rsi_val = float(df['rsi'].iloc[i])

        n = self.BREAKOUT_PERIOD
        prev_high = float(df['high'].iloc[max(0, i-n):i].max())
        prev_low = float(df['low'].iloc[max(0, i-n):i].min())

        direction = None
        sl = None

        if ema9 > ema21 and close > prev_high:
            direction = 'LONG'
            sl = self._swing_low(df, i, 5)
        elif ema9 < ema21 and close < prev_low:
            direction = 'SHORT'
            sl = self._swing_high(df, i, 5)

        if not direction:
            return None
        if adx < self.MIN_ADX:
            return None
        # Strict Volume check
        if vol_ratio < self.VOL_MULT:
            return None
        
        # Chop / Fakeout filters
        if direction == 'LONG' and rsi_val > 70:
            return None
        if direction == 'SHORT' and rsi_val < 30:
            return None

        # Determine Stop Loss distance and cap
        if direction == 'LONG':
            sl = max(sl, close * (1 - self.MAX_SL_PCT))
            if sl >= close * 0.997: return None # Too close
        else:
            sl = min(sl, close * (1 + self.MAX_SL_PCT))
            if sl <= close * 1.003: return None # Too close

        # Margin calculation: 20% of current overall balance
        risk_margin = min(self.balance * self._hybrid_margin_pct, self.balance)
        if risk_margin < 20: 
            return None # Insufficient funds
            
        qty = (risk_margin * self._hybrid_leverage) / current_price

        return {
            'symbol': symbol,
            'direction': direction,
            'entry_price': current_price,
            'sl_price': sl,
            'qty': qty,
            'margin': risk_margin,
            'strategy': 'hybrid_momentum',
            'reason': f"Zero-Lag Breakout (Vol: {vol_ratio:.1f})",
            'atr': float(df['atr'].iloc[i]) if 'atr' in df.columns else current_price * 0.01
        }

    def _check_hybrid_momentum_exit(self, t: Dict, current_price: float) -> Optional[Tuple[str, float]]:
        """Check for exit conditions for the Hybrid engine."""
        entry = t['entry_price']
        sl = t['sl_price']
        direction = t['side']
        
        if direction == 'LONG':
            if current_price > t.get('peak_price', entry):
                t['peak_price'] = current_price
            peak = t.get('peak_price', current_price)
            peak_pnl_pct = (peak - entry) / entry
        else:
            if current_price < t.get('peak_price', entry):
                t['peak_price'] = current_price
            peak = t.get('peak_price', current_price)
            peak_pnl_pct = (entry - peak) / entry

        # ── Tier 1: Breakeven ──
        if peak_pnl_pct >= 0.015 and not t.get('_be_active'):
            t['_be_active'] = True
            be = entry * 1.002 if direction == 'LONG' else entry * 0.998
            if (direction == 'LONG' and be > sl) or (direction == 'SHORT' and be < sl):
                t['sl_price'] = be
                sl = be

        # ── Tier 2: Tight Margin-Scaled Trail ──
        keep_ratio = 0.0
        if peak_pnl_pct >= self.TRAIL_START: 
            keep_ratio = self.TRAIL_KEEP 
            if peak_pnl_pct >= 0.050:
                keep_ratio = 0.85
            elif peak_pnl_pct >= 0.035:
                keep_ratio = 0.70
            elif peak_pnl_pct >= 0.025:
                keep_ratio = 0.60
                
        if keep_ratio > 0:
            if direction == 'LONG':
                trail_sl = entry * (1 + (peak_pnl_pct * keep_ratio))
                if trail_sl > sl: t['sl_price'] = trail_sl; sl = trail_sl
            else:
                trail_sl = entry * (1 - (peak_pnl_pct * keep_ratio))
                if trail_sl < sl: t['sl_price'] = trail_sl; sl = trail_sl

        # Exit triggered
        if direction == 'LONG' and current_price <= sl:
            return ('SL_OR_TRAIL', current_price)
        if direction == 'SHORT' and current_price >= sl:
            return ('SL_OR_TRAIL', current_price)

        # Timeout 48 bars (2 days to keep margin agile)
        # Using approximated time since we don't track bar counts accurately in live
        try:
            from datetime import datetime
            et = datetime.fromisoformat(str(t['entry_time']).replace('Z',''))
            age_hours = (datetime.now() - et).total_seconds() / 3600
            if age_hours >= 72:
                return ('TIMEOUT', current_price)
        except:
             pass

        return None

    def _hybrid_scan(self, fetcher):
        """Called by live_trader scanner loop"""
        for sym in self.scanned_symbols:
            if not self.is_running:
                break
                
            # Do not enter if maximum global capacity reached
            with self.trades_lock:
                if len(self.open_trades) >= self.max_open_trades_limit:
                    return

            try:
                df = fetcher.fetch_ohlcv(sym, "1h", limit=50)
                if df is None or len(df) < 25:
                    continue
                    
                from backtest.signals import add_all_indicators
                df = add_all_indicators(df)
                
                cp = float(df['close'].iloc[-1])
                res = self.check_hybrid_momentum_entry(sym, df, cp)
                
                if res:
                    # ── Real-Time Order Book (BBO) Imbalance Check ──
                    try:
                        ob = fetcher.exchange.fetch_order_book(sym, limit=20)
                        bids = ob.get('bids', [])
                        asks = ob.get('asks', [])
                        
                        bid_vol = sum(b[1] for b in bids)
                        ask_vol = sum(a[1] for a in asks)
                        
                        if bid_vol > 0 and ask_vol > 0:
                            if res['direction'] == 'LONG':
                                imbalance = bid_vol / ask_vol
                                if imbalance < 1.20:
                                    if hasattr(self, 'log'):
                                        self.log(f"[OB REJECT] {sym} LONG iptal: Alim Baskisi={imbalance:.2f}x (Min: 1.20x)")
                                    continue
                            else:
                                imbalance = ask_vol / bid_vol
                                if imbalance < 1.20:
                                    if hasattr(self, 'log'):
                                        self.log(f"[OB REJECT] {sym} SHORT iptal: Satis Baskisi={imbalance:.2f}x (Min: 1.20x)")
                                    continue
                    except Exception as ob_err:
                        if hasattr(self, 'log'):
                            self.log(f"  [OB WARN] {sym} Order Book error: {ob_err}")
                        pass # OB verisi alinamadiysa isleme gecikmesiz devam et

                    self._open(
                        symbol=sym,
                        side=res['direction'],
                        price=cp,
                        multiplier=1.0,  # We handle sizing inside the checked dictionary response
                        sl_price=res['sl_price'],
                        absolute_qty=res['qty'],
                        atr=res.get('atr', cp * 0.01),
                        signal_result=res
                    )
            except Exception as e:
                if hasattr(self, 'log'):
                    self.log(f"  [SCAN ERROR] {sym}: {e}")
            time.sleep(0.3)
            
        if hasattr(self, 'log'):
            symbols_count = len(getattr(self, 'scanned_symbols', []))
            self.log(f"[OK] Zero-Lag Hybrid finished scanning {symbols_count} markets.")

