import sys, os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from datetime import datetime
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators

class PortfolioSimV2:
    def __init__(self, balance=1000.0):
        self.balance = balance
        self.open_trades = []
        self.closed_trades = []
        self.peak_balance = balance
        self.max_dd = 0.0
        
        self.BP = 15
        self.VM = 1.2
        self.MIN_ADX = 20
        self.LEV = 10
        self.MAX_SL_PCT = 0.03
        
        self.TRAIL_START = 0.015
        self.TRAIL_KEEP = 0.40

    def get_same_dir_exposure(self, direction):
        return sum(1 for t in self.open_trades if t['side'] == direction)

    def run(self, days=180):
        fetcher = DataFetcher('binance')
        symbols = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
            "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "POL/USDT"
        ]
        
        dfs = {}
        all_times = set()
        print("Fetching and processing data...")
        for sym in symbols:
            df = fetcher.fetch_ohlcv(sym, "1h", limit=days * 24 + 100)
            if df is not None and len(df) > 50:
                df = add_all_indicators(df)
                dfs[sym] = df
                for t in df.index: all_times.add(t)
        
        sorted_times = sorted(list(all_times))
        print(f"Unified timeline: {len(sorted_times)} hours.")
        
        for t_idx, ts in enumerate(sorted_times):
            # 1. Update Exits
            for sym in list(set(op['symbol'] for op in self.open_trades)):
                if sym not in dfs or ts not in dfs[sym].index: continue
                
                row = dfs[sym].loc[ts]
                curr_price = float(row['close'])
                
                for op in self.open_trades[:]:
                    if op['symbol'] != sym: continue
                    
                    entry = op['entry_price']
                    direction = op['side']
                    sl = op['sl_price']
                    
                    if direction == 'LONG':
                        if curr_price > op['peak_price']: op['peak_price'] = curr_price
                        peak_pnl_pct = (op['peak_price'] - entry) / entry
                        curr_pnl_pct = (curr_price - entry) / entry
                    else:
                        if curr_price < op['peak_price']: op['peak_price'] = curr_price
                        peak_pnl_pct = (entry - op['peak_price']) / entry
                        curr_pnl_pct = (entry - curr_price) / entry
                    
                    # 3) Partial TP at 2.5%
                    if peak_pnl_pct >= 0.025 and not op.get('_tp1'):
                        op['_tp1'] = True
                        close_qty = op['qty'] * 0.40
                        pnl = (curr_price - entry) * close_qty if direction == 'LONG' else (entry - curr_price) * close_qty
                        margin_released = (close_qty * entry) / self.LEV
                        pnl -= (close_qty * entry + close_qty * curr_price) * 0.001
                        self.balance += margin_released + pnl
                        op['qty'] -= close_qty
                        op['margin'] -= margin_released
                        op['realized_pnl'] += pnl
                    
                    # Trail Tiers (V1 Logic)
                    if peak_pnl_pct >= 0.015 and not op.get('_be'):
                        op['_be'] = True
                        be = entry * 1.002 if direction == 'LONG' else entry * 0.998
                        if direction == 'LONG' and be > sl: op['sl_price'] = be
                        elif direction == 'SHORT' and be < sl: op['sl_price'] = be
                        
                    keep = 0.0
                    if peak_pnl_pct >= 0.015: keep = 0.40
                    if peak_pnl_pct >= 0.025: keep = 0.60
                    if peak_pnl_pct >= 0.035: keep = 0.70
                    if peak_pnl_pct >= 0.050: keep = 0.85
                    
                    if keep > 0:
                        if direction == 'LONG':
                            trail = entry * (1 + peak_pnl_pct * keep)
                            if trail > op['sl_price']: op['sl_price'] = trail
                        else:
                            trail = entry * (1 - peak_pnl_pct * keep)
                            if trail < op['sl_price']: op['sl_price'] = trail
                    
                    sl = op['sl_price']
                    
                    # Target Hits
                    exit_reason = None
                    if direction == 'LONG' and curr_price <= sl: exit_reason = 'SL_TRAIL'
                    if direction == 'SHORT' and curr_price >= sl: exit_reason = 'SL_TRAIL'
                    
                    # Age
                    age_h = (ts - op['entry_time']).total_seconds() / 3600
                    if age_h >= 72: exit_reason = 'TIMEOUT'
                    
                    if exit_reason:
                        pnl = (curr_price - entry) * op['qty'] if direction == 'LONG' else (entry - curr_price) * op['qty']
                        pnl -= (op['qty'] * entry + op['qty'] * curr_price) * 0.001
                        self.balance += op['margin'] + pnl
                        op['realized_pnl'] += pnl
                        op['exit_time'] = ts
                        op['exit_price'] = curr_price
                        op['reason'] = exit_reason
                        self.closed_trades.append(op)
                        self.open_trades.remove(op)
                        
                        if self.balance > self.peak_balance: self.peak_balance = self.balance
                        dd = (self.peak_balance - self.balance) / self.peak_balance
                        if dd > self.max_dd: self.max_dd = dd

            # 2. Update Entries
            for sym in symbols:
                if sym not in dfs: continue
                # We need historical positional index for iloc.
                # Find iloc of ts
                df = dfs[sym]
                if ts not in df.index: continue
                # Exact integer index isn't directly t_idx because symbols have different histories.
                # Use get_loc
                i = df.index.get_loc(ts)
                if i < self.BP + 2: continue
                
                close = float(df['close'].iloc[i])
                ema9 = float(df['ema9'].iloc[i])
                ema21 = float(df['ema21'].iloc[i])
                adx = float(df['adx'].iloc[i])
                vol = float(df['vol_ratio_20'].iloc[i])
                rsi = float(df['rsi'].iloc[i])
                
                prev_high = float(df['high'].iloc[max(0, i-self.BP):i].max())
                prev_low = float(df['low'].iloc[max(0, i-self.BP):i].min())
                
                direction = None
                sl = None
                
                if ema9 > ema21 and close > prev_high:
                    direction = 'LONG'
                    sl = float(df['low'].iloc[max(0, i-5):i+1].min())
                elif ema9 < ema21 and close < prev_low:
                    direction = 'SHORT'
                    sl = float(df['high'].iloc[max(0, i-5):i+1].max())
                
                if not direction: continue
                if adx < self.MIN_ADX or vol < self.VM: continue
                if direction == 'LONG' and rsi > 70: continue
                if direction == 'SHORT' and rsi < 30: continue
                
                if direction == 'LONG':
                    sl = max(sl, close * (1 - self.MAX_SL_PCT))
                    if sl >= close * 0.997: continue
                else:
                    sl = min(sl, close * (1 + self.MAX_SL_PCT))
                    if sl <= close * 1.003: continue
                    
                # User Requirement 2: Same direction max 2
                if self.get_same_dir_exposure(direction) >= 2: continue
                
                if len(self.open_trades) >= 3: continue # Max 3 concurrent total
                
                # User Requirement 1: Risk based sizing (Max 2% loss per trade)
                risk_amount = self.balance * 0.02
                sl_distance = abs(close - sl)
                qty = risk_amount / sl_distance if sl_distance > 0 else 0
                max_margin = self.balance * 0.15
                max_qty = (max_margin * self.LEV) / close
                qty = min(qty, max_qty)
                margin = (qty * close) / self.LEV
                
                if margin < 10 or self.balance < margin: continue
                
                self.balance -= margin
                self.open_trades.append({
                    'symbol': sym, 'side': direction, 'entry_price': close,
                    'qty': qty, 'margin': margin, 'sl_price': sl,
                    'entry_time': ts, 'peak_price': close, 'realized_pnl': 0.0
                })
        
        # Close open trades at the end
        for op in self.open_trades:
            close = float(dfs[op['symbol']]['close'].iloc[-1])
            pnl = (close - op['entry_price']) * op['qty'] if op['side'] == 'LONG' else (op['entry_price'] - close) * op['qty']
            self.balance += op['margin'] + pnl
            op['realized_pnl'] += pnl
            self.closed_trades.append(op)
            
        wins = sum(1 for t in self.closed_trades if t['realized_pnl'] > 0)
        total = len(self.closed_trades)
        wr = wins/total*100 if total else 0
        
        print("\n==================================================")
        print("   V2 RISK & PARTIAL TP SIMULATION (180 DAYS)   ")
        print("==================================================")
        print(f"Parametreler:")
        print("- Baslangic: $1000")
        print("- Sizing: Dinamik %2 Risk, %15 Max Margin")
        print("- Kural: Ayni yonde max 2 islem")
        print("- TP: %2.5 kârda pozisyonun %40'ini kapat")
        print("--------------------------------------------------")
        print(f"Sonuclar:")
        print(f"- Toplam Islem Sayisi: {total}")
        print(f"- Basarili: {wins} / Basarisiz: {total-wins}")
        print(f"- Kazanma Orani: %{wr:.2f}")
        print(f"- Maksimum Dusus (Drawdown): %{self.max_dd*100:.2f}")
        print("--------------------------------------------------")
        print(f"- Bitis Bakiyesi: ${self.balance:.2f}")
        print(f"- Net Kar/Zarar: ${self.balance-1000:.2f} (%{(self.balance-1000)/10:.2f})")
        print("==================================================")

if __name__ == '__main__':
    sim = PortfolioSimV2()
    sim.run()
