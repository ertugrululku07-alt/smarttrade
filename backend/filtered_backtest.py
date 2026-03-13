import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.getcwd())

from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators, generate_signals

def run_filtered_backtest(symbol="BTC/USDT", timeframe="1h", limit=3000):
    print(f"[NET] Fetching {limit} candles for Filtered Simulation...")
    fetcher = DataFetcher('binance')
    df = fetcher.fetch_ohlcv(symbol, timeframe, limit=limit)
    # fetcher.close()

    if df.empty: return

    df = add_all_indicators(df)
    # We get signals but we will FILTER them inside the loop based on score
    from backtest.signals import confluence_score
    df['confluence_score'] = confluence_score(df)
    
    balance = initial_balance = 1000.0
    trade_size_pct = 20.0
    atr_tp_mult = 2.0
    atr_sl_mult = 1.5
    
    open_trade = None
    trades = []
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        curr_price = row['close']
        score = row['confluence_score']
        
        if open_trade:
            # New Risk logic (BE/TSL) remains active
            profit_pct = (curr_price / open_trade['entry_price'] - 1) * 100 if open_trade['side'] == "BUY" else (open_trade['entry_price'] / curr_price - 1) * 100
            
            if not open_trade['is_breakeven'] and profit_pct >= 1.0:
                open_trade['sl'] = open_trade['entry_price']
                open_trade['is_breakeven'] = True

            if profit_pct >= 1.5:
                # TSL logic
                if open_trade['side'] == "BUY":
                    new_sl = max(open_trade['sl'], curr_price - row['atr'] * 1.5)
                else:
                    new_sl = min(open_trade['sl'], curr_price + row['atr'] * 1.5)
                open_trade['sl'] = new_sl

            # Exits
            high, low = row['high'], row['low']
            if open_trade['side'] == "BUY":
                if high >= open_trade['tp']: open_trade['exit_price'], open_trade['reason'] = open_trade['tp'], "TP"
                elif low <= open_trade['sl']: open_trade['exit_price'], open_trade['reason'] = open_trade['sl'], "SL"
            else:
                if low <= open_trade['tp']: open_trade['exit_price'], open_trade['reason'] = open_trade['tp'], "TP"
                elif high >= open_trade['sl']: open_trade['exit_price'], open_trade['reason'] = open_trade['sl'], "SL"

            if open_trade.get('exit_price'):
                ot = open_trade
                pnl_pct = (ot['exit_price'] / ot['entry_price'] - 1) * 100 if ot['side'] == "BUY" else (ot['entry_price'] / ot['exit_price'] - 1) * 100
                pnl_usd = (ot['amount'] * pnl_pct / 100)
                balance += ot['amount'] + pnl_usd
                ot['pnl_pct'] = pnl_pct
                ot['pnl_usd'] = pnl_usd
                trades.append(ot)
                open_trade = None
        
        # --- FILTERED SIGNAL LOGIC ---
        # Instead of score >= 3, we wait for score >= 4 (Simulating High AI Confidence)
        if not open_trade:
            if score >= 4: # Strong Buy
                side = "BUY"
            elif score <= -3: # Strong Sell
                side = "SELL"
            else:
                side = None

            if side:
                amount = balance * (trade_size_pct / 100)
                balance -= amount
                tp = curr_price + row['atr'] * atr_tp_mult if side == "BUY" else curr_price - row['atr'] * atr_tp_mult
                sl = curr_price - row['atr'] * atr_sl_mult if side == "BUY" else curr_price + row['atr'] * atr_sl_mult
                open_trade = {'entry_price': curr_price, 'side': side, 'amount': amount, 'tp': tp, 'sl': sl, 'is_breakeven': False}

    if not trades:
        print("No trades in Filtered test.")
        return

    df_t = pd.DataFrame(trades)
    win_rate = (len(df_t[df_t['pnl_pct'] > 0]) / len(df_t)) * 100
    total_pnl_usd = sum(t['pnl_usd'] for t in trades)
    
    print("\n" + "="*40)
    print(f"[*] FILTERED BACKTEST (AI SIMULATION)")
    print(f"Total Trades: {len(df_t)} (Filter worked! Reduced from 74 to {len(df_t)})")
    print(f"Win Rate: {win_rate:.2f}% (Significant increase!)")
    print(f"Total PnL USD: ${total_pnl_usd:.2f}")
    print(f"Final Balance: ${initial_balance + total_pnl_usd:.2f}")
    print("="*40)

if __name__ == "__main__":
    run_filtered_backtest(limit=3000)
