import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.getcwd())

from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators, generate_signals

def run_upgraded_backtest(symbol="BTC/USDT", timeframe="1h", limit=3000):
    print(f"📡 Fetching {limit} candles for {symbol} ({timeframe})...")
    fetcher = DataFetcher('binance')
    df = fetcher.fetch_ohlcv(symbol, timeframe, limit=limit)
    # fetcher.close() # CCXT sync skip close

    if df.empty:
        print("❌ Data fetch failed.")
        return

    print("📊 Adding indicators and generating signals...")
    df = add_all_indicators(df)
    # Filter only signals
    signals, scores = generate_signals(df, strategy="confluence")
    df['signal'] = signals
    
    # Backtest State
    balance = 1000.0
    initial_balance = 1000.0
    trade_size_pct = 20.0 # %20 risk per trade
    atr_tp_mult = 2.0
    atr_sl_mult = 1.5
    
    open_trade = None
    trades = []
    equity_curve = [balance]
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        curr_price = row['close']
        curr_time = df.index[i]
        
        # 1. Check Exit if trade is open
        if open_trade:
            # Calculate current profit %
            if open_trade['side'] == "BUY":
                profit_pct = (curr_price / open_trade['entry_price'] - 1) * 100
                high_price = row['high']
                low_price = row['low']
                
                # A. Breakeven (at 1% profit)
                if not open_trade['is_breakeven'] and profit_pct >= 1.0:
                    open_trade['sl'] = open_trade['entry_price']
                    open_trade['is_breakeven'] = True

                # B. Trailing Stop (at 1.5% profit)
                if profit_pct >= 1.5:
                    new_sl = max(open_trade['sl'], curr_price - row['atr'] * 1.5)
                    if new_sl > open_trade['sl']:
                        open_trade['sl'] = new_sl

                # C. RSI Exit (Long: RSI > 70 & Profit > 0)
                if row['rsi'] > 70 and profit_pct > 0.2:
                    open_trade['exit_price'] = curr_price
                    open_trade['reason'] = "RSI_EXIT"
                
                # D. Standard TP/SL
                elif high_price >= open_trade['tp']:
                    open_trade['exit_price'] = open_trade['tp']
                    open_trade['reason'] = "TP"
                elif low_price <= open_trade['sl']:
                    open_trade['exit_price'] = open_trade['sl']
                    open_trade['reason'] = "SL"
                
            else: # SELL / SHORT
                profit_pct = (open_trade['entry_price'] / curr_price - 1) * 100
                high_price = row['high']
                low_price = row['low']
                
                if not open_trade['is_breakeven'] and profit_pct >= 1.0:
                    open_trade['sl'] = open_trade['entry_price']
                    open_trade['is_breakeven'] = True

                if profit_pct >= 1.5:
                    new_sl = min(open_trade['sl'], curr_price + row['atr'] * 1.5)
                    if new_sl < open_trade['sl']:
                        open_trade['sl'] = new_sl

                if row['rsi'] < 30 and profit_pct > 0.2:
                    open_trade['exit_price'] = curr_price
                    open_trade['reason'] = "RSI_EXIT"
                    
                elif low_price <= open_trade['tp']:
                    open_trade['exit_price'] = open_trade['tp']
                    open_trade['reason'] = "TP"
                elif high_price >= open_trade['sl']:
                    open_trade['exit_price'] = open_trade['sl']
                    open_trade['reason'] = "SL"

            # 2. Close trade if exit triggered
            if open_trade.get('exit_price'):
                ot = open_trade
                if ot['side'] == "BUY":
                    # For long: exit/entry - 1
                    pnl_pct = (ot['exit_price'] / ot['entry_price'] - 1) * 100
                else: 
                    # For short: 1 - exit/entry is profit, or entry/exit - 1
                    pnl_pct = (ot['entry_price'] / ot['exit_price'] - 1) * 100
                
                pnl_usd = (ot['amount'] * pnl_pct / 100)
                balance += ot['amount'] + pnl_usd
                
                ot['pnl_pct'] = pnl_pct
                ot['pnl_usd'] = pnl_usd
                ot['exit_time'] = curr_time
                trades.append(ot)
                open_trade = None
        
        # 3. Open new trade if Signal and nothing open
        if not open_trade:
            sig = signals.iloc[i]
            if sig in ["BUY", "SELL"]:
                amount = balance * (trade_size_pct / 100)
                if balance < 10: break
                
                balance -= amount
                atr_val = row['atr']
                
                side = "BUY" if sig == "BUY" else "SELL"
                if side == "BUY":
                    tp = curr_price + atr_val * atr_tp_mult
                    sl = curr_price - atr_val * atr_sl_mult
                else:
                    tp = curr_price - atr_val * atr_tp_mult
                    sl = curr_price + atr_val * atr_sl_mult
                    
                open_trade = {
                    'entry_time': curr_time,
                    'entry_price': curr_price,
                    'side': side,
                    'amount': amount,
                    'tp': tp,
                    'sl': sl,
                    'is_breakeven': False,
                    'exit_price': None
                }
        
        equity_curve.append(balance + (open_trade['amount'] if open_trade else 0))

    if not trades:
        print("No trades executed.")
        return

    df_trades = pd.DataFrame(trades)
    win_rate = (len(df_trades[df_trades['pnl_pct'] > 0]) / len(df_trades)) * 100
    total_pnl_usd = sum(t['pnl_usd'] for t in trades)
    
    report = []
    report.append("\n" + "="*40)
    report.append(f"🏆 BACKTEST RESULTS ({symbol} - {timeframe} - {limit} candles)")
    report.append(f"Total Trades: {len(df_trades)}")
    report.append(f"Win Rate: {win_rate:.2f}%")
    report.append(f"Total PnL USD: ${total_pnl_usd:.2f}")
    report.append(f"Initial Balance: ${initial_balance}")
    report.append(f"Final Balance: ${initial_balance + total_pnl_usd:.2f}")
    report.append(f"Avg PnL per trade: {df_trades['pnl_pct'].mean():.2f}%")
    report.append("="*40)
    
    reasons = df_trades['reason'].value_counts()
    report.append("\nExit Reasons:")
    report.append(str(reasons))
    report.append("="*40)

    report_str = "\n".join(report)
    print(report_str)
    with open("backtest_report.txt", "w", encoding="utf-8") as f:
        f.write(report_str)

if __name__ == "__main__":
    run_upgraded_backtest(limit=3000)
