import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.getcwd())

from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators, generate_signals

def run_baseline_backtest(symbol="BTC/USDT", timeframe="1h", limit=3000):
    print(f"📡 Fetching {limit} candles for Baseline...")
    fetcher = DataFetcher('binance')
    df = fetcher.fetch_ohlcv(symbol, timeframe, limit=limit)
    # fetcher.close()

    if df.empty: return

    df = add_all_indicators(df)
    signals, scores = generate_signals(df, strategy="confluence")
    
    balance = initial_balance = 1000.0
    trade_size_pct = 20.0
    atr_tp_mult = 2.0
    atr_sl_mult = 1.5
    
    open_trade = None
    trades = []
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        curr_price = row['close']
        
        if open_trade:
            high_price, low_price = row['high'], row['low']
            if open_trade['side'] == "BUY":
                if high_price >= open_trade['tp']:
                    open_trade['exit_price'] = open_trade['tp']
                    open_trade['reason'] = "TP"
                elif low_price <= open_trade['sl']:
                    open_trade['exit_price'] = open_trade['sl']
                    open_trade['reason'] = "SL"
            else:
                if low_price <= open_trade['tp']:
                    open_trade['exit_price'] = open_trade['tp']
                    open_trade['reason'] = "TP"
                elif high_price >= open_trade['sl']:
                    open_trade['exit_price'] = open_trade['sl']
                    open_trade['reason'] = "SL"

            if open_trade.get('exit_price'):
                ot = open_trade
                pnl_pct = (ot['exit_price'] / ot['entry_price'] - 1) * 100 if ot['side'] == "BUY" else (ot['entry_price'] / ot['exit_price'] - 1) * 100
                pnl_usd = (ot['amount'] * pnl_pct / 100)
                balance += ot['amount'] + pnl_usd
                ot['pnl_usd'] = pnl_usd
                trades.append(ot)
                open_trade = None
        
        if not open_trade:
            sig = signals.iloc[i]
            if sig in ["BUY", "SELL"]:
                amount = balance * (trade_size_pct / 100)
                balance -= amount
                side = "BUY" if sig == "BUY" else "SELL"
                tp = curr_price + row['atr'] * atr_tp_mult if side == "BUY" else curr_price - row['atr'] * atr_tp_mult
                sl = curr_price - row['atr'] * atr_sl_mult if side == "BUY" else curr_price + row['atr'] * atr_sl_mult
                open_trade = {'entry_price': curr_price, 'side': side, 'amount': amount, 'tp': tp, 'sl': sl}

    total_pnl_usd = sum(t['pnl_usd'] for t in trades)
    report = f"OLD STRATEGY (No BE/TSL): Total PnL USD: ${total_pnl_usd:.2f} | Final Balance: ${initial_balance + total_pnl_usd:.2f}"
    print(report)
    with open("baseline_report.txt", "w") as f: f.write(report)

if __name__ == "__main__":
    run_baseline_backtest(limit=3000)
