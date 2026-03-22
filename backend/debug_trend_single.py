import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

from backtest.trend_backtest import TrendBacktest
from run_trend_simulation import UserTrendBacktest

def test_single():
    bt = UserTrendBacktest(initial_balance=1000.0, leverage=10)
    res = bt.run_backtest('BTC/USDT', days=90)
    print(f"Success: {res['success']}")
    if not res['success']:
        print(f"Error: {res.get('error')}")
    else:
        print(f"Total trades: {res['total_trades']}")
        print(f"Win Rate: {res['win_rate']}")
        print(f"Total PnL: {res['total_pnl']}")

if __name__ == '__main__':
    test_single()
