"""v3.0 Simple Trend Momentum — 20 coin test"""
import sys, os, warnings, logging, io
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)
sys.path.insert(0, '.')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators
from ai.adaptive_backtest import AdaptiveBacktest
from collections import Counter

fetcher = DataFetcher('binance')
coins = [
    'BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT','XRP/USDT',
    'LINK/USDT','AVAX/USDT','DOGE/USDT','DOT/USDT','LTC/USDT',
    'ATOM/USDT','UNI/USDT','NEAR/USDT','INJ/USDT','TIA/USDT',
    'RUNE/USDT','AAVE/USDT',
]

total_pnl = 0
profitable = 0
results = []

for sym in coins:
    try:
        df = fetcher.fetch_ohlcv(sym, '1h', limit=4320)
        if df is None or df.empty or len(df) < 100:
            print(f"  SKIP {sym}: insufficient data")
            continue
        df = add_all_indicators(df)
    except Exception as e:
        print(f"  SKIP {sym}: {e}")
        continue

    engine = AdaptiveBacktest(
        timeframe='1h',
        initial_capital=1000,
        use_simple_strategy=True,
    )
    old = sys.stdout; sys.stdout = io.StringIO()
    result = engine.run(df, symbol=sym)
    sys.stdout = old

    oc = Counter(t.outcome for t in result.trades)
    pnl_d = (result.equity_curve[-1] - 1000) if result.equity_curve else 0
    total_pnl += pnl_d
    if pnl_d > 0: profitable += 1

    tp_c = oc.get('TP', 0)
    sl_c = oc.get('SL', 0)
    to_c = oc.get('TIMEOUT', 0)
    tag = "+" if pnl_d > 0 else "-"

    # Direction breakdown
    longs = sum(1 for t in result.trades if t.direction == 'LONG')
    shorts = sum(1 for t in result.trades if t.direction == 'SHORT')

    print(f"  {tag} {sym:12s} {result.total_trades:3d}t WR:{result.win_rate:5.1f}% "
          f"PnL:{result.total_pnl_pct:+7.2f}% (${pnl_d:+7.2f}) "
          f"PF:{result.profit_factor:4.2f} DD:{result.max_drawdown_pct:5.1f}% "
          f"TP:{tp_c} SL:{sl_c} TO:{to_c} L:{longs} S:{shorts}")
    results.append((sym, pnl_d, result))

print(f"\n{'='*70}")
print(f"  TOTAL: ${total_pnl:+.2f} | Profitable: {profitable}/{len(results)}")
print(f"  Avg PnL per coin: ${total_pnl/max(len(results),1):+.2f}")
