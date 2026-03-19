'use client';

import { useState } from 'react';
import { getApiUrl } from '@/utils/api';

interface Trade {
  id: number;
  direction: string;
  entry_price: number;
  exit_price: number;
  entry_time: string;
  exit_time: string;
  pnl: number;
  pnl_pct: number;
  max_profit_pct: number;
  exit_reason: string;
}

interface BacktestResult {
  success: boolean;
  symbol?: string;
  timeframe?: string;
  days?: number;
  leverage?: number;
  initial_balance?: number;
  final_balance?: number;
  total_pnl?: number;
  total_pnl_pct?: number;
  total_trades?: number;
  long_trades?: number;
  short_trades?: number;
  wins?: number;
  losses?: number;
  win_rate?: number;
  avg_win?: number;
  avg_loss?: number;
  profit_factor?: number;
  max_profit_trade?: number;
  max_loss_trade?: number;
  trades?: Trade[];
  error?: string;
}

export default function ICTFullBacktestPage() {
  const [symbol, setSymbol] = useState('BANANAS31/USDT');
  const [days, setDays] = useState(30);
  const [initialBalance, setInitialBalance] = useState(1000);
  const [leverage, setLeverage] = useState(10);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);

  const runBacktest = async () => {
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(getApiUrl('/backtest/ict-full-backtest'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          symbol, 
          days, 
          initial_balance: initialBalance,
          leverage 
        }),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            ICT/SMC v2.5 Full Backtest
          </h1>
          <p className="text-gray-300">
            Complete strategy simulation with LONG/SHORT auto-detection
          </p>
        </div>

        {/* Input Form */}
        <div className="bg-gray-800 rounded-lg p-6 mb-6 border border-gray-700">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Symbol
              </label>
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                placeholder="BTC/USDT"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Days
              </label>
              <input
                type="number"
                value={days}
                onChange={(e) => setDays(parseInt(e.target.value))}
                className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                min="7"
                max="90"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Initial Balance ($)
              </label>
              <input
                type="number"
                value={initialBalance}
                onChange={(e) => setInitialBalance(parseInt(e.target.value))}
                className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                min="100"
                max="100000"
                step="100"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Leverage (x)
              </label>
              <input
                type="number"
                value={leverage}
                onChange={(e) => setLeverage(parseInt(e.target.value))}
                className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
                min="1"
                max="20"
              />
            </div>
          </div>

          <button
            onClick={runBacktest}
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white font-bold py-3 px-6 rounded-lg transition-colors"
          >
            {loading ? 'Running Full Backtest...' : 'Run Full Backtest'}
          </button>
        </div>

        {/* Results */}
        {result && (
          <div className="space-y-6">
            {result.success ? (
              <>
                {/* Performance Summary */}
                <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                  <h2 className="text-2xl font-bold text-white mb-4">Performance Summary</h2>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-gray-700 p-4 rounded-lg">
                      <div className="text-gray-400 text-sm">Symbol</div>
                      <div className="text-white text-xl font-bold">{result.symbol}</div>
                    </div>
                    <div className="bg-gray-700 p-4 rounded-lg">
                      <div className="text-gray-400 text-sm">Timeframe</div>
                      <div className="text-white text-xl font-bold">{result.timeframe}</div>
                    </div>
                    <div className="bg-gray-700 p-4 rounded-lg">
                      <div className="text-gray-400 text-sm">Days</div>
                      <div className="text-white text-xl font-bold">{result.days}</div>
                    </div>
                    <div className="bg-gray-700 p-4 rounded-lg">
                      <div className="text-gray-400 text-sm">Leverage</div>
                      <div className="text-white text-xl font-bold">{result.leverage}x</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-4">
                    <div className={`p-4 rounded-lg border-2 ${(result.total_pnl ?? 0) >= 0 ? 'bg-green-900/30 border-green-700' : 'bg-red-900/30 border-red-700'}`}>
                      <div className="text-gray-400 text-sm">Total PnL</div>
                      <div className={`text-3xl font-bold ${(result.total_pnl ?? 0) >= 0 ? 'text-green-300' : 'text-red-300'}`}>
                        ${result.total_pnl?.toFixed(2)} ({result.total_pnl_pct?.toFixed(2)}%)
                      </div>
                    </div>
                    <div className="bg-blue-900/30 border border-blue-700 p-4 rounded-lg">
                      <div className="text-blue-400 text-sm">Initial Balance</div>
                      <div className="text-blue-300 text-2xl font-bold">${result.initial_balance?.toFixed(2)}</div>
                    </div>
                    <div className="bg-purple-900/30 border border-purple-700 p-4 rounded-lg">
                      <div className="text-purple-400 text-sm">Final Balance</div>
                      <div className="text-purple-300 text-2xl font-bold">${result.final_balance?.toFixed(2)}</div>
                    </div>
                  </div>
                </div>

                {/* Trade Statistics */}
                <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                  <h2 className="text-2xl font-bold text-white mb-4">Trade Statistics</h2>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-gray-700 p-4 rounded-lg">
                      <div className="text-gray-400 text-sm">Total Trades</div>
                      <div className="text-white text-3xl font-bold">{result.total_trades}</div>
                    </div>
                    <div className="bg-blue-900/30 border border-blue-700 p-4 rounded-lg">
                      <div className="text-blue-400 text-sm">LONG Trades</div>
                      <div className="text-blue-300 text-3xl font-bold">{result.long_trades}</div>
                    </div>
                    <div className="bg-orange-900/30 border border-orange-700 p-4 rounded-lg">
                      <div className="text-orange-400 text-sm">SHORT Trades</div>
                      <div className="text-orange-300 text-3xl font-bold">{result.short_trades}</div>
                    </div>
                    <div className={`p-4 rounded-lg border-2 ${(result.win_rate ?? 0) >= 50 ? 'bg-green-900/30 border-green-700' : 'bg-red-900/30 border-red-700'}`}>
                      <div className="text-gray-400 text-sm">Win Rate</div>
                      <div className={`text-3xl font-bold ${(result.win_rate ?? 0) >= 50 ? 'text-green-300' : 'text-red-300'}`}>
                        {result.win_rate?.toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                    <div className="bg-green-900/30 border border-green-700 p-4 rounded-lg">
                      <div className="text-green-400 text-sm">Wins</div>
                      <div className="text-green-300 text-2xl font-bold">{result.wins}</div>
                    </div>
                    <div className="bg-red-900/30 border border-red-700 p-4 rounded-lg">
                      <div className="text-red-400 text-sm">Losses</div>
                      <div className="text-red-300 text-2xl font-bold">{result.losses}</div>
                    </div>
                    <div className="bg-gray-700 p-4 rounded-lg">
                      <div className="text-gray-400 text-sm">Avg Win</div>
                      <div className="text-green-300 text-xl font-bold">${result.avg_win?.toFixed(2)}</div>
                    </div>
                    <div className="bg-gray-700 p-4 rounded-lg">
                      <div className="text-gray-400 text-sm">Avg Loss</div>
                      <div className="text-red-300 text-xl font-bold">${result.avg_loss?.toFixed(2)}</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4 mt-4">
                    <div className="bg-gray-700 p-4 rounded-lg">
                      <div className="text-gray-400 text-sm">Profit Factor</div>
                      <div className="text-white text-2xl font-bold">{result.profit_factor?.toFixed(2)}</div>
                    </div>
                    <div className="bg-green-900/30 border border-green-700 p-4 rounded-lg">
                      <div className="text-green-400 text-sm">Best Trade</div>
                      <div className="text-green-300 text-xl font-bold">${result.max_profit_trade?.toFixed(2)}</div>
                    </div>
                    <div className="bg-red-900/30 border border-red-700 p-4 rounded-lg">
                      <div className="text-red-400 text-sm">Worst Trade</div>
                      <div className="text-red-300 text-xl font-bold">${result.max_loss_trade?.toFixed(2)}</div>
                    </div>
                  </div>
                </div>

                {/* Trade History */}
                {result.trades && result.trades.length > 0 && (
                  <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                    <h2 className="text-2xl font-bold text-white mb-4">Recent Trades (Last 20)</h2>
                    <div className="space-y-2">
                      {result.trades.map((trade) => (
                        <div key={trade.id} className={`p-4 rounded-lg border ${trade.pnl >= 0 ? 'bg-green-900/20 border-green-700' : 'bg-red-900/20 border-red-700'}`}>
                          <div className="flex justify-between items-center mb-2">
                            <div className="flex items-center gap-4">
                              <span className={`px-3 py-1 rounded font-bold ${trade.direction === 'LONG' ? 'bg-blue-600' : 'bg-orange-600'}`}>
                                {trade.direction}
                              </span>
                              <span className="text-gray-400 font-mono text-sm">{trade.entry_time}</span>
                            </div>
                            <div className="flex items-center gap-4">
                              <div className={`text-xl font-bold ${trade.pnl >= 0 ? 'text-green-300' : 'text-red-300'}`}>
                                ${trade.pnl.toFixed(2)} ({trade.pnl_pct.toFixed(2)}%)
                              </div>
                              <span className="text-gray-400 text-sm">{trade.exit_reason}</span>
                            </div>
                          </div>
                          <div className="grid grid-cols-3 gap-4 text-sm">
                            <div>
                              <span className="text-gray-400">Entry: </span>
                              <span className="text-white font-mono">${trade.entry_price.toFixed(6)}</span>
                            </div>
                            <div>
                              <span className="text-gray-400">Exit: </span>
                              <span className="text-white font-mono">${trade.exit_price.toFixed(6)}</span>
                            </div>
                            <div>
                              <span className="text-gray-400">Max Profit: </span>
                              <span className="text-white">{trade.max_profit_pct.toFixed(2)}%</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="bg-red-900/30 border border-red-700 rounded-lg p-6">
                <h2 className="text-2xl font-bold text-red-400 mb-2">Error</h2>
                <p className="text-red-300">{result.error}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
