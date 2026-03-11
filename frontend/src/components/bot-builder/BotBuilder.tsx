"use client"

import React, { useState } from 'react';
import { getApiUrl } from '@/utils/api';
import { SortableBlock } from './SortableBlock';
import { DraggableItem } from './DraggableItem';

const AVAILABLE_BLOCKS = [
  { id: 'rsi', type: 'indicator', title: 'RSI Indicator', desc: 'Relative Strength Index' },
  { id: 'macd', type: 'indicator', title: 'MACD', desc: 'Moving Average Convergence' },
  { id: 'if_condition', type: 'logic', title: 'IF Condition', desc: 'Logic Gate (>, <, =)' },
  { id: 'buy_market', type: 'action', title: 'Buy Market', desc: 'Execute Market Buy' },
  { id: 'sell_market', type: 'action', title: 'Sell Market', desc: 'Execute Market Sell' },
];

export default function BotBuilder() {
  const [isMounted, setIsMounted] = useState(false);
  const [activeBlocks, setActiveBlocks] = useState<{ id: string, type: string, title: string }[]>([]);

  React.useEffect(() => {
    setIsMounted(true);
  }, []);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault(); // Drop işlemine izin vermek için gerekli
    e.dataTransfer.dropEffect = 'copy';
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const dataString = e.dataTransfer.getData('application/json');

    if (!dataString) return;

    try {
      const originalBlock = JSON.parse(dataString);
      const newBlock = {
        id: `${originalBlock.id}_${Date.now()}`,
        type: originalBlock.type,
        title: originalBlock.title
      };

      setActiveBlocks(prev => [...prev, newBlock]);
    } catch (err) {
      console.error("Drop veri parse hatası:", err);
    }
  };

  const handleSave = async () => {
    try {
      // API Call to Python Backtest Engine
      console.log("Sending Strategy to Backtest Engine:", activeBlocks);
      const reqBody = {
        symbol: "BTC/USDT",
        timeframe: "1h",
        limit: 100, // Deneme amaçlı 100 mum
        initial_balance: 1000.0,
        strategy: activeBlocks.map(block => ({ id: block.id, type: block.type, title: block.title }))
      };

      const response = await fetch(getApiUrl("/backtest/run"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reqBody)
      });

      if (!response.ok) {
        throw new Error(`Backtest HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Backtest Metrics:", data);

      const metrics = data.metrics;
      alert(`Backtest Results:\nWin Rate: ${metrics.win_rate}%\nTotal PnL: $${metrics.total_pnl}\nTotal Trades: ${metrics.total_trades}`);

    } catch (error) {
      console.error("Backtest execution failed:", error);
      alert("Backtest çalıştırılırken bir hata oluştu. Lütfen konsolu kontrol edin.");
    }
  };

  const handleOptimize = async () => {
    if (activeBlocks.length === 0) {
      alert("Lütfen önce test edilecek blokları Canvas'a sürükleyin!");
      return;
    }

    try {
      alert("Yapay Zeka (AI) optimizasyon motoru başlatılıyor. Bu işlem biraz zaman alabilir...");
      console.log("Sending Strategy to AI Optimizer Engine:", activeBlocks);

      const reqBody = {
        symbol: "BTC/USDT",
        timeframe: "1h",
        limit: 100,
        initial_balance: 1000.0,
        n_trials: 50, // 50 farklı kombinasyon Optuna tarafından test edilecek
        strategy: activeBlocks.map(block => ({ id: block.id, type: block.type, title: block.title }))
      };

      const response = await fetch(getApiUrl("/backtest/optimize"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reqBody)
      });

      if (!response.ok) {
        throw new Error(`AI Optimizer HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Optimizer Results:", data);

      alert(`AI Optimization Results:\nBest Potential PnL: $${data.best_pnl}\nTotal Trial Combinations: ${data.total_trials}\n\nSuggested Parameters:\n${JSON.stringify(data.best_parameters, null, 2)}`);

    } catch (error) {
      console.error("AI Optimizer execution failed:", error);
      alert("Yapay Zeka optimizasyonunda bir hata oluştu. Lütfen konsolu kontrol edin.");
    }
  };

  if (!isMounted) return null;

  return (
    <div className="flex flex-col h-[calc(100vh-80px)] w-full bg-slate-900 text-white font-sans">

      {/* Topbar - Available Blocks */}
      <div className="w-full bg-slate-800 p-4 border-b border-slate-700 flex-shrink-0 z-10 shadow-md">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-emerald-400">Available Nodes</h2>
          <div className="flex space-x-3">
            <button
              onClick={handleOptimize}
              className="bg-purple-600 hover:bg-purple-500 text-white px-6 py-2 rounded-lg font-medium transition-colors shadow-lg shadow-purple-900/20 flex items-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Optimize Strategy (AI)
            </button>
            <button
              onClick={handleSave}
              className="bg-emerald-600 hover:bg-emerald-500 text-white px-6 py-2 rounded-lg font-medium transition-colors shadow-lg shadow-emerald-900/20"
            >
              Save Strategy
            </button>
          </div>
        </div>

        <div className="flex space-x-8 overflow-x-auto pb-2">
          <div className="min-w-max">
            <h3 className="text-sm font-semibold text-slate-400 mb-2 uppercase tracking-wider">Indicators & Data</h3>
            <div className="flex space-x-2">
              {AVAILABLE_BLOCKS.filter(b => b.type === 'indicator').map(block => (
                <DraggableItem key={block.id} item={block} />
              ))}
            </div>
          </div>
          <div className="min-w-max">
            <h3 className="text-sm font-semibold text-slate-400 mb-2 uppercase tracking-wider">Logic Gates</h3>
            <div className="flex space-x-2">
              {AVAILABLE_BLOCKS.filter(b => b.type === 'logic').map(block => (
                <DraggableItem key={block.id} item={block} />
              ))}
            </div>
          </div>
          <div className="min-w-max">
            <h3 className="text-sm font-semibold text-slate-400 mb-2 uppercase tracking-wider">Actions</h3>
            <div className="flex space-x-2">
              {AVAILABLE_BLOCKS.filter(b => b.type === 'action').map(block => (
                <DraggableItem key={block.id} item={block} />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Canvas - Drop Area */}
      <div
        className="flex-1 p-8 bg-slate-900 overflow-y-auto"
        id="canvas-droppable"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-white">Visual Bot Designer</h1>
          <p className="text-slate-400 mt-2">Drag and drop nodes from the top bar to build your strategy algorithm.</p>
        </div>

        <div className="min-h-[500px] w-full max-w-4xl mx-auto bg-slate-800/50 border border-slate-700/50 rounded-2xl p-6 shadow-2xl backdrop-blur-sm relative pointer-events-auto">
          {activeBlocks.length === 0 ? (
            <div className="flex flex-col items-center justify-center text-slate-500 h-full min-h-[400px]">
              <div className="w-20 h-20 mb-4 rounded-full border-2 border-dashed border-slate-600 flex items-center justify-center">
                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                </svg>
              </div>
              <p className="text-lg">Drag strategy nodes here</p>
            </div>
          ) : (
            <div className="space-y-3 relative">
              {activeBlocks.map((block, index) => (
                <div key={block.id} className="relative">
                  <SortableBlock id={block.id} title={block.title} type={block.type} />
                  {index < activeBlocks.length - 1 && (
                    <div className="w-px h-6 bg-slate-600 absolute -bottom-6 left-1/2 transform -translate-x-1/2 z-0" />
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

    </div>
  );
}
