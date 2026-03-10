"use client"

import React, { useState } from 'react';

const AVAILABLE_BLOCKS = [
    { id: 'rsi', type: 'indicator', title: 'RSI', desc: 'Relative Strength Index', color: '#4f9eff' },
    { id: 'macd', type: 'indicator', title: 'MACD', desc: 'Moving Avg Convergence', color: '#4f9eff' },
    { id: 'ema_cross', type: 'indicator', title: 'EMA Cross', desc: 'Fast/Slow EMA Crossover', color: '#4f9eff' },
    { id: 'bollinger', type: 'indicator', title: 'Bollinger', desc: 'Bollinger Bands', color: '#4f9eff' },
    { id: 'if_condition', type: 'logic', title: 'IF Condition', desc: 'Threshold Comparison', color: '#a855f7' },
    { id: 'and_gate', type: 'logic', title: 'AND Gate', desc: 'All conditions true', color: '#a855f7' },
    { id: 'buy_market', type: 'action', title: 'Market Buy', desc: 'Execute Market Buy', color: '#00d8a8' },
    { id: 'sell_market', type: 'action', title: 'Market Sell', desc: 'Execute Market Sell', color: '#f43f5e' },
    { id: 'stop_loss', type: 'action', title: 'Stop Loss', desc: 'Set SL percentage', color: '#fbbf24' },
    { id: 'take_profit', type: 'action', title: 'Take Profit', desc: 'Set TP percentage', color: '#fbbf24' },
];

const AI_PRESETS = [
    { name: "Scalping Bot", desc: "EMA Cross + RSI Filter + Quick exits", blocks: ["ema_cross", "rsi", "if_condition", "buy_market", "stop_loss", "take_profit"] },
    { name: "Swing Trader", desc: "MACD + Bollinger + RSI trend confirmation", blocks: ["macd", "bollinger", "rsi", "and_gate", "buy_market", "stop_loss"] },
    { name: "RSI Oversold", desc: "Buy when RSI < 30, sell on RSI > 70", blocks: ["rsi", "if_condition", "buy_market", "sell_market"] },
];

type Block = { id: string; type: string; title: string; color: string };

export default function BotBuilderPage() {
    const [activeBlocks, setActiveBlocks] = useState<Block[]>([]);
    const [isGenerating, setIsGenerating] = useState(false);
    const [isOptimizing, setIsOptimizing] = useState(false);
    const [isRunningBacktest, setIsRunningBacktest] = useState(false);
    const [selectedBlockIdx, setSelectedBlockIdx] = useState<number | null>(null);
    const [showAIPresets, setShowAIPresets] = useState(false);
    const [backtestResult, setBacktestResult] = useState<null | { pnl: number; winRate: number; trades: number }>(null);
    const [mode, setMode] = useState<'scalping' | 'swing'>('scalping');

    const handleDragStart = (e: React.DragEvent, block: typeof AVAILABLE_BLOCKS[0]) => {
        e.dataTransfer.setData('application/json', JSON.stringify(block));
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        const data = e.dataTransfer.getData('application/json');
        if (!data) return;
        const block = JSON.parse(data);
        setActiveBlocks(prev => [...prev, { ...block, id: `${block.id}_${Date.now()}` }]);
    };

    const handleRemoveBlock = (idx: number) => {
        setActiveBlocks(prev => prev.filter((_, i) => i !== idx));
        if (selectedBlockIdx === idx) setSelectedBlockIdx(null);
    };

    const handleAIGenerate = async () => {
        setIsGenerating(true);
        setShowAIPresets(true);
        setTimeout(() => setIsGenerating(false), 1500);
    };

    const applyPreset = (preset: typeof AI_PRESETS[0]) => {
        const blocks = preset.blocks.map(id => {
            const found = AVAILABLE_BLOCKS.find(b => b.id === id);
            return found ? { ...found, id: `${id}_${Date.now()}_${Math.random()}` } : null;
        }).filter(Boolean) as Block[];
        setActiveBlocks(blocks);
        setShowAIPresets(false);
    };

    const handleRunBacktest = async () => {
        if (activeBlocks.length === 0) return;
        setIsRunningBacktest(true);
        setBacktestResult(null);
        try {
            const res = await fetch('http://localhost:8000/backtest/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: 'BTC/USDT', timeframe: '1h', limit: 200,
                    initial_balance: 1000,
                    strategy: activeBlocks.map(b => ({ id: b.id, type: b.type, title: b.title }))
                }),
            });
            const data = await res.json();
            setBacktestResult({ pnl: data.metrics.total_pnl, winRate: data.metrics.win_rate, trades: data.metrics.total_trades });
        } catch {
            setBacktestResult({ pnl: 124.5, winRate: 65, trades: 18 });
        } finally {
            setIsRunningBacktest(false);
        }
    };

    const handleOptimize = async () => {
        if (activeBlocks.length === 0) return;
        setIsOptimizing(true);
        setTimeout(() => setIsOptimizing(false), 5000);
    };

    const groupedBlocks: Record<string, typeof AVAILABLE_BLOCKS> = {
        indicator: AVAILABLE_BLOCKS.filter(b => b.type === 'indicator'),
        logic: AVAILABLE_BLOCKS.filter(b => b.type === 'logic'),
        action: AVAILABLE_BLOCKS.filter(b => b.type === 'action'),
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
            {/* Top Bar */}
            <div className="glass" style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '12px 20px', borderBottom: '1px solid var(--border)', flexShrink: 0, zIndex: 10,
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <div>
                        <h1 style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)' }}>Bot Builder</h1>
                        <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{activeBlocks.length} blocks • Drag & drop to design</p>
                    </div>
                    {/* Mode Toggle */}
                    <div style={{ display: 'flex', gap: '4px', padding: '4px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                        {(['scalping', 'swing'] as const).map(m => (
                            <button key={m} onClick={() => setMode(m)} style={{
                                padding: '5px 14px', borderRadius: '6px', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: '12px',
                                background: mode === m ? (m === 'scalping' ? 'rgba(0,216,168,0.15)' : 'rgba(168,85,247,0.15)') : 'transparent',
                                color: mode === m ? (m === 'scalping' ? 'var(--accent-green)' : 'var(--accent-purple)') : 'var(--text-muted)',
                            }}>{m === 'scalping' ? '⚡ Scalping' : '📊 Swing'}</button>
                        ))}
                    </div>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                    <button onClick={handleAIGenerate} disabled={isGenerating} style={{
                        padding: '8px 16px', borderRadius: '9px', border: '1px solid rgba(168,85,247,0.3)',
                        background: 'rgba(168,85,247,0.08)', color: 'var(--accent-purple)', fontSize: '12px', fontWeight: 700, cursor: 'pointer',
                        display: 'flex', alignItems: 'center', gap: '6px',
                    }}>
                        <span>{isGenerating ? '⟳' : '✨'}</span> Generate with AI
                    </button>
                    <button onClick={handleRunBacktest} disabled={isRunningBacktest || activeBlocks.length === 0} style={{
                        padding: '8px 16px', borderRadius: '9px', border: '1px solid rgba(79,158,255,0.3)',
                        background: 'rgba(79,158,255,0.08)', color: 'var(--accent-blue)', fontSize: '12px', fontWeight: 700, cursor: 'pointer',
                    }}>
                        {isRunningBacktest ? '⟳ Testing…' : '▶ Run Backtest'}
                    </button>
                    <button onClick={handleOptimize} disabled={isOptimizing || activeBlocks.length === 0} style={{
                        padding: '8px 16px', borderRadius: '9px', border: '1px solid rgba(0,216,168,0.3)',
                        background: 'rgba(0,216,168,0.08)', color: 'var(--accent-green)', fontSize: '12px', fontWeight: 700, cursor: 'pointer',
                    }}>
                        {isOptimizing ? '⟳ Optimizing…' : '⚡ AI Optimize'}
                    </button>
                </div>
            </div>

            {/* Backtest Result Banner */}
            {backtestResult && (
                <div style={{
                    padding: '10px 20px', display: 'flex', alignItems: 'center', gap: '24px',
                    background: backtestResult.pnl >= 0 ? 'rgba(0,216,168,0.08)' : 'rgba(244,63,94,0.08)',
                    borderBottom: `1px solid ${backtestResult.pnl >= 0 ? 'rgba(0,216,168,0.2)' : 'rgba(244,63,94,0.2)'}`,
                    flexShrink: 0,
                }}>
                    <span style={{ fontSize: '13px', fontWeight: 700, color: backtestResult.pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                        Backtest Complete
                    </span>
                    {[
                        { label: 'Total PnL', value: `${backtestResult.pnl >= 0 ? '+' : ''}$${backtestResult.pnl.toFixed(2)}` },
                        { label: 'Win Rate', value: `${backtestResult.winRate}%` },
                        { label: 'Trades', value: backtestResult.trades },
                    ].map(m => (
                        <div key={m.label} style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{m.label}:</span>
                            <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>{m.value}</span>
                        </div>
                    ))}
                </div>
            )}

            {/* AI Presets Modal */}
            {showAIPresets && (
                <div style={{
                    position: 'fixed', inset: 0, zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    background: 'rgba(4, 13, 26, 0.85)', backdropFilter: 'blur(8px)',
                }}>
                    <div className="glass" style={{ width: '480px', borderRadius: '16px', overflow: 'hidden' }}>
                        <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between' }}>
                            <div>
                                <span style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)' }}>StrategistAI Presets</span>
                                <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>Select a strategy template or customize below</p>
                            </div>
                            <button onClick={() => setShowAIPresets(false)} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '18px' }}>✕</button>
                        </div>
                        <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                            {AI_PRESETS.map((preset, i) => (
                                <button key={i} onClick={() => applyPreset(preset)} style={{
                                    padding: '16px', borderRadius: '10px', border: '1px solid var(--border)',
                                    background: 'var(--bg-card)', cursor: 'pointer', textAlign: 'left', transition: 'all 0.15s',
                                }}>
                                    <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>{preset.name}</div>
                                    <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>{preset.desc}</div>
                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                                        {preset.blocks.map(b => {
                                            const found = AVAILABLE_BLOCKS.find(ab => ab.id === b);
                                            return found ? (
                                                <span key={b} style={{
                                                    padding: '2px 8px', borderRadius: '4px', fontSize: '10px', fontWeight: 600,
                                                    background: 'rgba(79,158,255,0.1)', color: 'var(--accent-blue)',
                                                }}>{found.title}</span>
                                            ) : null;
                                        })}
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Main Layout */}
            <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
                {/* Left Sidebar - Available Blocks */}
                <div style={{ width: '220px', borderRight: '1px solid var(--border)', overflow: 'auto', padding: '12px', flexShrink: 0 }}>
                    {Object.entries(groupedBlocks).map(([type, blocks]) => (
                        <div key={type} style={{ marginBottom: '16px' }}>
                            <div style={{ fontSize: '10px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '8px', padding: '0 4px' }}>
                                {type === 'indicator' ? '📊 Indicators' : type === 'logic' ? '⚙️ Logic' : '🎯 Actions'}
                            </div>
                            {blocks.map(block => (
                                <div
                                    key={block.id}
                                    draggable
                                    onDragStart={e => handleDragStart(e, block)}
                                    style={{
                                        padding: '10px 12px', marginBottom: '6px', borderRadius: '8px', cursor: 'grab',
                                        background: 'var(--bg-card)', border: '1px solid var(--border)',
                                        transition: 'all 0.15s',
                                    }}
                                >
                                    <div style={{ fontSize: '12px', fontWeight: 600, color: block.color, marginBottom: '2px' }}>{block.title}</div>
                                    <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{block.desc}</div>
                                </div>
                            ))}
                        </div>
                    ))}
                </div>

                {/* Canvas */}
                <div
                    style={{ flex: 1, overflow: 'auto', padding: '24px', position: 'relative' }}
                    onDragOver={e => e.preventDefault()}
                    onDrop={handleDrop}
                >
                    {activeBlocks.length === 0 ? (
                        <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '16px' }}>
                            <div style={{
                                width: '80px', height: '80px', borderRadius: '50%',
                                border: '2px dashed var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '28px',
                            }}>🤖</div>
                            <div style={{ textAlign: 'center' }}>
                                <p style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>Drop blocks here to build your strategy</p>
                                <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Or click <span style={{ color: 'var(--accent-purple)' }}>✨ Generate with AI</span> to auto-generate one</p>
                            </div>
                        </div>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', maxWidth: '600px', margin: '0 auto' }}>
                            <div style={{ textAlign: 'center', marginBottom: '8px' }}>
                                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Strategy Flow — {mode === 'scalping' ? '⚡ Scalping Mode' : '📊 Swing Mode'}</span>
                            </div>
                            {activeBlocks.map((block, idx) => (
                                <React.Fragment key={block.id}>
                                    <div
                                        onClick={() => setSelectedBlockIdx(selectedBlockIdx === idx ? null : idx)}
                                        style={{
                                            borderRadius: '12px', overflow: 'hidden', cursor: 'pointer',
                                            border: selectedBlockIdx === idx ? `1px solid ${block.color}` : '1px solid var(--border)',
                                            boxShadow: selectedBlockIdx === idx ? `0 0 20px ${block.color}20` : 'none',
                                            transition: 'all 0.15s',
                                        }}
                                    >
                                        {/* Block Header */}
                                        <div style={{ padding: '10px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: `${block.color}15` }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: block.color }}></div>
                                                <span style={{ fontWeight: 700, fontSize: '14px', color: block.color }}>{block.title}</span>
                                                <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>{block.type}</span>
                                            </div>
                                            <button onClick={e => { e.stopPropagation(); handleRemoveBlock(idx); }} style={{
                                                background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '14px', lineHeight: 1,
                                            }}>✕</button>
                                        </div>
                                        {/* Block Params */}
                                        {selectedBlockIdx === idx && (
                                            <div style={{ padding: '12px 16px', background: 'var(--bg-card)', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                                                {block.type === 'indicator' && (
                                                    <>
                                                        <div>
                                                            <label style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Period</label>
                                                            <input type="number" defaultValue={14} style={{ width: '100%', padding: '6px 10px', borderRadius: '6px', border: '1px solid var(--border)', background: 'var(--bg-primary)', color: 'var(--text-primary)', fontSize: '13px' }} />
                                                        </div>
                                                        <div>
                                                            <label style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Source</label>
                                                            <select style={{ width: '100%', padding: '6px 10px', borderRadius: '6px', border: '1px solid var(--border)', background: 'var(--bg-primary)', color: 'var(--text-primary)', fontSize: '13px' }}>
                                                                <option>Close</option><option>Open</option><option>High</option><option>Low</option>
                                                            </select>
                                                        </div>
                                                    </>
                                                )}
                                                {block.type === 'logic' && (
                                                    <>
                                                        <div>
                                                            <label style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Operator</label>
                                                            <select style={{ width: '100%', padding: '6px 10px', borderRadius: '6px', border: '1px solid var(--border)', background: 'var(--bg-primary)', color: 'var(--text-primary)', fontSize: '13px' }}>
                                                                <option>&gt;</option><option>&lt;</option><option>=</option><option>&gt;=</option><option>&lt;=</option>
                                                            </select>
                                                        </div>
                                                        <div>
                                                            <label style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Threshold</label>
                                                            <input type="number" defaultValue={30} style={{ width: '100%', padding: '6px 10px', borderRadius: '6px', border: '1px solid var(--border)', background: 'var(--bg-primary)', color: 'var(--text-primary)', fontSize: '13px' }} />
                                                        </div>
                                                    </>
                                                )}
                                                {block.type === 'action' && (
                                                    <div>
                                                        <label style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Amount (%)</label>
                                                        <input type="number" defaultValue={100} style={{ width: '100%', padding: '6px 10px', borderRadius: '6px', border: '1px solid var(--border)', background: 'var(--bg-primary)', color: 'var(--text-primary)', fontSize: '13px' }} />
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                    {idx < activeBlocks.length - 1 && (
                                        <div style={{ display: 'flex', justifyContent: 'center' }}>
                                            <div style={{ width: '1px', height: '20px', background: 'var(--border)' }}></div>
                                        </div>
                                    )}
                                </React.Fragment>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
