"use client"

import React, { useState } from 'react';
import { getApiUrl } from '@/utils/api';

const SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'AVAX/USDT', 'LINK/USDT'];
const TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d'];
const STRATEGIES = [
    { value: 'bb_mr', label: 'BB Mean Reversion', icon: '📊', desc: 'Bollinger Band bounce + RSI' },
    { value: 'ict_smc', label: 'ICT / SMC', icon: '🏦', desc: 'Smart Money Concepts + Order Flow' },
];
const LIMITS = [
    { label: '~3 days', value: 72 },
    { label: '~1 week', value: 168 },
    { label: '~1 month', value: 720 },
    { label: '~3 months', value: 2160 },
    { label: '~6 months', value: 4320 },
    { label: '~1 year', value: 8760 },
    { label: '~1.5 years', value: 13000 },
];
const REGIME_LABELS: Record<string, { label: string; color: string }> = {
    strong_trend_up: { label: '📈 Strong Trend ↑', color: '#00d8a8' },
    strong_trend_down: { label: '📉 Strong Trend ↓', color: '#f43f5e' },
    ranging: { label: '↔️ Ranging', color: '#fbbf24' },
    high_volatility: { label: '⚡ High Volatility', color: '#f97316' },
    breakout: { label: '💥 Breakout', color: '#a855f7' },
};

function EquityChart({ points, initialBalance }: { points: number[]; initialBalance: number }) {
    if (!points || points.length < 2) return null;
    const min = Math.min(...points), max = Math.max(...points);
    const range = max - min || 1;
    const W = 600, H = 140;
    const xs = points.map((_, i) => (i / (points.length - 1)) * W);
    const ys = points.map(p => H - ((p - min) / range) * (H - 20) - 10);
    const d = `M ${xs[0]} ${ys[0]} ` + xs.slice(1).map((x, i) => `L ${x} ${ys[i + 1]}`).join(' ');
    const fill = `${d} L ${xs[xs.length - 1]} ${H} L ${xs[0]} ${H} Z`;
    const profitable = points[points.length - 1] >= initialBalance;
    const color = profitable ? '#00d8a8' : '#f43f5e';
    const baseY = H - ((initialBalance - min) / range) * (H - 20) - 10;
    return (
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ height: '140px' }}>
            <defs>
                <linearGradient id="eqG2" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={color} stopOpacity="0.35" />
                    <stop offset="100%" stopColor={color} stopOpacity="0" />
                </linearGradient>
            </defs>
            <line x1="0" y1={baseY} x2={W} y2={baseY} stroke="rgba(255,255,255,0.08)" strokeDasharray="4 4" />
            <path d={fill} fill="url(#eqG2)" />
            <path d={d} fill="none" stroke={color} strokeWidth="2" />
        </svg>
    );
}

export default function BacktestPage() {
    const [symbol, setSymbol] = useState('BTC/USDT');
    const [timeframe, setTf] = useState('1h');
    const [limit, setLimit] = useState(720);
    const [balance, setBal] = useState(1000);
    const [tradeSize, setTS] = useState(15);
    const [minConf, setMC] = useState(0.55);
    const [strategy, setStrategy] = useState('bb_mr');

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [adaptiveRes, setARes] = useState<any>(null);

    const api = async (path: string, method = 'POST', body?: any) => {
        const res = await fetch(getApiUrl(path), {
            method,
            headers: body ? { 'Content-Type': 'application/json' } : {},
            body: body ? JSON.stringify(body) : undefined,
        });
        if (!res.ok) {
            const e = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
            throw new Error(e.detail || `HTTP ${res.status}`);
        }
        return res.json();
    };

    const run = async () => {
        setLoading(true); setError(''); setARes(null);
        try {
            const d = await api('/backtest/run-adaptive', 'POST', { symbol, timeframe, limit, initial_balance: balance, trade_size_pct: tradeSize, min_confidence: minConf, strategy });
            setARes(d);
        } catch (e: any) { setError(e.message); }
        finally { setLoading(false); }
    };

    return (
        <div style={{ padding: '24px', height: '100%', overflow: 'auto' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '12px' }}>
                <div>
                    <h1 style={{ fontSize: '24px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>AI Adaptive Backtest Lab</h1>
                    <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Real Binance data · Dynamic Regime Switching · Meta-Labelling</p>
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: '20px' }}>
                {/* ─── Settings Panel ─────────────────────────────────────── */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    {/* Market */}
                    <div className="glass" style={{ borderRadius: '14px', padding: '16px', border: '1px solid rgba(79,158,255,0.15)' }}>
                        <p style={{ fontSize: '10px', fontWeight: 700, color: 'var(--accent-blue)', marginBottom: '10px', textTransform: 'uppercase' }}>⚡ Strategy</p>
                        <div style={{ display: 'flex', gap: '6px', marginBottom: '0' }}>
                            {STRATEGIES.map(s => (
                                <button key={s.value} onClick={() => setStrategy(s.value)} style={{
                                    flex: 1, padding: '10px 8px', borderRadius: '8px', border: strategy === s.value ? '2px solid var(--accent-blue)' : '1px solid var(--border)',
                                    background: strategy === s.value ? 'rgba(79,158,255,0.1)' : 'var(--bg-card)', cursor: 'pointer', transition: 'all 0.2s',
                                }}>
                                    <div style={{ fontSize: '18px', marginBottom: '4px' }}>{s.icon}</div>
                                    <div style={{ fontSize: '11px', fontWeight: 700, color: strategy === s.value ? 'var(--accent-blue)' : 'var(--text-primary)' }}>{s.label}</div>
                                    <div style={{ fontSize: '9px', color: 'var(--text-muted)', marginTop: '2px' }}>{s.desc}</div>
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="glass" style={{ borderRadius: '14px', padding: '16px' }}>
                        <p style={{ fontSize: '10px', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '10px', textTransform: 'uppercase' }}>Market</p>
                        <div style={{ marginBottom: '10px' }}>
                            <label style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Symbol</label>
                            <select value={symbol} onChange={e => setSymbol(e.target.value)} style={{ width: '100%', padding: '8px', borderRadius: '7px', border: '1px solid var(--border)', background: 'var(--bg-card)', color: 'var(--text-primary)', fontSize: '13px' }}>
                                {SYMBOLS.map(s => <option key={s} value={s}>{s}</option>)}
                            </select>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                            <div>
                                <label style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Timeframe</label>
                                <select value={timeframe} onChange={e => setTf(e.target.value)} style={{ width: '100%', padding: '8px', borderRadius: '7px', border: '1px solid var(--border)', background: 'var(--bg-card)', color: 'var(--text-primary)', fontSize: '12px' }}>
                                    {TIMEFRAMES.map(t => <option key={t} value={t}>{t}</option>)}
                                </select>
                            </div>
                            <div>
                                <label style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Period</label>
                                <select value={limit} onChange={e => setLimit(Number(e.target.value))} style={{ width: '100%', padding: '8px', borderRadius: '7px', border: '1px solid var(--border)', background: 'var(--bg-card)', color: 'var(--text-primary)', fontSize: '11px' }}>
                                    {LIMITS.map((l, i) => <option key={i} value={l.value}>{l.label}</option>)}
                                </select>
                            </div>
                        </div>
                    </div>

                    {/* Capital */}
                    <div className="glass" style={{ borderRadius: '14px', padding: '16px' }}>
                        <p style={{ fontSize: '10px', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '10px', textTransform: 'uppercase' }}>Capital</p>
                        {[
                            { label: 'Initial Balance (USDT)', val: balance, set: setBal, min: 100 },
                            { label: 'Trade Size %', val: tradeSize, set: setTS, min: 1 },
                        ].map(f => (
                            <div key={f.label} style={{ marginBottom: '10px' }}>
                                <label style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>{f.label}</label>
                                <input type="number" value={f.val} onChange={e => f.set(Number(e.target.value))} min={f.min} style={{ width: '100%', padding: '8px', borderRadius: '7px', border: '1px solid var(--border)', background: 'var(--bg-card)', color: 'var(--text-primary)', fontSize: '13px', outline: 'none' }} />
                            </div>
                        ))}
                    </div>

                    {/* AI settings */}
                    <div className="glass" style={{ borderRadius: '14px', padding: '16px', border: '1px solid rgba(0,216,168,0.15)' }}>
                        <p style={{ fontSize: '10px', fontWeight: 700, color: 'var(--accent-green)', marginBottom: '10px', textTransform: 'uppercase' }}>🧠 AI Settings</p>
                        <label style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Min Confidence ({Math.round(minConf * 100)}%)</label>
                        <input type="range" min={0.4} max={0.9} step={0.05} value={minConf} onChange={e => setMC(Number(e.target.value))} style={{ width: '100%', accentColor: '#00d8a8' }} />
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: 'var(--text-muted)', marginTop: '2px' }}><span>More trades</span><span>Fewer, better</span></div>
                    </div>

                    <button onClick={run} disabled={loading} style={{
                        padding: '13px', borderRadius: '10px', border: 'none', cursor: loading ? 'wait' : 'pointer', fontWeight: 800, fontSize: '14px',
                        background: loading ? 'var(--bg-card)' : strategy === 'ict_smc' ? 'linear-gradient(135deg, #a855f7, #4f9eff)' : 'linear-gradient(135deg, #00d8a8, #4f9eff)',
                        color: loading ? 'var(--text-muted)' : 'white',
                        boxShadow: loading ? 'none' : '0 4px 20px rgba(0,216,168,0.2)',
                        transition: 'all 0.2s',
                    }}>{loading ? '⟳ Running...' : strategy === 'ict_smc' ? '🏦 Run ICT/SMC Backtest' : '📊 Run BB MR Backtest'}</button>

                    {error && <div style={{ padding: '10px', borderRadius: '8px', background: 'rgba(244,63,94,0.1)', border: '1px solid rgba(244,63,94,0.2)', fontSize: '11px', color: 'var(--accent-red)', whiteSpace: 'pre-wrap' }}>⚠️ {error}</div>}
                </div>

                {/* ─── Results Panel ───────────────────────────────────────── */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>

                    {/* ══ ADAPTIVE RESULTS ══ */}
                    {adaptiveRes?.metrics ? (() => {
                        const m = adaptiveRes.metrics;
                        return (
                            <>
                                <div style={{ padding: '10px 14px', borderRadius: '10px', background: strategy === 'ict_smc' ? 'rgba(168,85,247,0.06)' : 'rgba(0,216,168,0.06)', border: `1px solid ${strategy === 'ict_smc' ? 'rgba(168,85,247,0.15)' : 'rgba(0,216,168,0.15)'}`, display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
                                    <span style={{ fontSize: '12px', color: strategy === 'ict_smc' ? '#a855f7' : '#00d8a8', fontWeight: 700 }}>{strategy === 'ict_smc' ? '🏦 ICT/SMC Engine' : '📊 BB Mean Reversion'}</span>
                                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>📅 {adaptiveRes.date_range?.from?.slice(0, 10)} → {adaptiveRes.date_range?.to?.slice(0, 10)}</span>
                                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>🔄 {m.regime_changes} regime changes</span>
                                </div>
                                {m.strategy_usage && Object.keys(m.strategy_usage).length > 0 && (
                                    <div className="glass" style={{ borderRadius: '12px', padding: '12px 16px' }}>
                                        <p style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '8px' }}>Strategy Distribution</p>
                                        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                                            {Object.entries(m.strategy_usage).map(([s, c]: any) => (
                                                <div key={s} style={{ padding: '4px 10px', borderRadius: '6px', background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                                                    <span style={{ fontSize: '11px', color: 'var(--text-primary)', fontWeight: 600 }}>{s}</span>
                                                    <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: '6px' }}>{c}×</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: '10px' }}>
                                    {[
                                        { label: 'Total PnL', value: `${m.total_pnl >= 0 ? '+' : ''}$${m.total_pnl?.toFixed(2)}`, sub: `${m.total_pnl_pct?.toFixed(2)}%`, pos: m.total_pnl >= 0 },
                                        { label: 'Win Rate', value: `${m.win_rate?.toFixed(1)}%`, sub: `${m.win_trades}W/${m.loss_trades}L`, pos: m.win_rate >= 50 },
                                        { label: 'Sharpe', value: m.sharpe_ratio?.toFixed(3), sub: m.sharpe_ratio > 1 ? '✓ Good' : 'Fair', pos: m.sharpe_ratio > 0 },
                                        { label: 'Max Drawdown', value: `-${m.max_drawdown?.toFixed(2)}%`, sub: 'worst dip', pos: m.max_drawdown < 15 },
                                    ].map((c, i) => (
                                        <div key={i} className="glass" style={{ padding: '12px 14px', borderRadius: '10px' }}>
                                            <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '6px', fontWeight: 600, textTransform: 'uppercase' }}>{c.label}</div>
                                            <div style={{ fontSize: '18px', fontWeight: 800, color: c.pos ? 'var(--accent-green)' : 'var(--accent-red)', marginBottom: '2px' }}>{c.value}</div>
                                            <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{c.sub}</div>
                                        </div>
                                    ))}
                                </div>
                                <div className="glass" style={{ borderRadius: '14px', padding: '18px' }}>
                                    <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>Equity Curve — Adaptive AI</div>
                                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '12px' }}>${m.initial_balance?.toFixed(0)} → ${m.final_balance?.toFixed(2)}</div>
                                    <EquityChart points={m.equity_curve || []} initialBalance={m.initial_balance} />
                                </div>
                                {m.trades?.length > 0 && (
                                    <div className="glass" style={{ borderRadius: '14px', overflow: 'hidden' }}>
                                        <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between' }}>
                                            <span style={{ fontSize: '13px', fontWeight: 700 }}>Adaptive Trade Log (last 30)</span>
                                            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Found {m.total_trades} trades</span>
                                        </div>
                                        <div style={{ maxHeight: '280px', overflow: 'auto' }}>
                                            <table className="data-table">
                                                <thead><tr><th>Regime</th><th>Strategy</th><th>Side</th><th>Entry</th><th>Exit</th><th>PnL</th><th>PnL%</th><th>Reason</th></tr></thead>
                                                <tbody>
                                                    {m.trades.slice(-30).reverse().map((t: any, i: number) => (
                                                        <tr key={i}>
                                                            <td><span style={{ fontSize: '10px', fontWeight: 600, color: REGIME_LABELS[t.regime]?.color || 'var(--text-muted)' }}>{REGIME_LABELS[t.regime]?.label || t.regime}</span></td>
                                                            <td style={{ fontSize: '11px', color: 'var(--accent-blue)' }}>{t.strategy}</td>
                                                            <td><span style={{ padding: '2px 6px', borderRadius: '4px', fontSize: '10px', fontWeight: 700, background: t.direction === 'LONG' ? 'rgba(0,216,168,0.12)' : 'rgba(244,63,94,0.12)', color: t.direction === 'LONG' ? 'var(--accent-green)' : 'var(--accent-red)' }}>{t.direction}</span></td>
                                                            <td style={{ fontSize: '11px' }}>${t.entry_price?.toLocaleString()}</td>
                                                            <td style={{ fontSize: '11px' }}>${t.exit_price?.toLocaleString()}</td>
                                                            <td style={{ fontWeight: 700, color: t.pnl_pct >= 0 ? 'var(--accent-green)' : 'var(--accent-red)', fontSize: '11px' }}>{t.pnl_pct >= 0 ? '+' : ''}${(t.pnl_pct / 100 * m.initial_balance * (t.position_size || 0.15)).toFixed(2)}</td>
                                                            <td style={{ fontSize: '11px', color: t.pnl_pct >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>{t.pnl_pct >= 0 ? '+' : ''}{t.pnl_pct?.toFixed(2)}%</td>
                                                            <td><span style={{ fontSize: '9px', padding: '2px 5px', borderRadius: '3px', fontWeight: 600, background: t.outcome === 'TP' ? 'rgba(34,197,94,0.1)' : t.outcome === 'SL' ? 'rgba(244,63,94,0.1)' : 'rgba(249,115,22,0.1)', color: t.outcome === 'TP' ? '#22c55e' : t.outcome === 'SL' ? 'var(--accent-red)' : '#f97316' }}>{t.outcome}</span></td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                )}
                            </>
                        );
                    })() : !loading && (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '300px', gap: '16px' }}>
                            <div style={{ fontSize: '56px' }}>🧠</div>
                            <div style={{ textAlign: 'center' }}>
                                <p style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '8px' }}>
                                    Adaptive AI Engine Ready
                                </p>
                                <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                                    Configure your test and run to see dynamic regime shifting in action.
                                </p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div >
    );
}
