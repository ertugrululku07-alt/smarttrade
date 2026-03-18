"use client"

import React, { useState } from 'react';
import { getApiUrl } from '@/utils/api';

const ALL_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT',
    'MATIC/USDT', 'UNI/USDT', 'LTC/USDT', 'ATOM/USDT', 'FTM/USDT',
    'NEAR/USDT', 'INJ/USDT', 'TIA/USDT', 'RUNE/USDT', 'AAVE/USDT',
];

const PRESETS: Record<string, string[]> = {
    'Top 5': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT'],
    'Top 10': ALL_SYMBOLS.slice(0, 10),
    'DeFi': ['UNI/USDT', 'AAVE/USDT', 'LINK/USDT', 'INJ/USDT', 'RUNE/USDT'],
    'L1s': ['ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'NEAR/USDT', 'ATOM/USDT', 'DOT/USDT'],
    'All 20': ALL_SYMBOLS,
};

const TIMEFRAMES = ['15m', '1h', '4h'];
const STRATEGIES = [
    { value: 'bb_mr', label: 'BB MR', icon: '📊' },
    { value: 'ict_smc', label: 'ICT/SMC', icon: '🏦' },
];
const LIMITS = [
    { label: '~1 week', value: 168 },
    { label: '~1 month', value: 720 },
    { label: '~3 months', value: 2160 },
    { label: '~6 months', value: 4320 },
];

function MiniEquity({ points, initial }: { points: number[]; initial: number }) {
    if (!points || points.length < 2) return <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>--</span>;
    const min = Math.min(...points), max = Math.max(...points);
    const range = max - min || 1;
    const W = 120, H = 32;
    const xs = points.map((_, i) => (i / (points.length - 1)) * W);
    const ys = points.map(p => H - ((p - min) / range) * (H - 4) - 2);
    const d = `M ${xs[0]} ${ys[0]} ` + xs.slice(1).map((x, i) => `L ${x} ${ys[i + 1]}`).join(' ');
    const profitable = points[points.length - 1] >= initial;
    const color = profitable ? '#00d8a8' : '#f43f5e';
    return (
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
            <path d={d} fill="none" stroke={color} strokeWidth="1.5" />
        </svg>
    );
}

function BigEquity({ results, initial }: { results: any[]; initial: number }) {
    if (!results || results.length === 0) return null;
    const W = 700, H = 180;
    const colors = ['#00d8a8', '#4f9eff', '#f97316', '#a855f7', '#f43f5e', '#fbbf24', '#06b6d4', '#ec4899', '#84cc16', '#8b5cf6'];

    return (
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ height: '180px' }}>
            {/* Baseline */}
            <line x1="0" y1={H / 2} x2={W} y2={H / 2} stroke="rgba(255,255,255,0.06)" strokeDasharray="4 4" />
            {results.map((r, ri) => {
                const pts = r.equity_curve;
                if (!pts || pts.length < 2) return null;
                const allPts = results.flatMap((x: any) => x.equity_curve || []);
                const gMin = Math.min(...allPts), gMax = Math.max(...allPts);
                const range = gMax - gMin || 1;
                const xs = pts.map((_: number, i: number) => (i / (pts.length - 1)) * W);
                const ys = pts.map((p: number) => H - ((p - gMin) / range) * (H - 12) - 6);
                const d = `M ${xs[0]} ${ys[0]} ` + xs.slice(1).map((x: number, i: number) => `L ${x} ${ys[i + 1]}`).join(' ');
                return <path key={ri} d={d} fill="none" stroke={colors[ri % colors.length]} strokeWidth="1.5" opacity="0.85" />;
            })}
        </svg>
    );
}

export default function MultiBacktestPage() {
    const [selected, setSelected] = useState<string[]>(['BTC/USDT', 'ETH/USDT', 'SOL/USDT']);
    const [timeframe, setTf] = useState('1h');
    const [limit, setLimit] = useState(720);
    const [balance, setBal] = useState(1000);
    const [minConf, setMC] = useState(0.55);
    const [strategy, setStrategy] = useState('bb_mr');

    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState('');
    const [error, setError] = useState('');
    const [data, setData] = useState<any>(null);
    const [expandedSymbol, setExpandedSymbol] = useState<string | null>(null);

    const toggleSymbol = (s: string) => {
        setSelected(prev => prev.includes(s) ? prev.filter(x => x !== s) : [...prev, s]);
    };

    const applyPreset = (name: string) => {
        setSelected([...PRESETS[name]]);
    };

    const run = async () => {
        if (selected.length === 0) { setError('En az 1 sembol secin'); return; }
        setLoading(true); setError(''); setData(null);
        setProgress(`${selected.length} sembol test ediliyor...`);
        try {
            const res = await fetch(getApiUrl('/backtest/run-multi'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbols: selected,
                    timeframe,
                    limit,
                    initial_balance: balance,
                    min_confidence: minConf,
                    strategy,
                }),
            });
            if (!res.ok) {
                const e = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
                throw new Error(e.detail || `HTTP ${res.status}`);
            }
            const d = await res.json();
            setData(d);
            setProgress('');
        } catch (e: any) { setError(e.message); setProgress(''); }
        finally { setLoading(false); }
    };

    const colors = ['#00d8a8', '#4f9eff', '#f97316', '#a855f7', '#f43f5e', '#fbbf24', '#06b6d4', '#ec4899', '#84cc16', '#8b5cf6'];
    const summary = data?.summary;
    const results: any[] = data?.results || [];
    const sorted = [...results].sort((a, b) => b.total_pnl - a.total_pnl);

    return (
        <div style={{ padding: '24px', height: '100%', overflow: 'auto' }}>
            {/* Header */}
            <div style={{ marginBottom: '20px' }}>
                <h1 style={{ fontSize: '24px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>
                    Multi-Coin Backtest Lab
                </h1>
                <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                    Compare AI Adaptive Engine across multiple coins simultaneously
                </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '20px' }}>
                {/* ─── Settings Panel ─── */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    {/* Symbol Selection */}
                    <div className="glass" style={{ borderRadius: '14px', padding: '16px' }}>
                        <p style={{ fontSize: '10px', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '8px', textTransform: 'uppercase' }}>
                            Symbols ({selected.length})
                        </p>
                        {/* Presets */}
                        <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap', marginBottom: '10px' }}>
                            {Object.keys(PRESETS).map(p => (
                                <button key={p} onClick={() => applyPreset(p)} style={{
                                    padding: '3px 8px', borderRadius: '5px', border: '1px solid var(--border)',
                                    background: 'var(--bg-card)', color: 'var(--text-muted)', fontSize: '10px',
                                    cursor: 'pointer', fontWeight: 600,
                                }}>
                                    {p}
                                </button>
                            ))}
                        </div>
                        {/* Coin grid */}
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '4px', maxHeight: '240px', overflow: 'auto' }}>
                            {ALL_SYMBOLS.map(s => {
                                const active = selected.includes(s);
                                return (
                                    <button key={s} onClick={() => toggleSymbol(s)} style={{
                                        padding: '6px 8px', borderRadius: '6px', border: active ? '1px solid rgba(0,216,168,0.4)' : '1px solid var(--border)',
                                        background: active ? 'rgba(0,216,168,0.08)' : 'var(--bg-card)',
                                        color: active ? '#00d8a8' : 'var(--text-muted)',
                                        fontSize: '11px', fontWeight: active ? 700 : 500, cursor: 'pointer',
                                        textAlign: 'left', transition: 'all 0.15s',
                                    }}>
                                        {active ? '+ ' : ''}{s.replace('/USDT', '')}
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* Strategy */}
                    <div className="glass" style={{ borderRadius: '14px', padding: '12px 16px' }}>
                        <p style={{ fontSize: '10px', fontWeight: 700, color: 'var(--accent-blue)', marginBottom: '8px', textTransform: 'uppercase' }}>⚡ Strategy</p>
                        <div style={{ display: 'flex', gap: '6px' }}>
                            {STRATEGIES.map(s => (
                                <button key={s.value} onClick={() => setStrategy(s.value)} style={{
                                    flex: 1, padding: '8px', borderRadius: '7px', border: strategy === s.value ? '2px solid var(--accent-blue)' : '1px solid var(--border)',
                                    background: strategy === s.value ? 'rgba(79,158,255,0.1)' : 'var(--bg-card)', cursor: 'pointer', transition: 'all 0.15s',
                                    fontSize: '11px', fontWeight: 700, color: strategy === s.value ? 'var(--accent-blue)' : 'var(--text-muted)',
                                }}>
                                    {s.icon} {s.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Settings */}
                    <div className="glass" style={{ borderRadius: '14px', padding: '16px' }}>
                        <p style={{ fontSize: '10px', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '10px', textTransform: 'uppercase' }}>Settings</p>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '10px' }}>
                            <div>
                                <label style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Timeframe</label>
                                <select value={timeframe} onChange={e => setTf(e.target.value)} style={{ width: '100%', padding: '7px', borderRadius: '7px', border: '1px solid var(--border)', background: 'var(--bg-card)', color: 'var(--text-primary)', fontSize: '12px' }}>
                                    {TIMEFRAMES.map(t => <option key={t} value={t}>{t}</option>)}
                                </select>
                            </div>
                            <div>
                                <label style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Period</label>
                                <select value={limit} onChange={e => setLimit(Number(e.target.value))} style={{ width: '100%', padding: '7px', borderRadius: '7px', border: '1px solid var(--border)', background: 'var(--bg-card)', color: 'var(--text-primary)', fontSize: '11px' }}>
                                    {LIMITS.map((l, i) => <option key={i} value={l.value}>{l.label}</option>)}
                                </select>
                            </div>
                        </div>
                        <div style={{ marginBottom: '10px' }}>
                            <label style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>Initial Balance (USDT)</label>
                            <input type="number" value={balance} onChange={e => setBal(Number(e.target.value))} min={100}
                                style={{ width: '100%', padding: '7px', borderRadius: '7px', border: '1px solid var(--border)', background: 'var(--bg-card)', color: 'var(--text-primary)', fontSize: '13px', outline: 'none' }} />
                        </div>
                        <div>
                            <label style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                                Min Confidence ({Math.round(minConf * 100)}%)
                            </label>
                            <input type="range" min={0.4} max={0.9} step={0.05} value={minConf} onChange={e => setMC(Number(e.target.value))}
                                style={{ width: '100%', accentColor: '#00d8a8' }} />
                        </div>
                    </div>

                    <button onClick={run} disabled={loading || selected.length === 0} style={{
                        padding: '13px', borderRadius: '10px', border: 'none', cursor: loading ? 'wait' : 'pointer',
                        fontWeight: 800, fontSize: '14px',
                        background: loading ? 'var(--bg-card)' : 'linear-gradient(135deg, #4f9eff, #a855f7)',
                        color: loading ? 'var(--text-muted)' : 'white',
                        boxShadow: loading ? 'none' : '0 4px 20px rgba(79,158,255,0.2)',
                        transition: 'all 0.2s',
                    }}>
                        {loading ? `Running... ${progress}` : `Run ${selected.length} Coins`}
                    </button>

                    {error && <div style={{ padding: '10px', borderRadius: '8px', background: 'rgba(244,63,94,0.1)', border: '1px solid rgba(244,63,94,0.2)', fontSize: '11px', color: 'var(--accent-red)', whiteSpace: 'pre-wrap' }}>{error}</div>}

                    {data?.errors?.length > 0 && (
                        <div style={{ padding: '10px', borderRadius: '8px', background: 'rgba(249,115,22,0.08)', border: '1px solid rgba(249,115,22,0.2)', fontSize: '11px', color: '#f97316' }}>
                            {data.errors.map((e: any, i: number) => <div key={i}>{e.symbol}: {e.error}</div>)}
                        </div>
                    )}
                </div>

                {/* ─── Results Panel ─── */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                    {summary && summary.total_symbols > 0 ? (
                        <>
                            {/* Summary Cards */}
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '10px' }}>
                                {[
                                    { label: 'Total PnL', value: `${summary.total_pnl >= 0 ? '+' : ''}$${summary.total_pnl?.toFixed(2)}`, pos: summary.total_pnl >= 0 },
                                    { label: 'Avg Win Rate', value: `${summary.avg_win_rate?.toFixed(1)}%`, pos: summary.avg_win_rate >= 50 },
                                    { label: 'Avg Sharpe', value: summary.avg_sharpe?.toFixed(3), pos: summary.avg_sharpe > 0 },
                                    { label: 'Max Drawdown', value: `-${summary.max_drawdown?.toFixed(1)}%`, pos: summary.max_drawdown < 15 },
                                    { label: 'Profitable', value: `${summary.profitable_symbols}/${summary.total_symbols}`, pos: summary.profitable_symbols > summary.total_symbols / 2 },
                                ].map((c, i) => (
                                    <div key={i} className="glass" style={{ padding: '12px 14px', borderRadius: '10px' }}>
                                        <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px', fontWeight: 600, textTransform: 'uppercase' }}>{c.label}</div>
                                        <div style={{ fontSize: '18px', fontWeight: 800, color: c.pos ? 'var(--accent-green)' : 'var(--accent-red)' }}>{c.value}</div>
                                    </div>
                                ))}
                            </div>

                            {/* Combined Equity Chart */}
                            <div className="glass" style={{ borderRadius: '14px', padding: '16px' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                                    <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>Combined Equity Curves</span>
                                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                                        {sorted.slice(0, 8).map((r, i) => (
                                            <span key={r.symbol} style={{ fontSize: '9px', color: colors[i % colors.length], fontWeight: 600 }}>
                                                {r.symbol.replace('/USDT', '')}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                                <BigEquity results={sorted.slice(0, 8)} initial={balance} />
                            </div>

                            {/* Coin Comparison Table */}
                            <div className="glass" style={{ borderRadius: '14px', overflow: 'hidden' }}>
                                <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between' }}>
                                    <span style={{ fontSize: '13px', fontWeight: 700 }}>Coin Performance Comparison</span>
                                    <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                                        Best: {summary.best_symbol} (+${summary.best_pnl}) | Worst: {summary.worst_symbol} (${summary.worst_pnl})
                                    </span>
                                </div>
                                <div style={{ maxHeight: '500px', overflow: 'auto' }}>
                                    <table className="data-table" style={{ width: '100%' }}>
                                        <thead>
                                            <tr>
                                                <th style={{ textAlign: 'left' }}>Symbol</th>
                                                <th>Equity</th>
                                                <th>Trades</th>
                                                <th>Win Rate</th>
                                                <th>PnL</th>
                                                <th>PnL %</th>
                                                <th>Max DD</th>
                                                <th>Sharpe</th>
                                                <th>PF</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {sorted.map((r, ri) => {
                                                const isExpanded = expandedSymbol === r.symbol;
                                                return (
                                                    <React.Fragment key={r.symbol}>
                                                        <tr onClick={() => setExpandedSymbol(isExpanded ? null : r.symbol)}
                                                            style={{ cursor: 'pointer', background: isExpanded ? 'rgba(79,158,255,0.04)' : undefined }}>
                                                            <td style={{ fontWeight: 700, fontSize: '12px' }}>
                                                                <span style={{ color: colors[ri % colors.length], marginRight: '4px' }}>●</span>
                                                                {r.symbol.replace('/USDT', '')}
                                                            </td>
                                                            <td><MiniEquity points={r.equity_curve} initial={balance} /></td>
                                                            <td style={{ fontSize: '11px' }}>{r.total_trades}</td>
                                                            <td style={{ fontSize: '11px', fontWeight: 600, color: r.win_rate >= 50 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                                                {r.win_rate?.toFixed(1)}%
                                                            </td>
                                                            <td style={{ fontSize: '12px', fontWeight: 700, color: r.total_pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                                                {r.total_pnl >= 0 ? '+' : ''}${r.total_pnl?.toFixed(2)}
                                                            </td>
                                                            <td style={{ fontSize: '11px', color: r.total_pnl_pct >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                                                {r.total_pnl_pct >= 0 ? '+' : ''}{r.total_pnl_pct?.toFixed(2)}%
                                                            </td>
                                                            <td style={{ fontSize: '11px', color: r.max_drawdown < 15 ? 'var(--text-muted)' : 'var(--accent-red)' }}>
                                                                -{r.max_drawdown?.toFixed(1)}%
                                                            </td>
                                                            <td style={{ fontSize: '11px', fontWeight: 600, color: r.sharpe_ratio > 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                                                {r.sharpe_ratio?.toFixed(2)}
                                                            </td>
                                                            <td style={{ fontSize: '11px', fontWeight: 600 }}>
                                                                {r.profit_factor?.toFixed(2)}
                                                            </td>
                                                        </tr>
                                                        {/* Expanded Trade Details */}
                                                        {isExpanded && r.trades?.length > 0 && (
                                                            <tr>
                                                                <td colSpan={9} style={{ padding: '0' }}>
                                                                    <div style={{ padding: '12px 16px', background: 'rgba(79,158,255,0.02)', borderTop: '1px solid var(--border)' }}>
                                                                        <div style={{ display: 'flex', gap: '16px', marginBottom: '10px', flexWrap: 'wrap' }}>
                                                                            {r.strategy_usage && Object.entries(r.strategy_usage).map(([s, c]: any) => (
                                                                                <span key={s} style={{ fontSize: '10px', padding: '2px 8px', borderRadius: '4px', background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                                                                                    <b style={{ color: 'var(--accent-blue)' }}>{s}</b>
                                                                                    <span style={{ color: 'var(--text-muted)', marginLeft: '4px' }}>{c}x</span>
                                                                                </span>
                                                                            ))}
                                                                            <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                                                                                {r.date_range?.from?.slice(0, 10)} - {r.date_range?.to?.slice(0, 10)}
                                                                            </span>
                                                                        </div>
                                                                        <table className="data-table" style={{ width: '100%' }}>
                                                                            <thead>
                                                                                <tr><th>Side</th><th>Strategy</th><th>Entry</th><th>Exit</th><th>PnL%</th><th>Outcome</th><th>Bars</th></tr>
                                                                            </thead>
                                                                            <tbody>
                                                                                {r.trades.slice(-15).reverse().map((t: any, ti: number) => (
                                                                                    <tr key={ti}>
                                                                                        <td>
                                                                                            <span style={{ padding: '1px 5px', borderRadius: '3px', fontSize: '10px', fontWeight: 700, background: t.direction === 'LONG' ? 'rgba(0,216,168,0.1)' : 'rgba(244,63,94,0.1)', color: t.direction === 'LONG' ? '#00d8a8' : '#f43f5e' }}>
                                                                                                {t.direction}
                                                                                            </span>
                                                                                        </td>
                                                                                        <td style={{ fontSize: '10px', color: 'var(--accent-blue)' }}>{t.strategy}</td>
                                                                                        <td style={{ fontSize: '10px' }}>${t.entry_price?.toLocaleString()}</td>
                                                                                        <td style={{ fontSize: '10px' }}>${t.exit_price?.toLocaleString()}</td>
                                                                                        <td style={{ fontSize: '10px', fontWeight: 700, color: t.pnl_pct >= 0 ? '#00d8a8' : '#f43f5e' }}>
                                                                                            {t.pnl_pct >= 0 ? '+' : ''}{t.pnl_pct?.toFixed(2)}%
                                                                                        </td>
                                                                                        <td>
                                                                                            <span style={{ fontSize: '9px', padding: '1px 5px', borderRadius: '3px', fontWeight: 600, background: t.outcome === 'TP' ? 'rgba(34,197,94,0.1)' : t.outcome === 'SL' ? 'rgba(244,63,94,0.1)' : 'rgba(249,115,22,0.1)', color: t.outcome === 'TP' ? '#22c55e' : t.outcome === 'SL' ? '#f43f5e' : '#f97316' }}>
                                                                                                {t.outcome}
                                                                                            </span>
                                                                                        </td>
                                                                                        <td style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{t.bars_held}</td>
                                                                                    </tr>
                                                                                ))}
                                                                            </tbody>
                                                                        </table>
                                                                    </div>
                                                                </td>
                                                            </tr>
                                                        )}
                                                    </React.Fragment>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </>
                    ) : !loading && (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '400px', gap: '16px' }}>
                            <div style={{ fontSize: '56px' }}>📊</div>
                            <div style={{ textAlign: 'center' }}>
                                <p style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '8px' }}>
                                    Multi-Coin Backtest Ready
                                </p>
                                <p style={{ fontSize: '13px', color: 'var(--text-muted)', maxWidth: '400px' }}>
                                    Select coins, configure parameters, and run to compare AI engine performance across multiple markets.
                                </p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
