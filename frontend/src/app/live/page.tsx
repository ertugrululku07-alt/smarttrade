"use client"

import React, { useState, useEffect, useMemo } from 'react';
import { getApiUrl } from '@/utils/api';

// --- Mini PnL Chart ---
const PnLChart = ({ data }: { data: any[] }) => {
    if (!data || data.length < 2) return null;
    const values = data.map(d => d.pct);
    const min = Math.min(...values, -0.5);
    const max = Math.max(...values, 0.5);
    const range = max - min;
    const width = 120; const height = 40; const padding = 2;
    const points = data.map((d, i) => {
        const x = (i / (data.length - 1)) * width;
        const y = height - ((d.pct - min) / range) * (height - padding * 2) - padding;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    const isProfit = data[data.length - 1].pct >= 0;
    const color = isProfit ? '#00d8a8' : '#f43f5e';
    const gradientId = `grad-${Math.random().toString(36).substr(2, 9)}`;
    return (
        <div style={{ position: 'relative', width, height, marginLeft: 12 }}>
            <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
                <defs><linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stopColor={color} stopOpacity="0.2" /><stop offset="100%" stopColor={color} stopOpacity="0" /></linearGradient></defs>
                <path d={`M 0,${height} L ${points} L ${width},${height} Z`} fill={`url(#${gradientId})`} />
                <polyline fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" points={points} />
            </svg>
        </div>
    );
};

// --- Detailed Modal Chart ---
const DetailedPnLChart = ({ data, pair }: { data: any[], pair: string }) => {
    if (!data || data.length < 2) return <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>Not enough data...</div>;
    const values = data.map(d => d.pct);
    const min = Math.min(...values, -0.2); const max = Math.max(...values, 0.2); const range = max - min || 1;
    const width = 600; const height = 240; const padding = 30;
    const points = data.map((d, i) => {
        const x = padding + (i / (data.length - 1)) * (width - padding * 2);
        const y = (height - padding) - ((d.pct - min) / range) * (height - padding * 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    const isProfit = data[data.length - 1].pct >= 0;
    const color = isProfit ? '#00d8a8' : '#f43f5e';
    return (
        <div style={{ width: '100%', background: 'rgba(0,0,0,0.2)', borderRadius: '12px', padding: '20px', border: '1px solid var(--border)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Performance ({pair})</span>
                <span style={{ fontSize: '12px', fontWeight: 700, color }}>{data[data.length - 1].pct.toFixed(2)}%</span>
            </div>
            <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} style={{ overflow: 'visible' }}>
                {[min, 0, max].map((v, i) => { const y = (height - padding) - ((v - min) / range) * (height - padding * 2); return (<g key={`g-${i}`}><line x1={padding} y1={y} x2={width - padding} y2={y} stroke="rgba(255,255,255,0.05)" strokeDasharray="4" /><text x={padding - 5} y={y + 4} textAnchor="end" fill="var(--text-muted)" style={{ fontSize: '10px' }}>{v.toFixed(1)}%</text></g>); })}
                <polyline fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" points={points} />
                <defs><linearGradient id="detailed-grad" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" stopColor={color} /><stop offset="100%" stopColor="transparent" /></linearGradient></defs>
            </svg>
        </div>
    );
};

// --- Inline settings input ---
const SettingInput = ({ label, value, onChange, step, min, max, unit }: any) => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <label style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{label}</label>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            {unit === '$' && <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>$</span>}
            <input type="number" step={step} min={min} max={max} value={value} onChange={e => onChange(parseFloat(e.target.value) || 0)}
                style={{ width: '80px', padding: '5px 8px', borderRadius: '6px', border: '1px solid var(--border)', background: 'rgba(0,0,0,0.3)', color: 'var(--text-primary)', fontSize: '13px', textAlign: 'center' }} />
            {unit && unit !== '$' && <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{unit}</span>}
        </div>
    </div>
);

// --- Toggle Switch ---
const ToggleSwitch = ({ label, enabled, onToggle, color }: { label: string, enabled: boolean, onToggle: () => void, color: string }) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }} onClick={onToggle}>
        <div style={{
            width: 36, height: 20, borderRadius: 10, position: 'relative', transition: 'all 0.2s',
            background: enabled ? color : 'rgba(255,255,255,0.1)', border: `1px solid ${enabled ? color : 'rgba(255,255,255,0.15)'}`,
        }}>
            <div style={{
                width: 16, height: 16, borderRadius: '50%', background: '#fff', position: 'absolute', top: 1,
                left: enabled ? 17 : 1, transition: 'left 0.2s', boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
            }} />
        </div>
        <span style={{ fontSize: '12px', fontWeight: 600, color: enabled ? color : 'var(--text-muted)' }}>{label}</span>
    </div>
);

export default function LiveTradingPage() {
    const [statusData, setStatusData] = useState<any>(null);
    const [selectedTrade, setSelectedTrade] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [actionLoading, setActionLoading] = useState(false);
    const [closingTradeId, setClosingTradeId] = useState<string | null>(null);
    const [v3Stats, setV3Stats] = useState<any>(null);
    const [activeTab, setActiveTab] = useState<'trading' | 'analytics'>('trading');

    // General Settings
    const [riskSettings, setRiskSettings] = useState<any>(null);
    const [formMaxTrades, setFormMaxTrades] = useState(5);
    const [formBalance, setFormBalance] = useState(10000);
    const [savingSettings, setSavingSettings] = useState(false);
    const [settingsLoaded, setSettingsLoaded] = useState(false);

    // Global Trading Settings
    const [formLeverage, setFormLeverage] = useState(10);
    const [formMaxNotional, setFormMaxNotional] = useState(300);
    const [formMaxLoss, setFormMaxLoss] = useState(5);
    // Strategy Toggles
    const [bbEnabled, setBbEnabled] = useState(true);
    const [ictEnabled, setIctEnabled] = useState(true);
    const [trendEnabled, setTrendEnabled] = useState(true);

    // Analytics State
    const [analyticsData, setAnalyticsData] = useState<any[]>([]);
    const [filterStrategy, setFilterStrategy] = useState('all');
    const [filterSide, setFilterSide] = useState('all');
    const [filterResult, setFilterResult] = useState('all');
    const [filterStartDate, setFilterStartDate] = useState('');
    const [filterEndDate, setFilterEndDate] = useState('');
    const [resetBalance, setResetBalance] = useState(10000);
    const [showResetConfirm, setShowResetConfirm] = useState(false);

    const [isMounted, setIsMounted] = useState(false);
    useEffect(() => { setIsMounted(true); }, []);

    const fetchStatus = async () => {
        try {
            const [statusRes, statsRes] = await Promise.all([
                fetch(getApiUrl("/live/quant/status")),
                fetch(getApiUrl("/live/v3/stats")),
            ]);
            if (statusRes.ok) setStatusData(await statusRes.json());
            if (statsRes.ok) setV3Stats(await statsRes.json());
        } catch (e) { console.error(e); } finally { setLoading(false); }
    };

    const fetchRiskSettings = async () => {
        try {
            const res = await fetch(getApiUrl("/live/quant/risk-settings"));
            if (res.ok) {
                const data = await res.json();
                setRiskSettings(data);
                if (!settingsLoaded) {
                    setFormMaxTrades(data.max_open_trades || 5);
                    setFormBalance(data.balance || 10000);
                    setFormLeverage(data.leverage || 10);
                    setFormMaxNotional(data.max_notional || 300);
                    setFormMaxLoss(data.max_loss_cap || 5);
                    setBbEnabled(data.bb_mr_enabled !== false);
                    setIctEnabled(data.ict_smc_enabled !== false);
                    setTrendEnabled(data.trend_v4_enabled !== false);
                    setSettingsLoaded(true);
                }
            }
        } catch (e) { console.error(e); }
    };

    const fetchAnalytics = async () => {
        try {
            const params = new URLSearchParams();
            if (filterStartDate) {
                const ts = Math.floor(new Date(filterStartDate).getTime() / 1000);
                if (!isNaN(ts)) params.append('start_time', ts.toString());
            }
            if (filterEndDate) {
                const ts = Math.floor(new Date(filterEndDate).getTime() / 1000) + 86399; // end of day
                if (!isNaN(ts)) params.append('end_time', ts.toString());
            }
            
            const url = `/live/quant/analytics${params.toString() ? '?' + params.toString() : ''}`;
            const res = await fetch(getApiUrl(url));
            if (res.ok) {
                const data = await res.json();
                setAnalyticsData(data.trades || []);
            }
        } catch (e) { console.error(e); }
    };

    useEffect(() => {
        fetchStatus();
        fetchRiskSettings();
        const interval = setInterval(fetchStatus, 3000);
        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        if (activeTab === 'analytics') fetchAnalytics();
    }, [activeTab]);

    const handleAction = async (action: 'start' | 'stop') => {
        setActionLoading(true);
        try {
            const res = await fetch(getApiUrl(`/live/quant/${action}`), { method: 'POST' });
            if (res.ok) await fetchStatus();
        } catch (e) { console.error(e); } finally { setActionLoading(false); }
    };

    const handleCloseTrade = async (tradeId: string) => {
        setClosingTradeId(tradeId);
        try {
            const res = await fetch(getApiUrl(`/live/quant/close-trade/${tradeId}`), { method: 'POST' });
            if (res.ok) await fetchStatus();
            else { const err = await res.json(); alert(err.message || 'Error'); }
        } catch (e) { alert('Backend connection failed'); } finally { setClosingTradeId(null); }
    };

    const handleSaveAllSettings = async () => {
        setSavingSettings(true);
        try {
            const res = await fetch(getApiUrl('/live/quant/settings'), {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    max_open_trades: formMaxTrades,
                    balance: formBalance,
                    leverage: formLeverage,
                    max_notional: formMaxNotional,
                    max_loss_cap: formMaxLoss,
                    bb_mr_enabled: bbEnabled,
                    ict_smc_enabled: ictEnabled,
                    trend_v4_enabled: trendEnabled,
                })
            });
            if (res.ok) { await fetchStatus(); await fetchRiskSettings(); }
            else alert('Settings save failed');
        } catch (e) { console.error(e); } finally { setSavingSettings(false); }
    };

    const handleReset = async () => {
        try {
            const res = await fetch(getApiUrl('/live/quant/reset'), {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ new_balance: resetBalance })
            });
            if (res.ok) {
                setShowResetConfirm(false);
                await fetchStatus();
                await fetchRiskSettings();
                await fetchAnalytics();
            }
        } catch (e) { alert('Reset failed'); }
    };

    // Analytics computations
    const filteredTrades = useMemo(() => {
        return analyticsData.filter(t => {
            if (filterStrategy !== 'all' && t.strategy !== filterStrategy) return false;
            if (filterSide !== 'all' && t.side !== filterSide) return false;
            if (filterResult === 'win' && t.pnl <= 0) return false;
            if (filterResult === 'loss' && t.pnl >= 0) return false;
            return true;
        });
    }, [analyticsData, filterStrategy, filterSide, filterResult]);

    const analyticsStats = useMemo(() => {
        const trades = filteredTrades;
        if (!trades.length) return null;
        const wins = trades.filter(t => t.pnl > 0);
        const losses = trades.filter(t => t.pnl <= 0);
        const totalPnl = trades.reduce((s, t) => s + t.pnl, 0);
        const avgWin = wins.length ? wins.reduce((s, t) => s + t.pnl, 0) / wins.length : 0;
        const avgLoss = losses.length ? losses.reduce((s, t) => s + t.pnl, 0) / losses.length : 0;
        const maxWin = wins.length ? Math.max(...wins.map(t => t.pnl)) : 0;
        const maxLoss = losses.length ? Math.min(...losses.map(t => t.pnl)) : 0;
        const wr = (wins.length / trades.length * 100);
        const rr = avgLoss !== 0 ? Math.abs(avgWin / avgLoss) : 0;
        let equity = 0;
        const equityCurve = trades.map(t => { equity += t.pnl; return equity; });
        return { total: trades.length, wins: wins.length, losses: losses.length, totalPnl, avgWin, avgLoss, maxWin, maxLoss, wr, rr, equityCurve };
    }, [filteredTrades]);

    const strategies = useMemo(() => {
        const s = new Set(analyticsData.map(t => t.strategy));
        return Array.from(s).sort();
    }, [analyticsData]);

    const strategyComparison = useMemo(() => {
        if (!analyticsData.length) return [];
        const groups: Record<string, any[]> = {};
        analyticsData.forEach(t => {
            const s = t.strategy || 'unknown';
            if (!groups[s]) groups[s] = [];
            groups[s].push(t);
        });
        return Object.entries(groups).map(([name, trades]) => {
            const wins = trades.filter(t => t.pnl > 0);
            const losses = trades.filter(t => t.pnl <= 0);
            const totalPnl = trades.reduce((s, t) => s + t.pnl, 0);
            const avgWin = wins.length ? wins.reduce((s, t) => s + t.pnl, 0) / wins.length : 0;
            const avgLoss = losses.length ? losses.reduce((s, t) => s + t.pnl, 0) / losses.length : 0;
            const wr = trades.length ? (wins.length / trades.length * 100) : 0;
            const rr = avgLoss !== 0 ? Math.abs(avgWin / avgLoss) : 0;
            const maxWin = wins.length ? Math.max(...wins.map(t => t.pnl)) : 0;
            const maxLoss = losses.length ? Math.min(...losses.map(t => t.pnl)) : 0;
            return { name, total: trades.length, wins: wins.length, losses: losses.length, totalPnl, avgWin, avgLoss, wr, rr, maxWin, maxLoss };
        }).sort((a, b) => b.totalPnl - a.totalPnl);
    }, [analyticsData]);

    if (!isMounted) return null;
    if (loading) return <div style={{ padding: 24, color: 'var(--text-muted)' }}>Loading AI Auto-Trader...</div>;

    const isRunning = statusData?.status === "Running";
    const balance = statusData?.balance || 10000;
    const trades = statusData?.open_trades || [];
    const closedTrades = statusData?.closed_trades || [];
    const logs = statusData?.recent_logs || [];

    const tabBtn = (tab: 'trading' | 'analytics', label: string) => (
        <button onClick={() => setActiveTab(tab)} style={{
            padding: '8px 20px', borderRadius: '8px', border: 'none', fontSize: '13px', fontWeight: 600, cursor: 'pointer',
            background: activeTab === tab ? 'rgba(79, 158, 255, 0.15)' : 'transparent',
            color: activeTab === tab ? 'var(--accent-blue)' : 'var(--text-muted)',
        }}>{label}</button>
    );

    const sectionTitle = (text: string, color: string) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
            <div style={{ width: 3, height: 14, borderRadius: 2, background: color }} />
            <span style={{ fontSize: '11px', fontWeight: 700, color, textTransform: 'uppercase', letterSpacing: '0.5px' }}>{text}</span>
        </div>
    );

    return (
        <div style={{ padding: '24px', height: '100%', overflow: 'auto' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px' }}>
                <div>
                    <h1 style={{ fontSize: '24px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>Live Auto-Trader</h1>
                    <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>BB MR + ICT/SMC + Trend v4.4 (Paper Trading)</p>
                </div>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                    <div style={{
                        display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 14px', borderRadius: '8px',
                        background: isRunning ? 'rgba(0, 216, 168, 0.06)' : 'rgba(244, 63, 94, 0.06)',
                        border: isRunning ? '1px solid rgba(0, 216, 168, 0.15)' : '1px solid rgba(244, 63, 94, 0.15)',
                    }}>
                        <span className={`status-dot ${isRunning ? 'status-live' : ''}`} style={{ backgroundColor: isRunning ? 'var(--accent-green)' : 'var(--accent-red)' }}></span>
                        <span style={{ fontSize: '12px', color: isRunning ? 'var(--accent-green)' : 'var(--accent-red)', fontWeight: 600 }}>
                            {isRunning ? "Active" : "Stopped"}
                        </span>
                    </div>
                    <button onClick={() => handleAction(isRunning ? 'stop' : 'start')} disabled={actionLoading} style={{
                        padding: '8px 16px', borderRadius: '8px',
                        border: `1px solid ${isRunning ? 'rgba(244, 63, 94, 0.3)' : 'rgba(0, 216, 168, 0.3)'}`,
                        background: isRunning ? 'rgba(244, 63, 94, 0.08)' : 'rgba(0, 216, 168, 0.08)',
                        color: isRunning ? 'var(--accent-red)' : 'var(--accent-green)',
                        fontSize: '12px', fontWeight: 600, cursor: 'pointer',
                    }}>{actionLoading ? '...' : isRunning ? 'Stop' : 'Start'}</button>
                </div>
            </div>

            {/* Tab Navigation */}
            <div style={{ display: 'flex', gap: '4px', marginBottom: '20px', background: 'rgba(255,255,255,0.03)', borderRadius: '10px', padding: '4px', width: 'fit-content' }}>
                {tabBtn('trading', 'Trading')}
                {tabBtn('analytics', 'Analytics & Reports')}
            </div>

            {/* ═══════════ TRADING TAB ═══════════ */}
            {activeTab === 'trading' && (<>
                {/* Global + Save/Reset Row */}
                <div className="glass" style={{ padding: '16px 20px', borderRadius: '12px', marginBottom: '16px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                        <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>Global Settings</span>
                        <div style={{ display: 'flex', gap: 8 }}>
                            <button onClick={() => setShowResetConfirm(true)} style={{
                                padding: '5px 12px', borderRadius: '6px', border: '1px solid rgba(244, 63, 94, 0.3)',
                                background: 'rgba(244, 63, 94, 0.08)', color: 'var(--accent-red)', fontSize: '11px', fontWeight: 600, cursor: 'pointer',
                            }}>System Reset</button>
                            <button onClick={handleSaveAllSettings} disabled={savingSettings} style={{
                                padding: '5px 16px', borderRadius: '6px',
                                border: '1px solid rgba(0, 216, 168, 0.3)', background: 'rgba(0, 216, 168, 0.1)',
                                color: 'var(--accent-green)', fontSize: '11px', fontWeight: 600,
                                cursor: savingSettings ? 'not-allowed' : 'pointer', opacity: savingSettings ? 0.6 : 1,
                            }}>{savingSettings ? 'Saving...' : 'Save All Settings'}</button>
                        </div>
                    </div>
                    <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap', alignItems: 'flex-end' }}>
                        <SettingInput label="Max Trades" value={formMaxTrades} onChange={setFormMaxTrades} step={1} min={1} max={30} />
                        <SettingInput label="Balance" value={formBalance} onChange={setFormBalance} step={100} min={100} max={1000000} unit="$" />
                        <SettingInput label="Leverage" value={formLeverage} onChange={setFormLeverage} step={1} min={1} max={50} unit="x" />
                        <SettingInput label="Trade Size" value={formMaxNotional} onChange={setFormMaxNotional} step={50} min={10} max={5000} unit="$" />
                        <SettingInput label="Max Loss" value={formMaxLoss} onChange={setFormMaxLoss} step={0.5} min={0.5} max={100} unit="$" />
                    </div>
                    <div style={{ marginTop: 14, display: 'flex', gap: 20, flexWrap: 'wrap', alignItems: 'center' }}>
                        <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: 700 }}>Strategies</span>
                        <ToggleSwitch label="BB MR v7.1" enabled={bbEnabled} onToggle={() => setBbEnabled(!bbEnabled)} color="#4f9eff" />
                        <ToggleSwitch label="ICT/SMC v2" enabled={ictEnabled} onToggle={() => setIctEnabled(!ictEnabled)} color="#c084fc" />
                        <ToggleSwitch label="Trend v4.4" enabled={trendEnabled} onToggle={() => setTrendEnabled(!trendEnabled)} color="#f59e0b" />
                    </div>
                </div>

                {/* Live Stats Row */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '20px' }}>
                    <div className="glass" style={{ padding: '14px 16px', borderRadius: '12px' }}>
                        <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--accent-blue)', textTransform: 'uppercase', marginBottom: '4px' }}>Balance</div>
                        <div style={{ fontSize: '22px', fontWeight: 700, color: 'var(--accent-blue)' }}>${balance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
                    </div>
                    <div className="glass" style={{ padding: '14px 16px', borderRadius: '12px' }}>
                        <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>Open</div>
                        <div style={{ fontSize: '22px', fontWeight: 700, color: 'var(--text-primary)' }}>{trades.length}</div>
                    </div>
                    <div className="glass" style={{ padding: '14px 16px', borderRadius: '12px' }}>
                        <div style={{ fontSize: '11px', fontWeight: 700, color: v3Stats?.win_rate >= 50 ? 'var(--accent-green)' : 'var(--accent-red)', textTransform: 'uppercase', marginBottom: '4px' }}>Win Rate</div>
                        <div style={{ fontSize: '22px', fontWeight: 700, color: v3Stats?.win_rate >= 50 ? 'var(--accent-green)' : 'var(--accent-red)' }}>{v3Stats?.win_rate || 0}%</div>
                    </div>
                    <div className="glass" style={{ padding: '14px 16px', borderRadius: '12px' }}>
                        <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>Total PnL</div>
                        <div style={{ fontSize: '22px', fontWeight: 700, color: (v3Stats?.total_pnl || 0) >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                            {(v3Stats?.total_pnl || 0) >= 0 ? '+' : ''}${(v3Stats?.total_pnl || 0).toFixed(2)}
                        </div>
                    </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '20px' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                        {/* Active Trades */}
                        <div className="glass" style={{ borderRadius: '14px', overflow: 'hidden' }}>
                            <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--border)' }}>
                                <h2 style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>Active Trades</h2>
                            </div>
                            <div style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                                {trades.length === 0 ? (
                                    <div style={{ padding: '30px 0', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>Scanning for opportunities...</div>
                                ) : trades.map((trade: any) => (
                                    <div key={`ot-${trade.id}`} className="glass glass-hover" style={{ borderRadius: '10px', padding: '12px 14px', border: '1px solid var(--border)', cursor: 'zoom-in' }} onClick={() => setSelectedTrade(trade)}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                            <div>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px', flexWrap: 'wrap' }}>
                                                    <span style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>{trade.pair}</span>
                                                    <span style={{ padding: '1px 6px', borderRadius: '4px', fontSize: '10px', fontWeight: 700, background: trade.side === 'LONG' ? 'rgba(0,216,168,0.12)' : 'rgba(244,63,94,0.12)', color: trade.side === 'LONG' ? 'var(--accent-green)' : 'var(--accent-red)' }}>{trade.side}</span>
                                                    {trade.strategy && trade.strategy !== 'unknown' && <span style={{ padding: '1px 6px', borderRadius: '4px', fontSize: '9px', fontWeight: 600, background: 'rgba(255,153,0,0.1)', color: 'rgb(255,174,52)' }}>{trade.strategy}</span>}
                                                </div>
                                                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Entry: ${trade.entry} · {trade.entry_time}</div>
                                                <div style={{ display: 'flex', alignItems: 'center', marginTop: 6 }}>
                                                    <div style={{ fontSize: '9px', color: 'var(--text-muted)', background: 'rgba(255,255,255,0.05)', padding: '2px 6px', borderRadius: '4px' }}>Score: {trade.soft_score}/5</div>
                                                    <PnLChart data={trade.pnl_history} />
                                                </div>
                                            </div>
                                            <div style={{ textAlign: 'right' }}>
                                                <div style={{ fontSize: '15px', fontWeight: 700, color: trade.pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                                    {trade.pnl >= 0 ? '+' : ''}${trade.pnl?.toFixed(2) || '0.00'}
                                                    <span style={{ fontSize: '11px', marginLeft: 4 }}>({trade.pnl_pct >= 0 ? '+' : ''}{trade.pnl_pct?.toFixed(2) || '0'}%)</span>
                                                </div>
                                                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Margin: ${trade.margin?.toFixed(2)}</div>
                                                <button onClick={(e) => { e.stopPropagation(); handleCloseTrade(trade.id); }} disabled={closingTradeId === trade.id}
                                                    style={{ marginTop: '6px', padding: '3px 10px', borderRadius: '6px', background: 'rgba(244,63,94,0.1)', border: '1px solid rgba(244,63,94,0.3)', color: 'var(--accent-red)', fontSize: '10px', fontWeight: 600, cursor: 'pointer', opacity: closingTradeId === trade.id ? 0.5 : 1 }}>
                                                    {closingTradeId === trade.id ? '...' : 'Close'}
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Recent Closed Trades */}
                        <div className="glass" style={{ borderRadius: '14px', overflow: 'hidden' }}>
                            <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--border)' }}>
                                <h2 style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>Recent Trades</h2>
                            </div>
                            <div style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '350px', overflowY: 'auto' }}>
                                {closedTrades.length === 0 ? (
                                    <div style={{ padding: '30px 0', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>No trades yet.</div>
                                ) : closedTrades.map((t: any) => (
                                    <div key={`ct-${t.id}`} style={{ padding: '10px', borderRadius: '8px', backgroundColor: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }} onClick={() => setSelectedTrade(t)}>
                                        <div>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
                                                <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>{t.pair}</span>
                                                <span style={{ padding: '1px 5px', borderRadius: '4px', fontSize: '9px', fontWeight: 700, background: t.side === 'LONG' ? 'rgba(0,216,168,0.12)' : 'rgba(244,63,94,0.12)', color: t.side === 'LONG' ? 'var(--accent-green)' : 'var(--accent-red)' }}>{t.side}</span>
                                                {t.strategy && t.strategy !== 'unknown' && <span style={{ padding: '1px 5px', borderRadius: '4px', fontSize: '8px', fontWeight: 600, background: 'rgba(255,153,0,0.1)', color: 'rgb(255,174,52)' }}>{t.strategy}</span>}
                                                <span style={{ fontSize: '9px', color: 'var(--text-muted)' }}>{t.reason}</span>
                                            </div>
                                            <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{t.entry_time} → {t.exit_time}</div>
                                        </div>
                                        <div style={{ textAlign: 'right' }}>
                                            <div style={{ fontSize: '13px', fontWeight: 700, color: t.pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                                {t.pnl >= 0 ? '+' : ''}${t.pnl?.toFixed(2)} <span style={{ fontSize: '10px' }}>({t.pnl_pct >= 0 ? '+' : ''}{t.pnl_pct?.toFixed(2)}%)</span>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Logs */}
                    <div className="glass" style={{ borderRadius: '14px', overflow: 'hidden' }}>
                        <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>System Logs</span>
                            <span className="ai-badge">Live</span>
                        </div>
                        <div style={{ padding: '10px', display: 'flex', flexDirection: 'column', gap: '6px', height: '500px', overflowY: 'auto' }}>
                            {logs.length === 0 ? (
                                <div style={{ color: 'var(--text-muted)', fontSize: '12px', textAlign: 'center', marginTop: 20 }}>No logs yet.</div>
                            ) : logs.map((logObj: any, idx: number) => {
                                let log = typeof logObj === 'string' ? logObj : (logObj.text || '');
                                let color = 'var(--text-secondary)';
                                if (log.includes("[GRN]") || log.includes("[OK]")) color = 'var(--accent-green)';
                                if (log.includes("[PERF]") || log.includes("[SYNC]") || log.includes("[GEAR]")) color = 'var(--accent-blue)';
                                if (log.includes("[FAIL]") || log.includes("[STOP]") || log.includes("[CAP]")) color = 'var(--accent-red)';
                                if (log.includes("[WARN]") || log.includes("Error")) color = 'var(--accent-red)';
                                if (log.includes("[LOCK]") || log.includes("[TRAIL]")) color = '#ffae34';
                                if (log.includes("[SEARCH]")) color = 'var(--text-muted)';
                                if (log.includes("[RESET]")) color = '#c084fc';
                                const logKey = typeof logObj === 'string' ? `ls-${idx}` : `lo-${logObj.id}`;
                                return (
                                    <div key={logKey} style={{ padding: '8px', borderRadius: '6px', backgroundColor: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)', fontSize: '10px', color, lineHeight: '1.4', fontFamily: 'monospace' }}>{log}</div>
                                );
                            })}
                        </div>
                    </div>
                </div>
            </>)}

            {/* ═══════════ ANALYTICS TAB ═══════════ */}
            {activeTab === 'analytics' && (<>
                {/* Filters */}
                <div className="glass" style={{ padding: '14px 20px', borderRadius: '12px', marginBottom: '20px' }}>
                    <div style={{ display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap' }}>
                        <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>Filters:</span>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                            <label style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Strategy</label>
                            <select value={filterStrategy} onChange={e => setFilterStrategy(e.target.value)} style={{ padding: '5px 8px', borderRadius: '6px', border: '1px solid var(--border)', background: 'rgba(0,0,0,0.3)', color: 'var(--text-primary)', fontSize: '12px' }}>
                                <option value="all">All Strategies</option>
                                {strategies.map(s => <option key={s} value={s}>{s}</option>)}
                            </select>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                            <label style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Direction</label>
                            <select value={filterSide} onChange={e => setFilterSide(e.target.value)} style={{ padding: '5px 8px', borderRadius: '6px', border: '1px solid var(--border)', background: 'rgba(0,0,0,0.3)', color: 'var(--text-primary)', fontSize: '12px' }}>
                                <option value="all">All</option>
                                <option value="LONG">LONG</option>
                                <option value="SHORT">SHORT</option>
                            </select>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                            <label style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Result</label>
                            <select value={filterResult} onChange={e => setFilterResult(e.target.value)} style={{ padding: '5px 8px', borderRadius: '6px', border: '1px solid var(--border)', background: 'rgba(0,0,0,0.3)', color: 'var(--text-primary)', fontSize: '12px' }}>
                                <option value="all">All</option>
                                <option value="win">Wins</option>
                                <option value="loss">Losses</option>
                            </select>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                            <label style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Start Date</label>
                            <input type="date" value={filterStartDate} onChange={e => setFilterStartDate(e.target.value)} style={{ padding: '4px 8px', borderRadius: '6px', border: '1px solid var(--border)', background: 'rgba(0,0,0,0.3)', color: 'var(--text-primary)', fontSize: '12px', height: '28px' }} />
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                            <label style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>End Date</label>
                            <input type="date" value={filterEndDate} onChange={e => setFilterEndDate(e.target.value)} style={{ padding: '4px 8px', borderRadius: '6px', border: '1px solid var(--border)', background: 'rgba(0,0,0,0.3)', color: 'var(--text-primary)', fontSize: '12px', height: '28px' }} />
                        </div>
                        <button onClick={fetchAnalytics} style={{ padding: '5px 14px', borderRadius: '6px', border: '1px solid rgba(79,158,255,0.3)', background: 'rgba(79,158,255,0.1)', color: 'var(--accent-blue)', fontSize: '11px', fontWeight: 600, cursor: 'pointer', marginTop: 12 }}>Refresh</button>
                    </div>
                </div>

                {/* Strategy Comparison */}
                {strategyComparison.length > 1 && (
                    <div className="glass" style={{ padding: '16px 20px', borderRadius: '12px', marginBottom: '20px' }}>
                        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: 12 }}>Strategy Comparison</div>
                        <div style={{ display: 'grid', gridTemplateColumns: `repeat(${strategyComparison.length}, 1fr)`, gap: '12px' }}>
                            {strategyComparison.map((s, idx) => {
                                const colors: Record<string, string> = { 'bb_mr_v5.1': '#4f9eff', 'ict_smc_v1': '#c084fc', 'ict_smc_v2': '#c084fc' };
                                const clr = colors[s.name] || '#ffae34';
                                return (
                                    <div key={`cmp-${idx}`} style={{ padding: '14px', borderRadius: '10px', background: 'rgba(255,255,255,0.02)', border: `1px solid ${clr}33` }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10 }}>
                                            <div style={{ width: 3, height: 12, borderRadius: 2, background: clr }} />
                                            <span style={{ fontSize: '12px', fontWeight: 700, color: clr }}>{s.name}</span>
                                        </div>
                                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
                                            {[
                                                { l: 'Trades', v: `${s.total}`, c: 'var(--text-primary)' },
                                                { l: 'Win Rate', v: `${s.wr.toFixed(1)}%`, c: s.wr >= 50 ? '#00d8a8' : '#f43f5e' },
                                                { l: 'Total PnL', v: `${s.totalPnl >= 0 ? '+' : ''}$${s.totalPnl.toFixed(2)}`, c: s.totalPnl >= 0 ? '#00d8a8' : '#f43f5e' },
                                                { l: 'R:R', v: `${s.rr.toFixed(2)}:1`, c: s.rr >= 1 ? '#00d8a8' : '#ffae34' },
                                                { l: 'Avg Win', v: `+$${s.avgWin.toFixed(2)}`, c: '#00d8a8' },
                                                { l: 'Avg Loss', v: `$${s.avgLoss.toFixed(2)}`, c: '#f43f5e' },
                                                { l: 'Best', v: `+$${s.maxWin.toFixed(2)}`, c: '#00d8a8' },
                                                { l: 'Worst', v: `$${s.maxLoss.toFixed(2)}`, c: '#f43f5e' },
                                            ].map((item, j) => (
                                                <div key={`ci-${j}`} style={{ padding: '4px 0' }}>
                                                    <div style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>{item.l}</div>
                                                    <div style={{ fontSize: '13px', fontWeight: 700, color: item.c }}>{item.v}</div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}

                {/* Analytics Summary Cards */}
                {analyticsStats && (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '10px', marginBottom: '20px' }}>
                        {[
                            { label: 'Total Trades', value: analyticsStats.total, color: 'var(--accent-blue)' },
                            { label: 'Win Rate', value: `${analyticsStats.wr.toFixed(1)}%`, color: analyticsStats.wr >= 50 ? 'var(--accent-green)' : 'var(--accent-red)' },
                            { label: 'Total PnL', value: `${analyticsStats.totalPnl >= 0 ? '+' : ''}$${analyticsStats.totalPnl.toFixed(2)}`, color: analyticsStats.totalPnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' },
                            { label: 'R:R Ratio', value: `${analyticsStats.rr.toFixed(2)}:1`, color: analyticsStats.rr >= 1 ? 'var(--accent-green)' : '#ffae34' },
                            { label: 'Avg Win', value: `+$${analyticsStats.avgWin.toFixed(2)}`, color: 'var(--accent-green)' },
                            { label: 'Avg Loss', value: `$${analyticsStats.avgLoss.toFixed(2)}`, color: 'var(--accent-red)' },
                        ].map((c, i) => (
                            <div key={`sc-${i}`} className="glass" style={{ padding: '12px', borderRadius: '10px', textAlign: 'center' }}>
                                <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>{c.label}</div>
                                <div style={{ fontSize: '18px', fontWeight: 800, color: c.color }}>{c.value}</div>
                            </div>
                        ))}
                    </div>
                )}

                {/* Equity Curve */}
                {analyticsStats && analyticsStats.equityCurve.length > 1 && (
                    <div className="glass" style={{ padding: '16px 20px', borderRadius: '12px', marginBottom: '20px' }}>
                        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: 10 }}>Equity Curve ({filteredTrades.length} trades)</div>
                        <svg width="100%" height="120" viewBox={`0 0 800 120`} preserveAspectRatio="none">
                            {(() => {
                                const ec = analyticsStats.equityCurve;
                                const mn = Math.min(...ec, 0); const mx = Math.max(...ec, 0); const rng = mx - mn || 1;
                                const pts = ec.map((v: number, i: number) => {
                                    const x = (i / (ec.length - 1)) * 800;
                                    const y = 110 - ((v - mn) / rng) * 100;
                                    return `${x.toFixed(1)},${y.toFixed(1)}`;
                                }).join(' ');
                                const last = ec[ec.length - 1];
                                const clr = last >= 0 ? '#00d8a8' : '#f43f5e';
                                const zeroY = 110 - ((0 - mn) / rng) * 100;
                                return (<>
                                    <line x1="0" y1={zeroY} x2="800" y2={zeroY} stroke="rgba(255,255,255,0.1)" strokeDasharray="4" />
                                    <polyline fill="none" stroke={clr} strokeWidth="2" strokeLinecap="round" points={pts} />
                                </>);
                            })()}
                        </svg>
                    </div>
                )}

                {/* Detailed Trade List */}
                <div className="glass" style={{ borderRadius: '14px', overflow: 'hidden' }}>
                    <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between' }}>
                        <h2 style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>All Trades ({filteredTrades.length})</h2>
                        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                                W: {analyticsStats?.wins || 0} / L: {analyticsStats?.losses || 0} |
                                Best: ${analyticsStats?.maxWin?.toFixed(2) || '0'} |
                                Worst: ${analyticsStats?.maxLoss?.toFixed(2) || '0'}
                            </span>
                        </div>
                    </div>
                    <div style={{ maxHeight: '500px', overflowY: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
                            <thead>
                                <tr style={{ background: 'rgba(255,255,255,0.03)', position: 'sticky', top: 0 }}>
                                    {['#', 'Symbol', 'Side', 'Strategy', 'Notional', 'PnL', 'PnL%', 'MaxPnl%', 'Reason', 'Entry', 'Exit'].map(h => (
                                        <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: 'var(--text-muted)', fontWeight: 600, fontSize: '10px', textTransform: 'uppercase', borderBottom: '1px solid var(--border)' }}>{h}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {filteredTrades.map((t, i) => (
                                    <tr key={`at-${t.id}-${i}`} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                                        <td style={{ padding: '6px', color: 'var(--text-muted)' }}>{i + 1}</td>
                                        <td style={{ padding: '6px', fontWeight: 600, color: 'var(--text-primary)' }}>{t.symbol}</td>
                                        <td style={{ padding: '6px' }}><span style={{ padding: '1px 5px', borderRadius: '3px', fontSize: '9px', fontWeight: 700, background: t.side === 'LONG' ? 'rgba(0,216,168,0.12)' : 'rgba(244,63,94,0.12)', color: t.side === 'LONG' ? '#00d8a8' : '#f43f5e' }}>{t.side}</span></td>
                                        <td style={{ padding: '6px', color: 'var(--text-muted)' }}>{t.strategy}</td>
                                        <td style={{ padding: '6px', color: 'var(--text-muted)' }}>${t.notional?.toFixed(0)}</td>
                                        <td style={{ padding: '6px', fontWeight: 700, color: t.pnl >= 0 ? '#00d8a8' : '#f43f5e' }}>{t.pnl >= 0 ? '+' : ''}${t.pnl?.toFixed(2)}</td>
                                        <td style={{ padding: '6px', color: t.pnl_pct >= 0 ? '#00d8a8' : '#f43f5e' }}>{t.pnl_pct >= 0 ? '+' : ''}{t.pnl_pct?.toFixed(2)}%</td>
                                        <td style={{ padding: '6px', color: '#ffae34' }}>{t.max_pnl_pct?.toFixed(1)}%</td>
                                        <td style={{ padding: '6px', color: 'var(--text-muted)', fontSize: '10px' }}>{t.reason}</td>
                                        <td style={{ padding: '6px', color: 'var(--text-muted)', fontSize: '10px' }}>{t.entry_time}</td>
                                        <td style={{ padding: '6px', color: 'var(--text-muted)', fontSize: '10px' }}>{t.exit_time}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        {filteredTrades.length === 0 && (
                            <div style={{ padding: '30px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>No trades match the selected filters.</div>
                        )}
                    </div>
                </div>
            </>)}

            {/* ═══════════ RESET CONFIRMATION MODAL ═══════════ */}
            {showResetConfirm && (
                <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(8px)', zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center' }} onClick={() => setShowResetConfirm(false)}>
                    <div className="glass" style={{ width: 400, borderRadius: '16px', padding: '28px', border: '1px solid rgba(244,63,94,0.3)' }} onClick={e => e.stopPropagation()}>
                        <h3 style={{ fontSize: '18px', fontWeight: 700, color: 'var(--accent-red)', marginBottom: 12 }}>System Reset</h3>
                        <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: 16 }}>
                            All open trades, trade history, and logs will be permanently deleted. The balance will be reset.
                        </p>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 20 }}>
                            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>New Balance:</span>
                            <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>$</span>
                            <input type="number" value={resetBalance} onChange={e => setResetBalance(parseFloat(e.target.value) || 10000)} style={{ width: 100, padding: '6px 10px', borderRadius: '6px', border: '1px solid var(--border)', background: 'rgba(0,0,0,0.3)', color: 'var(--text-primary)', fontSize: '14px', textAlign: 'center' }} />
                        </div>
                        <div style={{ display: 'flex', gap: 10 }}>
                            <button onClick={() => setShowResetConfirm(false)} style={{ flex: 1, padding: '8px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-muted)', fontSize: '12px', fontWeight: 600, cursor: 'pointer' }}>Cancel</button>
                            <button onClick={handleReset} style={{ flex: 1, padding: '8px', borderRadius: '8px', border: '1px solid rgba(244,63,94,0.5)', background: 'rgba(244,63,94,0.15)', color: 'var(--accent-red)', fontSize: '12px', fontWeight: 700, cursor: 'pointer' }}>Confirm Reset</button>
                        </div>
                    </div>
                </div>
            )}

            {/* ═══════════ TRADE DETAIL MODAL ═══════════ */}
            {selectedTrade && (
                <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(8px)', zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px' }} onClick={() => setSelectedTrade(null)}>
                    <div className="glass" style={{ width: '100%', maxWidth: '700px', borderRadius: '20px', padding: '30px', border: '1px solid rgba(255,255,255,0.1)', position: 'relative' }} onClick={e => e.stopPropagation()}>
                        <button onClick={() => setSelectedTrade(null)} style={{ position: 'absolute', top: 20, right: 20, background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '20px' }}>X</button>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px' }}>
                            <div>
                                <h2 style={{ fontSize: '22px', fontWeight: 800, color: 'var(--text-primary)' }}>
                                    {selectedTrade.pair} <span style={{ color: selectedTrade.side === 'LONG' ? 'var(--accent-green)' : 'var(--accent-red)', fontSize: '14px' }}>{selectedTrade.side}</span>
                                </h2>
                                <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>ID: {selectedTrade.id} · {selectedTrade.strategy} · {selectedTrade.regime?.replace('_', ' ')}</p>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: 4 }}>{selectedTrade.entry_time} {selectedTrade.exit_time && `→ ${selectedTrade.exit_time}`}</div>
                            </div>
                            <div style={{ textAlign: 'right' }}>
                                <div style={{ fontSize: '24px', fontWeight: 800, color: selectedTrade.pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                    {selectedTrade.pnl >= 0 ? '+' : '-'}${Math.abs(selectedTrade.pnl).toFixed(2)}
                                </div>
                                <div style={{ fontSize: '14px', color: 'var(--text-muted)' }}>{selectedTrade.pnl_pct?.toFixed(2)}% ROI</div>
                                {selectedTrade.exit_time && selectedTrade.reason && (
                                    <div style={{ fontSize: '11px', fontWeight: 700, marginTop: 4, padding: '2px 8px', borderRadius: '4px', background: selectedTrade.pnl > 0 ? 'rgba(0,216,168,0.1)' : 'rgba(244,63,94,0.1)', color: selectedTrade.pnl > 0 ? 'var(--accent-green)' : 'var(--accent-red)', display: 'inline-block' }}>{selectedTrade.reason}</div>
                                )}
                            </div>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '25px' }}>
                            <div className="glass" style={{ padding: '15px', borderRadius: '12px', background: 'rgba(255,255,255,0.02)' }}>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '5px' }}>ENTRY</div>
                                <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>Score: <span style={{ color: 'var(--accent-blue)' }}>{selectedTrade.soft_score}/5</span></div>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: 4 }}>Price: ${selectedTrade.entry}</div>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: 2 }}>Margin: ${selectedTrade.margin?.toFixed(2)}</div>
                            </div>
                            <div className="glass" style={{ padding: '15px', borderRadius: '12px', background: 'rgba(255,255,255,0.02)' }}>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '5px' }}>PERFORMANCE</div>
                                <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--accent-green)' }}>Max: +{selectedTrade.max_pnl_pct?.toFixed(2)}%</div>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: 4 }}>{selectedTrade.exit ? `Exit: $${selectedTrade.exit}` : `Current: $${selectedTrade.current_price}`}</div>
                            </div>
                        </div>
                        <DetailedPnLChart data={selectedTrade.pnl_history} pair={selectedTrade.pair} />
                        <div style={{ marginTop: '20px', padding: '12px', background: 'rgba(255,255,255,0.03)', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.05)', display: 'flex', gap: '20px', fontSize: '12px', color: 'var(--text-primary)' }}>
                            <span>SL: ${selectedTrade.sl_price?.toFixed(6) || '--'}</span>
                            <span>TP: ${selectedTrade.tp_price?.toFixed(6) || '--'}</span>
                            {!selectedTrade.exit_time && selectedTrade.pm_stage >= 2 && <span style={{ color: 'var(--accent-green)' }}>Trailing Active</span>}
                            {!selectedTrade.exit_time && selectedTrade.pm_stage === 1 && <span style={{ color: 'var(--accent-blue)' }}>Breakeven Active</span>}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
