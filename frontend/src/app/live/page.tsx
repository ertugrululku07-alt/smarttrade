"use client"

import React, { useState, useEffect } from 'react';
import { getApiUrl } from '@/utils/api';

// --- Premium animated PnL Chart Component ---
const PnLChart = ({ data }: { data: any[] }) => {
    if (!data || data.length < 2) return null;

    const values = data.map(d => d.pct);
    const min = Math.min(...values, -0.5); // At least show some range
    const max = Math.max(...values, 0.5);
    const range = max - min;
    
    // SVG Coordinates
    const width = 120;
    const height = 40;
    const padding = 2;
    
    const points = data.map((d, i) => {
        const x = (i / (data.length - 1)) * width;
        const y = height - ((d.pct - min) / range) * (height - padding * 2) - padding;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');

    const isProfit = data[data.length - 1].pct >= 0;
    const color = isProfit ? '#00d8a8' : '#f43f5e';
    const gradientId = `grad-${Math.random().toString(36).substr(2, 9)}`;

    return (
        <div style={{ position: 'relative', width: width, height: height, marginLeft: 12 }}>
            <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
                <defs>
                    <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor={color} stopOpacity="0.2" />
                        <stop offset="100%" stopColor={color} stopOpacity="0" />
                    </linearGradient>
                </defs>
                {/* Area under curve */}
                <path
                    d={`M 0,${height} L ${points} L ${width},${height} Z`}
                    fill={`url(#${gradientId})`}
                    style={{ transition: 'all 0.3s ease' }}
                />
                {/* Main line */}
                <polyline
                    fill="none"
                    stroke={color}
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    points={points}
                    style={{ transition: 'all 0.3s ease' }}
                />
            </svg>
        </div>
    );
};

// --- Detailed Premium Modal Chart ---
const DetailedPnLChart = ({ data, pair }: { data: any[], pair: string }) => {
    if (!data || data.length < 2) return (
        <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
            Not enough data points yet...
        </div>
    );

    const values = data.map(d => d.pct);
    const min = Math.min(...values, -0.2);
    const max = Math.max(...values, 0.2);
    const range = max - min || 1;
    
    const width = 600;
    const height = 240;
    const padding = 30;
    
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
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Performance Timeline ({pair})</span>
                <span style={{ fontSize: '12px', fontWeight: 700, color: color }}>
                    Current: {data[data.length - 1].pct.toFixed(2)}%
                </span>
            </div>
            <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} style={{ overflow: 'visible' }}>
                {/* Horizontal Grid lines */}
                {[min, 0, max].map((v, i) => {
                    const y = (height - padding) - ((v - min) / range) * (height - padding * 2);
                    return (
                        <g key={`grid-${i}`}>
                            <line x1={padding} y1={y} x2={width - padding} y2={y} stroke="rgba(255,255,255,0.05)" strokeDasharray="4" />
                            <text x={padding - 5} y={y + 4} textAnchor="end" fill="var(--text-muted)" style={{ fontSize: '10px' }}>{v.toFixed(1)}%</text>
                        </g>
                    )
                })}
                
                {/* Path */}
                <path d={`M ${padding},${height - padding} L ${points} L ${width - padding},${height - padding} Z`} fill={`url(#detailed-grad)`} opacity="0.1" />
                <polyline fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" points={points} />
                
                {/* Time labels (start/end) */}
                <text x={padding} y={height - 10} fill="var(--text-muted)" style={{ fontSize: '10px' }}>{data[0].t}</text>
                <text x={width - padding} y={height - 10} textAnchor="end" fill="var(--text-muted)" style={{ fontSize: '10px' }}>{data[data.length - 1].t}</text>
                
                <defs>
                    <linearGradient id="detailed-grad" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor={color} />
                        <stop offset="100%" stopColor="transparent" />
                    </linearGradient>
                </defs>
            </svg>
        </div>
    );
};

export default function LiveTradingPage() {
    const [statusData, setStatusData] = useState<any>(null);
    const [selectedTrade, setSelectedTrade] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [actionLoading, setActionLoading] = useState(false);
    const [closingTradeId, setClosingTradeId] = useState<string | null>(null);
    const [maxOpenTrades, setMaxOpenTrades] = useState<number | null>(null);
    const [savingSettings, setSavingSettings] = useState(false);
    const [v3Stats, setV3Stats] = useState<any>(null);

    const [isMounted, setIsMounted] = useState(false);
    useEffect(() => { setIsMounted(true); }, []);

    const fetchStatus = async () => {
        try {
            const res = await fetch(getApiUrl("/live/quant/status"));
            if (res.ok) {
                const data = await res.json();
                setStatusData(data);
                if (maxOpenTrades === null && data.max_open_trades_limit !== undefined) {
                    setMaxOpenTrades(data.max_open_trades_limit);
                }
            }

            // Fetch V3 Stats
            const statsRes = await fetch(getApiUrl("/live/v3/stats"));
            if (statsRes.ok) {
                setV3Stats(await statsRes.json());
            }
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchStatus();
        const interval = setInterval(fetchStatus, 3000); // Her 3 saniyede bir guncelle
        return () => clearInterval(interval);
    }, []);

    const handleAction = async (action: 'start' | 'stop') => {
        setActionLoading(true);
        try {
            const res = await fetch(getApiUrl(`/live/quant/${action}`), { method: 'POST' });
            if (res.ok) {
                await fetchStatus();
            }
        } catch (e) {
            console.error(e);
        } finally {
            setActionLoading(false);
        }
    };

    const handleCloseTrade = async (tradeId: string) => {
        setClosingTradeId(tradeId);
        try {
            const res = await fetch(getApiUrl(`/live/quant/close-trade/${tradeId}`), { method: 'POST' });
            if (res.ok) {
                await fetchStatus();
            } else {
                const err = await res.json();
                alert(err.message || 'Error closing trade');
            }
        } catch (e) {
            console.error(e);
            alert('Failed to connect to backend.');
        } finally {
            setClosingTradeId(null);
        }
    };

    const handleSaveSettings = async () => {
        if (maxOpenTrades === null || maxOpenTrades < 1) return;
        setSavingSettings(true);
        try {
            const res = await fetch(getApiUrl('/live/quant/settings'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ max_open_trades: maxOpenTrades })
            });
            if (res.ok) {
                await fetchStatus();
            } else {
                alert('Hata olustu');
            }
        } catch (e) {
            console.error(e);
        } finally {
            setSavingSettings(false);
        }
    };

    if (!isMounted) return null;
    if (loading) return <div style={{ padding: 24, color: 'var(--text-muted)' }}>Loading AI Auto-Trader...</div>;

    const isRunning = statusData?.status === "Running";
    const balance = statusData?.balance || 10000;
    const trades = statusData?.open_trades || [];
    const closedTrades = statusData?.closed_trades || [];
    const logs = statusData?.recent_logs || [];

    return (
        <div style={{ padding: '24px', height: '100%', overflow: 'auto' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px' }}>
                <div>
                    <h1 style={{ fontSize: '24px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>Live Auto-Trader</h1>
                    <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Quant AI otonom market tarayici (Paper Trading)</p>
                </div>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                    <div style={{
                        display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 14px',
                        borderRadius: '8px',
                        background: isRunning ? 'rgba(0, 216, 168, 0.06)' : 'rgba(244, 63, 94, 0.06)',
                        border: isRunning ? '1px solid rgba(0, 216, 168, 0.15)' : '1px solid rgba(244, 63, 94, 0.15)',
                    }}>
                        <span className={`status-dot ${isRunning ? 'status-live' : ''}`} style={{ backgroundColor: isRunning ? 'var(--accent-green)' : 'var(--accent-red)' }}></span>
                        <span style={{ fontSize: '12px', color: isRunning ? 'var(--accent-green)' : 'var(--accent-red)', fontWeight: 600 }}>
                            {isRunning ? "Scanner Active" : "Scanner Stopped"}
                        </span>
                    </div>

                    {isRunning ? (
                        <button
                            onClick={() => handleAction('stop')} disabled={actionLoading}
                            style={{
                                padding: '8px 16px', borderRadius: '8px', border: '1px solid rgba(244, 63, 94, 0.3)',
                                background: 'rgba(244, 63, 94, 0.08)', color: 'var(--accent-red)', fontSize: '12px', fontWeight: 600, cursor: 'pointer',
                            }}>
                            {actionLoading ? 'Stopping...' : 'Stop Auto-Trader'}
                        </button>
                    ) : (
                        <button
                            onClick={() => handleAction('start')} disabled={actionLoading}
                            style={{
                                padding: '8px 16px', borderRadius: '8px', border: '1px solid rgba(0, 216, 168, 0.3)',
                                background: 'rgba(0, 216, 168, 0.08)', color: 'var(--accent-green)', fontSize: '12px', fontWeight: 600, cursor: 'pointer',
                            }}>
                            {actionLoading ? 'Starting...' : 'Start Auto-Trader'}
                        </button>
                    )}
                </div>
            </div>

            {/* Settings & Live Stats */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr repeat(3, 1fr)', gap: '12px', marginBottom: '24px' }}>
                <div className="glass" style={{ padding: '16px', borderRadius: '12px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '8px' }}>Trade Limit Config</div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                        <input
                            type="number" min="1" max="50"
                            value={maxOpenTrades || ''}
                            onChange={(e) => setMaxOpenTrades(parseInt(e.target.value) || 1)}
                            style={{
                                width: '60px', padding: '6px', borderRadius: '6px',
                                border: '1px solid var(--border)', background: 'rgba(0,0,0,0.2)',
                                color: 'var(--text-primary)', fontSize: '14px', textAlign: 'center'
                            }}
                        />
                        <button
                            onClick={handleSaveSettings}
                            disabled={savingSettings || maxOpenTrades === statusData?.max_open_trades_limit}
                            style={{
                                padding: '6px 12px', borderRadius: '6px',
                                background: (savingSettings || maxOpenTrades === statusData?.max_open_trades_limit) ? 'rgba(255,255,255,0.05)' : 'rgba(79, 158, 255, 0.1)',
                                border: '1px solid ' + ((savingSettings || maxOpenTrades === statusData?.max_open_trades_limit) ? 'rgba(255,255,255,0.1)' : 'rgba(79, 158, 255, 0.3)'),
                                color: (savingSettings || maxOpenTrades === statusData?.max_open_trades_limit) ? 'var(--text-muted)' : 'var(--accent-blue)',
                                fontSize: '12px', fontWeight: 600, cursor: (savingSettings || maxOpenTrades === statusData?.max_open_trades_limit) ? 'not-allowed' : 'pointer',
                            }}
                        >
                            {savingSettings ? '...' : 'Save'}
                        </button>
                    </div>
                </div>
                <div className="glass" style={{ padding: '16px', borderRadius: '12px' }}>
                    <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '8px' }}>Paper Balance</div>
                    <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--accent-blue)' }}>${balance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
                </div>
                <div className="glass" style={{ padding: '16px', borderRadius: '12px' }}>
                    <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '8px' }}>Open Positions</div>
                    <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--text-primary)' }}>{trades.length}</div>
                </div>
                <div className="glass" style={{ padding: '16px', borderRadius: '12px' }}>
                    <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '8px' }}>Scanned Markets</div>
                    <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--text-primary)' }}>{statusData?.scanned_markets_count || 10} <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>pairs</span></div>
                </div>
            </div>

            {/* V3.1 Performance Board */}
            {v3Stats && v3Stats.total > 0 && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '24px' }}>
                    <div className="glass" style={{ padding: '16px', borderRadius: '12px', border: '1px solid rgba(79, 158, 255, 0.2)', background: 'rgba(79, 158, 255, 0.03)' }}>
                        <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--accent-blue)', textTransform: 'uppercase', marginBottom: '4px' }}>Total History</div>
                        <div style={{ fontSize: '20px', fontWeight: 800, color: 'var(--text-primary)' }}>{v3Stats.total} <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>trades</span></div>
                    </div>
                    <div className="glass" style={{ padding: '16px', borderRadius: '12px', border: '1px solid rgba(0, 216, 168, 0.2)', background: 'rgba(0, 216, 168, 0.03)' }}>
                        <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--accent-green)', textTransform: 'uppercase', marginBottom: '4px' }}>Win Rate</div>
                        <div style={{ fontSize: '20px', fontWeight: 800, color: 'var(--accent-green)' }}>{v3Stats.win_rate}%</div>
                    </div>
                    <div className="glass" style={{ padding: '16px', borderRadius: '12px', border: '1px solid rgba(255, 174, 52, 0.2)', background: 'rgba(255, 174, 52, 0.03)' }}>
                        <div style={{ fontSize: '11px', fontWeight: 700, color: '#ffae34', textTransform: 'uppercase', marginBottom: '4px' }}>Avg Risk/Reward</div>
                        <div style={{ fontSize: '20px', fontWeight: 800, color: '#ffae34' }}>{v3Stats.avg_rr}R</div>
                    </div>
                    <div className="glass" style={{ padding: '16px', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.1)', background: 'rgba(255, 255, 255, 0.03)' }}>
                        <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>Acc. Growth</div>
                        <div style={{ fontSize: '20px', fontWeight: 800, color: v3Stats.total_pnl_pct >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                            {v3Stats.total_pnl_pct >= 0 ? '+' : ''}{v3Stats.total_pnl_pct}%
                        </div>
                    </div>
                </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 360px', gap: '20px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    {/* Open Positions */}
                    <div className="glass" style={{ borderRadius: '14px', overflow: 'hidden' }}>
                        <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)' }}>
                            <h2 style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>Active Trades</h2>
                        </div>
                        <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                            {trades.length === 0 ? (
                                <div key="empty-open" style={{ padding: '40px 0', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
                                    No open positions right now. Scanning for opportunities...
                                </div>
                            ) : (
                                <div key="list-open" className="list-container" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                    {trades.map((trade: any) => (
                                        <div 
                                            key={`open-trade-${trade.id}`} 
                                            className="glass glass-hover" 
                                            style={{ borderRadius: '10px', padding: '14px 16px', border: '1px solid var(--border)', cursor: 'zoom-in' }}
                                            onClick={() => setSelectedTrade(trade)}
                                        >
                                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                <div>
                                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px', flexWrap: 'wrap' }}>
                                                        <span style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)' }}>{trade.pair}</span>
                                                        <span style={{
                                                            padding: '1px 7px', borderRadius: '4px', fontSize: '10px', fontWeight: 700,
                                                            background: trade.side === 'LONG' ? 'rgba(0, 216, 168, 0.12)' : 'rgba(244, 63, 94, 0.12)',
                                                            color: trade.side === 'LONG' ? 'var(--accent-green)' : 'var(--accent-red)',
                                                        }}>{trade.side}</span>
                                                        {trade.regime && trade.regime !== 'unknown' && (
                                                            <span style={{
                                                                padding: '1px 7px', borderRadius: '4px', fontSize: '10px', fontWeight: 600,
                                                                background: 'rgba(79, 158, 255, 0.1)', color: 'var(--accent-blue)', whiteSpace: 'nowrap'
                                                            }}>{trade.regime.replace('_', ' ')}</span>
                                                        )}
                                                        {trade.strategy && trade.strategy !== 'unknown' && trade.strategy !== 'none' && (
                                                            <span style={{
                                                                padding: '1px 7px', borderRadius: '4px', fontSize: '10px', fontWeight: 600,
                                                                background: 'rgba(255, 153, 0, 0.1)', color: 'rgb(255, 174, 52)', whiteSpace: 'nowrap'
                                                            }}>{trade.strategy}</span>
                                                        )}
                                                    </div>
                                                    <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                                                        Entry: ${trade.entry} · Time: {trade.entry_time}
                                                    </div>
                                                    <div style={{ display: 'flex', alignItems: 'center', marginTop: 8 }}>
                                                        <div style={{ fontSize: '10px', color: 'var(--text-muted)', background: 'rgba(255,255,255,0.05)', padding: '2px 6px', borderRadius: '4px' }}>
                                                             Score: {trade.soft_score}/5 · {trade.entry_type}
                                                        </div>
                                                        <PnLChart data={trade.pnl_history} />
                                                    </div>
                                                </div>
                                                <div style={{ textAlign: 'right' }}>
                                                    <div style={{ fontSize: '15px', fontWeight: 700, color: trade.pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                                        {`${trade.pnl >= 0 ? '+' : ''}${trade.pnl !== undefined ? '$' + trade.pnl.toFixed(2) : '--'}`}
                                                        <span style={{ fontSize: '12px', marginLeft: 4 }}>
                                                            {`(${trade.pnl_pct >= 0 ? '+' : ''}${trade.pnl_pct !== undefined ? trade.pnl_pct.toFixed(2) + '%' : '--'})`}
                                                        </span>
                                                    </div>
                                                    <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                                                        Current: ${trade.current_price !== undefined ? trade.current_price.toFixed(4) : '--'} · Margin: ${trade.margin.toFixed(2)}
                                                    </div>
                                                    <button
                                                        onClick={() => handleCloseTrade(trade.id)}
                                                        disabled={closingTradeId === trade.id || !isRunning}
                                                        style={{
                                                            marginTop: '8px', padding: '4px 10px', borderRadius: '6px',
                                                            background: 'rgba(244, 63, 94, 0.1)', border: '1px solid rgba(244, 63, 94, 0.3)',
                                                            color: 'var(--accent-red)', fontSize: '11px', fontWeight: 600,
                                                            cursor: (closingTradeId === trade.id || !isRunning) ? 'not-allowed' : 'pointer',
                                                            opacity: (closingTradeId === trade.id || !isRunning) ? 0.6 : 1
                                                        }}
                                                    >
                                                        {closingTradeId === trade.id ? 'Kapatılıyor...' : 'Kapat'}
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Trade History (Closed) */}
                    <div className="glass" style={{ borderRadius: '14px', overflow: 'hidden' }}>
                        <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)' }}>
                            <h2 style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>Trade History</h2>
                        </div>
                        <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '400px', overflowY: 'auto' }}>
                            {closedTrades.length === 0 ? (
                                <div key="empty-closed" style={{ padding: '40px 0', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
                                    No completed trades yet.
                                </div>
                            ) : (
                                <div key="list-closed" className="list-container" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                    {closedTrades.map((t: any) => (
                                        <div key={`closed-trade-${t.id}`} style={{
                                            padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(255,255,255,0.02)',
                                            border: '1px solid rgba(255,255,255,0.05)', display: 'flex', justifyContent: 'space-between', alignItems: 'center'
                                        }}>
                                            <div>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px', flexWrap: 'wrap' }}>
                                                    <span style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>{t.pair}</span>
                                                    <span style={{
                                                        padding: '1px 6px', borderRadius: '4px', fontSize: '9px', fontWeight: 700,
                                                        background: t.side === 'LONG' ? 'rgba(0, 216, 168, 0.12)' : 'rgba(244, 63, 94, 0.12)',
                                                        color: t.side === 'LONG' ? 'var(--accent-green)' : 'var(--accent-red)',
                                                    }}>{t.side}</span>
                                                    {t.regime && t.regime !== 'unknown' && (
                                                        <span style={{
                                                            padding: '1px 6px', borderRadius: '4px', fontSize: '9px', fontWeight: 600,
                                                            background: 'rgba(79, 158, 255, 0.1)', color: 'var(--accent-blue)', whiteSpace: 'nowrap'
                                                        }}>{t.regime.replace('_', ' ')}</span>
                                                    )}
                                                    {t.strategy && t.strategy !== 'unknown' && t.strategy !== 'none' && (
                                                        <span style={{
                                                            padding: '1px 6px', borderRadius: '4px', fontSize: '9px', fontWeight: 600,
                                                            background: 'rgba(255, 153, 0, 0.1)', color: 'rgb(255, 174, 52)', whiteSpace: 'nowrap'
                                                        }}>{t.strategy}</span>
                                                    )}
                                                    <span style={{ fontSize: '10px', color: 'var(--text-muted)', marginLeft: 4 }}>
                                                        {t.reason}
                                                    </span>
                                                </div>
                                                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                                                    {t.entry_time} → {t.exit_time}
                                                </div>
                                                <div style={{ display: 'flex', alignItems: 'center', marginTop: 4 }}>
                                                    <div style={{ fontSize: '9px', color: 'var(--text-muted)', background: 'rgba(255,255,255,0.03)', padding: '1px 5px', borderRadius: '3px' }}>
                                                         Score: {t.soft_score}/5 · {t.entry_type}
                                                    </div>
                                                    <PnLChart data={t.pnl_history} />
                                                </div>
                                            </div>
                                            <div style={{ textAlign: 'right', cursor: 'zoom-in' }} onClick={() => setSelectedTrade(t)}>
                                                <div style={{ fontSize: '14px', fontWeight: 700, color: t.pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                                    {`${t.pnl >= 0 ? '+' : ''}$${t.pnl !== undefined ? t.pnl.toFixed(2) : '0.00'}`}
                                                    <span style={{ fontSize: '11px', marginLeft: 4 }}>
                                                        {`(${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct !== undefined ? t.pnl_pct.toFixed(2) : '0.00'}%)`}
                                                    </span>
                                                </div>
                                                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                                                    in: ${t.entry} out: ${t.exit}
                                                </div>
                                                <div style={{ fontSize: '9px', color: 'var(--accent-blue)', fontWeight: 600, marginTop: 4 }}>
                                                    DETAILS →
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* SupervisorAI Alerts / Logs (Sag Sutun) */}
                <div className="glass" style={{ borderRadius: '14px', overflow: 'hidden' }}>
                    <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>System Logs</span>
                        <span className="ai-badge">Live</span>
                    </div>
                    <div style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px', height: '400px', overflowY: 'auto' }}>
                        {logs.length === 0 ? (
                            <div key="empty-logs" style={{ color: 'var(--text-muted)', fontSize: '12px', textAlign: 'center', marginTop: 20 }}>No logs yet.</div>
                        ) : (
                            <div key="list-logs" className="list-container" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                {logs.map((logObj: any, idx: number) => {
                                    let log = typeof logObj === 'string' ? logObj : (logObj.text || '');
                                    let color = 'var(--text-secondary)';
                                    if (log.includes("🟢")) color = 'var(--accent-green)';
                                    if (log.includes("✅")) color = 'var(--accent-blue)';
                                    if (log.includes("❌")) color = 'var(--accent-red)';
                                    if (log.includes("🛑") || log.includes("Error")) color = 'var(--accent-red)';
                                    if (log.includes("🤖")) color = 'var(--accent-primary)';

                                    let logKey = typeof logObj === 'string' ? `log-str-${idx}` : `log-obj-${logObj.id}`;

                                    return (
                                        <div key={logKey} style={{
                                            padding: '10px', borderRadius: '8px', backgroundColor: 'rgba(255,255,255,0.02)',
                                            border: '1px solid rgba(255,255,255,0.05)', fontSize: '11px', color,
                                            lineHeight: '1.4', fontFamily: 'monospace'
                                        }}>
                                            {log}
                                        </div>
                                    )
                                })}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* --- DETAILED TRADE MODAL --- */}
            {selectedTrade && (
                <div style={{
                    position: 'fixed', top: 0, left: 0, width: '100%', height: '100%',
                    background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(8px)',
                    zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    padding: '20px'
                }} onClick={() => setSelectedTrade(null)}>
                    <div className="glass" style={{
                        width: '100%', maxWidth: '700px', borderRadius: '20px', padding: '30px',
                        border: '1px solid rgba(255,255,255,0.1)', position: 'relative'
                    }} onClick={e => e.stopPropagation()}>
                        <button 
                            onClick={() => setSelectedTrade(null)}
                            style={{ position: 'absolute', top: 20, right: 20, background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '20px' }}
                        >✕</button>
                        
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px' }}>
                            <div>
                                <h2 style={{ fontSize: '22px', fontWeight: 800, color: 'var(--text-primary)' }}>
                                    {selectedTrade.pair} <span style={{ color: selectedTrade.side === 'LONG' ? 'var(--accent-green)' : 'var(--accent-red)', fontSize: '14px' }}>{selectedTrade.side}</span>
                                </h2>
                                <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>ID: {selectedTrade.id} · {selectedTrade.strategy} · {selectedTrade.regime?.replace('_', ' ')}</p>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: 4 }}>
                                    {selectedTrade.entry_time} {selectedTrade.exit_time && `→ ${selectedTrade.exit_time}`}
                                </div>
                            </div>
                            <div style={{ textAlign: 'right' }}>
                                <div style={{ fontSize: '24px', fontWeight: 800, color: selectedTrade.pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                    {selectedTrade.pnl >= 0 ? '+' : '-'}${Math.abs(selectedTrade.pnl).toFixed(2)}
                                </div>
                                <div style={{ fontSize: '14px', color: 'var(--text-muted)' }}>{selectedTrade.pnl_pct.toFixed(2)}% ROI</div>
                                {selectedTrade.exit_time && (
                                    <div style={{ 
                                        fontSize: '11px', fontWeight: 700, marginTop: 4, padding: '2px 8px', borderRadius: '4px',
                                        background: selectedTrade.pnl > 0 ? 'rgba(0, 216, 168, 0.1)' : 'rgba(244, 63, 94, 0.1)',
                                        color: selectedTrade.pnl > 0 ? 'var(--accent-green)' : 'var(--accent-red)',
                                        display: 'inline-block'
                                    }}>
                                        {selectedTrade.reason}
                                    </div>
                                )}
                            </div>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '25px' }}>
                            <div className="glass" style={{ padding: '15px', borderRadius: '12px', background: 'rgba(255,255,255,0.02)' }}>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '5px' }}>ENTRY CONDITIONS</div>
                                <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
                                    Quality Score: <span style={{ color: 'var(--accent-blue)' }}>{selectedTrade.soft_score}/5</span>
                                </div>
                                <div style={{ fontSize: '12px', color: 'var(--accent-primary)', marginTop: 4 }}>Type: {selectedTrade.entry_type?.toUpperCase()}</div>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: 4 }}>Price: ${selectedTrade.entry.toFixed(6)}</div>
                            </div>
                            <div className="glass" style={{ padding: '15px', borderRadius: '12px', background: 'rgba(255,255,255,0.02)' }}>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '5px' }}>PERFORMANCE METRICS</div>
                                <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--accent-green)' }}>
                                    Max Profit: +{selectedTrade.max_pnl_pct?.toFixed(2)}%
                                </div>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: 4 }}>
                                    {selectedTrade.exit ? `Exit: $${selectedTrade.exit.toFixed(6)}` : `Size: $${selectedTrade.margin.toFixed(2)}`}
                                </div>
                            </div>
                        </div>

                        <DetailedPnLChart data={selectedTrade.pnl_history} pair={selectedTrade.pair} />
                        
                        <div style={{ marginTop: '25px', padding: '15px', background: selectedTrade.exit_time ? 'rgba(255, 255, 255, 0.03)' : 'rgba(244, 63, 94, 0.05)', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
                            <div style={{ fontSize: '11px', color: selectedTrade.exit_time ? 'var(--text-muted)' : 'var(--accent-red)', fontWeight: 700, marginBottom: '5px' }}>
                                {selectedTrade.exit_time ? 'POSITION SUMMARY' : 'ACTIVE PROTECTIONS'}
                            </div>
                            <div style={{ display: 'flex', gap: '20px', fontSize: '12px', color: 'var(--text-primary)' }}>
                                <span>SL: ${selectedTrade.sl_price?.toFixed(6) || '--'}</span>
                                <span>TP: ${selectedTrade.tp_price?.toFixed(6) || '--'}</span>
                                {!selectedTrade.exit_time && selectedTrade.max_pnl_pct > 0.8 && <span style={{ color: 'var(--accent-green)' }}>✓ Breakeven Active</span>}
                                {selectedTrade.exit_time && <span>Status: {selectedTrade.pnl > 0 ? 'PROFIT' : 'LOSS'}</span>}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
