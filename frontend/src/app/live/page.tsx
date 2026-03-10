"use client"

import React, { useState, useEffect } from 'react';

export default function LiveTradingPage() {
    const [statusData, setStatusData] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [actionLoading, setActionLoading] = useState(false);
    const [closingTradeId, setClosingTradeId] = useState<string | null>(null);
    const [maxOpenTrades, setMaxOpenTrades] = useState<number | null>(null);
    const [savingSettings, setSavingSettings] = useState(false);

    const fetchStatus = async () => {
        try {
            const res = await fetch("http://localhost:8000/live/quant/status");
            if (res.ok) {
                const data = await res.json();
                setStatusData(data);
                if (maxOpenTrades === null && data.max_open_trades_limit !== undefined) {
                    setMaxOpenTrades(data.max_open_trades_limit);
                }
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
            const res = await fetch(`http://localhost:8000/live/quant/${action}`, { method: 'POST' });
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
            const res = await fetch(`http://localhost:8000/live/quant/close-trade/${tradeId}`, { method: 'POST' });
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
            const res = await fetch('http://localhost:8000/live/quant/settings', {
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
                                        <div key={`open-trade-${trade.id}`} className="glass glass-hover" style={{
                                            borderRadius: '10px', padding: '14px 16px', border: '1px solid var(--border)'
                                        }}>
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
                                            </div>
                                            <div style={{ textAlign: 'right' }}>
                                                <div style={{ fontSize: '14px', fontWeight: 700, color: t.pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                                    {`${t.pnl >= 0 ? '+' : ''}$${t.pnl !== undefined ? t.pnl.toFixed(2) : '0.00'}`}
                                                    <span style={{ fontSize: '11px', marginLeft: 4 }}>
                                                        {`(${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct !== undefined ? t.pnl_pct.toFixed(2) : '0.00'}%)`}
                                                    </span>
                                                </div>
                                                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                                                    in: ${t.entry} out: ${t.exit}
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
        </div>
    );
}
