"use client"

import React, { useState, useEffect } from 'react';

const mockStats = [
    { label: "Total Portfolio Value", value: "$24,832.50", change: "+8.42%", positive: true, icon: "💼" },
    { label: "Today's PnL", value: "+$342.18", change: "+1.4%", positive: true, icon: "📈" },
    { label: "Active Bots", value: "5", change: "2 Scalping, 1 Swing, 2 MM", positive: true, icon: "🤖" },
    { label: "Win Rate (7d)", value: "68.4%", change: "↑ 3.2% vs last week", positive: true, icon: "🎯" },
];

const mockTrades = [
    { id: "T-1245", pair: "BTC/USDT", side: "BUY", entry: "43,200", exit: "44,100", pnl: "+$82.10", duration: "2h 14m", bot: "ScalpBot-1", status: "closed" },
    { id: "T-1244", pair: "ETH/USDT", side: "SELL", entry: "2,840", exit: "2,780", pnl: "+$48.30", duration: "45m", bot: "ScalpBot-1", status: "closed" },
    { id: "T-1243", pair: "BTC/USDT", side: "BUY", entry: "42,800", exit: "42,500", pnl: "-$28.50", duration: "1h 3m", bot: "SwingBot-1", status: "closed" },
    { id: "T-1242", pair: "SOL/USDT", side: "BUY", entry: "98.40", exit: "—", pnl: "+$12.40", duration: "Running", bot: "ScalpBot-2", status: "open" },
];

const mockAlerts = [
    { type: "info", message: "LearnerAI updated RSI parameters — best PnL improved 12%", time: "3m ago", ai: "LearnerAI" },
    { type: "warning", message: "SupervisorAI flagged consecutive losses on ETH/USDT. Reducing position size.", time: "18m ago", ai: "SupervisorAI" },
    { type: "success", message: "StrategistAI generated new scalping strategy for BTC/USDT 15m timeframe", time: "1h ago", ai: "StrategistAI" },
];

export default function DashboardPage() {
    const [time, setTime] = useState(new Date());

    useEffect(() => {
        const t = setInterval(() => setTime(new Date()), 1000);
        return () => clearInterval(t);
    }, []);

    return (
        <div style={{ padding: '24px', height: '100%', overflow: 'auto' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '28px' }}>
                <div>
                    <h1 style={{ fontSize: '24px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>
                        Dashboard
                    </h1>
                    <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                        {time.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })} · {time.toLocaleTimeString()}
                    </p>
                </div>
                <div style={{ display: 'flex', gap: '10px' }}>
                    <div style={{
                        display: 'flex', alignItems: 'center', gap: '8px',
                        padding: '8px 16px', borderRadius: '10px',
                        background: 'rgba(0, 216, 168, 0.06)', border: '1px solid rgba(0, 216, 168, 0.15)'
                    }}>
                        <span className="status-dot status-live"></span>
                        <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent-green)' }}>AI Systems Active</span>
                    </div>
                </div>
            </div>

            {/* Stats Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '24px' }}>
                {mockStats.map((stat, i) => (
                    <div key={i} className="glass glass-hover animate-fade-up" style={{
                        borderRadius: '14px', padding: '20px',
                        animationDelay: `${i * 0.05}s`,
                    }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                            <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: 500 }}>{stat.label}</span>
                            <span style={{ fontSize: '20px' }}>{stat.icon}</span>
                        </div>
                        <div style={{ fontSize: '22px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>{stat.value}</div>
                        <div style={{ fontSize: '12px', color: stat.positive ? 'var(--accent-green)' : 'var(--accent-red)', fontWeight: 500 }}>{stat.change}</div>
                    </div>
                ))}
            </div>

            {/* Main Content Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '20px' }}>
                {/* Recent Trades Table */}
                <div className="glass" style={{ borderRadius: '14px', overflow: 'hidden' }}>
                    <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <h2 style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>Recent Trades</h2>
                        <a href="/live" style={{ fontSize: '12px', color: 'var(--accent-blue)', textDecoration: 'none' }}>View all →</a>
                    </div>
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Pair</th><th>Side</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Duration</th><th>Bot</th>
                            </tr>
                        </thead>
                        <tbody>
                            {mockTrades.map(trade => (
                                <tr key={trade.id}>
                                    <td style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{trade.pair}</td>
                                    <td>
                                        <span style={{
                                            padding: '2px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 700,
                                            background: trade.side === 'BUY' ? 'rgba(0, 216, 168, 0.12)' : 'rgba(244, 63, 94, 0.12)',
                                            color: trade.side === 'BUY' ? 'var(--accent-green)' : 'var(--accent-red)',
                                        }}>{trade.side}</span>
                                    </td>
                                    <td>${trade.entry}</td>
                                    <td>{trade.status === 'open' ? <span style={{ color: 'var(--accent-yellow)' }}>Open</span> : `$${trade.exit}`}</td>
                                    <td style={{ color: trade.pnl.startsWith('+') ? 'var(--accent-green)' : 'var(--accent-red)', fontWeight: 600 }}>{trade.pnl}</td>
                                    <td>{trade.duration}</td>
                                    <td style={{ color: 'var(--accent-blue)', fontSize: '12px' }}>{trade.bot}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                {/* AI Alerts Panel */}
                <div className="glass" style={{ borderRadius: '14px', overflow: 'hidden' }}>
                    <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>AI Activity Feed</span>
                        <span className="ai-badge">Live</span>
                    </div>
                    <div style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        {mockAlerts.map((alert, i) => (
                            <div key={i} style={{
                                padding: '12px 14px', borderRadius: '10px',
                                background: alert.type === 'warning' ? 'rgba(251, 191, 36, 0.06)'
                                    : alert.type === 'success' ? 'rgba(0, 216, 168, 0.06)'
                                        : 'rgba(79, 158, 255, 0.06)',
                                border: `1px solid ${alert.type === 'warning' ? 'rgba(251, 191, 36, 0.15)'
                                    : alert.type === 'success' ? 'rgba(0, 216, 168, 0.15)'
                                        : 'rgba(79, 158, 255, 0.15)'}`,
                            }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                                    <span style={{
                                        fontSize: '10px', fontWeight: 700,
                                        color: alert.ai === 'SupervisorAI' ? 'var(--accent-yellow)' : alert.ai === 'LearnerAI' ? 'var(--accent-blue)' : 'var(--accent-green)',
                                    }}>{alert.ai}</span>
                                    <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{alert.time}</span>
                                </div>
                                <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.5' }}>{alert.message}</p>
                            </div>
                        ))}
                    </div>
                    {/* Quick Actions */}
                    <div style={{ padding: '12px 16px', borderTop: '1px solid var(--border)', display: 'flex', gap: '8px' }}>
                        <a href="/bot-builder" style={{
                            flex: 1, padding: '8px', textAlign: 'center', borderRadius: '8px', fontSize: '12px', fontWeight: 600,
                            background: 'rgba(0, 216, 168, 0.08)', border: '1px solid rgba(0, 216, 168, 0.15)',
                            color: 'var(--accent-green)', textDecoration: 'none',
                        }}>+ New Bot</a>
                        <a href="/backtest" style={{
                            flex: 1, padding: '8px', textAlign: 'center', borderRadius: '8px', fontSize: '12px', fontWeight: 600,
                            background: 'rgba(79, 158, 255, 0.08)', border: '1px solid rgba(79, 158, 255, 0.15)',
                            color: 'var(--accent-blue)', textDecoration: 'none',
                        }}>Backtest</a>
                    </div>
                </div>
            </div>
        </div>
    );
}
