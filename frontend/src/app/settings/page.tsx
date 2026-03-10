"use client"

import React, { useState } from 'react';

type TabKey = 'general' | 'risk' | 'notifications' | 'api';

const TABS: { key: TabKey; label: string; emoji: string }[] = [
    { key: 'general', label: 'General', emoji: '⚙️' },
    { key: 'risk', label: 'Risk Management', emoji: '🛡️' },
    { key: 'notifications', label: 'Notifications', emoji: '🔔' },
    { key: 'api', label: 'API Keys', emoji: '🔑' },
];

export default function SettingsPage() {
    const [tab, setTab] = useState<TabKey>('general');
    const [saved, setSaved] = useState(false);

    const handleSave = () => {
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
    };

    const Field = ({ label, help, children }: { label: string; help?: string; children: React.ReactNode }) => (
        <div style={{ marginBottom: '20px' }}>
            <label style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-muted)', display: 'block', marginBottom: '6px', textTransform: 'uppercase' }}>{label}</label>
            {children}
            {help && <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '5px' }}>{help}</p>}
        </div>
    );

    const Input = ({ defaultValue, type = 'text', placeholder = '' }: { defaultValue?: string | number; type?: string; placeholder?: string }) => (
        <input type={type} defaultValue={defaultValue} placeholder={placeholder} style={{
            width: '100%', padding: '10px 14px', borderRadius: '8px', border: '1px solid var(--border)',
            background: 'var(--bg-card)', color: 'var(--text-primary)', fontSize: '13px', outline: 'none',
        }} />
    );

    const Toggle = ({ label, defaultChecked = false }: { label: string; defaultChecked?: boolean }) => {
        const [on, setOn] = useState(defaultChecked);
        return (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px', borderRadius: '10px', background: 'var(--bg-card)', border: '1px solid var(--border)', marginBottom: '10px' }}>
                <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>{label}</span>
                <div onClick={() => setOn(!on)} style={{
                    width: '44px', height: '24px', borderRadius: '12px', cursor: 'pointer', position: 'relative', transition: 'background 0.2s',
                    background: on ? 'var(--accent-green)' : 'var(--border)',
                }}>
                    <div style={{
                        width: '18px', height: '18px', borderRadius: '50%', background: 'white', position: 'absolute', top: '3px',
                        left: on ? '23px' : '3px', transition: 'left 0.2s',
                    }} />
                </div>
            </div>
        );
    };

    return (
        <div style={{ padding: '24px', height: '100%', overflow: 'auto' }}>
            <div style={{ marginBottom: '28px' }}>
                <h1 style={{ fontSize: '24px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>Settings</h1>
                <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Configure your trading system</p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '200px 1fr', gap: '20px', maxWidth: '800px' }}>
                {/* Tabs */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    {TABS.map(t => (
                        <button key={t.key} onClick={() => setTab(t.key)} style={{
                            padding: '10px 14px', borderRadius: '8px', border: 'none', cursor: 'pointer', textAlign: 'left', fontSize: '13px', fontWeight: 600,
                            background: tab === t.key ? 'rgba(0,216,168,0.08)' : 'transparent',
                            color: tab === t.key ? 'var(--accent-green)' : 'var(--text-muted)',
                            borderLeft: `3px solid ${tab === t.key ? 'var(--accent-green)' : 'transparent'}`,
                        }}>
                            {t.emoji} {t.label}
                        </button>
                    ))}
                </div>

                {/* Content */}
                <div className="glass" style={{ borderRadius: '16px', padding: '24px' }}>
                    {tab === 'general' && (
                        <div>
                            <h2 style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '20px' }}>General Settings</h2>
                            <Field label="Preferred Currency"><Input defaultValue="USDT" /></Field>
                            <Field label="Default Exchange">
                                <select defaultValue="paper" style={{ width: '100%', padding: '10px 14px', borderRadius: '8px', border: '1px solid var(--border)', background: 'var(--bg-card)', color: 'var(--text-primary)', fontSize: '13px' }}>
                                    <option value="paper">📄 Paper Trading</option>
                                    <option value="binance">🟡 Binance</option>
                                    <option value="bybit">🔵 Bybit</option>
                                </select>
                            </Field>
                            <Field label="Default Timeframe">
                                <select style={{ width: '100%', padding: '10px 14px', borderRadius: '8px', border: '1px solid var(--border)', background: 'var(--bg-card)', color: 'var(--text-primary)', fontSize: '13px' }}>
                                    {['1m', '5m', '15m', '1h', '4h', '1d'].map(t => <option key={t}>{t}</option>)}
                                </select>
                            </Field>
                            <Toggle label="Auto-restart bots after system restart" defaultChecked={true} />
                            <Toggle label="Show paper trading indicator in UI" defaultChecked={true} />
                        </div>
                    )}

                    {tab === 'risk' && (
                        <div>
                            <h2 style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '20px' }}>Risk Management</h2>
                            <div style={{ padding: '12px 16px', borderRadius: '10px', background: 'rgba(0,216,168,0.06)', border: '1px solid rgba(0,216,168,0.15)', marginBottom: '20px' }}>
                                <p style={{ fontSize: '12px', color: 'var(--accent-green)', fontWeight: 600 }}>🛡️ SupervisorAI enforces these limits automatically</p>
                            </div>
                            <Field label="Max Daily Loss (USDT)" help="SupervisorAI pauses all bots if daily loss exceeds this amount"><Input type="number" defaultValue={150} /></Field>
                            <Field label="Max Drawdown (%)" help="Maximum portfolio drawdown before emergency stop"><Input type="number" defaultValue={7} /></Field>
                            <Field label="Max Concurrent Open Trades"><Input type="number" defaultValue={5} /></Field>
                            <Field label="Max Capital per Trade (%)" help="Never put more than this % in a single trade"><Input type="number" defaultValue={15} /></Field>
                            <Toggle label="Emergency stop on 3 consecutive losses" defaultChecked={true} />
                            <Toggle label="Auto-reduce position size after losses" defaultChecked={true} />
                            <Toggle label="Lock profits when daily target reached" defaultChecked={false} />
                        </div>
                    )}

                    {tab === 'notifications' && (
                        <div>
                            <h2 style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '20px' }}>Notifications</h2>
                            <Field label="Telegram Bot Token" help="Get notified on your phone for every trade">
                                <Input placeholder="110201543:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw..." />
                            </Field>
                            <Field label="Telegram Chat ID"><Input placeholder="-100123456789" /></Field>
                            <div style={{ marginBottom: '16px' }}>
                                <p style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '10px', textTransform: 'uppercase' }}>Notify me when:</p>
                                <Toggle label="Trade opened" defaultChecked={true} />
                                <Toggle label="Trade closed (with PnL)" defaultChecked={true} />
                                <Toggle label="SupervisorAI alert triggered" defaultChecked={true} />
                                <Toggle label="Daily summary (8:00 AM)" defaultChecked={true} />
                                <Toggle label="Bot paused due to losses" defaultChecked={true} />
                            </div>
                        </div>
                    )}

                    {tab === 'api' && (
                        <div>
                            <h2 style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '8px' }}>API Keys</h2>
                            <div style={{ padding: '12px 16px', borderRadius: '10px', background: 'rgba(251,191,36,0.06)', border: '1px solid rgba(251,191,36,0.2)', marginBottom: '20px' }}>
                                <p style={{ fontSize: '12px', color: 'var(--accent-yellow)' }}>⚠️ Only use API keys with Trade permission. Never enable Withdrawal.</p>
                            </div>
                            {['Binance', 'Bybit'].map(ex => (
                                <div key={ex} style={{ padding: '16px', borderRadius: '12px', background: 'var(--bg-card)', border: '1px solid var(--border)', marginBottom: '14px' }}>
                                    <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '12px' }}>{ex}</div>
                                    <Field label="API Key"><Input placeholder={`${ex} API Key...`} /></Field>
                                    <Field label="Secret Key"><Input type="password" placeholder="Secret..." /></Field>
                                    <Toggle label="Testnet mode" defaultChecked={true} />
                                </div>
                            ))}
                        </div>
                    )}

                    <button onClick={handleSave} style={{
                        marginTop: '12px', padding: '11px 28px', borderRadius: '10px', border: 'none', cursor: 'pointer',
                        background: saved ? 'rgba(0,216,168,0.1)' : 'linear-gradient(135deg, #00d8a8, #4f9eff)',
                        color: saved ? 'var(--accent-green)' : 'white', fontWeight: 700, fontSize: '13px', transition: 'all 0.2s',
                    }}>
                        {saved ? '✓ Saved!' : 'Save Changes'}
                    </button>
                </div>
            </div>
        </div>
    );
}
