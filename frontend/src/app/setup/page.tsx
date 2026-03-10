"use client"

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';

const EXCHANGES = [
    { id: 'binance', name: 'Binance', logo: '🟡', desc: 'Worlds largest exchange' },
    { id: 'bybit', name: 'Bybit', logo: '🔵', desc: 'Leading derivatives exchange' },
    { id: 'paper', name: 'Paper Trading', logo: '📄', desc: 'Simulate with no real money (Recommended)' },
];

const RISK_LEVELS = [
    {
        id: 'conservative',
        name: 'Conservative',
        emoji: '🛡️',
        color: '#00d8a8',
        desc: 'Low risk. Small, frequent profits. Mostly scalping in stable pairs.',
        params: { maxDrawdown: '3%', tradeSize: '5%', dailyLoss: '$50', tp: '1.5%', sl: '0.5%' },
    },
    {
        id: 'balanced',
        name: 'Balanced',
        emoji: '⚖️',
        color: '#4f9eff',
        desc: 'Medium risk. Mix of scalping and swing trades for steady growth.',
        params: { maxDrawdown: '7%', tradeSize: '15%', dailyLoss: '$150', tp: '4%', sl: '2%' },
    },
    {
        id: 'aggressive',
        name: 'Aggressive',
        emoji: '⚡',
        color: '#f43f5e',
        desc: 'Higher risk. Larger positions, faster moves. Not for beginners.',
        params: { maxDrawdown: '15%', tradeSize: '30%', dailyLoss: '$400', tp: '8%', sl: '4%' },
    },
];

const BOT_PRESETS = [
    {
        id: 'scalper', name: 'AI Scalper', emoji: '⚡',
        desc: 'Makes 10-30 quick trades per day. EMA crossover + RSI filter. Best for volatile markets.',
        timeframe: '5m', pairs: ['BTC/USDT', 'ETH/USDT'],
    },
    {
        id: 'swing', name: 'AI Swing Trader', emoji: '🌊',
        desc: 'Makes 1-5 trades per day. Catches big moves with MACD + Bollinger Bands.',
        timeframe: '4h', pairs: ['BTC/USDT', 'SOL/USDT'],
    },
    {
        id: 'grid', name: 'AI Grid Bot', emoji: '🔲',
        desc: 'Places a grid of buy/sell orders. Earns from sideways markets 24/7.',
        timeframe: '1h', pairs: ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
    },
];

export default function SetupPage() {
    const router = useRouter();
    const [step, setStep] = useState(1);
    const [exchange, setExchange] = useState('paper');
    const [apiKey, setApiKey] = useState('');
    const [secret, setSecret] = useState('');
    const [capital, setCapital] = useState(1000);
    const [riskLevel, setRiskLevel] = useState('balanced');
    const [selectedBots, setSelectedBots] = useState<string[]>(['scalper']);
    const [launching, setLaunching] = useState(false);

    const totalSteps = 4;
    const risk = RISK_LEVELS.find(r => r.id === riskLevel)!;

    const toggleBot = (id: string) => {
        setSelectedBots(prev => prev.includes(id) ? prev.filter(b => b !== id) : [...prev, id]);
    };

    const handleLaunch = async () => {
        setLaunching(true);
        // Simulate saving config / starting bots
        await new Promise(res => setTimeout(res, 2500));
        // Save to localStorage for demo
        localStorage.setItem('smarttrade_config', JSON.stringify({ exchange, capital, riskLevel, selectedBots, setupComplete: true }));
        router.push('/dashboard');
    };

    return (
        <div style={{
            minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'radial-gradient(ellipse at 50% 0%, rgba(0,216,168,0.08) 0%, var(--bg-primary) 60%)',
            padding: '24px',
        }}>
            <div style={{ width: '100%', maxWidth: '680px' }}>
                {/* Header */}
                <div style={{ textAlign: 'center', marginBottom: '40px' }}>
                    <div style={{
                        width: '56px', height: '56px', borderRadius: '16px', margin: '0 auto 16px',
                        background: 'linear-gradient(135deg, #00d8a8, #4f9eff)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '26px',
                    }}>⚡</div>
                    <h1 style={{ fontSize: '28px', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '8px' }}>
                        Welcome to SmartTrade AI
                    </h1>
                    <p style={{ fontSize: '15px', color: 'var(--text-muted)' }}>
                        Set up once. Sleep. Let AI trade for you.
                    </p>
                </div>

                {/* Progress */}
                <div style={{ display: 'flex', gap: '6px', marginBottom: '32px' }}>
                    {Array.from({ length: totalSteps }).map((_, i) => (
                        <div key={i} style={{
                            flex: 1, height: '4px', borderRadius: '2px',
                            background: i < step ? 'var(--accent-green)' : 'var(--border)',
                            transition: 'background 0.4s ease',
                        }} />
                    ))}
                </div>

                {/* Step Card */}
                <div className="glass" style={{ borderRadius: '20px', padding: '32px', marginBottom: '20px' }}>

                    {/* STEP 1 — Exchange */}
                    {step === 1 && (
                        <div className="animate-fade-up">
                            <h2 style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
                                Connect Your Exchange
                            </h2>
                            <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '24px' }}>
                                Start with Paper Trading — no API key needed. Switch to real money anytime.
                            </p>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginBottom: '24px' }}>
                                {EXCHANGES.map(ex => (
                                    <div key={ex.id} onClick={() => setExchange(ex.id)} style={{
                                        padding: '16px 20px', borderRadius: '12px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '14px',
                                        border: exchange === ex.id ? '2px solid var(--accent-green)' : '1px solid var(--border)',
                                        background: exchange === ex.id ? 'rgba(0,216,168,0.06)' : 'var(--bg-card)',
                                        transition: 'all 0.15s',
                                    }}>
                                        <span style={{ fontSize: '24px' }}>{ex.logo}</span>
                                        <div style={{ flex: 1 }}>
                                            <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>{ex.name}</div>
                                            <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{ex.desc}</div>
                                        </div>
                                        {exchange === ex.id && <span style={{ color: 'var(--accent-green)', fontSize: '18px' }}>✓</span>}
                                    </div>
                                ))}
                            </div>
                            {exchange !== 'paper' && (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', padding: '16px', borderRadius: '10px', background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                                    <div>
                                        <label style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginBottom: '6px', fontWeight: 600 }}>API KEY</label>
                                        <input value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="Enter your API key" style={{
                                            width: '100%', padding: '10px 14px', borderRadius: '8px', border: '1px solid var(--border)',
                                            background: 'var(--bg-primary)', color: 'var(--text-primary)', fontSize: '13px', outline: 'none',
                                        }} />
                                    </div>
                                    <div>
                                        <label style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginBottom: '6px', fontWeight: 600 }}>SECRET KEY</label>
                                        <input value={secret} onChange={e => setSecret(e.target.value)} type="password" placeholder="Enter your secret key" style={{
                                            width: '100%', padding: '10px 14px', borderRadius: '8px', border: '1px solid var(--border)',
                                            background: 'var(--bg-primary)', color: 'var(--text-primary)', fontSize: '13px', outline: 'none',
                                        }} />
                                    </div>
                                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', padding: '8px 12px', borderRadius: '6px', background: 'rgba(251,191,36,0.06)', border: '1px solid rgba(251,191,36,0.2)' }}>
                                        🔒 Keys are encrypted and never shared. Only use Read + Trade permissions. Do NOT add withdrawal permissions.
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* STEP 2 — Capital & Risk */}
                    {step === 2 && (
                        <div className="animate-fade-up">
                            <h2 style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>Capital & Risk Profile</h2>
                            <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '24px' }}>How much do you want to allocate and how much risk can you handle?</p>

                            {/* Capital Input */}
                            <div style={{ marginBottom: '24px' }}>
                                <label style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '10px' }}>STARTING CAPITAL (USDT)</label>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                    <input type="number" value={capital} onChange={e => setCapital(Number(e.target.value))} min={100} style={{
                                        flex: 1, padding: '12px 16px', borderRadius: '10px', border: '1px solid var(--border)',
                                        background: 'var(--bg-card)', color: 'var(--text-primary)', fontSize: '20px', fontWeight: 700, outline: 'none',
                                    }} />
                                    <span style={{ fontSize: '14px', color: 'var(--text-muted)', fontWeight: 600 }}>USDT</span>
                                </div>
                                <div style={{ display: 'flex', gap: '8px', marginTop: '10px' }}>
                                    {[500, 1000, 5000, 10000].map(v => (
                                        <button key={v} onClick={() => setCapital(v)} style={{
                                            padding: '5px 12px', borderRadius: '6px', border: `1px solid ${capital === v ? 'var(--accent-green)' : 'var(--border)'}`,
                                            background: capital === v ? 'rgba(0,216,168,0.1)' : 'var(--bg-card)',
                                            color: capital === v ? 'var(--accent-green)' : 'var(--text-muted)', fontSize: '12px', fontWeight: 600, cursor: 'pointer',
                                        }}>${v.toLocaleString()}</button>
                                    ))}
                                </div>
                            </div>

                            {/* Risk Level */}
                            <div>
                                <label style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)', display: 'block', marginBottom: '10px' }}>RISK PROFILE</label>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                                    {RISK_LEVELS.map(r => (
                                        <div key={r.id} onClick={() => setRiskLevel(r.id)} style={{
                                            padding: '16px', borderRadius: '12px', cursor: 'pointer',
                                            border: riskLevel === r.id ? `2px solid ${r.color}` : '1px solid var(--border)',
                                            background: riskLevel === r.id ? `${r.color}08` : 'var(--bg-card)',
                                            transition: 'all 0.15s',
                                        }}>
                                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                                    <span style={{ fontSize: '20px' }}>{r.emoji}</span>
                                                    <span style={{ fontSize: '14px', fontWeight: 700, color: r.id === riskLevel ? r.color : 'var(--text-primary)' }}>{r.name}</span>
                                                </div>
                                                {riskLevel === r.id && <span style={{ color: r.color }}>✓</span>}
                                            </div>
                                            <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '12px' }}>{r.desc}</p>
                                            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                                                {Object.entries(r.params).map(([k, v]) => (
                                                    <span key={k} style={{
                                                        padding: '3px 8px', borderRadius: '4px', fontSize: '10px', fontWeight: 600,
                                                        background: 'rgba(255,255,255,0.04)', color: 'var(--text-secondary)',
                                                    }}>{k}: {v}</span>
                                                ))}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* STEP 3 — Choose Bots */}
                    {step === 3 && (
                        <div className="animate-fade-up">
                            <h2 style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>Choose Your AI Bots</h2>
                            <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '24px' }}>Select one or more bots. They'll run 24/7 automatically.</p>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                {BOT_PRESETS.map(bot => {
                                    const selected = selectedBots.includes(bot.id);
                                    return (
                                        <div key={bot.id} onClick={() => toggleBot(bot.id)} style={{
                                            padding: '20px', borderRadius: '14px', cursor: 'pointer',
                                            border: selected ? '2px solid var(--accent-green)' : '1px solid var(--border)',
                                            background: selected ? 'rgba(0,216,168,0.05)' : 'var(--bg-card)',
                                            transition: 'all 0.15s',
                                        }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '10px' }}>
                                                <span style={{ fontSize: '24px' }}>{bot.emoji}</span>
                                                <div style={{ flex: 1 }}>
                                                    <div style={{ fontSize: '15px', fontWeight: 700, color: selected ? 'var(--accent-green)' : 'var(--text-primary)' }}>{bot.name}</div>
                                                    <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Timeframe: {bot.timeframe} · Pairs: {bot.pairs.join(', ')}</div>
                                                </div>
                                                <div style={{
                                                    width: '22px', height: '22px', borderRadius: '50%', flexShrink: 0, transition: 'all 0.15s',
                                                    border: `2px solid ${selected ? 'var(--accent-green)' : 'var(--border)'}`,
                                                    background: selected ? 'var(--accent-green)' : 'transparent',
                                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                }}>
                                                    {selected && <span style={{ color: '#040d1a', fontSize: '12px', fontWeight: 900 }}>✓</span>}
                                                </div>
                                            </div>
                                            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>{bot.desc}</p>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {/* STEP 4 — Review & Launch */}
                    {step === 4 && (
                        <div className="animate-fade-up">
                            <h2 style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>Ready to Launch 🚀</h2>
                            <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '24px' }}>Review your configuration and start your AI trading system.</p>

                            {/* Summary Cards */}
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginBottom: '28px' }}>
                                {[
                                    { label: '🔗 Exchange', value: EXCHANGES.find(e => e.id === exchange)?.name },
                                    { label: '💰 Capital', value: `$${capital.toLocaleString()} USDT` },
                                    { label: '⚖️ Risk Profile', value: RISK_LEVELS.find(r => r.id === riskLevel)?.name },
                                    { label: '🤖 Active Bots', value: selectedBots.map(id => BOT_PRESETS.find(b => b.id === id)?.name).join(', ') },
                                ].map(item => (
                                    <div key={item.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px', borderRadius: '10px', background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                                        <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>{item.label}</span>
                                        <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>{item.value}</span>
                                    </div>
                                ))}
                            </div>

                            {/* Supervisor AI Notice */}
                            <div style={{ padding: '14px 16px', borderRadius: '10px', background: 'rgba(0,216,168,0.06)', border: '1px solid rgba(0,216,168,0.15)', marginBottom: '24px' }}>
                                <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--accent-green)', marginBottom: '6px' }}>🛡️ AI Protection Active</div>
                                <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                                    SupervisorAI will monitor every trade 24/7. If it detects unusual losses or market conditions, it will automatically pause the bot and notify you. You're always in control.
                                </p>
                            </div>

                            {/* Launch Button */}
                            <button onClick={handleLaunch} disabled={launching || selectedBots.length === 0} style={{
                                width: '100%', padding: '16px', borderRadius: '12px', border: 'none', cursor: launching ? 'wait' : 'pointer',
                                background: launching ? 'var(--bg-card)' : 'linear-gradient(135deg, #00d8a8, #4f9eff)',
                                color: launching ? 'var(--text-muted)' : 'white', fontSize: '16px', fontWeight: 800,
                                letterSpacing: '0.02em', boxShadow: launching ? 'none' : '0 8px 32px rgba(0,216,168,0.25)',
                                transition: 'all 0.2s',
                            }}>
                                {launching ? '🚀 Launching AI System…' : '🚀 Launch AI Trading System'}
                            </button>
                        </div>
                    )}
                </div>

                {/* Navigation */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    {step > 1 ? (
                        <button onClick={() => setStep(s => s - 1)} style={{
                            padding: '10px 20px', borderRadius: '10px', border: '1px solid var(--border)',
                            background: 'transparent', color: 'var(--text-muted)', fontSize: '13px', fontWeight: 600, cursor: 'pointer',
                        }}>← Back</button>
                    ) : <div />}
                    {step < totalSteps && (
                        <button onClick={() => setStep(s => s + 1)} disabled={step === 3 && selectedBots.length === 0} style={{
                            padding: '10px 24px', borderRadius: '10px', border: 'none', cursor: 'pointer',
                            background: 'linear-gradient(135deg, #00d8a8, #4f9eff)', color: 'white', fontSize: '13px', fontWeight: 700,
                            boxShadow: '0 4px 16px rgba(0,216,168,0.2)', transition: 'all 0.15s',
                        }}>Continue →</button>
                    )}
                </div>

                {/* Step indicator */}
                <p style={{ textAlign: 'center', fontSize: '12px', color: 'var(--text-muted)', marginTop: '16px' }}>
                    Step {step} of {totalSteps}
                </p>
            </div>
        </div>
    );
}
