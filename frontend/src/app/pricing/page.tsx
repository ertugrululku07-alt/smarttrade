"use client"

import React, { useState } from 'react';

const PLANS = [
    {
        id: 'starter',
        name: 'Starter',
        emoji: '🌱',
        price: 0,
        period: 'Free forever',
        color: '#6b7280',
        highlight: false,
        desc: 'Learn and test with paper trading',
        features: [
            { text: '1 paper trading bot', ok: true },
            { text: 'Backtest Lab (all 7 strategies)', ok: true },
            { text: 'Market Scanner', ok: true },
            { text: 'AI Regime Detection', ok: true },
            { text: 'Live trading', ok: false },
            { text: 'Futures / leverage', ok: false },
            { text: 'Real-time alerts', ok: false },
            { text: 'Priority support', ok: false },
        ],
        cta: 'Start Free',
    },
    {
        id: 'pro',
        name: 'Pro',
        emoji: '⚡',
        price: 29,
        period: '/month',
        color: '#00d8a8',
        highlight: true,
        badge: 'Most Popular',
        desc: 'For serious traders who want automation',
        features: [
            { text: '5 live bots', ok: true },
            { text: 'Backtest Lab + AI Adaptive', ok: true },
            { text: 'Market Scanner (10 pairs)', ok: true },
            { text: 'AI Regime Detection', ok: true },
            { text: 'Live spot trading', ok: true },
            { text: 'Telegram / Email alerts', ok: true },
            { text: 'Futures / leverage', ok: false },
            { text: 'Priority support', ok: false },
        ],
        cta: 'Get Pro',
    },
    {
        id: 'elite',
        name: 'Elite',
        emoji: '🚀',
        price: 79,
        period: '/month',
        color: '#a855f7',
        highlight: false,
        badge: 'Best Returns',
        desc: 'Futures, leverage, maximum performance',
        features: [
            { text: 'Unlimited bots', ok: true },
            { text: 'Backtest Lab + AI Adaptive', ok: true },
            { text: 'Market Scanner (50+ pairs)', ok: true },
            { text: 'AI Regime Detection', ok: true },
            { text: 'Live spot + futures trading', ok: true },
            { text: 'Leverage up to 20x', ok: true },
            { text: 'Telegram / Email / SMS alerts', ok: true },
            { text: '1-on-1 setup call', ok: true },
        ],
        cta: 'Go Elite',
    },
];

const FAQS = [
    {
        q: 'Is my capital safe?',
        a: 'Your funds stay on your own Binance account. SmartTrade only reads market data and sends trade orders — it never has withdrawal access to your funds.',
    },
    {
        q: 'What returns can I realistically expect?',
        a: 'In spot mode, backtests show 3-15% monthly in trending markets. With futures (Elite), 10-30% monthly is achievable but comes with amplified risk. Past backtest results do not guarantee future returns.',
    },
    {
        q: 'What if the system loses money?',
        a: 'The system has a built-in Max Drawdown Protection: it automatically stops trading when losses exceed a preset threshold (default 15%). You set the risk limits.',
    },
    {
        q: 'Can I cancel anytime?',
        a: 'Yes. No contracts, no lock-in. Cancel anytime from your account settings.',
    },
    {
        q: 'Do I need trading experience?',
        a: 'No. The Setup Wizard guides you through everything in under 5 minutes. The AI handles all trading decisions automatically.',
    },
];

export default function PricingPage() {
    const [billing, setBilling] = useState<'monthly' | 'annual'>('monthly');
    const [openFaq, setOpenFaq] = useState<number | null>(null);

    const price = (base: number) => billing === 'annual' ? Math.round(base * 0.75) : base;

    return (
        <div style={{ padding: '32px 24px', maxWidth: '1100px', margin: '0 auto', overflow: 'auto', height: '100%' }}>

            {/* Header */}
            <div style={{ textAlign: 'center', marginBottom: '48px' }}>
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '6px 14px', borderRadius: '20px', background: 'rgba(0,216,168,0.1)', border: '1px solid rgba(0,216,168,0.2)', marginBottom: '16px' }}>
                    <span style={{ fontSize: '12px', color: 'var(--accent-green)', fontWeight: 700 }}>💎 Transparent Pricing</span>
                </div>
                <h1 style={{ fontSize: '36px', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '12px', lineHeight: 1.2 }}>
                    Start Automated Trading<br />
                    <span style={{ background: 'linear-gradient(135deg, #00d8a8, #4f9eff)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                        While You Sleep
                    </span>
                </h1>
                <p style={{ fontSize: '15px', color: 'var(--text-muted)', maxWidth: '500px', margin: '0 auto 24px', lineHeight: 1.7 }}>
                    Real Binance data · AI-driven decisions · 7/24 automated · Your funds stay on your exchange
                </p>

                {/* Billing toggle */}
                <div style={{ display: 'inline-flex', gap: '4px', padding: '4px', borderRadius: '10px', background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                    {(['monthly', 'annual'] as const).map(b => (
                        <button key={b} onClick={() => setBilling(b)} style={{
                            padding: '8px 20px', borderRadius: '7px', border: 'none', cursor: 'pointer', fontWeight: 700, fontSize: '13px',
                            background: billing === b ? 'var(--accent-green)' : 'transparent',
                            color: billing === b ? '#000' : 'var(--text-muted)',
                        }}>
                            {b === 'monthly' ? 'Monthly' : 'Annual'}{b === 'annual' && <span style={{ marginLeft: '6px', fontSize: '10px', padding: '1px 5px', borderRadius: '3px', background: 'rgba(0,0,0,0.2)' }}>-25%</span>}
                        </button>
                    ))}
                </div>
            </div>

            {/* Plans */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', marginBottom: '60px' }}>
                {PLANS.map(plan => (
                    <div key={plan.id} style={{
                        borderRadius: '20px',
                        padding: '28px 24px',
                        border: plan.highlight ? `2px solid ${plan.color}` : '1px solid var(--border)',
                        background: plan.highlight ? `rgba(0,216,168,0.04)` : 'var(--bg-card)',
                        position: 'relative',
                        transform: plan.highlight ? 'translateY(-6px)' : 'none',
                        boxShadow: plan.highlight ? `0 20px 60px rgba(0,216,168,0.15)` : 'none',
                        transition: 'all 0.2s',
                    }}>
                        {plan.badge && (
                            <div style={{ position: 'absolute', top: '-12px', left: '50%', transform: 'translateX(-50%)', padding: '4px 14px', borderRadius: '20px', background: plan.color, color: plan.id === 'pro' ? '#000' : '#fff', fontSize: '11px', fontWeight: 800, whiteSpace: 'nowrap' }}>
                                {plan.badge}
                            </div>
                        )}

                        <div style={{ fontSize: '28px', marginBottom: '8px' }}>{plan.emoji}</div>
                        <div style={{ fontSize: '18px', fontWeight: 800, color: plan.color, marginBottom: '4px' }}>{plan.name}</div>
                        <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '20px', lineHeight: 1.5 }}>{plan.desc}</div>

                        <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px', marginBottom: '4px' }}>
                            <span style={{ fontSize: '38px', fontWeight: 800, color: 'var(--text-primary)' }}>
                                {plan.price === 0 ? 'Free' : `$${price(plan.price)}`}
                            </span>
                            {plan.price > 0 && <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>/month</span>}
                        </div>
                        {billing === 'annual' && plan.price > 0 && (
                            <div style={{ fontSize: '11px', color: 'var(--accent-green)', marginBottom: '20px' }}>
                                Save ${(plan.price - price(plan.price)) * 12}/year
                            </div>
                        )}
                        {plan.price === 0 && <div style={{ marginBottom: '20px' }} />}

                        <button style={{
                            width: '100%', padding: '12px', borderRadius: '10px', border: 'none', cursor: 'pointer',
                            fontWeight: 800, fontSize: '14px',
                            background: plan.highlight ? plan.color : `${plan.color}15`,
                            color: plan.highlight ? '#000' : plan.color,
                            marginBottom: '24px',
                        }}>
                            {plan.cta}
                        </button>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                            {plan.features.map((f, i) => (
                                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <span style={{ fontSize: '14px', color: f.ok ? plan.color : 'var(--border)', flexShrink: 0 }}>{f.ok ? '✓' : '✗'}</span>
                                    <span style={{ fontSize: '13px', color: f.ok ? 'var(--text-secondary)' : 'var(--text-muted)' }}>{f.text}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>

            {/* Performance Promise */}
            <div style={{ borderRadius: '20px', padding: '32px', background: 'linear-gradient(135deg, rgba(0,216,168,0.05), rgba(79,158,255,0.05))', border: '1px solid rgba(0,216,168,0.15)', marginBottom: '48px' }}>
                <h2 style={{ fontSize: '22px', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '8px' }}>What You're Actually Buying</h2>
                <p style={{ fontSize: '14px', color: 'var(--text-muted)', marginBottom: '24px' }}>Not "guaranteed profits" — but a professional system that works while you don't.</p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
                    {[
                        { icon: '🤖', title: 'AI Strategy Selection', desc: 'System detects trending vs ranging market and picks the right approach automatically' },
                        { icon: '🛡️', title: 'Risk Protection', desc: '15% max drawdown stop, automatic position sizing, no emotional mistakes' },
                        { icon: '📊', title: 'Real Backtesting', desc: 'Test any strategy on real Binance data before risking a single dollar' },
                        { icon: '😴', title: '24/7 Automated', desc: 'Works while you sleep. No screen time, no manual decisions, no missed setups' },
                    ].map((item, i) => (
                        <div key={i} style={{ padding: '16px', borderRadius: '12px', background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                            <div style={{ fontSize: '24px', marginBottom: '8px' }}>{item.icon}</div>
                            <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>{item.title}</div>
                            <div style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.5 }}>{item.desc}</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* FAQ */}
            <div style={{ marginBottom: '48px' }}>
                <h2 style={{ fontSize: '22px', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '20px', textAlign: 'center' }}>Frequently Asked Questions</h2>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {FAQS.map((faq, i) => (
                        <div key={i} style={{ borderRadius: '12px', border: '1px solid var(--border)', background: 'var(--bg-card)', overflow: 'hidden' }}>
                            <button onClick={() => setOpenFaq(openFaq === i ? null : i)} style={{
                                width: '100%', padding: '16px 20px', border: 'none', background: 'transparent', cursor: 'pointer',
                                display: 'flex', justifyContent: 'space-between', alignItems: 'center', textAlign: 'left',
                            }}>
                                <span style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>{faq.q}</span>
                                <span style={{ fontSize: '18px', color: 'var(--text-muted)', transform: openFaq === i ? 'rotate(45deg)' : 'none', transition: 'transform 0.2s' }}>+</span>
                            </button>
                            {openFaq === i && (
                                <div style={{ padding: '0 20px 16px', fontSize: '13px', color: 'var(--text-muted)', lineHeight: 1.7 }}>
                                    {faq.a}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* CTA */}
            <div style={{ textAlign: 'center', padding: '40px', borderRadius: '20px', background: 'linear-gradient(135deg, rgba(168,85,247,0.06), rgba(79,158,255,0.06))', border: '1px solid rgba(168,85,247,0.15)' }}>
                <h2 style={{ fontSize: '24px', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '8px' }}>Ready to start earning passively?</h2>
                <p style={{ fontSize: '14px', color: 'var(--text-muted)', marginBottom: '24px' }}>Start free and upgrade when you're ready. No card required.</p>
                <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
                    <button style={{ padding: '14px 32px', borderRadius: '10px', border: 'none', cursor: 'pointer', fontWeight: 800, fontSize: '15px', background: 'linear-gradient(135deg, #00d8a8, #4f9eff)', color: '#000' }}>
                        Start Free — No Card Required
                    </button>
                    <button style={{ padding: '14px 32px', borderRadius: '10px', border: '1px solid var(--border)', cursor: 'pointer', fontWeight: 700, fontSize: '15px', background: 'transparent', color: 'var(--text-secondary)' }}>
                        View Backtest Results →
                    </button>
                </div>
            </div>
        </div>
    );
}
