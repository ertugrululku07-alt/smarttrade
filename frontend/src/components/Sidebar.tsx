"use client"

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const navItems = [
    {
        href: '/dashboard',
        label: 'Dashboard',
        icon: (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" />
                <rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" />
            </svg>
        ),
    },
    {
        href: '/bots',
        label: 'My Bots',
        badge: '3',
        icon: (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="11" width="18" height="11" rx="2" /><path d="M7 11V7a5 5 0 0 1 10 0v4" />
                <circle cx="12" cy="16" r="1" />
            </svg>
        ),
    },
    {
        href: '/bot-builder',
        label: 'Bot Builder',
        icon: (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
            </svg>
        ),
    },
    {
        href: '/backtest',
        label: 'Backtest',
        icon: (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
        ),
    },
    {
        href: '/multi-backtest',
        label: 'Multi Backtest',
        badge: 'NEW',
        icon: (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 3v18h18" /><path d="M7 16l4-8 4 4 4-8" />
            </svg>
        ),
    },
    {
        href: '/live',
        label: 'Live Monitor',
        badge: 'LIVE',
        icon: (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" /><polygon points="10 8 16 12 10 16 10 8" />
            </svg>
        ),
    },
];

const bottomItems = [
    {
        href: '/pricing',
        label: 'Pricing',
        icon: (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <line x1="12" y1="1" x2="12" y2="23" /><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
            </svg>
        ),
    },
    {
        href: '/settings',
        label: 'Settings',
        icon: (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="3" />
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14" />
                <path d="M12 2v2M12 20v2M2 12h2M20 12h2" />
            </svg>
        ),
    },
    {
        href: '/setup',
        label: 'Setup Wizard',
        icon: (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
        ),
    },
];

export default function Sidebar() {
    const pathname = usePathname();
    const [collapsed, setCollapsed] = useState(false);

    const isActive = (href: string) => pathname === href || (href !== '/' && pathname.startsWith(href));

    return (
        <aside
            className="flex flex-col h-full transition-all duration-300"
            style={{
                width: collapsed ? '60px' : '216px',
                background: 'var(--bg-secondary)',
                borderRight: '1px solid var(--border)',
                flexShrink: 0,
                transition: 'width 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
            }}
        >
            {/* Logo */}
            <div className="flex items-center h-14 border-b" style={{ borderColor: 'var(--border)', padding: collapsed ? '0 16px' : '0 14px' }}>
                <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                    style={{ background: 'linear-gradient(135deg, #00d8a8, #4f9eff)', minWidth: '32px' }}>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="white">
                        <path d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                </div>
                {!collapsed && (
                    <div style={{ marginLeft: '10px', flex: 1, overflow: 'hidden' }}>
                        <div style={{ fontSize: '13px', fontWeight: 800, color: 'var(--text-primary)', whiteSpace: 'nowrap' }}>SmartTrade</div>
                        <div className="ai-badge" style={{ marginTop: '2px' }}>AI v2</div>
                    </div>
                )}
                <button
                    onClick={() => setCollapsed(!collapsed)}
                    style={{ marginLeft: collapsed ? 'auto' : '4px', padding: '4px', borderRadius: '6px', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}
                >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        {collapsed ? <path d="M9 18l6-6-6-6" /> : <path d="M15 18l-6-6 6-6" />}
                    </svg>
                </button>
            </div>

            {/* Nav */}
            <nav style={{ flex: 1, padding: '10px 8px', overflowY: 'auto', overflowX: 'hidden' }}>
                {navItems.map(item => {
                    const active = isActive(item.href);
                    return (
                        <Link key={item.href} href={item.href} style={{ textDecoration: 'none', display: 'block', marginBottom: '2px' }}>
                            <div style={{
                                display: 'flex', alignItems: 'center', gap: '10px', padding: '9px 10px', borderRadius: '8px',
                                background: active ? 'rgba(0,216,168,0.08)' : 'transparent',
                                color: active ? 'var(--accent-green)' : 'var(--text-muted)',
                                border: active ? '1px solid rgba(0,216,168,0.15)' : '1px solid transparent',
                                transition: 'all 0.12s',
                                overflow: 'hidden',
                            }}>
                                <span style={{ flexShrink: 0, display: 'flex' }}>{item.icon}</span>
                                {!collapsed && (
                                    <>
                                        <span style={{ fontSize: '13px', fontWeight: 600, flex: 1, whiteSpace: 'nowrap' }}>{item.label}</span>
                                        {item.badge && (
                                            <span style={{
                                                fontSize: '9px', fontWeight: 800, padding: '2px 5px', borderRadius: '4px',
                                                background: item.badge === 'LIVE' ? 'rgba(0,216,168,0.15)' : 'rgba(79,158,255,0.15)',
                                                color: item.badge === 'LIVE' ? 'var(--accent-green)' : 'var(--accent-blue)',
                                                flexShrink: 0,
                                            }}>{item.badge}</span>
                                        )}
                                    </>
                                )}
                            </div>
                        </Link>
                    );
                })}
            </nav>

            {/* Bottom */}
            <div style={{ padding: '8px', borderTop: '1px solid var(--border)' }}>
                {bottomItems.map(item => {
                    const active = isActive(item.href);
                    return (
                        <Link key={item.href} href={item.href} style={{ textDecoration: 'none', display: 'block', marginBottom: '2px' }}>
                            <div style={{
                                display: 'flex', alignItems: 'center', gap: '10px', padding: '9px 10px', borderRadius: '8px',
                                color: active ? 'var(--accent-green)' : 'var(--text-muted)',
                                transition: 'all 0.12s', overflow: 'hidden',
                            }}>
                                <span style={{ flexShrink: 0, display: 'flex' }}>{item.icon}</span>
                                {!collapsed && <span style={{ fontSize: '13px', fontWeight: 600, whiteSpace: 'nowrap' }}>{item.label}</span>}
                            </div>
                        </Link>
                    );
                })}

                {!collapsed && (
                    <div style={{ margin: '8px 2px 0', padding: '10px 12px', borderRadius: '8px', background: 'rgba(0,216,168,0.04)', border: '1px solid rgba(0,216,168,0.1)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                            <span className="status-dot status-live"></span>
                            <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--accent-green)' }}>AI Online</span>
                        </div>
                        <p style={{ fontSize: '10px', color: 'var(--text-muted)', lineHeight: 1.4 }}>Strategist · Supervisor · Learner</p>
                    </div>
                )}
            </div>
        </aside>
    );
}
