"use client"

import React, { useState, useEffect, useCallback } from 'react';
import { getApiUrl } from '@/utils/api';

const API = getApiUrl();

const STRATEGY_LABELS: Record<string, string> = {
    quant_grid: 'RSI Grid Scalper',
    turtle: 'Turtle Trend Follower',
    adaptive: 'Regime Adaptive',
    market_maker: 'Liquidity Grab / Sniper',
    smart_money: 'Smart Money / Market Maker',
    xgboost: 'XGBoost AI Predictor',
    predator: 'Predator Piyasa Avcısı',
    director: '🎩 Direktör AI (Meta-Model)',
};

const BOT_COLORS: Record<string, string> = {
    'quant-grid': '#00d8a8',
    'turtle-trend': '#f59e0b',
    'adaptive-regime': '#a78bfa',
    'market-maker': '#ec4899',
    'smart-money': '#3b82f6',
    'xgboost-ai': '#f97316',
    'predator-ai': '#ef4444',
    'director-1m': '#c084fc',
    'director-5m': '#d8b4fe',
    'director-15m': '#e9d5ff',
    'director-1h': '#f3e8ff',
    'director-4h': '#faf5ff',
};

interface TrainingStatus {
    status: string;
    progress: number;
    detail: string;
    result?: {
        accuracy: number;
        symbols_count: number;
        total_samples: number;
        timeframe: string;
        top_features: { feature: string; importance: number }[];
        class_report: Record<string, { precision: number; recall: number; f1: number }>;
        trained_at: string;
    };
}

interface OpenTrade {
    id: string;
    side: string;
    symbol: string;
    entry_price: number;
    current_price: number;
    qty: number;
    margin: number;
    entry_time: string;
    unrealized_pnl: number;
    unrealized_pnl_pct: number;
}

interface ClosedTrade {
    id: string;
    side: string;
    symbol: string;
    entry_price: number;
    exit_price: number;
    margin: number;
    pnl: number;
    reason: string;
    entry_time: string;
    exit_time: string;
}

interface BotStatus {
    id: string;
    name: string;
    symbol: string;
    timeframe: string;
    strategy: string;
    status: string;
    balance: number;
    initial_balance: number;
    total_pnl: number;
    total_pnl_pct: number;
    today_pnl: number;
    win_rate: number;
    total_trades: number;
    open_trades_count: number;
    unrealized_pnl: number;
    recent_logs: { id: string; text: string }[];
    open_trades: OpenTrade[];
    closed_trades: ClosedTrade[];
}

interface Summary {
    total_pnl: number;
    today_pnl: number;
    running_bots: number;
    total_bots: number;
}

type Tab = 'bots' | 'positions' | 'history';

const fmt = (n: number, dec = 2) => `${n >= 0 ? '+' : ''}${n.toFixed(dec)}`;
const fmtPrice = (p: number) => p >= 1000 ? p.toFixed(2) : p >= 1 ? p.toFixed(4) : p.toFixed(6);

export default function BotsPage() {
    const [bots, setBots] = useState<BotStatus[]>([]);
    const [summary, setSummary] = useState<Summary | null>(null);
    const [loading, setLoading] = useState(true);
    const [allStarted, setAllStarted] = useState(false);
    const [tab, setTab] = useState<Tab>('bots');
    const [selectedLog, setSelectedLog] = useState<BotStatus | null>(null);

    // XGBoost Training State
    const [trainStatus, setTrainStatus] = useState<TrainingStatus>({ status: 'idle', progress: 0, detail: '' });
    const [trainOpen, setTrainOpen] = useState(false);
    const [trainTf, setTrainTf] = useState('15m');
    const [trainLimit, setTrainLimit] = useState(1500);

    // XGBoost Enhancements State
    const [schedStatus, setSchedStatus] = useState<{ status: string, next_run?: string }>({ status: 'stopped' });
    const [ensembStatus, setEnsembStatus] = useState<{ ensemble_active: boolean, total_models: number }>({ ensemble_active: false, total_models: 0 });

    // Settings Modal State
    const [settingsOpen, setSettingsOpen] = useState<BotStatus | null>(null);
    const [settingsForm, setSettingsForm] = useState<any>({});

    // PnL Details Modal State
    const [pnlDetailsOpen, setPnlDetailsOpen] = useState<BotStatus | null>(null);

    // Trade History Filter State
    const [filterBot, setFilterBot] = useState<string>('all');
    const [filterSide, setFilterSide] = useState<string>('all');
    const [filterResult, setFilterResult] = useState<string>('all');
    const [filterSymbol, setFilterSymbol] = useState<string>('');

    // Fetch configs mapped by bot ID
    const [configs, setConfigs] = useState<Record<string, any>>({});

    // Prevent hydration mismatch
    const [isMounted, setIsMounted] = useState(false);
    useEffect(() => {
        setIsMounted(true);
    }, []);

    const fetchConfigs = async () => {
        try {
            const res = await fetch(getApiUrl('/live/bots/configs'));
            if (res.ok) {
                const data = await res.json();
                if (data.success && data.configs) {
                    setConfigs(data.configs);
                }
            }
        } catch { }
    };

    const fetchAIExtras = async () => {
        try {
            const [sRes, eRes] = await Promise.all([
                fetch(getApiUrl('/ai/xgboost/scheduler/status')),
                fetch(getApiUrl('/ai/xgboost/ensemble/status'))
            ]);
            if (sRes.ok) setSchedStatus(await sRes.json());
            if (eRes.ok) setEnsembStatus(await eRes.json());
        } catch { }
    };

    useEffect(() => {
        fetchConfigs();
        fetchAIExtras();
        // Check training status on mount as well so we don't lose state if we refresh
        fetch(getApiUrl('/ai/xgboost/status')).then(res => res.json()).then(resData => {
            if (resData.success && resData.data) {
                const data = resData.data;
                if (data.status === 'fetching' || data.status === 'training') {
                    setTrainStatus(data);
                }
            }
        }).catch(err => console.error(err));
    }, []);

    // Poll training status when active
    useEffect(() => {
        let iv: NodeJS.Timeout;
        const poll = async () => {
            try {
                const res = await fetch(getApiUrl('/ai/xgboost/status'));
                if (res.ok) {
                    const resData = await res.json();
                    if (resData.success && resData.data) {
                        const data = resData.data;
                        setTrainStatus(data);
                        // Stop polling if done or error
                        if (data.status === 'done' || data.status === 'error' || data.status === 'idle') {
                            clearInterval(iv);
                        }
                    } else if (resData.status) {
                        // Fallback in case endpoint returns direct object
                        setTrainStatus(resData);
                        if (resData.status === 'done' || resData.status === 'error' || resData.status === 'idle') {
                            clearInterval(iv);
                        }
                    }
                }
            } catch { }
        };

        if (trainStatus.status === 'fetching' || trainStatus.status === 'training') {
            iv = setInterval(poll, 2000);
            return () => clearInterval(iv);
        }
    }, [trainStatus.status]);

    const handleStartTraining = async () => {
        try {
            const endpoint = trainTf === 'all' ? getApiUrl('/ai/xgboost/director/train-all') : getApiUrl('/ai/xgboost/train');
            const payload = trainTf === 'all'
                ? { timeframe: '15m', limit: trainLimit } // Backend ignores tf for train-all but requires it in schema
                : { timeframe: trainTf, limit: trainLimit };

            const res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (res.ok) {
                setTrainStatus({ status: 'fetching', progress: 5, detail: trainTf === 'all' ? 'Tüm zaman dilimleri için eğitim başlıyor...' : 'Eğitim başlatılıyor...' });
            }
        } catch { }
    };

    const handleResetTraining = async () => {
        try {
            const res = await fetch(getApiUrl('/ai/xgboost/reset'), { method: 'POST' });
            if (res.ok) {
                setTrainStatus({ status: 'idle', progress: 0, detail: 'Eğitim durumu sıfırlandı.' });
            }
        } catch { }
    };

    const toggleScheduler = async () => {
        const action = schedStatus.status === 'running' ? 'stop' : 'start';
        await fetch(getApiUrl(`/ai/xgboost/scheduler/${action}`), { method: 'POST' });
        setTimeout(fetchAIExtras, 1000);
    };

    const handleOpenSettings = (bot: BotStatus) => {
        const botCfg = configs[bot.id] || {};
        const isXGB = bot.id === 'xgboost-ai';
        setSettingsForm({
            timeframe: botCfg.timeframe || bot.timeframe,
            leverage: botCfg.leverage || 5,
            risk_pct: botCfg.risk_pct * 100 || 2.0,
            atr_tp_mult: botCfg.atr_tp_mult || 1.5,
            atr_sl_mult: botCfg.atr_sl_mult || (isXGB ? 1.0 : 2.5),
            min_confidence: botCfg.min_confidence || 0.6,
        });
        setSettingsOpen(bot);
    };

    const handleSaveSettings = async () => {
        if (!settingsOpen) return;
        try {
            const payload = { ...settingsForm, risk_pct: settingsForm.risk_pct / 100.0 };
            const res = await fetch(getApiUrl(`/live/bots/${settingsOpen.id}/config`), {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (res.ok) {
                setSettingsOpen(null);
                fetchConfigs();
                setTimeout(fetchStatus, 800);
            }
        } catch { }
    };

    const fetchStatus = useCallback(async () => {
        try {
            const res = await fetch(getApiUrl('/live/bots/status'));
            if (!res.ok) return;
            const data = await res.json();
            setBots(data.bots || []);
            setSummary(data.summary || null);
            setAllStarted((data.bots || []).some((b: BotStatus) => b.status === 'running'));
        } catch { }
        finally { setLoading(false); }
    }, []);

    useEffect(() => {
        fetchStatus();
        const iv = setInterval(fetchStatus, 8000);
        return () => clearInterval(iv);
    }, [fetchStatus]);

    const handleStartAll = async () => { await fetch(getApiUrl('/live/bots/start-all'), { method: 'POST' }); setTimeout(fetchStatus, 1000); };
    const handleStopAll = async () => { await fetch(getApiUrl('/live/bots/stop-all'), { method: 'POST' }); setTimeout(fetchStatus, 1000); };
    const handleToggle = async (bot: BotStatus) => {
        const action = bot.status === 'running' ? 'stop' : 'start';
        await fetch(getApiUrl(`/live/bots/${bot.id}/${action}`), { method: 'POST' });
        setTimeout(fetchStatus, 800);
    };
    const handleReset = async (id: string) => { await fetch(getApiUrl(`/live/bots/${id}/reset`), { method: 'POST' }); setTimeout(fetchStatus, 800); };

    const handleResetAll = async () => {
        const balStr = prompt("Tüm botları sıfırlamak istediğinize emin misiniz?\n\n(Yeni Başlangıç Bakiyesini girebilir, veya boş bırakarak mevcut bakiyeyle sıfırlayabilirsiniz)", "1000");
        if (balStr === null) return; // User cancelled

        let initial_balance = parseFloat(balStr);
        const payload = !isNaN(initial_balance) && initial_balance > 0 ? { initial_balance } : {};

        try {
            const res = await fetch(getApiUrl('/live/bots/reset-all'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (res.ok) {
                fetchConfigs();
                setTimeout(fetchStatus, 800);
            }
        } catch { }
    };

    const handleCloseAllTrades = async () => {
        if (!confirm("DİKKAT! Çalışan tüm botlardaki AÇIK İŞLEMLERİ (zararda olsalar dahi) piyasa fiyatından anında kapatmak istediğinize emin misiniz?")) {
            return;
        }
        try {
            const res = await fetch(`${API}/live/bots/close-all`, { method: 'POST' });
            if (res.ok) setTimeout(fetchStatus, 800);
        } catch { }
    };

    const totalPnl = summary?.total_pnl ?? bots.reduce((s, b) => s + b.total_pnl, 0);
    const todayPnl = summary?.today_pnl ?? bots.reduce((s, b) => s + b.today_pnl, 0);
    const runningCount = summary?.running_bots ?? bots.filter(b => b.status === 'running').length;
    const totalUnrealized = bots.reduce((s, b) => s + (b.unrealized_pnl ?? 0), 0);

    // Flatten all open/closed trades across bots
    const allOpen: (OpenTrade & { botId: string; botName: string })[] = bots.flatMap(b =>
        (b.open_trades || []).map(t => ({ ...t, botId: b.id, botName: b.name }))
    );
    const allClosed: (ClosedTrade & { botId: string; botName: string })[] = bots.flatMap(b =>
        (b.closed_trades || []).map(t => ({ ...t, botId: b.id, botName: b.name }))
    ).sort((a, b) => (b.exit_time || '').localeCompare(a.exit_time || ''));

    const tabs: { key: Tab; label: string; badge?: number }[] = [
        { key: 'bots', label: '🤖 Botlar' },
        { key: 'positions', label: '📊 Açık Pozisyonlar', badge: allOpen.length },
        { key: 'history', label: '📋 İşlem Geçmişi', badge: allClosed.length },
    ];

    return (
        <div style={{ padding: '24px', height: '100%', overflow: 'auto' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <div>
                    <h1 style={{ fontSize: '22px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '3px' }}>5-Bot Demo Yarışı</h1>
                    <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                        <span className="status-dot status-live" style={{ marginRight: '5px' }} />
                        {runningCount} / {bots.length || 5} bot çalışıyor · Her 8 saniyede güncellenir
                    </p>
                </div>
                <div>
                    <div style={{ display: 'flex', gap: '8px' }}>
                        <button onClick={handleCloseAllTrades} style={{ padding: '9px 15px', borderRadius: '10px', border: '1px solid rgba(251,191,36,0.3)', cursor: 'pointer', background: 'rgba(251,191,36,0.1)', color: '#fbbf24', fontSize: '13px', fontWeight: 700 }} title="Açık işlemleri anında kapat">📉 Açıkları Kapat</button>
                        <button onClick={handleResetAll} style={{ padding: '9px 15px', borderRadius: '10px', border: '1px dashed var(--border)', cursor: 'pointer', background: 'transparent', color: 'var(--text-muted)', fontSize: '13px', fontWeight: 600 }}>Tümünü Sıfırla</button>

                        {!allStarted
                            ? <button onClick={handleStartAll} style={{ padding: '9px 18px', borderRadius: '10px', border: 'none', cursor: 'pointer', background: 'linear-gradient(135deg,#00d8a8,#4f9eff)', color: 'white', fontSize: '13px', fontWeight: 700 }}>▶ Tüm Botları Başlat</button>
                            : <button onClick={handleStopAll} style={{ padding: '9px 18px', borderRadius: '10px', border: '1px solid rgba(244,63,94,0.3)', cursor: 'pointer', background: 'rgba(244,63,94,0.1)', color: '#f43f5e', fontSize: '13px', fontWeight: 700 }}>⏹ Tümünü Durdur</button>
                        }
                    </div>
                </div>
            </div>

            {/* Summary Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '20px' }}>
                {[
                    { label: 'Toplam Kapalı PnL', value: `${fmt(totalPnl)}$`, pos: totalPnl >= 0 },
                    { label: 'Açık Pozisyon PnL', value: `${fmt(totalUnrealized)}$`, pos: totalUnrealized >= 0 },
                    { label: "Bugünkü PnL", value: `${fmt(todayPnl)}$`, pos: todayPnl >= 0 },
                    { label: 'Sistem', value: runningCount > 0 ? '🤖 Canlı' : '⏸ Bekliyor', pos: runningCount > 0 },
                ].map((c, i) => (
                    <div key={i} className="glass" style={{ padding: '14px 18px', borderRadius: '12px' }}>
                        <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '6px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>{c.label}</div>
                        <div style={{ fontSize: '20px', fontWeight: 800, color: c.pos ? 'var(--accent-green)' : 'var(--accent-red)' }}>{c.value}</div>
                    </div>
                ))}
            </div>

            {/* Tabs */}
            <div style={{ display: 'flex', gap: '4px', marginBottom: '18px', borderBottom: '1px solid var(--border)', paddingBottom: '0' }}>
                {tabs.map(t => (
                    <button key={t.key} onClick={() => setTab(t.key)} style={{
                        padding: '8px 16px', borderRadius: '8px 8px 0 0', border: 'none', cursor: 'pointer', fontSize: '13px', fontWeight: 600,
                        background: tab === t.key ? 'var(--bg-card)' : 'transparent',
                        color: tab === t.key ? 'var(--text-primary)' : 'var(--text-muted)',
                        borderBottom: tab === t.key ? '2px solid var(--accent-green)' : '2px solid transparent',
                        transition: 'all 0.15s',
                        display: 'flex', alignItems: 'center', gap: '6px',
                    }}>
                        {t.label}
                        {isMounted && t.badge !== undefined && t.badge > 0 && (
                            <span style={{ background: 'rgba(0,216,168,0.2)', color: 'var(--accent-green)', fontSize: '10px', fontWeight: 700, padding: '1px 6px', borderRadius: '10px' }}>
                                {t.badge}
                            </span>
                        )}
                    </button>
                ))}
            </div>

            {/* ── TAB: BOTLAR ─────────────────────────────────────── */}
            {tab === 'bots' && (
                <>
                    {/* XGBoost Training Panel */}
                    <div className="glass" style={{ borderRadius: '16px', marginBottom: '20px', border: '1px solid rgba(249,115,22,0.2)', overflow: 'hidden' }}>
                        <div style={{ padding: '14px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border)', background: 'rgba(249,115,22,0.04)' }}>
                            <div>
                                <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)' }}>🧠 XGBoost AI Model Eğitimi</div>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>Çoklu pariteden veri çekerek AI modelini eğitin</div>
                            </div>
                            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                                {trainStatus.status === 'done' && trainStatus.result && (
                                    <span style={{ fontSize: '11px', color: '#00d8a8', fontWeight: 700, background: 'rgba(0,216,168,0.1)', padding: '4px 10px', borderRadius: '8px' }}>
                                        ✅ Doğruluk: %{trainStatus.result.accuracy}
                                    </span>
                                )}
                                <button onClick={() => setTrainOpen(!trainOpen)} style={{ padding: '8px 16px', borderRadius: '10px', border: 'none', cursor: 'pointer', background: trainOpen ? 'rgba(249,115,22,0.15)' : 'linear-gradient(135deg, #f97316, #fb923c)', color: trainOpen ? '#f97316' : 'white', fontWeight: 700, fontSize: '12px' }}>
                                    {trainOpen ? '▲ Kapat' : '🚀 Modeli Eğit'}
                                </button>
                            </div>
                        </div>

                        {trainOpen && (
                            <div style={{ padding: '18px 20px' }}>
                                {/* Parameters */}
                                <div style={{ display: 'flex', gap: '12px', marginBottom: '14px', flexWrap: 'wrap' }}>
                                    <div>
                                        <label style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: '4px' }}>Timeframe</label>
                                        <select value={trainTf} onChange={e => setTrainTf(e.target.value)} style={{ padding: '7px 12px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)', fontSize: '12px', fontWeight: 600 }}>
                                            <option value="all">Tüm Zaman Dilimleri (Otonom)</option>
                                            <option value="1m">1m (Scalp)</option>
                                            <option value="5m">5m</option>
                                            <option value="15m">15m</option>
                                            <option value="1h">1h</option>
                                            <option value="4h">4h</option>
                                        </select>
                                    </div>
                                    <div>
                                        <label style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: '4px' }}>Mum Sayısı (Parite Başı)</label>
                                        <select value={trainLimit} onChange={e => setTrainLimit(Number(e.target.value))} style={{ padding: '7px 12px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)', fontSize: '12px', fontWeight: 600 }}>
                                            <option value={500}>500 (Hızlı)</option>
                                            <option value={1000}>1,000</option>
                                            <option value={1500}>1,500</option>
                                            <option value={3000}>3,000 (Önerilen)</option>
                                            <option value={5000}>5,000</option>
                                            <option value={10000}>10,000</option>
                                            <option value={20000}>20,000 (Derin Geçmiş)</option>
                                        </select>
                                    </div>
                                    {isMounted && (
                                        <div style={{ display: 'flex', alignItems: 'flex-end', gap: '8px' }}>
                                            <button onClick={handleStartTraining}
                                                disabled={trainStatus.status === 'fetching' || trainStatus.status === 'training'}
                                                style={{ padding: '8px 22px', borderRadius: '8px', border: 'none', cursor: 'pointer', background: (trainStatus.status === 'fetching' || trainStatus.status === 'training') ? 'rgba(249,115,22,0.2)' : 'linear-gradient(135deg, #f97316, #ea580c)', color: 'white', fontWeight: 700, fontSize: '12px', opacity: (trainStatus.status === 'fetching' || trainStatus.status === 'training') ? 0.6 : 1 }}>
                                                {(trainStatus.status === 'fetching' || trainStatus.status === 'training') ? '⏳ Eğitim Devam Ediyor...' : '▶ Eğitimi Başlat'}
                                            </button>
                                            {(trainStatus.status === 'fetching' || trainStatus.status === 'training' || trainStatus.status === 'error') && (
                                                <button onClick={handleResetTraining}
                                                    style={{ padding: '8px 14px', borderRadius: '8px', border: '1px solid rgba(244,63,94,0.3)', cursor: 'pointer', background: 'rgba(244,63,94,0.1)', color: '#f43f5e', fontWeight: 700, fontSize: '12px' }}
                                                    title="Takılı kalan eğitimi sıfırla">
                                                    ↺ Sıfırla
                                                </button>
                                            )}
                                        </div>
                                    )}
                                </div>

                                {/* Progress Bar */}
                                {isMounted && (trainStatus.status === 'fetching' || trainStatus.status === 'training') && (
                                    <div style={{ marginBottom: '14px' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                                            <span style={{ fontSize: '11px', color: '#f97316', fontWeight: 600 }}>{trainStatus.detail}</span>
                                            <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 700 }}>%{trainStatus.progress}</span>
                                        </div>
                                        <div style={{ width: '100%', height: '8px', borderRadius: '4px', background: 'rgba(249,115,22,0.1)', overflow: 'hidden' }}>
                                            <div style={{ width: `${trainStatus.progress}%`, height: '100%', borderRadius: '4px', background: 'linear-gradient(90deg, #f97316, #fb923c)', transition: 'width 0.5s ease' }} />
                                        </div>
                                    </div>
                                )}

                                {/* Results */}
                                {isMounted && trainStatus.status === 'done' && trainStatus.result && (
                                    <div style={{ background: 'rgba(0,216,168,0.04)', borderRadius: '12px', padding: '14px', border: '1px solid rgba(0,216,168,0.15)' }}>
                                        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '10px' }}>📊 Eğitim Sonuçları</div>
                                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginBottom: '12px' }}>
                                            {[
                                                { label: 'Doğruluk', value: `%${trainStatus.result.accuracy}`, color: '#00d8a8' },
                                                { label: 'Parite', value: trainStatus.result.symbols_count, color: '#f59e0b' },
                                                { label: 'Veri', value: `${(trainStatus.result.total_samples / 1000).toFixed(1)}K`, color: '#a78bfa' },
                                                { label: 'Timeframe', value: trainStatus.result.timeframe, color: '#3b82f6' },
                                            ].map((s, i) => (
                                                <div key={i} style={{ textAlign: 'center', padding: '8px', borderRadius: '8px', background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                                                    <div style={{ fontSize: '9px', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '3px', textTransform: 'uppercase' }}>{s.label}</div>
                                                    <div style={{ fontSize: '16px', fontWeight: 800, color: s.color }}>{s.value}</div>
                                                </div>
                                            ))}
                                        </div>
                                        {trainStatus.result.top_features && (
                                            <div>
                                                <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '6px' }}>EN ÖNEMLİ ÖZELLİKLER</div>
                                                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                                                    {trainStatus.result.top_features.slice(0, 6).map((f, i) => (
                                                        <span key={i} style={{ fontSize: '10px', padding: '3px 8px', borderRadius: '6px', background: 'rgba(249,115,22,0.1)', color: '#f97316', fontWeight: 600, border: '1px solid rgba(249,115,22,0.15)' }}>
                                                            {f.feature}: {(f.importance * 100).toFixed(1)}%
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}

                                {trainStatus.status === 'error' && (
                                    <div style={{ padding: '12px', borderRadius: '8px', background: 'rgba(244,63,94,0.1)', border: '1px solid rgba(244,63,94,0.2)', color: '#f43f5e', fontSize: '12px' }}>
                                        ❌ {trainStatus.detail}
                                    </div>
                                )}

                                <div style={{ display: 'flex', gap: '12px', marginTop: '16px', paddingTop: '16px', borderTop: '1px dashed rgba(249,115,22,0.2)' }}>
                                    <div style={{ flex: 1, padding: '12px', borderRadius: '10px', background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                                            <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)' }}>🗓 Oto-Eğitim (Haftalık)</div>
                                            <button onClick={toggleScheduler} style={{ padding: '4px 10px', borderRadius: '6px', fontSize: '10px', fontWeight: 700, border: 'none', cursor: 'pointer', background: schedStatus.status === 'running' ? 'rgba(244,63,94,0.15)' : 'rgba(0,216,168,0.15)', color: schedStatus.status === 'running' ? '#f43f5e' : '#00d8a8' }}>
                                                {schedStatus.status === 'running' ? 'Durdur' : 'Başlat'}
                                            </button>
                                        </div>
                                        <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                                            {schedStatus.status === 'running'
                                                ? <span style={{ color: '#00d8a8' }}>Aktif (Sonraki: {new Date(schedStatus.next_run || '').toLocaleString('tr-TR')})</span>
                                                : 'Kapalı (Model sadece manuel güncellenir)'}
                                        </div>
                                    </div>

                                    <div style={{ flex: 1, padding: '12px', borderRadius: '10px', background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                                        <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '6px' }}>🧠 Ensemble Modu (M-TF)</div>
                                        <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                                            {ensembStatus.ensemble_active
                                                ? <span style={{ color: '#fbbf24' }}>Aktif ({ensembStatus.total_models} model ağırlıklı oy)</span>
                                                : 'Kapalı (Sadece tek timeframe modeli aktif)'}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))', gap: '16px' }}>
                        {(bots.length > 0 ? bots : [
                            { id: 'quant-grid', name: '⚡ Quant Grid Scalper', symbol: 'Top 50 USDT', timeframe: '15m', strategy: 'quant_grid', status: 'stopped', balance: 1000, initial_balance: 1000, total_pnl: 0, total_pnl_pct: 0, today_pnl: 0, win_rate: 0, total_trades: 0, open_trades_count: 0, unrealized_pnl: 0, recent_logs: [], open_trades: [], closed_trades: [] },
                            { id: 'turtle-trend', name: '🐢 Turtle Trend Follower', symbol: 'Top 50 USDT', timeframe: '1h', strategy: 'turtle', status: 'stopped', balance: 1000, initial_balance: 1000, total_pnl: 0, total_pnl_pct: 0, today_pnl: 0, win_rate: 0, total_trades: 0, open_trades_count: 0, unrealized_pnl: 0, recent_logs: [], open_trades: [], closed_trades: [] },
                            { id: 'adaptive-regime', name: '🧠 Regime Adaptive Bot', symbol: 'Top 50 USDT', timeframe: '1h', strategy: 'adaptive', status: 'stopped', balance: 1000, initial_balance: 1000, total_pnl: 0, total_pnl_pct: 0, today_pnl: 0, win_rate: 0, total_trades: 0, open_trades_count: 0, unrealized_pnl: 0, recent_logs: [], open_trades: [], closed_trades: [] },
                            { id: 'market-maker', name: '🎯 Market Maker / Sniper', symbol: 'Top 50 USDT', timeframe: '15m', strategy: 'market_maker', status: 'stopped', balance: 1000, initial_balance: 1000, total_pnl: 0, total_pnl_pct: 0, today_pnl: 0, win_rate: 0, total_trades: 0, open_trades_count: 0, unrealized_pnl: 0, recent_logs: [], open_trades: [], closed_trades: [] },
                            { id: 'smart-money', name: '🏦 Smart Money + Market Maker', symbol: 'Top 50 USDT', timeframe: '15m', strategy: 'smart_money', status: 'stopped', balance: 1000, initial_balance: 1000, total_pnl: 0, total_pnl_pct: 0, today_pnl: 0, win_rate: 0, total_trades: 0, open_trades_count: 0, unrealized_pnl: 0, recent_logs: [], open_trades: [], closed_trades: [] },
                            { id: 'director-1m', name: '🎩 Direktör AI — 1m Scalper', symbol: 'Top 50 USDT', timeframe: '1m', strategy: 'director', status: 'stopped', balance: 1000, initial_balance: 1000, total_pnl: 0, total_pnl_pct: 0, today_pnl: 0, win_rate: 0, total_trades: 0, open_trades_count: 0, unrealized_pnl: 0, recent_logs: [], open_trades: [], closed_trades: [] },
                            { id: 'director-5m', name: '🎩 Direktör AI — 5m Momentum', symbol: 'Top 50 USDT', timeframe: '5m', strategy: 'director', status: 'stopped', balance: 1000, initial_balance: 1000, total_pnl: 0, total_pnl_pct: 0, today_pnl: 0, win_rate: 0, total_trades: 0, open_trades_count: 0, unrealized_pnl: 0, recent_logs: [], open_trades: [], closed_trades: [] },
                            { id: 'director-15m', name: '🎩 Direktör AI — 15m Swing', symbol: 'Top 50 USDT', timeframe: '15m', strategy: 'director', status: 'stopped', balance: 1000, initial_balance: 1000, total_pnl: 0, total_pnl_pct: 0, today_pnl: 0, win_rate: 0, total_trades: 0, open_trades_count: 0, unrealized_pnl: 0, recent_logs: [], open_trades: [], closed_trades: [] },
                            { id: 'director-1h', name: '🎩 Direktör AI — 1h Trend', symbol: 'Top 50 USDT', timeframe: '1h', strategy: 'director', status: 'stopped', balance: 1000, initial_balance: 1000, total_pnl: 0, total_pnl_pct: 0, today_pnl: 0, win_rate: 0, total_trades: 0, open_trades_count: 0, unrealized_pnl: 0, recent_logs: [], open_trades: [], closed_trades: [] },
                            { id: 'director-4h', name: '🎩 Direktör AI — 4h Position', symbol: 'Top 50 USDT', timeframe: '4h', strategy: 'director', status: 'stopped', balance: 1000, initial_balance: 1000, total_pnl: 0, total_pnl_pct: 0, today_pnl: 0, win_rate: 0, total_trades: 0, open_trades_count: 0, unrealized_pnl: 0, recent_logs: [], open_trades: [], closed_trades: [] },
                        ] as BotStatus[]).map(bot => {
                            const color = BOT_COLORS[bot.id] || '#00d8a8';
                            const isRunning = bot.status === 'running';
                            return (
                                <div key={bot.id} className="glass" style={{ borderRadius: '16px', overflow: 'hidden', border: isRunning ? `1px solid ${color}40` : '1px solid var(--border)' }}>
                                    <div style={{ padding: '14px 18px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: isRunning ? `${color}08` : 'transparent', borderBottom: '1px solid var(--border)' }}>
                                        <div>
                                            <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '2px' }}>{bot.name}</div>
                                            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{bot.symbol} · {bot.timeframe} · <span style={{ color }}>{STRATEGY_LABELS[bot.strategy] || bot.strategy}</span></div>
                                        </div>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                                            {isRunning ? <span className="status-dot status-live" /> : <span className="status-dot status-idle" />}
                                            <span style={{ fontSize: '11px', fontWeight: 600, color: isRunning ? 'var(--accent-green)' : 'var(--text-muted)' }}>
                                                {isRunning ? 'Canlı' : 'Durduruldu'}
                                            </span>
                                        </div>
                                    </div>

                                    <div style={{ padding: '14px 18px' }}>
                                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '12px' }}>
                                            {[
                                                { label: 'Kapalı PnL', value: `${fmt(bot.total_pnl)}$`, sub: `${fmt(bot.total_pnl_pct)}%`, pos: bot.total_pnl >= 0 },
                                                { label: 'Açık PnL', value: `${fmt(bot.unrealized_pnl ?? 0)}$`, sub: 'anlık', pos: (bot.unrealized_pnl ?? 0) >= 0 },
                                                { label: 'Win Rate', value: `${bot.win_rate}%`, sub: `${bot.total_trades} işlem`, pos: true },
                                                { label: 'Bakiye', value: `$${bot.balance.toFixed(2)}`, sub: `${bot.open_trades_count} açık`, pos: bot.balance >= bot.initial_balance },
                                            ].map((s, i) => (
                                                <div key={i} style={{ padding: '9px 11px', borderRadius: '9px', background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                                                    <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '3px', fontWeight: 600 }}>{s.label}</div>
                                                    <div style={{ fontSize: '15px', fontWeight: 800, color: s.pos ? 'var(--accent-green)' : 'var(--accent-red)' }}>{s.value}</div>
                                                    <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '1px' }}>{s.sub}</div>
                                                </div>
                                            ))}
                                        </div>

                                        {bot.recent_logs?.length > 0 && (
                                            <div style={{ marginBottom: '10px', padding: '9px 11px', borderRadius: '8px', background: 'rgba(0,0,0,0.2)', border: '1px solid var(--border)', maxHeight: '64px', overflow: 'hidden' }}>
                                                {bot.recent_logs.slice(0, 3).map(log => (
                                                    <div key={log.id} style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '2px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{log.text}</div>
                                                ))}
                                            </div>
                                        )}

                                        <div style={{ display: 'flex', gap: '7px' }}>
                                            <button onClick={() => handleToggle(bot)} style={{ flex: 1, padding: '8px', borderRadius: '8px', border: 'none', cursor: 'pointer', fontWeight: 700, fontSize: '12px', background: isRunning ? 'rgba(244,63,94,0.1)' : `${color}18`, color: isRunning ? 'var(--accent-red)' : color }}>
                                                {isRunning ? '⏸ Durdur' : '▶ Başlat'}
                                            </button>
                                            <button onClick={() => handleOpenSettings(bot)} style={{ padding: '8px 11px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-muted)', fontSize: '12px', cursor: 'pointer' }} title="Ayarlar">⚙️</button>
                                            <button onClick={() => setPnlDetailsOpen(bot)} style={{ padding: '8px 11px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-muted)', fontSize: '12px', cursor: 'pointer' }} title="Detaylı PnL">📊</button>
                                            <button onClick={() => setSelectedLog(bot)} style={{ padding: '8px 11px', borderRadius: '8px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-muted)', fontSize: '12px', cursor: 'pointer' }}>📋</button>
                                            <button onClick={() => handleReset(bot.id)} style={{ padding: '8px 11px', borderRadius: '8px', border: '1px solid rgba(244,63,94,0.2)', background: 'transparent', color: '#f43f5e', fontSize: '12px', cursor: 'pointer' }} title="Sıfırla">↺</button>
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </>
            )}

            {/* ── TAB: AÇIK POZİSYONLAR ────────────────────────── */}
            {tab === 'positions' && (
                <div>
                    {allOpen.length === 0 ? (
                        <div className="glass" style={{ padding: '60px', textAlign: 'center', borderRadius: '16px' }}>
                            <div style={{ fontSize: '40px', marginBottom: '12px' }}>📭</div>
                            <div style={{ color: 'var(--text-muted)', fontSize: '14px' }}>Şu an açık pozisyon yok</div>
                            <div style={{ color: 'var(--text-muted)', fontSize: '12px', marginTop: '6px' }}>Botlar sinyal ürettiğinde burada görünür</div>
                        </div>
                    ) : (
                        <>
                            {/* Summary row */}
                            <div style={{ display: 'flex', gap: '12px', marginBottom: '16px' }}>
                                <div className="glass" style={{ padding: '12px 18px', borderRadius: '10px', flex: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: 600 }}>Toplam Açık Pozisyon</span>
                                    <span style={{ fontSize: '18px', fontWeight: 800, color: '#fbbf24' }}>{allOpen.length}</span>
                                </div>
                                <div className="glass" style={{ padding: '12px 18px', borderRadius: '10px', flex: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: 600 }}>Toplam Açık PnL</span>
                                    <span style={{ fontSize: '18px', fontWeight: 800, color: totalUnrealized >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                        {fmt(totalUnrealized)}$
                                    </span>
                                </div>
                                <div className="glass" style={{ padding: '12px 18px', borderRadius: '10px', flex: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: 600 }}>Karda Pozisyon</span>
                                    <span style={{ fontSize: '18px', fontWeight: 800, color: 'var(--accent-green)' }}>
                                        {allOpen.filter(t => t.unrealized_pnl > 0).length}/{allOpen.length}
                                    </span>
                                </div>
                            </div>

                            {/* Positions Table */}
                            <div className="glass" style={{ borderRadius: '14px', overflow: 'hidden' }}>
                                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                    <thead>
                                        <tr style={{ borderBottom: '1px solid var(--border)', background: 'rgba(0,0,0,0.2)' }}>
                                            {['Bot', 'Sembol', 'Yön', 'Miktar(Margin)', 'Giriş Fiyatı', 'Anlık Fiyat', 'Değişim', 'Unrealized PnL', 'Giriş'].map(h => (
                                                <th key={h} style={{ padding: '11px 14px', textAlign: 'left', fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', whiteSpace: 'nowrap' }}>{h}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {allOpen.map((t, i) => {
                                            const pnlPos = t.unrealized_pnl >= 0;
                                            const pricePct = ((t.current_price - t.entry_price) / t.entry_price) * 100;
                                            const color = BOT_COLORS[t.botId] || '#00d8a8';
                                            return (
                                                <tr key={t.id + i} style={{ borderBottom: '1px solid var(--border)', transition: 'background 0.15s' }}
                                                    onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.02)')}
                                                    onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
                                                    {/* Bot */}
                                                    <td style={{ padding: '12px 14px' }}>
                                                        <span style={{ fontSize: '10px', fontWeight: 700, color, background: `${color}15`, padding: '2px 7px', borderRadius: '5px' }}>
                                                            {t.botId.split('-').map(w => w[0].toUpperCase() + w.slice(1)).join(' ').slice(0, 8)}
                                                        </span>
                                                    </td>
                                                    {/* Sembol */}
                                                    <td style={{ padding: '12px 14px', fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', whiteSpace: 'nowrap' }}>{t.symbol}</td>
                                                    {/* Yön */}
                                                    <td style={{ padding: '12px 14px' }}>
                                                        <span style={{
                                                            fontSize: '11px', fontWeight: 700, padding: '3px 9px', borderRadius: '6px',
                                                            background: t.side === 'LONG' ? 'rgba(0,216,168,0.12)' : 'rgba(244,63,94,0.12)',
                                                            color: t.side === 'LONG' ? '#00d8a8' : '#f43f5e',
                                                        }}>
                                                            {t.side === 'LONG' ? '▲ LONG' : '▼ SHORT'}
                                                        </span>
                                                    </td>
                                                    {/* Margin */}
                                                    <td style={{ padding: '12px 14px', fontSize: '12px', color: 'var(--text-secondary)' }}>${t.margin.toFixed(2)}</td>
                                                    {/* Entry */}
                                                    <td style={{ padding: '12px 14px', fontSize: '12px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{fmtPrice(t.entry_price)}</td>
                                                    {/* Current */}
                                                    <td style={{ padding: '12px 14px', fontSize: '12px', color: 'var(--text-primary)', fontFamily: 'monospace', fontWeight: 600 }}>{fmtPrice(t.current_price)}</td>
                                                    {/* Değişim % */}
                                                    <td style={{ padding: '12px 14px' }}>
                                                        <span style={{ fontSize: '12px', fontWeight: 700, color: pricePct >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                                            {pricePct >= 0 ? '▲' : '▼'} {Math.abs(pricePct).toFixed(3)}%
                                                        </span>
                                                    </td>
                                                    {/* Unrealized PnL */}
                                                    <td style={{ padding: '12px 14px' }}>
                                                        <div style={{ fontSize: '13px', fontWeight: 800, color: pnlPos ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                                            {fmt(t.unrealized_pnl)}$
                                                        </div>
                                                        <div style={{ fontSize: '10px', color: pnlPos ? 'var(--accent-green)' : 'var(--accent-red)', opacity: 0.7 }}>
                                                            {fmt(t.unrealized_pnl_pct)}%
                                                        </div>
                                                    </td>
                                                    {/* Zaman */}
                                                    <td style={{ padding: '12px 14px', fontSize: '11px', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>{t.entry_time}</td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        </>
                    )}
                </div>
            )}

            {/* ── TAB: İŞLEM GEÇMİŞİ ──────────────────────────── */}
            {tab === 'history' && (() => {
                // Apply filters
                const uniqueBots = Array.from(new Set(allClosed.map(t => t.botId)));
                const filteredTrades = allClosed.filter(t => {
                    if (filterBot !== 'all' && t.botId !== filterBot) return false;
                    if (filterSide !== 'all' && t.side !== filterSide) return false;
                    if (filterResult === 'win' && t.pnl <= 0) return false;
                    if (filterResult === 'loss' && t.pnl > 0) return false;
                    if (filterSymbol && !t.symbol.toLowerCase().includes(filterSymbol.toLowerCase())) return false;
                    return true;
                });
                const activeFilters = [filterBot !== 'all', filterSide !== 'all', filterResult !== 'all', filterSymbol !== ''].filter(Boolean).length;
                const filteredPnl = filteredTrades.reduce((s, t) => s + t.pnl, 0);

                return (
                    <div>
                        {allClosed.length === 0 ? (
                            <div className="glass" style={{ padding: '60px', textAlign: 'center', borderRadius: '16px' }}>
                                <div style={{ fontSize: '40px', marginBottom: '12px' }}>📭</div>
                                <div style={{ color: 'var(--text-muted)', fontSize: '14px' }}>Henüz kapatılmış işlem yok</div>
                            </div>
                        ) : (
                            <>
                                {/* Stats row - based on filtered trades */}
                                <div style={{ display: 'flex', gap: '12px', marginBottom: '16px' }}>
                                    {[
                                        { label: 'Toplam İşlem', value: `${filteredTrades.length}${activeFilters > 0 ? ` / ${allClosed.length}` : ''}`, color: '#fbbf24' },
                                        { label: 'Karda', value: filteredTrades.filter(t => t.pnl > 0).length, color: 'var(--accent-green)' },
                                        { label: 'Zararda', value: filteredTrades.filter(t => t.pnl <= 0).length, color: 'var(--accent-red)' },
                                        { label: 'Toplam PnL', value: `${fmt(filteredPnl)}$`, color: filteredPnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' },
                                    ].map((s, i) => (
                                        <div key={i} className="glass" style={{ padding: '12px 18px', borderRadius: '10px', flex: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                            <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600 }}>{s.label}</span>
                                            <span style={{ fontSize: '18px', fontWeight: 800, color: s.color }}>{s.value}</span>
                                        </div>
                                    ))}
                                </div>

                                {/* Filter Bar */}
                                <div className="glass" style={{ borderRadius: '12px', padding: '12px 16px', marginBottom: '14px', display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
                                    <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 700, marginRight: '4px' }}>🔍 Filtrele:</span>

                                    {/* Symbol Search */}
                                    <input
                                        type="text"
                                        placeholder="Sembol ara..."
                                        value={filterSymbol}
                                        onChange={e => setFilterSymbol(e.target.value)}
                                        style={{
                                            padding: '6px 12px', borderRadius: '8px', border: '1px solid var(--border)',
                                            background: 'var(--bg-card)', color: 'var(--text-primary)', fontSize: '12px',
                                            outline: 'none', width: '130px',
                                        }}
                                    />

                                    {/* Bot filter */}
                                    <select value={filterBot} onChange={e => setFilterBot(e.target.value)}
                                        style={{ padding: '6px 10px', borderRadius: '8px', border: '1px solid var(--border)', background: 'var(--bg-card)', color: 'var(--text-primary)', fontSize: '12px', fontWeight: 600, cursor: 'pointer', outline: 'none' }}>
                                        <option value="all">Tüm Botlar</option>
                                        {uniqueBots.map(id => (
                                            <option key={id} value={id}>{id.split('-').map((w: string) => w[0].toUpperCase() + w.slice(1)).join(' ')}</option>
                                        ))}
                                    </select>

                                    {/* Side filter */}
                                    {(['all', 'LONG', 'SHORT'] as const).map(s => (
                                        <button key={s} onClick={() => setFilterSide(s)} style={{
                                            padding: '5px 12px', borderRadius: '8px', fontSize: '11px', fontWeight: 700, cursor: 'pointer', border: 'none',
                                            background: filterSide === s
                                                ? (s === 'LONG' ? 'rgba(0,216,168,0.2)' : s === 'SHORT' ? 'rgba(244,63,94,0.2)' : 'rgba(79,158,255,0.2)')
                                                : 'rgba(255,255,255,0.05)',
                                            color: filterSide === s
                                                ? (s === 'LONG' ? '#00d8a8' : s === 'SHORT' ? '#f43f5e' : '#4f9eff')
                                                : 'var(--text-muted)',
                                        }}>
                                            {s === 'all' ? 'L+S' : s === 'LONG' ? '▲ Long' : '▼ Short'}
                                        </button>
                                    ))}

                                    {/* Result filter */}
                                    {[{ k: 'all', label: 'Tümü' }, { k: 'win', label: '✅ Karda' }, { k: 'loss', label: '❌ Zararda' }].map(r => (
                                        <button key={r.k} onClick={() => setFilterResult(r.k)} style={{
                                            padding: '5px 12px', borderRadius: '8px', fontSize: '11px', fontWeight: 700, cursor: 'pointer', border: 'none',
                                            background: filterResult === r.k ? 'rgba(79,158,255,0.2)' : 'rgba(255,255,255,0.05)',
                                            color: filterResult === r.k ? '#4f9eff' : 'var(--text-muted)',
                                        }}>
                                            {r.label}
                                        </button>
                                    ))}

                                    {/* Active filter badge + reset */}
                                    {activeFilters > 0 && (
                                        <button onClick={() => { setFilterBot('all'); setFilterSide('all'); setFilterResult('all'); setFilterSymbol(''); }}
                                            style={{ marginLeft: 'auto', padding: '5px 12px', borderRadius: '8px', fontSize: '11px', fontWeight: 700, cursor: 'pointer', border: '1px solid rgba(244,63,94,0.3)', background: 'rgba(244,63,94,0.08)', color: '#f43f5e' }}>
                                            ✕ Temizle ({activeFilters})
                                        </button>
                                    )}
                                </div>

                                <div className="glass" style={{ borderRadius: '14px', overflow: 'hidden' }}>
                                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                        <thead>
                                            <tr style={{ borderBottom: '1px solid var(--border)', background: 'rgba(0,0,0,0.2)' }}>
                                                {['Bot', 'Sembol', 'Yön', 'Margin', 'Giriş', 'Çıkış', 'PnL', 'PnL %', 'Neden', 'Çıkış Zamanı'].map(h => (
                                                    <th key={h} style={{ padding: '11px 12px', textAlign: 'left', fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', whiteSpace: 'nowrap' }}>{h}</th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {filteredTrades.length === 0 ? (
                                                <tr>
                                                    <td colSpan={10} style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
                                                        Bu filtreyle eşleşen işlem bulunamadı
                                                    </td>
                                                </tr>
                                            ) : filteredTrades.map((t, i) => {
                                                const win = t.pnl > 0;
                                                const color = BOT_COLORS[t.botId] || '#00d8a8';
                                                const pnlPct = t.margin > 0 ? (t.pnl / t.margin) * 100 : 0;
                                                return (
                                                    <tr key={t.id + i} style={{ borderBottom: '1px solid var(--border)' }}
                                                        onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.02)')}
                                                        onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
                                                        <td style={{ padding: '10px 12px' }}>
                                                            <span style={{ fontSize: '10px', fontWeight: 700, color, background: `${color}15`, padding: '2px 6px', borderRadius: '5px' }}>
                                                                {t.botId.split('-').map(w => w[0].toUpperCase() + w.slice(1)).join(' ').slice(0, 8)}
                                                            </span>
                                                        </td>
                                                        <td style={{ padding: '10px 12px', fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)', whiteSpace: 'nowrap' }}>{t.symbol}</td>
                                                        <td style={{ padding: '10px 12px' }}>
                                                            <span style={{ fontSize: '10px', fontWeight: 700, padding: '2px 7px', borderRadius: '5px', background: t.side === 'LONG' ? 'rgba(0,216,168,0.12)' : 'rgba(244,63,94,0.12)', color: t.side === 'LONG' ? '#00d8a8' : '#f43f5e' }}>
                                                                {t.side === 'LONG' ? '▲ L' : '▼ S'}
                                                            </span>
                                                        </td>
                                                        <td style={{ padding: '10px 12px', fontSize: '11px', color: 'var(--text-secondary)' }}>${t.margin?.toFixed(2) || '–'}</td>
                                                        <td style={{ padding: '10px 12px', fontSize: '11px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{fmtPrice(t.entry_price)}</td>
                                                        <td style={{ padding: '10px 12px', fontSize: '11px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{fmtPrice(t.exit_price)}</td>
                                                        <td style={{ padding: '10px 12px' }}>
                                                            <span style={{ fontSize: '13px', fontWeight: 800, color: win ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                                                {win ? '✅' : '❌'} {fmt(t.pnl)}$
                                                            </span>
                                                        </td>
                                                        <td style={{ padding: '10px 12px' }}>
                                                            <span style={{ fontSize: '11px', fontWeight: 700, color: win ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                                                {fmt(pnlPct)}%
                                                            </span>
                                                        </td>
                                                        <td style={{ padding: '10px 12px' }}>
                                                            <span style={{ fontSize: '10px', padding: '2px 6px', borderRadius: '4px', background: 'var(--bg-card)', color: 'var(--text-muted)', border: '1px solid var(--border)', whiteSpace: 'nowrap' }}>
                                                                {t.reason || '–'}
                                                            </span>
                                                        </td>
                                                        <td style={{ padding: '10px 12px', fontSize: '10px', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>{t.exit_time || '–'}</td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                            </>
                        )}
                    </div>
                );
            })()}

            {/* Log Modal */}
            {selectedLog && (
                <div style={{ position: 'fixed', inset: 0, zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(4,13,26,0.85)', backdropFilter: 'blur(8px)' }}
                    onClick={() => setSelectedLog(null)}>
                    <div className="glass" style={{ width: '520px', maxHeight: '70vh', borderRadius: '18px', overflow: 'hidden' }} onClick={e => e.stopPropagation()}>
                        <div style={{ padding: '16px 22px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between' }}>
                            <div>
                                <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>{selectedLog.name} — Loglar</div>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>{selectedLog.symbol} · {selectedLog.timeframe}</div>
                            </div>
                            <button onClick={() => setSelectedLog(null)} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '18px' }}>✕</button>
                        </div>
                        <div style={{ padding: '14px 22px', overflow: 'auto', maxHeight: 'calc(70vh - 70px)' }}>
                            {selectedLog.recent_logs.length === 0
                                ? <div style={{ color: 'var(--text-muted)', fontSize: '13px', textAlign: 'center', padding: '30px' }}>Henüz log yok</div>
                                : selectedLog.recent_logs.map(log => (
                                    <div key={log.id} style={{
                                        fontSize: '11px', padding: '5px 0', borderBottom: '1px solid var(--border)',
                                        color: log.text.includes('✅') ? '#00d8a8' : log.text.includes('❌') ? '#f43f5e' : log.text.includes('🟢') ? '#4ade80' : 'var(--text-muted)'
                                    }}>{log.text}</div>
                                ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Settings Modal */}
            {settingsOpen && (
                <div style={{ position: 'fixed', inset: 0, zIndex: 110, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(4,13,26,0.85)', backdropFilter: 'blur(8px)' }}
                    onClick={() => setSettingsOpen(null)}>
                    <div className="glass" style={{ width: '400px', borderRadius: '18px', overflow: 'hidden' }} onClick={e => e.stopPropagation()}>
                        <div style={{ padding: '16px 22px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between' }}>
                            <div>
                                <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>{settingsOpen.name} — Ayarlar</div>
                                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>Değişiklikler anında uygulanır</div>
                            </div>
                            <button onClick={() => setSettingsOpen(null)} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '18px' }}>✕</button>
                        </div>
                        <div style={{ padding: '20px 22px' }}>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                                <div>
                                    <label style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: '6px' }}>Zaman Dilimi (Timeframe)</label>
                                    <select value={settingsForm.timeframe} onChange={e => setSettingsForm({ ...settingsForm, timeframe: e.target.value })}
                                        style={{ width: '100%', padding: '10px 14px', borderRadius: '10px', background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)', fontSize: '13px', outline: 'none' }}>
                                        <option value="1m">1m (Skalping)</option>
                                        <option value="5m">5m (Agresif)</option>
                                        <option value="15m">15m (Dengeli)</option>
                                        <option value="1h">1h (Momentum)</option>
                                        <option value="4h">4h (Trend)</option>
                                    </select>
                                </div>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                                    <div>
                                        <label style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: '6px' }}>Kaldıraç (x)</label>
                                        <input type="number" min="1" max="125" value={settingsForm.leverage} onChange={e => setSettingsForm({ ...settingsForm, leverage: Number(e.target.value) })}
                                            style={{ width: '100%', padding: '10px 14px', borderRadius: '10px', background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)', fontSize: '13px', outline: 'none' }} />
                                    </div>
                                    <div>
                                        <label style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: '6px' }}>Risk (% / İşlem)</label>
                                        <input type="number" min="0.1" max="100" step="0.1" value={settingsForm.risk_pct} onChange={e => setSettingsForm({ ...settingsForm, risk_pct: Number(e.target.value) })}
                                            style={{ width: '100%', padding: '10px 14px', borderRadius: '10px', background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)', fontSize: '13px', outline: 'none' }} />
                                    </div>
                                    <div>
                                        <label style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: '6px' }}>TP (ATR Çarpanı)</label>
                                        <input type="number" min="0.1" max="10" step="0.1" value={settingsForm.atr_tp_mult} onChange={e => setSettingsForm({ ...settingsForm, atr_tp_mult: Number(e.target.value) })}
                                            style={{ width: '100%', padding: '10px 14px', borderRadius: '10px', background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)', fontSize: '13px', outline: 'none' }} />
                                    </div>
                                    <div>
                                        <label style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: '6px' }}>SL (ATR Çarpanı)</label>
                                        <input type="number" min="0.1" max="10" step="0.1" value={settingsForm.atr_sl_mult} onChange={e => setSettingsForm({ ...settingsForm, atr_sl_mult: Number(e.target.value) })}
                                            style={{ width: '100%', padding: '10px 14px', borderRadius: '10px', background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)', fontSize: '13px', outline: 'none' }} />
                                    </div>
                                </div>
                                {settingsOpen.id === 'xgboost-ai' && (
                                    <div>
                                        <label style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: '6px' }}>Min Confidence (0.0 - 1.0)</label>
                                        <input type="number" min="0.4" max="0.99" step="0.01" value={settingsForm.min_confidence} onChange={e => setSettingsForm({ ...settingsForm, min_confidence: Number(e.target.value) })}
                                            style={{ width: '100%', padding: '10px 14px', borderRadius: '10px', background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)', fontSize: '13px', outline: 'none' }} />
                                    </div>
                                )}
                            </div>
                            <button onClick={handleSaveSettings} style={{ width: '100%', marginTop: '24px', padding: '12px', borderRadius: '10px', border: 'none', cursor: 'pointer', background: 'linear-gradient(135deg, #00d8a8, #4f9eff)', color: 'white', fontWeight: 700, fontSize: '14px' }}>
                                Ayarları Kaydet
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* PnL Details Modal */}
            {pnlDetailsOpen && (() => {
                const bot = pnlDetailsOpen;
                const closed = bot.closed_trades || [];
                const wins = closed.filter(t => t.pnl > 0);
                const losses = closed.filter(t => t.pnl <= 0);

                const grossProfit = wins.reduce((sum, t) => sum + t.pnl, 0);
                const grossLoss = losses.reduce((sum, t) => sum + t.pnl, 0);
                const netPnl = grossProfit + grossLoss;
                const winRate = closed.length > 0 ? ((wins.length / closed.length) * 100).toFixed(1) : '0.0';

                return (
                    <div style={{ position: 'fixed', inset: 0, zIndex: 120, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(4,13,26,0.85)', backdropFilter: 'blur(8px)' }}
                        onClick={() => setPnlDetailsOpen(null)}>
                        <div className="glass" style={{ width: '480px', borderRadius: '18px', overflow: 'hidden' }} onClick={e => e.stopPropagation()}>
                            <div style={{ padding: '16px 22px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div>
                                    <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)' }}>{bot.name} — Detaylı PnL Analizi</div>
                                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>{bot.symbol} · {STRATEGY_LABELS[bot.strategy] || bot.strategy}</div>
                                </div>
                                <button onClick={() => setPnlDetailsOpen(null)} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '18px' }}>✕</button>
                            </div>

                            <div style={{ padding: '20px 22px' }}>
                                {/* Overall Summary Cards */}
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', marginBottom: '16px' }}>
                                    <div style={{ padding: '12px', borderRadius: '10px', background: 'var(--bg-card)', border: '1px solid var(--border)', textAlign: 'center' }}>
                                        <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: '4px' }}>Net PnL</div>
                                        <div style={{ fontSize: '18px', fontWeight: 800, color: netPnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>{fmt(netPnl)}$</div>
                                    </div>
                                    <div style={{ padding: '12px', borderRadius: '10px', background: 'var(--bg-card)', border: '1px solid var(--border)', textAlign: 'center' }}>
                                        <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: '4px' }}>Win Rate</div>
                                        <div style={{ fontSize: '18px', fontWeight: 800, color: 'var(--accent-blue)' }}>%{winRate}</div>
                                    </div>
                                    <div style={{ padding: '12px', borderRadius: '10px', background: 'var(--bg-card)', border: '1px solid var(--border)', textAlign: 'center' }}>
                                        <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: '4px' }}>İşlem Sayısı</div>
                                        <div style={{ fontSize: '18px', fontWeight: 800, color: 'var(--text-primary)' }}>{closed.length}</div>
                                    </div>
                                </div>

                                {/* Deep Dive Metrics */}
                                <div style={{ borderRadius: '12px', border: '1px solid var(--border)', overflow: 'hidden', marginBottom: '20px' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 16px', background: 'rgba(255,255,255,0.02)', borderBottom: '1px solid var(--border)' }}>
                                        <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Kârlı İşlem Sayısı (Wins)</span>
                                        <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--accent-green)' }}>{wins.length} İşlem</span>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 16px', borderBottom: '1px solid var(--border)' }}>
                                        <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Brüt Kâr (Gross Profit)</span>
                                        <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--accent-green)' }}>+{grossProfit.toFixed(2)}$</span>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 16px', background: 'rgba(255,255,255,0.02)', borderBottom: '1px solid var(--border)' }}>
                                        <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Zararlı İşlem Sayısı (Losses)</span>
                                        <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--accent-red)' }}>{losses.length} İşlem</span>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 16px' }}>
                                        <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Brüt Zarar (Gross Loss)</span>
                                        <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--accent-red)' }}>{grossLoss.toFixed(2)}$</span>
                                    </div>
                                </div>

                                {/* Recent 5 Trades Mini-Table */}
                                <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '8px' }}>Son 5 İşlem Özeti</div>
                                {closed.length === 0 ? (
                                    <div style={{ fontSize: '12px', color: 'var(--text-muted)', textAlign: 'center', padding: '20px', background: 'var(--bg-card)', borderRadius: '10px', border: '1px solid var(--border)' }}>
                                        Henüz kapatılmış işlem bulunmuyor
                                    </div>
                                ) : (
                                    <div style={{ borderRadius: '10px', border: '1px solid var(--border)', overflow: 'hidden' }}>
                                        {closed.slice(0, 5).map((t, i) => (
                                            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 14px', borderBottom: i !== 4 && i !== closed.length - 1 ? '1px solid var(--border)' : 'none', background: i % 2 === 0 ? 'rgba(255,255,255,0.02)' : 'transparent' }}>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                    <span style={{ fontSize: '10px', fontWeight: 700, color: t.side === 'LONG' ? '#00d8a8' : '#f43f5e', background: t.side === 'LONG' ? 'rgba(0,216,168,0.1)' : 'rgba(244,63,94,0.1)', padding: '2px 6px', borderRadius: '4px' }}>
                                                        {t.side}
                                                    </span>
                                                    <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>{t.exit_time || t.entry_time}</span>
                                                </div>
                                                <div style={{ fontSize: '12px', fontWeight: 700, color: t.pnl > 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                                    {t.pnl > 0 ? '+' : ''}{t.pnl.toFixed(2)}$
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                );
            })()}
        </div>
    );
}
