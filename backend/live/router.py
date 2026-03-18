import os
import time

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, List
from live_trader import LivePaperTrader

_DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

router = APIRouter(prefix="/live", tags=["Live Trading Engine"])

# ─────────────────────────────────────────────────────────────────────────────
# Quant AI / Adaptive Auto-Trader Endpoints
# ─────────────────────────────────────────────────────────────────────────────

quant_trader = LivePaperTrader()

@router.post("/quant/start")
def start_quant_trader():
    """Tüm marketi tarayan AI Auto-Trader'ı başlatır."""
    if quant_trader.is_running:
        return {"success": False, "message": "Auto-Trader is already running"}
    quant_trader.start()
    return {"success": True, "message": "Auto-Trader started successfully."}

@router.post("/quant/stop")
def stop_quant_trader():
    """Auto-Trader'ı durdurur."""
    if not quant_trader.is_running:
        return {"success": False, "message": "Auto-Trader is not running"}
    quant_trader.stop()
    return {"success": True, "message": "Auto-Trader stopped successfully."}

@router.get("/quant/status")
def status_quant_trader():
    """Auto-Trader anlık durumunu döndürür."""
    return quant_trader.get_status()

@router.post("/quant/close-trade/{trade_id}")
def close_quant_trade(trade_id: str):
    """Belirli bir işlem ID'sini (trade_id) manuel olarak kapatır."""
    result = quant_trader.close_trade(trade_id)
    return result

class QuantSettingsRequest(BaseModel):
    max_open_trades: Optional[int] = Field(None, ge=1, le=30)
    max_notional: Optional[float] = Field(None, ge=10, le=5000)
    max_loss_cap: Optional[float] = Field(None, ge=1, le=100)
    min_rr: Optional[float] = Field(None, ge=1.0, le=5.0)
    tp_min: Optional[float] = Field(None, ge=0.1, le=10.0)
    balance: Optional[float] = Field(None, ge=100, le=1000000)
    # Strategy enable/disable
    bb_mr_enabled: Optional[bool] = None
    ict_smc_enabled: Optional[bool] = None
    # ICT/SMC params
    ict_min_confluence: Optional[int] = Field(None, ge=1, le=4)
    ict_min_rr: Optional[float] = Field(None, ge=1.0, le=10.0)
    ict_max_sl_pct: Optional[float] = Field(None, ge=0.5, le=5.0)
    ict_require_sweep: Optional[bool] = None
    ict_require_displacement: Optional[bool] = None
    ict_killzone_only: Optional[bool] = None
    ict_max_notional: Optional[float] = Field(None, ge=10, le=5000)
    ict_max_loss_cap: Optional[float] = Field(None, ge=1, le=100)

@router.post("/quant/settings")
def update_quant_settings(req: QuantSettingsRequest):
    """Auto-Trader ayarlarını günceller (BB MR + ICT/SMC)."""
    return quant_trader.update_settings(
        max_open_trades=req.max_open_trades,
        max_notional=req.max_notional,
        max_loss_cap=req.max_loss_cap,
        min_rr=req.min_rr,
        tp_min=req.tp_min,
        balance=req.balance,
        bb_mr_enabled=req.bb_mr_enabled,
        ict_smc_enabled=req.ict_smc_enabled,
        ict_min_confluence=req.ict_min_confluence,
        ict_min_rr=req.ict_min_rr,
        ict_max_sl_pct=req.ict_max_sl_pct,
        ict_require_sweep=req.ict_require_sweep,
        ict_require_displacement=req.ict_require_displacement,
        ict_killzone_only=req.ict_killzone_only,
        ict_max_notional=req.ict_max_notional,
        ict_max_loss_cap=req.ict_max_loss_cap,
    )

@router.get("/quant/risk-settings")
def get_risk_settings():
    """Mevcut risk/trade ayarlarını döndürür."""
    return quant_trader.get_risk_settings()

class ResetRequest(BaseModel):
    new_balance: Optional[float] = 10000.0

@router.post("/quant/reset")
def reset_system(req: ResetRequest):
    """Sistemi sıfırlar — tüm trade geçmişi silinir, bakiye resetlenir."""
    return quant_trader.reset_system(req.new_balance)

@router.get("/quant/analytics")
def get_analytics():
    """Tüm kapalı trade geçmişi + aggregate metrikler."""
    raw = quant_trader.get_analytics()
    trades = raw.get('trades', [])

    if not trades:
        return {**raw, "analytics": {}}

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    analytics = {
        "total_trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "total_pnl": round(sum(t['pnl'] for t in trades), 2),
        "avg_win": round(sum(t['pnl'] for t in wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(t['pnl'] for t in losses) / len(losses), 2) if losses else 0,
        "best_trade": round(max(t['pnl'] for t in trades), 2) if trades else 0,
        "worst_trade": round(min(t['pnl'] for t in trades), 2) if trades else 0,
    }

    return {**raw, "analytics": analytics}

@router.get("/v3/stats")
def get_v3_stats():
    """Engine performans istatistiklerini (Win Rate, Avg RR, Profit Factor) döndürür."""
    with quant_trader.trades_lock:
        trades = list(quant_trader.closed_trades)

    if not trades:
        return {
            "total": 0, "total_trades": 0, "win_rate": 0,
            "avg_rr": 0, "avg_pnl": 0, "profit_factor": 0,
            "total_pnl": 0, "total_pnl_pct": 0, "wins": 0, "losses": 0,
            "by_strategy": {},
        }

    wins = [t for t in trades if t.get('pnl', 0) > 0]
    losses = [t for t in trades if t.get('pnl', 0) <= 0]
    total_pnl = sum(t.get('pnl', 0) for t in trades)
    total_margin = sum(t.get('margin', 0) for t in trades)
    total_pnl_pct = round((total_pnl / total_margin) * 100, 2) if total_margin > 0 else 0

    # ★ Ortalama PnL ($ per trade)
    avg_pnl = round(total_pnl / len(trades), 2) if trades else 0

    # ★ Gerçek avg R:R hesaplama
    rr_values = []
    for t in trades:
        entry = t.get('entry_price', 0)
        sl = t.get('sl_price', 0)
        exit_p = t.get('exit_price', 0)
        if entry > 0 and sl > 0 and exit_p > 0:
            risk = abs(entry - sl)
            if risk > 0:
                reward = abs(exit_p - entry)
                rr = reward / risk
                if t.get('pnl', 0) < 0:
                    rr = -rr
                rr_values.append(rr)
    avg_rr = round(sum(rr_values) / len(rr_values), 2) if rr_values else 0

    # ★ Profit Factor
    gross_profit = sum(t.get('pnl', 0) for t in wins) if wins else 0
    gross_loss = abs(sum(t.get('pnl', 0) for t in losses)) if losses else 0
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf')

    # ★ Strateji bazlı breakdown
    by_strategy = {}
    for t in trades:
        strat = t.get('strategy', 'unknown')
        if strat not in by_strategy:
            by_strategy[strat] = {'wins': 0, 'losses': 0, 'pnl': 0.0}
        if t.get('pnl', 0) > 0:
            by_strategy[strat]['wins'] += 1
        else:
            by_strategy[strat]['losses'] += 1
        by_strategy[strat]['pnl'] += t.get('pnl', 0)

    for strat, data in by_strategy.items():
        total_s = data['wins'] + data['losses']
        data['win_rate'] = round(data['wins'] / total_s * 100, 1) if total_s > 0 else 0
        data['pnl'] = round(data['pnl'], 2)
        data['total'] = total_s

    return {
        "total": len(trades),
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "avg_rr": avg_rr,
        "avg_pnl": avg_pnl,
        "profit_factor": min(profit_factor, 99.99),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": total_pnl_pct,
        "by_strategy": by_strategy,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic / Health Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/quant/health")
def get_health():
    """
    Sistem sağlık kontrolü — model, strateji, pipeline durumu.
    Hangi modeller yüklü, hangi stratejiler aktif, pass rate ne?
    """
    from ai.adaptive_live_adapter import _get_engine
    engine = _get_engine()
    return engine.get_health_check()


@router.get("/quant/diagnose/{symbol}")
def diagnose_symbol(symbol: str):
    """
    Belirli bir sembol için tüm karar pipeline'ını adım adım trace eder.

    7 Adım:
      1. Rejim Tespiti (1h)
      2. Strateji Sinyali (primary → secondary fallback)
      3. MTF Filtre (4h trend yönü)
      4. Entry Timing (15m RSI/momentum/mum)
      5. Meta Filter (XGBoost model)
      6. Final Decision (confidence, size)
      7. Quality Gate (soft_score≥3, meta>0)

    Kullanım: GET /live/quant/diagnose/BTCUSDT
    """
    import traceback
    try:
        from backtest.data_fetcher import DataFetcher
        from backtest.signals import add_all_indicators, add_meta_context_features
        from ai.xgboost_trainer import generate_features
        from ai.adaptive_live_adapter import _get_engine, _ensure_features

        fetcher = DataFetcher('binance')

        # Format symbol
        sym = symbol.upper()
        if '/' not in sym and sym.endswith('USDT'):
            sym_ccxt = sym[:-4] + '/USDT'
        elif '/' not in sym:
            sym_ccxt = sym + '/USDT'
        else:
            sym_ccxt = sym

        # Fetch data for all timeframes
        data_status = {}

        df_1h = fetcher.fetch_ohlcv(sym_ccxt, '1h', limit=100)
        if df_1h is None or df_1h.empty or len(df_1h) < 50:
            return {"error": f"1h veri yetersiz ({len(df_1h) if df_1h is not None else 0} bars)", "symbol": sym}
        df_1h = add_all_indicators(df_1h)
        df_1h = generate_features(df_1h)
        df_1h = _ensure_features(df_1h, sym)
        data_status["1h"] = f"{len(df_1h)} bars ✅"

        df_15m = fetcher.fetch_ohlcv(sym_ccxt, '15m', limit=100)
        if df_15m is not None and not df_15m.empty and len(df_15m) >= 20:
            df_15m = add_all_indicators(df_15m)
            data_status["15m"] = f"{len(df_15m)} bars ✅"
        else:
            df_15m = None
            data_status["15m"] = "yetersiz ⚠️"

        df_4h = fetcher.fetch_ohlcv(sym_ccxt, '4h', limit=100)
        if df_4h is not None and not df_4h.empty and len(df_4h) >= 30:
            df_4h = add_all_indicators(df_4h)
            data_status["4h"] = f"{len(df_4h)} bars ✅"
        else:
            df_4h = None
            data_status["4h"] = "yetersiz ⚠️"

        # Run diagnostic
        engine = _get_engine()
        trace = engine.diagnose(
            df=df_1h,
            df_secondary=df_15m,
            df_4h=df_4h,
            symbol=sym,
        )
        trace["data_fetch"] = data_status
        return trace

    except Exception as e:
        resp = {"error": str(e), "symbol": symbol}
        if _DEBUG:
            resp["traceback"] = traceback.format_exc()
        return resp


@router.get("/quant/features/{symbol}")
def get_features(symbol: str):
    """
    Meta model'e giren 43 feature'ın tam değerlerini döndürür.
    Eğitim vs production karşılaştırması için kullanılır.
    """
    import traceback
    try:
        from backtest.data_fetcher import DataFetcher
        from backtest.signals import add_all_indicators
        from ai.xgboost_trainer import generate_features
        from ai.adaptive_live_adapter import _get_engine, _ensure_features

        fetcher = DataFetcher('binance')
        sym = symbol.upper()
        if '/' not in sym and sym.endswith('USDT'):
            sym_ccxt = sym[:-4] + '/USDT'
        elif '/' not in sym:
            sym_ccxt = sym + '/USDT'
        else:
            sym_ccxt = sym

        df_1h = fetcher.fetch_ohlcv(sym_ccxt, '1h', limit=100)
        if df_1h is None or df_1h.empty or len(df_1h) < 50:
            return {"error": "1h veri yetersiz"}
        df_1h = add_all_indicators(df_1h)
        df_1h = generate_features(df_1h)
        df_1h = _ensure_features(df_1h, sym)

        df_4h = fetcher.fetch_ohlcv(sym_ccxt, '4h', limit=100)
        if df_4h is not None and not df_4h.empty and len(df_4h) >= 30:
            df_4h = add_all_indicators(df_4h)
        else:
            df_4h = None

        engine = _get_engine()
        engine._inject_htf_features(df_1h, df_4h)

        # Run predict with debug to get feature values
        from ai.regime_detector import detect_regime
        regime, _ = detect_regime(df_1h)

        engine.meta_predictor.predict(
            df=df_1h, regime=regime, signal_direction="LONG",
            signal_confidence=0.8, timeframe="1h", debug=True,
        )

        debug = engine.meta_predictor.last_debug
        if not debug:
            return {"error": "debug info not available"}

        # Sort features by category
        feature_vals = debug.get("feature_values", {})
        categorized = {"signal": {}, "htf": {}, "context": {}, "technical": {}}
        for fname, info in feature_vals.items():
            val = info["value"]
            src = info["source"]
            if src == "signal" or fname.startswith("signal_") or fname.startswith("regime_"):
                categorized["signal"][fname] = val
            elif fname.startswith("htf_"):
                categorized["htf"][fname] = val
            elif fname in ("close_pct_change", "vol_sma_ratio", "di_diff", "vol_ratio_20"):
                categorized["technical"][fname] = val
            else:
                categorized["context"][fname] = val

        return {
            "symbol": sym,
            "regime": regime.value,
            "model_key": debug.get("model_key"),
            "total_features": debug.get("total_features"),
            "ok_nonzero": debug.get("ok_nonzero"),
            "missing": debug.get("missing_from_df"),
            "zero_count": debug.get("zero_or_nan"),
            "features_by_category": categorized,
        }

    except Exception as e:
        resp = {"error": str(e), "symbol": symbol}
        if _DEBUG:
            resp["traceback"] = traceback.format_exc()
        return resp


# ─────────────────────────────────────────────────────────────────────────────
# Cooldown Status Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/quant/cooldown")
def get_cooldown_status():
    """Aktif cooldown durumlarını döndürür."""
    now = time.time()
    return {
        "global_cooldown": {
            "active": now < quant_trader._cooldown_until,
            "remaining_sec": max(0, int(quant_trader._cooldown_until - now)),
            "consecutive_losses": quant_trader.consecutive_losses,
        },
        "bb_mr": {
            "cooldown_active": now < quant_trader._bb_cooldown_until,
            "remaining_sec": max(0, int(quant_trader._bb_cooldown_until - now)),
            "consecutive_sl": quant_trader._bb_consecutive_sl,
            "last_trade_age_sec": int(now - quant_trader._bb_last_trade_time) if quant_trader._bb_last_trade_time > 0 else -1,
            "recent_outcomes": quant_trader._bb_recent_outcomes[-5:],
        },
        "ict": {
            "cooldown_active": now < quant_trader._ict_cooldown_until,
            "remaining_sec": max(0, int(quant_trader._ict_cooldown_until - now)),
            "consecutive_sl": quant_trader._ict_consecutive_sl,
            "last_trade_age_sec": int(now - quant_trader._ict_last_trade_time) if quant_trader._ict_last_trade_time > 0 else -1,
            "recent_outcomes": quant_trader._ict_recent_outcomes[-5:],
        },
        "symbol_cooldowns": {
            "bb_mr": {s: int(t - now) for s, t in quant_trader._bb_symbol_cooldown.items() if t > now},
            "ict": {s: int(t - now) for s, t in quant_trader._ict_symbol_cooldown.items() if t > now},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standard Bot Placeholders (Frontend 404 Fix)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/bots/configs")
def get_bot_configs():
    return []

@router.get("/bots/status")
def get_bots_status():
    return {"summary": {"active": 0, "total": 0}, "bots": []}

@router.post("/bots/start-all")
def start_all_bots():
    return {"success": True, "message": "No standard bots configured."}
