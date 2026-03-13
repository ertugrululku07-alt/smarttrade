from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List
from live_trader import LivePaperTrader

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
    if not quant_trader.is_running:
        return {"success": False, "message": "Auto-Trader is not running"}
    
    result = quant_trader.close_trade(trade_id)
    return result

class QuantSettingsRequest(BaseModel):
    max_open_trades: int

@router.post("/quant/settings")
def update_quant_settings(req: QuantSettingsRequest):
    """Auto-Trader ayarlarını günceller."""
    return quant_trader.update_settings(req.max_open_trades)

@router.get("/v3/stats")
def get_v3_stats():
    """Engine performans istatistiklerini (Win Rate, Avg RR) döndürür."""
    with quant_trader.trades_lock:
        trades = list(quant_trader.closed_trades)

    if not trades:
        return {
            "total_trades": 0, "win_rate": 0, "avg_rr": 0,
            "total_pnl": 0, "wins": 0, "losses": 0,
        }

    wins = [t for t in trades if t.get('pnl', 0) > 0]
    losses = [t for t in trades if t.get('pnl', 0) <= 0]
    total_pnl = sum(t.get('pnl', 0) for t in trades)
    avg_rr = (total_pnl / len(trades)) if trades else 0

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "avg_rr": round(avg_rr, 2),
        "total_pnl": round(total_pnl, 2),
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
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "symbol": symbol,
        }


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
        return {"error": str(e), "traceback": traceback.format_exc()}


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
