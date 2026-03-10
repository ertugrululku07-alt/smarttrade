from fastapi import APIRouter, HTTPException
from schemas import BacktestRequest
from backtest.data_fetcher import DataFetcher
from backtest.engine import BacktestEngine
from ai.adaptive_backtest import AdaptiveBacktest
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/backtest", tags=["Backtest Intelligence"])


# ────────────────────────────────────────────────────────────
#  Adaptive AI Backtest  (Ana endpoint)
# ────────────────────────────────────────────────────────────

class AdaptiveBacktestRequest(BaseModel):
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = 500
    initial_balance: float = 1000.0
    trade_size_pct: float = 15.0
    min_confidence: float = 0.55


@router.post("/run-adaptive")
def run_adaptive_backtest(request: AdaptiveBacktestRequest):
    """
    AI Adaptif Backtest — Piyasa rejimine göre strateji otomatik değişir.
    """
    import numpy as np

    def _to_python(obj):
        """numpy / dataclass nesnelerini JSON-safe Python tiplerine çevirir."""
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_python(v) for v in obj]
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    try:
        from ai.data_sources.futures_data import enrich_ohlcv_with_futures
        from ai.xgboost_trainer import generate_features
        from backtest.signals import add_all_indicators
        from dataclasses import asdict

        fetcher = DataFetcher('binance')
        df = fetcher.fetch_ohlcv(request.symbol, request.timeframe, limit=request.limit)
        if df.empty:
            raise HTTPException(status_code=400, detail="No market data available.")

        df = add_all_indicators(df)
        df = enrich_ohlcv_with_futures(df, request.symbol, silent=True)
        df = generate_features(df)

        engine = AdaptiveBacktest(
            timeframe=request.timeframe,
            initial_capital=request.initial_balance,
            use_meta_filter=True,   # Predictor fix uygulandı, aktif edildi
        )

        result_obj = engine.run(df, symbol=request.symbol)
        
        # ── v1.3: Debug stats terminale yazdır ──────────────
        if hasattr(engine, 'print_debug_stats'):
            engine.print_debug_stats()

        raw = asdict(result_obj)
        metrics = _to_python(raw)

        metrics['initial_balance'] = request.initial_balance
        metrics['final_balance'] = float(result_obj.equity_curve[-1]) if result_obj.equity_curve else request.initial_balance
        metrics['total_pnl'] = metrics['final_balance'] - metrics['initial_balance']
        metrics['win_trades'] = result_obj.winners
        metrics['loss_trades'] = result_obj.losers
        metrics['max_drawdown'] = float(result_obj.max_drawdown_pct)
        metrics['regime_changes'] = len(set(t.regime for t in result_obj.trades)) if result_obj.trades else 0
        metrics['strategy_usage'] = {s: v['trades'] for s, v in result_obj.strategy_stats.items()}

        return {
            "success": True,
            "metrics": metrics,
            "date_range": {
                "from": str(df.index[0]),
                "to": str(df.index[-1]),
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Adaptive backtest failed: {str(e)}")


# ────────────────────────────────────────────────────────────
#  BotBuilder Backtest  (Bot oluşturucu sayfası için)
# ────────────────────────────────────────────────────────────

@router.post("/run")
def run_backtest(request: BacktestRequest):
    """
    Frontend BotBuilder'dan gelen blok dizisiyle backtest yapar.
    Gerçek Binance verisi + signals.py hesaplamaları.
    """
    try:
        fetcher = DataFetcher('binance')
        df = fetcher.fetch_ohlcv(request.symbol, request.timeframe, limit=request.limit)
        if df.empty:
            raise HTTPException(status_code=400, detail="No market data available.")

        engine = BacktestEngine(
            df,
            initial_balance=request.initial_balance,
        )
        engine.load_strategy([b.model_dump() for b in request.strategy])
        results = engine.run()
        return {
            "success": True,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "candles_tested": len(df),
            "date_range": {"from": str(df.index[0]), "to": str(df.index[-1])},
            "metrics": results,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")
