from fastapi import APIRouter, HTTPException
from schemas import BacktestRequest
from backtest.data_fetcher import DataFetcher
from backtest.engine import BacktestEngine
from ai.adaptive_backtest import AdaptiveBacktest
from ai.ict_backtest import ICTBacktest
from backtest.ict_quick_test import quick_backtest_ict
from pydantic import BaseModel
from typing import Optional, List
import traceback

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
    strategy: str = "bb_mr"  # 'bb_mr' | 'ict_smc'


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

        # ── Strateji seçimine göre engine ──
        if request.strategy == 'ict_smc':
            # ICT/SMC Backtest — 4h HTF verisi gerekli
            df_4h = None
            try:
                df_4h = fetcher.fetch_ohlcv(request.symbol, "4h", limit=min(request.limit, 500))
                if df_4h is not None and (df_4h.empty or len(df_4h) < 30):
                    df_4h = None
                else:
                    df_4h = add_all_indicators(df_4h)
            except Exception:
                pass

            engine = ICTBacktest(
                initial_capital=request.initial_balance,
                min_quality=8,
                min_rr=2.0,
            )
            result_obj = engine.run(df, df_4h=df_4h, symbol=request.symbol)
        else:
            # BB Mean Reversion (varsayılan)
            use_simple = True
            engine = AdaptiveBacktest(
                timeframe=request.timeframe,
                initial_capital=request.initial_balance,
                use_simple_strategy=use_simple,
                use_meta_filter=not use_simple,
            )

            if not use_simple:
                df = enrich_ohlcv_with_futures(df, request.symbol, silent=True)
                df = generate_features(df)
                from backtest.signals import add_meta_context_features
                df = add_meta_context_features(df)

            df_4h = None
            if not use_simple:
                try:
                    df_4h = fetcher.fetch_ohlcv(request.symbol, "4h", limit=min(request.limit, 500))
                    if df_4h is not None and (df_4h.empty or len(df_4h) < 30):
                        df_4h = None
                except Exception:
                    pass

            result_obj = engine.run(df, df_4h=df_4h, symbol=request.symbol)
        
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


# ────────────────────────────────────────────────────────────
#  Multi-Coin Backtest  (Çoklu coin karşılaştırmalı)
# ────────────────────────────────────────────────────────────

class MultiCoinBacktestRequest(BaseModel):
    symbols: List[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    timeframe: str = "1h"
    limit: int = 500
    initial_balance: float = 1000.0
    min_confidence: float = 0.55
    strategy: str = "bb_mr"  # 'bb_mr' | 'ict_smc'


@router.post("/run-multi")
def run_multi_coin_backtest(request: MultiCoinBacktestRequest):
    """
    Çoklu coin backtest — her sembol için ayrı adaptive backtest çalıştırır,
    sonuçları karşılaştırmalı olarak döndürür.
    """
    import numpy as np

    def _to_python(obj):
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

    if not request.symbols or len(request.symbols) == 0:
        raise HTTPException(status_code=400, detail="En az 1 sembol seçin")
    if len(request.symbols) > 20:
        raise HTTPException(status_code=400, detail="Max 20 sembol desteklenir")

    from ai.xgboost_trainer import generate_features
    from backtest.signals import add_all_indicators
    from dataclasses import asdict

    fetcher = DataFetcher('binance')
    results = []
    errors = []

    for symbol in request.symbols:
        try:
            df = fetcher.fetch_ohlcv(symbol, request.timeframe, limit=request.limit)
            if df.empty or len(df) < 50:
                errors.append({"symbol": symbol, "error": "Yetersiz veri"})
                continue

            df = add_all_indicators(df)

            # ── Strateji seçimi ──
            if request.strategy == 'ict_smc':
                df_4h = None
                try:
                    df_4h = fetcher.fetch_ohlcv(symbol, "4h", limit=min(request.limit, 500))
                    if df_4h is not None and (df_4h.empty or len(df_4h) < 30):
                        df_4h = None
                    else:
                        df_4h = add_all_indicators(df_4h)
                except Exception:
                    pass

                engine = ICTBacktest(
                    initial_capital=request.initial_balance,
                    min_quality=8,
                    min_rr=2.0,
                )
                result_obj = engine.run(df, df_4h=df_4h, symbol=symbol)
            else:
                use_simple = True
                if not use_simple:
                    try:
                        from ai.data_sources.futures_data import enrich_ohlcv_with_futures
                        df = enrich_ohlcv_with_futures(df, symbol, silent=True)
                    except Exception:
                        pass
                    df = generate_features(df)
                    from backtest.signals import add_meta_context_features
                    df = add_meta_context_features(df)

                df_4h = None
                if not use_simple:
                    try:
                        df_4h = fetcher.fetch_ohlcv(symbol, "4h", limit=min(request.limit, 500))
                        if df_4h is not None and (df_4h.empty or len(df_4h) < 30):
                            df_4h = None
                    except Exception:
                        pass

                engine = AdaptiveBacktest(
                    timeframe=request.timeframe,
                    initial_capital=request.initial_balance,
                    use_simple_strategy=use_simple,
                    use_meta_filter=not use_simple,
                )
                result_obj = engine.run(df, df_4h=df_4h, symbol=symbol)
            raw = _to_python(asdict(result_obj))

            final_bal = float(result_obj.equity_curve[-1]) if result_obj.equity_curve else request.initial_balance
            total_pnl = final_bal - request.initial_balance

            results.append({
                "symbol": symbol,
                "total_trades": raw.get('total_trades', 0),
                "winners": raw.get('winners', 0),
                "losers": raw.get('losers', 0),
                "win_rate": raw.get('win_rate', 0),
                "profit_factor": raw.get('profit_factor', 0),
                "total_pnl": round(total_pnl, 2),
                "total_pnl_pct": raw.get('total_pnl_pct', 0),
                "max_drawdown": raw.get('max_drawdown_pct', 0),
                "sharpe_ratio": raw.get('sharpe_ratio', 0),
                "avg_bars_held": raw.get('avg_bars_held', 0),
                "initial_balance": request.initial_balance,
                "final_balance": round(final_bal, 2),
                "equity_curve": raw.get('equity_curve', []),
                "strategy_usage": {s: v.get('trades', 0) for s, v in raw.get('strategy_stats', {}).items()},
                "regime_stats": raw.get('regime_stats', {}),
                "trades": raw.get('trades', [])[-30:],  # Son 30 trade
                "date_range": {
                    "from": str(df.index[0]) if len(df) > 0 else "",
                    "to": str(df.index[-1]) if len(df) > 0 else "",
                },
            })
        except Exception as e:
            errors.append({"symbol": symbol, "error": str(e)})

    # Özet metrikleri hesapla
    if results:
        total_pnl_all = sum(r['total_pnl'] for r in results)
        avg_wr = np.mean([r['win_rate'] for r in results if r['total_trades'] > 0])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results if r['total_trades'] > 0])
        max_dd = max(r['max_drawdown'] for r in results) if results else 0
        best = max(results, key=lambda r: r['total_pnl'])
        worst = min(results, key=lambda r: r['total_pnl'])

        summary = {
            "total_symbols": len(results),
            "total_pnl": round(total_pnl_all, 2),
            "avg_win_rate": round(float(avg_wr), 2),
            "avg_sharpe": round(float(avg_sharpe), 3),
            "max_drawdown": round(float(max_dd), 2),
            "best_symbol": best['symbol'],
            "best_pnl": best['total_pnl'],
            "worst_symbol": worst['symbol'],
            "worst_pnl": worst['total_pnl'],
            "profitable_symbols": sum(1 for r in results if r['total_pnl'] > 0),
        }
    else:
        summary = {"total_symbols": 0, "total_pnl": 0}

    return {
        "success": True,
        "summary": summary,
        "results": results,
        "errors": errors,
    }


# ────────────────────────────────────────────────────────────
#  ICT/SMC v2.5 Full Backtest (Complete Strategy Simulation)
# ────────────────────────────────────────────────────────────

class TrendBacktestRequest(BaseModel):
    symbol: str = "BTC/USDT"
    days: int = 30
    initial_balance: float = 1000.0
    leverage: int = 10


@router.post("/trend-backtest")
def trend_backtest_endpoint(request: TrendBacktestRequest):
    """Trend Following Backtest — Supertrend + EMA + ADX + Smart Trail"""
    try:
        from backtest.trend_backtest import full_backtest_trend
        result = full_backtest_trend(
            symbol=request.symbol,
            days=request.days,
            initial_balance=request.initial_balance,
            leverage=request.leverage
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


class ICTFullBacktestRequest(BaseModel):
    symbol: str = "BTC/USDT"
    days: int = 30
    initial_balance: float = 1000.0
    leverage: int = 10


@router.post("/ict-full-backtest")
def ict_full_backtest_endpoint(request: ICTFullBacktestRequest):
    """
    ICT/SMC v2.5 Full Strategy Backtest
    Complete simulation with entry, exit, profit protection, PnL tracking
    Auto-detects both LONG and SHORT opportunities
    
    Example:
        POST /backtest/ict-full-backtest
        {
            "symbol": "BANANAS31/USDT",
            "days": 30,
            "initial_balance": 1000.0,
            "leverage": 10
        }
    
    Returns:
        - Total trades (LONG + SHORT count)
        - Win rate, profit factor
        - Complete trade history
        - Final balance and PnL
    """
    try:
        from backtest.ict_full_backtest import full_backtest_ict
        
        result = full_backtest_ict(
            symbol=request.symbol,
            days=request.days,
            initial_balance=request.initial_balance,
            leverage=request.leverage
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
