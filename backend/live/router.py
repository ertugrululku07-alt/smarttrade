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
