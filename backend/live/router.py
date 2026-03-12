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
    """V3.1 Engine performans istatistiklerini (Win Rate, Avg RR) döndürür."""
    return quant_trader.engine_v3.logger.get_stats()

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
