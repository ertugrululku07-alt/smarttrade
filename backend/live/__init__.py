"""Live Trading Package — Güvenli import ile."""

try:
    from live.trading_engine import TradingEngine, engine
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"trading_engine yüklenemedi: {e}")

    # Placeholder — uygulama çökmez
    class TradingEngine:
        """Placeholder — trading_engine.py bulunamadı."""
        def __init__(self):
            self.bots = {}
        def start_bot(self, config):
            return {"success": False, "message": "trading_engine.py bulunamadı"}
        def stop_bot(self, bot_id):
            return {"success": False, "message": "trading_engine.py bulunamadı"}
        def get_all_status(self):
            return {"summary": {}, "bots": []}

    engine = TradingEngine()

__all__ = ["TradingEngine", "engine"]