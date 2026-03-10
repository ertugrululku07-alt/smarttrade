import sys
import os

print("\n" + "="*50)
print("🚀 SMARTTRADE BACKEND STARTING (ENTRY POINT)")
print(f"📍 Python Path: {sys.path}")
print(f"📂 CWD: {os.getcwd()}")
print("="*50 + "\n")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Define app EARLY before heavy imports
app = FastAPI(
    title="SmartTrade AI Platform",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0", "msg": "Early health check active"}

@app.get("/")
def read_root():
    return {"message": "SmartTrade AI Platform v2.0"}

# Delay heavy imports
try:
    import models
    from database import engine
    from routers import router as auth_router
    from backtest.router import router as backtest_router
    from live.router import router as live_router
    from ai.router_ai import router as ai_router
    print("✅ All modules imported successfully.")
except Exception as e:
    print(f"❌ MODULE IMPORT ERROR: {e}")
    # We still want the app to run so we can see this error in logs

# Create all database tables (moved to lifespan)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 SmartTrade AI Platform Başlıyor...")
    try:
        models.Base.metadata.create_all(bind=engine)
        print("✅ Veritabanı tabloları hazır.")
    except Exception as e:
        print(f"❌ Veritabanı başlatma hatası (ama uygulama devam ediyor): {e}")
    yield
    # Shutdown
    print("🛑 SmartTrade AI Platform Kapanıyor...")


if "models" in locals():
    app.include_router(auth_router)
    app.include_router(backtest_router)
    app.include_router(live_router)
    app.include_router(ai_router)
