from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import models
from database import engine
from routers import router as auth_router
from backtest.router import router as backtest_router
from live.router import router as live_router
from ai.router_ai import router as ai_router

# Create all database tables (moved to lifespan)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 SmartTrade AI Platform Başlıyor...")
    models.Base.metadata.create_all(bind=engine)
    yield
    # Shutdown
    print("🛑 SmartTrade AI Platform Kapanıyor...")


app = FastAPI(
    title="SmartTrade AI Platform",
    version="2.0.0",
    description="AI-powered crypto trading platform with StrategistAI, SupervisorAI, LearnerAI and Auto-Trainer",
    lifespan=lifespan
)

# CORS yapılandırması
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(backtest_router)
app.include_router(live_router)
app.include_router(ai_router)

@app.get("/")
def read_root():
    return {
        "message": "SmartTrade AI Platform v2.0",
        "ai_systems": ["StrategistAI", "SupervisorAI", "LearnerAI"],
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0"}
