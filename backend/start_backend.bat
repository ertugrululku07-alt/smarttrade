@echo off
chcp 65001 >nul
title SmartTrade Backend

echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║         🎩 SMARTTRADE AI BACKEND BAŞLATILIYOR       ║
echo  ╚══════════════════════════════════════════════════════╝
echo.

:: Virtual environment'ı aktive et
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [HATA] venv bulunamadı! Önce: python -m venv venv
    pause
    exit /b 1
)

echo [1/4] Python ortamı hazır ✓

:: Model klasörü yoksa oluştur
if not exist "ai\models" (
    mkdir ai\models
    echo [2/4] Model klasörü oluşturuldu: ai\models\
) else (
    echo [2/4] Model klasörü mevcut ✓
)

:: Adaptive AI model dosyaları kontrolü
echo.
echo [3/4] Model durumu kontrol ediliyor...
python -c "import os; regimes=['TRENDING','MEAN_REVERTING','HIGH_VOLATILE','LOW_VOLATILE']; import glob; m=[r for r in regimes if not glob.glob(f'ai/models/meta_{r.lower()}_*.joblib')]; print('\n  Adaptive AI icin', len(m), 'model eksik.') if m else print('\n  Tum modeller hazir.'); print('  Egitmek icin: python -m ai.meta_labelling.meta_trainer\n') if m else None"

echo.
echo [4/4] FastAPI sunucusu başlatılıyor...
echo.
echo  ┌─────────────────────────────────────────┐
echo  │  API:   http://localhost:8000           │
echo  │  Docs:  http://localhost:8000/docs      │
echo  │  Bots:  http://localhost:3000/bots      │
echo  └─────────────────────────────────────────┘
echo.

:: Sunucuyu başlat (Canlı Mod - Otomatik Yeniden Başlama Kapalı)
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info

pause
