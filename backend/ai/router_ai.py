from fastapi import APIRouter

router = APIRouter(prefix="/ai", tags=["AI Status Intelligence"])

@router.get("/xgboost/status")
def get_xgboost_status():
    """XGBoost model durumunu döndürür (Placeholder)."""
    return {
        "status": "active",
        "models_loaded": 4,
        "last_training": "N/A",
        "ensemble_ready": True
    }

@router.get("/xgboost/scheduler/status")
def get_scheduler_status():
    return {"active": False, "next_run": "Manual Only"}

@router.get("/xgboost/ensemble/status")
def get_ensemble_status():
    return {"status": "synced", "weights": [0.25, 0.25, 0.25, 0.25]}
