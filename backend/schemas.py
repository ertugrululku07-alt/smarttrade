from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# --- User Schemas ---
class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    is_premium: bool
    created_at: datetime

    class Config:
        from_attributes = True

# --- Auth Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# --- Exchange Key Schemas ---
class ExchangeKeyCreate(BaseModel):
    exchange_name: str
    api_key: str
    secret_key: str
    is_testnet: bool = True

class ExchangeKeyResponse(BaseModel):
    id: int
    exchange_name: str
    is_testnet: bool

    class Config:
        from_attributes = True

# --- Bot & Bot Config Schemas ---
class BotBase(BaseModel):
    name: str

class BotCreate(BotBase):
    pass

class BotResponse(BotBase):
    id: int
    user_id: int
    status: str
    created_at: datetime

    class Config:
        from_attributes = True

class BotConfigBase(BaseModel):
    config_json: str

class BotConfigCreate(BotConfigBase):
    pass

class BotConfigResponse(BotConfigBase):
    id: int
    bot_id: int

    class Config:
        from_attributes = True

# --- Backtest Engine Schemas ---
class BotStrategyBlock(BaseModel):
    id: str
    type: str
    title: Optional[str] = None

class BacktestRequest(BaseModel):
    symbol: str = 'BTC/USDT'
    timeframe: str = '1h'
    limit: int = 200
    initial_balance: float = 1000.0
    strategy: List[BotStrategyBlock]

# --- AI Optimizer Schemas ---
class OptimizationRequest(BaseModel):
    symbol: str = 'BTC/USDT'
    timeframe: str = '1h'
    limit: int = 500
    initial_balance: float = 1000.0
    n_trials: int = 50 # Optuna tarafından denenecek kombinasyon sayısı
    strategy: List[BotStrategyBlock]