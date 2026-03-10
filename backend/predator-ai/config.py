import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

load_dotenv()

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"

@dataclass
class DatabaseConfig:
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    postgres_url: str = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost/predator")

@dataclass
class ExchangeConfig:
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_secret: str = os.getenv("BINANCE_SECRET", "")
    bybit_api_key: str = os.getenv("BYBIT_API_KEY", "")
    bybit_secret: str = os.getenv("BYBIT_SECRET", "")
    
@dataclass
class Web3Config:
    eth_rpc: str = os.getenv("ETH_RPC", "https://mainnet.infura.io/v3/your-project-id")
    bsc_rpc: str = os.getenv("BSC_RPC", "https://bsc-dataseed.binance.org")
    solana_rpc: str = os.getenv("SOLANA_RPC", "https://api.mainnet-beta.solana.com")
    moralis_api_key: str = os.getenv("MORALIS_API_KEY", "")

@dataclass
class SocialConfig:
    twitter_bearer_token: str = os.getenv("TWITTER_BEARER_TOKEN", "")
    twitter_api_key: str = os.getenv("TWITTER_API_KEY", "")
    twitter_api_secret: str = os.getenv("TWITTER_API_SECRET", "")
    telegram_api_id: str = os.getenv("TELEGRAM_API_ID", "")
    telegram_api_hash: str = os.getenv("TELEGRAM_API_HASH", "")
    discord_token: str = os.getenv("DISCORD_TOKEN", "")

@dataclass
class AIConfig:
    ensemble_weights: Dict[str, float] = None
    rl_learning_rate: float = 0.0003
    rl_buffer_size: int = 10000
    max_drawdown: float = 0.15
    max_leverage: int = 3
    daily_loss_limit: float = 0.05
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'xgboost': 0.2,
                'lstm': 0.25,
                'graph_nn': 0.3,
                'transformer': 0.25
            }

@dataclass
class TradingConfig:
    mode: TradingMode = TradingMode.PAPER
    min_confidence: float = 0.75
    max_position_size: float = 10000  # USD
    min_position_size: float = 100     # USD
    max_slippage: float = 0.01          # 1%
    gas_limit: int = 500000
    gas_price_multiplier: float = 1.1

@dataclass
class Config:
    env: Environment = Environment(os.getenv("ENVIRONMENT", "development"))
    db: DatabaseConfig = DatabaseConfig()
    exchanges: ExchangeConfig = ExchangeConfig()
    web3: Web3Config = Web3Config()
    social: SocialConfig = SocialConfig()
    ai: AIConfig = AIConfig()
    trading: TradingConfig = TradingConfig()
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = "logs/predator.log"
    
    # Monitoring
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", 8000))
    telegram_alert_token: str = os.getenv("TELEGRAM_ALERT_TOKEN", "")
    telegram_alert_chat_id: str = os.getenv("TELEGRAM_ALERT_CHAT_ID", "")

config = Config()