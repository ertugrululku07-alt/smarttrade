#!/usr/bin/env python3
"""
Predator AI - Piyasa Avcısı Robot
Ana çalıştırma dosyası
"""

import asyncio
import signal
import sys
from datetime import datetime
from loguru import logger
import argparse

from config import config, TradingMode
from core.data_harvester import DataHarvester
from core.analyzers.onchain import OnChainAnalyzer
from core.analyzers.social import SocialSentimentAnalyzer
from core.analyzers.microstructure import MarketMicrostructureAnalyzer
from core.decision.ensemble import EnsembleDecisionMaker
from core.decision.rl_agent import RLAgent
from core.decision.risk_governor import RiskGovernor
from core.execution.executor import ExecutionEngine

class PredatorAI:
    """Ana robot sınıfı"""
    
    def __init__(self):
        self.config = config
        self.running = False
        
        # Initialize components
        logger.info("Initializing Predator AI components...")
        
        self.harvester = DataHarvester(config)
        self.onchain_analyzer = OnChainAnalyzer(
            self.harvester.w3_eth, 
            config.web3.moralis_api_key
        )
        self.social_analyzer = SocialSentimentAnalyzer()
        self.market_analyzer = MarketMicrostructureAnalyzer()
        self.ensemble = EnsembleDecisionMaker(config)
        self.rl_agent = RLAgent(config)
        self.risk_governor = RiskGovernor(config)
        self.executor = ExecutionEngine(config)
        
        # Metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.start_time = datetime.now()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        """Graceful shutdown"""
        logger.info("Shutdown signal received")
        self.running = False
        
    async def analyze_opportunity(self, data: dict) -> dict:
        """Bir fırsatı analiz et"""
        
        # 1. On-chain analiz
        onchain_result = await self.onchain_analyzer.analyze_token(
            data.get('token_address', ''),
            data.get('chain', 'ethereum')
        )
        
        # 2. Sosyal medya analizi
        social_result = await self.social_analyzer.analyze_project(
            data.get('project_name', ''),
            data.get('keywords', [])
        )
        
        # 3. Market mikroyapı analizi
        market_result = await self.market_analyzer.analyze_order_book(
            data.get('bids', []),
            data.get('asks', [])
        )
        
        # Pump & dump tespiti
        pump_dump = await self.market_analyzer.detect_pump_dump(
            data.get('trades', [])
        )
        
        # 4. Tüm feature'ları birleştir
        features = {
            'onchain': onchain_result,
            'social': social_result,
            'market': market_result,
            'pump_dump': pump_dump
        }
        
        # 5. Ensemble model ile tahmin
        prediction = await self.ensemble.predict(features)
        
        return {
            'features': features,
            'prediction': prediction,
            'timestamp': datetime.now()
        }
        
    async def trading_cycle(self):
        """Ana trading döngüsü"""
        
        logger.info("Starting trading cycle...")
        
        while self.running:
            try:
                # 1. Yeni token'ları kontrol et
                new_pairs = self.harvester.dex_listener.new_pairs[-10:]  # Son 10 yeni token
                
                for pair in new_pairs:
                    # Token'ı analiz et
                    analysis = await self.analyze_opportunity({
                        'token_address': pair['token0'],
                        'chain': pair['chain'],
                        'project_name': f"Token_{pair['token0'][:6]}",
                        'keywords': ['crypto', 'new', 'token']
                    })
                    
                    # Eğer fırsat varsa
                    if analysis['prediction']['action'] != 'HOLD' and \
                       analysis['prediction']['confidence'] > self.config.trading.min_confidence:
                        
                        # Mevcut piyasa durumunu al
                        market_data = {
                            'price': 0.01,  # Gerçek fiyatı çek
                            'volatility': 0.05,
                            'volume': 100000
                        }
                        
                        # Risk kontrolünden geçir
                        approved, reason, signal = await self.risk_governor.approve_trade(
                            {
                                'action': analysis['prediction']['action'],
                                'symbol': f"{pair['token0']}/USDT",
                                'entry_price': 0.01,
                                'position_size': 1000,
                                'leverage': 1,
                                'sector': 'defi',
                                'confidence': analysis['prediction']['confidence']
                            },
                            market_data
                        )
                        
                        if approved:
                            # İşlemi icra et
                            result = await self.executor.execute_signal(signal)
                            
                            if result['status'] == 'executed':
                                self.total_trades += 1
                                logger.success(f"Trade executed: {result}")
                                
                                # RL agent'ı güncelle
                                state = self.rl_agent.get_state(market_data, {
                                    'total_value': self.risk_governor.portfolio_value,
                                    'position_size': signal['position_size']
                                })
                                
                                # Reward hesapla (ileride PnL ile güncellenecek)
                                self.rl_agent.remember(
                                    state, 
                                    signal.get('action_id', 0),
                                    0,  # Geçici reward
                                    state,
                                    False
                                )
                                
                # 2. Açık pozisyonları kontrol et
                for position in self.risk_governor.open_positions:
                    # Stop-loss ve take-profit kontrolü
                    pass
                    
                # 3. RL agent'ı eğit
                if len(self.rl_agent.memory) > 32:
                    self.rl_agent.train()
                    
                # 4. Metrikleri logla
                if self.total_trades > 0:
                    win_rate = self.winning_trades / self.total_trades
                    logger.info(f"Win rate: {win_rate:.2%}, Trades: {self.total_trades}")
                    
            except Exception as e:
                logger.error(f"Trading cycle error: {e}")
                
            await asyncio.sleep(10)  # 10 saniye bekle
            
    async def run(self):
        """Ana çalıştırma fonksiyonu"""
        
        logger.info("=" * 60)
        logger.info("PREDATOR AI - Piyasa Avcısı Robot")
        logger.info(f"Mode: {self.config.trading.mode.value}")
        logger.info(f"Environment: {self.config.env.value}")
        logger.info("=" * 60)
        
        self.running = True
        
        # Paralel görevleri başlat
        tasks = [
            self.harvester.run(),
            self.trading_cycle(),
            self.executor.monitor_positions()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            await self.shutdown()
            
    async def shutdown(self):
        """Güvenli kapanış"""
        
        logger.info("Shutting down Predator AI...")
        
        # Açık pozisyonları kapat
        await self.risk_governor.emergency_close_all()
        
        # Metrikleri raporla
        uptime = datetime.now() - self.start_time
        logger.info(f"Uptime: {uptime}")
        logger.info(f"Total trades: {self.total_trades}")
        logger.info(f"Final portfolio: ${self.risk_governor.portfolio_value:.2f}")
        
        logger.info("Shutdown complete")

def main():
    parser = argparse.ArgumentParser(description='Predator AI Trading Bot')
    parser.add_argument('--mode', type=str, default='paper',
                       choices=['paper', 'live', 'backtest'],
                       help='Trading mode')
    parser.add_argument('--config', type=str, default='.env',
                       help='Config file path')
    
    args = parser.parse_args()
    
    # Configure logging
    logger.add(
        "logs/predator_{time}.log",
        rotation="500 MB",
        retention="10 days",
        level=config.log_level
    )
    
    # Set trading mode
    if args.mode == 'paper':
        config.trading.mode = TradingMode.PAPER
    elif args.mode == 'live':
        config.trading.mode = TradingMode.LIVE
    elif args.mode == 'backtest':
        config.trading.mode = TradingMode.BACKTEST
        
    # Run bot
    bot = PredatorAI()
    
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()