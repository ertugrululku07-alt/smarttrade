import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger
import asyncio

class RiskGovernor:
    """Risk yönetim katmanı"""
    
    def __init__(self, config):
        self.config = config
        self.max_drawdown = config.ai.max_drawdown
        self.max_leverage = config.ai.max_leverage
        self.daily_loss_limit = config.ai.daily_loss_limit
        
        # Portfolio metrics
        self.portfolio_value = 100000  # Başlangıç
        self.peak_value = self.portfolio_value
        self.current_drawdown = 0
        self.daily_pnl = 0
        self.daily_start_value = self.portfolio_value
        
        # Position tracking
        self.open_positions = []
        self.position_history = []
        self.daily_trades = 0
        
        # Risk limits
        self.max_positions = 5
        self.max_position_size_pct = 0.2  # Portföyün %20'si
        self.max_sector_exposure = 0.4  # Tek sektöre max %40
        
        # VaR parameters
        self.var_confidence = 0.95
        self.var_lookback = 100
        
    async def approve_trade(self, signal: Dict, market_data: Dict) -> Tuple[bool, str, Dict]:
        """İşlemi risk kontrollerinden geçir"""
        
        logger.info(f"Approving trade: {signal}")
        
        # 1. Günlük kayıp limiti kontrolü
        if self.daily_pnl < -self.daily_loss_limit * self.portfolio_value:
            return False, "Daily loss limit reached", {}
            
        # 2. Maksimum pozisyon sayısı kontrolü
        if len(self.open_positions) >= self.max_positions:
            return False, "Max positions reached", {}
            
        # 3. Pozisyon büyüklüğü kontrolü
        position_size = signal.get('position_size', 0)
        if position_size > self.portfolio_value * self.max_position_size_pct:
            position_size = self.portfolio_value * self.max_position_size_pct
            
        # 4. Kaldıraç kontrolü
        leverage = signal.get('leverage', 1)
        if leverage > self.max_leverage:
            leverage = self.max_leverage
            
        # 5. Sektör konsantrasyonu kontrolü
        sector = signal.get('sector', 'unknown')
        sector_exposure = self.get_sector_exposure(sector)
        if sector_exposure + position_size > self.portfolio_value * self.max_sector_exposure:
            return False, f"Sector exposure limit reached for {sector}", {}
            
        # 6. VaR kontrolü
        var_95 = self.calculate_var(position_size, leverage)
        if var_95 > self.portfolio_value * 0.02:  # Maksimum VaR portföyün %2'si
            return False, "VaR limit exceeded", {'var': var_95}
            
        # 7. Korelasyon riski
        correlation_risk = self.check_correlation_risk(signal)
        if correlation_risk > 0.7:  # Yüksek korelasyon
            logger.warning(f"High correlation risk: {correlation_risk}")
            
        # 8. Stop-loss kontrolü
        stop_loss = signal.get('stop_loss')
        if not stop_loss:
            # Varsayılan stop-loss
            stop_loss = signal.get('entry_price', 0) * 0.95 if signal.get('action') == 'LONG' else signal.get('entry_price', 0) * 1.05
            
        # 9. Take-profit kontrolü
        take_profit = signal.get('take_profit')
        if not take_profit:
            # Varsayılan take-profit
            take_profit = signal.get('entry_price', 0) * 1.15 if signal.get('action') == 'LONG' else signal.get('entry_price', 0) * 0.85
            
        # Risk-adjusted position size
        risk_per_trade = 0.01  # Portföyün %1'i
        volatility = market_data.get('volatility', 0.02)
        
        # Kelly Criterion based sizing
        win_prob = signal.get('confidence', 0.5)
        win_loss_ratio = 1.5  # Örnek
        
        kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Kelly'nin %25'i
        
        # Volatiliteye göre ayarla
        if volatility > 0.05:  # Yüksek volatilite
            kelly_fraction *= 0.5
            
        position_size = self.portfolio_value * kelly_fraction
        
        approved_signal = {
            **signal,
            'position_size': position_size,
            'leverage': leverage,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_per_trade': risk_per_trade,
            'kelly_fraction': kelly_fraction,
            'approved_at': datetime.now()
        }
        
        return True, "Approved", approved_signal
        
    def calculate_var(self, position_size: float, leverage: int, days: int = 1) -> float:
        """Value at Risk hesapla"""
        
        # Tarihsel volatilite (simülasyon)
        historical_returns = np.random.randn(self.var_lookback) * 0.02
        
        # VaR hesapla
        var_percentile = np.percentile(historical_returns, (1 - self.var_confidence) * 100)
        
        # Pozisyon büyüklüğü ve kaldıraç ile çarp
        var = abs(position_size * leverage * var_percentile * np.sqrt(days))
        
        return var
        
    def get_sector_exposure(self, sector: str) -> float:
        """Belirli bir sektördeki toplam exposure'ı hesapla"""
        
        exposure = 0
        for pos in self.open_positions:
            if pos.get('sector') == sector:
                exposure += pos.get('position_size', 0) * pos.get('leverage', 1)
                
        return exposure
        
    def check_correlation_risk(self, signal: Dict) -> float:
        """Yeni sinyalin mevcut pozisyonlarla korelasyon riskini hesapla"""
        
        if not self.open_positions:
            return 0
            
        # Basit korelasyon simülasyonu
        correlations = []
        for pos in self.open_positions:
            # Aynı yön
            if pos.get('action') == signal.get('action'):
                # Aynı sektör
                if pos.get('sector') == signal.get('sector'):
                    correlations.append(0.8)
                # Farklı sektör
                else:
                    correlations.append(0.3)
            # Ters yön
            else:
                correlations.append(-0.2)
                
        return np.mean(correlations) if correlations else 0
        
    def update_portfolio(self, pnl: float):
        """Portföy değerini güncelle"""
        
        self.portfolio_value += pnl
        self.daily_pnl += pnl
        
        # Peak tracking
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
            
        # Drawdown hesapla
        self.current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        
        # Drawdown limit kontrolü
        if self.current_drawdown > self.max_drawdown:
            logger.warning(f"Max drawdown reached: {self.current_drawdown:.2%}")
            asyncio.create_task(self.emergency_close_all())
            
    async def emergency_close_all(self):
        """Acil durumda tüm pozisyonları kapat"""
        
        logger.error("EMERGENCY: Closing all positions!")
        
        for pos in self.open_positions:
            # Pozisyonları kapat
            pass
            
        self.open_positions = []
        
    def get_risk_metrics(self) -> Dict:
        """Risk metriklerini raporla"""
        
        return {
            'portfolio_value': self.portfolio_value,
            'peak_value': self.peak_value,
            'current_drawdown': self.current_drawdown,
            'daily_pnl': self.daily_pnl,
            'open_positions': len(self.open_positions),
            'daily_trades': self.daily_trades,
            'var_95': self.calculate_var(self.portfolio_value, 1),
            'sector_exposure': {
                sector: self.get_sector_exposure(sector) 
                for sector in set(p.get('sector', 'unknown') for p in self.open_positions)
            }
        }