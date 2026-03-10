from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime

class Trade(BaseModel):
    entry_time: datetime
    entry_price: float
    side: str  # "BUY" or "SELL"
    size: float
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None

    def close_trade(self, time: datetime, price: float):
        self.exit_time = time
        self.exit_price = price
        
        # Kar/Zarar Hesaplama (Long işlemler için basitleştirilmiş)
        if self.side == "BUY":
            self.pnl = (self.exit_price - self.entry_price) * self.size
            self.pnl_percent = (self.exit_price - self.entry_price) / self.entry_price * 100
        elif self.side == "SELL":
            self.pnl = (self.entry_price - self.exit_price) * self.size
            self.pnl_percent = (self.entry_price - self.exit_price) / self.entry_price * 100

class Portfolio(BaseModel):
    initial_balance: float
    current_balance: float
    equity: float
    open_trades: List[Trade] = []
    closed_trades: List[Trade] = []

    def open_trade(self, time: datetime, price: float, side: str, amount_percent: float):
        # amount_percent = portföyün ne kadarı ile işleme girilecek (100 = hepsi)
        position_value = (self.current_balance * amount_percent) / 100
        size = position_value / price
        
        trade = Trade(entry_time=time, entry_price=price, side=side, size=size)
        
        # Basitlik açısından bakiyeden anında düşüyoruz (komisyon hariç)
        self.current_balance -= position_value
        self.open_trades.append(trade)
        return trade
        
    def close_trade(self, trade_index: int, time: datetime, price: float):
        if trade_index < len(self.open_trades):
            trade = self.open_trades.pop(trade_index)
            trade.close_trade(time, price)
            
            # İşlemden dönen anapara + kar/zarar bakiyeye eklenir
            position_value_on_exit = trade.size * price
            self.current_balance += position_value_on_exit
            
            self.closed_trades.append(trade)
            self.equity = self.current_balance
            return trade
        return None

    def calculate_metrics(self) -> dict:
        total_trades = len(self.closed_trades)
        winning_trades = sum(1 for t in self.closed_trades if (t.pnl or 0) > 0)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_profit = sum(t.pnl for t in self.closed_trades if t.pnl)
        
        return {
            "initial_balance": self.initial_balance,
            "final_balance": self.current_balance,
            "total_return_percent": ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": total_profit
        }
