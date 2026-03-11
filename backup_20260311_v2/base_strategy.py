"""
Base Strategy — Tüm stratejilerin ana sınıfı
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class Signal:
    """Strateji sinyali"""
    direction: Optional[str]   # 'LONG', 'SHORT', None
    confidence: float          # 0.0 - 1.0
    strategy_name: str         # Hangi strateji üretti
    regime: str                # Hangi rejimde üretildi
    reason: str                # İnsan-okunabilir neden
    entry_price: float = 0.0
    tp_atr_mult: float = 2.0   # TP = entry ± tp_mult × ATR
    sl_atr_mult: float = 1.3   # SL = entry ∓ sl_mult × ATR

    @property
    def is_valid(self) -> bool:
        return self.direction is not None and self.confidence > 0.0

    def __repr__(self):
        if not self.is_valid:
            return f"Signal(HOLD | {self.strategy_name})"
        return (f"Signal({self.direction} | conf={self.confidence:.2f} | "
                f"{self.strategy_name} | {self.reason})")


class BaseStrategy(ABC):
    """Tüm stratejilerin implement etmesi gereken arayüz"""

    name: str = "base"
    regime: str = "unknown"

    # Her strateji kendi TP/SL oranlarını belirler
    default_tp_mult: float = 2.0
    default_sl_mult: float = 1.3

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        DataFrame'den sinyal üret.
        En az son 50 bar gerekli.
        """
        pass

    def _no_signal(self, reason: str = "") -> Signal:
        """Sinyal yok döndür"""
        return Signal(
            direction=None,
            confidence=0.0,
            strategy_name=self.name,
            regime=self.regime,
            reason=reason or "No signal",
        )

    def _long_signal(self, confidence: float, reason: str,
                     entry_price: float = 0.0) -> Signal:
        return Signal(
            direction='LONG',
            confidence=confidence,
            strategy_name=self.name,
            regime=self.regime,
            reason=reason,
            entry_price=entry_price,
            tp_atr_mult=self.default_tp_mult,
            sl_atr_mult=self.default_sl_mult,
        )

    def _short_signal(self, confidence: float, reason: str,
                      entry_price: float = 0.0) -> Signal:
        return Signal(
            direction='SHORT',
            confidence=confidence,
            strategy_name=self.name,
            regime=self.regime,
            reason=reason,
            entry_price=entry_price,
            tp_atr_mult=self.default_tp_mult,
            sl_atr_mult=self.default_sl_mult,
        )
