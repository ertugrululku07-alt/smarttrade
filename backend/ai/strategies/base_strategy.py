"""
Base Strategy v1.1 — Tüm stratejilerin ana sınıfı

v1.1 Değişiklikler:
  - _validate_columns helper eklendi
  - _safe_get helper eklendi
  - confidence clamp [0, 1]
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import pandas as pd


@dataclass
class Signal:
    """Strateji sinyali"""
    direction: Optional[str]
    confidence: float
    strategy_name: str
    regime: str
    reason: str
    entry_price: float = 0.0
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.3

    hard_pass: bool = False
    soft_score: int = 0
    entry_type: str = "none"
    tp_price: float = 0.0
    sl_price: float = 0.0

    @property
    def is_valid(self) -> bool:
        return (self.direction is not None
                and self.hard_pass
                and self.soft_score >= 3)

    def __repr__(self):
        if not self.is_valid:
            if self.direction is None:
                return f"Signal(HOLD | {self.strategy_name} | {self.reason})"
            fail = "HardFail" if not self.hard_pass else f"LowScore({self.soft_score})"
            return f"Signal(HOLD | {fail} | {self.strategy_name})"
        return (f"Signal({self.direction} | {self.entry_type} | "
                f"score={self.soft_score} | conf={self.confidence:.2f} | "
                f"{self.strategy_name} | {self.reason})")


class BaseStrategy(ABC):
    """Tüm stratejilerin implement etmesi gereken arayüz"""

    name: str = "base"
    regime: str = "unknown"
    default_tp_mult: float = 2.0
    default_sl_mult: float = 1.3

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        pass

    # ══════════════════════════════════════════════════
    # v1.1 EKLENDİ: Helper Metodlar
    # ══════════════════════════════════════════════════

    def _validate_columns(self, df: pd.DataFrame,
                          required: List[str]) -> Optional[str]:
        """
        Gerekli kolonları kontrol et.
        Returns:
            str: Eksik kolon mesajı (hata varsa)
            None: Tüm kolonlar mevcut
        """
        missing = [col for col in required if col not in df.columns]
        if missing:
            return f"Eksik kolonlar: {missing}"
        return None

    def _safe_get(self, df: pd.DataFrame, col: str,
                  default=None, idx: int = -1):
        """
        Güvenli kolon erişimi.
        Kolon yoksa veya değer NaN ise default döndürür.
        """
        if col not in df.columns:
            return default
        val = df[col].iloc[idx]
        if pd.isna(val):
            return default
        return val

    # ══════════════════════════════════════════════════
    # Signal Builders (değişmedi)
    # ══════════════════════════════════════════════════

    def _no_signal(self, reason: str = "",
                   hard_pass: bool = False,
                   soft_score: int = 0) -> Signal:
        return Signal(
            direction=None,
            confidence=0.0,
            strategy_name=self.name,
            regime=self.regime,
            reason=reason or "No signal",
            hard_pass=hard_pass,
            soft_score=soft_score
        )

    def _long_signal(self, soft_score: int, reason: str,
                     entry_price: float = 0.0,
                     entry_type: str = "none",
                     hard_pass: bool = True,
                     tp_price: float = 0.0,
                     sl_price: float = 0.0) -> Signal:
        confidence = min(1.0, 0.5 + (soft_score / 10.0)) if hard_pass else 0.0
        return Signal(
            direction='LONG',
            confidence=confidence,
            strategy_name=self.name,
            regime=self.regime,
            reason=reason,
            entry_price=entry_price,
            tp_atr_mult=self.default_tp_mult,
            sl_atr_mult=self.default_sl_mult,
            hard_pass=hard_pass,
            soft_score=soft_score,
            entry_type=entry_type,
            tp_price=tp_price,
            sl_price=sl_price
        )

    def _short_signal(self, soft_score: int, reason: str,
                      entry_price: float = 0.0,
                      entry_type: str = "none",
                      hard_pass: bool = True,
                      tp_price: float = 0.0,
                      sl_price: float = 0.0) -> Signal:
        confidence = min(1.0, 0.5 + (soft_score / 10.0)) if hard_pass else 0.0
        return Signal(
            direction='SHORT',
            confidence=confidence,
            strategy_name=self.name,
            regime=self.regime,
            reason=reason,
            entry_price=entry_price,
            tp_atr_mult=self.default_tp_mult,
            sl_atr_mult=self.default_sl_mult,
            hard_pass=hard_pass,
            soft_score=soft_score,
            entry_type=entry_type,
            tp_price=tp_price,
            sl_price=sl_price
        )
