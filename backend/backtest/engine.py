"""
SmartTrade Backtest Engine v2 — Gerçek Sinyal Tabanlı

Bu motor:
1. CCXT ile Binance'den gerçek OHLCV verisi çeker
2. signals.py ile teknik indikatörleri hesaplar
3. Seçilen stratejiye göre BUY/SELL sinyalleri üretir
4. ATR tabanlı dinamik SL/TP uygular
5. Kapsamlı performans metrikleri hesaplar
6. LearnerAI hafızasına sonucu kaydeder
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from backtest.signals import add_all_indicators, generate_signals


class Trade:
    def __init__(self, entry_time, entry_price: float, side: str,
                 amount: float, tp: float, sl: float):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.side = side
        self.amount = amount          # Kaç USD kullanıldı
        self.units = amount / entry_price
        self.tp = tp
        self.sl = sl
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.close_reason = None
        self.status = "open"

    def close(self, exit_time, exit_price: float, reason: str):
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.close_reason = reason
        self.status = "closed"
        if self.side == "BUY":
            self.pnl = (exit_price - self.entry_price) * self.units
            self.pnl_pct = (exit_price / self.entry_price - 1) * 100
        else:  # SELL / SHORT
            self.pnl = (self.entry_price - exit_price) * self.units
            self.pnl_pct = (self.entry_price / exit_price - 1) * 100

    def to_dict(self) -> dict:
        return {
            "entry_time": str(self.entry_time),
            "exit_time": str(self.exit_time),
            "side": self.side,
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4) if self.exit_price else None,
            "pnl": round(self.pnl, 4),
            "pnl_pct": round(self.pnl_pct, 3),
            "close_reason": self.close_reason,
            "amount": round(self.amount, 2),
        }


class BacktestEngine:
    def __init__(self, df: pd.DataFrame, initial_balance: float = 1000.0,
                 trade_size_pct: float = 20.0, atr_tp_mult: float = 2.0,
                 atr_sl_mult: float = 1.0, max_open: int = 1):
        """
        :param df:              Çiğ OHLCV DataFrame (Binance'den)
        :param initial_balance: Başlangıç bakiyesi (USD)
        :param trade_size_pct:  Her işlemde kullanılacak bakiye yüzdesi
        :param atr_tp_mult:     ATR × bu çarpan = TP mesafesi
        :param atr_sl_mult:     ATR × bu çarpan = SL mesafesi
        :param max_open:        Aynı anda açık olabilecek max işlem
        """
        self.df = add_all_indicators(df)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trade_size_pct = trade_size_pct
        self.atr_tp_mult = atr_tp_mult
        self.atr_sl_mult = atr_sl_mult
        self.max_open = max_open
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_balance]
        self.strategy_config = []

    def load_strategy(self, config: list):
        self.strategy_config = config

    def _get_strategy_type(self) -> str:
        """Frontend blok dizisinden strateji tipini çıkar."""
        ids = [b.get("id", "") for b in self.strategy_config]
        ids_str = " ".join(ids)
        if "macd" in ids_str and "bollinger" in ids_str:
            return "swing"
        elif "ema" in ids_str:
            return "scalping"
        elif "rsi" in ids_str and not "macd" in ids_str:
            return "rsi_only"
        return "confluence"

    def _check_exits(self, i: int, row: pd.Series):
        """Her mumda açık işlemlerin TP/SL çarpıp çarpmadığını kontrol et."""
        still_open = []
        for trade in self.open_trades:
            high = row['high']
            low  = row['low']
            time = self.df.index[i]

            hit_tp = hit_sl = False
            if trade.side == "BUY":
                hit_tp = high >= trade.tp
                hit_sl = low  <= trade.sl
            else:
                hit_tp = low  <= trade.tp
                hit_sl = high >= trade.sl

            if hit_sl:
                exit_price = trade.sl
                trade.close(time, exit_price, "SL")
                self.balance += trade.amount + trade.pnl
                self.closed_trades.append(trade)
            elif hit_tp:
                exit_price = trade.tp
                trade.close(time, exit_price, "TP")
                self.balance += trade.amount + trade.pnl
                self.closed_trades.append(trade)
            else:
                still_open.append(trade)

        self.open_trades = still_open

    def _open_trade(self, i: int, side: str):
        """Yeni işlem aç (bakiye, ATR tabanlı SL/TP hesapla)."""
        if len(self.open_trades) >= self.max_open:
            return

        row = self.df.iloc[i]
        price = row['close']
        atr_val = row['atr']
        amount = self.balance * (self.trade_size_pct / 100)

        if amount < 1 or self.balance < 1:
            return

        if side == "BUY":
            tp = price + atr_val * self.atr_tp_mult
            sl = price - atr_val * self.atr_sl_mult
        else:
            tp = price - atr_val * self.atr_tp_mult
            sl = price + atr_val * self.atr_sl_mult

        self.balance -= amount
        trade = Trade(self.df.index[i], price, side, amount, tp, sl)
        self.open_trades.append(trade)

    def run(self, strategy_override: Optional[str] = None) -> Dict[str, Any]:
        """Ana backtest döngüsü. Gerçek sinyal hesaplamalarını kullanır."""
        strategy = strategy_override or self._get_strategy_type()
        signals, scores = generate_signals(self.df, strategy=strategy)
        self.df['signal'] = signals
        self.df['confluence_score'] = scores

        # Warm-up: İlk 50 mum hesaplama için kullanılır, işlem açılmaz
        warmup = 50

        for i in range(warmup, len(self.df)):
            row = self.df.iloc[i]
            sig = signals.iloc[i]

            # Önce TP/SL kontrol
            self._check_exits(i, row)

            # Sonra yeni sinyal
            if sig == "BUY" and len(self.open_trades) == 0:
                self._open_trade(i, "BUY")
            elif sig == "SELL":
                # Açık BUY işlemlerini kapat
                for t in self.open_trades[:]:
                    if t.side == "BUY":
                        t.close(self.df.index[i], row['close'], "Signal")
                        self.balance += t.amount + t.pnl
                        self.open_trades.remove(t)
                        self.closed_trades.append(t)

            # Equity kaydı
            unrealized = sum(
                (row['close'] - t.entry_price) * t.units if t.side == "BUY"
                else (t.entry_price - row['close']) * t.units
                for t in self.open_trades
            )
            self.equity_curve.append(round(self.balance + unrealized, 2))

        # Biten bakiye — kalan açık işlemleri son fiyata kapat
        final_price = self.df['close'].iloc[-1]
        final_time = self.df.index[-1]
        for t in self.open_trades[:]:
            t.close(final_time, final_price, "EndOfData")
            self.balance += t.amount + t.pnl
            self.closed_trades.append(t)
        self.open_trades = []

        return self._calculate_metrics(strategy)

    def _calculate_metrics(self, strategy: str) -> Dict[str, Any]:
        trades = self.closed_trades
        if not trades:
            return {
                "strategy": strategy, "total_trades": 0, "win_rate": 0,
                "total_pnl": 0, "total_pnl_pct": 0, "max_drawdown": 0,
                "sharpe_ratio": 0, "profit_factor": 0,
                "avg_win": 0, "avg_loss": 0,
                "equity_curve": self.equity_curve[:200:2],
                "trades": [],
            }

        wins  = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in trades)

        win_rate = len(wins) / len(trades) * 100
        avg_win  = np.mean([t.pnl for t in wins])  if wins  else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        profit_factor = (
            abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses))
            if losses and sum(t.pnl for t in losses) != 0 else float('inf')
        )

        # Max Drawdown
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        max_dd = float(abs(dd.min()))

        # Sharpe Ratio (daily returns approximation)
        if len(self.equity_curve) > 2:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(365) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        return {
            "strategy": strategy,
            "total_trades": len(trades),
            "win_trades": len(wins),
            "loss_trades": len(losses),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / self.initial_balance * 100, 2),
            "initial_balance": self.initial_balance,
            "final_balance": round(self.balance, 2),
            "max_drawdown": round(max_dd, 2),
            "sharpe_ratio": round(float(sharpe), 3),
            "profit_factor": round(float(profit_factor), 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "best_trade": round(max((t.pnl for t in trades), default=0), 2),
            "worst_trade": round(min((t.pnl for t in trades), default=0), 2),
            # Equity curve downsampled to max 200 points
            "equity_curve": self.equity_curve[::max(1, len(self.equity_curve) // 200)],
            "trades": [t.to_dict() for t in trades[-50:]],  # Son 50 işlem
        }
