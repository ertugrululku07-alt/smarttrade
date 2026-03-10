"""
Adaptive Regime + Meta-Label Backtest Engine v1.0

Tüm sistemi geçmiş veri üzerinde test eder:
  1. Her bar'da rejim tespit
  2. Rejime uygun sinyal üret
  3. Meta-model ile filtrele
  4. TP/SL simülasyonu
  5. PnL, WR, PF, Sharpe hesapla

Kullanım:
  python -m ai.adaptive_backtest --timeframe 15m --limit 3000
"""

import os
import sys
import gc
import json
import warnings
import numpy as np
import pandas as pd
import time as _time
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators
from ai.data_sources.futures_data import enrich_ohlcv_with_futures
from ai.xgboost_trainer import generate_features
from ai.regime_detector import Regime, detect_regime
from ai.meta_labelling.meta_predictor import MetaPredictor
from ai.adaptive_engine import AdaptiveEngine, TradeDecision


MAKER_FEE = 0.0004

DEFAULT_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT',
    'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT',
    'UNIUSDT', 'LTCUSDT', 'ATOMUSDT', 'ETCUSDT', 'BCHUSDT',
    'FTMUSDT', 'NEARUSDT', 'RUNEUSDT', 'AAVEUSDT', 'SANDUSDT',
    'GRTUSDT', 'INJUSDT', 'TIAUSDT', 'WLDUSDT',
]


# ===================================================================
# Trade Kaydi
# ===================================================================

@dataclass
class TradeRecord:
    """Tek bir trade kaydı"""
    symbol: str
    bar_index: int
    timestamp: str
    direction: str          # 'LONG' / 'SHORT'
    entry_price: float
    exit_price: float
    tp_price: float
    sl_price: float
    atr_at_entry: float
    pnl_pct: float          # Yüzde PnL
    pnl_atr: float          # ATR cinsinden PnL
    outcome: str            # 'TP' / 'SL' / 'TRAIL' / 'TIMEOUT'
    regime: str
    strategy: str
    signal_confidence: float
    meta_confidence: float
    position_size: float
    bars_held: int
    is_winner: bool
    fee_paid: float


@dataclass
class BacktestResult:
    """Backtest sonucu"""
    trades: List[TradeRecord] = field(default_factory=list)
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl_pct: float = 0.0
    total_pnl_atr: float = 0.0
    avg_pnl_per_trade: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    avg_bars_held: float = 0.0
    signals_generated: int = 0
    signals_filtered: int = 0
    filter_rate: float = 0.0
    regime_stats: Dict = field(default_factory=dict)
    strategy_stats: Dict = field(default_factory=dict)
    by_direction: Dict = field(default_factory=dict)
    equity_curve: List[float] = field(default_factory=list)


# ===================================================================
# Backtest Engine
# ===================================================================

class AdaptiveBacktest:
    """
    Adaptive Regime + Meta-Label backtest motoru.

    Özellikler:
      - Bar-by-bar simülasyon
      - Trailing stop desteği
      - Rejim bazlı strateji seçimi
      - Meta-model filtreleme
      - Position sizing (Kelly)
      - Detaylı istatistikler
    """

    def __init__(
        self,
        timeframe: str = "15m",
        trail_activation: float = 0.6,
        regime_lookback: int = 50,
        initial_capital: float = 10000.0,
        max_concurrent_trades: int = 3,
        use_meta_filter: bool = True,
    ):
        self.timeframe = timeframe
        self.trail_activation = trail_activation
        self.regime_lookback = regime_lookback
        self.initial_capital = initial_capital
        self.max_concurrent = max_concurrent_trades
        self.use_meta_filter = use_meta_filter

        # ── v1.4: AdaptiveEngine-v1.4 ───────────────────────
        self.engine = AdaptiveEngine(timeframe=timeframe)
        if not use_meta_filter:
            self.engine.meta_predictor = None

        # TF bazlı lookahead
        self.lookahead = self._get_lookahead(timeframe)
        self.lookahead = self._get_lookahead(timeframe)

        # ── v1.3: Debug sayaçları ────────────────────────────
        self._debug_counts = {
            'total_steps': 0,
            'signals_generated': 0,
            'signals_filtered': 0,
            'no_strategy': 0,
            'no_signal': 0,
            'meta_rejected': 0,
            'regime_distribution': {},
        }

    def _get_lookahead(self, tf: str) -> int:
        mapping = {
            "1m": 16, "3m": 32, "5m": 24, "15m": 16,
            "30m": 12, "1h": 12, "4h": 12, "1d": 10,
        }
        return mapping.get(tf, 16)

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        verbose: bool = False,
    ) -> BacktestResult:
        """
        Tek coin üzerinde backtest çalıştır.
        """
        N = len(df)
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        atrs = df['atr'].values

        result = BacktestResult()
        result.equity_curve = [self.initial_capital]

        capital = self.initial_capital
        open_trades = []  # Açık pozisyonlar

        signals_total = 0
        signals_filtered = 0

        # Rejim/strateji istatistikleri
        regime_trades = {r.value: {'trades': 0, 'wins': 0, 'pnl': 0.0}
                         for r in Regime}
        strategy_trades = {}
        direction_stats = {'LONG': {'trades': 0, 'wins': 0, 'pnl': 0.0},
                           'SHORT': {'trades': 0, 'wins': 0, 'pnl': 0.0}}

        for i in range(self.regime_lookback, N - 1):
            atr_val = atrs[i]
            if np.isnan(atr_val) or atr_val < 1e-10:
                result.equity_curve.append(capital)
                continue

            # ── Açık pozisyonları güncelle ────────────────────
            closed_indices = []
            for t_idx, trade in enumerate(open_trades):
                outcome = self._check_trade_exit(
                    trade, highs[i], lows[i], closes[i], i
                )
                if outcome is not None:
                    # Trade kapandı
                    trade_record = self._close_trade(
                        trade, outcome, closes[i], i, symbol
                    )
                    result.trades.append(trade_record)

                    # Capital güncelle
                    trade_pnl = trade_record.pnl_pct / 100 * capital * trade_record.position_size
                    capital += trade_pnl

                    # İstatistik güncelle
                    regime_trades[trade['regime']]['trades'] += 1
                    regime_trades[trade['regime']]['pnl'] += trade_record.pnl_atr
                    direction_stats[trade['direction']]['trades'] += 1
                    direction_stats[trade['direction']]['pnl'] += trade_record.pnl_atr

                    sname = trade.get('strategy', 'unknown')
                    if sname not in strategy_trades:
                        strategy_trades[sname] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
                    strategy_trades[sname]['trades'] += 1

                    if trade_record.is_winner:
                        regime_trades[trade['regime']]['wins'] += 1
                        direction_stats[trade['direction']]['wins'] += 1
                        strategy_trades[sname]['wins'] += 1
                    strategy_trades[sname]['pnl'] += trade_record.pnl_atr

                    closed_indices.append(t_idx)

            # Kapalı trade'leri kaldır
            for idx in sorted(closed_indices, reverse=True):
                open_trades.pop(idx)

            result.equity_curve.append(capital)

            # ── Max concurrent kontrol ───────────────────────
            if len(open_trades) >= self.max_concurrent:
                continue

            # ── v1.4: Karar Mekanizmasını Engine'e Bırak ──────
            decision = self.engine.decide(df.iloc[:i + 1])
            
            # Rejim istatistiği için
            regime = decision.regime
            
            if decision.action == 'HOLD':
                continue

            signals_total += 1
            meta_conf = decision.meta_confidence
            position_size = decision.position_size

            # Trade verileri (Signal nesnesi gibi davranalım)
            # engine.decide() içindeki gizli signal verilerine erişim
            # engine._last_signal'dan çekebiliriz (debug için eklemiştik)
            signal = getattr(self.engine, '_last_signal', None)
            if not signal: continue
            
            # ── Aynı yönde açık trade var mı? ────────────────
            same_dir_open = any(
                t['direction'] == signal.direction for t in open_trades
            )
            if same_dir_open:
                continue

            # ── Trade aç ─────────────────────────────────────
            entry = closes[i]
            tp_mult = signal.tp_atr_mult
            sl_mult = signal.sl_atr_mult

            if signal.direction == 'LONG':
                tp_price = entry + atr_val * tp_mult
                sl_price = entry - atr_val * sl_mult
            else:
                tp_price = entry - atr_val * tp_mult
                sl_price = entry + atr_val * sl_mult

            trade = {
                'direction': signal.direction,
                'entry_price': entry,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'tp_mult': tp_mult,
                'sl_mult': sl_mult,
                'atr': atr_val,
                'entry_bar': i,
                'regime': regime,
                'strategy': signal.strategy_name,
                'signal_conf': signal.confidence,
                'meta_conf': meta_conf,
                'position_size': position_size,
                'best_price': entry,
                'trail_sl': sl_price,
                'trail_activated': False,
                'reason': signal.reason,
            }
            open_trades.append(trade)

            if verbose:
                print(f"  [{i}] OPEN {signal.direction}: "
                      f"entry={entry:.4f} tp={tp_price:.4f} sl={sl_price:.4f} "
                      f"regime={regime.value} meta={meta_conf:.2f} "
                      f"size={position_size:.0%}")

        # ── Açık kalan trade'leri timeout ile kapat ──────────
        for trade in open_trades:
            trade_record = self._close_trade(
                trade, {'type': 'TIMEOUT', 'exit_price': closes[-1]},
                closes[-1], N - 1, symbol,
            )
            result.trades.append(trade_record)

        # ── v1.3: Debug stats ────────────────────────────────
        # Engine'den debug bilgisi al (eğer AdaptiveEngine kullanıyorsak)

        # Sonuçları hesapla
        result = self._compute_stats(
            result, signals_total, signals_filtered,
            regime_trades, strategy_trades, direction_stats,
        )

        return result

    def _check_trade_exit(
        self, trade: Dict, high: float, low: float, close: float, bar: int
    ) -> Optional[Dict]:
        """Açık trade'in TP/SL/Trail/Timeout kontrolü"""

        direction = trade['direction']
        entry = trade['entry_price']
        tp = trade['tp_price']
        sl = trade['sl_price']
        atr = trade['atr']
        entry_bar = trade['entry_bar']

        # Timeout kontrolü
        if bar - entry_bar >= self.lookahead:
            return {'type': 'TIMEOUT', 'exit_price': close}

        if direction == 'LONG':
            # Best price güncelle
            if high > trade['best_price']:
                trade['best_price'] = high

            # Trailing stop aktivasyonu
            trail_level = entry + (tp - entry) * self.trail_activation
            if trade['best_price'] >= trail_level:
                trade['trail_activated'] = True

            if trade['trail_activated']:
                new_trail = trade['best_price'] - atr * 1.5  # 1.0 -> 1.5
                trade['trail_sl'] = max(trade['trail_sl'], new_trail)
                if low <= trade['trail_sl']:
                    exit_price = max(trade['trail_sl'], low)
                    return {'type': 'TRAIL', 'exit_price': exit_price}

            # Normal TP
            if high >= tp:
                return {'type': 'TP', 'exit_price': tp}

            # Normal SL
            if low <= sl:
                return {'type': 'SL', 'exit_price': sl}

        else:  # SHORT
            if low < trade['best_price']:
                trade['best_price'] = low

            trail_level = entry - (entry - tp) * self.trail_activation
            if trade['best_price'] <= trail_level:
                trade['trail_activated'] = True

            if trade['trail_activated']:
                new_trail = trade['best_price'] + atr * 1.5  # 1.0 -> 1.5
                trade['trail_sl'] = min(trade['trail_sl'], new_trail)
                if high >= trade['trail_sl']:
                    exit_price = min(trade['trail_sl'], high)
                    return {'type': 'TRAIL', 'exit_price': exit_price}

            if low <= tp:
                return {'type': 'TP', 'exit_price': tp}

            if high >= sl:
                return {'type': 'SL', 'exit_price': sl}

        return None

    def _close_trade(
        self, trade: Dict, outcome: Dict, close: float, bar: int, symbol: str
    ) -> TradeRecord:
        """Trade'i kapat ve kayıt oluştur"""
        exit_price = outcome['exit_price']
        entry = trade['entry_price']
        direction = trade['direction']
        atr = trade['atr']

        fee = MAKER_FEE * 2  # Giriş + çıkış

        if direction == 'LONG':
            pnl_raw = (exit_price - entry) / entry
        else:
            pnl_raw = (entry - exit_price) / entry

        pnl_pct = (pnl_raw - fee) * 100
        pnl_atr = (exit_price - entry) / atr if direction == 'LONG' \
            else (entry - exit_price) / atr

        bars_held = bar - trade['entry_bar']
        is_winner = pnl_pct > 0

        return TradeRecord(
            symbol=symbol,
            bar_index=trade['entry_bar'],
            timestamp=str(bar),
            direction=direction,
            entry_price=entry,
            exit_price=exit_price,
            tp_price=trade['tp_price'],
            sl_price=trade['sl_price'],
            atr_at_entry=atr,
            pnl_pct=round(pnl_pct, 4),
            pnl_atr=round(pnl_atr, 4),
            outcome=outcome['type'],
            regime=trade['regime'],
            strategy=trade['strategy'],
            signal_confidence=trade['signal_conf'],
            meta_confidence=trade['meta_conf'],
            position_size=trade['position_size'],
            bars_held=bars_held,
            is_winner=is_winner,
            fee_paid=round(fee * 100, 4),
        )

    def _position_size(
        self, signal_conf: float, meta_conf: float, regime: Regime
    ) -> float:
        """Kelly-inspired position sizing"""
        combined = signal_conf * 0.3 + meta_conf * 0.7

        regime_max = {
            Regime.TRENDING: 1.0,
            Regime.MEAN_REVERTING: 0.80,
            Regime.HIGH_VOLATILE: 0.50,
            Regime.LOW_VOLATILE: 0.60,
        }
        max_size = regime_max.get(regime, 0.50)

        if combined >= 0.75:
            size = 1.0
        elif combined >= 0.65:
            size = 0.75
        elif combined >= 0.58:
            size = 0.50
        elif combined >= 0.52:
            size = 0.25
        else:
            size = 0.0

        return min(size, max_size)

    def _compute_stats(
        self, result: BacktestResult,
        signals_total: int, signals_filtered: int,
        regime_trades: Dict, strategy_trades: Dict,
        direction_stats: Dict,
    ) -> BacktestResult:
        """Backtest istatistiklerini hesapla"""
        trades = result.trades

        result.total_trades = len(trades)
        result.signals_generated = signals_total
        result.signals_filtered = signals_filtered
        result.filter_rate = (
            signals_filtered / signals_total * 100
            if signals_total > 0 else 0
        )

        if not trades:
            return result

        result.winners = sum(1 for t in trades if t.is_winner)
        result.losers = result.total_trades - result.winners
        result.win_rate = result.winners / result.total_trades * 100

        # PnL
        pnls = [t.pnl_pct for t in trades]
        pnl_atrs = [t.pnl_atr for t in trades]
        result.total_pnl_pct = sum(pnls)
        result.total_pnl_atr = sum(pnl_atrs)
        result.avg_pnl_per_trade = np.mean(pnls)

        # Profit Factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        result.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else 999.0
        )

        # Sharpe Ratio (günlük yaklaşım)
        if len(pnls) > 1:
            result.sharpe_ratio = (
                np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)
            )

        # Max Drawdown
        equity = result.equity_curve
        peak = equity[0]
        max_dd = 0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown_pct = round(max_dd, 2)

        # Avg bars held
        result.avg_bars_held = np.mean([t.bars_held for t in trades])

        # Regime stats
        for regime_val, data in regime_trades.items():
            t = data['trades']
            w = data['wins']
            pnl = data['pnl']
            result.regime_stats[regime_val] = {
                'trades': t,
                'wins': w,
                'wr': round(w / t * 100, 1) if t > 0 else 0,
                'pnl_atr': round(pnl, 2),
                'avg_pnl': round(pnl / t, 4) if t > 0 else 0,
            }

        # Strategy stats
        result.strategy_stats = {}
        for sname, data in strategy_trades.items():
            t = data['trades']
            w = data['wins']
            pnl = data['pnl']
            result.strategy_stats[sname] = {
                'trades': t,
                'wins': w,
                'wr': round(w / t * 100, 1) if t > 0 else 0,
                'pnl_atr': round(pnl, 2),
            }

        # Direction stats
        for d in ['LONG', 'SHORT']:
            data = direction_stats[d]
            t = data['trades']
            w = data['wins']
            pnl = data['pnl']
            result.by_direction[d] = {
                'trades': t,
                'wins': w,
                'wr': round(w / t * 100, 1) if t > 0 else 0,
                'pnl_atr': round(pnl, 2),
            }

        return result


# ═══════════════════════════════════════════════════════════════════
# Çoklu Coin Backtest
# ═══════════════════════════════════════════════════════════════════

def run_full_backtest(
    symbols: Optional[List[str]] = None,
    timeframe: str = "15m",
    limit: int = 3000,
    use_meta_filter: bool = True,
    use_cache: bool = False,
    test_split: float = 0.30,
    verbose: bool = False,
) -> Dict:
    """
    Tüm coinler üzerinde backtest çalıştır.

    Args:
        test_split: Verinin son %X'ini test olarak kullan
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    print(f"\n{'═' * 65}")
    print(f"  📊 ADAPTIVE BACKTEST v1.0")
    print(f"  TF: {timeframe} | Coins: {len(symbols)} | "
          f"Meta-filter: {'ON' if use_meta_filter else 'OFF'}")
    print(f"{'=' * 65}\n")

    engine = AdaptiveBacktest(
        timeframe=timeframe,
        use_meta_filter=use_meta_filter,
    )

    fetcher = DataFetcher('binance')
    all_results = []
    all_trades = []
    skipped = []

    for idx, sym in enumerate(symbols):
        try:
            if use_cache:
                try:
                    from ai.data_cache import get_cached_ohlcv
                    df = get_cached_ohlcv(sym, timeframe, limit=limit, fetcher=fetcher)
                except ImportError:
                    df = fetcher.fetch_ohlcv(sym, timeframe, limit=limit)
            else:
                df = fetcher.fetch_ohlcv(sym, timeframe, limit=limit)

            if df is None or df.empty or len(df) < 300:
                skipped.append(sym)
                continue

            df = add_all_indicators(df)
            df = enrich_ohlcv_with_futures(df, sym, silent=True)
            df = generate_features(df)

            # Son %30'u test olarak kullan
            test_start = int(len(df) * (1 - test_split))
            test_df = df.iloc[test_start:].copy()

            if len(test_df) < 100:
                skipped.append(sym)
                continue

            # Backtest çalıştır
            bt_result = engine.run(test_df, symbol=sym, verbose=verbose)

            if bt_result.total_trades > 0:
                all_results.append({
                    'symbol': sym,
                    'result': bt_result,
                })
                all_trades.extend(bt_result.trades)

                if (idx + 1) % 5 == 0 or bt_result.total_trades > 20:
                    print(f"  [{idx + 1}/{len(symbols)}] {sym}: "
                          f"Trades={bt_result.total_trades} "
                          f"WR=%{bt_result.win_rate:.1f} "
                          f"PF={bt_result.profit_factor:.2f} "
                          f"PnL={bt_result.total_pnl_pct:.2f}%")
            else:
                if (idx + 1) % 10 == 0:
                    print(f"  [{idx + 1}/{len(symbols)}] {sym}: No trades")

            del df, test_df
            gc.collect()

            if not use_cache:
                _time.sleep(0.25)

        except Exception as e:
            print(f"  ⚠️ {sym}: {e}")
            skipped.append(sym)

    # ── v1.3: Engine debug stats ─────────────────────────
    # Backtest engine'den strateji istatistiklerini al
    if hasattr(engine, 'strategies'):
        print(f"\n  📊 Strateji Debug:")
        print(f"     Signals generated: {sum(1 for _ in all_trades)}")

    if not all_trades:
        print(f"\n  ❌ 0 trade! Olası sorunlar:")
        print(f"     1. Stratejiler hiç sinyal üretmiyor")
        print(f"     2. Meta-filter her şeyi reddediyor")
        print(f"     3. Confidence eşiği çok yüksek")
        print(f"     → --no-meta ile tekrar dene")
        return {"error": "No trades", "debug": "Check strategy thresholds"}

    summary = _compute_summary(all_trades, all_results, symbols, skipped)
    _print_summary(summary, use_meta_filter)

    return summary


def _compute_summary(
    all_trades: List[TradeRecord],
    all_results: List[Dict],
    symbols: List[str],
    skipped: List[str],
) -> Dict:
    """Toplam backtest istatistikleri"""
    total = len(all_trades)
    winners = sum(1 for t in all_trades if t.is_winner)
    losers = total - winners

    pnls = [t.pnl_pct for t in all_trades]
    pnl_atrs = [t.pnl_atr for t in all_trades]

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))

    # Rejim bazlı
    regime_agg = {}
    for t in all_trades:
        r = t.regime
        if r not in regime_agg:
            regime_agg[r] = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'pnl_atr': 0.0}
        regime_agg[r]['trades'] += 1
        regime_agg[r]['pnl'] += t.pnl_pct
        regime_agg[r]['pnl_atr'] += t.pnl_atr
        if t.is_winner:
            regime_agg[r]['wins'] += 1

    # Strateji bazlı
    strat_agg = {}
    for t in all_trades:
        s = t.strategy
        if s not in strat_agg:
            strat_agg[s] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
        strat_agg[s]['trades'] += 1
        strat_agg[s]['pnl'] += t.pnl_pct
        if t.is_winner:
            strat_agg[s]['wins'] += 1

    # Yön bazlı
    dir_agg = {'LONG': {'trades': 0, 'wins': 0, 'pnl': 0.0},
               'SHORT': {'trades': 0, 'wins': 0, 'pnl': 0.0}}
    for t in all_trades:
        dir_agg[t.direction]['trades'] += 1
        dir_agg[t.direction]['pnl'] += t.pnl_pct
        if t.is_winner:
            dir_agg[t.direction]['wins'] += 1

    # Outcome bazlı
    outcome_agg = {}
    for t in all_trades:
        o = t.outcome
        if o not in outcome_agg:
            outcome_agg[o] = {'count': 0, 'wins': 0}
        outcome_agg[o]['count'] += 1
        if t.is_winner:
            outcome_agg[o]['wins'] += 1

    sharpe = np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252) if len(pnls) > 1 else 0

    return {
        'total_trades': total,
        'winners': winners,
        'losers': losers,
        'win_rate': round(winners / total * 100, 2),
        'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999,
        'total_pnl_pct': round(sum(pnls), 2),
        'total_pnl_atr': round(sum(pnl_atrs), 2),
        'avg_pnl_per_trade': round(np.mean(pnls), 4),
        'median_pnl': round(np.median(pnls), 4),
        'sharpe_ratio': round(sharpe, 2),
        'avg_bars_held': round(np.mean([t.bars_held for t in all_trades]), 1),
        'coins_traded': len(set(t.symbol for t in all_trades)),
        'coins_total': len(symbols),
        'coins_skipped': len(skipped),
        'by_regime': {
            r: {
                'trades': d['trades'],
                'wr': round(d['wins'] / d['trades'] * 100, 1) if d['trades'] > 0 else 0,
                'pnl_pct': round(d['pnl'], 2),
                'pnl_atr': round(d['pnl_atr'], 2),
            }
            for r, d in regime_agg.items()
        },
        'by_strategy': {
            s: {
                'trades': d['trades'],
                'wr': round(d['wins'] / d['trades'] * 100, 1) if d['trades'] > 0 else 0,
                'pnl_pct': round(d['pnl'], 2),
            }
            for s, d in strat_agg.items()
        },
        'by_direction': {
            d: {
                'trades': data['trades'],
                'wr': round(data['wins'] / data['trades'] * 100, 1) if data['trades'] > 0 else 0,
                'pnl_pct': round(data['pnl'], 2),
            }
            for d, data in dir_agg.items()
        },
        'by_outcome': outcome_agg,
    }


def _print_summary(summary: Dict, use_meta: bool):
    """Sonuçları güzel formatta yazdır"""
    meta_str = "META-FILTER ON ✅" if use_meta else "META-FILTER OFF ❌"

    print(f"\n{'═' * 65}")
    print(f"  🏆 BACKTEST SONUÇLARI ({meta_str})")
    print(f"{'═' * 65}")
    print(f"  📊 Total Trades : {summary['total_trades']:,}")
    print(f"  ✅ Winners      : {summary['winners']:,}")
    print(f"  ❌ Losers       : {summary['losers']:,}")
    print(f"  🎯 Win Rate     : %{summary['win_rate']}")
    print(f"  💰 Profit Factor: {summary['profit_factor']}")
    print(f"  📈 Total PnL    : %{summary['total_pnl_pct']} "
          f"({summary['total_pnl_atr']:.1f} ATR)")
    print(f"  💵 Avg PnL/Trade: %{summary['avg_pnl_per_trade']:.4f}")
    print(f"  📉 Sharpe Ratio : {summary['sharpe_ratio']}")
    print(f"  ⏱️  Avg Hold     : {summary['avg_bars_held']:.1f} bars")
    print(f"  🪙 Coins        : {summary['coins_traded']} / {summary['coins_total']}")

    print(f"\n{'─' * 55}")
    print(f"  📊 REJİM BAZLI:")
    for regime, data in summary['by_regime'].items():
        print(f"    {regime:16}: T={data['trades']:4} "
              f"WR=%{data['wr']:5.1f} PnL=%{data['pnl_pct']:8.2f} "
              f"({data['pnl_atr']:.1f} ATR)")

    print(f"\n{'─' * 55}")
    print(f"  📊 STRATEJİ BAZLI:")
    for strat, data in summary['by_strategy'].items():
        print(f"    {strat:16}: T={data['trades']:4} "
              f"WR=%{data['wr']:5.1f} PnL=%{data['pnl_pct']:8.2f}")

    print(f"\n{'─' * 55}")
    print(f"  📊 YÖN BAZLI:")
    for d, data in summary['by_direction'].items():
        print(f"    {d:16}: T={data['trades']:4} "
              f"WR=%{data['wr']:5.1f} PnL=%{data['pnl_pct']:8.2f}")

    print(f"\n{'─' * 55}")
    print(f"  📊 SONUÇ BAZLI:")
    for outcome, data in summary.get('by_outcome', {}).items():
        ct = data['count']
        w = data['wins']
        wr = w / ct * 100 if ct > 0 else 0
        print(f"    {outcome:8}: {ct:4} trade | WR=%{wr:.1f}")

    print(f"{'=' * 65}\n")


# ═══════════════════════════════════════════════════════════════════
# A/B Test: Meta-filter ON vs OFF
# ═══════════════════════════════════════════════════════════════════

def run_ab_test(
    symbols: Optional[List[str]] = None,
    timeframe: str = "15m",
    limit: int = 3000,
    use_cache: bool = False,
) -> Dict:
    """Meta-filter ON vs OFF karşılaştırması"""
    print(f"\n{'=' * 65}")
    print(f"  [DEBUG] A/B TEST: Meta-Filter ON vs OFF")
    print(f"{'=' * 65}\n")

    # A: Meta-filter OFF
    print("--- A: META-FILTER OFF ---")
    result_off = run_full_backtest(
        symbols=symbols, timeframe=timeframe, limit=limit,
        use_meta_filter=False, use_cache=use_cache,
    )

    # B: Meta-filter ON
    print("\n--- B: META-FILTER ON ---")
    result_on = run_full_backtest(
        symbols=symbols, timeframe=timeframe, limit=limit,
        use_meta_filter=True, use_cache=use_cache,
    )

    # Karşılaştırma
    print(f"\n  {'-' * 50}")
    print(f"  BACKTEST DEBUG STATS")
    print(f"  {'-' * 50}")

    metrics = ['total_trades', 'win_rate', 'profit_factor',
               'total_pnl_pct', 'sharpe_ratio', 'avg_pnl_per_trade']

    for m in metrics:
        val_off = result_off.get(m, 0)
        val_on = result_on.get(m, 0)

        if isinstance(val_off, (int, float)) and isinstance(val_on, (int, float)):
            diff = val_on - val_off
            arrow = "[UP]" if diff > 0 else "[DOWN]" if diff < 0 else "[=]"
            print(f"  {m:20}: OFF={val_off:10} | ON={val_on:10} | "
                  f"Diff={diff:+.2f} {arrow}")

    print(f"{'=' * 65}\n")

    return {
        'without_meta': result_off,
        'with_meta': result_on,
    }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive Backtest v1.0")
    parser.add_argument('--timeframe', type=str, default='15m')
    parser.add_argument('--limit', type=int, default=3000)
    parser.add_argument('--symbols', type=str, default='')
    parser.add_argument('--no-meta', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--ab-test', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    syms = [s.strip() for s in args.symbols.split(',') if s.strip()] or None

    if args.ab_test:
        run_ab_test(
            symbols=syms, timeframe=args.timeframe,
            limit=args.limit, use_cache=args.cache,
        )
    else:
        run_full_backtest(
            symbols=syms, timeframe=args.timeframe, limit=args.limit,
            use_meta_filter=not args.no_meta,
            use_cache=args.cache, verbose=args.verbose,
        )
