"""
ICT/SMC Backtest Engine v1.0

Live trader'daki _ict_smc_signal mantığını bar-by-bar simülasyona çevirir.
Aynı ict_core.analyze() altyapısını kullanır.

Kullanım:
  from ai.ict_backtest import ICTBacktest
  engine = ICTBacktest(initial_capital=1000)
  result = engine.run(df_1h, df_4h=df_4h, symbol="BTC/USDT")
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai import ict_core
from ai.adaptive_backtest import TradeRecord, BacktestResult

MAKER_FEE = 0.0004


class ICTBacktest:
    """
    ICT/SMC backtest motoru.

    Live trader'daki _ict_smc_signal mantığını bar-by-bar simüle eder:
      - HTF (4h) yapı analizi → trend yönü
      - 1h ICT analizi → sinyal üretimi
      - Kalite skoru + minimum eşik
      - Kademeli TP (TP1 %30 kapat, TP2 %40 kapat, TP3 trail)
      - Yapısal SL trailing
      - CHoCH çıkış
      - Timeout
    """

    def __init__(
        self,
        initial_capital: float = 1000.0,
        min_quality: int = 8,
        min_rr: float = 2.0,
        max_sl_pct: float = 2.5,
        max_concurrent: int = 1,
        cooldown_bars: int = 6,
        timeout_bars: int = 48,
        use_partial_tp: bool = True,
    ):
        self.initial_capital = initial_capital
        self.min_quality = min_quality
        self.min_rr = min_rr
        self.max_sl_pct = max_sl_pct / 100.0
        self.max_concurrent = max_concurrent
        self.cooldown_bars = cooldown_bars
        self.timeout_bars = timeout_bars
        self.use_partial_tp = use_partial_tp

    # ══════════════════════════════════════════════════════════════
    # Ana run loop
    # ══════════════════════════════════════════════════════════════
    def run(
        self,
        df_1h: pd.DataFrame,
        df_4h: Optional[pd.DataFrame] = None,
        symbol: str = "UNKNOWN",
        verbose: bool = False,
    ) -> BacktestResult:
        """Tek coin üzerinde ICT/SMC backtest çalıştır."""
        from backtest.signals import add_all_indicators

        df = df_1h.copy()
        if 'atr' not in df.columns:
            df = add_all_indicators(df)

        N = len(df)
        if N < 80:
            return BacktestResult()

        closes = df['close'].astype(float).values
        highs = df['high'].astype(float).values
        lows = df['low'].astype(float).values

        result = BacktestResult()
        result.equity_curve = [self.initial_capital]
        capital = self.initial_capital

        open_trades: List[Dict] = []
        last_trade_bar = -20
        consecutive_sl = 0
        cooldown_until = 0
        signals_total = 0

        # Stats
        strategy_trades: Dict = {}
        regime_trades: Dict = {}
        direction_stats = {
            'LONG': {'trades': 0, 'wins': 0, 'pnl': 0.0},
            'SHORT': {'trades': 0, 'wins': 0, 'pnl': 0.0},
        }

        # HTF analiz (4h) — tüm veri üzerinde bir kez
        htf_ms = 'ranging'
        if df_4h is not None and len(df_4h) >= 30:
            try:
                if 'atr' not in df_4h.columns:
                    df_4h = add_all_indicators(df_4h.copy())
                ict_htf = ict_core.analyze(df_4h, '', swing_left=5, swing_right=3)
                htf_ms = ict_htf.market_structure or 'ranging'
            except Exception:
                htf_ms = 'ranging'

        # ── Bar-by-bar simülasyon ──
        lookback = 60  # ICT analizi için gereken minimum bar

        for i in range(lookback, N - 1):
            # ── Açık trade çıkış kontrolü ──
            closed_indices = []
            for t_idx, trade in enumerate(open_trades):
                outcome = self._check_exit(
                    trade, highs[i], lows[i], closes[i], i,
                    df.iloc[max(0, i - 30):i + 1],
                )
                if outcome is not None:
                    rec = self._close_trade(trade, outcome, closes[i], i, symbol)
                    result.trades.append(rec)
                    trade_pnl = rec.pnl_pct / 100 * capital * rec.position_size
                    capital += trade_pnl
                    closed_indices.append(t_idx)

                    # Stats tracking
                    d = trade['direction']
                    s = trade.get('strategy', 'ict_smc')
                    r = trade.get('regime', 'ict')
                    direction_stats[d]['trades'] += 1
                    direction_stats[d]['pnl'] += rec.pnl_atr
                    if s not in strategy_trades:
                        strategy_trades[s] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
                    strategy_trades[s]['trades'] += 1
                    if r not in regime_trades:
                        regime_trades[r] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
                    regime_trades[r]['trades'] += 1

                    if rec.is_winner:
                        direction_stats[d]['wins'] += 1
                        strategy_trades[s]['wins'] += 1
                        regime_trades[r]['wins'] += 1
                        consecutive_sl = 0
                    else:
                        if outcome['type'] == 'SL':
                            consecutive_sl += 1
                            if consecutive_sl >= 3:
                                cooldown_until = i + self.cooldown_bars * 2
                                consecutive_sl = 0

                    strategy_trades[s]['pnl'] += rec.pnl_atr
                    regime_trades[r]['pnl'] += rec.pnl_atr

            for idx in sorted(closed_indices, reverse=True):
                open_trades.pop(idx)

            result.equity_curve.append(capital)

            # ── Yeni sinyal ara ──
            if len(open_trades) >= self.max_concurrent:
                continue
            if i - last_trade_bar < self.cooldown_bars:
                continue
            if i < cooldown_until:
                continue

            # ICT analizi (son 60 bar penceresi)
            window = df.iloc[max(0, i - lookback):i + 1].copy()
            sig = self._generate_signal(window, htf_ms, closes[i])
            if sig is None:
                continue

            signals_total += 1

            # Trade aç
            trade = {
                'direction': sig['direction'],
                'entry_price': sig['entry_price'],
                'tp_price': sig['tp_price'],
                'sl_price': sig['sl_price'],
                'atr': sig['atr'],
                'entry_bar': i,
                'regime': htf_ms,
                'strategy': 'ict_smc',
                'signal_conf': sig.get('quality', 0) / 20.0,
                'meta_conf': 0.0,
                'position_size': 1.0,
                'best_price': sig['entry_price'],
                'quality': sig.get('quality', 0),
                'rr': sig.get('rr', 0),
                # Partial TP tracking
                '_tp1_hit': False,
                '_tp2_hit': False,
                '_tp3_hit': False,
                '_trail_mode': False,
                '_original_qty_pct': 1.0,  # kalan pozisyon yüzdesi
            }
            open_trades.append(trade)
            last_trade_bar = i

            if verbose:
                print(
                    f"  [{i}] {sig['direction']} entry={sig['entry_price']:.2f} "
                    f"TP={sig['tp_price']:.2f} SL={sig['sl_price']:.2f} "
                    f"Q={sig.get('quality', 0)} RR={sig.get('rr', 0):.1f}"
                )

        # ── Kalan açık trade'leri kapat ──
        for trade in open_trades:
            rec = self._close_trade(
                trade,
                {'type': 'TIMEOUT', 'exit_price': closes[-1]},
                closes[-1], N - 1, symbol,
            )
            result.trades.append(rec)

        # İstatistikleri hesapla
        result = self._compute_stats(
            result, signals_total, 0,
            regime_trades, strategy_trades, direction_stats,
        )
        return result

    # ══════════════════════════════════════════════════════════════
    # ICT Sinyal Üretimi (live_trader._ict_smc_signal'ın backtest versiyonu)
    # ══════════════════════════════════════════════════════════════
    def _generate_signal(
        self, df_window: pd.DataFrame, htf_ms: str, current_price: float,
    ) -> Optional[Dict]:
        """Bar-by-bar ICT sinyal üretimi."""

        cp = current_price
        atr_val = float(df_window['atr'].iloc[-1]) if 'atr' in df_window.columns else cp * 0.01
        if np.isnan(atr_val) or atr_val <= 0:
            atr_val = cp * 0.01

        # ── 1) HTF yön belirleme ──
        if htf_ms == 'bullish':
            direction = 'LONG'
        elif htf_ms == 'bearish':
            direction = 'SHORT'
        else:
            # Ranging — 1h yapıya bak
            ict_1h_auto = ict_core.analyze(df_window, '', swing_left=3, swing_right=2)
            if ict_1h_auto.market_structure == 'bullish':
                direction = 'LONG'
            elif ict_1h_auto.market_structure == 'bearish':
                direction = 'SHORT'
            else:
                return None  # Yön belli değil → bekle

        # ── 2) 1H ICT analizi (yönlü) ──
        ict_1h = ict_core.analyze(df_window, direction, swing_left=3, swing_right=2)

        # ── 3) Yapı filtresi: BOS veya CHoCH gerekli ──
        has_bos = False
        has_choch = False
        dir_str = 'bullish' if direction == 'LONG' else 'bearish'

        for sb in ict_1h.structure_breaks:
            if sb.direction == dir_str:
                if sb.type == 'BOS':
                    has_bos = True
                elif sb.type == 'CHoCH':
                    has_choch = True

        if not has_bos and not has_choch:
            return None  # Yapı kırılması yok → sinyal yok

        # ── 4) Sweep kontrolü ──
        has_sweep = ict_1h.sweep_detected and ict_1h.sweep_type in (
            'SSL' if direction == 'LONG' else 'BSL',
            'SSL_sweep' if direction == 'LONG' else 'BSL_sweep',
        )

        # ── 5) Displacement kontrolü ──
        has_disp = (ict_1h.displacement and
                    ict_1h.displacement_direction == dir_str)

        # ── 6) POI Confluence ──
        ob_type = 'bullish' if direction == 'LONG' else 'bearish'
        active_obs = [ob for ob in ict_1h.order_blocks
                      if ob.type == ob_type and not ob.mitigated]
        active_fvgs = [fvg for fvg in ict_1h.fvg_zones
                       if fvg.type == ob_type and not fvg.filled]
        cc = ict_1h.poi_confluence

        # ── 7) POI Proximity Gate ──
        poi_ok, poi_dist, poi_reason = self._check_poi_proximity(
            direction, cp, ict_1h, atr_val)
        if not poi_ok:
            return None

        # ── 8) SL / TP hesaplama ──
        sl_price = ict_core.get_sweep_sl(
            direction, ict_1h.sweep_level, cp, atr_val,
            ict_1h.swing_highs, ict_1h.swing_lows,
        )

        # Max SL cap
        sl_dist = abs(cp - sl_price)
        if sl_dist > cp * self.max_sl_pct:
            if direction == 'LONG':
                sl_price = cp * (1 - self.max_sl_pct)
            else:
                sl_price = cp * (1 + self.max_sl_pct)
            sl_dist = abs(cp - sl_price)

        tp_price = ict_core.get_liquidity_tp(
            direction, cp,
            ict_1h.equal_highs, ict_1h.equal_lows,
            ict_1h.swing_highs, ict_1h.swing_lows,
        )

        # R:R kontrolü
        tp_dist = abs(tp_price - cp)
        if sl_dist <= 0:
            return None
        rr = tp_dist / sl_dist
        if rr < self.min_rr:
            # TP'yi ayarla
            tp_dist = sl_dist * self.min_rr
            if direction == 'LONG':
                tp_price = cp + tp_dist
            else:
                tp_price = cp - tp_dist
            rr = self.min_rr

        # ── 9) Kalite Skoru ──
        quality = 0
        if has_sweep:
            quality += 2
        if has_disp:
            quality += 2
        if has_bos:
            quality += 2
        if has_choch:
            quality += 3
        if cc >= 3:
            quality += 2
        elif cc >= 2:
            quality += 1
        if active_obs:
            quality += 1
        if active_fvgs:
            quality += 1
        # OTE zone
        in_ote = False
        if ict_1h.ote:
            ote = ict_1h.ote
            if ote.bottom <= cp <= ote.top:
                quality += 2
                in_ote = True
        # Volume spike
        if 'volume' in df_window.columns and 'vol_ma' in df_window.columns:
            vol = float(df_window['volume'].iloc[-1])
            vol_ma = float(df_window['vol_ma'].iloc[-1]) if 'vol_ma' in df_window.columns else vol
            if not np.isnan(vol) and not np.isnan(vol_ma) and vol_ma > 0:
                if vol > vol_ma * 1.5:
                    quality += 1
        # R:R bonus
        if rr >= 3.0:
            quality += 1
        # POI inside bonus
        if poi_reason in ('inside_ob', 'inside_fvg', 'inside_ote'):
            quality += 2

        # Minimum kalite eşiği
        min_q = self.min_quality
        if not in_ote and cc < 3:
            min_q = max(min_q, 10)

        if quality < min_q:
            return None

        return {
            'direction': direction,
            'entry_price': cp,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'atr': atr_val,
            'quality': quality,
            'rr': round(rr, 2),
            'has_sweep': has_sweep,
            'has_disp': has_disp,
            'has_bos': has_bos,
            'has_choch': has_choch,
            'confluence': cc,
            'poi_reason': poi_reason,
            'in_ote': in_ote,
        }

    # ══════════════════════════════════════════════════════════════
    # POI Proximity Gate (live_trader'dan taşındı)
    # ══════════════════════════════════════════════════════════════
    def _check_poi_proximity(self, direction, cp, ict_1h, atr):
        """Fiyat aktif POI'ye yeterince yakın mı? (max 2 ATR)"""
        ob_type = 'bullish' if direction == 'LONG' else 'bearish'
        max_distance = atr * 2.0

        best_ob_dist = float('inf')
        for ob in ict_1h.order_blocks:
            if ob.type == ob_type and not ob.mitigated:
                if ob.bottom <= cp <= ob.top:
                    return True, 0.0, 'inside_ob'
                if direction == 'LONG':
                    dist = cp - ob.top
                    if 0 <= dist <= max_distance:
                        best_ob_dist = min(best_ob_dist, dist)
                else:
                    dist = ob.bottom - cp
                    if 0 <= dist <= max_distance:
                        best_ob_dist = min(best_ob_dist, dist)

        best_fvg_dist = float('inf')
        for fvg in ict_1h.fvg_zones:
            if fvg.type == ob_type and not fvg.filled:
                if fvg.bottom <= cp <= fvg.top:
                    return True, 0.0, 'inside_fvg'
                dist = abs(cp - fvg.ce)
                if dist <= max_distance:
                    best_fvg_dist = min(best_fvg_dist, dist)

        if ict_1h.ote:
            ote = ict_1h.ote
            if ote.bottom <= cp <= ote.top:
                return True, 0.0, 'inside_ote'

        min_dist = min(best_ob_dist, best_fvg_dist)
        if min_dist <= max_distance:
            return True, min_dist, 'near_poi'

        return False, min_dist, 'too_far'

    # ══════════════════════════════════════════════════════════════
    # Çıkış Kontrolü (Partial TP + Trail + CHoCH + Timeout)
    # ══════════════════════════════════════════════════════════════
    def _check_exit(
        self, trade: Dict, high: float, low: float, close: float, bar: int,
        df_recent: pd.DataFrame = None,
    ) -> Optional[Dict]:
        """
        ICT çıkış mantığı:
          1. SL hit → tam çıkış
          2. Kademeli TP (TP1/TP2/TP3)
          3. Trail mode → CHoCH veya SL trail
          4. Timeout
        """
        direction = trade['direction']
        entry = trade['entry_price']
        tp = trade['tp_price']
        sl = trade['sl_price']
        entry_bar = trade['entry_bar']
        bars_held = bar - entry_bar

        # ── 1. SL check ──
        if direction == 'LONG' and low <= sl:
            return {'type': 'SL', 'exit_price': sl}
        if direction == 'SHORT' and high >= sl:
            return {'type': 'SL', 'exit_price': sl}

        # ── 2. Kademeli TP (partial simülasyonu) ──
        tp_dist = abs(tp - entry)
        if tp_dist > 0 and self.use_partial_tp:
            if direction == 'LONG':
                progress = (high - entry) / tp_dist if high > entry else 0
            else:
                progress = (entry - low) / tp_dist if low < entry else 0

            # TP3 hit → trail mode
            if progress >= 1.0 and not trade.get('_tp3_hit'):
                trade['_tp3_hit'] = True
                trade['_trail_mode'] = True
                trade['_trail_bar'] = bar
                # SL → kârın %75'ini koru
                if direction == 'LONG':
                    trade['sl_price'] = entry + tp_dist * 0.75
                else:
                    trade['sl_price'] = entry - tp_dist * 0.75
                # Pozisyon %30'u kaldı (simülasyon için PnL ayarlaması yapılacak)
                trade['_remaining_pct'] = 0.30
                # TP3'te kapatma yok — trail devam

            # TP2 hit → SL güncelle, %40 kapat (simülasyonda: pozisyon küçült)
            elif progress >= 0.65 and not trade.get('_tp2_hit'):
                trade['_tp2_hit'] = True
                if direction == 'LONG':
                    trade['sl_price'] = entry + tp_dist * 0.33
                else:
                    trade['sl_price'] = entry - tp_dist * 0.33
                trade['_remaining_pct'] = trade.get('_remaining_pct', 1.0) * 0.60

            # TP1 hit → SL → breakeven
            elif progress >= 0.33 and not trade.get('_tp1_hit'):
                trade['_tp1_hit'] = True
                buffer = entry * 0.001
                if direction == 'LONG':
                    trade['sl_price'] = max(trade['sl_price'], entry + buffer)
                else:
                    trade['sl_price'] = min(trade['sl_price'], entry - buffer)
                trade['_remaining_pct'] = trade.get('_remaining_pct', 1.0) * 0.70

        # ── 3. Trail mode ──
        if trade.get('_trail_mode'):
            trail_start = trade.get('_trail_bar', bar)

            # Trail timeout: 24 bar (~24h)
            if bar - trail_start > 24:
                return {'type': 'TRAIL_TIMEOUT', 'exit_price': close}

            # CHoCH yapı bozulma kontrolü (her 5 barda bir)
            if df_recent is not None and len(df_recent) >= 10 and bars_held % 5 == 0:
                try:
                    ict_trail = ict_core.analyze(
                        df_recent, direction, swing_left=3, swing_right=2)
                    bad_dir = 'bearish' if direction == 'LONG' else 'bullish'
                    if (ict_trail.last_choch and
                            ict_trail.last_choch.direction == bad_dir):
                        return {'type': 'CHOCH_EXIT', 'exit_price': close}
                except Exception:
                    pass

            # Yapısal SL trailing (her 5 barda)
            if df_recent is not None and len(df_recent) >= 10 and bars_held % 5 == 0:
                try:
                    ict_trail = ict_core.analyze(
                        df_recent, direction, swing_left=3, swing_right=2)
                    atr = trade['atr']
                    buf = atr * 0.2
                    if direction == 'LONG':
                        for sw in reversed(ict_trail.swing_lows):
                            candidate = sw.price - buf
                            if candidate > entry and candidate > trade['sl_price']:
                                trade['sl_price'] = candidate
                                break
                    else:
                        for sw in reversed(ict_trail.swing_highs):
                            candidate = sw.price + buf
                            if candidate < entry and candidate < trade['sl_price']:
                                trade['sl_price'] = candidate
                                break
                except Exception:
                    pass

            return None  # Trail mode'da timeout kontrolüne girme

        # ── 4. Basit TP (partial kapalı, sadece tam TP) ──
        if not self.use_partial_tp:
            if direction == 'LONG' and high >= tp:
                return {'type': 'TP', 'exit_price': tp}
            if direction == 'SHORT' and low <= tp:
                return {'type': 'TP', 'exit_price': tp}

        # ── 5. Timeout ──
        max_bars = self.timeout_bars * 2 if trade.get('_tp3_hit') else self.timeout_bars
        if bars_held >= max_bars:
            return {'type': 'TIMEOUT', 'exit_price': close}

        return None

    # ══════════════════════════════════════════════════════════════
    # Trade kapatma ve kayıt
    # ══════════════════════════════════════════════════════════════
    def _close_trade(
        self, trade: Dict, outcome: Dict, close: float, bar: int, symbol: str,
    ) -> TradeRecord:
        """Trade'i kapat ve TradeRecord oluştur."""
        exit_price = outcome['exit_price']
        entry = trade['entry_price']
        direction = trade['direction']
        atr = trade['atr']
        fee = MAKER_FEE * 2

        if direction == 'LONG':
            pnl_raw = (exit_price - entry) / entry
        else:
            pnl_raw = (entry - exit_price) / entry

        # Partial TP simülasyonu: kalan pozisyon yüzdesi ile PnL ayarla
        remaining_pct = trade.get('_remaining_pct', 1.0)
        # Partial kârlar: TP1 ve TP2'de kapatılan kısımların kârı
        partial_pnl = 0.0
        tp_dist_pct = abs(trade['tp_price'] - entry) / entry if entry > 0 else 0

        if trade.get('_tp1_hit') and self.use_partial_tp:
            # TP1'de %30 kapatıldı, %33 progress'te
            partial_pnl += 0.30 * (tp_dist_pct * 0.33 - fee)
        if trade.get('_tp2_hit') and self.use_partial_tp:
            # TP2'de %40 kapatıldı (orijinalin), %65 progress'te
            partial_pnl += 0.40 * (tp_dist_pct * 0.65 - fee)

        # Kalan kısmın PnL'si
        final_pnl = remaining_pct * (pnl_raw - fee) + partial_pnl
        pnl_pct = final_pnl * 100

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
            regime=trade.get('regime', 'ict'),
            strategy=trade.get('strategy', 'ict_smc'),
            signal_confidence=trade.get('signal_conf', 0),
            meta_confidence=trade.get('meta_conf', 0),
            position_size=trade.get('position_size', 1.0),
            bars_held=bars_held,
            is_winner=is_winner,
            fee_paid=round(fee * 100, 4),
        )

    # ══════════════════════════════════════════════════════════════
    # İstatistik hesaplama (adaptive_backtest ile aynı format)
    # ══════════════════════════════════════════════════════════════
    def _compute_stats(
        self, result: BacktestResult,
        signals_total: int, signals_filtered: int,
        regime_trades: Dict, strategy_trades: Dict,
        direction_stats: Dict,
    ) -> BacktestResult:
        """Backtest istatistiklerini hesapla."""
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

        pnls = [t.pnl_pct for t in trades]
        pnl_atrs = [t.pnl_atr for t in trades]
        result.total_pnl_pct = sum(pnls)
        result.total_pnl_atr = sum(pnl_atrs)
        result.avg_pnl_per_trade = np.mean(pnls)

        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        result.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else 999.0
        )

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

        result.avg_bars_held = np.mean([t.bars_held for t in trades])

        # Regime stats
        for r, data in regime_trades.items():
            t = data['trades']
            w = data['wins']
            pnl = data['pnl']
            result.regime_stats[r] = {
                'trades': t, 'wins': w,
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
                'trades': t, 'wins': w,
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
                'trades': t, 'wins': w,
                'wr': round(w / t * 100, 1) if t > 0 else 0,
                'pnl_atr': round(pnl, 2),
            }

        return result
