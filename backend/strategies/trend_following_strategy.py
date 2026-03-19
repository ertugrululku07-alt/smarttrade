"""
Trend Following v4.4 Strategy — Live Trading Mixin
===================================================

Backtest sonucu: +$88.30 (7 coin × 30 gün, 75 trade, %45.3 WR)

Entry:
  - Supertrend flip + Triple EMA alignment (9>21>50 veya 9<21<50)
  - EMA cross + MACD + Volume + ADX > 20

Exit:
  - SL: Swing low/high (5 bar), max %2 cap
  - Trail: +6% kardan sonra peak karın %35'ini kilitle
  - ST flip = karda çık (>%0.3)
  - Timeout 72 bar, Max $4 kayıp
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators


# ═══════════════════════════════════════════════════════════════════
#  Supertrend Hesaplama
# ═══════════════════════════════════════════════════════════════════

def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """Supertrend indicator — direction (+1 bullish, -1 bearish) + value."""
    hl2 = (df['high'] + df['low']) / 2
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1 / period, adjust=False).mean()

    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    st_direction = pd.Series(1, index=df.index, dtype=int)
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    for i in range(1, len(df)):
        if lower_band.iloc[i] > final_lower.iloc[i - 1] or df['close'].iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if upper_band.iloc[i] < final_upper.iloc[i - 1] or df['close'].iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if st_direction.iloc[i - 1] == 1:
            if df['close'].iloc[i] < final_lower.iloc[i]:
                st_direction.iloc[i] = -1
            else:
                st_direction.iloc[i] = 1
        else:
            if df['close'].iloc[i] > final_upper.iloc[i]:
                st_direction.iloc[i] = 1
            else:
                st_direction.iloc[i] = -1

    st_value = pd.Series(0.0, index=df.index)
    for i in range(len(df)):
        st_value.iloc[i] = final_lower.iloc[i] if st_direction.iloc[i] == 1 else final_upper.iloc[i]

    return st_direction, st_value


# ═══════════════════════════════════════════════════════════════════
#  Trend Following Mixin
# ═══════════════════════════════════════════════════════════════════

class TrendFollowingMixin:
    """
    Trend Following v4.4 — Live Trading Strategy Mixin.

    Backtest'ten kanıtlanmış strateji:
      +$88.30 / 7 coin / 30 gün / %45.3 WR

    ICTSMCStrategyMixin ve BBMRStrategyMixin ile aynı pattern'i takip eder.
    """

    # ── Parameters ──
    _TREND_PARAMS = {
        'st_period': 10,
        'st_multiplier': 3.0,
        'min_adx': 20,
        'min_vol_ratio': 0.8,
        'max_sl_pct': 0.02,
        'swing_lookback': 5,
        'be_threshold': 0.03,        # +3% price profit → move SL to breakeven
        'trail_start': 0.06,         # +6% price profit → trailing start
        'trail_keep': 0.55,          # Lock 55% of peak profit (was 35%)
        'timeout_hours': 72,
        'max_loss_cap': 4.0,         # $4 max loss per trade
        'max_notional': 300.0,
        'cooldown_hours': 2,
    }

    # ── State Initialization ──
    def _trend_init_state(self):
        if not hasattr(self, '_trend_last_trade_time'):
            self._trend_last_trade_time = 0
        if not hasattr(self, '_trend_consecutive_sl'):
            self._trend_consecutive_sl = 0
        if not hasattr(self, '_trend_cooldown_until'):
            self._trend_cooldown_until = 0
        if not hasattr(self, '_trend_recent_outcomes'):
            self._trend_recent_outcomes = []
        if not hasattr(self, '_trend_symbol_cooldown'):
            self._trend_symbol_cooldown = {}

    # ── Correlation Groups (shared with ICT) ──
    _TREND_CORRELATED_GROUPS = {
        'BTC_GROUP': {'BTC/USDT', 'ETH/USDT'},
        'ALT_L1': {'SOL/USDT', 'AVAX/USDT', 'DOT/USDT', 'ADA/USDT', 'NEAR/USDT'},
        'ALT_L2': {'LINK/USDT', 'AAVE/USDT', 'UNI/USDT'},
        'MEME': {'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'WIF/USDT', 'FLOKI/USDT'},
    }

    def _trend_check_correlation(self, symbol: str) -> bool:
        """Aynı korelasyon grubundan açık trend trade var mı?"""
        my_group = None
        for group_name, members in self._TREND_CORRELATED_GROUPS.items():
            if symbol in members:
                my_group = group_name
                break
        if my_group is None:
            return True  # Grupta değil, serbestçe aç

        with self.trades_lock:
            for t in self.open_trades:
                if t.get('strategy', '') != 'trend_v4.4':
                    continue
                t_sym = t.get('symbol', '')
                for members in self._TREND_CORRELATED_GROUPS.values():
                    if t_sym in members and symbol in members and t_sym != symbol:
                        return False
        return True

    # ── Signal Generation ──
    def _trend_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Trend Following v4.4 sinyal üretimi.
        Backtest'teki _check_entry mantığının birebir kopyası.
        """
        if len(df) < 65:
            return None

        # Supertrend hesapla
        p = self._TREND_PARAMS
        st_dir, st_val = _supertrend(df, p['st_period'], p['st_multiplier'])

        i = len(df) - 1
        if i < 2:
            return None

        close = float(df['close'].iloc[i])
        adx = float(df['adx'].iloc[i]) if 'adx' in df.columns else 0
        rsi_val = float(df['rsi'].iloc[i]) if 'rsi' in df.columns else 50
        vol_ratio = float(df['vol_ratio_20'].iloc[i]) if 'vol_ratio_20' in df.columns else 1.0
        macd_hist = float(df['macd_hist'].iloc[i]) if 'macd_hist' in df.columns else 0

        if any(np.isnan(v) for v in [adx, rsi_val, vol_ratio, macd_hist]):
            return None

        ema9 = float(df['ema9'].iloc[i]) if 'ema9' in df.columns else close
        ema21 = float(df['ema21'].iloc[i]) if 'ema21' in df.columns else close

        st_now = int(st_dir.iloc[i])
        st_prev = int(st_dir.iloc[i - 1])

        # ── Entry Detection ──
        st_flip_bull = (st_now == 1 and st_prev == -1)
        st_flip_bear = (st_now == -1 and st_prev == 1)
        ema_bull = ema9 > ema21
        ema_bear = ema9 < ema21

        # Recent EMA cross (3 bar)
        recent_cross_bull = False
        recent_cross_bear = False
        if 'ema_cross' in df.columns:
            for j in range(max(0, i - 3), i + 1):
                ec = df['ema_cross'].iloc[j]
                if not pd.isna(ec):
                    if float(ec) > 0:
                        recent_cross_bull = True
                    elif float(ec) < 0:
                        recent_cross_bear = True

        direction = None
        if st_flip_bull and ema_bull:
            direction = 'LONG'
        elif st_now == 1 and recent_cross_bull and ema_bull and macd_hist > 0:
            direction = 'LONG'
        if direction is None:
            if st_flip_bear and ema_bear:
                direction = 'SHORT'
            elif st_now == -1 and recent_cross_bear and ema_bear and macd_hist < 0:
                direction = 'SHORT'

        if direction is None:
            return None

        # ── Quality Filters ──
        if adx < p['min_adx']:
            return None
        if vol_ratio < p['min_vol_ratio']:
            return None
        if direction == 'LONG' and rsi_val > 70:
            return None
        if direction == 'SHORT' and rsi_val < 30:
            return None

        # ── Swing SL (capped at %2) ──
        lookback = p['swing_lookback']
        start = max(0, i - lookback)
        if direction == 'LONG':
            sl = float(df['low'].iloc[start:i + 1].min())
            sl = max(sl, close * (1 - p['max_sl_pct']))
            if sl >= close * 0.997:
                return None
        else:
            sl = float(df['high'].iloc[start:i + 1].max())
            sl = min(sl, close * (1 + p['max_sl_pct']))
            if sl <= close * 1.003:
                return None

        sl_dist = abs(close - sl)
        # Safety TP (çok uzak — trail ile çıkılacak)
        tp = close + sl_dist * 20 if direction == 'LONG' else close - sl_dist * 20

        atr_val = float(df['atr'].iloc[i]) if 'atr' in df.columns else sl_dist

        return {
            'direction': direction,
            'sl_price': sl,
            'tp_price': tp,
            'atr': atr_val,
            'sl_dist': sl_dist,
            'strategy': 'trend_v4.4',
            'adx': round(adx, 1),
            'rsi': round(rsi_val, 1),
            'vol_ratio': round(vol_ratio, 2),
            'st_direction': st_now,
            'st_flip': st_flip_bull or st_flip_bear,
            'ema_aligned': ema_bull if direction == 'LONG' else ema_bear,
            'signal_high': float(df['high'].iloc[i]),
            'signal_low': float(df['low'].iloc[i]),
        }

    # ── Exit Logic (called from _ticker_loop) ──
    def _check_trend_exit(self, t, current_price):
        """
        Trend v4.4 çıkış mantığı — ticker loop'tan çağrılır.
        1. SL hit
        2. Late trail (+6% → peak karın %35 kilitle)
        3. Max dollar loss
        4. Timeout (72h)
        """
        entry = t['entry_price']
        sl = t['sl_price']
        side = t['side']
        qty = t['qty']
        p = self._TREND_PARAMS

        # PnL hesapla
        if side == 'LONG':
            pnl_pct = (current_price - entry) / entry
            peak = t.get('_trend_peak', entry)
            if current_price > peak:
                peak = current_price
                t['_trend_peak'] = peak
            peak_pnl_pct = (peak - entry) / entry
        else:
            pnl_pct = (entry - current_price) / entry
            peak = t.get('_trend_peak', entry)
            if current_price < peak:
                peak = current_price
                t['_trend_peak'] = peak
            peak_pnl_pct = (entry - peak) / entry

        # ── Tier 1: Breakeven — +3% peak'ten sonra SL'yi giriş fiyatına taşı ──
        be_thresh = p.get('be_threshold', 0.03)
        if peak_pnl_pct >= be_thresh and not t.get('_trend_be_active'):
            t['_trend_be_active'] = True
            if side == 'LONG':
                be_price = entry * 1.001  # fee buffer
                if be_price > sl:
                    t['sl_price'] = be_price
                    sl = be_price
                    self.log(f"[LOCK] {t['symbol']} BE activated @ {be_price:.6f} (peak {peak_pnl_pct*100:.1f}%)")
            else:
                be_price = entry * 0.999
                if be_price < sl:
                    t['sl_price'] = be_price
                    sl = be_price
                    self.log(f"[LOCK] {t['symbol']} BE activated @ {be_price:.6f} (peak {peak_pnl_pct*100:.1f}%)")

        # ── Tier 2: Trail — +6% kardan sonra, peak karın %55'ini kilitle ──
        if peak_pnl_pct >= p['trail_start']:
            t['_trend_trail_active'] = True
            keep = peak_pnl_pct * p['trail_keep']
            if side == 'LONG':
                new_sl = entry * (1 + keep)
                if new_sl > sl:
                    t['sl_price'] = new_sl
                    sl = new_sl
            else:
                new_sl = entry * (1 - keep)
                if new_sl < sl:
                    t['sl_price'] = new_sl
                    sl = new_sl

        # ── SL Hit ──
        sl_hit = (side == 'LONG' and current_price <= sl) or \
                 (side == 'SHORT' and current_price >= sl)
        if sl_hit:
            reason = 'TRAIL_SL' if t.get('_trend_trail_active') else 'SL'
            self._close_all_locked([t], t['symbol'], current_price, reason)
            self._trend_record_outcome('SL', t['symbol'])
            return

        # ── Max Dollar Loss ──
        pnl_dollar = (current_price - entry) * qty if side == 'LONG' else (entry - current_price) * qty
        if pnl_dollar < -p['max_loss_cap']:
            self._close_all_locked([t], t['symbol'], current_price, 'MAXLOSS')
            self._trend_record_outcome('SL', t['symbol'])
            return

        # ── Timeout ──
        age_hours = (time.time() - t.get('entry_timestamp', time.time())) / 3600
        if age_hours >= p['timeout_hours']:
            self._close_all_locked([t], t['symbol'], current_price, 'TIMEOUT')
            outcome = 'TP' if pnl_pct > 0 else 'SL'
            self._trend_record_outcome(outcome, t['symbol'])
            return

        # ── Supertrend Flip Exit (kârda) ──
        # Not: Bu kontrolü scan loop'ta yapıyoruz çünkü df gerekiyor
        # _ticker_loop'ta sadece fiyat bazlı çıkışlar kontrol edilir

    # ── Supertrend Flip Check (scan loop'ta çağrılır) ──
    def _trend_check_st_flip_exit(self, t, df: pd.DataFrame):
        """
        Supertrend flip → kârda çık.
        Bu metod scan loop sırasında df mevcut olduğunda çağrılır.
        """
        if t.get('strategy', '') != 'trend_v4.4':
            return

        p = self._TREND_PARAMS
        st_dir, _ = _supertrend(df, p['st_period'], p['st_multiplier'])
        st_now = int(st_dir.iloc[-1])

        entry = t['entry_price']
        close = float(df['close'].iloc[-1])
        side = t['side']

        pnl_pct = ((close - entry) / entry) if side == 'LONG' else ((entry - close) / entry)

        # ST flip + kârda = çık
        if side == 'LONG' and st_now == -1 and pnl_pct > 0.003:
            with self.trades_lock:
                if t in self.open_trades:
                    self._close_all_locked([t], t['symbol'], close, 'ST_FLIP')
                    self._trend_record_outcome('TP', t['symbol'])

        if side == 'SHORT' and st_now == 1 and pnl_pct > 0.003:
            with self.trades_lock:
                if t in self.open_trades:
                    self._close_all_locked([t], t['symbol'], close, 'ST_FLIP')
                    self._trend_record_outcome('TP', t['symbol'])

    # ── Outcome Recording ──
    def _trend_record_outcome(self, outcome_type: str, symbol: str = ''):
        """Trade sonucunu kaydet, cooldown tetikle."""
        self._trend_init_state()
        now = time.time()

        if outcome_type == 'SL':
            self._trend_consecutive_sl += 1
            if self._trend_consecutive_sl >= 3:
                self._trend_cooldown_until = now + 6 * 3600  # 6 saat mola
                self.log(f"[COOL] Trend 3 ardışık SL → 6 saat mola")
                self._trend_consecutive_sl = 0
            if symbol:
                self._trend_symbol_cooldown[symbol] = now + 3 * 3600  # Sembol 3h cooldown
        elif outcome_type == 'TP':
            self._trend_consecutive_sl = 0

        self._trend_recent_outcomes.append(outcome_type)
        if len(self._trend_recent_outcomes) > 10:
            self._trend_recent_outcomes.pop(0)

    def _trend_check_cooldown(self) -> bool:
        """Cooldown aktif mi?"""
        self._trend_init_state()
        return time.time() >= self._trend_cooldown_until

    # ═══════════════════════════════════════════════════════════════
    #  SCAN LOOP — BB MR scan pattern'ini takip eder
    # ═══════════════════════════════════════════════════════════════

    def _trend_scan(self, fetcher: DataFetcher):
        """
        Trend Following v4.4 — Scan Loop.
        Tüm coinleri tara, sinyal bul, en iyisini aç.
        """
        self._trend_init_state()

        if not self._check_loss_limits():
            return

        if not self._trend_check_cooldown():
            remaining = max(0, self._trend_cooldown_until - time.time())
            if remaining > 0:
                self.log(f"[COOL] Trend cooldown: {int(remaining / 3600)}h {int((remaining % 3600) / 60)}m kaldı")
            return

        # Trade arası minimum bekleme
        p = self._TREND_PARAMS
        if time.time() - self._trend_last_trade_time < p['cooldown_hours'] * 3600:
            return

        candidates = []
        skipped = 0

        with self.trades_lock:
            active_symbols = set(t['symbol'] for t in self.open_trades)
            pending_symbols = set(p['symbol'] for p in self.pending_orders)
            active_count = len(self.open_trades) + len(self.pending_orders)
            can_open = active_count < self.max_open_trades_limit

        if not can_open:
            self.log("[PERF] Trend scan: limit dolu")
            return

        for i, symbol in enumerate(self.scanned_symbols[:]):
            if not self.is_running:
                break

            try:
                if i % 30 == 0:
                    self.log(f"  [TREND] Scanning {symbol} ({i}/{len(self.scanned_symbols)})")

                if symbol in active_symbols or symbol in pending_symbols:
                    continue

                # Symbol cooldown
                sym_cd = self._trend_symbol_cooldown.get(symbol, 0)
                if time.time() < sym_cd:
                    continue

                # Korelasyon kontrolü
                if not self._trend_check_correlation(symbol):
                    continue

                df_1h = fetcher.fetch_ohlcv(symbol, '1h', limit=100)
                if df_1h is None or df_1h.empty or len(df_1h) < 65:
                    continue

                df_1h = add_all_indicators(df_1h)

                sig = self._trend_signal(df_1h)
                if sig is None:
                    skipped += 1
                    continue

                sig['symbol'] = symbol
                sig['close'] = float(df_1h['close'].iloc[-1])
                candidates.append(sig)

                # Açık trend trade'ler için ST flip çıkış kontrolü
                with self.trades_lock:
                    for t in self.open_trades[:]:
                        if t.get('strategy') == 'trend_v4.4' and t['symbol'] == symbol:
                            self._trend_check_st_flip_exit(t, df_1h)

            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str:
                    time.sleep(1.0)
                else:
                    self.log(f"[ERR] Trend scan {symbol}: {e}")
                    time.sleep(0.3)

        if not candidates:
            self.log(f"[PERF] Trend scan: 0 signal, {skipped} skipped")
            return

        # En iyi sinyali seç: ADX * vol_ratio sıralaması
        candidates.sort(
            key=lambda x: x.get('adx', 0) * x.get('vol_ratio', 1),
            reverse=True,
        )

        top_n = min(len(candidates), 5)
        self.log(f"[RANK] Trend v4.4: {len(candidates)} sinyal - Top {top_n}:")
        for j, c in enumerate(candidates[:top_n]):
            marker = ">>>" if j == 0 else "   "
            self.log(
                f"  {marker} #{j + 1} {c['symbol']} {c['direction']} | "
                f"ADX:{c.get('adx', 0)} RSI:{c.get('rsi', 0)} "
                f"Vol:{c.get('vol_ratio', 0)} ST_flip:{c.get('st_flip', False)}"
            )

        best = candidates[0]
        symbol = best['symbol']
        direction = best['direction']
        cp = best['close']
        atr_val = best['atr']
        sl_dist = best['sl_dist']

        # Position sizing: risk-based
        max_loss = p['max_loss_cap']
        qty_risk = max_loss / sl_dist if sl_dist > 0 else 0
        max_notional = min(p['max_notional'], self.balance * 0.20)
        qty_notional = max_notional / cp
        qty = min(qty_risk, qty_notional)

        if qty <= 0:
            return

        logger_id = f"{symbol}_{direction}_{int(time.time())}"

        with self.trades_lock:
            tid = self._open_locked(
                symbol=symbol,
                side=direction,
                price=cp,
                multiplier=1.0,
                tp_price=best['tp_price'],
                sl_price=best['sl_price'],
                signal_result={
                    'strategy': 'trend_v4.4',
                    'regime': 'trending',
                    'entry_type': 'market',
                    'soft_score': 5,
                    'signal': direction,
                    'adx': best.get('adx', 0),
                    'rsi': best.get('rsi', 0),
                    'vol_ratio': best.get('vol_ratio', 0),
                    'st_flip': best.get('st_flip', False),
                },
                absolute_qty=qty,
                atr=atr_val,
                logger_id=logger_id,
            )
            if tid:
                for t in self.open_trades:
                    if t.get('id') == tid:
                        t['_trend_peak'] = cp
                        t['_trend_trail_active'] = False
                        break
                self._trend_last_trade_time = time.time()

        self.log(
            f"[PERF] Trend v4.4: {len(candidates)} signal(s), {skipped} skipped, "
            f"BEST: {symbol} ADX={best.get('adx', 0)} Vol={best.get('vol_ratio', 0)}"
        )
