import time
import threading
import pandas as pd
from typing import Dict, List, Optional
import json
import os
import uuid
from datetime import datetime

from backtest.data_fetcher import DataFetcher
from ai.engine_v3 import PositionManagerV3
from ai.adaptive_live_adapter import (
    generate_signal, should_open_position, get_tp_sl_prices
)

SCAN_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "POL/USDT"
]

import ccxt

def get_all_usdt_pairs():
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    excluded = {"USDC/USDT", "USD1/USDT", "TUSD/USDT", "FDUSD/USDT"}
    usdt_pairs = [
        symbol for symbol in markets
        if symbol.endswith('/USDT')
        and symbol not in excluded
        and markets[symbol].get('active')
        and markets[symbol].get('spot')
    ]
    return usdt_pairs


class LivePaperTrader:
    """
    Background'da calisan Otonom Paper Trading (Sanal Bakiye) servisi.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LivePaperTrader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, initial_balance=10000.0, leverage=10, risk_pct=1.5):
        if self._initialized:
            return
        self._initialized = True

        state_dir = os.getenv("LIVE_TRADER_STATE_DIR", os.path.dirname(__file__))
        self.state_file = os.path.join(state_dir, "live_trader_state.json")

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.risk_pct = risk_pct / 100.0
        self.max_open_trades_limit = 5

        self.is_running = False
        self._thread = None
        self._ticker_thread = None

        self.open_trades = []
        self.pending_orders = []
        self.closed_trades = []
        self.logs = []
        self.current_prices = {}
        self.trades_lock = threading.RLock()
        self._last_save_time = 0
        self._save_interval = 10

        self.last_scan_time = None
        self.trade_counter = 0
        self.max_dca_levels = 5
        self.dca_spacing_pct = 0.01

        self.scanned_symbols = SCAN_SYMBOLS
        self.timeframe = "1h"
        self.secondary_tf = "15m"

        self.symbol_consecutive_losses = {}
        self.consecutive_losses = 0

        self.position_managers = {}

        self.load_state()

    # ──────────────────────────── State Persistence ────────────────────────────

    def load_state(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)

            self.balance = state.get('balance', self.initial_balance)
            self.open_trades = state.get('open_trades', [])
            self.pending_orders = state.get('pending_orders', [])
            self.closed_trades = state.get('closed_trades', [])
            self.trade_counter = state.get('trade_counter', 0)
            self.max_open_trades_limit = state.get('max_open_trades_limit', 5)
            self.consecutive_losses = state.get('consecutive_losses', 0)
            self.symbol_consecutive_losses = state.get('symbol_consecutive_losses', {})

            # Normalize legacy string logs → dict logs
            loaded_logs = state.get('logs', [])
            self.logs = []
            for entry in loaded_logs:
                if isinstance(entry, str):
                    self.logs.append({"id": str(uuid.uuid4())[:8], "text": entry})
                else:
                    self.logs.append(entry)

            self.log(
                f"[SAVE] State loaded. Balance: ${self.balance:.2f}, "
                f"Open: {len(self.open_trades)}, Pending: {len(self.pending_orders)}"
            )
        except Exception as e:
            self.log(f"[WARN] Error loading state: {e}. Starting fresh.")

    def _sanitize_for_json(self, obj):
        """Recursively convert numpy/pandas types to native Python."""
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(v) for v in obj]
        elif hasattr(obj, 'item') and not isinstance(obj, (list, tuple, dict, str)): # numpy scalar
            return obj.item()
        elif hasattr(obj, 'isoformat'):  # datetime/Timestamp
            return obj.isoformat()
        elif isinstance(obj, float) and (obj != obj):  # NaN check
            return None
        return obj

    def save_state(self):
        try:
            with self.trades_lock:
                state = {
                    'balance': self.balance,
                    'open_trades': self.open_trades,
                    'pending_orders': self.pending_orders,
                    'closed_trades': self.closed_trades,
                    'logs': self.logs,
                    'trade_counter': self.trade_counter,
                    'max_open_trades_limit': self.max_open_trades_limit,
                    'consecutive_losses': self.consecutive_losses,
                    'symbol_consecutive_losses': self.symbol_consecutive_losses,
                }
            
            clean_state = self._sanitize_for_json(state)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(clean_state, f, indent=4)
        except Exception as e:
            self.log(f"[WARN] Error saving state: {e}")

    def _force_save(self):
        """Immediate save for important events (trade open/close)."""
        self.save_state()
        self._last_save_time = time.time()

    def _throttled_save(self):
        """Save at most once per `_save_interval` seconds."""
        now = time.time()
        if now - self._last_save_time > self._save_interval:
            self.save_state()
            self._last_save_time = now

    # ──────────────────────────── Lifecycle ────────────────────────────────────

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.log("[BOT] Quant AI Live Paper Trader STARTED.")
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ticker_thread = threading.Thread(target=self._ticker_loop, daemon=True)
        self._ticker_thread.start()

    def stop(self):
        self.is_running = False
        self.log("[STOP] Quant AI Live Paper Trader STOPPED.")
        self._force_save()

    def log(self, msg: str):
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fmsg = f"[{t}] {msg}"
        print(fmsg)
        self.logs.insert(0, {"id": str(uuid.uuid4())[:8], "text": fmsg})
        if len(self.logs) > 100:
            self.logs.pop()
        self._throttled_save()

    # ──────────────────────────── Status / Serialisation ──────────────────────

    def get_status(self):
        with self.trades_lock:
            return {
                "status": "Running" if self.is_running else "Stopped",
                "balance": round(self.balance, 2),
                "open_trades_count": len(self.open_trades),
                "pending_orders_count": len(self.pending_orders),
                "open_trades": [self._trade_dict(t) for t in self.open_trades],
                "closed_trades_count": len(self.closed_trades),
                "closed_trades": [
                    self._closed_trade_dict(t) for t in self.closed_trades[-20:]
                ],
                "recent_logs": self.logs[:10],
                "scanned_markets_count": len(self.scanned_symbols),
                "max_open_trades_limit": self.max_open_trades_limit,
            }

    def _compute_pnl(self, t, current_price):
        """Central PnL calculation (avoids copy-paste drift)."""
        n_in = t['qty'] * t['entry_price']
        n_out = t['qty'] * current_price
        fee = (n_in + n_out) * 0.0002
        if t['side'] == 'LONG':
            pnl = (n_out - n_in) - fee
        else:
            pnl = (n_in - n_out) - fee
        pnl_pct = (pnl / t['margin']) * 100 if t['margin'] > 0 else 0
        return pnl, pnl_pct

    def _trade_dict(self, t):
        sym = t['symbol']
        cp = self.current_prices.get(sym, t['entry_price'])
        pnl, pnl_pct = self._compute_pnl(t, cp)
        return {
            "id": t['id'], "pair": sym, "side": t['side'],
            "entry": round(t['entry_price'], 4), "size": t['qty'],
            "margin": t['margin'], "entry_time": t['entry_time'],
            "current_price": round(cp, 4),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "max_pnl_pct": round(t.get('max_pnl_pct', 0), 2),
            "regime": t.get('regime', 'unknown'),
            "strategy": t.get('strategy', 'unknown'),
            "sl_price": t.get('sl_price', 0),
            "tp_price": t.get('tp_price', 0),
            "soft_score": t.get('soft_score', 0),
            "entry_type": t.get('entry_type', 'unknown'),
            "pnl_history": t.get('pnl_history', []),
        }

    def _closed_trade_dict(self, t):
        pnl_pct = (t['pnl'] / t['margin']) * 100 if t['margin'] > 0 else 0
        return {
            "id": t['id'], "pair": t['symbol'], "side": t['side'],
            "entry": round(t['entry_price'], 4),
            "exit": round(t['exit_price'], 4),
            "size": t['qty'], "margin": t['margin'],
            "entry_time": t['entry_time'],
            "exit_time": t.get('exit_time', '--'),
            "pnl": round(t['pnl'], 2),
            "pnl_pct": round(pnl_pct, 2),
            "max_pnl_pct": round(t.get('max_pnl_pct', 0), 2),
            "reason": t.get('reason', 'UNKNOWN'),
            "regime": t.get('regime', 'unknown'),
            "strategy": t.get('strategy', 'unknown'),
            "soft_score": t.get('soft_score', 0),
            "entry_type": t.get('entry_type', 'unknown'),
            "sl_price": t.get('sl_price', 0),
            "tp_price": t.get('tp_price', 0),
            "pnl_history": t.get('pnl_history', []),
        }

    # ──────────────────────────── Ticker Loop ─────────────────────────────────

    def _ticker_loop(self):
        exchange = ccxt.binance({'enableRateLimit': True})

        while self.is_running:
            try:
                with self.trades_lock:
                    open_syms = [t['symbol'] for t in self.open_trades]
                    pending_syms = [p['symbol'] for p in self.pending_orders]
                    symbols = list(set(open_syms + pending_syms))

                if not symbols:
                    time.sleep(3)
                    continue

                tickers = exchange.fetch_tickers(symbols)

                for sym, data in tickers.items():
                    if not (data and data.get('last')):
                        continue
                    new_price = data['last']
                    self.current_prices[sym] = new_price  # atomic dict write

                    with self.trades_lock:
                        self._process_pending_for_symbol(sym, new_price)
                        self._process_open_for_symbol(sym, new_price)

            except Exception:
                pass  # Ticker failures are non-critical

            time.sleep(3)

    def _process_pending_for_symbol(self, sym, price):
        """Check pending limit orders — MUST be called under trades_lock."""
        for p in self.pending_orders[:]:
            if p['symbol'] != sym:
                continue

            # Expiration (2 hours)
            age = time.time() - p.get('created_at', 0)
            if age > 7200:
                self.pending_orders.remove(p)
                self.log(
                    f"⌛ PENDING EXPIRED: {sym} {p['side']} "
                    f"({int(age / 60)}m old)"
                )
                self._force_save()
                continue

            side = p['side']
            entry = p['entry_price']
            triggered = (
                (side == 'LONG' and price <= entry)
                or (side == 'SHORT' and price >= entry)
            )
            if triggered:
                self.log(
                    f"[ZAP] PENDING FILLED: {sym} {side} @ {price:.6f} "
                    f"(Target: {entry:.6f})"
                )
                self.pending_orders.remove(p)
                self._open_locked(
                    p['symbol'], p['side'], price, 1.0, 0,
                    p['sl_price'], p['signal_result'],
                    p['absolute_qty'], p['atr'], p['logger_id'],
                )

    def _process_open_for_symbol(self, sym, price):
        """Update PnL snapshots & check position manager — under trades_lock."""
        for t in self.open_trades[:]:
            if t['symbol'] != sym:
                continue
            self._record_pnl_snapshot(t, price)
            self._check_v3_manager(t, price)

    def _record_pnl_snapshot(self, t, current_price):
        if 'pnl_history' not in t:
            t['pnl_history'] = []

        pnl, pnl_pct = self._compute_pnl(t, current_price)

        # Track max PnL
        if pnl_pct > t.get('max_pnl_pct', -999):
            t['max_pnl_pct'] = pnl_pct
            t['max_pnl_val'] = pnl

        t['pnl_history'].append({
            "t": datetime.now().strftime("%H:%M:%S"),
            "p": round(current_price, 6),
            "pnl": round(pnl, 2),
            "pct": round(pnl_pct, 2),
        })
        if len(t['pnl_history']) > 200:
            t['pnl_history'].pop(0)

    # ──────────────────────────── Position Manager ────────────────────────────

    def _get_fallback_atr(self, symbol, entry_price):
        """Estimate ATR without network I/O if possible. Real ATR update happens in scan loop."""
        # v3.6: Avoid network I/O under lock. Use 0.5% estimate as safety.
        estimate = entry_price * 0.005
        return estimate

    def _check_v3_manager(self, t, current_price):
        """PositionManagerV3 exit logic. Called under trades_lock."""
        tid = t['id']

        if tid not in self.position_managers:
            atr = t.get('atr') or self._get_fallback_atr(t['symbol'], t['entry_price'])
            t['atr'] = atr
            pm = PositionManagerV3(t, atr)
            pm.stage = t.get('pm_stage', 0)
            pm.highest_seen = t.get('pm_highest', t['entry_price'])
            pm.trailing_stop = t.get('pm_trail', None)
            pm.stop = t.get('sl_price', t['entry_price'])
            self.position_managers[tid] = pm

        pm = self.position_managers[tid]
        res = pm.update(current_price)

        if res['action'] == 'EXIT':
            self._close_all_locked([t], t['symbol'], current_price, res['reason'])
            return # v3.6: Don't modify 't' after it's moved to closed_trades

        if res['action'] == 'PARTIAL':
            t['sl_price'] = res['stop']
            self.log(
                f"[MONEY] PARTIAL {t['symbol']}: {res['reason']} "
                f"| New SL: {res['stop']:.6f}"
            )
        elif res['action'] == 'UPDATE_STOP':
            t['sl_price'] = res['stop']

        # Persist PM state (only if trade is still open)
        t['pm_stage'] = pm.stage
        t['pm_highest'] = pm.highest_seen
        t['pm_trail'] = pm.trailing_stop
        t['sl_price'] = pm.trailing_stop if pm.trailing_stop else pm.stop

    # ──────────────────────────── Scan Loop ────────────────────────────────────

    def _run_loop(self):
        fetcher = DataFetcher('binance')

        while self.is_running:
            try:
                # Dynamic symbol refresh
                try:
                    all_pairs = get_all_usdt_pairs()
                    with self.trades_lock:
                        self.scanned_symbols = all_pairs
                    self.log(
                        f"[SYNC] Dynamic symbols: {len(self.scanned_symbols)} USDT pairs"
                    )
                except Exception as e:
                    msg = str(e).lower()
                    if "451" in msg or "restricted" in msg:
                        self.log(
                            "[WARN] Binance region restriction (HTTP 451). "
                            "Using fallback symbols."
                        )
                    else:
                        self.log(f"[WARN] Symbol fetch failed: {e}. Using fallback.")

                self.log(f"[SEARCH] Scanning {len(self.scanned_symbols)} markets...")
                self._scan_and_trade(fetcher)

            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate" in err:
                    self.log(f"[WARN] Rate limit: {e}. Backing off...")
                    time.sleep(5)
                else:
                    self.log(f"[FAIL] Scan loop error: {e}")
                time.sleep(0.5)

            # Wait between scan cycles
            for _ in range(60):
                if not self.is_running:
                    break
                time.sleep(1)

    def _scan_and_trade(self, fetcher: DataFetcher):
        # BTC bias
        try:
            btc_4h = fetcher.fetch_ohlcv("BTC/USDT", "4h", limit=100)
            if btc_4h.empty:
                self.log("[WARN] BTC 4h empty — skipping scan.")
                return
        except Exception as e:
            self.log(f"[WARN] BTC 4h fetch error: {e}")
            return

        for i, symbol in enumerate(self.scanned_symbols[:]):
            if not self.is_running:
                break

            try:
                df_1h = fetcher.fetch_ohlcv(symbol, self.timeframe, limit=100)
                if df_1h.empty or len(df_1h) < 50:
                    continue

                df_15m = fetcher.fetch_ohlcv(symbol, self.secondary_tf, limit=100)
                if df_15m.empty or len(df_15m) < 50:
                    continue

                df_4h = fetcher.fetch_ohlcv(symbol, "4h", limit=100)
                if df_4h.empty or len(df_4h) < 50:
                    continue

                if i % 30 == 0:
                    self.log(f"[*][*] Scanning {symbol} ({i}/{len(self.scanned_symbols)})")

                with self.trades_lock:
                    has_active = any(
                        t['symbol'] == symbol for t in self.open_trades
                    )
                    has_pending = any(
                        p['symbol'] == symbol for p in self.pending_orders
                    )
                    active_count = len(self.open_trades) + len(self.pending_orders)
                    can_open = active_count < self.max_open_trades_limit

                if has_active or has_pending or not can_open:
                    continue

                # --- HYBRID AI ENGINE DECISION ---
                decision = generate_signal(
                    df=df_1h,
                    df_secondary=df_15m,
                    df_4h=df_4h,
                    symbol=symbol,
                    timeframe=self.timeframe,
                    secondary_tf=self.secondary_tf
                )

                if should_open_position(decision):
                    if self.consecutive_losses >= 10:
                        self.log("[STOP] ENGINE HALT: Max consecutive losses limit reached")
                        self.stop()
                        break

                    cp = float(df_1h['close'].iloc[-1])
                    
                    tp_price = decision.get('tp_price', 0.0)
                    sl_price = decision.get('sl_price', 0.0)
                    
                    if tp_price == 0.0 or sl_price == 0.0:
                        # Fallback TP/SL
                        atr_proxy = df_1h['high'].iloc[-14:].max() - df_1h['low'].iloc[-14:].min()
                        tp_price, sl_price = get_tp_sl_prices(decision, cp, atr_proxy)

                    risk_abs = abs(cp - sl_price)
                    if risk_abs <= 0: risk_abs = cp * 0.01
                    
                    # Size calculation
                    loss_scaling = {0: 1.0, 1: 1.0, 2: 0.8, 3: 0.6}.get(self.consecutive_losses, 0.5)
                    position_conf = decision.get('position_size', 0.5)
                    risk_amount = self.balance * self.risk_pct * loss_scaling * position_conf
                    raw_qty = risk_amount / risk_abs
                    max_qty = (self.balance * self.leverage) / cp
                    qty = min(raw_qty, max_qty)

                    logger_id = f"{symbol}_{decision['signal']}_{int(time.time())}"
                    atr_val = risk_abs / decision.get('sl_mult', 1.0)

                    with self.trades_lock:
                        self._open_locked(
                            symbol=symbol,
                            side=decision['signal'],
                            price=cp,
                            multiplier=1.0,
                            tp_price=tp_price,
                            sl_price=sl_price,
                            signal_result=decision,
                            absolute_qty=qty,
                            atr=atr_val,
                            logger_id=logger_id
                        )

                time.sleep(0.1)

            except Exception as e:
                # [OK] FIX: Single except — log + rate-limit backoff
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str:
                    time.sleep(1.0)
                else:
                    self.log(f"[FAIL] Error scanning {symbol}: {e}")
                    time.sleep(0.5)

    # ──────────────────────────── Trade Execution ─────────────────────────────

    def _open(self, symbol, side, price, multiplier,
              tp_price=0.0, sl_price=0.0, signal_result=None,
              absolute_qty=None, atr=0.001, logger_id=None):
        with self.trades_lock:
            return self._open_locked(
                symbol, side, price, multiplier,
                tp_price, sl_price, signal_result,
                absolute_qty, atr, logger_id,
            )

    def _open_locked(self, symbol, side, price, multiplier,
                     tp_price=0.0, sl_price=0.0, signal_result=None,
                     absolute_qty=None, atr=0.001, logger_id=None):
        """Must be called under trades_lock."""
        if absolute_qty:
            qty = absolute_qty
        else:
            qty = (
                self.initial_balance * self.risk_pct * self.leverage * multiplier
            ) / price

        margin = (qty * price) / self.leverage

        if qty <= 0 or self.balance < margin:
            return None

        self.balance -= margin
        self.trade_counter += 1
        tid = f"LVT-{self.trade_counter}"

        sig = signal_result or {}
        t = {
            "id": tid,
            "symbol": symbol,
            "side": side,
            "entry_price": price,
            "qty": qty,
            "margin": margin,
            "entry_time": datetime.now().strftime("%H:%M:%S"),
            "tp_price": tp_price,
            "sl_price": sl_price,
            "atr": atr,
            "strategy": sig.get('type', 'ICT_Hybrid'),
            "regime": sig.get('regime', 'unknown'),
            "entry_type": sig.get('type', 'none'),
            "soft_score": sig.get('score', 0),
            "signal_result": signal_result,
            "logger_id": logger_id or (
                f"{symbol}_{side}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            ),
            "max_pnl_pct": 0,
            "pnl_history": [],
        }

        self.open_trades.append(t)
        self.position_managers[tid] = PositionManagerV3(t, atr)

        self.log(
            f"[GRN] OPEN {side} {symbol} @ {price:.6f} | "
            f"Score: {t['soft_score']}/5 | Type: {t['entry_type']} | "
            f"Qty: {qty:.4f}"
        )
        self._force_save()
        return tid

    def _close_all(self, trades, symbol, exit_price, reason):
        """Thread-safe wrapper."""
        with self.trades_lock:
            self._close_all_locked(trades, symbol, exit_price, reason)

    def _close_all_locked(self, trades, symbol, exit_price, reason):
        """Must be called under trades_lock."""
        for t in trades:
            if t not in self.open_trades:
                continue

            pnl, pnl_pct = self._compute_pnl(t, exit_price)
            self.balance += t['margin'] + pnl

            t['exit_price'] = exit_price
            t['exit_time'] = datetime.now().strftime("%H:%M:%S")
            t['pnl'] = pnl
            t['reason'] = reason

            # Logger exit
            try:
                self.engine_v3.logger.log_exit(
                    t.get('logger_id', ''), exit_price, reason
                )
            except Exception:
                pass

            self.closed_trades.append(t)
            self.open_trades.remove(t)

            # Manager cleanup
            self.position_managers.pop(t['id'], None)

            # is_win tracking directly replacing the EngineV3 logic
            is_win = pnl > 0
            if is_win:
                self.consecutive_losses = max(0, self.consecutive_losses - 1)
            else:
                self.consecutive_losses += 1

            # Symbol consecutive loss tracking
            if is_win:
                self.symbol_consecutive_losses[symbol] = 0
            else:
                self.symbol_consecutive_losses[symbol] = (
                    self.symbol_consecutive_losses.get(symbol, 0) + 1
                )

            max_p = t.get('max_pnl_pct', 0)
            pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
            icon = "[OK]" if pnl > 0 else "[FAIL]"
            self.log(
                f"{icon} CLOSE {t['side']} {symbol} @ {exit_price:.6f} | "
                f"PnL: {pnl_str} ({pnl_pct:.2f}%) | "
                f"Max: {max_p:.2f}% | Reason: {reason}"
            )

        if trades:
            self._force_save()

    # ──────────────────────────── Manual Actions ──────────────────────────────

    def close_trade(self, trade_id: str):
        with self.trades_lock:
            trade = next(
                (t for t in self.open_trades if t['id'] == trade_id), None
            )
            if not trade:
                return {"success": False, "message": "Trade not found."}

            sym = trade['symbol']
            cp = self.current_prices.get(sym, trade['entry_price'])

            # Live price fallback (Network I/O is risky here but 1m is fast)
            if cp == trade['entry_price']:
                try:
                    fetcher = DataFetcher('binance')
                    df = fetcher.fetch_ohlcv(sym, '1m', limit=2)
                    if not df.empty:
                        cp = df.iloc[-1]['close']
                except Exception:
                    pass

            self._close_all_locked([trade], sym, cp, "MANUAL_CLOSE")
            
        return {
            "success": True,
            "message": f"Trade {trade_id} closed at {cp:.4f}.",
        }

    def update_settings(self, max_open_trades: int):
        if max_open_trades <= 0:
            return {"success": False, "message": "Invalid value"}
        self.max_open_trades_limit = max_open_trades
        self._force_save()
        self.log(f"[GEAR] Max open trades → {max_open_trades}")
        return {"success": True, "message": "Settings updated"}
