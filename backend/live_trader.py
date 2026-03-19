import time
import threading
import pandas as pd
from typing import Dict, List, Optional
import json
import os
import uuid
from datetime import datetime

import numpy as np
from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators
from ai.engine_v3 import PositionManagerV3
from ai.adaptive_live_adapter import (
    generate_signal, should_open_position, get_tp_sl_prices, has_1h_signal
)
from ai.entry_gate import five_layer_gate, GateResult
from strategies.bb_mr_strategy import BBMRStrategyMixin
from strategies.ict_smc_strategy import ICTSMCStrategyMixin
from strategies.vwap_scalping_strategy import VWAPScalpingMixin
from strategies.trend_following_strategy import TrendFollowingMixin

SCAN_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "POL/USDT"
]

CORE_V1_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT",
]
CORE_V1_SYMBOL_LIMIT = 40
CORE_V2_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
]
CORE_V2_SYMBOL_LIMIT = 50
CORE_V2_AGGR_SYMBOL_LIMIT = 60

import ccxt
import requests as _requests

TOP_SYMBOLS_LIMIT = 150  # En likit 150 coin (3 grup × 50)

def get_all_usdt_pairs():
    """
    Top 150 USDT pair by 24h volume.
    3 grup halinde 50'şer coin taranır (toplam 150).
    """
    try:
        # 1. 24h ticker ile volume bilgisi çek
        resp = _requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=20)
        resp.raise_for_status()
        tickers = resp.json()

        excluded = {"USDCUSDT", "USD1USDT", "TUSDUSDT", "FDUSDUSDT"}
        usdt_tickers = []
        for t in tickers:
            sym = t.get('symbol', '')
            if (sym.endswith('USDT')
                    and sym not in excluded
                    and float(t.get('quoteVolume', 0)) > 0):
                usdt_tickers.append({
                    'symbol': sym.replace('USDT', '') + '/USDT',
                    'volume': float(t.get('quoteVolume', 0)),
                })

        # 2. Volume'a göre sırala, top N al
        usdt_tickers.sort(key=lambda x: x['volume'], reverse=True)
        top_pairs = [t['symbol'] for t in usdt_tickers[:TOP_SYMBOLS_LIMIT]]

        if len(top_pairs) < 10:
            return SCAN_SYMBOLS
        return top_pairs
    except Exception:
        return SCAN_SYMBOLS


class LivePaperTrader(TrendFollowingMixin, VWAPScalpingMixin, ICTSMCStrategyMixin):
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

        # Shared exchange instance (rate limit shared across all paths)
        self._exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot', 'fetchMarkets': ['spot']}
        })

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

        # ── Strategy Enable/Disable ──
        self._strategy_enabled = {
            'bb_mr': True,
            'ict_smc': True,
            'trend_v4': True,
        }
        self._active_profile = 'custom'

        # ── v5.1: Backtest-aligned BB MR cooldown ──
        self._bb_consecutive_sl = 0
        self._bb_cooldown_until = 0        # timestamp
        self._bb_recent_outcomes = []      # Son 10 trade sonucu
        self._bb_last_trade_time = 0       # Son trade zamanı (3h cooldown)
        self._bb_symbol_cooldown = {}     # {symbol: timestamp} — SL olan sembol cooldown

        # ── ICT/SMC Strategy params ──
        self._ICT_PARAMS = {
            'min_confluence': 3,      # Min POI confluence skoru (1-4)
            'min_rr': 1.45,           # Min Risk:Reward oranı
            'max_sl_pct': 2.5,        # Max SL yüzdesi
            'require_sweep': True,    # Likidite sweep zorunlu mu
            'require_displacement': False,  # Displacement zorunlu mu
            'killzone_only': False,   # Sadece London/NY killzone
            'max_notional': 300.0,    # ICT trade başına max notional
            'max_loss_cap': 5.8,      # ICT trade başına max kayıp
            'sr_proximity_limit': 0.05,  # HTF support/resistance yakınlık eşiği
            'max_poi_distance_atr': 5.0,
            'max_zone_age_bars': 72,
            'entry_pullback_min_pct': 0.015,
            'entry_range_lookback': 20,
            'entry_range_top_ceil': 0.85,
            'entry_range_bot_floor': 0.15,
            'entry_max_ema21_ext': 0.045,
            'trend_recent_bars_4h': 48,
            'trend_min_labels': 8,
            'trend_max_labels': 20,
            'swing_params': {
                '15m': {'left': 3, 'right': 2},
                '1h': {'left': 5, 'right': 3},
                '4h': {'left': 5, 'right': 5},
                '1d': {'left': 7, 'right': 5},
            },
            'rsi_overbought': 72.0,
            'rsi_oversold': 28.0,
            'rsi_extreme_overbought': 82.0,
            'rsi_extreme_oversold': 18.0,
        }
        self._ict_consecutive_sl = 0
        self._ict_cooldown_until = 0
        self._ict_last_trade_time = 0
        self._ict_symbol_cooldown = {}
        self._ict_recent_outcomes = []

        self.position_managers = {}

        # ── BB MR user-configurable params ──
        self._user_max_notional = 150.0   # BB MR trade başına max notional ($)
        self._user_max_loss_cap = 5.0     # BB MR trade başına max kayıp ($)

        # ── v2.1: Korelasyon Filtresi ──
        self._btc_returns = None       # BTC 1h return serisi (cache)
        self._btc_returns_ts = 0       # Son güncelleme zamanı

        # ── v2.1: Günlük/Haftalık Loss Limiti ──
        self._daily_start_balance = None
        self._daily_date = None
        self._weekly_start_balance = None
        self._weekly_date = None       # ISO week number
        self._cooldown_until = 0       # 3 ardışık kayıp sonrası mola timestamp

        # ── v2.1: Adaptive Threshold ──
        self._adaptive_threshold = 0.65  # Başlangıç

        self.load_state()

    # ──────────────────────────── State Persistence ────────────────────────────

    def load_state(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
        except Exception as e:
            self.log(f"[WARN] State corrupt: {e}. Trying backup...")
            backup_file = self.state_file + ".bak"
            if os.path.exists(backup_file):
                try:
                    with open(backup_file, 'r', encoding='utf-8') as f:
                        state = json.load(f)
                    self.log(f"[OK] Backup state restored.")
                except Exception as e2:
                    self.log(f"[WARN] Backup also corrupt: {e2}. Starting fresh.")
                    return
            else:
                self.log(f"[WARN] No backup found. Starting fresh.")
                return

        try:
            self.balance = state.get('balance', self.initial_balance)
            self.open_trades = state.get('open_trades', [])
            self.pending_orders = state.get('pending_orders', [])
            self.closed_trades = state.get('closed_trades', [])
            self.trade_counter = state.get('trade_counter', 0)
            self.max_open_trades_limit = min(state.get('max_open_trades_limit', 8), 8)
            self.consecutive_losses = state.get('consecutive_losses', 0)
            self.symbol_consecutive_losses = state.get('symbol_consecutive_losses', {})

            # Strategy settings restore
            saved_enabled = state.get('strategy_enabled', {})
            if saved_enabled:
                self._strategy_enabled.update(saved_enabled)
            self._active_profile = state.get('active_profile', 'custom')
            saved_symbols = state.get('scanned_symbols', [])
            if isinstance(saved_symbols, list) and len(saved_symbols) >= 5:
                self.scanned_symbols = saved_symbols
            saved_ict = state.get('ict_params', {})
            if saved_ict:
                self._ICT_PARAMS.update(saved_ict)
            if 'user_max_notional' in state:
                self._user_max_notional = state['user_max_notional']
            if 'user_max_loss_cap' in state:
                self._user_max_loss_cap = state['user_max_loss_cap']

            # BB MR cooldown state restore
            self._bb_consecutive_sl = state.get('bb_consecutive_sl', 0)
            self._bb_cooldown_until = state.get('bb_cooldown_until', 0)
            self._bb_recent_outcomes = state.get('bb_recent_outcomes', [])
            self._bb_last_trade_time = state.get('bb_last_trade_time', 0)
            self._bb_symbol_cooldown = state.get('bb_symbol_cooldown', {})

            # ICT cooldown state restore
            self._ict_consecutive_sl = state.get('ict_consecutive_sl', 0)
            self._ict_cooldown_until = state.get('ict_cooldown_until', 0)
            self._ict_last_trade_time = state.get('ict_last_trade_time', 0)
            self._ict_symbol_cooldown = state.get('ict_symbol_cooldown', {})
            self._ict_recent_outcomes = state.get('ict_recent_outcomes', [])

            # Risk management state restore
            self._daily_start_balance = state.get('daily_start_balance', None)
            daily_date_str = state.get('daily_date', None)
            if daily_date_str:
                try:
                    from datetime import date as _date
                    self._daily_date = _date.fromisoformat(daily_date_str)
                except Exception:
                    self._daily_date = None
            self._weekly_start_balance = state.get('weekly_start_balance', None)
            self._weekly_date = state.get('weekly_date', None)
            self._cooldown_until = state.get('cooldown_until', 0)
            self._adaptive_threshold = state.get('adaptive_threshold', 0.65)

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
            self.log(f"[WARN] Error parsing state: {e}. Starting fresh.")

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
                    'active_profile': self._active_profile,
                    'scanned_symbols': self.scanned_symbols,
                    # Strategy settings
                    'strategy_enabled': self._strategy_enabled,
                    'ict_params': self._ICT_PARAMS,
                    'user_max_notional': getattr(self, '_user_max_notional', 150.0),
                    'user_max_loss_cap': getattr(self, '_user_max_loss_cap', 5.0),
                    # BB MR cooldown state
                    'bb_consecutive_sl': self._bb_consecutive_sl,
                    'bb_cooldown_until': self._bb_cooldown_until,
                    'bb_recent_outcomes': self._bb_recent_outcomes,
                    'bb_last_trade_time': self._bb_last_trade_time,
                    'bb_symbol_cooldown': self._bb_symbol_cooldown,
                    # ICT cooldown state
                    'ict_consecutive_sl': self._ict_consecutive_sl,
                    'ict_cooldown_until': self._ict_cooldown_until,
                    'ict_last_trade_time': self._ict_last_trade_time,
                    'ict_symbol_cooldown': self._ict_symbol_cooldown,
                    'ict_recent_outcomes': self._ict_recent_outcomes,
                    # Risk management state
                    'daily_start_balance': self._daily_start_balance,
                    'daily_date': str(self._daily_date) if self._daily_date else None,
                    'weekly_start_balance': self._weekly_start_balance,
                    'weekly_date': self._weekly_date,
                    'cooldown_until': self._cooldown_until,
                    'adaptive_threshold': self._adaptive_threshold,
                }
            
            clean_state = self._sanitize_for_json(state)
            
            # Atomic write: temp file → rename (prevents corruption on crash)
            tmp_file = self.state_file + ".tmp"
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(clean_state, f, indent=4)
            
            # Backup current state before overwrite
            if os.path.exists(self.state_file):
                backup_file = self.state_file + ".bak"
                try:
                    import shutil
                    shutil.copy2(self.state_file, backup_file)
                except Exception:
                    pass
            
            # Atomic rename
            os.replace(tmp_file, self.state_file)
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
        # Reset consecutive losses on fresh start (old session losses shouldn't halt new session)
        self.consecutive_losses = 0
        self.symbol_consecutive_losses = {}

        # Immediately fetch prices for open trades (no 30s wait for ticker loop)
        self._prefetch_prices()

        self.log("[BOT] Quant AI Live Paper Trader STARTED.")
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ticker_thread = threading.Thread(target=self._ticker_loop, daemon=True)
        self._ticker_thread.start()

    def _prefetch_prices(self):
        """Fetch current prices for all open trades immediately on startup."""
        try:
            with self.trades_lock:
                syms = list(set(t['symbol'] for t in self.open_trades))
            if not syms:
                return
            tickers = self._exchange.fetch_tickers(syms)
            for sym, data in tickers.items():
                if data and data.get('last'):
                    self.current_prices[sym] = data['last']
            self.log(f"[OK] Prefetch prices: {len(self.current_prices)} symbols updated")
        except Exception as e:
            self.log(f"[WARN] Prefetch prices failed: {e}")

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

    # ──────────────────────────── Status / Serialisation ──────────────────────

    def get_status(self):
        with self.trades_lock:
            return {
                "status": "Running" if self.is_running else "Stopped",
                "is_running": self.is_running,
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
            "pm_stage": t.get('pm_stage', 0),
            "pnl_history": t.get('pnl_history', []),
        }

    def _closed_trade_dict(self, t):
        orig_margin = t.get('original_margin', t.get('margin', 0))
        pnl_pct = (t['pnl'] / orig_margin) * 100 if orig_margin > 0 else 0
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
        while self.is_running:
            try:
                # 1. Collect symbols (short lock)
                with self.trades_lock:
                    open_syms = [t['symbol'] for t in self.open_trades]
                    pending_syms = [p['symbol'] for p in self.pending_orders]
                    symbols = list(set(open_syms + pending_syms))
                    # Collect BB MR trades needing RSI update
                    rsi_needed = set()
                    for t in self.open_trades:
                        strat = t.get('strategy', '')
                        if strat in ('bb_mr_v5.1', 'bb_mr_v6.0', 'bb_mr_v7.1'):
                            if time.time() - t.get('_last_rsi_update', 0) > 300:
                                rsi_needed.add(t['symbol'])

                if not symbols:
                    time.sleep(3)
                    continue

                # 2. Fetch prices — NO LOCK
                tickers = self._exchange.fetch_tickers(symbols)
                for sym, data in tickers.items():
                    if data and data.get('last'):
                        self.current_prices[sym] = data['last']

                # 3. RSI fetch for BB MR trades — NO LOCK (blocking I/O safe)
                rsi_cache = {}
                for sym in rsi_needed:
                    try:
                        _df = DataFetcher('binance').fetch_ohlcv(sym, '1h', limit=20)
                        if _df is not None and len(_df) >= 14:
                            _df = add_all_indicators(_df)
                            if 'rsi' in _df.columns:
                                _rsi = float(_df['rsi'].iloc[-1])
                                if not np.isnan(_rsi):
                                    rsi_cache[sym] = _rsi
                    except Exception:
                        pass

                # 4. Apply prices + RSI + exit checks (short lock)
                with self.trades_lock:
                    for sym, data in tickers.items():
                        if not (data and data.get('last')):
                            continue
                        price = data['last']
                        self._process_pending_for_symbol(sym, price)
                        self._process_open_for_symbol(sym, price, rsi_cache)

            except Exception as e:
                self.log(f"[WARN] Ticker loop error: {e}")

            self._throttled_save()
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

    def _process_open_for_symbol(self, sym, price, rsi_cache=None):
        """Update PnL snapshots & check exit — under trades_lock."""
        for t in self.open_trades[:]:
            if t['symbol'] != sym:
                continue
            self._record_pnl_snapshot(t, price)

            # VWAP Scalping disabled - not profitable
            # strat = t.get('strategy', '')
            # if 'vwap_scalp' in strat:
            #     self._check_vwap_exit(t, price)
            
            if t.get('strategy', '') in ('ict_smc_v1', 'ict_smc_v2', 'ict_smc_v3'):
                # ICT/SMC → ATR-based BE/trailing
                self._check_ict_exit(t, price)
            elif t.get('strategy', '') == 'trend_v4.4':
                # Trend Following → Late trail + SL + max loss
                self._check_trend_exit(t, price)
            else:
                # Legacy trades → PositionManager
                self._check_v3_manager(t, price)

    def _calc_partial_pnl(self, t, current_price, close_qty):
        """Partial close için PnL hesapla (fee dahil)."""
        n_in = close_qty * t['entry_price']
        n_out = close_qty * current_price
        fee = (n_in + n_out) * 0.0002
        if t['side'] == 'LONG':
            return (n_out - n_in) - fee
        else:
            return (n_in - n_out) - fee

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
        """PositionManagerV3 exit logic + TP check + real partial close. Called under trades_lock."""
        tid = t['id']

        # ── TP Price Check → Partial + Tight Trailing (trend runner) ──
        tp = t.get('tp_price', 0)
        if tp and tp > 0 and not t.get('tp_hit_converted', False):
            hit_tp = (t['side'] == 'LONG' and current_price >= tp) or \
                     (t['side'] == 'SHORT' and current_price <= tp)
            if hit_tp:
                # TP'ye ulaşıldı: %30 kârı al, kalanı sıkı trailing ile tut
                t['tp_hit_converted'] = True
                partial_pct = 0.30
                partial_qty = t['qty'] * partial_pct
                pnl_per_unit = (current_price - t['entry_price']) if t['side'] == 'LONG' \
                               else (t['entry_price'] - current_price)
                partial_pnl = partial_qty * pnl_per_unit

                t['qty'] -= partial_qty
                margin_released = (partial_qty * t['entry_price']) / self.leverage
                t['margin'] = max(0.0, t.get('margin', 0.0) - margin_released)
                self.balance += margin_released + partial_pnl
                t['partial_profit'] = t.get('partial_profit', 0) + partial_pnl

                # PM'i Stage 4'e (sıkı trailing) zorla
                if tid in self.position_managers:
                    pm_ref = self.position_managers[tid]
                    pm_ref.stage = max(pm_ref.stage, 4)
                    atr_val = t.get('atr', t['entry_price'] * 0.005)
                    if t['side'] == 'LONG':
                        tight_trail = current_price - 0.5 * atr_val
                        pm_ref.trailing_stop = max(pm_ref.trailing_stop or 0, tight_trail)
                    else:
                        tight_trail = current_price + 0.5 * atr_val
                        pm_ref.trailing_stop = min(pm_ref.trailing_stop or 999999, tight_trail)
                    t['sl_price'] = pm_ref.trailing_stop

                # TP hedefini kaldır — bundan sonra sadece trailing yönetir
                t['tp_price'] = 0

                self.log(
                    f"[ROCKET] {t['symbol']} TP HIT → TRAIL MODE: "
                    f"Partial {partial_pct:.0%} (+${partial_pnl:.2f}) | "
                    f"Tight trail: {t['sl_price']:.6f} (trend devam ettikçe içeride)"
                )
                self._force_save()
                return

        # ── Initialize PM ──
        if tid not in self.position_managers:
            atr = t.get('atr') or self._get_fallback_atr(t['symbol'], t['entry_price'])
            t['atr'] = atr
            pm = PositionManagerV3(t, atr, leverage=self.leverage)
            pm.stage = t.get('pm_stage', 0)
            pm.peak_price = t.get('pm_highest', t['entry_price'])
            pm.trailing_stop = t.get('pm_trail', None)
            pm.stop = t.get('sl_price', t['entry_price'])
            pm.peak_roi = t.get('pm_peak_roi', 0.0)
            self.position_managers[tid] = pm

        pm = self.position_managers[tid]
        res = pm.update(current_price)

        if res['action'] == 'EXIT':
            self._close_all_locked([t], t['symbol'], current_price, res['reason'])
            return

        if res['action'] == 'PARTIAL':
            # Real partial close: book profit for the partial amount
            partial_pct = res.get('amount', 0.25)
            partial_qty = t['qty'] * partial_pct
            pnl_per_unit = (current_price - t['entry_price']) if t['side'] == 'LONG' \
                           else (t['entry_price'] - current_price)
            partial_pnl = partial_qty * pnl_per_unit

            # Update trade: reduce qty, release margin, book profit
            t['qty'] -= partial_qty
            margin_released = (partial_qty * t['entry_price']) / self.leverage
            t['margin'] = max(0.0, t.get('margin', 0.0) - margin_released)
            self.balance += margin_released + partial_pnl
            t['partial_profit'] = t.get('partial_profit', 0) + partial_pnl

            t['sl_price'] = res['stop']
            self.log(
                f"[MONEY] PARTIAL {t['symbol']}: {res['reason']} "
                f"| Closed {partial_pct:.0%} (+${partial_pnl:.2f}) "
                f"| New SL: {res['stop']:.6f}"
            )
            self._force_save()

        elif res['action'] == 'UPDATE_STOP':
            t['sl_price'] = res['stop']

        # Persist PM state
        t['pm_stage'] = pm.stage
        t['pm_highest'] = pm.peak_price
        t['pm_trail'] = pm.trailing_stop
        t['pm_peak_roi'] = pm.peak_roi
        t['sl_price'] = pm.trailing_stop if pm.trailing_stop else pm.stop

    # ──────────────────────────── v2.1 Risk Filters ────────────────────────────

    def _update_btc_returns(self, fetcher):
        """BTC 1h return serisini 10dk cache ile güncelle."""
        now = time.time()
        if self._btc_returns is not None and now - self._btc_returns_ts < 600:
            return
        try:
            btc = fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=100)
            if btc is not None and len(btc) > 20:
                self._btc_returns = btc['close'].pct_change().dropna()
                self._btc_returns_ts = now
        except Exception:
            pass

    def _check_correlation_limit(self, symbol: str, direction: str, fetcher) -> bool:
        """
        Korelasyon Filtresi: BTC ile yüksek korelasyonlu coinlerde
        aynı yönde max 3 pozisyon.
        BTC/ETH hariç — onlar major, ayrı tutulur.
        """
        if symbol in ("BTC/USDT", "ETH/USDT"):
            return True  # Major'lar her zaman geçer

        self._update_btc_returns(fetcher)
        if self._btc_returns is None or len(self._btc_returns) < 20:
            return True  # Veri yoksa engelleme

        # Bu coin'in BTC korelasyonunu hesapla
        try:
            coin_df = fetcher.fetch_ohlcv(symbol, "1h", limit=100)
            if coin_df is None or len(coin_df) < 30:
                return True
            coin_returns = coin_df['close'].pct_change().dropna()
            min_len = min(len(self._btc_returns), len(coin_returns))
            if min_len < 20:
                return True
            corr = self._btc_returns.iloc[-min_len:].reset_index(drop=True).corr(
                coin_returns.iloc[-min_len:].reset_index(drop=True)
            )
        except Exception:
            return True

        if abs(corr) < 0.80:
            return True  # Düşük korelasyon — sınır yok

        # Yüksek korelasyon: aynı yönde max 3
        with self.trades_lock:
            same_dir_correlated = 0
            for t in self.open_trades:
                if t['side'] == direction and t['symbol'] not in ("BTC/USDT", "ETH/USDT"):
                    same_dir_correlated += 1

        if same_dir_correlated >= 3:
            self.log(f"  [CORR] {symbol} {direction}: BTC corr={corr:.2f}, same-dir={same_dir_correlated}/3 → SKIP")
            return False
        return True

    def _check_loss_limits(self) -> bool:
        """
        Günlük/Haftalık kayıp limiti ve ardışık kayıp molası.
        Returns False → trade açma.
        """
        now = datetime.now()
        today = now.date()
        week_num = now.isocalendar()[1]

        # Günlük reset
        if self._daily_date != today:
            self._daily_date = today
            self._daily_start_balance = self.balance

        # Haftalık reset
        if self._weekly_date != week_num:
            self._weekly_date = week_num
            self._weekly_start_balance = self.balance

        # Cooldown kontrolü (3 ardışık kayıp → 30dk mola)
        if self._cooldown_until > 0:
            if time.time() < self._cooldown_until:
                remaining = int(self._cooldown_until - time.time())
                if remaining % 60 == 0:
                    self.log(f"[COOL] Cooldown aktif: {remaining // 60}dk kaldı")
                return False
            else:
                # Cooldown bitti → sıfırla
                self._cooldown_until = 0
                self.consecutive_losses = 0
                self.log("[COOL] Mola bitti, trade açılabilir")

        # 3 ardışık kayıp → 30 dakika mola
        if self.consecutive_losses >= 3:
            self._cooldown_until = time.time() + 1800  # 30 dk
            self.log(f"[COOL] 3 ardışık kayıp → 30dk mola başladı")
            return False

        # Günlük max %4 kayıp
        if self._daily_start_balance and self._daily_start_balance > 0:
            daily_loss_pct = (self._daily_start_balance - self.balance) / self._daily_start_balance * 100
            if daily_loss_pct >= 4.0:
                self.log(f"[LIMIT] Günlük kayıp limiti: -%{daily_loss_pct:.1f} ≥ %4 → Bugün dur")
                return False

        # Haftalık max %8 kayıp
        if self._weekly_start_balance and self._weekly_start_balance > 0:
            weekly_loss_pct = (self._weekly_start_balance - self.balance) / self._weekly_start_balance * 100
            if weekly_loss_pct >= 8.0:
                self.log(f"[LIMIT] Haftalık kayıp limiti: -%{weekly_loss_pct:.1f} ≥ %8 → Haftayı kapat")
                return False

        return True

    def _get_killzone_multiplier(self) -> float:
        """
        Killzone bazlı position size çarpanı.
        London/NY = 1.2x, Asia = 0.6x, Dead zone = 0 (trade açma).
        """
        utc_now = datetime.utcnow()
        hour = utc_now.hour

        # London Killzone: 07:00-10:00 UTC
        if 7 <= hour < 10:
            return 1.2
        # NY AM Killzone: 12:00-15:00 UTC
        if 12 <= hour < 15:
            return 1.2
        # NY PM / London Close: 15:00-17:00 UTC
        if 15 <= hour < 17:
            return 1.0
        # Asia Session: 00:00-03:00 UTC
        if 0 <= hour < 3:
            return 0.6
        # Dead Zone: 03:00-07:00 UTC (gece yarısı sonrası)
        if 3 <= hour < 7:
            return 0.0  # Trade açma
        # Diğer saatler
        return 0.8

    def _get_scan_interval(self) -> int:
        """Adaptive scan interval: killzone'da hızlı, off-hours'da yavaş."""
        hour = datetime.utcnow().hour
        # NY AM Killzone (12-15 UTC): Her 30s
        if 12 <= hour < 15:
            return 30
        # London Killzone (07-10 UTC): Her 45s
        if 7 <= hour < 10:
            return 45
        # Active hours (10-17 UTC): Her 60s
        if 10 <= hour < 17:
            return 60
        # Off-hours: Her 120s
        return 120

    def _check_daily_structure(self, symbol: str, direction: str, fetcher) -> float:
        """
        1D trend kontrolü. HTF uyum varsa büyük pozisyon, çatışma varsa küçük.
        Returns: multiplier (0.5 = çatışma, 1.0 = nötr, 1.3 = tam uyum)
        """
        try:
            df_1d = fetcher.fetch_ohlcv(symbol, "1d", limit=50)
            if df_1d is None or len(df_1d) < 20:
                return 1.0  # Veri yoksa nötr

            c = df_1d['close'].astype(float)
            ema20 = c.ewm(span=20, adjust=False).mean()
            ema50 = c.ewm(span=50, adjust=False).mean() if len(c) >= 50 else ema20

            cp = float(c.iloc[-1])
            ema20_val = float(ema20.iloc[-1])
            ema50_val = float(ema50.iloc[-1])

            # Daily trend tespiti
            daily_bullish = cp > ema20_val and ema20_val > ema50_val
            daily_bearish = cp < ema20_val and ema20_val < ema50_val

            # Weekly high/low awareness
            weekly_high = float(df_1d['high'].iloc[-5:].max())
            weekly_low = float(df_1d['low'].iloc[-5:].min())
            weekly_range = weekly_high - weekly_low
            near_weekly_high = (weekly_high - cp) / weekly_range < 0.15 if weekly_range > 0 else False
            near_weekly_low = (cp - weekly_low) / weekly_range < 0.15 if weekly_range > 0 else False

            if direction == 'LONG':
                if daily_bullish and not near_weekly_high:
                    return 1.3   # Tam uyum — büyük pozisyon
                elif daily_bearish:
                    return 0.5   # Çatışma — küçük pozisyon
                elif near_weekly_high:
                    return 0.6   # Weekly resistance yakın
            elif direction == 'SHORT':
                if daily_bearish and not near_weekly_low:
                    return 1.3
                elif daily_bullish:
                    return 0.5
                elif near_weekly_low:
                    return 0.6

            return 1.0  # Nötr
        except Exception:
            return 1.0

    def _get_adaptive_threshold(self) -> float:
        """
        Son 20 trade WR'ye göre dinamik meta threshold.
        WR > %70 → threshold düşür (daha çok trade)
        WR < %50 → threshold yükselt (daha az trade)
        """
        with self.trades_lock:
            recent = self.closed_trades[-20:] if len(self.closed_trades) >= 10 else []

        if not recent:
            return 0.65  # Default

        wins = sum(1 for t in recent if t.get('pnl', 0) > 0)
        wr = wins / len(recent)

        if wr >= 0.70:
            thr = 0.55   # İyi dönem → daha çok trade
        elif wr >= 0.60:
            thr = 0.60
        elif wr >= 0.50:
            thr = 0.65   # Normal
        elif wr >= 0.40:
            thr = 0.70   # Kötü dönem → daha az trade
        else:
            thr = 0.75   # Çok kötü → çok seçici

        if thr != self._adaptive_threshold:
            self.log(f"[ADAPT] Threshold: {self._adaptive_threshold:.2f} → {thr:.2f} (WR={wr:.0%}, N={len(recent)})")
            self._adaptive_threshold = thr

        return thr

    # ══════════════════════════════════════════════════════════════════════════
    # BB MR Strategy → strategies/bb_mr_strategy.py (BBMRStrategyMixin)
    # ICT/SMC Strategy → strategies/ict_smc_strategy.py (ICTSMCStrategyMixin)
    # ══════════════════════════════════════════════════════════════════════════

    # ──────────────────────────── Scan Loop ────────────────────────────────────

    def _run_loop(self):
        fetcher = DataFetcher('binance')
        _last_symbol_refresh = 0
        _SYMBOL_REFRESH_INTERVAL = 1800  # 30 dakikada bir

        while self.is_running:
            try:
                # Dynamic symbol refresh (30 dakikada bir)
                if time.time() - _last_symbol_refresh > _SYMBOL_REFRESH_INTERVAL:
                    try:
                        all_pairs = get_all_usdt_pairs()
                        profile = getattr(self, '_active_profile', 'custom')
                        with self.trades_lock:
                            if profile == 'core_v1':
                                limited = all_pairs[:CORE_V1_SYMBOL_LIMIT]
                                self.scanned_symbols = limited if len(limited) >= 8 else list(CORE_V1_SYMBOLS)
                            elif profile == 'core_v2':
                                limited = all_pairs[:CORE_V2_SYMBOL_LIMIT]
                                self.scanned_symbols = limited if len(limited) >= 10 else list(CORE_V2_SYMBOLS)
                            elif profile == 'core_v2_aggressive':
                                limited = all_pairs[:CORE_V2_AGGR_SYMBOL_LIMIT]
                                self.scanned_symbols = limited if len(limited) >= 15 else list(CORE_V2_SYMBOLS)
                            else:
                                self.scanned_symbols = all_pairs
                        if profile == 'core_v1':
                            self.log(
                                f"[SYNC] Core V1 universe refreshed: {len(self.scanned_symbols)} top-liquid pairs"
                            )
                        elif profile == 'core_v2':
                            self.log(
                                f"[SYNC] Core V2 universe refreshed: {len(self.scanned_symbols)} top-liquid pairs"
                            )
                        elif profile == 'core_v2_aggressive':
                            self.log(
                                f"[SYNC] Core V2 Aggressive universe refreshed: {len(self.scanned_symbols)} top-liquid pairs"
                            )
                        else:
                            self.log(
                                f"[SYNC] Top {len(self.scanned_symbols)} liquid USDT pairs (volume-filtered)"
                            )
                        _last_symbol_refresh = time.time()
                    except Exception as e:
                        msg = str(e).lower()
                        if "451" in msg or "restricted" in msg:
                            self.log(
                                "[WARN] Binance region restriction (HTTP 451). "
                                "Using fallback symbols."
                            )
                        else:
                            self.log(f"[WARN] Symbol fetch failed: {e}. Using fallback.")

                # ── Strategy 1: VWAP Scalping (DISABLED - not profitable) ──
                # if self._strategy_enabled.get('vwap_scalp', False):
                #     self.log(f"[SEARCH] VWAP Scalping v1.0 scanning {len(self.scanned_symbols)} markets...")
                #     self._vwap_scan(fetcher)

                # ── Strategy 2: ICT/SMC ──
                if self._strategy_enabled.get('ict_smc', True):
                    self.log(f"[SEARCH] ICT/SMC v2.4 scanning {len(self.scanned_symbols)} markets...")
                    self._ict_smc_scan(fetcher)

                # ── Strategy 3: Trend Following v4.4 ──
                if self._strategy_enabled.get('trend_v4', True):
                    self.log(f"[SEARCH] Trend v4.4 scanning {len(self.scanned_symbols)} markets...")
                    self._trend_scan(fetcher)

            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate" in err:
                    self.log(f"[WARN] Rate limit: {e}. Backing off...")
                    time.sleep(5)
                else:
                    self.log(f"[FAIL] Scan loop error: {e}")
                time.sleep(0.5)

            # Adaptive scan interval (killzone bazlı)
            scan_wait = self._get_scan_interval()
            for _ in range(scan_wait):
                if not self.is_running:
                    break
                time.sleep(1)


    def _scan_and_trade(self, fetcher: DataFetcher):
        # ── DEPRECATED: Legacy scan — artık kullanılmıyor. _bb_mr_scan + _ict_smc_scan aktif. ──
        # ── v2.1: Günlük/Haftalık kayıp limiti ──
        if not self._check_loss_limits():
            return

        # ── v2.1: Killzone dead zone kontrolü ──
        kz_mult = self._get_killzone_multiplier()
        if kz_mult <= 0:
            self.log("[KZ] Dead zone (03:00-07:00 UTC) — trade açılmaz")
            return

        # ── v2.1: BTC korelasyon cache güncelle ──
        self._update_btc_returns(fetcher)

        # ── v2.1: Adaptive threshold güncelle ──
        adaptive_thr = self._get_adaptive_threshold()

        # BTC bias (non-blocking: scan continues even if BTC fails)
        btc_4h = None
        try:
            btc_4h = fetcher.fetch_ohlcv("BTC/USDT", "4h", limit=100)
            if btc_4h is not None and btc_4h.empty:
                btc_4h = None
        except Exception as e:
            self.log(f"[WARN] BTC 4h fetch error: {e}. Continuing without BTC bias.")
            btc_4h = None

        skipped_prefilter = 0

        for i, symbol in enumerate(self.scanned_symbols[:]):
            if not self.is_running:
                break

            try:
                if i % 30 == 0:
                    self.log(f"[*][*] Scanning {symbol} ({i}/{len(self.scanned_symbols)})")

                # ── STEP 0: Early checks BEFORE any API calls ──
                with self.trades_lock:
                    has_active = any(
                        t['symbol'] == symbol for t in self.open_trades
                    )
                    has_pending = any(
                        p['symbol'] == symbol for p in self.pending_orders
                    )
                    active_count = len(self.open_trades) + len(self.pending_orders)
                    can_open = active_count < self.max_open_trades_limit

                    # Direction diversity: max %60 aynı yön
                    long_count = sum(1 for t in self.open_trades if t['side'] == 'LONG')
                    short_count = sum(1 for t in self.open_trades if t['side'] == 'SHORT')
                    max_one_dir = max(3, int(self.max_open_trades_limit * 0.6))

                    # Strategy diversity: max 4 trade per strategy
                    strategy_counts = {}
                    for t in self.open_trades:
                        sname = t.get('strategy', 'unknown')
                        strategy_counts[sname] = strategy_counts.get(sname, 0) + 1

                if has_active or has_pending:
                    continue

                # ── STEP 1: Fetch only 1h data ──
                df_1h = fetcher.fetch_ohlcv(symbol, self.timeframe, limit=100)
                if df_1h.empty or len(df_1h) < 50:
                    continue

                # ── STEP 2: Quick 1h pre-filter (no 15m/4h API calls yet) ──
                if not has_1h_signal(df_1h, symbol, self.timeframe, self.secondary_tf):
                    skipped_prefilter += 1
                    continue

                # ── STEP 3: Signal found! Fetch 15m + 4h ──
                if not can_open:
                    continue

                df_15m = fetcher.fetch_ohlcv(symbol, self.secondary_tf, limit=100)
                if df_15m.empty or len(df_15m) < 50:
                    continue

                df_4h = fetcher.fetch_ohlcv(symbol, "4h", limit=100)
                if df_4h.empty or len(df_4h) < 50:
                    continue

                # ── STEP 4: Full signal generation with all timeframes ──
                decision = generate_signal(
                    df=df_1h,
                    df_secondary=df_15m,
                    df_4h=df_4h,
                    symbol=symbol,
                    timeframe=self.timeframe,
                    secondary_tf=self.secondary_tf
                )

                if should_open_position(decision):
                    sig_dir = decision['signal']

                    # Direction diversity check
                    if sig_dir == 'LONG' and long_count >= max_one_dir:
                        print(f"  [SKIP] {symbol} LONG: direction diversity (L={long_count}/{max_one_dir})")
                        continue
                    if sig_dir == 'SHORT' and short_count >= max_one_dir:
                        print(f"  [SKIP] {symbol} SHORT: direction diversity (S={short_count}/{max_one_dir})")
                        continue

                    # Strategy diversity check (max 4 per strategy)
                    sig_strat = decision.get('strategy', 'unknown')
                    if strategy_counts.get(sig_strat, 0) >= 4:
                        print(f"  [SKIP] {symbol}: strategy diversity ({sig_strat}={strategy_counts[sig_strat]}/4)")
                        continue
                    if self.consecutive_losses >= 10:
                        self.log("[STOP] ENGINE HALT: Max consecutive losses limit reached")
                        self.stop()
                        break

                    # ── v2.1: Korelasyon filtresi ──
                    if not self._check_correlation_limit(symbol, sig_dir, fetcher):
                        continue

                    # ── v2.1: Adaptive threshold ──
                    meta_conf = decision.get('meta_confidence', 0.0)
                    if meta_conf > 0 and meta_conf < adaptive_thr:
                        print(f"  [ADAPT] {symbol}: meta={meta_conf:.2f} < thr={adaptive_thr:.2f} → SKIP")
                        continue

                    # ══════════════════════════════════════════════
                    # 5-LAYER ENTRY GATE v2.1
                    # ══════════════════════════════════════════════
                    cp = float(df_1h['close'].iloc[-1])

                    # Ön TP/SL hesapla
                    tp_price = decision.get('tp_price', 0.0)
                    sl_price = decision.get('sl_price', 0.0)
                    if tp_price == 0.0 or sl_price == 0.0:
                        atr_proxy = float(df_1h['high'].iloc[-14:].max() - df_1h['low'].iloc[-14:].min())
                        tp_price, sl_price = get_tp_sl_prices(decision, cp, atr_proxy)

                    # 5m veri çek (Katman 3 & 4 için)
                    df_5m = None
                    try:
                        df_5m = fetcher.fetch_ohlcv(symbol, "5m", limit=60)
                        if df_5m is not None and len(df_5m) < 15:
                            df_5m = None
                    except Exception:
                        pass

                    gate = five_layer_gate(
                        direction=sig_dir,
                        df_1h=df_1h,
                        df_5m=df_5m,
                        df_4h=df_4h,
                        entry_price=cp,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        meta_confidence=decision.get('meta_confidence', 0.5),
                        regime=decision.get('regime', 'trending'),
                        strategy=decision.get('strategy', 'unknown'),
                    )

                    if not gate.passed:
                        print(f"  [GATE] {symbol} {sig_dir}: {gate.summary()}")
                        continue

                    # Gate geçti — yapısal SL/TP kullan
                    sl_price = gate.sl_price if gate.sl_price > 0 else sl_price
                    tp_price = gate.tp_price if gate.tp_price > 0 else tp_price

                    # Hard SL distance cap: max 2.5% (gate zaten kontrol etti ama güvenlik)
                    max_sl_dist = cp * 0.025
                    raw_sl_dist = abs(cp - sl_price)
                    if raw_sl_dist > max_sl_dist:
                        if sig_dir == 'LONG':
                            sl_price = cp - max_sl_dist
                        else:
                            sl_price = cp + max_sl_dist

                    risk_abs = abs(cp - sl_price)
                    if risk_abs <= 0: risk_abs = cp * 0.01

                    # Size calculation (v2.1: killzone + daily structure çarpanları)
                    loss_scaling = {0: 1.0, 1: 1.0, 2: 0.8, 3: 0.6}.get(self.consecutive_losses, 0.5)
                    position_conf = decision.get('position_size', 0.5)
                    htf_mult = self._check_daily_structure(symbol, sig_dir, fetcher)
                    risk_amount = self.balance * self.risk_pct * loss_scaling * position_conf * kz_mult * htf_mult
                    raw_qty = risk_amount / risk_abs
                    max_qty = (self.balance * self.leverage) / cp
                    max_margin = self.balance * 0.02 * position_conf
                    max_qty_margin = (max_margin * self.leverage) / cp
                    qty = min(raw_qty, max_qty, max_qty_margin)

                    logger_id = f"{symbol}_{sig_dir}_{int(time.time())}"
                    atr_val = risk_abs / decision.get('sl_mult', 1.0)

                    print(f"  [ENTRY] {symbol} {sig_dir}: {gate.summary()}")

                    with self.trades_lock:
                        self._open_locked(
                            symbol=symbol,
                            side=sig_dir,
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

        if skipped_prefilter > 0:
            self.log(f"[PERF] Pre-filter: {skipped_prefilter}/{len(self.scanned_symbols)} symbols skipped (no 1h signal)")

    # ──────────────────────────── 5m Precision Entry ──────────────────────────

    @staticmethod
    def _confirm_5m_entry(df_5m: pd.DataFrame, direction: str, symbol: str) -> bool:
        """
        DEPRECATED: entry_gate.py Layer 3 (TETİK) bu işlevi üstlendi.
        5m chart'ta yapısal giriş onayı.
        1h sinyal + 15m zamanlama geçtikten sonra, 5m'de micro-structure aranır.

        LONG onayı:
          - Son 3 mum içinde bullish yapı (engulfing/hammer/higher low)
          - EMA9 üstünde veya EMA9'a doğru bounce
          - RSI 35-70 arası (ne aşırı satılmış ne aşırı alınmış)
          - Momentum pozitif (son close > 3 bar önceki close)

        SHORT onayı: Tam tersi.
        """
        try:
            if df_5m is None or len(df_5m) < 20:
                return True  # 5m verisi yoksa engelleme

            c = df_5m['close'].astype(float)
            o = df_5m['open'].astype(float)
            h = df_5m['high'].astype(float)
            l = df_5m['low'].astype(float)

            # EMA9 hesapla
            ema9 = c.ewm(span=9, adjust=False).mean()
            cp = float(c.iloc[-1])
            ema9_val = float(ema9.iloc[-1])

            # RSI14 hesapla
            delta = c.diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / loss.replace(0, 1e-10)
            rsi = 100 - (100 / (1 + rs))
            rsi_val = float(rsi.iloc[-1]) if not rsi.empty else 50.0

            # Son 3 mum analizi
            bodies = []
            wicks_upper = []
            wicks_lower = []
            for i in range(-3, 0):
                body = float(c.iloc[i] - o.iloc[i])
                full_range = float(h.iloc[i] - l.iloc[i])
                if full_range > 0:
                    upper_wick = float(h.iloc[i] - max(c.iloc[i], o.iloc[i])) / full_range
                    lower_wick = float(min(c.iloc[i], o.iloc[i]) - l.iloc[i]) / full_range
                else:
                    upper_wick = 0.0
                    lower_wick = 0.0
                bodies.append(body)
                wicks_upper.append(upper_wick)
                wicks_lower.append(lower_wick)

            # Momentum: son close vs 5 bar önceki
            mom_5 = (cp - float(c.iloc[-6])) / float(c.iloc[-6]) * 100 if len(c) >= 6 else 0.0

            # Higher low / Lower high
            recent_lows = [float(l.iloc[i]) for i in range(-4, 0)]
            recent_highs = [float(h.iloc[i]) for i in range(-4, 0)]
            higher_low = recent_lows[-1] > min(recent_lows[:-1]) if len(recent_lows) >= 2 else False
            lower_high = recent_highs[-1] < max(recent_highs[:-1]) if len(recent_highs) >= 2 else False

            if direction == 'LONG':
                # Bullish checks
                bullish_candle = bodies[-1] > 0  # Son mum yeşil
                bullish_engulf = bodies[-1] > 0 and bodies[-2] < 0 and abs(bodies[-1]) > abs(bodies[-2])
                hammer = lower_wick[-1] > 0.5 and bodies[-1] > 0  # Uzun alt fitil + yeşil
                ema_ok = cp >= ema9_val * 0.998  # EMA9 üstü veya çok yakın
                rsi_ok = 30 < rsi_val < 72  # Aşırı alınmış değil
                mom_ok = mom_5 > -0.3  # Güçlü düşüş yok

                structure_ok = bullish_candle or bullish_engulf or hammer or higher_low
                confirmed = structure_ok and ema_ok and rsi_ok and mom_ok

                if not confirmed:
                    reasons = []
                    if not structure_ok: reasons.append("no_bull_structure")
                    if not ema_ok: reasons.append(f"below_ema9")
                    if not rsi_ok: reasons.append(f"rsi={rsi_val:.0f}")
                    if not mom_ok: reasons.append(f"mom={mom_5:.1f}%")
                    print(f"  [5m REJECT] {symbol} LONG: {', '.join(reasons)}")
                    return False

            elif direction == 'SHORT':
                # Bearish checks
                bearish_candle = bodies[-1] < 0  # Son mum kırmızı
                bearish_engulf = bodies[-1] < 0 and bodies[-2] > 0 and abs(bodies[-1]) > abs(bodies[-2])
                shooting_star = wicks_upper[-1] > 0.5 and bodies[-1] < 0
                ema_ok = cp <= ema9_val * 1.002  # EMA9 altı veya çok yakın
                rsi_ok = 28 < rsi_val < 70  # Aşırı satılmış değil
                mom_ok = mom_5 < 0.3  # Güçlü yükseliş yok

                structure_ok = bearish_candle or bearish_engulf or shooting_star or lower_high
                confirmed = structure_ok and ema_ok and rsi_ok and mom_ok

                if not confirmed:
                    reasons = []
                    if not structure_ok: reasons.append("no_bear_structure")
                    if not ema_ok: reasons.append(f"above_ema9")
                    if not rsi_ok: reasons.append(f"rsi={rsi_val:.0f}")
                    if not mom_ok: reasons.append(f"mom={mom_5:.1f}%")
                    print(f"  [5m REJECT] {symbol} SHORT: {', '.join(reasons)}")
                    return False

            return True
        except Exception as e:
            print(f"  [5m WARN] {symbol}: {e}")
            return True  # Hata durumunda engelleme

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
        # Hard guard: never exceed max open trades (prevents TOCTOU race)
        if len(self.open_trades) >= self.max_open_trades_limit:
            return None

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
            "original_margin": margin,
            "entry_time": datetime.now().strftime("%H:%M:%S"),
            "entry_timestamp": time.time(),
            "tp_price": tp_price,
            "sl_price": sl_price,
            "atr": atr,
            "strategy": sig.get('strategy', 'unknown'),
            "regime": sig.get('regime', 'unknown'),
            "entry_type": sig.get('entry_type', 'none'),
            "soft_score": sig.get('soft_score', 0),
            "signal_result": signal_result,
            "logger_id": logger_id or (
                f"{symbol}_{side}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            ),
            "max_pnl_pct": 0,
            "pnl_history": [],
        }

        self.open_trades.append(t)
        # BB MR trades → pure TP/SL, PositionManager gereksiz
        if sig.get('strategy', '') not in ('bb_mr_v5.1', 'bb_mr_v6.0', 'bb_mr_v7.1'):
            self.position_managers[tid] = PositionManagerV3(t, atr, leverage=self.leverage)

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

            # Total PnL = kalan kısmın PnL'si + partial close'lardan gelen kâr
            partial_profit = t.get('partial_profit', 0)
            total_pnl = pnl + partial_profit

            t['exit_price'] = exit_price
            t['exit_time'] = datetime.now().strftime("%H:%M:%S")
            t['pnl'] = total_pnl
            t['reason'] = reason

            self.closed_trades.append(t)
            self.open_trades.remove(t)

            # Manager cleanup
            self.position_managers.pop(t['id'], None)

            # is_win total PnL'e bakmalı (partial profit dahil)
            is_win = total_pnl > 0
            if is_win:
                self.consecutive_losses = max(0, self.consecutive_losses - 1)
                # v2.1: Win gelince cooldown sıfırla
                if self.consecutive_losses < 3:
                    self._cooldown_until = 0
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
            pnl_str = f"+${total_pnl:.2f}" if total_pnl > 0 else f"-${abs(total_pnl):.2f}"
            icon = "[WIN]" if total_pnl > 0 else "[LOSS]"
            partial_note = f" (incl ${partial_profit:+.2f} partial)" if partial_profit != 0 else ""
            self.log(
                f"{icon} CLOSE {t['side']} {symbol} @ {exit_price:.6f} | "
                f"PnL: {pnl_str} ({pnl_pct:.2f}%){partial_note} | "
                f"Max: {max_p:.2f}% | Reason: {reason}"
            )

        if trades:
            self._force_save()

    # ──────────────────────────── Manual Actions ──────────────────────────────

    def close_trade(self, trade_id: str):
        # 1. Find trade + symbol (short lock)
        with self.trades_lock:
            trade = next(
                (t for t in self.open_trades if t['id'] == trade_id), None
            )
            if not trade:
                return {"success": False, "message": "Trade not found."}
            sym = trade['symbol']
            cp = self.current_prices.get(sym, trade['entry_price'])

        # 2. Live price fallback — NO LOCK (blocking I/O safe)
        if cp == trade['entry_price']:
            try:
                ticker = self._exchange.fetch_ticker(sym)
                if ticker and ticker.get('last'):
                    cp = ticker['last']
            except Exception:
                pass

        # 3. Close trade (short lock)
        with self.trades_lock:
            if trade not in self.open_trades:
                return {"success": False, "message": "Trade already closed."}
            self._close_all_locked([trade], sym, cp, "MANUAL_CLOSE")

        return {
            "success": True,
            "message": f"Trade {trade_id} closed at {cp:.4f}.",
        }

    def apply_profile_core_v1(self):
        """
        Core V1 profile:
          - Trend-only execution
          - Curated high-liquidity symbol universe
          - Conservative risk envelope
        """
        with self.trades_lock:
            self._active_profile = 'core_v1'
            self._strategy_enabled['bb_mr'] = False
            self._strategy_enabled['ict_smc'] = False
            self._strategy_enabled['trend_v4'] = True

            all_pairs = get_all_usdt_pairs()
            limited = all_pairs[:CORE_V1_SYMBOL_LIMIT]
            self.scanned_symbols = limited if len(limited) >= 8 else list(CORE_V1_SYMBOLS)
            self.max_open_trades_limit = 3

            if self.leverage > 10:
                self.leverage = 10

            self._user_max_notional = 250.0
            self._user_max_loss_cap = 4.5
            self._TREND_PARAMS['min_adx'] = 18
            self._TREND_PARAMS['min_vol_ratio'] = 0.70
            self._TREND_PARAMS['cooldown_hours'] = 1.5
            self._TREND_PARAMS['timeout_hours'] = 72
            self._TREND_PARAMS['max_notional'] = 250.0
            self._TREND_PARAMS['max_loss_cap'] = 4.5
            self._ICT_PARAMS['max_notional'] = 250.0
            self._ICT_PARAMS['max_loss_cap'] = 4.5

            self._adaptive_threshold = max(self._adaptive_threshold, 0.70)

        self._force_save()
        self.log(
            "[PROFILE] Core V1 applied: Trend-only, max_trades=3, "
            f"symbols={len(self.scanned_symbols)}, leverage<=10, notional=$250, max_loss=$4.5"
        )
        return {
            "success": True,
            "profile": "core_v1",
            "message": "Core V1 profile applied",
            "settings": {
                "max_open_trades": self.max_open_trades_limit,
                "leverage": self.leverage,
                "max_notional": self._user_max_notional,
                "max_loss_cap": self._user_max_loss_cap,
                "trend_min_adx": self._TREND_PARAMS.get('min_adx'),
                "trend_min_vol_ratio": self._TREND_PARAMS.get('min_vol_ratio'),
                "bb_mr_enabled": self._strategy_enabled.get('bb_mr', False),
                "ict_smc_enabled": self._strategy_enabled.get('ict_smc', False),
                "trend_v4_enabled": self._strategy_enabled.get('trend_v4', True),
                "symbols": self.scanned_symbols,
            },
        }

    def apply_profile_core_v2_aggressive(self):
        """
        Core V2 Aggressive profile:
          - ICT/SMC-only execution with higher throughput
          - Dynamic top-60 high-liquidity universe
          - Higher risk budget for monthly growth targeting
        """
        with self.trades_lock:
            self._active_profile = 'core_v2_aggressive'
            self._strategy_enabled['bb_mr'] = False
            self._strategy_enabled['ict_smc'] = True
            self._strategy_enabled['trend_v4'] = False

            all_pairs = get_all_usdt_pairs()
            limited = all_pairs[:CORE_V2_AGGR_SYMBOL_LIMIT]
            self.scanned_symbols = limited if len(limited) >= 15 else list(CORE_V2_SYMBOLS)
            self.max_open_trades_limit = 5

            if self.leverage > 10:
                self.leverage = 10

            self._user_max_notional = 300.0
            self._user_max_loss_cap = 5.8
            self._ICT_PARAMS['max_notional'] = 300.0
            self._ICT_PARAMS['max_loss_cap'] = 5.8
            self._ICT_PARAMS['min_rr'] = 1.45
            self._ICT_PARAMS['max_sl_pct'] = 2.5
            self._ICT_PARAMS['min_confluence'] = 2
            self._ICT_PARAMS['sr_proximity_limit'] = 0.035
            self._ICT_PARAMS['max_poi_distance_atr'] = 6.0
            self._ICT_PARAMS['max_zone_age_bars'] = 84
            self._ICT_PARAMS['entry_pullback_min_pct'] = 0.015
            self._ICT_PARAMS['entry_range_lookback'] = 20
            self._ICT_PARAMS['entry_range_top_ceil'] = 0.85
            self._ICT_PARAMS['entry_range_bot_floor'] = 0.15
            self._ICT_PARAMS['entry_max_ema21_ext'] = 0.045
            self._ICT_PARAMS['trend_recent_bars_4h'] = 42
            self._ICT_PARAMS['trend_min_labels'] = 7
            self._ICT_PARAMS['trend_max_labels'] = 20
            self._ICT_PARAMS['swing_params'] = {
                '15m': {'left': 3, 'right': 2},
                '1h': {'left': 4, 'right': 2},
                '4h': {'left': 5, 'right': 4},
                '1d': {'left': 7, 'right': 5},
            }
            self._ICT_PARAMS['rsi_overbought'] = 76.0
            self._ICT_PARAMS['rsi_oversold'] = 24.0
            self._ICT_PARAMS['rsi_extreme_overbought'] = 88.0
            self._ICT_PARAMS['rsi_extreme_oversold'] = 12.0
            self._ICT_PARAMS['entry_trigger_tf'] = '15m'
            self._ICT_PARAMS['structure_tf'] = '1h'
            self._ICT_PARAMS['hard_exit_tf'] = '4h'
            self._ICT_PARAMS['use_dynamic_partials'] = True

            self._adaptive_threshold = min(self._adaptive_threshold, 0.65)

        self._force_save()
        self.log(
            "[PROFILE] Core V2 Aggressive applied: ICT-only, max_trades=5, "
            f"symbols={len(self.scanned_symbols)}, leverage<=10, notional=$300, max_loss=$5.8"
        )
        return {
            "success": True,
            "profile": "core_v2_aggressive",
            "message": "Core V2 Aggressive profile applied",
            "settings": {
                "max_open_trades": self.max_open_trades_limit,
                "leverage": self.leverage,
                "max_notional": self._user_max_notional,
                "max_loss_cap": self._user_max_loss_cap,
                "bb_mr_enabled": self._strategy_enabled.get('bb_mr', False),
                "ict_smc_enabled": self._strategy_enabled.get('ict_smc', True),
                "trend_v4_enabled": self._strategy_enabled.get('trend_v4', False),
                "entry_trigger_tf": self._ICT_PARAMS.get('entry_trigger_tf', '15m'),
                "structure_tf": self._ICT_PARAMS.get('structure_tf', '1h'),
                "hard_exit_tf": self._ICT_PARAMS.get('hard_exit_tf', '4h'),
                "symbols": self.scanned_symbols,
            },
        }

    def apply_profile_core_v2(self):
        """
        Core V2 profile:
          - ICT/SMC-only execution (multi-setup: liquidity + MSS/BOS + OB/FVG)
          - Dynamic top-50 high-liquidity universe
          - Conservative, runner-friendly risk envelope
        """
        with self.trades_lock:
            self._active_profile = 'core_v2'
            self._strategy_enabled['bb_mr'] = False
            self._strategy_enabled['ict_smc'] = True
            self._strategy_enabled['trend_v4'] = False

            all_pairs = get_all_usdt_pairs()
            limited = all_pairs[:CORE_V2_SYMBOL_LIMIT]
            self.scanned_symbols = limited if len(limited) >= 10 else list(CORE_V2_SYMBOLS)
            self.max_open_trades_limit = 4

            if self.leverage > 10:
                self.leverage = 10

            self._user_max_notional = 300.0
            self._user_max_loss_cap = 5.8
            self._ICT_PARAMS['max_notional'] = 300.0
            self._ICT_PARAMS['max_loss_cap'] = 5.8
            self._ICT_PARAMS['min_rr'] = 1.45
            self._ICT_PARAMS['max_sl_pct'] = 2.5
            self._ICT_PARAMS['min_confluence'] = 2
            self._ICT_PARAMS['sr_proximity_limit'] = 0.04
            self._ICT_PARAMS['max_poi_distance_atr'] = 5.0
            self._ICT_PARAMS['max_zone_age_bars'] = 72
            self._ICT_PARAMS['entry_pullback_min_pct'] = 0.015
            self._ICT_PARAMS['entry_range_lookback'] = 20
            self._ICT_PARAMS['entry_range_top_ceil'] = 0.85
            self._ICT_PARAMS['entry_range_bot_floor'] = 0.15
            self._ICT_PARAMS['entry_max_ema21_ext'] = 0.045
            self._ICT_PARAMS['trend_recent_bars_4h'] = 48
            self._ICT_PARAMS['trend_min_labels'] = 8
            self._ICT_PARAMS['trend_max_labels'] = 20
            self._ICT_PARAMS['swing_params'] = {
                '15m': {'left': 3, 'right': 2},
                '1h': {'left': 5, 'right': 3},
                '4h': {'left': 5, 'right': 5},
                '1d': {'left': 7, 'right': 5},
            }
            self._ICT_PARAMS['rsi_overbought'] = 75.0
            self._ICT_PARAMS['rsi_oversold'] = 25.0
            self._ICT_PARAMS['rsi_extreme_overbought'] = 85.0
            self._ICT_PARAMS['rsi_extreme_oversold'] = 15.0
            self._ICT_PARAMS['entry_trigger_tf'] = '15m'
            self._ICT_PARAMS['structure_tf'] = '1h'
            self._ICT_PARAMS['hard_exit_tf'] = '4h'
            self._ICT_PARAMS['use_dynamic_partials'] = True

            self._adaptive_threshold = max(self._adaptive_threshold, 0.70)

        self._force_save()
        self.log(
            "[PROFILE] Core V2 applied: ICT-only, max_trades=4, "
            f"symbols={len(self.scanned_symbols)}, leverage<=10, notional=$300, max_loss=$5.8"
        )
        return {
            "success": True,
            "profile": "core_v2",
            "message": "Core V2 profile applied",
            "settings": {
                "max_open_trades": self.max_open_trades_limit,
                "leverage": self.leverage,
                "max_notional": self._user_max_notional,
                "max_loss_cap": self._user_max_loss_cap,
                "bb_mr_enabled": self._strategy_enabled.get('bb_mr', False),
                "ict_smc_enabled": self._strategy_enabled.get('ict_smc', True),
                "trend_v4_enabled": self._strategy_enabled.get('trend_v4', False),
                "entry_trigger_tf": self._ICT_PARAMS.get('entry_trigger_tf', '15m'),
                "structure_tf": self._ICT_PARAMS.get('structure_tf', '1h'),
                "hard_exit_tf": self._ICT_PARAMS.get('hard_exit_tf', '4h'),
                "symbols": self.scanned_symbols,
            },
        }

    def update_settings(self, max_open_trades: int = None, balance: float = None,
                         leverage: int = None, max_notional: float = None,
                         max_loss_cap: float = None,
                         bb_mr_enabled: bool = None, ict_smc_enabled: bool = None,
                         trend_v4_enabled: bool = None, **kwargs):
        changes = []
        # ── Global Ayarlar ──
        if max_open_trades is not None and max_open_trades > 0:
            self.max_open_trades_limit = max_open_trades
            changes.append(f"max_trades={max_open_trades}")
        if balance is not None and balance > 0:
            self.balance = balance
            changes.append(f"balance=${balance:.2f}")
        if leverage is not None and leverage > 0:
            self.leverage = leverage
            changes.append(f"leverage={leverage}x")
        if max_notional is not None and max_notional > 0:
            self._user_max_notional = max_notional
            # Tüm stratejilere uygula
            self._ICT_PARAMS['max_notional'] = max_notional
            self._TREND_PARAMS['max_notional'] = max_notional
            changes.append(f"trade_size=${max_notional:.0f}")
        if max_loss_cap is not None and max_loss_cap > 0:
            self._user_max_loss_cap = max_loss_cap
            # Tüm stratejilere uygula
            self._ICT_PARAMS['max_loss_cap'] = max_loss_cap
            self._TREND_PARAMS['max_loss_cap'] = max_loss_cap
            changes.append(f"max_loss=${max_loss_cap:.1f}")

        # ── Strateji enable/disable ──
        if bb_mr_enabled is not None:
            self._strategy_enabled['bb_mr'] = bb_mr_enabled
            changes.append(f"bb_mr={'ON' if bb_mr_enabled else 'OFF'}")
        if ict_smc_enabled is not None:
            self._strategy_enabled['ict_smc'] = ict_smc_enabled
            changes.append(f"ict_smc={'ON' if ict_smc_enabled else 'OFF'}")
        if trend_v4_enabled is not None:
            self._strategy_enabled['trend_v4'] = trend_v4_enabled
            changes.append(f"trend_v4={'ON' if trend_v4_enabled else 'OFF'}")

        if not changes:
            return {"success": False, "message": "No valid settings provided"}
        self._active_profile = 'custom'
        self._force_save()
        self.log(f"[GEAR] Settings updated: {', '.join(changes)}")
        return {"success": True, "message": f"Updated: {', '.join(changes)}"}

    def get_risk_settings(self):
        return {
            "profile": getattr(self, '_active_profile', 'custom'),
            "symbol_count": len(self.scanned_symbols),
            "max_open_trades": self.max_open_trades_limit,
            "balance": round(self.balance, 2),
            "leverage": self.leverage,
            "max_notional": getattr(self, '_user_max_notional', 300.0),
            "max_loss_cap": getattr(self, '_user_max_loss_cap', 5.0),
            "bb_mr_enabled": self._strategy_enabled.get('bb_mr', True),
            "ict_smc_enabled": self._strategy_enabled.get('ict_smc', True),
            "trend_v4_enabled": self._strategy_enabled.get('trend_v4', True),
        }

    def reset_system(self, new_balance: float = 10000.0):
        was_running = self.is_running
        if was_running:
            self.stop()
            time.sleep(2)
        with self.trades_lock:
            self.balance = new_balance
            self.open_trades = []
            self.pending_orders = []
            self.closed_trades = []
            self.trade_counter = 0
            self.consecutive_losses = 0
            self.symbol_consecutive_losses = {}
            # BB MR reset
            self._bb_consecutive_sl = 0
            self._bb_cooldown_until = 0
            self._bb_recent_outcomes = []
            self._bb_last_trade_time = 0
            self._bb_symbol_cooldown = {}
            # ICT reset
            self._ict_consecutive_sl = 0
            self._ict_cooldown_until = 0
            self._ict_last_trade_time = 0
            self._ict_symbol_cooldown = {}
            self._ict_recent_outcomes = []
            self.logs = []
        self._force_save()
        self.log(f"[RESET] System reset. Balance: ${new_balance:.2f}")
        return {"success": True, "message": f"System reset. Balance: ${new_balance:.2f}"}

    def get_analytics(self):
        with self.trades_lock:
            all_closed = list(self.closed_trades)
        result = []
        for t in all_closed:
            pnl = t.get('pnl', 0)
            margin = t.get('original_margin', t.get('margin', 0))
            pnl_pct = (pnl / margin * 100) if margin > 0 else 0
            notional = t.get('entry_price', 0) * t.get('qty', 0)
            result.append({
                "id": t.get('id', ''),
                "symbol": t.get('symbol', ''),
                "side": t.get('side', ''),
                "entry_price": t.get('entry_price', 0),
                "exit_price": t.get('exit_price', 0),
                "qty": t.get('qty', 0),
                "margin": margin,
                "notional": round(notional, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "max_pnl_pct": round(t.get('max_pnl_pct', 0), 2),
                "strategy": t.get('strategy', 'unknown'),
                "entry_time": t.get('entry_time', ''),
                "exit_time": t.get('exit_time', ''),
                "reason": t.get('close_reason', t.get('reason', 'unknown')),
                "killzone": t.get('killzone', ''),
            })
        return {"trades": result, "total": len(result)}
