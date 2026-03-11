import time
import threading
import pandas as pd
from typing import Dict, List, Optional
import json
import os
from datetime import datetime

from backtest.data_fetcher import DataFetcher
from backtest.signals import add_all_indicators
from ai.data_sources.futures_data import enrich_ohlcv_with_futures
from ai.xgboost_trainer import generate_features
from ai.adaptive_live_adapter import generate_signal, should_open_position, calculate_position_amount, get_tp_sl_prices

# Taranacak on tanimli market cap'i yuksek hacimli coinler
SCAN_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", 
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "POL/USDT"
]

import ccxt
def get_all_usdt_pairs():
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    excluded = ["USDC/USDT", "USD1/USDT", "TUSD/USDT", "FDUSD/USDT"]
    usdt_pairs = [symbol for symbol in markets if symbol.endswith('/USDT') and symbol not in excluded and markets[symbol].get('active') and markets[symbol].get('spot')]
    return usdt_pairs

class LivePaperTrader:
    """
    Background'da calisan Otonom Paper Trading (Sanal Bakiye) servisi.
    Belirlenen coin'leri periyodik(orn. 1dk'da bir 15m verisine gore) tarar, 
    Quant AI mantigina gore sinyal bulursa islem acar/kapatir.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LivePaperTrader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, initial_balance=10000.0, leverage=10, risk_pct=1.5):
        if self._initialized: return
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
        
        self.open_trades = []
        self.closed_trades = []
        self.logs = []
        self.current_prices = {}

        
        # Sinyalleri yonetmek adina gecmis engine orneklerini tutmak isteyebiliriz
        # Su anlik stateless bir live scalping yapiyoruz (Son muma bakip islem karar verilir)
        self.last_scan_time = None
        self.trade_counter = 0
        self.max_dca_levels = 5
        self.dca_spacing_pct = 0.01
        
        self.scanned_symbols = SCAN_SYMBOLS
        self.timeframe = "1h"
        self.secondary_tf = "15m"

        # v2.0 Tracking
        self.consecutive_losses = 0
        self.symbol_consecutive_losses = {} # symbol -> count

        self.load_state()

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    self.balance = state.get('balance', self.initial_balance)
                    self.open_trades = state.get('open_trades', [])
                    self.closed_trades = state.get('closed_trades', [])
                    self.trade_counter = state.get('trade_counter', 0)
                    self.max_open_trades_limit = state.get('max_open_trades_limit', 5)
                    
                    loaded_logs = state.get('logs', [])
                    self.logs = []
                    import uuid
                    for l in loaded_logs:
                        if isinstance(l, str):
                            self.logs.append({"id": str(uuid.uuid4())[:8], "text": l})
                        else:
                            self.logs.append(l)
                            
                self.log(f"💾 State loaded from {self.state_file}. Balance: ${self.balance:.2f}, Open Trades: {len(self.open_trades)}")
            except Exception as e:
                self.log(f"⚠️ Error loading state: {str(e)}. Starting fresh.")

    def save_state(self):
        try:
            state = {
                'balance': self.balance,
                'open_trades': self.open_trades,
                'closed_trades': self.closed_trades,
                'logs': self.logs,
                'trade_counter': self.trade_counter,
                'max_open_trades_limit': self.max_open_trades_limit,
                'consecutive_losses': self.consecutive_losses,
                'symbol_consecutive_losses': self.symbol_consecutive_losses
            }
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            self.log(f"⚠️ Error saving state: {str(e)}")

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.log("🤖 Quant AI Live Paper Trader STARTED.")
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        self._ticker_thread = threading.Thread(target=self._ticker_loop, daemon=True)
        self._ticker_thread.start()

    def stop(self):
        self.is_running = False
        self.log("🛑 Quant AI Live Paper Trader STOPPED.")
        if self._thread:
            # Join bloklayabilir eger thread icinde uyuyorsa, o yuzden degiskeni False yapmak yeterli
            pass 

    def log(self, msg: str):
        import uuid
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fmsg = f"[{t}] {msg}"
        print(fmsg)
        self.logs.insert(0, {"id": str(uuid.uuid4())[:8], "text": fmsg})
        if len(self.logs) > 100:
            self.logs.pop()
        self.save_state()

    def get_status(self):
        # Aktif kar/zarari hesapla (Guncel fiyati _run_loop'ta guncellemiyoruz ama kabaca pnl gostermek adina)
        # Gercek pnl icin her saniye ticker cekmek lazim. Su an sadece entry bazli state donulur.
        return {
            "status": "Running" if self.is_running else "Stopped",
            "balance": round(self.balance, 2),
            "open_trades_count": len(self.open_trades),
            "open_trades": [self._trade_dict(t) for t in self.open_trades],
            "closed_trades_count": len(self.closed_trades),
            "closed_trades": [self._closed_trade_dict(t) for t in self.closed_trades[-20:]], # Son 20 islem
            "recent_logs": self.logs[:10],
            "scanned_markets_count": len(self.scanned_symbols),
            "max_open_trades_limit": self.max_open_trades_limit
        }

    def _trade_dict(self, t):
        sym = t['symbol']
        cp = self.current_prices.get(sym, t['entry_price'])
        n_in = t['qty'] * t['entry_price']
        n_out = t['qty'] * cp
        
        if t['side'] == 'LONG':
            pnl = (n_out - n_in) - (n_in + n_out) * 0.0002
        else:
            pnl = (n_in - n_out) - (n_in + n_out) * 0.0002
            
        pnl_pct = (pnl / t['margin']) * 100 if t['margin'] > 0 else 0
        
        return {
            "id": t['id'], "pair": sym, "side": t['side'], 
            "entry": round(t['entry_price'], 4), "size": t['qty'],
            "margin": t['margin'], "entry_time": t['entry_time'],
            "current_price": round(cp, 4),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "regime": t.get('regime', 'unknown'),
            "strategy": t.get('strategy', 'unknown'),
            "sl_price": t.get('sl_price', 0),
            "tp_price": t.get('tp_price', 0),
            "pnl_history": t.get('pnl_history', [])
        }

    def _closed_trade_dict(self, t):
        return {
            "id": t['id'], "pair": t['symbol'], "side": t['side'],
            "entry": round(t['entry_price'], 4), "exit": round(t['exit_price'], 4),
            "size": t['qty'], "margin": t['margin'], 
            "entry_time": t['entry_time'], "exit_time": t.get('exit_time', '--'),
            "pnl": round(t['pnl'], 2), 
            "pnl_pct": round((t['pnl']/t['margin'])*100, 2) if t['margin'] > 0 else 0,
            "reason": t.get('reason', 'UNKNOWN'),
            "regime": t.get('regime', 'unknown'),
            "strategy": t.get('strategy', 'unknown'),
            "pnl_history": t.get('pnl_history', [])
        }

    def _ticker_loop(self):
        import ccxt
        exchange = ccxt.binance({'enableRateLimit': True})
        while self.is_running:
            if self.open_trades:
                try:
                    symbols = list(set([t['symbol'] for t in self.open_trades]))
                    for sym, data in tickers.items():
                        if data and 'last' in data and data['last']:
                            new_price = data['last']
                            self.current_prices[sym] = new_price
                            # Update PnL history for open trades of this symbol
                            for t in self.open_trades:
                                if t['symbol'] == sym:
                                    self._record_pnl_snapshot(t, new_price)
                except Exception as e:
                    pass
            time.sleep(3)

    def _record_pnl_snapshot(self, t, current_price):
        """Her 3 saniyede bir PnL kaydı al (Grafik için)"""
        if 'pnl_history' not in t:
            t['pnl_history'] = []
            
        entry = t['entry_price']
        side = t['side']
        qty = t['qty']
        
        n_in = qty * entry
        n_out = qty * current_price
        
        if side == 'LONG':
            pnl = (n_out - n_in) - (n_in + n_out) * 0.0002
        else:
            pnl = (n_in - n_out) - (n_in + n_out) * 0.0002
            
        pnl_pct = (pnl / t['margin']) * 100 if t['margin'] > 0 else 0
        
        snapshot = {
            "t": datetime.now().strftime("%H:%M:%S"),
            "p": round(current_price, 6),
            "pnl": round(pnl, 2),
            "pct": round(pnl_pct, 2)
        }
        
        t['pnl_history'].append(snapshot)
        
        # Hafıza yönetimi: Son 200 kaydı tut (yaklaşık 10 dakikalık detaylı iz)
        if len(t['pnl_history']) > 200:
            t['pnl_history'].pop(0)

    def _run_loop(self):
        fetcher = DataFetcher('binance')
        
        while self.is_running:
            try:
                # Dinamik market tarama (Tüm aktif USDT spot)
                try:
                    self.scanned_symbols = get_all_usdt_pairs()
                    self.log(f"🔄 Dynamic symbols fetched: {len(self.scanned_symbols)} USDT pairs")
                except Exception as e:
                    if "451" in str(e) or "restricted" in str(e).lower():
                        self.log("⚠️ Binance Region Restriction: Access denied from this server's location (HTTP 451). Using fallback symbols.")
                    else:
                        self.log(f"⚠️ Dynamic symbols fetch failed: {str(e)}. Using fallback.")

                self.log(f"🔍 Scanning {len(self.scanned_symbols)} markets...")
                self._scan_and_trade(fetcher)
            except Exception as e:
                self.log(f"❌ Error in scan loop: {str(e)}")
            
            # Gunde cok rate limit yememek icin 15 dakikalik mum stratejisinde 
            # dakikada 1 kez kontrol etmek (60 sn) veya 2 dakikada bir kontrol etmek uygundur.
            # Demo hizli calissin diye 15 saniyede bir taratiyoruz (CCXT limitine takilmamasi umulur)
            for _ in range(15):
                if not self.is_running: break
                time.sleep(1)

    def _scan_and_trade(self, fetcher: DataFetcher):
        # Tum sembolleri gez
        for i, symbol in enumerate(self.scanned_symbols):
            if not self.is_running: break
            
            try:
                # ── v1.5: Hybrid Data Fetching (1H Primary + 15M Secondary) ──
                df_1h = fetcher.fetch_ohlcv(symbol, self.timeframe, limit=200)
                if df_1h.empty or len(df_1h) < 100: 
                    continue
                
                df_15m = fetcher.fetch_ohlcv(symbol, self.secondary_tf, limit=250)
                if df_15m.empty or len(df_15m) < 100:
                    continue

                if i % 30 == 0:
                    self.log(f"🕯️ Scanning {symbol} (Hybrid: {self.timeframe}/{self.secondary_tf})")

                # Note: Adapter automatically handles indicators/features if missing
                df_1h = add_all_indicators(df_1h)
                df_1h = enrich_ohlcv_with_futures(df_1h, symbol, silent=True)
                df_1h = generate_features(df_1h)

                df_15m = add_all_indicators(df_15m)
                df_15m = enrich_ohlcv_with_futures(df_15m, symbol, silent=True)
                df_15m = generate_features(df_15m)
                
                # Sinyal tespiti için 1H dominanttır, ama TP/SL için 15m hassasiyeti de alınabilir.
                last_row = df_1h.iloc[-1]
                last_15m = df_15m.iloc[-1]
                
                c = last_row['close']
                atr = last_row['atr']
                
                # O sembole ait acik islemleri bul
                sym_longs = [t for t in self.open_trades if t['symbol'] == symbol and t['side'] == 'LONG']
                sym_shorts = [t for t in self.open_trades if t['symbol'] == symbol and t['side'] == 'SHORT']
                
                # --- 1. TP ve SL KONTROLU (Real-time Price Engine) ---
                for t in (sym_longs + sym_shorts):
                    side = t['side']
                    entry = t['entry_price']
                    current_sl = t.get('sl_price', 0)
                    target_tp = t.get('tp_price', 0)
                    
                    # KRİTİK: OHLCV yerine anlık ticker fiyatını kullan
                    real_tp = self.current_prices.get(symbol, c)
                    
                    # Eğer AI fiyatları yoksa fallback yüzdelerini hesapla (Eski işlemler için)
                    if current_sl == 0 or target_tp == 0:
                        is_volatile = 'BTC' not in symbol
                        tp_fallback = 0.015 if is_volatile else 0.007
                        sl_fallback = 0.04 if is_volatile else 0.02
                        if side == 'LONG':
                            target_tp = entry * (1 + tp_fallback)
                            current_sl = entry * (1 - sl_fallback)
                        else:
                            target_tp = entry * (1 - tp_fallback)
                            current_sl = entry * (1 + sl_fallback)
                        t['sl_price'] = current_sl
                        t['tp_price'] = target_tp

                    # Profit/Loss Calculation
                    profit_pct = (real_tp - entry) / entry if side == 'LONG' else (entry - real_tp) / entry
                    
                    # 1. Take Profit (Hard Exit)
                    hit_tp = (side == 'LONG' and real_tp >= target_tp) or (side == 'SHORT' and real_tp <= target_tp)
                    if hit_tp:
                        self._close_all([t], symbol, real_tp, f"TP_{side}_REALTIME")
                        continue

                    # 2. Stop Loss (Protective Exit)
                    hit_sl = (side == 'LONG' and real_tp <= current_sl) or (side == 'SHORT' and real_tp >= current_sl)
                    if hit_sl:
                        self._close_all([t], symbol, real_tp, f"SL_{side}_REALTIME")
                        continue

                    # 3. Breakeven logic (Kâr %1'i geçerse stop'u girişe çek)
                    if profit_pct >= 0.008 and t.get('be_active') is not True:
                        t['sl_price'] = entry # Stop can seviyesine çekildi
                        t['be_active'] = True
                        self.log(f"🛡️ BREAKEVEN active for {symbol} {side} @ {entry:.4f}")

                    # 4. Dynamic Trailing Stop logic (Geliştirilmiş - Agresif Kilitleme)
                    if profit_pct >= 0.012:
                        # Sliding Scale: Kâr arttıkça stopu çok daha yakına çek
                        if profit_pct < 0.02:
                            protect_ratio = 0.40  # Kârın %40'ını koru
                        elif profit_pct < 0.05:
                            protect_ratio = 0.65  # Kârın %65'ini koru
                        elif profit_pct < 0.10:
                            protect_ratio = 0.80  # Kârın %80'ini koru
                        else:
                            protect_ratio = 0.90  # %10+ kârda %90 koru
                            
                        trail_distance = (real_tp - entry) * protect_ratio if side == 'LONG' else (entry - real_tp) * protect_ratio
                        new_tsl = entry + trail_distance if side == 'LONG' else entry - trail_distance
                        
                        # Sadece stop'u daha iyiye taşıyorsak güncelle
                        if side == 'LONG' and new_tsl > t.get('sl_price', 0):
                            t['sl_price'] = new_tsl
                        elif side == 'SHORT' and (t.get('sl_price', 999999) == 0 or new_tsl < t['sl_price']):
                            t['sl_price'] = new_tsl

                    # 5. Trend-Aware Smart Exit (Trend yorulmasını bekle)
                    # RSI tepedeyken hemen satma, trendin bozulmasını bekle
                    is_exhausted = False
                    if side == 'LONG' and last_row['rsi'] > 75 and profit_pct > 0.03:
                        # Sinyal: Fiyat EMA9 altına inerse veya MACD histogramı düşerse
                        if c < last_row['ema9'] or last_row['macd_hist'] < df.iloc[-2]['macd_hist']:
                            is_exhausted = True
                            exit_reason = "SMART_EXIT_LONG_RSI_EMA"
                            
                    elif side == 'SHORT' and last_row['rsi'] < 25 and profit_pct > 0.03:
                        if c > last_row['ema9'] or last_row['macd_hist'] > df.iloc[-2]['macd_hist']:
                            is_exhausted = True
                            exit_reason = "SMART_EXIT_SHORT_RSI_EMA"

                    if is_exhausted:
                        self._close_all([t], symbol, c, exit_reason)

                # --- 2. GİRİŞ (SİNYAL) KONTROLU ---
                can_open_new = len(self.open_trades) < self.max_open_trades_limit
                
                if can_open_new and len(sym_longs) == 0 and len(sym_shorts) == 0:
                    # Hybrid Engine'i çağır
                    signal_result = generate_signal(
                        df_1h, 
                        df_secondary=df_15m, 
                        symbol=symbol, 
                        timeframe=self.timeframe,
                        secondary_tf=self.secondary_tf
                    )
                    
                    if should_open_position(signal_result, min_confidence=0.55):
                        # Calculate consecutive loss multiplier
                        sym_loss = self.symbol_consecutive_losses.get(symbol, 0)
                        loss_mult = 1.0
                        if sym_loss == 1: loss_mult = 1.0
                        elif sym_loss == 2: loss_mult = 0.75
                        elif sym_loss >= 3: loss_mult = 0.50
                        
                        # Apply score-based position sizing from engine
                        # (Kelly already considered in adapter, but we add our loss protection)
                        multiplier = calculate_position_amount(
                            signal_result, 
                            self.initial_balance, 
                            risk_per_trade=self.risk_pct, 
                            entry_price=c, 
                            atr=atr
                        ) * c / self.initial_balance / self.risk_pct / self.leverage
                        
                        # Scored scaling
                        final_multiplier = multiplier * loss_mult
                        final_multiplier = max(0.1, min(final_multiplier, 2.0))
                        
                        # Determine TP/SL from signal (v2.0 Scored Logic)
                        tp_price = signal_result.get('tp_price', 0.0)
                        sl_price = signal_result.get('sl_price', 0.0)
                        
                        # Fallback to ATR if strategy didn't provide explicit prices (Breakouts focus)
                        if tp_price == 0 or sl_price == 0:
                            tp_p, sl_p = get_tp_sl_prices(signal_result, c, atr)
                            if tp_price == 0: tp_price = tp_p
                            if sl_price == 0: sl_price = sl_p

                        self._open(symbol, signal_result['signal'], c, final_multiplier, 
                                  tp_price=tp_price, sl_price=sl_price, signal_result=signal_result)
                    else:
                        # Log why it didn't open
                        reason = signal_result.get('reason', 'Unknown reason')
                        conf = signal_result.get('confidence', 0)
                        # Sadece belirli aralıklarla log atalım ki UI dolmasın (veya önemliyse hepsini atalım)
                        if i % 50 == 0 or signal_result['signal'] != 'HOLD':
                           self.log(f"🔍 {symbol} Signal: {signal_result['signal']} | Reason: {reason}")

                # API limit yememek icin az bekle
                time.sleep(0.5)
                
            except Exception as e:
                self.log(f"❌ Error scanning {symbol}: {str(e)}")

    def _open(self, symbol, side, price, multiplier, tp_price=0.0, sl_price=0.0, signal_result=None):
        qty = (self.initial_balance * self.risk_pct * self.leverage * multiplier) / price
        margin = (qty * price) / self.leverage
        
        if qty > 0 and self.balance >= margin:
            self.balance -= margin
            self.trade_counter += 1
            tid = f"LVT-{self.trade_counter}"
            
            t = {
                "id": tid, "symbol": symbol, "side": side, 
                "entry_price": price, "qty": qty, "margin": margin, 
                "entry_time": datetime.now().strftime("%H:%M:%S"),
                "tp_price": tp_price,
                "sl_price": sl_price,
                "strategy": signal_result['strategy'] if signal_result else 'unknown',
                "regime": signal_result['regime'] if signal_result else 'unknown',
                "entry_type": signal_result.get('entry_type', 'none'),
                "soft_score": signal_result.get('soft_score', 0)
            }
            self.open_trades.append(t)
            self.log(f"🟢 OPEN {side} {symbol} @ {price:.2f} (x{multiplier})")
            self.save_state()

    def _close_all(self, trades, symbol, exit_price, reason):
        for t in trades:
            if t not in self.open_trades:
                continue
                
            n_in = t['qty'] * t['entry_price']
            n_out = t['qty'] * exit_price
            
            if t['side'] == 'LONG':
                pnl = (n_out - n_in) - (n_in + n_out) * 0.0002
            else:
                pnl = (n_in - n_out) - (n_in + n_out) * 0.0002
                
            self.balance += t['margin'] + pnl
            
            t['exit_price'] = exit_price
            t['exit_time'] = datetime.now().strftime("%H:%M:%S")
            t['pnl'] = pnl
            t['reason'] = reason
            
            self.closed_trades.append(t)
            self.open_trades.remove(t)
            
            pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
            icon = "✅" if pnl > 0 else "❌"
            self.log(f"{icon} CLOSE {t['side']} {symbol} @ {exit_price:.2f} | PNL: {pnl_str} [{reason}]")
            
            # --- consecutive loss tracking ---
            if pnl < 0:
                self.consecutive_losses += 1
                self.symbol_consecutive_losses[symbol] = self.symbol_consecutive_losses.get(symbol, 0) + 1
            else:
                self.consecutive_losses = 0
                self.symbol_consecutive_losses[symbol] = 0
        if trades:
            self.save_state()

    def close_trade(self, trade_id: str):
        trade_to_close = next((t for t in self.open_trades if t['id'] == trade_id), None)
        if not trade_to_close:
            return {"success": False, "message": "Trade not found."}
            
        sym = trade_to_close['symbol']
        cp = self.current_prices.get(sym, trade_to_close['entry_price'])
        
        # Eğer current_price yoksa Binance API'den anlık çekmeyi deneyelim
        if cp == trade_to_close['entry_price']:
            try:
                fetcher = DataFetcher('binance')
                df = fetcher.fetch_ohlcv(sym, '1m', limit=2)
                if not df.empty:
                    cp = df.iloc[-1]['close']
            except Exception:
                pass

        self._close_all([trade_to_close], sym, cp, "MANUAL_CLOSE")
        return {"success": True, "message": f"Trade {trade_id} closed manually at {cp:.4f}."}

    def update_settings(self, max_open_trades: int):
        if max_open_trades > 0:
            self.max_open_trades_limit = max_open_trades
            self.save_state()
            self.log(f"⚙️ Settings updated: Max open trades set to {max_open_trades}")
            return {"success": True, "message": "Settings updated"}
        return {"success": False, "message": "Invalid value"}
