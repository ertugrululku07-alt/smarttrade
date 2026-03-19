import pandas as pd
import numpy as np
import datetime
import os

# ════════════════════════════════════════════════════════════
# HybridTradingEngineV31 + TradeLogger: DEPRECATED
# Sadece PositionManagerV3 aktif kullanılıyor.
# BB MR ve ICT stratejileri kendi exit logic'lerini kullanır.
# ════════════════════════════════════════════════════════════

class TradeLogger:
    def __init__(self, path="trade_logs.csv"):
        self.path = path
        self.active_trades = {}
        self._reload_active_from_csv()

    def _reload_active_from_csv(self):
        """Recover active trades from CSV after restart."""
        if not os.path.exists(self.path):
            return
        try:
            df = pd.read_csv(self.path)
            # Reconstruct active trades from OPEN events that don't have a matching EXIT
            opens = df[df['result'] == 'OPEN']
            exits = df[df['result'].isin(['WIN', 'LOSS'])]
            exit_ids = set(exits['trade_id'].unique()) if 'trade_id' in exits.columns else set()
            
            for _, row in opens.iterrows():
                tid = row.get('trade_id')
                if tid and tid not in exit_ids:
                    self.active_trades[tid] = row.to_dict()
        except:
            pass

    def log_entry(self, trade, context):
        trade_id = f"{context.get('coin', 'COIN')}_{trade['side']}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        record = {
            'trade_id': trade_id,
            'event': 'ENTRY',
            **trade, **context,
            'open_time': datetime.datetime.now().isoformat(),
            'result': 'OPEN',
            'exit_price': None,
            'exit_reason': None,
            'actual_rr': None,
            'pnl_pct': None
        }
        self.active_trades[trade_id] = record
        self._append_csv(record)
        return trade_id

    def log_exit(self, trade_id, exit_price, exit_reason):
        if trade_id not in self.active_trades:
            return False
        
        trade = self.active_trades[trade_id]
        entry = trade.get('entry', trade.get('entry_price', 0))
        stop = trade.get('stop', trade.get('sl_price', entry))
        side = trade['side']
        risk = max(abs(entry - stop), 0.0001)
        
        pnl = (exit_price - entry) if side == 'LONG' else (entry - exit_price)
        actual_rr = round(pnl / risk, 2) if risk > 0 else 0
        
        exit_record = {
            'trade_id': trade_id,
            'event': 'EXIT',
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'actual_rr': actual_rr,
            'pnl_pct': round((pnl / entry) * 100, 3) if entry > 0 else 0,
            'result': 'WIN' if pnl > 0 else 'LOSS',
            'close_time': datetime.datetime.now().isoformat()
        }
        
        self._append_csv(exit_record)
        if trade_id in self.active_trades:
            del self.active_trades[trade_id]
        return exit_record['result'] == 'WIN'

    def get_stats(self):
        if not os.path.exists(self.path): return {'total': 0}
        try:
            df = pd.read_csv(self.path)
            closed = df[df['result'].isin(['WIN', 'LOSS'])]
            if closed.empty: return {'total': 0}
            
            wins = closed[closed['result'] == 'WIN']
            return {
                'total': len(closed),
                'win_rate': round(len(wins) / len(closed) * 100, 1),
                'avg_rr': round(closed['actual_rr'].mean(), 2),
                'total_pnl_pct': round(closed['pnl_pct'].sum(), 2)
            }
        except:
            return {'total': 0}

    def _append_csv(self, record):
        try:
            pd.DataFrame([record]).to_csv(self.path, mode='a', header=not os.path.exists(self.path), index=False)
        except:
            pass

class PositionManagerV3:
    """
    Kâr Koruma v3.3 — Balanced R:R System

    v3.2 Sorunu: PeakDD küçük peak'lerde erken çıkış → avg win $3, avg loss $14
    = 1:4.7 R:R → kasa eritiyor.

    v3.3 Çözümü:
    1. MaxLoss sıkılaştı: -%3.5 (kayıpları küçült)
    2. PeakDD sadece büyük peak'ler: peak > +6% ve %50 drawdown
    3. Trailing geniş başlar, giderek sıkılaşır (kazançları büyüt)
    4. Breakeven erken: +1.5% ROI

    Stage 0: İlk SL aktif
    Stage 1: 0.3R VEYA +1.5% ROI → Breakeven
    Stage 2: 0.8R VEYA +3% ROI → %20 partial + geniş trailing (1.5 ATR)
    Stage 3: 1.5R VEYA +5% ROI → %20 partial + normal trailing (1.0 ATR)
    Stage 4: 2.5R VEYA +8% ROI → %20 partial + sıkı trailing (0.5 ATR)

    Hedef: avg win ~$5-7, avg loss ~$4-5 → R:R ~1.2:1
    """

    # ROI% eşikleri (leverage dahil) — v3.4 ICT Optimized
    ROI_STAGE1 = 5.0     # +5% ROI → breakeven
    ROI_STAGE2 = 10.0    # +10% ROI → partial + wide trail
    ROI_STAGE3 = 15.0    # +15% ROI → partial + normal trail
    ROI_STAGE4 = 20.0    # +20% ROI → partial + tight trail
    MAX_LOSS_ROI = -2.0   # -%2.0 ROI → force exit (sıkı kayıp kontrolü)
    PEAK_DD_MIN = 6.0     # PeakDD peak > +6% olunca aktif
    PEAK_DRAWDOWN = 0.50  # Kâr zirvesinden %50 geri çekilme → exit

    def __init__(self, trade_data, atr, leverage=10):
        self.entry = trade_data.get('entry', trade_data.get('entry_price'))
        self.stop = trade_data.get('stop', trade_data.get('sl_price', self.entry))
        self.side = trade_data['side']
        self.leverage = leverage

        if self.entry is None:
            raise ValueError(f"PositionManagerV3: No entry price found in trade_data keys: {list(trade_data.keys())}")

        self.risk = max(abs(self.entry - self.stop), 0.000001)
        self.atr = atr
        self.stage = 0
        self.peak_price = self.entry  # LONG: en yüksek, SHORT: en düşük
        self.trailing_stop = None
        self.peak_roi = 0.0

    def _roi_pct(self, current_price):
        """Kaldıraçlı ROI% hesapla."""
        if self.side == 'LONG':
            return ((current_price - self.entry) / self.entry) * self.leverage * 100
        else:
            return ((self.entry - current_price) / self.entry) * self.leverage * 100

    def _trail_distance(self):
        """Stage'e göre trailing — geniş başla, kârla sıkılaştır."""
        if self.stage >= 4:
            return 0.5 * self.atr   # sıkı — kârı kilitle
        elif self.stage >= 3:
            return 1.0 * self.atr   # normal
        elif self.stage >= 2:
            return 1.5 * self.atr   # geniş — nefes aldır
        return 2.0 * self.atr       # fallback

    def _move_stop(self, new_stop):
        """Stop'u sadece kâr yönüne taşı."""
        if self.side == 'LONG':
            self.trailing_stop = max(self.trailing_stop or 0, new_stop)
        else:
            self.trailing_stop = min(self.trailing_stop or 999999, new_stop)

    def _hit_stage(self, r_mult, roi_pct, r_threshold, roi_threshold):
        """Hybrid check: R-mult VEYA ROI% eşiği aşıldı mı?"""
        return r_mult >= r_threshold or roi_pct >= roi_threshold

    def update(self, current_price):
        if self.side == 'LONG':
            self.peak_price = max(self.peak_price, current_price)
            r_mult = (current_price - self.entry) / self.risk
        else:
            self.peak_price = min(self.peak_price, current_price)
            r_mult = (self.entry - current_price) / self.risk

        roi_pct = self._roi_pct(current_price)
        self.peak_roi = max(self.peak_roi, roi_pct)

        # ── MAX LOSS: -%3.5 ROI → force exit ──
        if roi_pct <= self.MAX_LOSS_ROI:
            return {'action': 'EXIT', 'reason': f'MaxLoss {roi_pct:.1f}%'}

        # ── PEAK DRAWDOWN: sadece büyük peak'ler (>6%) için ──
        if self.peak_roi >= self.PEAK_DD_MIN and roi_pct > 0:
            drawdown_from_peak = (self.peak_roi - roi_pct) / self.peak_roi
            if drawdown_from_peak >= self.PEAK_DRAWDOWN:
                return {'action': 'EXIT', 'reason': f'PeakDD {self.peak_roi:.1f}%→{roi_pct:.1f}%'}

        # ── Stage 1: Breakeven ──
        if self._hit_stage(r_mult, roi_pct, 0.3, self.ROI_STAGE1) and self.stage == 0:
            self.stage = 1
            self.stop = self.entry
            return {'action': 'UPDATE_STOP', 'stop': self.stop,
                    'reason': f'BE R={r_mult:.2f} ROI={roi_pct:.1f}%'}

        # ── Stage 2: Partial %20 + Geniş Trailing ──
        if self._hit_stage(r_mult, roi_pct, 0.8, self.ROI_STAGE2) and self.stage <= 1:
            self.stage = 2
            trail_dist = self._trail_distance()
            if self.side == 'LONG':
                self.trailing_stop = max(self.entry, self.peak_price - trail_dist)
            else:
                self.trailing_stop = min(self.entry, self.peak_price + trail_dist)
            return {'action': 'PARTIAL', 'amount': 0.20, 'stop': self.trailing_stop,
                    'reason': f'TP1 R={r_mult:.2f} ROI={roi_pct:.1f}%'}

        # ── Stage 3: Partial %20 + Normal Trailing ──
        if self._hit_stage(r_mult, roi_pct, 1.5, self.ROI_STAGE3) and self.stage == 2:
            self.stage = 3
            trail_dist = self._trail_distance()
            new_trail = self.peak_price - trail_dist if self.side == 'LONG' else self.peak_price + trail_dist
            self._move_stop(new_trail)
            return {'action': 'PARTIAL', 'amount': 0.20, 'stop': self.trailing_stop,
                    'reason': f'TP2 R={r_mult:.2f} ROI={roi_pct:.1f}%'}

        # ── Stage 4: Partial %20 + Sıkı Trailing ──
        if self._hit_stage(r_mult, roi_pct, 2.5, self.ROI_STAGE4) and self.stage == 3:
            self.stage = 4
            trail_dist = self._trail_distance()
            new_trail = self.peak_price - trail_dist if self.side == 'LONG' else self.peak_price + trail_dist
            self._move_stop(new_trail)
            return {'action': 'PARTIAL', 'amount': 0.20, 'stop': self.trailing_stop,
                    'reason': f'TP3 R={r_mult:.2f} ROI={roi_pct:.1f}%'}

        # ── Trailing Update (stage >= 2) ──
        if self.stage >= 2 and self.trailing_stop is not None:
            trail_dist = self._trail_distance()
            new_trail = self.peak_price - trail_dist if self.side == 'LONG' else self.peak_price + trail_dist
            old_trail = self.trailing_stop
            self._move_stop(new_trail)
            if self.trailing_stop != old_trail:
                return {'action': 'UPDATE_STOP', 'stop': self.trailing_stop}

        # ── Exit Check ──
        active_stop = self.trailing_stop if self.trailing_stop else self.stop
        if (self.side == 'LONG' and current_price <= active_stop) or \
           (self.side == 'SHORT' and current_price >= active_stop):
            return {'action': 'EXIT', 'reason': 'Stop Hit'}

        return {'action': 'HOLD', 'stop': active_stop}

from ai.mtf_analyzer import MTFAnalyzer

class HybridTradingEngineV31:
    def __init__(self, config=None):
        self.config = config or {
            'max_daily_loss_pct': 1.0,
            'max_consecutive_losses': 999,
            'risk_per_trade': 0.01,
            'min_rr': 2.0,
            'score_threshold': 7,
            'max_same_side_positions': 2,
            'total_account_risk_limit': 0.04
        }
        self._validate_config()
        self.consecutive_losses = 0
        self.daily_start_balance = None
        self.last_reset_date = datetime.date.today()
        self.logger = TradeLogger()
        self.mtf_analyzer = MTFAnalyzer()

    def _validate_config(self):
        c = self.config
        assert 0 < c['max_daily_loss_pct'] <= 1.0, "Daily loss limit hatası"
        assert 1 <= c['max_consecutive_losses'] <= 1000, "Consecutive loss hatası"

    def safe_divide(self, num, den, default=0):
        return num / den if den and den != 0 else default

    def get_current_atr(self, df, period=14):
        tr = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
        atr_val = tr.rolling(period).mean().iloc[-1]
        return atr_val if not pd.isna(atr_val) and atr_val > 0 else (df['high'] - df['low']).iloc[-period:].mean()

    def find_swings(self, df, direction='HIGH', strength=3):
        swings = []
        col = 'high' if direction == 'HIGH' else 'low'
        for i in range(strength, len(df) - strength):
            val = df[col].iloc[i]
            if direction == 'HIGH':
                if all(val > df[col].iloc[i-j] for j in range(1, strength+1)) and all(val > df[col].iloc[i+j] for j in range(1, strength+1)):
                    swings.append({'price': val, 'index': i})
            else:
                if all(val < df[col].iloc[i-j] for j in range(1, strength+1)) and all(val < df[col].iloc[i+j] for j in range(1, strength+1)):
                    swings.append({'price': val, 'index': i})
        return swings

    def check_liquidity_sweep(self, df_15m, direction='LONG'):
        level = df_15m['low'].iloc[-21:-1].min() if direction == 'LONG' else df_15m['high'].iloc[-21:-1].max()
        for lookback in range(1, 4):
            candle = df_15m.iloc[-lookback]
            swept = candle['low'] < level if direction == 'LONG' else candle['high'] > level
            closed_ok = candle['close'] > level if direction == 'LONG' else candle['close'] < level
            wick = (candle['close'] - candle['low']) if direction == 'LONG' else (candle['high'] - candle['close'])
            body = max(abs(candle['close'] - candle['open']), 0.0001)
            if swept and closed_ok and wick > body * 1.5:
                return {'is_swept': True, 'level': level, 'bars_ago': lookback}
        return {'is_swept': False, 'level': None, 'bars_ago': 0}

    def check_mss(self, df_15m, atr, direction='LONG', sweep_data=None):
        swings = self.find_swings(df_15m, 'HIGH' if direction == 'LONG' else 'LOW')
        if not swings: return {'detected': False}
        last_s = swings[-1]
        last_c = df_15m.iloc[-1]
        broken = last_c['close'] > last_s['price'] if direction == 'LONG' else last_c['close'] < last_s['price']
        vol_ok = last_c['volume'] > (df_15m['volume'].iloc[-20:].mean() * 1.2)
        sweep_sync = sweep_data['bars_ago'] < 10 if sweep_data and sweep_data['is_swept'] else True
        detected = broken and vol_ok and (len(df_15m)-1-last_s['index'] < 15) and sweep_sync
        return {'detected': detected, 'is_god': (abs(last_c['high']-last_c['low']) > 2.5*atr), 'level': last_s['price']}

    def calculate_fvg_entry(self, df_15m, direction='LONG'):
        start, end = len(df_15m) - 4, max(len(df_15m) - 14, 0)
        for i in range(start, end, -1):
            if i + 2 >= len(df_15m): continue
            c1, c3 = df_15m.iloc[i], df_15m.iloc[i+2]
            if direction == 'LONG' and c3['low'] > c1['high'] and df_15m['close'].iloc[-1] > c3['low']:
                return {'entry': (c3['low'] + c1['high']) / 2, 'type': 'FVG'}
            if direction == 'SHORT' and c3['high'] < c1['low'] and df_15m['close'].iloc[-1] < c3['high']:
                return {'entry': (c3['high'] + c1['low']) / 2, 'type': 'FVG'}
        swings = self.find_swings(df_15m, 'LOW' if direction == 'LONG' else 'HIGH')
        if swings: return {'entry': swings[-1]['price'] + (self.get_current_atr(df_15m)*0.2*(1 if direction=='LONG' else -1)), 'type': 'SWING_PB'}
        return None

    def execute_decision_cycle(self, data, open_positions):
        """
        data expected keys:
        - balance
        - df_15m (DataFrame)
        - df_4h (DataFrame)
        - btc_4h (DataFrame)
        - coin (String)
        """
        # 1. Risk Guard
        today = datetime.date.today()
        if self.last_reset_date != today or self.daily_start_balance is None:
            self.daily_start_balance = data['balance']
            self.last_reset_date = today
            
        if self.daily_start_balance and (data['balance'] - self.daily_start_balance)/self.daily_start_balance < -self.config['max_daily_loss_pct']:
            return {'status': 'HALT', 'reason': 'Daily Limit'}
        if self.consecutive_losses >= self.config['max_consecutive_losses']:
            return {'status': 'HALT', 'reason': 'Consecutive Loss'}

        # 2. Nuke Filter
        atr = self.get_current_atr(data['df_15m'])
        if (data['df_15m']['high'].iloc[-1] - data['df_15m']['low'].iloc[-1]) > 3.0 * atr:
            return {'status': 'WAIT', 'reason': 'NUKE'}

        # 3. Sinyal Tarama
        # Phase 1: MTF Context (4h)
        htf_ctx = self.mtf_analyzer.analyze_htf(data['df_4h'], data.get('coin', 'UNKNOWN'))
        
        results = []
        for side in ['LONG', 'SHORT']:
            same_dir = [p for p in open_positions if p.get('side') == side]
            if len(same_dir) >= self.config['max_same_side_positions']: continue
            
            # MTF Adjustment
            mtf_adj = self.mtf_analyzer.get_signal_adjustment(htf_ctx, side)
            if not mtf_adj.allowed: continue

            sweep = self.check_liquidity_sweep(data['df_15m'], side)
            mss = self.check_mss(data['df_15m'], atr, side, sweep)
            
            # Base Skoru
            base_score = (3 if sweep['is_swept'] else 0) + (4 if mss.get('detected') and not mss.get('is_god') else 2 if mss.get('detected') else 0)
            
            # MTF Modifikasyonu (Phase 1)
            final_score = base_score + mtf_adj.score_modifier + 3 # +3 baseline compatibility
            
            if final_score >= self.config['score_threshold']:
                entry_data = self.calculate_fvg_entry(data['df_15m'], side)
                if entry_data:
                    stop = (sweep['level'] - atr*0.5 if side=='LONG' else sweep['level'] + atr*0.5) if sweep['level'] else (entry_data['entry'] + atr*2*(-1 if side=='LONG' else 1))
                    
                    # MTF SL/TP Adjustments
                    risk = max(abs(entry_data['entry'] - stop), 0.0001)
                    target = data['df_4h']['high' if side=='LONG' else 'low'].iloc[-50:].max() if side=='LONG' else data['df_4h']['low'].iloc[-50:].min()
                    reward = (target - entry_data['entry']) if side=='LONG' else (entry_data['entry'] - target)
                    
                    # Apply MTF Multipliers
                    risk *= mtf_adj.sl_multiplier
                    reward *= mtf_adj.tp_multiplier
                    
                    # Recalculate Stop/TP
                    if side == 'LONG':
                        stop = entry_data['entry'] - risk
                    else:
                        stop = entry_data['entry'] + risk

                    if (reward / risk) >= self.config['min_rr']:
                        results.append({
                            **entry_data, 
                            'side': side, 
                            'score': final_score, 
                            'stop': stop, 
                            'rr': round(reward/risk, 2), 
                            'risk_dist': risk,
                            'atr': atr,
                            'mtf_reason': mtf_adj.reason
                        })

        if not results: return {'status': 'WAIT', 'reason': 'No Setup'}
        best = max(results, key=lambda x: x['score'])
        
        # Risk management: scaling down with losses
        loss_scaling = {0: 1.0, 1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4}.get(self.consecutive_losses, 0.2)
        qty_usd = data['balance'] * self.config['risk_per_trade'] * loss_scaling
        qty = self.safe_divide(qty_usd, best['risk_dist'])
        
        trade_packet = {
            **best, 
            'status': 'TRADE', 
            'quantity': round(qty, 4),
            'atr': atr,
            'signal': best['side']
        }
        logger_id = self.logger.log_entry(trade_packet, {'coin': data.get('coin', 'COIN')})
        trade_packet['logger_id'] = logger_id
        return trade_packet

    def record_trade_result(self, is_win):
        if is_win: 
            self.consecutive_losses = max(0, self.consecutive_losses - 1)
        else: 
            self.consecutive_losses += 1
