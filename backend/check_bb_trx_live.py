"""BB MR TRX Live Performance Check"""
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

d = json.load(open('live_trader_state.json', 'r'))

# BB MR TRX trades
bb_trx_open = [t for t in d.get('open_trades', []) 
               if 'TRX' in t.get('symbol', '') and 'bb_mr' in t.get('strategy', '')]
bb_trx_closed = [t for t in d.get('closed_trades', []) 
                 if 'TRX' in t.get('symbol', '') and 'bb_mr' in t.get('strategy', '')]

print("=" * 70)
print("  BB MR TRX/USDT — Live Performance Report")
print("=" * 70)
print(f"\n  Fix sonrası durum:")
print(f"    ATR threshold: 0.30% → 0.15%")
print(f"    BE/Trail: margin PnL% → ATR-bazlı")
print(f"    Sinyal adayı (72h): 5 → 24")

print(f"\n  Açık Trades: {len(bb_trx_open)}")
for t in bb_trx_open:
    side = t.get('side', '')
    entry = t.get('entry_price', 0)
    margin = t.get('margin', 0)
    qty = t.get('qty', 0)
    sl = t.get('sl_price', 0)
    tp = t.get('tp_price', 0)
    entry_time = t.get('entry_time', '?')
    max_pnl = t.get('max_pnl_pct', 0)
    atr = t.get('atr', 0)
    quality = t.get('signal_result', {}).get('quality_score', 0)
    
    print(f"\n  {side} TRX/USDT")
    print(f"    Entry: ${entry:.6f} @ {entry_time}")
    print(f"    SL: ${sl:.6f} | TP: ${tp:.6f}")
    print(f"    Margin: ${margin:.2f} | Qty: {qty:.2f}")
    print(f"    ATR: ${atr:.6f} | Quality: {quality}")
    print(f"    Max PnL: {max_pnl:.1f}%")
    
    # BE/Trail thresholds
    if atr > 0:
        be_threshold = 1.5 * atr
        trail_threshold = 3.0 * atr
        if side == 'LONG':
            be_price = entry + be_threshold
            trail_price = entry + trail_threshold
        else:
            be_price = entry - be_threshold
            trail_price = entry - trail_threshold
        print(f"    BE trigger: ${be_price:.6f} ({be_threshold/entry*100:.2f}%)")
        print(f"    Trail trigger: ${trail_price:.6f} ({trail_threshold/entry*100:.2f}%)")

print(f"\n  Kapalı Trades: {len(bb_trx_closed)}")
if bb_trx_closed:
    total_pnl = sum(t.get('pnl', 0) for t in bb_trx_closed)
    wins = sum(1 for t in bb_trx_closed if t.get('pnl', 0) > 0)
    losses = sum(1 for t in bb_trx_closed if t.get('pnl', 0) < 0)
    print(f"    Win/Loss: {wins}/{losses}")
    print(f"    Total PnL: ${total_pnl:.2f}")
    
    for t in bb_trx_closed:
        side = t.get('side', '')
        entry = t.get('entry_price', 0)
        exit_p = t.get('exit_price', 0)
        pnl = t.get('pnl', 0)
        reason = t.get('close_reason', t.get('reason', '?'))
        entry_time = t.get('entry_time', '?')
        exit_time = t.get('exit_time', '?')
        max_pnl = t.get('max_pnl_pct', 0)
        atr = t.get('atr', 0)
        
        print(f"\n  {side} ${entry:.6f} → ${exit_p:.6f}")
        print(f"    PnL: ${pnl:+.2f} | Max: {max_pnl:.1f}%")
        print(f"    Reason: {reason}")
        print(f"    Time: {entry_time} → {exit_time}")
        if atr > 0:
            move_atr = abs(exit_p - entry) / atr
            print(f"    Move: {move_atr:.1f} ATR")

# Son BB MR scan log'larını kontrol et
print(f"\n{'='*70}")
print(f"  Son BB MR Scan Aktivitesi")
print(f"{'='*70}")
print(f"  (live_trader.log'dan son BB MR scan'leri kontrol edin)")
print(f"  Eğer TRX sinyali varsa ama trade açılmadıysa:")
print(f"    - HTF filter (4h EMA trend) reject etmiş olabilir")
print(f"    - Quality score threshold'u geçememiş olabilir")
print(f"    - Diğer coinler daha yüksek quality ile öncelik almış olabilir")
