"""
Test ICT/SMC Profit Protection Mechanisms
Simulates WAL/USDT scenario: Max +21.51% → Final -13.08%
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_ict_breakeven_trailing():
    """Test ATR-based breakeven and trailing logic"""
    
    # WAL/USDT trade data
    entry = 0.0867
    tp = 0.0956
    sl = 0.084500
    side = 'LONG'
    
    # Simulate price movement to +21.51% peak
    peak_price = entry * 1.2151  # 0.1054
    
    # ATR estimation (typical 0.5% for crypto)
    atr_val = entry * 0.005  # 0.000434
    
    print("=" * 60)
    print("ICT/SMC Profit Protection Test")
    print("=" * 60)
    print(f"Entry: ${entry:.6f}")
    print(f"TP: ${tp:.6f} (+{((tp/entry - 1) * 100):.2f}%)")
    print(f"SL: ${sl:.6f} (-{((1 - sl/entry) * 100):.2f}%)")
    print(f"Peak: ${peak_price:.6f} (+21.51%)")
    print(f"ATR: ${atr_val:.6f} ({(atr_val/entry * 100):.2f}%)")
    print()
    
    # Calculate profit in ATR terms
    peak_profit_atr = (peak_price - entry) / atr_val
    print(f"Peak Profit in ATR: {peak_profit_atr:.2f} ATR")
    print()
    
    # Test Breakeven trigger (1.5 ATR)
    print("--- BREAKEVEN TEST (1.5 ATR threshold) ---")
    if peak_profit_atr >= 1.5:
        be_sl = entry + 0.3 * atr_val
        print(f"✅ Breakeven SHOULD trigger at {peak_profit_atr:.2f} ATR")
        print(f"   New SL: ${be_sl:.6f} (entry + 0.3 ATR)")
        print(f"   Protection: +{((be_sl/entry - 1) * 100):.2f}%")
    else:
        print(f"❌ Breakeven NOT triggered ({peak_profit_atr:.2f} < 1.5 ATR)")
    print()
    
    # Test Trailing trigger (3.0 ATR)
    print("--- TRAILING TEST (3.0 ATR threshold) ---")
    if peak_profit_atr >= 3.0:
        trail_sl = peak_price - 1.5 * atr_val
        print(f"✅ Trailing SHOULD trigger at {peak_profit_atr:.2f} ATR")
        print(f"   New SL: ${trail_sl:.6f} (peak - 1.5 ATR)")
        print(f"   Protection: +{((trail_sl/entry - 1) * 100):.2f}%")
    else:
        print(f"❌ Trailing NOT triggered ({peak_profit_atr:.2f} < 3.0 ATR)")
    print()
    
    # Test Partial TP levels
    tp_dist = abs(tp - entry)
    progress = (peak_price - entry) / tp_dist
    
    print("--- PARTIAL TP TEST ---")
    print(f"Progress to TP: {progress * 100:.1f}%")
    
    if progress >= 0.33:
        print(f"✅ TP1 (33%) SHOULD trigger - Close 30%, SL→BE")
    if progress >= 0.65:
        print(f"✅ TP2 (65%) SHOULD trigger - Close 40%, SL→TP1")
    if progress >= 1.0:
        print(f"✅ TP3 (100%) SHOULD trigger - Trail mode, SL→75% TP")
    print()
    
    # Root cause analysis
    print("=" * 60)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 60)
    print("Possible reasons profit protection failed:")
    print()
    print("1. ❌ ATR value too large (wrong calculation)")
    print(f"   If ATR > ${(peak_price - entry) / 1.5:.6f}, breakeven won't trigger")
    print()
    print("2. ❌ _ict_peak_price not updating in ticker loop")
    print("   Peak tracking might be broken")
    print()
    print("3. ❌ _check_ict_exit not being called frequently enough")
    print("   3-second ticker interval might miss peaks")
    print()
    print("4. ❌ Partial TP logic has bugs (qty calculation)")
    print("   Might close full position instead of partial")
    print()
    print("5. ❌ SL update logic has condition bugs")
    print("   'if be_sl > sl' might fail if sl already moved")
    print()

if __name__ == '__main__':
    test_ict_breakeven_trailing()
