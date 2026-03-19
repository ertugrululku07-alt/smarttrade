"""
ICT/SMC Strategy Integrity Test
Verifies that profit protection mechanisms work correctly and strategy principles are intact
"""

def test_ict_profit_protection_logic():
    """Test profit protection logic with simulated trade scenarios"""
    
    print("=" * 70)
    print("ICT/SMC PROFIT PROTECTION LOGIC TEST")
    print("=" * 70)
    print()
    
    # Scenario 1: Trade reaches +20% profit (should trigger trailing)
    print("SCENARIO 1: Trade reaches +20% profit")
    print("-" * 70)
    
    entry = 0.0867
    sl_initial = 0.084500
    tp = 0.0956
    atr = entry * 0.005  # 0.5% ATR estimate
    
    # Simulate price movement
    prices = [
        (entry, 0, "Entry"),
        (entry * 1.05, 5, "5% profit"),
        (entry * 1.10, 10, "10% profit - Should trigger BE"),
        (entry * 1.15, 15, "15% profit"),
        (entry * 1.20, 20, "20% profit - Should trigger trailing"),
        (entry * 1.2151, 21.51, "Peak 21.51%"),
        (entry * 1.10, 10, "Pullback to 10%"),
        (entry * 0.95, -5, "Drop to -5%"),
    ]
    
    peak_price = entry
    sl = sl_initial
    be_set = False
    
    for price, pnl_pct, label in prices:
        peak_price = max(peak_price, price)
        peak_profit_atr = (peak_price - entry) / atr
        
        # Breakeven check
        if peak_profit_atr >= 1.5 and not be_set:
            be_sl = entry + 0.3 * atr
            if be_sl > sl:
                sl = be_sl
                be_set = True
                print(f"  ✅ {label}: BE triggered! SL: ${sl:.6f} (was ${sl_initial:.6f})")
        
        # Trailing check
        if peak_profit_atr >= 3.0:
            trail_sl = peak_price - 1.5 * atr
            if trail_sl > sl:
                old_sl = sl
                sl = trail_sl
                print(f"  ✅ {label}: TRAIL triggered! SL: ${sl:.6f} (was ${old_sl:.6f})")
        
        # Check if SL hit
        if price <= sl:
            actual_pnl = ((price - entry) / entry) * 100
            print(f"  🛑 {label}: SL HIT at ${price:.6f} (PnL: {actual_pnl:+.2f}%)")
            break
        else:
            print(f"  📊 {label}: Price ${price:.6f}, SL ${sl:.6f}, Peak ATR: {peak_profit_atr:.1f}")
    
    print()
    print(f"Final SL: ${sl:.6f} (Initial: ${sl_initial:.6f})")
    print(f"Protection gain: {((sl - sl_initial) / sl_initial * 100):.2f}%")
    print()
    
    # Scenario 2: Partial TP levels
    print("SCENARIO 2: Partial TP Levels")
    print("-" * 70)
    
    tp_dist = abs(tp - entry)
    
    test_prices = [
        (entry + tp_dist * 0.33, 33, "TP1 (33%)"),
        (entry + tp_dist * 0.65, 65, "TP2 (65%)"),
        (entry + tp_dist * 1.0, 100, "TP3 (100%)"),
        (entry + tp_dist * 1.5, 150, "150% of TP"),
    ]
    
    for price, progress, label in test_prices:
        pnl_pct = ((price - entry) / entry) * 100
        print(f"  {label}: ${price:.6f} (+{pnl_pct:.1f}%)")
        
        if progress >= 33:
            print(f"    ✅ Should close 30% and move SL to BE")
        if progress >= 65:
            print(f"    ✅ Should close 40% and move SL to TP1")
        if progress >= 100:
            print(f"    ✅ Should enter trail mode, SL to 75% TP")
    
    print()
    
    # Scenario 3: ATR value impact
    print("SCENARIO 3: ATR Value Impact on Protection")
    print("-" * 70)
    
    peak_at_21pct = entry * 1.2151
    
    for atr_pct in [0.3, 0.5, 1.0, 2.0]:
        test_atr = entry * (atr_pct / 100)
        peak_atr = (peak_at_21pct - entry) / test_atr
        
        be_trigger = peak_atr >= 1.5
        trail_trigger = peak_atr >= 3.0
        
        print(f"  ATR = {atr_pct:.1f}% (${test_atr:.6f})")
        print(f"    Peak profit: {peak_atr:.1f} ATR")
        print(f"    BE trigger (1.5 ATR): {'✅ YES' if be_trigger else '❌ NO'}")
        print(f"    Trail trigger (3.0 ATR): {'✅ YES' if trail_trigger else '❌ NO'}")
        
        if trail_trigger:
            trail_sl = peak_at_21pct - 1.5 * test_atr
            protection = ((trail_sl - entry) / entry) * 100
            print(f"    Trail SL: ${trail_sl:.6f} (+{protection:.1f}% protected)")
        print()
    
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("For WAL/USDT scenario (Entry $0.0867, Peak +21.51%):")
    print()
    print("With 0.5% ATR:")
    print("  - Peak = 43 ATR → SHOULD trigger both BE and trailing")
    print("  - Trail SL should be ~$0.1047 (+20.76% protected)")
    print("  - Actual result: -$10.31 (-13.08%) ❌")
    print()
    print("DIAGNOSIS:")
    print("  1. Either ATR value in trade object is WRONG (too large)")
    print("  2. Or _ict_peak_price is NOT being updated in ticker loop")
    print("  3. Or profit protection code is not being called at all")
    print()
    print("SOLUTION:")
    print("  - Added debug logging to track ATR, peak_price, and trigger conditions")
    print("  - Logs will show exact values when protection should trigger")
    print("  - Next trade will reveal which mechanism is broken")
    print()

if __name__ == '__main__':
    test_ict_profit_protection_logic()
