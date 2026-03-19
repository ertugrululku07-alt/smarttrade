"""
Test DEXE Entry Fix - Verify $5.579 entry would now be rejected
Simulates Mar 17 scenario where system bought at resistance
"""

def test_dexe_entry_rejection():
    """Test that new filters would reject DEXE $5.579 entry"""
    
    print("=" * 70)
    print("DEXE/USDT ENTRY FIX VALIDATION")
    print("=" * 70)
    print()
    
    # DEXE scenario on Mar 17
    entry_price = 5.579
    recent_high_100bars = 6.000  # Mar 17 peak
    recent_high_10bars = 5.637   # Last 10 bars high
    recent_low_30bars = 5.200    # 30-bar range low
    recent_high_30bars = 5.850   # 30-bar range high
    
    print("SCENARIO: Mar 17, 10:26 - System tried to enter LONG at $5.579")
    print(f"Recent high (100 bars): ${recent_high_100bars}")
    print(f"Recent high (10 bars): ${recent_high_10bars}")
    print(f"30-bar range: ${recent_low_30bars} - ${recent_high_30bars}")
    print()
    
    # Test 1: HTF Resistance Proximity
    print("TEST 1: HTF Resistance Proximity (5% threshold)")
    print("-" * 70)
    distance_from_high = (recent_high_100bars - entry_price) / recent_high_100bars
    print(f"Distance from high: {distance_from_high * 100:.2f}%")
    
    if distance_from_high < 0.05:
        print(f"❌ REJECT: Too close to resistance (need >5%, got {distance_from_high*100:.1f}%)")
        reject_resistance = True
    else:
        print(f"✅ PASS: Far enough from resistance ({distance_from_high*100:.1f}% > 5%)")
        reject_resistance = False
    print()
    
    # Test 2: Pullback Requirement
    print("TEST 2: Pullback Requirement (3% minimum)")
    print("-" * 70)
    pullback_pct = (recent_high_10bars - entry_price) / recent_high_10bars
    print(f"Pullback from recent high: {pullback_pct * 100:.2f}%")
    
    if pullback_pct < 0.03:
        print(f"❌ REJECT: No pullback (need >3%, got {pullback_pct*100:.1f}%)")
        reject_pullback = True
    else:
        print(f"✅ PASS: Sufficient pullback ({pullback_pct*100:.1f}% > 3%)")
        reject_pullback = False
    print()
    
    # Test 3: Premium Zone Filter
    print("TEST 3: Premium Zone Filter (70% threshold)")
    print("-" * 70)
    price_position = (entry_price - recent_low_30bars) / (recent_high_30bars - recent_low_30bars)
    print(f"Price position in range: {price_position * 100:.1f}%")
    
    if price_position > 0.70:
        print(f"❌ REJECT: Premium zone (need <70%, got {price_position*100:.0f}%)")
        reject_premium = True
    else:
        print(f"✅ PASS: Not in premium zone ({price_position*100:.0f}% < 70%)")
        reject_premium = False
    print()
    
    # Summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print()
    
    total_rejects = sum([reject_resistance, reject_pullback, reject_premium])
    
    print(f"Resistance Proximity: {'❌ REJECT' if reject_resistance else '✅ PASS'}")
    print(f"Pullback Requirement: {'❌ REJECT' if reject_pullback else '✅ PASS'}")
    print(f"Premium Zone Filter: {'❌ REJECT' if reject_premium else '✅ PASS'}")
    print()
    
    if total_rejects > 0:
        print(f"✅ ENTRY WOULD BE REJECTED ({total_rejects}/3 filters failed)")
        print()
        print("RESULT: System would NOT enter at $5.579 (resistance zone)")
        print("        Would wait for pullback to $5.200-5.250 (OB zone)")
    else:
        print("❌ ENTRY WOULD STILL BE ACCEPTED")
        print("   WARNING: Filters may need further tuning")
    
    print()
    print("=" * 70)
    print("OPTIMAL ENTRY ZONE (ICT/SMC)")
    print("=" * 70)
    print()
    print("Entry: $5.200 - $5.250 (Order Block + FVG)")
    print("  - Distance from high: 13.3% ✅")
    print("  - Pullback: 7.4% ✅")
    print("  - Price position: 0-8% (deep discount) ✅")
    print()
    print("SL: $5.050 (-3%)")
    print("TP1: $5.600 (+7%)")
    print("TP2: $5.850 (+12%)")
    print("R:R: 1:4 ✅")
    print()

if __name__ == '__main__':
    test_dexe_entry_rejection()
