#!/usr/bin/env python3
"""
Quick verification script for the BIST Signal Scheduler.

This script tests the scheduler functionality without requiring full dependencies.

Usage:
    python verify_scheduler.py
"""

import sys
from pathlib import Path
from datetime import datetime, time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("BIST Signal Scheduler - Verification Script")
print("=" * 80)

# Test 1: Import check
print("\n[1/6] Testing imports...")
try:
    from src.signals.scheduler import (
        SignalScheduler,
        BISTMarketHours,
        create_default_scheduler
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check constants
print("\n[2/6] Checking market constants...")
try:
    assert BISTMarketHours.MARKET_OPEN == time(10, 0), "Market open time incorrect"
    assert BISTMarketHours.MARKET_CLOSE == time(18, 0), "Market close time incorrect"
    assert BISTMarketHours.PRE_MARKET_OPEN == time(9, 40), "Pre-market open time incorrect"
    assert BISTMarketHours.POST_MARKET_CLOSE == time(18, 10), "Post-market close time incorrect"
    print("✓ Market hours: 10:00 - 18:00 (Turkey Time)")
    print("✓ Pre-market: 09:40 - 10:00")
    print("✓ Post-market: 18:00 - 18:10")
except AssertionError as e:
    print(f"✗ Constant check failed: {e}")
    sys.exit(1)

# Test 3: Check timezone
print("\n[3/6] Checking timezone configuration...")
try:
    from zoneinfo import ZoneInfo
    tz = BISTMarketHours.TURKEY_TZ
    assert tz == ZoneInfo("Europe/Istanbul"), "Timezone is not Europe/Istanbul"
    print(f"✓ Timezone: {tz}")
except Exception as e:
    print(f"✗ Timezone check failed: {e}")
    sys.exit(1)

# Test 4: Check holiday list
print("\n[4/6] Checking holiday calendar...")
try:
    holidays = BISTMarketHours.TURKISH_HOLIDAYS
    assert len(holidays) > 0, "No holidays configured"
    print(f"✓ {len(holidays)} holidays configured")
    print(f"  Sample holidays: {list(holidays)[:3]}")
except Exception as e:
    print(f"✗ Holiday check failed: {e}")
    sys.exit(1)

# Test 5: Test market hours detection
print("\n[5/6] Testing market hours detection...")
try:
    # Test current time
    now = datetime.now(BISTMarketHours.TURKEY_TZ)
    is_trading_day = BISTMarketHours.is_trading_day()
    is_open = BISTMarketHours.is_market_open()

    print(f"✓ Current time (Turkey): {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"✓ Is trading day: {is_trading_day}")
    print(f"✓ Is market open: {is_open}")

    if is_open:
        remaining = BISTMarketHours.get_trading_minutes_remaining()
        print(f"✓ Minutes until close: {remaining}")
    else:
        next_open = BISTMarketHours.get_next_market_open()
        print(f"✓ Next market open: {next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}")

except Exception as e:
    print(f"✗ Market hours detection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Create scheduler instance
print("\n[6/6] Testing scheduler creation...")
try:
    def dummy_30m_generator():
        return {"status": "success", "signals": []}

    def dummy_1h_generator():
        return {"status": "success", "signals": []}

    scheduler = create_default_scheduler(
        signal_generator_30m=dummy_30m_generator,
        signal_generator_1h=dummy_1h_generator,
        run_on_start=False
    )

    print("✓ Scheduler created successfully")

    # Check jobs
    jobs = scheduler.scheduler.get_jobs()
    print(f"✓ Scheduled jobs: {len(jobs)}")
    for job in jobs:
        print(f"  - {job.id}: {job.name}")

    # Get market status
    status = scheduler.get_market_status()
    print("✓ Market status retrieved:")
    for key, value in status.items():
        print(f"  - {key}: {value}")

    # Clean up
    if scheduler.scheduler.running:
        scheduler.stop(wait=False)

    print("✓ Scheduler cleaned up")

except Exception as e:
    print(f"✗ Scheduler creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("VERIFICATION COMPLETE - ALL TESTS PASSED ✓")
print("=" * 80)
print("\nThe BIST Signal Scheduler is ready to use!")
print("\nNext steps:")
print("  1. Review: docs/SCHEDULER_GUIDE.md")
print("  2. Configure: configs/scheduler_config.yaml")
print("  3. Run example: python examples/run_signal_scheduler.py --status")
print("  4. Run tests: pytest tests/test_scheduler.py -v")
print("\n" + "=" * 80)
