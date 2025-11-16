#!/usr/bin/env python3
"""
Dashboard Launcher for BIST AI Trading System

This script launches the Streamlit dashboard with proper configuration.

Usage:
    python run_dashboard.py
    # or
    ./run_dashboard.py

Author: BIST AI Trading System
Date: 2025-11-16
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ['STREAMLIT_SERVER_PORT'] = '8501'
os.environ['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Import streamlit
try:
    import streamlit.web.cli as stcli
except ImportError:
    print("Error: Streamlit is not installed.")
    print("Please install it using: pip install streamlit")
    sys.exit(1)

# Dashboard path
dashboard_path = project_root / "src" / "ui" / "dashboard.py"

if not dashboard_path.exists():
    print(f"Error: Dashboard file not found at {dashboard_path}")
    sys.exit(1)

# Launch dashboard
if __name__ == "__main__":
    print("=" * 80)
    print("BIST AI Trading System - Dashboard")
    print("=" * 80)
    print(f"\nLaunching dashboard from: {dashboard_path}")
    print(f"Dashboard will be available at: http://localhost:8501")
    print("\nPress Ctrl+C to stop the dashboard\n")
    print("=" * 80)

    sys.argv = [
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port=8501",
        "--server.address=localhost",
        "--browser.gatherUsageStats=false",
        "--theme.primaryColor=#1f77b4",
        "--theme.backgroundColor=#FFFFFF",
        "--theme.secondaryBackgroundColor=#f0f2f6",
        "--theme.textColor=#262730",
        "--theme.font=sans serif"
    ]

    sys.exit(stcli.main())
