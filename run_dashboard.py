#!/usr/bin/env python3
"""
Soccer Analytics Dashboard Launcher

Simple script to launch the soccer analytics dashboard web application.
Make sure you have run the pipeline first to generate the required data.

Author: Michael Xu
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import flask
        import plotly
        import pandas
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("💡 Install requirements with: pip install -r Soccer Analytics Pipeline/webapp_requirements.txt")
        return False

def check_data():
    """Check if required data files exist"""
    stats_dir = Path("Statistics")
    required_files = [
        "statistics_performance_scores.csv",
        "statistics_totals.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not (stats_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required data files:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n💡 Please run the pipeline first:")
        print("   1. python3 01_data_preparation.py")
        print("   2. python3 02_player_level_transformer.py")
        print("   3. python3 03_statistical_aggregator.py")
        return False
    
    print("✅ All required data files found")
    return True

def main():
    """Main launcher function"""
    print("⚽ Soccer Analytics Dashboard Launcher")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check data
    if not check_data():
        return
    
    # Import and run the app
    try:
        from soccer_dashboard_app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

if __name__ == "__main__":
    main()
