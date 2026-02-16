import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from ancser_quant.backtest import BacktestEngine
from ancser_quant.data.constituents import TICKERS
import pandas as pd
import numpy as np

def calculate_max_drawdown(equity_series):
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    return drawdown.min()

def calculate_total_return(equity_series):
    return (equity_series.iloc[-1] / equity_series.iloc[0]) - 1

def run_comparison():
    engine = BacktestEngine(initial_capital=100000.0, data_source='alpaca')
    
    # 500+ Symbols from Constituents
    symbols = TICKERS
    print(f"Loaded {len(symbols)} symbols from constituents.")
    
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    # 1. Fetch Data Once
    data = engine.fetch_and_prepare_data(symbols, start_date, end_date)
    
    if data.empty:
        print("Error: No data fetched.")
        return

    print("\n" + "="*50)
    print("RUNNING COMPARISON: Raw Reversion vs Drift-Filtered")
    print("="*50 + "\n")

    # 2. Run Raw Reversion
    print("Running Raw Reversion...")
    res_raw, _ = engine.run_simulation(
        data, 
        active_factors=['Reversion'], 
        leverage=1.0, 
        use_mwu=False, 
        use_vol_target=False
    )
    
    # 3. Run Drift-Filtered Reversion
    print("Running Drift-Filtered Reversion...")
    res_drift, _ = engine.run_simulation(
        data, 
        active_factors=['Drift-Reversion'], 
        leverage=1.0, 
        use_mwu=False, 
        use_vol_target=False
    )
    
    if res_raw.empty or res_drift.empty:
        print("Simulation returned empty results.")
        return

    # 4. Compare Results
    mdd_raw = calculate_max_drawdown(res_raw['equity'])
    ret_raw = calculate_total_return(res_raw['equity'])
    
    mdd_drift = calculate_max_drawdown(res_drift['equity'])
    ret_drift = calculate_total_return(res_drift['equity'])
    
    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)
    print(f"{'Metric':<20} | {'Raw Reversion':<15} | {'Drift-Filtered':<15}")
    print("-" * 56)
    print(f"{'Total Return':<20} | {ret_raw:.2%}          | {ret_drift:.2%}")
    print(f"{'Max Drawdown':<20} | {mdd_raw:.2%}          | {mdd_drift:.2%}")
    print("-" * 56)
    
    if mdd_drift > mdd_raw: # Note: specific Drawdown is negative number, so smaller negative number (closer to 0) is better.
        # e.g. -0.10 > -0.20
        print("\nSUCCESS: Drift Filter reduced Max Drawdown!")
    else:
        print("\nOBSERVATION: Drift Filter did not reduce Max Drawdown in this period.")
        
    print(f"\nDebug: Final Equity Raw:   ${res_raw['equity'].iloc[-1]:,.2f}")
    print(f"Debug: Final Equity Drift: ${res_drift['equity'].iloc[-1]:,.2f}")

if __name__ == "__main__":
    run_comparison()
