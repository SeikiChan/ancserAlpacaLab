# Scripts & Utilities

This folder contains utility scripts and tools for testing and analysis.

## Available Scripts

### 1. run_drift_comparison.py
**Purpose:** Compare performance of Raw Reversion vs Drift-Filtered Reversion factors

**Usage:**
```bash
python scripts/run_drift_comparison.py
```

**What it does:**
- Fetches historical data for 500+ symbols
- Runs two backtests in parallel:
  1. Raw Reversion (using RSI factor)
  2. Drift-Filtered Reversion (RSI only in non-drift regime)
- Compares metrics (Total Return, Max Drawdown, Calmar Ratio)
- Validates that drift filter reduces risk

**Expected Output:**
```
SUCCESS: Drift Filter reduced Max Drawdown!
```

---

### 2. debug_benchmark.py
**Purpose:** Debug benchmark data fetching and processing

**Usage:**
```bash
python scripts/debug_benchmark.py
```

**What it does:**
- Fetches SPY, QQQ, GLD data from Yahoo Finance
- Tests pivot/reindex logic
- Checks for NaN values and data quality
- Validates benchmark alignment for dashboard charts

**Use Case:** When dashboard benchmark charts are not displaying correctly

---

## Adding New Scripts

When adding new utility scripts to this folder:
1. Add a clear docstring at the top of the file
2. Update this README with usage instructions
3. Use `sys.path.append()` to ensure imports work:
   ```python
   import sys
   import os
   sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
   ```

---

## Best Practices

- Keep scripts focused on testing/debugging single features
- Use descriptive names (verb_noun.py, e.g., test_mwu.py)
- Add command-line arguments for flexibility (using argparse)
- Print clear progress messages and results
