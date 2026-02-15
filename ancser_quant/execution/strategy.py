import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ancser_quant.data.alpaca_adapter import AlpacaAdapter
from ancser_quant.backtest import BacktestEngine

class LiveStrategy:
    """
    Shared logic for calculating target portfolio state.
    Used by both the Dashboard (Preview) and Main Loop (Execution).
    """
    
    def __init__(self):
        self.alpaca = AlpacaAdapter()
        
    def calculate_targets(self, config: dict, current_equity: float = None) -> dict:
        """
        Calculate target weights and volatility scalar based on live config.
        """
        universe = config.get('universe', [])
        factors = config.get('active_factors', [])
        
        if not universe or not factors:
            return {"error": "Universe or Factors empty"}

        # 1. Volatility Targeting
        target_scalar = 1.0
        vol_metrics = {}
        
        use_vol_target = config.get('use_vol_target', False)
        vol_target = config.get('vol_target', 0.20)
        leverage_cap = config.get('leverage', 1.0)
        
        # We need history for both Vol and Factors
        # Fetch 60 days to be safe (Factors need ~20-50 days, Vol needs ~20 days)
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=90)
        
        try:
            # Re-use BacktestEngine's robust fetcher but for Alpaca source
            # Or just use adapter directly to get Polars DataFrame
            hist_pl = self.alpaca.fetch_history(universe, start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')).collect()
            
            if hist_pl.is_empty():
                 return {"error": "No historical data fetched"}
                 
            hist = hist_pl.to_pandas()
            hist['timestamp'] = pd.to_datetime(hist['timestamp'])
             
            # Calculate Volatility Scalar
            # 2. Factor Calculation
            # Get latest available data for each symbol
            # We need to construct a DataFrame similar to what BacktestEngine uses
            
            # Pivot prices
            closes = hist.pivot(index='timestamp', columns='symbol', values='close')
            
            # Initialize scores
            scores = pd.Series(0.0, index=closes.columns)
            
            # Simple Equal Weight per Factor for now (or load from config)
            # In BacktestEngine, we use MWU weights if enabled. 
            # For simplicity in Live Preview, let's assume Equal Weight across selected factors.
            weight_per_factor = 1.0 / len(factors)
            
            for f in factors:
                factor_rank = pd.Series(0.0, index=closes.columns)
                
                if f == 'Momentum':
                    # ROC 126 (6-month)
                    mom = closes.pct_change(126).iloc[-1]
                    factor_rank = mom.rank(pct=True, ascending=True)
                    
                elif f == 'Reversion':
                    # 5-day RSI (via -ReturnValue)
                    rev = closes.pct_change(5).iloc[-1]
                    factor_rank = rev.rank(pct=True, ascending=False) # Lower return -> Higher rank
                    
                elif f == 'Volatility':
                    # 20-day Vol (Lower is better)
                    vol = closes.pct_change().tail(20).std()
                    factor_rank = vol.rank(pct=True, ascending=False)
                    
                elif f == 'Skew':
                    # 60-day Skew (Higher is better)
                    skew = closes.pct_change().tail(60).skew()
                    factor_rank = skew.rank(pct=True, ascending=True)
                
                elif f == 'Microstructure':
                    # Amihud Illiquidity (High Volume/Return ratio - complicated for just price data)
                    # Proxy: Lower Volume Variance? Or just 1-day Reversion
                    # Let's use 1-day Reversion as placeholder for Microstructure
                    rev1 = closes.pct_change(1).iloc[-1]
                    factor_rank = rev1.rank(pct=True, ascending=False)

                scores += factor_rank.fillna(0.5) * weight_per_factor

            # 3. Portfolio Construction
            # Select Top 5 Stocks
            top_n = 5
            top_stocks = scores.nlargest(top_n)
            
            # Calculate Target Weights
            # Equal Weight * Vol Scalar
            target_weight = (1.0 / top_n) * target_scalar
            
            allocations = {}
            for sym, score in top_stocks.items():
                allocations[sym] = target_weight
                
            return {
                "allocations": allocations, # Symbol -> Weight (e.g. 0.20)
                "vol_metrics": vol_metrics,
                "latest_prices": closes.iloc[-1].to_dict(),
                "factor_scores": scores.to_dict()
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

