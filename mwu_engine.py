"""
Multiplicative Weights Update (MWU) Engine
------------------------------------------
Implements the Hedge algorithm for online learning of factor weights.

The goal is to dynamically adjust weights of different factors based on their
recent performance (loss/gain) to minimize regret relative to the best factor.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MWUEngine:
    def __init__(self, factors, learning_rate=0.05):
        """
        Args:
            factors (list): List of factor names.
            learning_rate (float): Eta parameter, controls adaptation speed (default 0.05).
        """
        self.factors = factors
        self.eta = learning_rate
        self.weights = {f: 1.0 / len(factors) for f in factors}
        self.history = []

    def update(self, factor_returns):
        """
        Update weights based on factor returns for a single period.
        
        Args:
            factor_returns (dict): {factor_name: return_value}
                                   Returns should be scaled roughly [-1, 1] or similar.
        """
        # Clip returns to avoid explosion [-1.0, 1.0]
        # We assume returns are roughly percentages like 0.01 (1%) to 0.10 (10%)
        # If returns are large, eta should be small.
        clipped_returns = {k: np.clip(v, -1.0, 1.0) for k, v in factor_returns.items()}
        
        current_weights = self.weights.copy()
        
        sum_weights = 0.0
        new_weights = {}
        
        for f in self.factors:
            r = clipped_returns.get(f, 0.0)
            w = current_weights.get(f, 0.0)
            
            # Multiplicative update: w_new = w * (1 + eta * r)
            # This rewards positive returns and penalizes negative returns
            nw = w * (1.0 + self.eta * r)
            
            # Avoid weights reaching exactly 0
            if nw < 1e-6:
                nw = 1e-6
                
            new_weights[f] = nw
            sum_weights += nw
            
        # Normalize to sum to 1.0
        if sum_weights > 0:
            self.weights = {k: v / sum_weights for k, v in new_weights.items()}
        else:
            # Fallback to equal weights
            n = len(self.factors)
            self.weights = {f: 1.0 / n for f in self.factors}
            
        self.history.append({
            'weights': self.weights.copy(),
            'returns': factor_returns
        })
        
        return self.weights

    def run_simulation(self, factor_returns_df):
        """
        Run MWU over a history of factor returns to get a time series of weights.
        
        Args:
            factor_returns_df (pd.DataFrame): Index=Date, Columns=Factors, Values=Returns
        
        Returns:
            pd.DataFrame: Weights over time (index same as input, shifted by 1 period effectively)
                          The weight at time T is formed using info up to T-1.
        """
        weights_history = []
        
        # Reset weights to equal
        self.weights = {f: 1.0 / len(self.factors) for f in self.factors}
        
        # We iterate through history.
        # At step i, we use current weights. Then we observe returns(i) and update for i+1.
        
        for dt, row in factor_returns_df.iterrows():
            rets = row.to_dict()
            
            # Record CURRENT weights (before seeing this return)
            # These are the weights we would have traded with at time dt
            weights_history.append(self.weights.copy())
            
            # Update weights for NEXT period
            self.update(rets)
            
        return pd.DataFrame(weights_history, index=factor_returns_df.index)
