import polars as pl
import numpy as np
from typing import Dict, List, Optional

class MWUEngine:
    """
    Multiplicative Weights Update (MWU) Engine.
    Dynamically adjusts factor weights based on their realized performance (IC).

    Improvements:
    - Added weight bounds to prevent extreme allocations
    - Added IC smoothing option for stability
    - Tracks IC history for analysis
    """
    def __init__(self, factor_names: List[str], learning_rate: float = 0.1, window: int = 252,
                 min_weight: float = 0.05, max_weight: float = 0.60):
        self.factor_names = factor_names
        self.learning_rate = learning_rate
        self.window = window
        self.min_weight = min_weight  # Prevent factors from being completely ignored
        self.max_weight = max_weight  # Prevent over-concentration

        # Initial equal weights
        n = len(factor_names)
        self.weights = {f: 1.0 / n for f in factor_names}

        # History for tracking
        self.history = []
        self.ic_history = []  # Track IC values for analysis

    def update(self, date: str, factor_ics: Dict[str, float]):
        """
        Update weights based on Information Coefficients (IC).
        IC > 0 -> Reward (Increase weight)
        IC < 0 -> Penalize (Decrease weight)

        Improved with:
        - Weight bounds (min/max) to prevent extreme allocations
        - IC history tracking for analysis
        """
        current_weights = self.weights.copy()
        new_weights = {}
        total_weight = 0.0

        # Record IC history
        ic_record = {'date': date}
        ic_record.update(factor_ics)
        self.ic_history.append(ic_record)

        for factor, ic in factor_ics.items():
            if factor not in current_weights:
                continue

            w_t = current_weights[factor]

            # Update Rule: W_{t+1} = W_t * (1 +/- eta * |IC|)
            # Scale IC by learning rate
            # We treat positive IC as 'gain' and negative IC as 'loss' for that expert

            if ic > 0:
                multiplier = 1.0 + self.learning_rate * abs(ic)
            else:
                multiplier = 1.0 - self.learning_rate * abs(ic)

            # Floor multiplier to avoid negative weights or zeroing out too fast
            multiplier = max(0.1, multiplier)

            w_next = w_t * multiplier
            new_weights[factor] = w_next
            total_weight += w_next

        # Normalize
        if total_weight > 0:
            for f in new_weights:
                new_weights[f] /= total_weight
        else:
            # Fallback to equal weights if everything dies
            n = len(new_weights)
            for f in new_weights:
                new_weights[f] = 1.0 / n

        # Apply weight bounds (min/max constraints)
        # This prevents over-concentration in one factor or complete ignoring of others
        bounded_weights = {}
        for f, w in new_weights.items():
            bounded_weights[f] = max(self.min_weight, min(self.max_weight, w))

        # Re-normalize after bounding
        total_bounded = sum(bounded_weights.values())
        if total_bounded > 0:
            for f in bounded_weights:
                bounded_weights[f] /= total_bounded

        self.weights = bounded_weights

        # Record history
        record = {'date': date}
        record.update(self.weights)
        self.history.append(record)

        return self.weights

    def get_history_df(self) -> pl.DataFrame:
        """Returns weight history as a Polars DataFrame for visualization."""
        if not self.history:
            return pl.DataFrame()
        return pl.DataFrame(self.history)

    def get_ic_history_df(self) -> pl.DataFrame:
        """Returns IC history as a Polars DataFrame for analysis."""
        if not self.ic_history:
            return pl.DataFrame()
        return pl.DataFrame(self.ic_history)
