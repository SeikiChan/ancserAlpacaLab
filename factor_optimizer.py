"""
Factor Optimizer — Random Search
Automatically explores factor/weight/time combinations to find
the best-performing configuration ranked by Calmar ratio.
"""

import warnings
warnings.filterwarnings('ignore')

import threading
import time
import logging
import json
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np

from factor_library import FACTOR_REGISTRY

logger = logging.getLogger(__name__)

# Exclude factors with external data dependencies from optimizer
# (graham requires yfinance calls per-ticker which is too slow for mass iteration)
SLOW_FACTORS = {'graham'}

# Search space
REBALANCE_OPTIONS = ['weekly_fri', 'weekly_mon', '2week', 'monthly']
TOP_N_OPTIONS = [5, 10, 15, 20, 30]
YEARS_OPTIONS = [3, 5, 7, 10]


class FactorOptimizer:
    """Random search optimizer over factor configurations."""

    def __init__(self, run_backtest_fn, save_strategy_fn=None):
        """
        Args:
            run_backtest_fn: callable(factor_config, rebalance_rule, years, top_n, **kw) -> result dict
            save_strategy_fn: callable(name, strategy_dict) -> None  (for auto-saving top results)
        """
        self._run_backtest = run_backtest_fn
        self._save_strategy = save_strategy_fn
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # State
        self._status = 'idle'  # idle | running | done | stopped | error
        self._progress = 0
        self._total = 0
        self._results = []  # list of {config, stats, rank}
        self._best = None
        self._error = None
        self._start_time = None

    # ------ Public API ------

    def start(self, n_iterations=50, min_factors=1, max_factors=None,
              universe_mode='sp500_nasdaq100', tickers=None):
        """Start optimization in background thread."""
        if self._status == 'running':
            raise RuntimeError("Optimizer already running")

        available = [k for k in FACTOR_REGISTRY if k not in SLOW_FACTORS]
        if max_factors is None:
            max_factors = min(len(available), 5)
        max_factors = min(max_factors, len(available))
        min_factors = max(1, min(min_factors, max_factors))

        self._stop_event.clear()
        self._status = 'running'
        self._progress = 0
        self._total = n_iterations
        self._results = []
        self._best = None
        self._error = None
        self._start_time = time.time()

        self._thread = threading.Thread(
            target=self._run_loop,
            args=(n_iterations, available, min_factors, max_factors,
                  universe_mode, tickers),
            daemon=True
        )
        self._thread.start()

    def stop(self):
        """Gracefully stop the optimizer."""
        self._stop_event.set()
        self._status = 'stopped'

    def get_status(self):
        """Return current status for the API."""
        with self._lock:
            elapsed = time.time() - self._start_time if self._start_time else 0
            eta = 0
            if self._progress > 0 and self._total > 0:
                per_iter = elapsed / self._progress
                remaining = self._total - self._progress
                eta = per_iter * remaining

            top_results = sorted(
                self._results,
                key=lambda r: r.get('calmar', -999),
                reverse=True
            )[:20]

            return {
                'status': self._status,
                'progress': self._progress,
                'total': self._total,
                'elapsed_sec': round(elapsed, 1),
                'eta_sec': round(eta, 1),
                'results_count': len(self._results),
                'top_results': top_results,
                'best': self._best,
                'error': self._error,
            }

    # ------ Internal ------

    def _run_loop(self, n_iterations, available_factors, min_factors, max_factors,
                  universe_mode, tickers):
        """Main optimization loop (runs in background thread)."""
        seen_configs = set()

        for i in range(n_iterations):
            if self._stop_event.is_set():
                break

            try:
                # Generate unique random config
                for _attempt in range(20):
                    config = self._random_config(available_factors, min_factors, max_factors)
                    config_key = self._config_hash(config)
                    if config_key not in seen_configs:
                        seen_configs.add(config_key)
                        break

                years = config.pop('_years')
                rebalance = config.pop('_rebalance')
                top_n = config.pop('_top_n')

                # Run backtest
                result = self._run_backtest(
                    factor_config=config,
                    rebalance_rule=rebalance,
                    years=years,
                    top_n=top_n,
                    universe_mode=universe_mode,
                    tickers=tickers,
                )

                stats = result.get('stats', {}).get('strategy', {})
                calmar = stats.get('calmar', -999)
                sharpe = stats.get('sharpe', -999)
                cagr = stats.get('cagr', 0)
                max_dd = stats.get('max_dd', 0)

                # Build readable factor summary
                enabled = {k: v for k, v in config.items() if v.get('enabled')}
                factor_summary = {k: round(v.get('weight', 0), 3) for k, v in enabled.items()}

                entry = {
                    'iteration': i + 1,
                    'calmar': round(calmar, 4),
                    'sharpe': round(sharpe, 4),
                    'cagr': round(cagr, 4),
                    'max_dd': round(max_dd, 4),
                    'factors': factor_summary,
                    'top_n': top_n,
                    'rebalance': rebalance,
                    'years': years,
                    'full_config': config,
                }

                with self._lock:
                    self._results.append(entry)
                    self._progress = i + 1

                    if self._best is None or calmar > self._best.get('calmar', -999):
                        self._best = entry

            except Exception as e:
                logger.warning(f"Optimizer iteration {i+1} failed: {e}")
                with self._lock:
                    self._progress = i + 1

        # Done — auto-save top 10
        with self._lock:
            self._status = 'done'
            if self._save_strategy and self._results:
                self._auto_save_top(10)

    def _auto_save_top(self, n):
        """Save top N results to strategy library with optimizer tag."""
        sorted_results = sorted(
            self._results,
            key=lambda r: r.get('calmar', -999),
            reverse=True
        )[:n]

        for rank, entry in enumerate(sorted_results, 1):
            name = f"⚡OPT #{rank} Calmar={entry['calmar']:.2f}"
            strategy = {
                'source': 'optimizer',
                'rank': rank,
                'calmar': entry['calmar'],
                'sharpe': entry['sharpe'],
                'cagr': entry['cagr'],
                'max_dd': entry['max_dd'],
                'factors': entry['full_config'],
                'rebalance': entry['rebalance'],
                'years': entry['years'],
                'top_n': entry['top_n'],
                'timestamp': datetime.now().isoformat(),
            }
            try:
                self._save_strategy(name, strategy)
            except Exception as e:
                logger.warning(f"Failed to save optimizer result #{rank}: {e}")

    def _random_config(self, available, min_k, max_k):
        """Generate a random factor configuration."""
        rng = np.random.default_rng()

        # Pick how many factors
        k = rng.integers(min_k, max_k + 1)
        chosen = list(rng.choice(available, size=k, replace=False))

        # Random Dirichlet weights (always sum to 1.0)
        weights = rng.dirichlet(np.ones(k))

        config = {}
        for key in available:
            if key in chosen:
                idx = chosen.index(key)
                w = float(round(weights[idx], 3))
                factor_meta = FACTOR_REGISTRY[key]
                # Use default params
                params = {pk: pv['default'] for pk, pv in factor_meta['params'].items()}
                config[key] = {'enabled': True, 'weight': w, **params}
            else:
                config[key] = {'enabled': False, 'weight': 0}

        # Add meta fields (popped by caller)
        config['_years'] = int(rng.choice(YEARS_OPTIONS))
        config['_rebalance'] = str(rng.choice(REBALANCE_OPTIONS))
        config['_top_n'] = int(rng.choice(TOP_N_OPTIONS))

        return config

    @staticmethod
    def _config_hash(config):
        """Create a hashable key from a config to avoid exact duplicates."""
        parts = []
        for k in sorted(config.keys()):
            if k.startswith('_'):
                parts.append(f"{k}={config[k]}")
            elif isinstance(config[k], dict) and config[k].get('enabled'):
                parts.append(f"{k}={config[k].get('weight', 0):.3f}")
        return '|'.join(parts)
