"""
Factor Analysis Web Dashboard
Flask server wrapping existing quant modules
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import math
import uuid
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from copy import deepcopy

import pandas as pd
import numpy as np
import yaml
from flask import Flask, request, jsonify, send_from_directory

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_manager import DataManager
from factor_library import FactorEngine, zscore_cross_sectional, FACTOR_REGISTRY
from mwu_engine import MWUEngine
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from dotenv import load_dotenv

load_dotenv()

from factor_optimizer import FactorOptimizer

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# Flask App
# ==========================================

app = Flask(__name__, static_folder='static', static_url_path='/static')
# Initialize Alpaca Clients
API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
PAPER = True  # Default to paper, can be made configurable

trading_client = None
if API_KEY and SECRET_KEY:
    try:
        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
        logger.info("Alpaca Trading Client initialized")
    except Exception as e:
        logger.error(f"Failed to init Alpaca Client: {e}")

SAVED_DIR = Path(__file__).parent / 'saved_data'
SAVED_DIR.mkdir(exist_ok=True)
FACTORS_FILE = SAVED_DIR / 'factors.json'
STRATEGIES_FILE = SAVED_DIR / 'strategies.json'

BACKTEST_LOG_DIR = Path(__file__).parent / 'logs' / 'backtest'
BACKTEST_LOG_DIR.mkdir(parents=True, exist_ok=True)

# --------------- auto-derived from factor_library decorators ---------------
FACTOR_INFO = {
    k: {kk: vv for kk, vv in v.items() if kk not in ('func', 'data_source')}
    for k, v in FACTOR_REGISTRY.items()
}
FACTOR_FUNCTIONS = {k: v['func'] for k, v in FACTOR_REGISTRY.items()}

# Rebalance frequency mapping
REBALANCE_MAP = {
    'daily': 'D',
    'weekly_fri': 'W-FRI',
    'weekly_mon': 'W-MON',
    '2week': '2W-FRI',
    'monthly': 'ME',
    'quarterly': 'QS',
    'half_year': '2QS',
    'yearly': 'YS',
}


# ==========================================
# Helpers
# ==========================================

def _load_json(filepath):
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _save_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _equity_to_json(series):
    """Convert a pandas Series to {date_string: value} dict."""
    result = {}
    for dt, val in series.items():
        if pd.isna(val):
            continue
        date_str = dt.strftime('%Y-%m-%d') if hasattr(dt, 'strftime') else str(dt)
        result[date_str] = round(float(val), 6)
    return result


def _compute_stats(equity_series, returns_series):
    """Compute performance stats from equity curve and returns."""
    if len(equity_series) < 2:
        return {}

    days = (equity_series.index[-1] - equity_series.index[0]).days
    years = max(days / 365.25, 0.01)
    final = float(equity_series.iloc[-1])
    cagr = float(final ** (1 / years) - 1) if final > 0 else 0

    peak = equity_series.cummax()
    dd = equity_series / peak - 1
    max_dd = float(dd.min())

    std = returns_series.std()
    sharpe = float((returns_series.mean() / std) * math.sqrt(252)) if std > 0 else 0
    win_rate = float((returns_series > 0).mean())

    # Calmar Ratio: CAGR / |MaxDD| — prefers high CAGR, penalizes large drawdowns
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-8 else 0.0

    return {
        'calmar': round(calmar, 4),
        'cagr': round(cagr, 4),
        'max_dd': round(max_dd, 4),
        'sharpe': round(sharpe, 4),
        'win_rate': round(win_rate, 4),
        'final_value': round(final, 4),
        'years': round(years, 2)
    }


def _run_backtest_with_config(factor_config, rebalance_rule, years, top_n,
                              tickers=None, universe_mode='sp500_nasdaq100',
                              start_date=None, end_date=None,
                              data_start_date=None, enable_mwu=False):
    """
    Run a backtest using the provided factor configuration.
    Returns equity curves + stats.
    """
    # Load base config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Override settings
    config['portfolio']['top_n'] = top_n
    config['portfolio']['rebalance'] = rebalance_rule
    config['backtest']['years'] = years

    # Date range
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')

    # Get data — override DataManager's config AFTER init
    dm = DataManager(str(config_path))

    if tickers:
        universe = tickers
    else:
        # Override universe mode in the DataManager's own config
        dm.config['universe']['mode'] = universe_mode
        universe = dm.get_universe_list()

    logger.info(f"Universe: {len(universe)} symbols, mode: {universe_mode}, period: {start_date} to {end_date}")

    # Use data_start_date for fetching (allows factor warm-up before actual backtest start)
    fetch_start = data_start_date if data_start_date else start_date

    close_df, volume_df = dm.get_market_data(
        universe=universe,
        start_date=fetch_start,
        end_date=end_date,
        use_cache=True
    )

    if close_df.empty:
        raise ValueError("No market data available")

    benchmarks_list = config['data']['benchmarks']
    defensive_list = config['data'].get('defensive_assets', [])

    spy = close_df[benchmarks_list[0]] if benchmarks_list[0] in close_df.columns else None
    qqq = close_df[benchmarks_list[1]] if len(benchmarks_list) > 1 and benchmarks_list[1] in close_df.columns else None

    # Separate stock columns
    exclude_cols = set(benchmarks_list + defensive_list)
    stock_cols = [c for c in close_df.columns if c not in exclude_cols]
    close_stocks = close_df[stock_cols] if stock_cols else close_df
    volume_stocks = volume_df[[c for c in stock_cols if c in volume_df.columns]] if stock_cols else volume_df

    if close_stocks.empty:
        raise ValueError("No stock data after filtering")

    # Compute factors based on user config
    factors = {}
    total_weight = 0

    for factor_name, fcfg in factor_config.items():
        if not fcfg.get('enabled', False):
            continue
        weight = fcfg.get('weight', 0)
        if weight <= 0:
            continue
        total_weight += weight

        func = FACTOR_FUNCTIONS.get(factor_name)
        if func is None:
            continue

        params = {k: v for k, v in fcfg.items() if k not in ('enabled', 'weight')}
        
        try:
            data_source = FACTOR_REGISTRY.get(factor_name, {}).get('data_source', 'close')
            if data_source == 'volume':
                factor_df = func(volume_stocks, **params)
            else:
                factor_df = func(close_stocks, **params)
            factors[factor_name] = (factor_df, weight)
        except Exception as e:
            logger.warning(f"Factor {factor_name} computation failed: {e}")

    if not factors:
        raise ValueError("No valid factors computed")

    # Compute composite score
    sample_df = list(factors.values())[0][0]
    score = pd.DataFrame(0.0, index=sample_df.index, columns=sample_df.columns)

    if enable_mwu and len(factors) > 1:
        logger.info("MWU Backtest: Calculating dynamic weights...")
        try:
            # 1. Calculate Factor Returns
            # We need to see how each factor performs period-over-period
            # We'll use weekly returns for stability
            factor_names = list(factors.keys())
            common_idx = sample_df.index
            
            # Resample prices to weekly (Friday)
            wk_close = close_stocks.resample('W-FRI').last()
            wk_ret = wk_close.pct_change()
            
            # Align factors to weekly
            wk_factors = {}
            for fname, (fdf, _) in factors.items():
                wk_factors[fname] = fdf.reindex(wk_close.index).ffill()
                
            # Compute factor returns history
            factor_ret_history = []
            valid_wk_dates = wk_close.index
            
            for i in range(1, len(valid_wk_dates)):
                dt = valid_wk_dates[i]
                prev_dt = valid_wk_dates[i-1]
                
                # Returns from prev to curr
                period_ret_stk = wk_ret.loc[dt]
                
                period_f_rets = {}
                for fname in factor_names:
                    # Pick top N stocks based on factor at prev_dt
                    f_vals = wk_factors[fname].loc[prev_dt]
                    valid = f_vals.dropna().sort_values(ascending=False)
                    picks = valid.head(top_n).index
                    
                    if len(picks) > 0:
                        # Avg return of these picks
                        r = period_ret_stk[picks].mean()
                        period_f_rets[fname] = r if not pd.isna(r) else 0.0
                    else:
                        period_f_rets[fname] = 0.0
                        
                factor_ret_history.append(period_f_rets)
                
            # DF of Factor Returns
            f_ret_df = pd.DataFrame(factor_ret_history, index=valid_wk_dates[1:])
            
            # Scale returns for MWU (since weekly returns are small)
            f_ret_df_scaled = f_ret_df * 5.0
            
            # 2. Run MWU
            mwu = MWUEngine(factor_names, learning_rate=0.1)
            # weights_df has index of dates where weights are APPLICABLE
            # run_simulation returns weights at T based on info < T.
            # So weights at T are ready to be used for trading at T.
            weights_df = mwu.run_simulation(f_ret_df_scaled)
            
            # 3. Compute Dynamic Composite Score
            # Reindex weights to daily
            daily_weights = weights_df.reindex(score.index).ffill().fillna(1.0/len(factor_names))
            
            # ZScore all factors once
            zscored_factors = {}
            for fname, (fdf, _) in factors.items():
                zscored_factors[fname] = zscore_cross_sectional(fdf).reindex(index=score.index, columns=score.columns).fillna(0)
                
            # Weighted sum
            for fname in factor_names:
                w_series = daily_weights[fname] # Series of weights over time
                # Expand w_series to match DataFrame shape for broadcasting
                # w_df = pd.DataFrame(np.outer(w_series.values, np.ones(score.shape[1])), index=score.index, columns=score.columns)
                # Pandas broadcasting works column-wise by default, we need row-wise
                # Multiply term: factor_matrix * weight_vector (elementwise along time)
                term = zscored_factors[fname].multiply(w_series, axis=0)
                score += term
                
        except Exception as e:
            logger.error(f"MWU Backtest Logic Failed: {e}")
            logger.info("Falling back to static weights")
            enable_mwu = False # invalidates the flag to fall through to static logic logic if we could, 
                               # but we already initialized score.
            # Fallback static calc
            score = pd.DataFrame(0.0, index=sample_df.index, columns=sample_df.columns)
            for fname, (fdf, w) in factors.items():
                norm = zscore_cross_sectional(fdf)
                norm = norm.reindex(index=score.index, columns=score.columns).fillna(0)
                score += (w / total_weight) * norm

    else:
        # Static Weights (Original Logic)
        for fname, (fdf, w) in factors.items():
            norm = zscore_cross_sectional(fdf)
            norm = norm.reindex(index=score.index, columns=score.columns).fillna(0)
            score += (w / total_weight) * norm

    # Filters
    filter_cfg = config['filters']
    dvol = (close_stocks * volume_stocks.reindex(columns=close_stocks.columns, fill_value=0)).rolling(
        filter_cfg['adv_window'], min_periods=10).mean()
    eligible = (close_stocks >= filter_cfg['min_price']) & (dvol >= filter_cfg['min_adv_dollar'])

    # Regime filter
    regime_cfg = config['regime']
    dates = close_stocks.index
    ret_stk = close_stocks.pct_change().fillna(0)

    if regime_cfg['enabled'] and spy is not None:
        indicator = spy
        sma = indicator.rolling(regime_cfg['sma_length'], min_periods=regime_cfg['sma_length'] // 2).mean()
        mom = indicator.pct_change(regime_cfg['momentum_length'])
        risk_on = (indicator > sma) & (mom > 0)
    else:
        risk_on = pd.Series(True, index=dates)

    # Rebalance dates
    idx = pd.DatetimeIndex(dates).sort_values()
    s = pd.Series(1.0, index=idx)
    if rebalance_rule.upper() == 'D':
        rebal_dates = set(idx)
    else:
        rebal = s.resample(rebalance_rule).last().dropna().index
        rebal_dates = set(rebal)

    # Cash return
    cash_asset_name = defensive_list[0] if defensive_list and defensive_list[0] in close_df.columns else None
    ret_cash = close_df[cash_asset_name].pct_change().fillna(0) if cash_asset_name else pd.Series(0, index=dates)

    # Core backtest loop
    portfolio_cfg = config['portfolio']
    costs_cfg = config['costs']
    w_prev = pd.Series(0.0, index=close_stocks.columns)
    port_ret = pd.Series(0.0, index=dates)

    # --- Track per-period holdings ---
    rebal_log = []  # list of (rebal_date, picks_list)

    for i in range(1, len(dates)):
        dt = dates[i]
        sig_dt = dates[i - 1]
        w_tgt = w_prev.copy()

        if dt in rebal_dates:
            if not risk_on.get(sig_dt, True) and regime_cfg.get('risk_off_mode') == 'defensive':
                w_tgt[:] = 0.0
                rebal_log.append((dt, []))
            else:
                mask = eligible.loc[sig_dt] if sig_dt in eligible.index else pd.Series(True, index=close_stocks.columns)
                scores = score.loc[sig_dt].where(mask).dropna() if sig_dt in score.index else pd.Series(dtype=float)

                if len(scores) >= portfolio_cfg.get('min_names_to_trade', 1):
                    ranks = scores.rank(ascending=False, method='first')
                    candidates = list(scores.sort_values(ascending=False).index)
                    picks = candidates[:top_n]

                    if picks:
                        base_w = 1.0 / len(picks)
                        w_tgt[:] = 0.0
                        for ticker in picks:
                            w_tgt.loc[ticker] = min(base_w, portfolio_cfg.get('max_weight', 0.4))
                        total = w_tgt.sum()
                        if total > 0:
                            w_tgt = w_tgt / total
                        rebal_log.append((dt, picks[:]))
                    else:
                        rebal_log.append((dt, []))
                else:
                    w_tgt[:] = 0.0
                    rebal_log.append((dt, []))

        turn = 0.5 * float(np.abs(w_tgt - w_prev).sum())
        cost = turn * (costs_cfg['total_bps'] / 10000.0)

        stock_ret = float((w_tgt * ret_stk.loc[dt]).sum())
        if float(w_tgt.sum()) < 1e-12:
            stock_ret = float(ret_cash.get(dt, 0))

        port_ret.loc[dt] = stock_ret - cost
        w_prev = w_tgt

    # Trim to actual backtest window (exclude factor warm-up period)
    if data_start_date and start_date:
        actual_start = pd.Timestamp(start_date)
        trim_mask = port_ret.index >= actual_start
        port_ret = port_ret[trim_mask]

    # Build equity curves
    eq_strategy = (1 + port_ret).cumprod()

    # Also trim benchmarks to match
    eq_dates = eq_strategy.index
    ret_spy = spy.pct_change().fillna(0).reindex(eq_dates, fill_value=0) if spy is not None else pd.Series(0, index=eq_dates)
    ret_qqq = qqq.pct_change().fillna(0).reindex(eq_dates, fill_value=0) if qqq is not None else pd.Series(0, index=eq_dates)
    eq_spy = (1 + ret_spy).cumprod()
    eq_qqq = (1 + ret_qqq).cumprod()

    # --- Compute per-period holdings returns ---
    holdings_history = _compute_holdings_history(rebal_log, close_stocks, start_date)
    trade_summary = _compute_trade_summary(holdings_history)

    return {
        'strategy': _equity_to_json(eq_strategy),
        'spy': _equity_to_json(eq_spy),
        'qqq': _equity_to_json(eq_qqq),
        'stats': {
            'strategy': _compute_stats(eq_strategy, port_ret),
            'spy': _compute_stats(eq_spy, ret_spy),
            'qqq': _compute_stats(eq_qqq, ret_qqq),
        },
        'holdings_history': holdings_history,
        'trade_summary': trade_summary,
    }


def _compute_holdings_history(rebal_log, close_stocks, start_date):
    """
    Compute per-period holdings with their returns.
    Returns a list of periods (most recent first), each with:
      - label: date range string
      - holdings: list of {ticker, return_pct} sorted best-to-worst
    """
    if not rebal_log:
        return []

    # Filter to only rebalances after actual start_date
    if start_date:
        actual_start = pd.Timestamp(start_date)
        rebal_log = [(dt, picks) for dt, picks in rebal_log if pd.Timestamp(dt) >= actual_start]

    if not rebal_log:
        return []

    periods = []
    for idx in range(len(rebal_log)):
        dt, picks = rebal_log[idx]
        if not picks:
            continue

        # Determine end of this holding period
        if idx + 1 < len(rebal_log):
            next_dt = rebal_log[idx + 1][0]
        else:
            # Last rebalance — use the last available date
            next_dt = close_stocks.index[-1]

        # Compute per-stock return during this holding period
        holdings = []
        for ticker in picks:
            if ticker not in close_stocks.columns:
                continue
            try:
                price_start = close_stocks.loc[dt, ticker] if dt in close_stocks.index else None
                price_end = close_stocks.loc[next_dt, ticker] if next_dt in close_stocks.index else None
                if price_start and price_end and price_start > 0:
                    ret = (price_end / price_start - 1)
                    holdings.append({'ticker': ticker, 'return_pct': round(ret * 100, 2)})
                else:
                    holdings.append({'ticker': ticker, 'return_pct': 0.0})
            except Exception:
                holdings.append({'ticker': ticker, 'return_pct': 0.0})

        # Sort by return descending (best first)
        holdings.sort(key=lambda h: h['return_pct'], reverse=True)

        dt_str = pd.Timestamp(dt).strftime('%Y-%m-%d')
        next_str = pd.Timestamp(next_dt).strftime('%Y-%m-%d')
        periods.append({
            'label': f"{dt_str} ~ {next_str}",
            'date': dt_str,
            'holdings': holdings,
        })

    # Most recent first
    periods.reverse()
    return periods


def _compute_trade_summary(holdings_history):
    """
    Aggregate all trades across all periods into a summary:
      - top_gainers: top 5 best individual trades
      - top_losers: top 5 worst individual trades
      - total_ops: total number of stock-period entries
      - total_periods: number of rebalance periods
      - win_count / loss_count
    """
    all_trades = []
    for period in holdings_history:
        for h in period.get('holdings', []):
            all_trades.append({
                'ticker': h['ticker'],
                'return_pct': h['return_pct'],
                'period': period['label'],
            })

    if not all_trades:
        return {'top_gainers': [], 'top_losers': [], 'total_ops': 0,
                'total_periods': 0, 'win_count': 0, 'loss_count': 0, 'win_rate': 0}

    sorted_trades = sorted(all_trades, key=lambda t: t['return_pct'], reverse=True)
    win_count = sum(1 for t in all_trades if t['return_pct'] > 0)
    loss_count = sum(1 for t in all_trades if t['return_pct'] <= 0)

    return {
        'top_gainers': sorted_trades[:5],
        'top_losers': sorted_trades[-5:][::-1],  # worst first
        'total_ops': len(all_trades),
        'total_periods': len(holdings_history),
        'win_count': win_count,
        'loss_count': loss_count,
        'win_rate': round(win_count / len(all_trades) * 100, 1) if all_trades else 0,
    }


def _log_backtest(run_type, config_data, result, extra=None):
    """Write a structured JSON log for every backtest run."""
    try:
        ts = datetime.now()
        short_id = uuid.uuid4().hex[:8]
        filename = f"backtest_{ts.strftime('%Y%m%d_%H%M%S')}_{short_id}.json"

        # Summarise equity curve (don't dump full daily data)
        strat_eq = result.get('strategy', {})
        eq_dates = sorted(strat_eq.keys()) if strat_eq else []

        log_entry = {
            'timestamp': ts.isoformat(),
            'run_type': run_type,
            'config': {
                'factors': config_data.get('factors', {}),
                'rebalance': config_data.get('rebalance', ''),
                'years': config_data.get('years', 0),
                'top_n': config_data.get('top_n', 0),
                'universe_mode': config_data.get('universe_mode', ''),
                'tickers': config_data.get('tickers', None),
            },
            'stats': result.get('stats', {}),
            'equity_summary': {
                'start_date': eq_dates[0] if eq_dates else None,
                'end_date': eq_dates[-1] if eq_dates else None,
                'start_value': strat_eq.get(eq_dates[0]) if eq_dates else None,
                'end_value': strat_eq.get(eq_dates[-1]) if eq_dates else None,
                'num_days': len(eq_dates),
            },
        }
        if extra:
            log_entry['extra'] = extra

        log_path = BACKTEST_LOG_DIR / filename
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Backtest log saved: {log_path}")
    except Exception as e:
        logger.warning(f"Failed to write backtest log: {e}")


# ==========================================
# Routes
# ==========================================

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/factors', methods=['GET'])
def get_factors():
    return jsonify(FACTOR_INFO)


@app.route('/api/alpaca/account', methods=['GET'])
def get_alpaca_account():
    if not trading_client:
        return jsonify({'error': 'Alpaca client not initialized'}), 503
    try:
        acct = trading_client.get_account()
        return jsonify({
            'equity': float(acct.equity),
            'cash': float(acct.cash),
            'buying_power': float(acct.buying_power),
            'day_trade_count': int(acct.daytrade_count),
            'status': acct.status,
            'currency': acct.currency
        })
    except Exception as e:
        logger.error(f"Alpaca Account Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alpaca/positions', methods=['GET'])
def get_alpaca_positions():
    if not trading_client:
        return jsonify({'error': 'Alpaca client not initialized'}), 503
    try:
        positions = trading_client.get_all_positions()
        res = []
        for p in positions:
            res.append({
                'symbol': p.symbol,
                'qty': float(p.qty),
                'market_value': float(p.market_value),
                'current_price': float(p.current_price),
                'avg_entry_price': float(p.avg_entry_price),
                'unrealized_pl': float(p.unrealized_pl),
                'unrealized_plpc': float(p.unrealized_plpc),
                'change_today': float(p.change_today)
            })
        return jsonify(res)
    except Exception as e:
        logger.error(f"Alpaca Positions Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alpaca/orders', methods=['GET'])
def get_alpaca_orders():
    if not trading_client:
        return jsonify({'error': 'Alpaca client not initialized'}), 503
    try:
        # Get open orders
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=50)
        orders = trading_client.get_orders(filter=req)
        
        # Get closed orders (recent)
        req_closed = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=20)
        closed_orders = trading_client.get_orders(filter=req_closed)
        
        def fmt_order(o):
            return {
                'id': str(o.id),
                'symbol': o.symbol,
                'qty': float(o.qty) if o.qty else 0,
                'notional': float(o.notional) if o.notional else 0,
                'side': o.side,
                'type': o.type,
                'time_in_force': o.time_in_force,
                'status': o.status,
                'created_at': o.created_at.isoformat(),
                'filled_avg_price': float(o.filled_avg_price) if o.filled_avg_price else None
            }
            
        return jsonify({
            'open': [fmt_order(o) for o in orders],
            'closed': [fmt_order(o) for o in closed_orders]
        })
    except Exception as e:
        logger.error(f"Alpaca Orders Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alpaca/history', methods=['GET'])
def get_alpaca_history():
    if not trading_client:
        return jsonify({'error': 'Alpaca client not initialized'}), 503
    try:
        # Get portfolio history (1M default)
        # Note: In a real app we might want more options
        hist = trading_client.get_portfolio_history(period="1M", timeframe="1D")
        
        # Format for Chart.js
        return jsonify({
            'timestamp': hist.timestamp, # list of unix timestamps
            'equity': hist.equity,       # list of equity values
            'profit_loss': hist.profit_loss,
            'profit_loss_pct': hist.profit_loss_pct
        })
    except Exception as e:
        logger.error(f"Alpaca History Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config/current', methods=['GET'])
def get_current_config():
    """Return the current active config (from file)"""
    try:
        file_path = Path(__file__).parent / 'config.yaml'
        with open(file_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return jsonify(cfg)
    except Exception as e:
        logger.error(f"Config Read Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    try:
        data = request.get_json()
        factor_config = data.get('factors', {})
        rebalance_key = data.get('rebalance', 'weekly_fri')
        rebalance_rule = REBALANCE_MAP.get(rebalance_key, 'W-FRI')
        years = data.get('years', 5)
        top_n = data.get('top_n', 10)
        tickers = data.get('tickers', None)
        universe_mode = data.get('universe_mode', 'sp500_nasdaq100')
        enable_mwu = data.get('enable_mwu', False)

        if tickers and isinstance(tickers, str):
            tickers = [t.strip().upper() for t in tickers.split(',') if t.strip()]
        if not tickers:
            tickers = None

        logger.info(f"Backtest request: factors={list(factor_config.keys())}, "
                    f"rebalance={rebalance_rule}, years={years}, top_n={top_n}, "
                    f"universe={universe_mode}, tickers={tickers}, mwu={enable_mwu}")

        result = _run_backtest_with_config(
            factor_config, rebalance_rule, years, top_n,
            tickers=tickers, universe_mode=universe_mode,
            enable_mwu=enable_mwu
        )

        # Log the backtest run
        _log_backtest('main', data, result)

        return jsonify({'status': 'ok', 'data': result})

    except Exception as e:
        logger.exception("Backtest failed")
        return jsonify({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/rolling', methods=['POST'])
def run_rolling():
    """Run rolling 1-year window backtests and return overlaid curves."""
    try:
        data = request.get_json()
        factor_config = data.get('factors', {})
        rebalance_key = data.get('rebalance', 'weekly_fri')
        rebalance_rule = REBALANCE_MAP.get(rebalance_key, 'W-FRI')
        total_years = data.get('years', 5)
        window_years = data.get('window_years', 1)
        top_n = data.get('top_n', 10)
        tickers = data.get('tickers', None)
        universe_mode = data.get('universe_mode', 'sp500_nasdaq100')

        if tickers and isinstance(tickers, str):
            tickers = [t.strip().upper() for t in tickers.split(',') if t.strip()]
        if not tickers:
            tickers = None

        windows = []
        now = datetime.now()

        # Lookback buffer: extra data before each window so factors can warm up
        # 378 calendar days ≈ 252 trading days * 1.5 for safety
        lookback_buffer_days = 378

        num_windows = total_years - window_years + 1
        logger.info(f"Rolling request: total_years={total_years}, window={window_years}yr, "
                    f"generating {num_windows} windows (with {lookback_buffer_days}d lookback buffer)")

        for start_year_offset in range(total_years - window_years, -1, -1):
            w_end_dt = now - timedelta(days=start_year_offset * 365)
            w_start_dt = w_end_dt - timedelta(days=window_years * 365)
            w_start = w_start_dt.strftime('%Y-%m-%d')
            w_end = w_end_dt.strftime('%Y-%m-%d')

            # Fetch extra data before window start for factor warm-up
            data_start_dt = w_start_dt - timedelta(days=lookback_buffer_days)
            data_start = data_start_dt.strftime('%Y-%m-%d')

            try:
                logger.info(f"  Rolling window: {w_start} to {w_end} (data from {data_start})")
                result = _run_backtest_with_config(
                    factor_config, rebalance_rule, window_years, top_n,
                    tickers=tickers, universe_mode=universe_mode,
                    start_date=w_start, end_date=w_end,
                    data_start_date=data_start
                )
                # Normalize to start at 1.0
                strat_data = result['strategy']
                if strat_data:
                    first_val = list(strat_data.values())[0]
                    if first_val > 0:
                        normalized = {k: round(v / first_val, 6) for k, v in strat_data.items()}
                    else:
                        normalized = strat_data
                else:
                    normalized = strat_data

                # Use month-level label: "2021.02-2022.02" for clarity
                label = f"{w_start[:7]}-{w_end[:7]}".replace('-', '.', 1).replace('-', '.', 1)
                # Simplify: "2021.02 ~ 2022.02"
                label = f"{w_start_dt.strftime('%Y.%m')} ~ {w_end_dt.strftime('%Y.%m')}"

                windows.append({
                    'label': label,
                    'start': w_start,
                    'end': w_end,
                    'equity': normalized,
                    'stats': result['stats']['strategy']
                })
            except Exception as e:
                logger.warning(f"Rolling window {w_start}-{w_end} failed: {e}")
                continue

        # Log rolling backtest
        _log_backtest('rolling', data, {
            'stats': {w['label']: w['stats'] for w in windows},
            'strategy': {},
        }, extra={
            'num_windows': len(windows),
            'window_years': window_years,
            'total_years': total_years,
            'windows_summary': [{'label': w['label'], 'stats': w['stats']} for w in windows]
        })

        return jsonify({'status': 'ok', 'windows': windows})

    except Exception as e:
        logger.exception("Rolling backtest failed")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/save-factor', methods=['POST'])
def save_factor():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        config = data.get('config', {})

        if not name:
            return jsonify({'status': 'error', 'message': 'Name is required'}), 400

        saved = _load_json(FACTORS_FILE)
        saved[name] = {
            'config': config,
            'saved_at': datetime.now().isoformat()
        }
        _save_json(FACTORS_FILE, saved)

        return jsonify({'status': 'ok', 'message': f'Factor config "{name}" saved'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/save-strategy', methods=['POST'])
def save_strategy():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        strategy = data.get('strategy', {})

        if not name:
            return jsonify({'status': 'error', 'message': 'Name is required'}), 400

        saved = _load_json(STRATEGIES_FILE)
        saved[name] = {
            'strategy': strategy,
            'saved_at': datetime.now().isoformat()
        }
        _save_json(STRATEGIES_FILE, saved)

        return jsonify({'status': 'ok', 'message': f'Strategy "{name}" saved'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/saved', methods=['GET'])
def get_saved():
    factors = _load_json(FACTORS_FILE)
    strategies = _load_json(STRATEGIES_FILE)
    return jsonify({'factors': factors, 'strategies': strategies})


@app.route('/api/delete-saved', methods=['POST'])
def delete_saved():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        save_type = data.get('type', 'factor')

        if save_type == 'factor':
            saved = _load_json(FACTORS_FILE)
            saved.pop(name, None)
            _save_json(FACTORS_FILE, saved)
        else:
            saved = _load_json(STRATEGIES_FILE)
            saved.pop(name, None)
            _save_json(STRATEGIES_FILE, saved)

        return jsonify({'status': 'ok', 'message': f'Deleted "{name}"'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ==========================================
# Optimizer
# ==========================================

def _save_optimizer_strategy(name, strategy_dict):
    """Save an optimizer-generated strategy to the strategies file."""
    saved = _load_json(STRATEGIES_FILE)
    # Remove old optimizer entries first
    saved = {k: v for k, v in saved.items() if not k.startswith('⚡OPT')}
    saved[name] = strategy_dict
    _save_json(STRATEGIES_FILE, saved)


_optimizer = FactorOptimizer(
    run_backtest_fn=_run_backtest_with_config,
    save_strategy_fn=_save_optimizer_strategy
)


@app.route('/optimizer')
def optimizer_page():
    return send_from_directory('static', 'optimizer.html')


@app.route('/api/optimizer/start', methods=['POST'])
def optimizer_start():
    try:
        data = request.get_json() or {}
        n_iter = int(data.get('n_iterations', 50))
        min_f = int(data.get('min_factors', 1))
        max_f = int(data.get('max_factors', 5))
        universe = data.get('universe_mode', 'sp500_nasdaq100')
        tickers_str = data.get('tickers', '')
        tickers = [t.strip() for t in tickers_str.split(',') if t.strip()] if tickers_str else None

        _optimizer.start(
            n_iterations=n_iter,
            min_factors=min_f,
            max_factors=max_f,
            universe_mode=universe,
            tickers=tickers
        )
        return jsonify({'status': 'ok', 'message': f'Started {n_iter} iterations'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/optimizer/status')
def optimizer_status():
    return jsonify(_optimizer.get_status())


@app.route('/api/optimizer/stop', methods=['POST'])
def optimizer_stop():
    _optimizer.stop()
    return jsonify({'status': 'ok', 'message': 'Stopping...'})


# ==========================================
# Main
# ==========================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Factor Analysis Web Dashboard")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
