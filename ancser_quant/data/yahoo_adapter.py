import yfinance as yf
import polars as pl
from .schema import MARKET_DATA_SCHEMA
from datetime import datetime
from typing import List
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

class YahooAdapter:
    """
    Fetches historical data from Yahoo Finance and converts to Polars LazyFrame.
    Primarily used for Backfill.
    Supports caching and parallel downloads.
    """
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, symbols: List[str], start_date: str, end_date: str) -> str:
        """Generate cache key from symbols and dates"""
        symbols_str = "_".join(sorted(symbols))
        key_str = f"{symbols_str}_{start_date}_{end_date}"
        # Use hash if too long
        if len(key_str) > 200:
            return hashlib.md5(key_str.encode()).hexdigest()
        return key_str.replace(":", "").replace("/", "-")

    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{cache_key}.parquet")

    def _load_from_cache(self, cache_path: str) -> pl.LazyFrame:
        """Load data from cache if exists"""
        if os.path.exists(cache_path):
            try:
                print(f"[Cache] Loading from {os.path.basename(cache_path)}")
                return pl.scan_parquet(cache_path)
            except Exception as e:
                print(f"[Cache] Failed to load: {e}")
        return None

    def _save_to_cache(self, df: pl.DataFrame, cache_path: str):
        """Save data to cache"""
        try:
            df.write_parquet(cache_path)
            print(f"[Cache] Saved to {os.path.basename(cache_path)}")
        except Exception as e:
            print(f"[Cache] Failed to save: {e}")

    def fetch_history(self, symbols: List[str], start_date: str, end_date: str = None) -> pl.LazyFrame:
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # Check cache first
        cache_key = self._get_cache_key(symbols, start_date, end_date)
        cache_path = self._get_cache_path(cache_key)
        cached_data = self._load_from_cache(cache_path)

        if cached_data is not None:
            return cached_data

        print(f"[YahooAdapter] Fetching {len(symbols)} symbols from {start_date} to {end_date}")

        # Download using yfinance with parallel threads
        try:
            df_pandas = yf.download(
                symbols, 
                start=start_date, 
                end=end_date, 
                group_by='ticker', 
                auto_adjust=True,
                threads=True,
                progress=False
            )
        except Exception as e:
            print(f"[YahooAdapter] Download failed: {e}")
            return pl.LazyFrame({}) # Empty

        # Process into long format
        # Pandas MultiIndex (Ticker, Attributes) -> Ticker column
        
        # Helper function to align schema before concat
        def align_schema(pdf):
            # Ensure columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'symbol']
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col not in pdf.columns:
                    pdf[col] = 0.0
            
            # Convert critical columns to consistent types
            pdf['Volume'] = pdf['Volume'].fillna(0).astype(float)
            pdf['Open'] = pdf['Open'].astype(float)
            pdf['High'] = pdf['High'].astype(float)
            pdf['Low'] = pdf['Low'].astype(float)
            pdf['Close'] = pdf['Close'].astype(float)
            
            # Strict Column Selection to avoid ShapeError (7 vs 8 cols)
            # Some tickers might return 'Adj Close', 'Dividends', etc.
            return pdf[required_cols]

        frames = []
        if len(symbols) == 1:
             # Single ticker structure is flat
             sym = symbols[0]
             df_pandas['symbol'] = sym
             df_pandas = align_schema(df_pandas)
             frames.append(pl.from_pandas(df_pandas.reset_index()))
        else: 
            # Multi-ticker
            # If MultiIndex (Ticker, Attributes), yfinance structure varies by version.
            for sym in symbols:
                try:
                    # Check if symbol exists in columns (Top level)
                    if sym not in df_pandas.columns:
                        continue
                        
                    subset = df_pandas[sym].copy()
                    subset['symbol'] = sym
                    subset = align_schema(subset)
                    
                    # Reset index to get Date column
                    subset = subset.reset_index()
                    frames.append(pl.from_pandas(subset))
                except KeyError:
                    pass # Symbol not found in data
                except Exception as e:
                    print(f"Error processing {sym}: {e}")
        
        if not frames:
            print("No valid data frames collected")
            return pl.LazyFrame({})

        # Concat all Polars DataFrames
        df_pl = pl.concat(frames)

        # Check if dataframe is empty
        if df_pl.height == 0:
            print("Concatenated dataframe is empty")
            return pl.LazyFrame({})

        # Rename columns to match Schema (check if columns exist first)
        # Yahoo: Date, Open, High, Low, Close, Volume
        current_columns = df_pl.columns

        rename_map = {}
        if "Date" in current_columns:
            rename_map["Date"] = "timestamp"
        if "Open" in current_columns:
            rename_map["Open"] = "open"
        if "High" in current_columns:
            rename_map["High"] = "high"
        if "Low" in current_columns:
            rename_map["Low"] = "low"
        if "Close" in current_columns:
            rename_map["Close"] = "close"
        if "Volume" in current_columns:
            rename_map["Volume"] = "volume"

        if rename_map:
            df_pl = df_pl.rename(rename_map)
        else:
            print(f"Warning: No standard Yahoo columns found. Available: {current_columns}")
            return pl.LazyFrame({})

        # Feature Engineering for missing columns
        # VWAP approx = (High + Low + Close) / 3
        # Trade Count = 0 (Yahoo doesn't provide)
        
        df_pl = df_pl.with_columns([
            ( (pl.col("high") + pl.col("low") + pl.col("close")) / 3 ).alias("vwap"),
            pl.lit(0).cast(pl.UInt32).alias("trade_count")
        ])
        
        # Enforce Types
        df_pl = df_pl.with_columns([
            pl.col("timestamp").cast(pl.Datetime),
            pl.col("symbol").cast(pl.Categorical),
            pl.col("open").cast(pl.Float32),
            pl.col("high").cast(pl.Float32),
            pl.col("low").cast(pl.Float32),
            pl.col("close").cast(pl.Float32),
            pl.col("volume").cast(pl.Float32),
            pl.col("vwap").cast(pl.Float32),
            pl.col("trade_count").cast(pl.UInt32)
        ])
        
        # Filter 0 volume
        df_pl = df_pl.filter(pl.col("volume") > 0)
        
        # Sort
        df_pl = df_pl.sort(["symbol", "timestamp"])

        # Save to cache
        self._save_to_cache(df_pl, cache_path)

        return df_pl.lazy()
