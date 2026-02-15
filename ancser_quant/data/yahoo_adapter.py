import yfinance as yf
import polars as pl
from .schema import MARKET_DATA_SCHEMA
from datetime import datetime
from typing import List

class YahooAdapter:
    """
    Fetches historical data from Yahoo Finance and converts to Polars LazyFrame.
    Primarily used for Backfill.
    """
    def fetch_history(self, symbols: List[str], start_date: str, end_date: str = None) -> pl.LazyFrame:
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"[YahooAdapter] Fetching {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Download using yfinance (Pandas based)
        # We download individually to avoid MultiIndex complexity with Polars for now,
        # or download all and process.
        
        # Optimized: Group download
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
            return pl.LazyFrame({})
        
        if not frames:
            return pl.LazyFrame({})
            
        # Concat all Polars DataFrames
        df_pl = pl.concat(frames)

        # Rename columns to match Schema (case insensitive check)
        # Yahoo: Date, Open, High, Low, Close, Volume
        
        df_pl = df_pl.rename({
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })

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

        return df_pl.lazy()
