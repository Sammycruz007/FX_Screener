# src/data_ingestion.py
import logging
import time
import yfinance as yf
import pandas as pd

logger = logging.getLogger("SystemLogger.DataIngestion")

class YFinanceDataIngestor:
    def __init__(self, pairs: list, period: str = "5y", interval: str = "1d"):
        self.pairs = pairs
        self.period = period
        self.interval = interval

    def fetch_data(self) -> dict:
        logger.info(f"Initiating yfinance data download for pairs: {self.pairs}")
        data_dict = {}
        
        for pair in self.pairs:
            yf_symbol = pair.replace("/", "") + "=X"
            df = pd.DataFrame()
            
            # --- PRODUCTION RETRY ENGINE ---
            max_retries = 3
            backoff_seconds = 2
            
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"Downloading {yf_symbol} (Attempt {attempt}/{max_retries})...")
                    ticker = yf.Ticker(yf_symbol)
                    df = ticker.history(period=self.period, interval=self.interval)
                    
                    if not df.empty:
                        break # Success! Break out of the retry loop
                        
                except Exception as e:
                    logger.warning(f"Attempt {attempt} failed due to network hiccup: {str(e)}")
                
                if attempt < max_retries:
                    time.sleep(backoff_seconds * attempt) # Wait longer with each failure
            # --------------------------------
            
            if df.empty:
                logger.error(f"No data returned for {yf_symbol} after {max_retries} attempts. API rate limit breached.")
                continue
                
            # Standardize and clean columns
            df = df.reset_index()
            df.columns = [col.lower() for col in df.columns]
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            
            # Defensive Slice: Drop today's incomplete candle if it's still printing
            today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
            df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
            if df.iloc[-1]['date_str'] == today_str:
                logger.info("Detected live, unclosed daily candle row. Slicing it off safely.")
                df = df.iloc[:-1].reset_index(drop=True)
            df = df.drop(columns=['date_str'])
            
            data_dict[pair] = df
            logger.info(f"Successfully processed {len(df)} finalized bars for {pair}.")
                
        return data_dict