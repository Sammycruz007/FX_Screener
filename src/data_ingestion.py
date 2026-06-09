# src/data_ingestion.py
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical.forex import ForexHistoricalDataClient
from alpaca.data.requests import ForexBarsRequest
from alpaca.data.timeframe import TimeFrame

# Configure structured logging for the data module
logger = logging.getLogger("SystemLogger.DataIngestion")

class AlpacaDataIngestor:
    """Handles secure connection and raw daily bar data ingestion from Alpaca."""
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
            logger.critical("Alpaca API Credentials missing from system environment (.env)!")
            raise ValueError("Credentials missing. Ensure ALPACA_API_KEY and ALPACA_SECRET_KEY are set.")
        
        try:
            self.client = ForexHistoricalDataClient(api_key=self.api_key, secret_key=self.secret_key)
            logger.info("Successfully authenticated and initialized Alpaca Forex client.")
        except Exception as e:
            logger.exception("Failed to initialize Alpaca historical client wrapper:")
            raise

    def fetch_daily_candles(self, pair: str, days_back: int = 1000) -> pd.DataFrame:
        """
        Retrieves historical D1 candlestick arrays for a targeted currency pair.
        Ex: pair='EUR/USD'
        """
        logger.info(f"Initiating historical fetch for {pair} looking back {days_back} days...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        try:
            request_params = ForexBarsRequest(
                symbol_or_symbols=pair,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            # Fetch raw structural objects from the endpoint
            bars = self.client.get_forex_bars(request_params)
            
            if not bars or not bars.df.empty:
                df = bars.df.reset_index()
                logger.info(f"Ingestion successful for {pair}. Retrieved {len(df)} total records.")
                return df
            else:
                logger.warning(f"API request completed but returned an empty dataset for {pair}!")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to fetch market data from Alpaca endpoints for target {pair}: {str(e)}")
            logger.exception(e)
            return pd.DataFrame()