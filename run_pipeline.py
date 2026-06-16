import yfinance as yf
import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime

# Suppress yfinance warnings for clean terminal output
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
LOOKBACK_WINDOW = 400
PERIOD = "1mo"
INTERVAL = "1h"

# Multi-Asset Forex Matrix
ASSETS = [
    'EURUSD=X', 'AUDUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'USDCAD=X', 
    'AUDCAD=X', 'EURCAD=X', 'GBPCAD=X', 'NZDCAD=X', 'AUDCHF=X', 'EURCHF=X', 
    'GBPCHF=X', 'NZDCHF=X', 'CADCHF=X', 'AUDJPY=X', 'EURJPY=X', 'GBPJPY=X', 
    'NZDJPY=X', 'CADJPY=X', 'EURAUD=X', 'GBPAUD=X', 'EURNZD=X', 'GBPNZD=X', 
    'AUDNZD=X'
]

# Display Names for formatting
DISPLAY_NAMES = {ticker: ticker.replace('=X', '').replace('USD', '/USD').replace('CAD', '/CAD').replace('CHF', '/CHF').replace('JPY', '/JPY').replace('AUD', '/AUD').replace('NZD', '/NZD').replace('EUR', 'EUR/').replace('GBP', 'GBP/').replace('AUD/', 'AUD').replace('NZD/', 'NZD') for ticker in ASSETS}
# A quick clean up for display logic
for k, v in DISPLAY_NAMES.items():
    if len(v) == 6: DISPLAY_NAMES[k] = f"{v[:3]}/{v[3:]}"

# ==========================================
# PRODUCTION LOGGER SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger_main = logging.getLogger("SystemLogger.Main")
logger_data = logging.getLogger("SystemLogger.DataIngestion")

# ==========================================
# ENGINE FUNCTIONS
# ==========================================

def fetch_data(ticker):
    """Downloads H1 data, flattens index, and drops the live unclosed candle."""
    logger_data.info(f"Downloading {ticker} (Attempt 1/3)...")
    
    # 1. Force the Hourly interval and cap the period to 1 month
    data = yf.download(
        tickers=ticker,
        period=PERIOD,
        interval=INTERVAL,
        progress=False
    )
    
    if data.empty:
        logger_data.warning(f"Failed to fetch data for {ticker}.")
        return None

    # 2. Safety Net: Flatten Yahoo Finance Multi-Index columns if they occur
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # 3. Slice off the active, unfinalized hour
    logger_data.info("Detected live, unclosed hourly candle row. Slicing it off safely.")
    df_finalized = data.iloc[:-1].copy()
    
    bars_processed = len(df_finalized)
    logger_data.info(f"Successfully processed {bars_processed} finalized bars for {DISPLAY_NAMES[ticker]}.")
    
    return df_finalized

def calculate_linreg_channels(df):
    """Calculates the 200-period Linear Regression Slope and Standard Deviation Bands."""
    if len(df) < LOOKBACK_WINDOW:
        return None, None, None, None, None

    # Get the last 200 closing prices
    closes = df['Close'].iloc[-LOOKBACK_WINDOW:].values
    x = np.arange(len(closes))
    
    # Linear Regression Formula: y = mx + c
    slope, intercept = np.polyfit(x, closes, 1)
    
    # Calculate regression line values
    reg_line = (slope * x) + intercept
    
    # Calculate standard deviation of residuals (volatility bands)
    residuals = closes - reg_line
    std_dev = np.std(residuals)
    
    current_close = closes[-1]
    current_mean = reg_line[-1]
    
    # Define bounds based on the final point of the regression line
    plus_2_std = current_mean + (2 * std_dev)
    minus_2_std = current_mean - (2 * std_dev)
    
    return current_close, slope, current_mean, plus_2_std, minus_2_std

def detect_price_action(df):
    """Checks the most recently finalized H1 bars for a reversal pattern."""
    prev_candle = df.iloc[-2]
    curr_candle = df.iloc[-1]
    
    # Basic Bullish Engulfing
    bullish_engulfing = (
        prev_candle['Close'] < prev_candle['Open'] and # Prev was red
        curr_candle['Close'] > curr_candle['Open'] and # Curr is green
        curr_candle['Open'] <= prev_candle['Close'] and 
        curr_candle['Close'] > prev_candle['Open']
    )
    
    # Basic Bearish Engulfing
    bearish_engulfing = (
        prev_candle['Close'] > prev_candle['Open'] and # Prev was green
        curr_candle['Close'] < curr_candle['Open'] and # Curr is red
        curr_candle['Open'] >= prev_candle['Close'] and 
        curr_candle['Close'] < prev_candle['Open']
    )
    
    if bullish_engulfing: return "BULLISH_REVERSAL"
    if bearish_engulfing: return "BEARISH_REVERSAL"
    return "NONE"

def evaluate_setup(ticker, df):
    """Runs the 3-stage confluence validation on the asset."""
    close_price, slope, mean_val, upper_band, lower_band = calculate_linreg_channels(df)
    
    if close_price is None:
        return None

    pa_signal = detect_price_action(df)
    
    # 1. Determine Macro Regime
    regime = "📈 UP" if slope > 0 else "📉 DOWN"
    
    # 2. Determine Structural Zone (Volatility Stretch)
    zone_status = "EQUILIBRIUM"
    if close_price >= upper_band:
        zone_status = "PREMIUM" # Overbought (+2 StdDev)
    elif close_price <= lower_band:
        zone_status = "DISCOUNT" # Oversold (-2 StdDev)

    # 3. Strategy Logic Validation
    strategy_signal = "💤 IDLE"
    
    if regime == "📈 UP":
        if zone_status == "PREMIUM":
            strategy_signal = "❌ MISMATCH (UP)"
        elif zone_status == "DISCOUNT":
            if pa_signal == "BULLISH_REVERSAL":
                strategy_signal = "🔥 BUY TRIGGER"
            else:
                strategy_signal = "⏳ WATCHING BUY"
                
    elif regime == "📉 DOWN":
        if zone_status == "DISCOUNT":
            strategy_signal = "❌ MISMATCH (DOWN)"
        elif zone_status == "PREMIUM":
            if pa_signal == "BEARISH_REVERSAL":
                strategy_signal = "🔥 SELL TRIGGER"
            else:
                strategy_signal = "⏳ WATCHING SELL"

    return {
        "ASSET": DISPLAY_NAMES.get(ticker, ticker),
        "CLOSE": close_price,
        "SLOPE": slope,
        "REGIME": regime,
        "ZONE": zone_status,
        "SIGNAL": strategy_signal
    }

# ==========================================
# MAIN EXECUTION THREAD
# ==========================================
def main():
    logger_main.info("Initializing Multi-Asset Production Driver...")
    logger_main.info("Invoking data extraction via: fetch_data")
    logger_data.info(f"Initiating yfinance data download for pairs: {list(DISPLAY_NAMES.values())}")
    
    results = []
    
    for ticker in ASSETS:
        df = fetch_data(ticker)
        if df is not None:
            setup = evaluate_setup(ticker, df)
            if setup:
                results.append(setup)

    # Output Production Matrix
    print("\n" + "="*95)
    print("                          MLOps MULTI-ASSET LIVE PRODUCTION SCREENER                           ")
    print("="*95)
    print(f"{'ASSET':<12} | {'CLOSE':<10} | {'SLOPE (200)':<13} | {'REGIME':<12} | {'ZONE STATUS':<14} | {'STRATEGY SIGNAL'}")
    print("-" * 95)
    
    for r in results:
        close_fmt = f"{r['CLOSE']:.5f}"
        slope_fmt = f"{r['SLOPE']:.7f}"
        print(f"{r['ASSET']:<12} | {close_fmt:<10} | {slope_fmt:>13} | {r['REGIME']:<12} | {r['ZONE']:<14} | {r['SIGNAL']}")
    
    print("="*95)

if __name__ == "__main__":
    main()