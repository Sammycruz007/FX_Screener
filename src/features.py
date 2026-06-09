# src/features.py
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("SystemLogger.Features")

def compute_linear_regression_channel(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """
    Computes a strict 200-candle rolling linear regression channel center line and slope.
    Bands are generated using a localized volatility multiplier to match visual charting.
    """
    logger.info(f"Computing rolling Linear Regression Channels using a strict {window}-period frame...")
    
    length = len(df)
    lrc_center = np.full(length, np.nan)
    lrc_slope = np.full(length, np.nan)
    
    x = np.arange(window)
    x_mean = x.mean()
    x_dev = x - x_mean
    x_var = (x_dev ** 2).sum()

    for i in range(window - 1, length):
        # Strict 200-candle selection: newest candle is always the 200th item
        y_window = df['close'].values[i - window + 1 : i + 1]
        y_mean = y_window.mean()
        
        slope = np.dot(x_dev, y_window - y_mean) / x_var
        intercept = y_mean - slope * x_mean
        
        lrc_center[i] = slope * (window - 1) + intercept
        lrc_slope[i] = slope

    df['lrc_center'] = lrc_center
    df['lrc_slope'] = lrc_slope
    
    # LOCALIZED CALIBRATION: Use a rolling short-term volatility window to keep bands tight to price action
    # This matches real-world linear regression channels like LuxAlgo
    rolling_volatility = df['close'].rolling(20).std()
    
    df['lrc_p1'] = df['lrc_center'] + (1.0 * rolling_volatility)
    df['lrc_p3'] = df['lrc_center'] + (2.5 * rolling_volatility)
    
    return df

def identify_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scans the data for Forex-optimized Bearish Engulfing and Evening Star formations.
    """
    logger.info("Scanning price action for Forex-calibrated candlestick formations...")
    
    df['is_bearish_engulfing'] = False
    df['is_evening_star'] = False
    
    for i in range(3, len(df)):
        c0_open, c0_close, c0_high = df.loc[i, 'open'], df.loc[i, 'close'], df.loc[i, 'high']
        c1_open, c1_close, c1_high = df.loc[i-1, 'open'], df.loc[i-1, 'close'], df.loc[i-1, 'high']
        c2_open, c2_close, c2_high = df.loc[i-2, 'open'], df.loc[i-2, 'close'], df.loc[i-2, 'high']
        
        c0_body = abs(c0_close - c0_open)
        c1_body = abs(c1_close - c1_open)
        c2_body = abs(c2_close - c2_open)
        
        # Bearish Engulfing
        c1_is_green = c1_close > c1_open
        c0_is_red = c0_close < c0_open
        if c1_is_green and c0_is_red and (c0_body > c1_body) and (c0_close < c1_open):
            df.loc[i, 'is_bearish_engulfing'] = True
                
        # Evening Star
        c2_is_green = c2_close > c2_open
        if c2_is_green and c0_is_red:
            is_star_body = c1_body < (c2_body * 0.5)
            is_star_peak = (c1_high > c2_high) and (c1_high >= c0_high)
            closes_deep = c0_close <= (c2_open + c2_close) / 2
            if is_star_body and is_star_peak and closes_deep:
                df.loc[i, 'is_evening_star'] = True
                
    return df