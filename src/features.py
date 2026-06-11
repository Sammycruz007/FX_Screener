# src/features.py
import pandas as pd
import numpy as np

def compute_linear_regression_channel(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """
    Matches LINREG by DGT exactly.
    Uses 1-to-200 indexing and Standard Error of the Estimate with 198 Degrees of Freedom.
    """
    length = len(df)
    lrc_center = np.full(length, np.nan)
    lrc_slope = np.full(length, np.nan)
    
    lrc_p1 = np.full(length, np.nan)
    lrc_p2 = np.full(length, np.nan)
    lrc_p3 = np.full(length, np.nan)
    
    lrc_n1 = np.full(length, np.nan)
    lrc_n2 = np.full(length, np.nan)
    lrc_n3 = np.full(length, np.nan)
    
    # 1-Based Data Indexing (1 to 200)
    x = np.arange(1, window + 1)
    x_sum = x.sum()
    x_sq_sum = (x ** 2).sum()
    denominator = (window * x_sq_sum) - (x_sum ** 2)

    for i in range(window - 1, length):
        y_window = df['close'].values[i - window + 1 : i + 1]
        y_sum = y_window.sum()
        xy_sum = np.dot(x, y_window)
        
        # OLS Slope (m) & Intercept (b)
        m = (window * xy_sum - x_sum * y_sum) / denominator
        b = (y_sum - m * x_sum) / window
        
        # Plotted center baseline value at current candle index (x = 200)
        current_center = (m * window) + b
        lrc_center[i] = current_center
        lrc_slope[i] = m
        
        # Standard Error of the Estimate (SE)
        fitted_line = (m * x) + b
        residuals = y_window - fitted_line
        sum_squared_residuals = (residuals ** 2).sum()
        
        # Degrees of Freedom correction (n - 2 = 198)
        se = np.sqrt(sum_squared_residuals / (window - 2))
        
        # Plotted Parallel Channel Bands
        lrc_p1[i] = current_center + (1 * se)
        lrc_p2[i] = current_center + (2 * se)
        lrc_p3[i] = current_center + (3 * se)
        
        lrc_n1[i] = current_center - (1 * se)
        lrc_n2[i] = current_center - (2 * se)
        lrc_n3[i] = current_center - (3 * se)

    df['lrc_center'] = lrc_center
    df['lrc_slope'] = lrc_slope
    df['lrc_p1'] = lrc_p1
    df['lrc_p2'] = lrc_p2
    df['lrc_p3'] = lrc_p3
    df['lrc_n1'] = lrc_n1
    df['lrc_n2'] = lrc_n2
    df['lrc_n3'] = lrc_n3
    
    return df

def identify_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detects entry candlestick triggers."""
    df['is_bearish_engulfing'] = (df['close'].shift(1) > df['open'].shift(1)) & \
                                 (df['close'] < df['open']) & \
                                 (df['open'] >= df['close'].shift(1)) & \
                                 (df['close'] <= df['open'].shift(1))
                                 
    df['is_bullish_engulfing'] = (df['close'].shift(1) < df['open'].shift(1)) & \
                                 (df['close'] > df['open']) & \
                                 (df['open'] <= df['close'].shift(1)) & \
                                 (df['close'] >= df['open'].shift(1))
    
    body = abs(df['close'] - df['open'])
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    
    df['is_shooting_star'] = (upper_wick >= 2 * body) & (lower_wick <= body * 0.5)
    df['is_hammer'] = (lower_wick >= 2 * body) & (upper_wick <= body * 0.5)
    return df