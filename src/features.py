# src/features_ml.py
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("SystemLogger.MLFeatures")

def extract_scale_invariant_features(df: pd.DataFrame, atr_window: int = 14) -> pd.DataFrame:
    """
    Transforms raw OHLCV pricing columns into normalized, scale-invariant 
    features optimized for time-series ML classification.
    """
    if len(df) < atr_window + 1:
        logger.error("Dataset too short to calculate ATR features!")
        return df

    logger.info("Extracting scale-invariant candlestick geometries and momentum features...")
    
    try:
        # 1. Compute True Range (TR) and Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close_prev = (df['high'] - df['close'].shift(1)).abs()
        low_close_prev = (df['low'] - df['close'].shift(1)).abs()
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=atr_window).mean()
        
        # Handle edge case: replace zeroes or NaNs in ATR to prevent division errors
        df['atr'] = df['atr'].replace(0, np.nan).bfill()
        
        # 2. Extract Normalized Candle Geometries
        max_open_close = df[['open', 'close']].max(axis=1)
        min_open_close = df[['open', 'close']].min(axis=1)
        
        df['feat_upper_wick_rel'] = (df['high'] - max_open_close) / df['atr']
        df['feat_lower_wick_rel'] = (min_open_close - df['low']) / df['atr']
        df['feat_body_rel'] = (df['close'] - df['open']).abs() / df['atr']
        
        # 3. Contextual Location Features (Z-Score relative to the channel)
        df['feat_lrc_zscore'] = (df['close'] - df['lrc_center']) / df['lrc_std']
        df['feat_lrc_slope_norm'] = df['lrc_slope'] / df['close'] # Normalized by price to make it scale-invariant
        
        # 4. Lagged Historical Features (Lookback momentum over the last 2 days)
        for lag in [1, 2]:
            df[f'feat_upper_wick_rel_lag_{lag}'] = df['feat_upper_wick_rel'].shift(lag)
            df[f'feat_lower_wick_rel_lag_{lag}'] = df['feat_lower_wick_rel'].shift(lag)
            df[f'feat_body_rel_lag_{lag}'] = df['feat_body_rel'].shift(lag)
            
        # Drop temporary math columns to keep the dataset clean
        df = df.dropna(subset=['feat_upper_wick_rel_lag_2', 'target_label']).reset_index(drop=True)
        
        logger.info(f"Feature engineering complete. Prepared matrix shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.exception(f"Fatal error during ML feature extraction: {str(e)}")
        raise