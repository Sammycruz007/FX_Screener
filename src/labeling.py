# src/labeling.py
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("SystemLogger.Labeling")

def apply_price_action_double_barrier(df: pd.DataFrame, zone_lookback: int = 3) -> pd.DataFrame:
    """
    Executes trades when a pattern prints near the premium zone under a 
    downward or flat macro structural regime.
    """
    logger.info(f"Executing calibrated location engine ({zone_lookback}-candle lookback + 10-pip grace zone)...")
    
    df['target_label'] = np.nan
    df['executed_trade'] = False
    df['pattern_sl_level'] = np.nan
    
    length = len(df)
    skip_until_idx = -1
    
    # Forex pip configuration buffers
    PIP_GRACE_BUFFER = 0.0010  # 10 pips tolerance window for zone touches
    MAX_SLOPE_THRESHOLD = 0.00002  # Allows flat/turning channels to qualify
    
    for idx in range(200, length - 1):
        if idx <= skip_until_idx:
            continue
            
        # 1. Calibrated Macro Filter (Down-sloping or rolling over flat)
        macro_regime_valid = df.loc[idx, 'lrc_slope'] <= MAX_SLOPE_THRESHOLD
        
        # 2. Calibrated Context Memory (Touched or came within 10 pips of the +1σ band)
        start_lookback = max(200, idx - zone_lookback + 1)
        
        zone_touched_recently = False
        for lookback_idx in range(start_lookback, idx + 1):
            high_price = df.loc[lookback_idx, 'high']
            p1_band = df.loc[lookback_idx, 'lrc_p1']
            
            # If high physically crossed OR came within 10 pips of the band
            if high_price >= (p1_band - PIP_GRACE_BUFFER):
                zone_touched_recently = True
                break
        
        # 3. Micro Trigger Filter
        has_pattern = df.loc[idx, 'is_bearish_engulfing'] or df.loc[idx, 'is_evening_star']
        
        if macro_regime_valid and zone_touched_recently and has_pattern:
            entry_idx = idx + 1
            df.loc[entry_idx, 'executed_trade'] = True
            
            # Determine stop loss and target levels
            if df.loc[idx, 'is_evening_star']:
                highest_wick = max(df.loc[idx, 'high'], df.loc[idx-1, 'high'], df.loc[idx-2, 'high'])
            else:
                highest_wick = max(df.loc[idx, 'high'], df.loc[idx-1, 'high'])
                
            static_stop_loss = highest_wick + 0.0002  # 2 pip breathing buffer
            static_take_profit = df.loc[idx, 'lrc_center']
            
            df.loc[entry_idx, 'pattern_sl_level'] = static_stop_loss
            
            # Trace trade resolution path
            resolved = False
            for future_idx in range(entry_idx, length):
                future_high = df.loc[future_idx, 'high']
                future_low = df.loc[future_idx, 'low']
                
                if future_high > static_stop_loss:
                    df.loc[entry_idx, 'target_label'] = 0.0
                    skip_until_idx = future_idx
                    resolved = True
                    break
                    
                elif future_low <= static_take_profit:
                    df.loc[entry_idx, 'target_label'] = 1.0
                    skip_until_idx = future_idx
                    resolved = True
                    break
                    
            if not resolved:
                skip_until_idx = length

    logger.info("Price action trade confirmation sequence finalized.")
    return df