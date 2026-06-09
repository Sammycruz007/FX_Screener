# src/labeling.py
import logging
import pandas as pd

logger = logging.getLogger("SystemLogger.Labeling")

def apply_triple_barrier_labels(df: pd.DataFrame, max_holding_days: int = 20) -> pd.DataFrame:
    """
    Scans future price behaviors to extract target labels for machine learning:
    1 = Success (Mean reversion back to center OLS line completed)
    0 = Failure (Price broke above +3 SD line; validation breached)
    2 = Timeout (Price drifted sideways without hitting targets within deadline)
    """
    if 'short_setup' not in df.columns:
        logger.error("Dataframe missing the 'short_setup' condition flag. Cannot apply labels.")
        return df
        
    logger.info(f"Applying future-window Triple-Barrier targets (Max holding cap: {max_holding_days} days)...")
    
    try:
        labels = []
        total_rows = len(df)
        
        for idx in range(total_rows):
            if not df.loc[idx, 'short_setup']:
                labels.append(None)
                continue
                
            # Snapshot the localized targets at entry time
            stop_loss = df.loc[idx, 'lrc_p3']
            take_profit = df.loc[idx, 'lrc_center']
            
            # Constrain window bounds safely to avoid array clipping
            horizon_end = min(idx + max_holding_days, total_rows - 1)
            future_data = df.loc[idx + 1 : horizon_end]
            
            assigned_label = 2  # Initialize as timeout baseline
            
            for _, future_row in future_data.iterrows():
                if future_row['high'] >= stop_loss:
                    assigned_label = 0  # Invalidated/Stopped
                    break
                if future_row['low'] <= take_profit:
                    assigned_label = 1  # Mean Reversion Success
                    break
                    
            labels.append(assigned_label)
            
        df['target_label'] = labels
        logger.info("Target variable calculations finalized successfully.")
        return df
        
    except Exception as e:
        logger.exception(f"Fatal crash processing forward labeling: {str(e)}")
        raise