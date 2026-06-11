# src/labeling.py
import pandas as pd
import numpy as np

def apply_final_convergence_model(df: pd.DataFrame, zone_lookback: int = 2) -> pd.DataFrame:
    """
    Enforces strict direction matching based on the synchronized 200-count DGT geometry.
    - BUY ONLY when slope >= 0 and price hits the discount zone (N1 to N3) below center.
    - SELL ONLY when slope <= 0 and price hits the premium zone (P1 to P3) above center.
    """
    df['target_label'] = np.nan
    df['executed_trade'] = False
    length = len(df)
    skip = -1
    
    for idx in range(200, length - 1):
        if idx <= skip: 
            continue
        
        # Look back to see if price entered the structured bands
        side = None
        for i in range(idx - zone_lookback + 1, idx + 1):
            if (df.loc[i, 'high'] >= df.loc[i, 'lrc_p1']) and (df.loc[i, 'low'] <= df.loc[i, 'lrc_p3']): 
                side = 'sell'
            elif (df.loc[i, 'low'] <= df.loc[i, 'lrc_n1']) and (df.loc[i, 'high'] >= df.loc[i, 'lrc_n3']): 
                side = 'buy'
        
        trade = False
        slope = df.loc[idx, 'lrc_slope']
        
        # RULE 1: SELL ONLY when slope is flat or down (<= 0)
        if side == 'sell' and slope <= 0 and (df.loc[idx, 'is_bearish_engulfing'] or df.loc[idx, 'is_shooting_star']):
            target, stop = df.loc[idx + 1, 'lrc_center'], df.loc[idx + 1, 'lrc_p3'] + 0.0010
            for f in range(idx + 1, length):
                if df.loc[f, 'low'] <= target: 
                    df.loc[idx+1, 'target_label'], trade, skip = 1.0, True, f
                    break
                if df.loc[f, 'high'] >= stop: 
                    df.loc[idx+1, 'target_label'], trade, skip = 0.0, True, f
                    break
        
        # RULE 2: BUY ONLY when slope is flat or up (>= 0)
        elif side == 'buy' and slope >= 0 and (df.loc[idx, 'is_bullish_engulfing'] or df.loc[idx, 'is_hammer']):
            target, stop = df.loc[idx + 1, 'lrc_center'], df.loc[idx + 1, 'lrc_n3'] - 0.0010
            for f in range(idx + 1, length):
                if df.loc[f, 'high'] >= target: 
                    df.loc[idx+1, 'target_label'], trade, skip = 1.0, True, f
                    break
                if df.loc[f, 'low'] <= stop: 
                    df.loc[idx+1, 'target_label'], trade, skip = 0.0, True, f
                    break
                
        df.loc[idx + 1, 'executed_trade'] = trade
    return df