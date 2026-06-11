# run_pipeline.py
import logging
import pandas as pd
import src.data_ingestion as di
from src.features import compute_linear_regression_channel, identify_candlestick_patterns

# Configure a clean output format
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("SystemLogger.Main")

if __name__ == "__main__":
    logger.info("Initializing Multi-Asset Production Driver...")
    
    # 1. Instantiate the ingestor with all 25 pairs
    ingestor = di.YFinanceDataIngestor(pairs=[
        'EUR/USD', 'AUD/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'USD/CAD', 
        'AUD/CAD', 'EUR/CAD', 'GBP/CAD', 'NZD/CAD', 'AUD/CHF', 'EUR/CHF', 
        'GBP/CHF', 'NZD/CHF', 'CAD/CHF', 'AUD/JPY', 'EUR/JPY', 'GBP/JPY', 
        'NZD/JPY', 'CAD/JPY', 'EUR/AUD', 'GBP/AUD', 'EUR/NZD', 'GBP/NZD', 'AUD/NZD'
    ])
    
    # Identify the extraction method dynamically
    download_method = None
    for attr in dir(ingestor):
        if callable(getattr(ingestor, attr)) and not attr.startswith("_"):
            if any(k in attr.lower() for k in ['download', 'fetch', 'get', 'run', 'execute']):
                download_method = getattr(ingestor, attr)
                break
                
    if download_method is None:
        raise AttributeError("Could not identify execution method on YFinanceDataIngestor class.")
        
    logger.info(f"Invoking data extraction via: {download_method.__name__}")
    result = download_method()
    
    # 2. Normalize incoming payload to a workable dictionary
    asset_dict = {}
    if isinstance(result, dict):
        asset_dict = result
    elif isinstance(result, pd.DataFrame):
        asset_dict = {'Dataset': result}
    else:
        # Check backup data frame assignments inside class states
        for attr_name in ['df', 'data', 'all_data']:
            if hasattr(ingestor, attr_name):
                val = getattr(ingestor, attr_name)
                if isinstance(val, dict):
                    asset_dict = val
                    break
                elif isinstance(val, pd.DataFrame):
                    asset_dict = {'Dataset': val}
                    break

    if not asset_dict:
        raise TypeError("Failed to extract a valid dataset dictionary from data engine.")

    # 3. Print the Comprehensive Multi-Asset Screener Table
    print("\n" + "="*95)
    print(f"{'MLOps MULTI-ASSET LIVE PRODUCTION SCREENER':^95}")
    print("="*95)
    print(f"{'ASSET':<12} | {'CLOSE':<8} | {'SLOPE (200)':<12} | {'REGIME':<12} | {'ZONE STATUS':<14} | {'STRATEGY SIGNAL'}")
    print("-" * 95)

    for pair_name, raw_df in asset_dict.items():
        if raw_df is None or len(raw_df) < 200:
            continue
            
        # Guard index and copy context cleanly
        df_clean = raw_df.copy().reset_index(drop=True)
        
        # Calculate your audited mathematical parameters (DGT Style)
        df_features = compute_linear_regression_channel(df_clean, window=200)
        df_patterns = identify_candlestick_patterns(df_features)
        
        latest_row = df_patterns.iloc[-1]
        
        current_close = latest_row['close']
        slope = latest_row['lrc_slope']
        n1 = latest_row['lrc_n1']
        p1 = latest_row['lrc_p1']
        
        # Determine directional structural trends
        regime = '📈 UP' if slope >= 0 else '📉 DOWN'
        
        # Classify geographic zones
        if current_close <= n1:
            zone_status = "DISCOUNT"
        elif current_close >= p1:
            zone_status = "PREMIUM"
        else:
            zone_status = "EQUILIBRIUM"
            
        # Apply defensive rules live
        signal = "💤 IDLE"
        if zone_status == "DISCOUNT":
            if slope >= 0:
                if latest_row['is_hammer'] or latest_row['is_bullish_engulfing']:
                    signal = "🚀 BUY TRIGGER"
                else:
                    signal = "⏳ WATCHING BUY"
            else:
                signal = "❌ MISMATCH (DOWN)"
                
        elif zone_status == "PREMIUM":
            if slope <= 0:
                if latest_row['is_shooting_star'] or latest_row['is_bearish_engulfing']:
                    signal = "🔥 SELL TRIGGER"
                else:
                    signal = "⏳ WATCHING SELL"
            else:
                signal = "❌ MISMATCH (UP)"

        print(f"{pair_name:<12} | {current_close:>8.5f} | {slope:>12.7f} | {regime:<12} | {zone_status:<14} | {signal}")

    print("="*95 + "\n")