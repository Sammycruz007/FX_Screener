# run_pipeline.py
import logging
import yaml
import os
import pandas as pd
from src.data_ingestion import YFinanceDataIngestor
from src.features import compute_linear_regression_channel, identify_candlestick_patterns
from src.labeling import apply_price_action_double_barrier

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    root_logger = logging.getLogger("SystemLogger.Main")
    root_logger.info("Launching Production Screener Evaluation...")
    
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    target_pair = config['data']['pairs'][0]
    
    raw_data_dict = YFinanceDataIngestor(pairs=[target_pair], period="5y", interval="1d").fetch_data()
    raw_bars = raw_data_dict.get(target_pair)
    
    df_features = compute_linear_regression_channel(raw_bars, window=200)
    df_patterns = identify_candlestick_patterns(df_features)
    final_dataset = apply_price_action_double_barrier(df_patterns, zone_lookback=3)
    
    unique_trades = final_dataset['executed_trade'].sum()
    root_logger.info("================ PRICE ACTION METRICS ================")
    root_logger.info(f"Unique Contextual Trades Found: {unique_trades}")
    if unique_trades > 0:
        distribution = final_dataset['target_label'].value_counts()
        root_logger.info(f"  -> Validated Wins (1.0): {distribution.get(1.0, 0)}")
        root_logger.info(f"  -> Stopped Out Losses (0.0): {distribution.get(0.0, 0)}")
    root_logger.info("=====================================================")
    
    os.makedirs("data", exist_ok=True)
    final_dataset.to_csv("data/processed_training_base.csv", index=False)

if __name__ == "__main__":
    main()