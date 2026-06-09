# run_pipeline.py
import logging
import yaml
from src.data_ingestion import AlpacaDataIngestor
from src.features import compute_linear_regression_channel
from src.labeling import apply_triple_barrier_labels

def configure_production_logging():
    """Initializes a unified log format for stdout tracking."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def main():
    configure_production_logging()
    root_logger = logging.getLogger("SystemLogger.Main")
    root_logger.info("Starting historical data and processing pipeline execution loop...")
    
    try:
        # Load configuration file
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        lrc_window = config['strategy']['lrc_window']
        max_holding_days = config['strategy']['max_holding_days']
        target_pair = config['data']['pairs'][0]  # Grab primary default ('EUR/USD')
        
        # 1. Execute Data Fetching
        ingestor = AlpacaDataIngestor()
        raw_bars = ingestor.fetch_daily_candles(pair=target_pair, days_back=1000)
        
        if raw_bars.empty:
            root_logger.critical("Pipeline halted: Initial historical data pull returned empty matrix.")
            return
            
        # 2. Execute LRC Channel Generation
        processed_data = compute_linear_regression_channel(raw_bars, window=lrc_window)
        
        # 3. Apply Multi-Barrier Optimization Labeling
        final_dataset = apply_triple_barrier_labels(processed_data, max_holding_days=max_holding_days)
        
        # 4. Extract metrics for logging output
        total_setups = final_dataset['short_setup'].sum()
        root_logger.info("--- Execution Run Summary Metrics ---")
        root_logger.info(f"Processed Bars    : {len(final_dataset)}")
        root_logger.info(f"Identified Setups : {total_setups}")
        
        if total_setups > 0:
            distribution = final_dataset['target_label'].value_counts()
            root_logger.info(f"  -> Hits on Reversion Center (1) : {distribution.get(1, 0)}")
            root_logger.info(f"  -> Hits on Boundary Invalidation (0): {distribution.get(0, 0)}")
            root_logger.info(f"  -> Reached Temporal Timeout (2)     : {distribution.get(2, 0)}")
            
        # Save historical processing output to file storage
        final_dataset.to_csv("data/processed_training_base.csv", index=False)
        root_logger.info("Pipeline processing runtime finalized successfully. Base dataset cached.")
        
    except Exception as e:
        root_logger.critical(f"Pipeline execution broke down due to unhandled engine crash: {str(e)}")

if __name__ == "__main__":
    main()