# src/train.py
import logging
import warnings
import mlflow
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Mute noisy algorithm warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger("SystemLogger.Training")

class ModelTournamentOrchestrator:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        # Isolate engineered features from structural metrics
        self.feature_cols = [col for col in self.df.columns if col.startswith('feat_')]
        
        # Filter down strictly to our setup instances where a target exists
        self.train_data = self.df[self.df['short_setup'] == True].dropna(subset=['target_label'])
        
        self.X = self.train_data[self.feature_cols].values
        self.y = self.train_data['target_label'].values
        
        # In trading, we care IMMENSELY about precision for class 1 (Success)
        # We don't want a model that over-trades; we want a model that is right when it triggers.
        logger.info(f"Tournament ready. Features: {len(self.feature_cols)} | Samples: {len(self.X)}")

    def objective(self, trial, model_name):
        """Optuna objective function tailored per algorithm framework."""
        # 5-Fold Time Series Split matrix
        tscv = TimeSeriesSplit(n_splits=5)
        precisions = []
        
        # Hyperparameter Search Spaces
        if model_name == "xgboost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'eval_metric': 'logloss'
            }
            model_factory = lambda: XGBClassifier(**params, random_state=42)
            
        elif model_name == "lightgbm":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 8, 64),
                'verbose': -1
            }
            model_factory = lambda: LGBMClassifier(**params, random_state=42)
            
        elif model_name == "catboost":
            params = {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'verbose': 0
            }
            model_factory = lambda: CatBoostClassifier(**params, random_state=42)

        # Execute walk-forward cross-validation
        for train_idx, val_idx in tscv.split(self.X):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            # Skip if slice lacks representation of our success class
            if 1 not in y_train or 1 not in y_val:
                continue
                
            model = model_factory()
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            
            # Calculate precision score specifically for the Mean Reversion Success label (1)
            score = precision_score(y_val, preds, labels=[1], average='macro', zero_division=0)
            precisions.append(score)
            
        return np.mean(precisions) if precisions else 0.0

    def run_tournament(self):
        """Runs optimization across frameworks and registers the global winner to MLflow."""
        mlflow.set_experiment("FX_Mean_Reversion_Trigger_Optimization")
        
        best_overall_score = -1
        best_overall_model = None
        best_overall_params = None
        winning_framework = ""
        
        frameworks = ["xgboost", "lightgbm", "catboost"]
        
        for framework in frameworks:
            logger.info(f"Starting Optuna optimization study for: {framework}")
            
            with mlflow.start_run(run_name=f"Optuna_{framework}", nested=True):
                study = optuna.create_study(direction="maximize")
                study.optimize(lambda trial: self.objective(trial, framework), n_trials=20)
                
                # Log best parameters found for this specific framework to MLflow
                mlflow.log_params(study.best_params)
                mlflow.log_metric("best_cv_precision_class_1", study.best_value)
                
                logger.info(f"Finished {framework}. Best CV Precision (Class 1): {study.best_value:.4f}")
                
                if study.best_value > best_overall_score:
                    best_overall_score = study.best_value
                    best_overall_params = study.best_params
                    winning_framework = framework

        logger.info(f"🏆 TOURNAMENT WINNER: {winning_framework} with CV Precision Score: {best_overall_score:.4f}")
        
        # Retrain the ultimate winner on the full historical dataset and register it
        with mlflow.start_run(run_name=f"Final_Production_{winning_framework}"):
            if winning_framework == "xgboost":
                production_model = XGBClassifier(**best_overall_params, random_state=42)
            elif winning_framework == "lightgbm":
                production_model = LGBMClassifier(**best_overall_params, random_state=42, verbose=-1)
            elif winning_framework == "catboost":
                production_model = CatBoostClassifier(**best_overall_params, random_state=42, verbose=0)
                
            production_model.fit(self.X, self.y)
            
            # Log final telemetry and artifacts
            mlflow.log_params(best_overall_params)
            mlflow.log_metric("final_precision_class_1", best_overall_score)
            mlflow.sklearn.log_model(production_model, artifact_path="model", registered_model_name="LRC_Candle_Trigger")
            
            logger.info("Production model trained on full dataset history and registered cleanly in MLflow Model Registry.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    try:
        tournament = ModelTournamentOrchestrator(data_path="data/processed_training_base.csv")
        tournament.run_tournament()
    except Exception as e:
        logger.critical(f"Tournament execution crashed: {str(e)}")