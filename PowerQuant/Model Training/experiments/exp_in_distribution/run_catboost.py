import argparse
import os
import sys
from pathlib import Path

import dotenv

# Set up path before importing from experiments/src
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)

# Load environment variables if .env exists
dotenv.load_dotenv()
if os.getenv('PROJECT_ROOT'):
    project_root = os.getenv('PROJECT_ROOT')
    sys.path.insert(0, project_root)

# Now import from experiments and src
import optuna

from experiments.exp_in_distribution.logger import ExperimentLogger
from experiments.exp_in_distribution.utils import train_and_evaluate_model
from src.base_classifier.catboost import CatBoost, CatBoostConfig


def objective(trial, data_path: str):
    # Define hyperparameters
    config = CatBoostConfig(
        depth=trial.suggest_int('depth', 1, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        iterations=trial.suggest_categorical('iterations', [1000, 2000, 5000]),
        min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 10, 100),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
        loss_function=trial.suggest_categorical('loss_function', ['RMSE', 'MAE']),
    )
    
    # Train and evaluate
    model = CatBoost(config)
    model, metrics = train_and_evaluate_model(model, data_path=data_path, quantile=False)
    
    # Log results
    logger = ExperimentLogger(experiment_id='in_distribution_catboost')
    logger.log_result(config.to_json_dict(), metrics)
    
    return metrics['r2']


def main():
    parser = argparse.ArgumentParser(
        description='In-distribution training: train and test on same architecture distribution'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset file: "data/combined_df.csv" for Dataset-1 or "data/combined_static_*.csv" for Dataset-2'
    )
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("IN-DISTRIBUTION EXPERIMENT")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*60}\n")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, args.dataset), n_trials=20)
    
    print(f"\nBest R² Score: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")


if __name__ == '__main__':
    main()
