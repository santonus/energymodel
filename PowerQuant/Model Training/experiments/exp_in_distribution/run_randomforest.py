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
from src.base_classifier.randomforest import RandomForest, RandomForestConfig


def objective(trial, data_path: str):
    # Define hyperparameters
    config = RandomForestConfig(
        n_estimators=trial.suggest_categorical('n_estimators', [50, 100, 200, 500]),
        max_depth=trial.suggest_categorical('max_depth', [5, 10, 20, None]),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
        max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2']),
    )
    
    # Train and evaluate
    model = RandomForest(config)
    model, metrics = train_and_evaluate_model(model, data_path=data_path, quantile=False)
    
    # Log results
    logger = ExperimentLogger(experiment_id='in_distribution_randomforest')
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
    print("IN-DISTRIBUTION EXPERIMENT - RANDOM FOREST")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*60}\n")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, args.dataset), n_trials=20)
    
    print(f"\nBest R² Score: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")


if __name__ == '__main__':
    main()
