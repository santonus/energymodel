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
from src.base_classifier.svr import SVR, SVRConfig


def objective(trial, data_path: str):
    # Define hyperparameters
    config = SVRConfig(
        kernel=trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
        C=trial.suggest_float('C', 0.1, 100.0, log=True),
        epsilon=trial.suggest_float('epsilon', 0.01, 1.0, log=True),
        gamma=trial.suggest_categorical('gamma', ['scale', 'auto']),
    )
    
    # Train and evaluate
    model = SVR(config)
    model, metrics = train_and_evaluate_model(model, data_path=data_path, quantile=False)
    
    # Log results
    logger = ExperimentLogger(experiment_id='in_distribution_svr')
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
    print("IN-DISTRIBUTION EXPERIMENT - SUPPORT VECTOR REGRESSION")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*60}\n")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, args.dataset), n_trials=20)
    
    print(f"\nBest R² Score: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")


if __name__ == '__main__':
    main()
