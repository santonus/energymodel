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

from experiments.exp_out_of_distribution.logger import ExperimentLogger
from experiments.exp_out_of_distribution.utils import train_and_evaluate_model
from src.base_classifier.svr import SVR, SVRConfig
from src.quantile_classifier.quant_classifier import QuantileClassifier


def objective(trial, data_path: str, test_arch_idx: int):
    # Define hyperparameters
    config = SVRConfig(
        kernel=trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
        C=trial.suggest_float('C', 0.1, 100.0, log=True),
        epsilon=trial.suggest_float('epsilon', 0.01, 1.0, log=True),
        gamma=trial.suggest_categorical('gamma', ['scale', 'auto']),
    )
    
    # Train and evaluate with quantile wrapping
    base_model = SVR(config)
    model = QuantileClassifier(base_model)
    model, metrics = train_and_evaluate_model(
        model, 
        data_path=data_path, 
        test_arch_idx=test_arch_idx,
        quantile=True
    )
    
    # Log results
    logger = ExperimentLogger(experiment_id=f'out_of_distribution_quant_svr_arch{test_arch_idx}')
    logger.log_result(config.to_json_dict(), metrics)
    
    return metrics['r2']


def main():
    parser = argparse.ArgumentParser(
        description='Out-of-distribution training: test on unseen architecture (Quantile-wrapped)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset file: "data/combined_df.csv" for Dataset-1 or "data/combined_static_*.csv" for Dataset-2'
    )
    parser.add_argument(
        '--test-arch',
        type=int,
        default=0,
        help='Architecture index to use as test set (0-3)'
    )
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("OUT-OF-DISTRIBUTION EXPERIMENT - QUANTILE SUPPORT VECTOR REGRESSION")
    print(f"Dataset: {args.dataset}")
    print(f"Test Architecture Index: {args.test_arch}")
    print(f"{'='*60}\n")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, args.dataset, args.test_arch), n_trials=20)
    
    print(f"\nBest R² Score: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")


if __name__ == '__main__':
    main()
