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
from src.base_classifier.randomforest import RandomForest, RandomForestConfig
from src.quantile_classifier.quant_classifier import QuantileClassifier


def objective(trial, data_path: str, test_arch_idx: int):
    # Define hyperparameters
    config = RandomForestConfig(
        n_estimators=trial.suggest_categorical('n_estimators', [50, 100, 200, 500]),
        max_depth=trial.suggest_categorical('max_depth', [5, 10, 20, None]),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
        max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2']),
    )
    
    # Train and evaluate with quantile wrapping
    base_model = RandomForest(config)
    model = QuantileClassifier(base_model)
    model, metrics = train_and_evaluate_model(
        model, 
        data_path=data_path, 
        test_arch_idx=test_arch_idx,
        quantile=True
    )
    
    # Log results
    logger = ExperimentLogger(experiment_id=f'out_of_distribution_quant_randomforest_arch{test_arch_idx}')
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
    print("OUT-OF-DISTRIBUTION EXPERIMENT - QUANTILE RANDOM FOREST")
    print(f"Dataset: {args.dataset}")
    print(f"Test Architecture Index: {args.test_arch}")
    print(f"{'='*60}\n")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, args.dataset, args.test_arch), n_trials=20)
    
    print(f"\nBest R² Score: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")


if __name__ == '__main__':
    main()
