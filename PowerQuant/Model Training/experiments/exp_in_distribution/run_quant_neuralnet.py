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
from src.base_classifier.neuralnet import NeuralNet, NeuralNetConfig
from src.quantile_classifier.quant_classifier import QuantileClassifier


def objective(trial, data_path: str):
    # Define hyperparameters
    config = NeuralNetConfig(
        hidden_dim=trial.suggest_categorical('hidden_dim', [32, 64, 128, 256]),
        num_layers=trial.suggest_int('num_layers', 1, 4),
        dropout=trial.suggest_float('dropout', 0.0, 0.5),
        learning_rate=trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
        batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),
        epochs=trial.suggest_categorical('epochs', [50, 100, 200]),
    )
    
    # Train and evaluate with quantile wrapping
    base_model = NeuralNet(config)
    model = QuantileClassifier(base_model)
    model, metrics = train_and_evaluate_model(model, data_path=data_path, quantile=True)
    
    # Log results
    logger = ExperimentLogger(experiment_id='in_distribution_quant_neuralnet')
    logger.log_result(config.to_json_dict(), metrics)
    
    return metrics['r2']


def main():
    parser = argparse.ArgumentParser(
        description='In-distribution training: train and test on same architecture distribution (Quantile-wrapped)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset file: "data/combined_df.csv" for Dataset-1 or "data/combined_static_*.csv" for Dataset-2'
    )
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("IN-DISTRIBUTION EXPERIMENT - QUANTILE NEURAL NETWORK")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*60}\n")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, args.dataset), n_trials=20)
    
    print(f"\nBest R² Score: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")


if __name__ == '__main__':
    main()
