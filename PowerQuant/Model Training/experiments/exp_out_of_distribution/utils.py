import os
import sys
from pathlib import Path
from typing import Any

import dotenv
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Dynamically find project root (3 levels up from this file)
project_root = str(Path(__file__).resolve().parent.parent.parent)

# Load environment variables if .env exists and override if PROJECT_ROOT is set
dotenv.load_dotenv()
if os.getenv('PROJECT_ROOT'):
    project_root = os.getenv('PROJECT_ROOT')

# Dataset-1 (C++ CUDA) features
DATASET1_FEATURES = [
    'avg_comp_lat',
    'glob_inst_kernel',
    'avg_glob_lat',
    'glob_load_sm',
    'glob_store_sm',
    'misc_inst_kernel',
    'inst_issue_cycles',
    'cache_penalty',
]

# Dataset-2 (PyTorch) features
DATASET2_FEATURES = [
    'total_bytes_read_mb',
    'total_bytes_written_mb',
    'total_bytes_mb',
    'total_flops_m',
    'arithmetic_intensity',
    'num_nodes',
    'count_unique_ops',
]


def detect_dataset_type(df: pd.DataFrame) -> str:
    """Detect if dataset is Dataset-1 (C++ CUDA) or Dataset-2 (PyTorch)."""
    dataset1_cols = set(DATASET1_FEATURES + ['Avg', 'Architecture'])
    dataset2_cols = set(DATASET2_FEATURES + ['architecture', 'power_consumption'])
    
    df_cols = set(df.columns)
    
    if dataset1_cols.issubset(df_cols):
        return 'dataset-1'
    elif dataset2_cols.issubset(df_cols):
        return 'dataset-2'
    else:
        raise ValueError(f"Unknown dataset type. Columns: {list(df.columns)}")


def load_data(data_path: str) -> pd.DataFrame:
    """Load the preprocessed GPU kernel performance data."""
    full_path = os.path.join(project_root, data_path)
    df = pd.read_csv(full_path)
    print(f'Data loaded successfully. Shape: {df.shape}')
    print(f'Dataset type: {detect_dataset_type(df)}')
    return df


def prepare_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare features and target variable from the dataset."""
    dataset_type = detect_dataset_type(df)
    
    if dataset_type == 'dataset-1':
        # Dataset-1: C++ CUDA kernels
        features = DATASET1_FEATURES + ['Avg', 'Architecture']
        df = df[features]
        y = np.array(df['Avg'])
        
        le = LabelEncoder()
        ind = le.fit_transform(df['Architecture'])
        
        X = np.array(df.drop(columns=['Avg', 'Architecture']))
        
    elif dataset_type == 'dataset-2':
        # Dataset-2: PyTorch models
        features = DATASET2_FEATURES + ['architecture', 'power_consumption']
        df = df[features]
        y = np.array(df['power_consumption'])
        
        le = LabelEncoder()
        ind = le.fit_transform(df['architecture'])
        
        X = np.array(df.drop(columns=['power_consumption', 'architecture']))
    
    print(f'Features shape: {X.shape}')
    print(f'Target shape: {y.shape}')
    
    return X, y, ind


def split_data_by_architecture(
    X: np.ndarray,
    y: np.ndarray,
    ind: np.ndarray,
    test_arch_idx: int,
) -> tuple[Any, Any]:
    """Split data by architecture for out-of-distribution testing.
    
    Args:
        X: Feature matrix
        y: Target values
        ind: Architecture indices (0, 1, 2, 3 for different architectures)
        test_arch_idx: Architecture index to use as test set
    
    Returns:
        (Xtrain, ytrain, indtrain), (Xtest, ytest, indtest)
    """
    # Split by architecture
    train_mask = ind != test_arch_idx
    test_mask = ind == test_arch_idx
    
    Xtrain = X[train_mask]
    ytrain = y[train_mask]
    indtrain = ind[train_mask]
    
    Xtest = X[test_mask]
    ytest = y[test_mask]
    indtest = ind[test_mask]
    
    print(f"\nOut-of-distribution split:")
    print(f"Train size: {len(Xtrain)} (excluding architecture {test_arch_idx})")
    print(f"Test size: {len(Xtest)} (architecture {test_arch_idx} only)")
    
    return (Xtrain, ytrain, indtrain), (Xtest, ytest, indtest)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ind_train: np.ndarray,
    model: Any,
    quantile: bool = True,
) -> Any:
    """Train a model."""
    if quantile:
        model.fit(X_train, y_train, ind_train)
    else:
        model.fit(X_train, y_train)
    
    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ind_test: np.ndarray,
    quantile: bool = True,
) -> dict:
    """Evaluate the trained model on test data."""
    if quantile:
        y_pred = model.predict(X_test, ind_test)
    else:
        y_pred = model.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
    }
    
    print('\n' + '=' * 50)
    print('MODEL EVALUATION RESULTS')
    print('=' * 50)
    print(f'Root Mean Squared Error (RMSE): {metrics["rmse"]:.4f}')
    print(f'Mean Absolute Error (MAE): {metrics["mae"]:.4f}')
    print(f'R² Score: {metrics["r2"]:.4f}')
    print('=' * 50)
    
    return metrics


def train_and_evaluate_model(model: Any, data_path: str, test_arch_idx: int = 0, quantile: bool = True):
    """Complete training and evaluation pipeline for out-of-distribution setting."""
    # Step 1: Load data
    print('\n1. Loading data...')
    df = load_data(data_path)
    X, y, ind = prepare_data(df)
    
    # Step 2: Split data by architecture (out-of-distribution)
    print('\n2. Splitting data by architecture (out-of-distribution)...')
    trainset, testset = split_data_by_architecture(X, y, ind, test_arch_idx)
    Xtrain, ytrain, indtrain = trainset
    Xtest, ytest, indtest = testset
    
    # Step 3: Train model
    print('\n3. Training model...')
    model = train_model(Xtrain, ytrain, indtrain, model, quantile=quantile)
    
    # Step 4: Evaluate model
    print('\n4. Evaluating model on unseen architecture...')
    metrics = evaluate_model(model, Xtest, ytest, indtest, quantile=quantile)
    
    print('\nPipeline completed successfully!')
    return model, metrics
