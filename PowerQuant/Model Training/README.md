# Model Training Subsystem

Model Training is the core ML pipeline for training quantile regression models to predict GPU power consumption.

## Overview

The subsystem implements an ensemble of 4 quantile regression models:
1. **CatBoost** - Gradient boosting on categorical features
2. **NeuralNet** - Multi-layer perceptron with configurable hidden layers
3. **RandomForest** - Ensemble of decision trees
4. **SVR** - Support Vector Regression with RBF/polynomial kernels

Each model is trained on GPU kernel performance data and wrapped in a `QuantileClassifier` for ensemble predictions.

## Directory Structure

```
Model Training/
├── data/                          # Symlink to Dataset files
├── models/                        # Trained model storage (joblib format)
├── experiments/
│   ├── exp_in_distribution/       # NEW: In-distribution training (same arch distribution)
│   │   ├── utils.py               # Unified utils with auto-dataset detection
│   │   ├── logger.py              # Result logging
│   │   ├── __init__.py
│   │   │
│   │   ├── run_catboost.py        # 8 training scripts:
│   │   ├── run_neuralnet.py       # - 4 base models
│   │   ├── run_randomforest.py    # - 4 quantile-wrapped versions
│   │   ├── run_svr.py             #
│   │   ├── run_quant_catboost.py  #
│   │   ├── run_quant_neuralnet.py #
│   │   ├── run_quant_randomforest.py
│   │   └── run_quant_svr.py
│   │
│   └── exp_out_of_distribution/   # NEW: Out-of-distribution testing (leave-one-arch-out)
│       ├── utils.py               # Architecture-based splitting
│       ├── logger.py
│       ├── __init__.py
│       └── [same 8 run_*.py scripts with test-arch support]
│
├── src/
│   ├── base_classifier/           # Core model implementations
│   │   ├── catboost.py            # CatBoost config + trainer
│   │   ├── neuralnet.py           # Neural net implementation
│   │   ├── randomforest.py        # Random forest implementation
│   │   ├── svr.py                 # SVR implementation
│   │   └── xgboost.py             # XGBoost (optional)
│   │
│   ├── quantile_classifier/       # Quantile regression wrapper
│   │   └── quant_classifier.py    # QuantileClassifier for any base model
│   │
│   └── utils/                     # Utilities
│       ├── clean_datasets.py      # Data cleaning helpers
│       ├── scribble.py            # Experimental code
│       └── summarize_results.py   # Result summarization
│
├── main.py                        # Legacy entry point
└── README.md                      # This file
```

## Experiments

### **exp_in_distribution: In-Distribution Training** (NEW)

Standard training on mixed architecture data with 80/20 split.

**Use Case:** Production deployment, overall accuracy
**Parameters:** `--dataset dataset-1 | dataset-2`

**Run:**
```bash
# All 8 models
python ../build.py build-model in-distribution --dataset dataset-1

# Specific model
python ../build.py build-model in-distribution --dataset dataset-1 --file run_catboost.py
```

**Key Features:**
- ✅ Auto-detects Dataset-1 vs Dataset-2
- ✅ 80/20 train/test split
- ✅ Hyperparameter tuning with Optuna (20 trials per model)
- ✅ Saves models to `models/in_distribution_*.pkl`
- ✅ Logs results to `Results/in_distribution_*_results.jsonl`
- ✅ 8 models: 4 base + 4 quantile-wrapped

### **exp_out_of_distribution: Out-of-Distribution Testing** (NEW)

Leave-one-architecture-out cross-validation to test generalization.

**Use Case:** Testing robustness to unseen hardware
**Parameters:** `--dataset dataset-1 | dataset-2 --test-arch 0|1|2|3`

**Run:**
```bash
# Test on unseen architecture 0 (train on 1,2,3)
python ../build.py build-model out-of-distribution --dataset dataset-2 --test-arch 0

# Specific model
python ../build.py build-model out-of-distribution --dataset dataset-2 --test-arch 0 --file run_svr.py
```

**Key Features:**
- ✅ Requires `--test-arch 0|1|2|3` parameter
- ✅ Trains on 3 architectures, tests on 1 holdout
- ✅ 4 architectures = 4 different splits
- ✅ Better evaluation of real-world generalization
- ✅ Saves models with architecture info: `models/out_of_distribution_arch0_*.pkl`
- ✅ Same 8 models as in-distribution

## Training a Model

### Using the Build System (Recommended)

```bash
cd /path/to/PowerQuant

# In-distribution (standard)
python build.py build-model in-distribution --dataset dataset-1

# Out-of-distribution (generalization test)
python build.py build-model out-of-distribution --dataset dataset-2 --test-arch 0

# Specific model file
python build.py build-model in-distribution --dataset dataset-1 --file run_catboost.py
```

### Running Directly

```bash
cd Model\ Training
export PROJECT_ROOT=$(pwd)/..

# In-distribution
python experiments/exp_in_distribution/run_catboost.py --dataset data/combined_df.csv

# Out-of-distribution (requires test-arch)
python experiments/exp_out_of_distribution/run_svr.py --dataset data/combined_static_20260127_092347.csv --test-arch 0
```

## Data Preparation

The `prepare_data()` function in each experiment's `utils.py`:

1. **Auto-detects dataset type** (Dataset-1 vs Dataset-2)
2. **Loads CSV** with dynamic column detection
3. **Filters to relevant features** (experiment-specific)
4. **Encodes architecture** as LabelEncoder indices for stratification
5. **Separates features (X), target (y), and architecture indices (ind)**

### Dataset-1 (C++ CUDA Kernels)
```
Input columns: avg_comp_lat, avg_glob_lat, glob_inst_kernel, ...
Features (8): All numeric columns except architecture/Avg
Target: Avg (power consumption in watts)
Architecture: 4 unique values (K80, Tesla, Ampere, Ada)
```

### Dataset-2 (PyTorch Models)
```
Input columns: total_bytes_read_mb, total_flops_m, arithmetic_intensity, ...
Features (7): Specified in DATASET2_FEATURES
Target: power_consumption (watts)
Architecture: Multiple GPU models
```

### Auto-Detection Example

```python
# In experiments/exp_in_distribution/utils.py
def detect_dataset_type(df):
    # Checks if Dataset-1 features exist
    if {'avg_comp_lat', 'avg_glob_lat', 'Architecture'}.issubset(df.columns):
        return 'dataset-1'
    # Checks if Dataset-2 features exist
    elif {'total_bytes_read_mb', 'architecture', 'power_consumption'}.issubset(df.columns):
        return 'dataset-2'
    else:
        raise ValueError("Unknown dataset")
```

## Model Architecture

### Base Models

Each model inherits from a base class with standard `fit()` and `predict()` interfaces:

**CatBoost:**
```python
from src.base_classifier.catboost import CatBoost, CatBoostConfig

config = CatBoostConfig(
    depth=trial.suggest_int('depth', 1, 10),
    learning_rate=trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
    iterations=trial.suggest_categorical('iterations', [1000, 2000, 5000]),
)
model = CatBoost(config)
model.fit(X_train, y_train)  # No architecture indices for base model
y_pred = model.predict(X_test)
```

**NeuralNet, RandomForest, SVR** follow the same pattern with their own hyperparameter options.

### Quantile Wrapper

Wraps any base model to make quantile-aware predictions:

```python
from src.quantile_classifier.quant_classifier import QuantileClassifier

# Create base model
base_model = CatBoost(config)

# Wrap with quantile classifier
quant_model = QuantileClassifier(base_model)

# Training requires architecture indices
quant_model.fit(X_train, y_train, ind_train)

# Prediction returns normalized quantiles
y_pred = quant_model.predict(X_test, ind_test)
```

The quantile wrapper:
1. Encodes targets as quantiles within each architecture
2. Trains base model on quantile-transformed targets
3. Decodes predictions back to original scale
4. Provides better uncertainty quantification

## Hyperparameter Tuning

Each model uses **Optuna** for automatic hyperparameter optimization over 20 trials:

**CatBoost Tuning:**
```python
config = CatBoostConfig(
    depth=trial.suggest_int('depth', 1, 10),
    learning_rate=trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
    iterations=trial.suggest_categorical('iterations', [1000, 2000, 5000]),
    min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 10, 100),
    l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
    loss_function=trial.suggest_categorical('loss_function', ['RMSE', 'MAE']),
)
```

**NeuralNet Tuning:**
```python
config = NeuralNetConfig(
    hidden_dim=trial.suggest_categorical('hidden_dim', [32, 64, 128, 256]),
    num_layers=trial.suggest_int('num_layers', 1, 4),
    dropout=trial.suggest_float('dropout', 0.0, 0.5),
    learning_rate=trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
    batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),
    epochs=trial.suggest_categorical('epochs', [50, 100, 200]),
)
```

**RandomForest & SVR:** Similar Optuna-based tuning with model-specific parameters.

**Optimization Target:** Maximize R² score on validation set

## Evaluation Metrics

All models report standardized metrics:
- **R² Score** - Coefficient of determination (0-1, higher = better)
- **MAE** - Mean Absolute Error in watts (lower = better)
- **RMSE** - Root Mean Squared Error in watts (lower = better)

Example from a training run:
```
==================================================
MODEL EVALUATION RESULTS
==================================================
Root Mean Squared Error (RMSE): 16.9078
Mean Absolute Error (MAE): 11.4030
R² Score: 0.8408
==================================================
```

**Interpretation:**
- R² = 0.84 means the model explains 84% of power variance
- MAE = 11.4 W means average prediction error is ±11.4 watts
- RMSE = 16.9 W emphasizes larger errors more than MAE

## Model Saving

Trained models are serialized with `joblib`:

```python
import joblib

# Save after training
joblib.dump(model, "models/in_distribution_catboost.pkl")

# Load for inference
model = joblib.load("models/in_distribution_catboost.pkl")
y_pred = model.predict(X_new)
```

Model files stored in `Model\ Training/models/`:
```
in_distribution_catboost.pkl
in_distribution_quant_neuralnet.pkl
out_of_distribution_arch0_svr.pkl
...etc
```

## Integration with Web App

The `www/app.py` Flask backend loads trained models at startup:

```python
import joblib
from pathlib import Path

# Load in-distribution models
models = {
    "catboost": joblib.load("Model Training/models/in_distribution_catboost.pkl"),
    "neuralnet": joblib.load("Model Training/models/in_distribution_quant_neuralnet.pkl"),
    "randomforest": joblib.load("Model Training/models/in_distribution_quant_randomforest.pkl"),
    "svr": joblib.load("Model Training/models/in_distribution_quant_svr.pkl"),
}

# Use for predictions
y_pred = models["catboost"].predict(X_test)
```

## Troubleshooting

### Memory Error During Training
- Reduce batch_size in NeuralNet config
- Use smaller dataset or subset
- Enable virtual memory swap

### Slow Hyperparameter Tuning
- Reduce n_trials in Optuna (default is 20)
- Use faster samplers (RandomSampler instead of TPESampler)
- Parallelize with `n_jobs=-1`

### Feature Mismatch at Prediction Time
- Ensure input features match training features
- Check dataset auto-detection: Dataset-1 vs Dataset-2
- Verify architecture encoding matches

### Missing Dataset
```bash
# Check data availability
python ../build.py status

# Should show all data files exist
```

### "out-of-distribution requires --test-arch"
```bash
# Correct: include --test-arch parameter
python ../build.py build-model out-of-distribution --dataset dataset-2 --test-arch 0

# Incorrect: forgot test-arch
python ../build.py build-model out-of-distribution --dataset dataset-2
#                                                    ^ Missing --test-arch
```

## Advanced Usage

### Custom Dataset
```bash
python build.py build-model in-distribution --dataset data/my_custom.csv
```

### Retraining on New Data
```bash
# New data with same format
cp new_data.csv Model\ Training/data/combined_df.csv

# Retrain models
python build.py build-model in-distribution --dataset data/combined_df.csv
```

### Model Comparison
```bash
# Compare in-distribution vs out-of-distribution
python build.py build-model in-distribution --dataset dataset-1
python build.py build-model out-of-distribution --dataset dataset-2 --test-arch 0
python build.py build-model out-of-distribution --dataset dataset-2 --test-arch 1
# Compare results in Model Training/results/
```

## Configuration Files

Each experiment contains:
- `utils.py` - Data loading, preparation, splitting, model evaluation
- `logger.py` - Result logging to JSONL files
- `run_*.py` - 8 individual model training scripts (4 base + 4 quantile)
- `experiments/exp0X/utils.py` - Data loading & preprocessing
- `experiments/exp0X/run_*.py` - Model training scripts

## Related Documentation

- [Build System](../BUILD_SYSTEM.md) - How to run training
- [KernelBench README](../Dataset%20Collection/Dataset-2/Benchmark%20Suite/KernelBench/README.md) - Data collection
- [Web App README](../www/README.md) - Model deployment
