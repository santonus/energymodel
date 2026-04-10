# PowerQuant Build System

Python-based build system for data collection and model training, using `uv` for dependency management.

## Quick Start

### Install Dependencies
```bash
uv sync
```

### List Available Experiments
```bash
python build.py list-experiments
```

### Collect Dataset
Runs the KernelBench baseline timing script to generate performance data:
```bash
python build.py collect-dataset
```

Optional: Specify a custom dataset path:
```bash
python build.py collect-dataset --dataset data/my_dataset.csv
```

### Build Models
Train a quantile regression model for a specific experiment:

```bash
# Default: exp01 with all files
python build.py build-model exp01

# Specific experiment
python build.py build-model exp06

# Specific file only
python build.py build-model exp06 --file run_quant_catboost.py

# Custom dataset
python build.py build-model exp06 --dataset data/custom.csv
```

### Check Status
```bash
python build.py status
python build.py status --verbose
```

## Available Experiments

| Experiment | Description | Files |
|------------|-------------|-------|
| **exp01** | Baseline models with stratified split | 8 files (4 base + 4 quantile) |
| **exp02** | Alternative baseline configuration | 8 files |
| **exp04** | Another variant | 8 files |
| **exp05** | Quantile models with 3+1 architecture split | 8 files |
| **exp06** | **NEW** Quantile models on entire dataset (80/20 random split) | 4 files (quantile only) |

### Exp06 - The New Experiment

Exp06 differs from exp05 by:
- Uses **entire dataset** instead of per-architecture splits
- **Random 80/20 train/test split** (not stratified)
- **Quantile models only** (4 models: CatBoost, NeuralNet, RandomForest, SVR)
- **Automatic model saving** to `Model Training/models/exp06_quant_*.pkl`
- **Dynamic feature selection** from CSV headers (numeric columns only)

Perfect for production deployment and web app integration.

## Data Management

The build system automatically:
- Creates a symlink from `Model Training/data/` to `Dataset Collection/Dataset-2/Dataset/`
- Uses the dataset for both data collection and model training
- Supports custom datasets via `--dataset` flag

## Project Structure

```
PowerQuant/
├── pyproject.toml                 # Project config + dependencies (uv)
├── build.py                       # Build system CLI (this file)
├── Dataset Collection/
│   └── Dataset-2/
│       └── Benchmark Suite/
│           └── KernelBench/
│               └── scripts/
│                   └── generate_baseline_time.py
├── Model Training/
│   ├── data/                      # Symlink to Dataset Collection/Dataset-2/Dataset
│   ├── models/                    # Trained models saved here
│   └── experiments/
│       ├── exp01/ - exp06/        # Training scripts
└── www/                           # Web app (future integration)
```

## Dependencies

Core (auto-installed by `uv sync`):
- `torch>=2.0.0` - Deep learning framework
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - ML models
- `optuna>=3.0.0` - Hyperparameter tuning
- `catboost>=1.2.0` - Gradient boosting
- `joblib>=1.3.0` - Model serialization
- `click>=8.1.0` - CLI framework
- `rich>=13.0.0` - Pretty terminal output
- `python-dotenv>=1.0.0` - Environment management

Optional:
- `fvcore>=0.1.5` - Advanced feature extraction

Dev:
- `pytest>=7.0.0` - Testing
- `black>=23.0.0` - Code formatting
- `ruff>=0.1.0` - Linting

## Commands Reference

```bash
# List all commands
python build.py --help

# Get command-specific help
python build.py build-model --help
python build.py collect-dataset --help
python build.py list-experiments --help
python build.py status --help

# Common workflows
python build.py collect-dataset              # Collect baseline timing
python build.py build-model exp06            # Train all exp06 models
python build.py build-model exp06 --file run_quant_catboost.py  # Train single model

# Check everything is working
python build.py status --verbose
```

## Trained Models

After running `python build.py build-model exp06`, trained models are saved to:
```
Model Training/models/
├── exp06_quant_catboost.pkl      # CatBoost quantile model
├── exp06_quant_neuralnet.pkl     # NeuralNet quantile model
├── exp06_quant_randomforest.pkl  # RandomForest quantile model
└── exp06_quant_svr.pkl           # SVR quantile model
```

These can be loaded in the web app for power prediction:
```python
import joblib
model = joblib.load("Model Training/models/exp06_quant_catboost.pkl")
predictions = model.predict(X)  # Returns quantile predictions
```

## Future Integration

The build system is designed to integrate with:
- **Web App** (`www/app.py`) - Load trained models for power prediction
- **CI/CD** - Automated training pipelines
- **Deployment** - Package models and data for production

## Troubleshooting

### "Module not found" errors
```bash
# Ensure dependencies are installed
uv sync

# Verify PROJECT_ROOT is set
python build.py status
```

### Script not found errors
```bash
# Check experiment exists
python build.py list-experiments

# Check working directory (should be PowerQuant root)
pwd
```

### Data not loading
```bash
# Verify data symlink
python build.py status --verbose

# Check dataset file exists
ls Model\ Training/data/
```
