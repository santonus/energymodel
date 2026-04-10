# PowerQuant: GPU Power Consumption Prediction

A complete ML pipeline for predicting GPU power consumption from GPU kernel and PyTorch model characteristics using multiple regression models (CatBoost, Neural Networks, Random Forest, SVR) with support for both standard and quantile predictions.

## 🎯 Quick Start (30 seconds)

```bash
cd /path/to/PowerQuant

# 1. Install dependencies
pip install -r pyproject.toml  # or: uv sync

# 2. List available experiments
python build.py list-experiments

# 3. Train models (in-distribution or out-of-distribution)
python build.py build-model in-distribution --dataset dataset-1
# or: python build.py build-model out-of-distribution --dataset dataset-2 --test-arch 0

# 4. Launch web app
cd www && python app.py
# Opens at http://localhost:5000
```

## 🏗️ Project Structure

```
PowerQuant/
├── README.md                                    # This file
├── BUILD_SYSTEM.md                             # Build system documentation
├── SUBSYSTEMS.md                               # Subsystem overview & workflows
├── build.py                                    # Python-based build CLI
├── pyproject.toml                              # Dependencies (uv)
│
├── Dataset Collection/
│   ├── Dataset-1(works for C++ CUDA Codes)/
│   │   ├── README.md                          # C++ CUDA dataset details
│   │   └── Datasets/
│   │       └── combined_df.csv                # Processed C++ kernel data
│   │
│   └── Dataset-2(works for Python PyTorch Models)/
│       ├── README.md                          # PyTorch dataset details
│       ├── Benchmark Suite/KernelBench/
│       │   ├── README.md                      # KernelBench documentation
│       │   ├── scripts/
│       │   │   ├── generate_baseline_time.py
│       │   │   ├── extract_model_features.py
│       │   │   └── variable_scaler.py
│       │   └── Dataset/
│       │
│       └── Dataset/
│           ├── combined_df.csv                # For Dataset-1
│           └── combined_static_20260127_092347.csv  # For Dataset-2
│
├── Model Training/
│   ├── README.md                              # Training pipeline docs
│   ├── data/                                  # Symlink to datasets
│   ├── models/                                # Trained models (joblib)
│   ├── experiments/
│   │   ├── exp_in_distribution/               # NEW: In-distribution experiment
│   │   │   ├── utils.py                       # Unified utils (auto-detects dataset)
│   │   │   ├── logger.py                      # Result logging
│   │   │   ├── run_catboost.py
│   │   │   ├── run_neuralnet.py
│   │   │   ├── run_randomforest.py
│   │   │   ├── run_svr.py
│   │   │   ├── run_quant_catboost.py          # Quantile-wrapped versions
│   │   │   ├── run_quant_neuralnet.py
│   │   │   ├── run_quant_randomforest.py
│   │   │   └── run_quant_svr.py
│   │   │
│   │   └── exp_out_of_distribution/           # NEW: Out-of-distribution experiment
│   │       ├── utils.py                       # Architecture-based splitting
│   │       ├── logger.py
│   │       └── [same 8 run_*.py scripts]
│   │
│   └── src/
│       ├── base_classifier/                  # Model implementations
│       │   ├── catboost.py
│       │   ├── neuralnet.py
│       │   ├── randomforest.py
│       │   └── svr.py
│       ├── quantile_classifier/              # Quantile regression wrapper
│       │   └── quant_classifier.py
│       └── utils/                            # Helper utilities
│
├── www/
│   ├── README.md                             # Web app documentation
│   ├── app.py                                # Flask backend
│   ├── index.html                            # CodeMirror editor UI
│   └── models/                               # Symlink to trained models
│
└── Results/                                  # Experiment results
```

## 🚀 Main Components

### 1. Build System (`build.py`)
Python-based CLI using Click framework for managing datasets and training models.

```bash
# List available experiments
python build.py list-experiments

# Train models
python build.py build-model <experiment> --dataset <dataset> [--test-arch <N>]

# Available experiments:
#   - in-distribution   : Train and test on same architecture distribution
#   - out-of-distribution: Test on unseen architecture (leave-one-out)

# Available datasets:
#   - dataset-1: C++ CUDA kernel data (combined_df.csv)
#   - dataset-2: PyTorch model data (combined_static_20260127_092347.csv)

# Examples:
python build.py build-model in-distribution --dataset dataset-1
python build.py build-model out-of-distribution --dataset dataset-2 --test-arch 0

# Train specific model
python build.py build-model in-distribution --dataset dataset-1 --file run_catboost.py

# Check system status
python build.py status
```

### 2. Two Experiment Types

#### **In-Distribution (Standard)**
- Train and test on same architecture distribution
- Uses standard 80/20 train/test split
- Good for: Overall power prediction accuracy
- Command: `python build.py build-model in-distribution --dataset dataset-1`

#### **Out-of-Distribution (Generalization)**
- Train on 3 architectures, test on 1 holdout architecture
- Leave-one-architecture-out cross-validation
- Good for: Testing generalization to unseen hardware
- Requires: `--test-arch 0|1|2|3` parameter
- Command: `python build.py build-model out-of-distribution --dataset dataset-2 --test-arch 0`

### 3. Dataset Support

#### **Dataset-1: C++ CUDA Kernels**
- **File:** `combined_df.csv`
- **Rows:** 15,054 kernel kernels
- **Features (8):** avg_comp_lat, avg_glob_lat, glob_inst_kernel, glob_load_sm, glob_store_sm, misc_inst_kernel, inst_issue_cycles, cache_penalty
- **Target:** Avg (power consumption)
- **Architectures:** K80, Tesla, Ampere, Ada

#### **Dataset-2: PyTorch Models**
- **File:** `combined_static_20260127_092347.csv`
- **Rows:** Diverse PyTorch models
- **Features (7):** total_bytes_read_mb, total_bytes_written_mb, total_bytes_mb, total_flops_m, arithmetic_intensity, num_nodes, count_unique_ops
- **Target:** power_consumption
- **Architectures:** Various GPU models

**Auto-Detection:** The system automatically detects which dataset is provided and loads appropriate features.

### 4. Available Models (8 per Experiment)

Each experiment includes **8 training scripts**:

**Standard Models:**
- `run_catboost.py` - Gradient boosting with categorical feature support
- `run_neuralnet.py` - Multi-layer perceptron (1-4 layers, 32-256 hidden dims)
- `run_randomforest.py` - Ensemble of decision trees
- `run_svr.py` - Support Vector Regression (linear, RBF, poly kernels)

**Quantile-Wrapped Models:**
- `run_quant_catboost.py` - CatBoost with quantile normalization
- `run_quant_neuralnet.py` - Neural Network with quantile normalization
- `run_quant_randomforest.py` - Random Forest with quantile normalization
- `run_quant_svr.py` - SVR with quantile normalization

**Hyperparameter Tuning:** All models use Optuna with 20 trials per run.

### 4. Web Application (Flask) (Under Development)
Interactive web UI for PyTorch model analysis and power prediction.

**Features:**
- CodeMirror editor with syntax highlighting
- Automatic feature extraction from code
- Power predictions from 4 quantile models
- REST API endpoints 
- Model comparison view

**Deploy:** `cd www && python app.py` then open `http://localhost:5000` (Under Development)

## 📊 Data Flow

```
GPU Benchmarks
    ↓
KernelBench (generate_baseline_time.py)
    ├─→ Extract features (FLOPs, memory, intensity)
    ├─→ Measure execution time
    ├─→ Record power consumption
    └─→ Output: combined_*.csv
    ↓
Model Training/data (symlink)
    ↓
Training Pipeline (exp01-05)
    ├─→ Load CSV dynamically
    ├─→ Select numeric columns
    ├─→ One-hot encode architecture
    ├─→ Stratified or configured split
    ├─→ Optuna hyperparameter tuning
    ├─→ Train 4 models
    └─→ Save to models/exp*_quant_*.pkl
```

## 🎯 Experiment Comparison

| Aspect | In-Distribution | Out-of-Distribution |
|--------|-----------------|---------------------|
| **Purpose** | Standard prediction | Generalization test |
| **Data Split** | 80/20 random split | Leave-one-arch-out |
| **Train/Test** | Same architecture mix | Train: 3 archs, Test: 1 arch |
| **Best For** | Overall accuracy | Hardware generalization |
| **Test Arch Param** | Not used | Required (0-3) |
| **Use Case** | Production deployment | Robustness testing |

## 📈 Performance Metrics

All models report:
- **R² Score** - Coefficient of determination (range: 0-1, higher = better)
- **MAE** - Mean Absolute Error in watts (lower = better)
- **RMSE** - Root Mean Squared Error in watts (lower = better)

**Example output**:
```
Test R²:   0.854 (explains 85.4% of variance)
Test MAE:  15.23 W (average error)
Test RMSE: 22.15 W (root mean squared error)
```

## 🔗 Feature Names

Automatically detected from CSV headers:

| Feature | Type | Example |
|---------|------|---------|
| `total_flops_m` | Numeric | 1805.3 M |
| `total_bytes_read_mb` | Numeric | 512.5 MB |
| `total_bytes_written_mb` | Numeric | 256.2 MB |
| `arithmetic_intensity` | Numeric | 2.14 FLOP/B |
| `num_nodes` | Numeric | 5 layers |
| `count_unique_ops` | Numeric | 3 types |
| `architecture` | Categorical | K80, Tesla, Ampere |
| `power_consumption` | Target | 185.4 W |

## 🛠️ Common Workflows

### Workflow 1: Full In-Distribution Pipeline (Recommended)
```bash
# Standard training on Dataset-1 (C++ CUDA kernels)
python build.py list-experiments              # See available experiments
python build.py build-model in-distribution --dataset dataset-1
# Trains all 8 models (4 base + 4 quantile-wrapped)
```

### Workflow 2: Out-of-Distribution Testing
```bash
# Test generalization to unseen architecture
python build.py build-model out-of-distribution --dataset dataset-2 --test-arch 0
# Leave architecture 0 as test, train on architectures 1,2,3
```

### Workflow 3: Train Single Model
```bash
# Train only CatBoost for in-distribution
python build.py build-model in-distribution --dataset dataset-1 --file run_catboost.py

# Train only Quantile SVR for out-of-distribution
python build.py build-model out-of-distribution --dataset dataset-2 --test-arch 0 --file run_quant_svr.py
```

### Workflow 4: Compare Models
```bash
# Run all 8 models in-distribution with Dataset-1
python build.py build-model in-distribution --dataset dataset-1

# Then compare results:
ls Model\ Training/results/in_distribution_*_results.jsonl
```

### Workflow 5: Test Different Dataset
```bash
# Switch to PyTorch dataset (Dataset-2)
python build.py build-model in-distribution --dataset dataset-2

# Or test out-of-distribution with leave-one-arch-out
python build.py build-model out-of-distribution --dataset dataset-2 --test-arch 1
```

## 📦 Dependencies

**Core** (auto-installed by `uv sync`):
```
torch>=2.0.0              # Deep learning framework
pandas>=1.5.0             # Data manipulation
numpy>=1.24.0             # Numerical computing
scikit-learn>=1.3.0       # ML utilities
optuna>=3.0.0             # Hyperparameter tuning
catboost>=1.2.0           # One model type
joblib>=1.3.0             # Model serialization
click>=8.1.0              # CLI framework
rich>=13.0.0              # Terminal UI
flask>=2.3.0              # Web framework
python-dotenv>=1.0.0      # Env vars
```

**Optional:**
```
fvcore>=0.1.5             # Advanced FLOP counting
```

**Dev:**
```
pytest>=7.0.0
black>=23.0.0
ruff>=0.1.0
```

## 🔧 Installation

### From Scratch
```bash
# Clone or download project
cd /path/to/PowerQuant

# Install with uv (recommended)
pip install uv
uv sync

# Or install manually
pip install -r requirements.txt
```

### Verify Installation
```bash
python build.py status
# Should show all green checkmarks ✓
```

## 🚢 Deployment

### Local Development
```bash
python build.py build-model exp05
cd www && python app.py
```

### Production (Gunicorn) (Under Development)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 www/app:app
```

### Docker
```bash
# Build
docker build -t powerquant .

# Run
docker run -p 5000:5000 powerquant
```

## ❓ Troubleshooting

### Missing Dependencies
```bash
# Install all required packages
python -m pip install torch pandas numpy scikit-learn optuna catboost joblib click rich python-dotenv flask

# Or use uv
uv sync
```

### "No CUDA GPUs found"
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"

# Data collection requires GPU - install NVIDIA drivers
```

### "Port 5000 already in use"
```bash
# Change port in www/app.py
app.run(port=5001)

# Or kill existing process
lsof -ti:5000 | xargs kill -9
```

### "Experiment not found"
```bash
# List available experiments
python build.py list-experiments

# Should show:
# - in-distribution (8 files)
# - out-of-distribution (8 files)
```

### "Models not found for predictions"
```bash
# Check if models are trained
ls Model\ Training/models/

# Should show files like:
# in_distribution_catboost_model.pkl
# in_distribution_quant_neuralnet_model.pkl
# out_of_distribution_arch0_svr_model.pkl
# etc.

# If missing, retrain:
python build.py build-model in-distribution --dataset dataset-1
```

### "Dataset-1 vs Dataset-2 confusion"
```bash
# Dataset-1: C++ CUDA kernels
python build.py build-model in-distribution --dataset dataset-1

# Dataset-2: PyTorch models
python build.py build-model in-distribution --dataset dataset-2

# System auto-detects columns and selects appropriate features
```

### "test-arch parameter required"
```bash
# Out-of-distribution REQUIRES test-arch (0-3)
python build.py build-model out-of-distribution --dataset dataset-2 --test-arch 0
#                                                                    ^^^^^^^^^^^^^^
# In-distribution DOES NOT use test-arch
python build.py build-model in-distribution --dataset dataset-1
```

## 📖 For More Details

- **[BUILD_SYSTEM.md](BUILD_SYSTEM.md)** - Comprehensive build system documentation
- **[SUBSYSTEMS.md](SUBSYSTEMS.md)** - Overview of all subsystems and workflows
- **[Model Training/README.md](Model%20Training/README.md)** - Training pipeline details
- **[KernelBench README](Dataset%20Collection/Dataset-2/Benchmark%20Suite/KernelBench/README.md)** - Data collection

## 🤝 Support

- **Issues & Bugs:** Check [Troubleshooting](#troubleshooting) section
- **Questions:** See subsystem-specific READMEs
- **Suggestions:** Open an issue with feature request

---

**Python:** 3.10+  
**CUDA:** Required for data collection (optional for predictions)
