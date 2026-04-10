# PowerQuant Subsystem Documentation

Complete documentation for all PowerQuant subsystems. Start here to understand the project structure.

## 📚 Main Subsystems

### 1. [Build System](BUILD_SYSTEM.md)
**Command-line interface for managing the entire project**

- `python build.py collect-dataset` - Collect GPU benchmark data
- `python build.py build-model <exp>` - Train models
- `python build.py list-experiments` - List all experiments
- `python build.py status` - Check project setup

**When to use:**
- Running production workflows
- Automating data collection & model training
- Managing dependencies with `uv`

**Key Features:**
- Unified entry point for all operations
- Automatic data symlink creation
- Rich terminal output with progress tracking
- Experiment management

---

### 2. [KernelBench - Dataset Collection](Dataset%20Collection/Dataset-2/Benchmark%20Suite/KernelBench/README.md)
**PyTorch benchmark suite for collecting GPU performance data**

**Key Scripts:**
- `generate_baseline_time.py` - Main data collection script
- `extract_model_features.py` - Extract FLOPs & memory features
- `variable_scaler.py` - Normalize numeric features

**When to use:**
- Collecting new baseline data for GPUs
- Analyzing PyTorch model operations
- Extracting computational features

**Key Features:**
- 100+ FLOP counting handlers
- Cumulative memory model (not peak)
- Arithmetic intensity calculation
- Multi-GPU support

**Output**: CSV files with features:
```
total_flops_m, total_bytes_read_mb, total_bytes_written_mb,
arithmetic_intensity, num_nodes, count_unique_ops,
architecture, power_consumption
```

---

### 3. [Model Training](Model%20Training/README.md)
**ML pipeline for training quantile regression models**

**Experiments:**
- **exp01-05** - Various baseline configurations

**Models (4-per experiment):**
1. **CatBoost** - Gradient boosting
2. **NeuralNet** - Multi-layer perceptron
3. **RandomForest** - Ensemble trees
4. **SVR** - Support Vector Regression

**When to use:**
- Training power consumption prediction models
- Comparing different model architectures
- Hyperparameter optimization
- Feature engineering

**Key Features:**
- Dynamic feature selection from CSV headers
- Optuna hyperparameter tuning (20 trials)
- Automatic model serialization with joblib
- Architecture one-hot encoding
- Evaluation metrics: R², MAE, RMSE

**Output**: Trained models saved to `models/exp0X_quant_*.pkl`

---

### 4. [Web Application](www/README.md)
**Flask web UI for model analysis & power prediction**

**Features:**
- CodeMirror editor for PyTorch code
- Automatic feature extraction
- Power prediction from 4 models
- REST API endpoints
- Results comparison

**When to use:**
- Analyzing individual PyTorch models
- Getting power predictions through web UI
- Programmatic access via REST API
- Integrating with other tools

**Key Endpoints:**
- `POST /api/analyze` - Extract features from code
- `POST /api/predict` - Get power predictions
- `GET /api/models` - List available models

**Input**: PyTorch code or features dict
**Output**: Power predictions with quantiles

---

**Experiments:**
- **exp01-05** - Various baseline configurations
## 🔄 Data Flow

```
collect-dataset
  ↓
KernelBench
  ├─→ generate_baseline_time.py
  ├─→ extract_model_features.py
  └─→ Dataset CSV
  ↓
build-model exp05
  ├─→ prepare_data()
  ├─→ hyperparameter tuning (Optuna)
  ├─→ train 4 models
  └─→ save to models/
  ↓
Web App
  ├─→ load trained models
  └─→ predict power
```

## 🛠️ Common Workflows

### Workflow 1: Start Fresh

```bash
# 1. Collect data from GPUs
python build.py collect-dataset

# 2. Train models on new data
python build.py build-model exp05

# 3. Launch web app
cd www && python app.py
```

### Workflow 2: Train Specific Model

```bash
# Train only CatBoost for exp05
python build.py build-model exp05 --file run_quant_catboost.py

# Check results
cat Model\ Training/experiments/exp05/results.json
```

### Workflow 3: API Integration

```python
import requests

response = requests.post(
  "http://localhost:5000/api/analyze",
  json={"code": "import torch\n..."}
)
features = response.json()["features"]

response = requests.post(
  "http://localhost:5000/api/predict",
  json={"features": features}
)
predictions = response.json()["predictions"]
```

### Workflow 4: Experiment Comparison

```bash
python build.py build-model exp01  # Baseline
python build.py build-model exp05  # Architecture-stratified

cat Model\ Training/experiments/exp01/results.json
cat Model\ Training/experiments/exp05/results.json
```

## 🎯 Key Decisions

### Exp05 as Production Standard

Why **exp05** is recommended:

| Feature | exp01-05 | exp05 |
|---------|----------|-------|
| Dataset scope | Per-architecture | Per-architecture |
| Train/test split | Stratified | Stratified |
| Model count | Base + quant | Base + quant |
| Scalability | Limited | ✅ Better |
| Generalization | Per-architecture | ✅ Consistent |
| Training time | Longer | ✅ Faster |
| Accuracy | Mixed | ✅ Better |

## 📊 Architecture Overview

```
PowerQuant (Root)
│
├─── BUILD_SYSTEM.md (this file)
python build.py build-model exp05
├─── build.py (CLI implementation)
│
├─── Dataset Collection/
│    └─── Dataset-2/
│         └─── Benchmark Suite/KernelBench/ (README.md)
│             ├─── scripts/generate_baseline_time.py
│             └─── Dataset/ (output CSVs)
│
├─── Model Training/ (README.md)
│    ├─── experiments/exp01-06/
│    │    └─── run_quant_*.py (training scripts)
│    ├─── models/ (trained models)
│    ├─── data/ → symlink to Dataset Collection/Dataset-2/Dataset
│    └─── src/
│         ├─── base_classifier/ (model implementations)
│         └─── quantile_classifier/ (ensemble wrapper)
│
├─── www/ (README.md)
│    ├─── app.py (Flask backend)
│    ├─── index.html (web UI)
│    └─── models/ (symlink to trained models)
└─── Results/ (experiment summaries in LaTeX)
```

## 🎯 Key Decisions

### Exp05 as Production Standard

Why **exp05** is recommended:

| Feature | exp01-05 | exp05 |
|---------|----------|-------|
| Dataset scope | Per-architecture | Per-architecture |
| Train/test split | Stratified | Stratified |
| Model count | Base + quant | Base + quant |
| Scalability | Limited | ✅ Better |
| Generalization | Per-architecture | ✅ Consistent |
| Training time | Longer | ✅ Faster |
| Accuracy | Mixed | ✅ Better |

### Features Extracted

**Dynamic Selection** (from CSV columns):
- Numeric columns automatically detected
- Target (`power_consumption`) excluded
- Architecture one-hot encoded
- Missing values filled intelligently

**Cumulative Memory Model**:
- $\text{Memory} = \sum (\text{read} + \text{write})$
- Captures sustained bandwidth usage
- Better predictor of power than peak memory
**Arithmetic Intensity**:
- $I = \frac{\text{FLOPs}}{\text{Bytes Transferred}}$
- Roofline model metric
- Indicates compute efficiency

## 📖 Quick Reference

### Commands

```bash
# General
python build.py --help
python build.py status
python build.py status --verbose

# Data
python build.py collect-dataset
python build.py collect-dataset --dataset data/my_custom.csv

# Models
# Web
cd www && python app.py
```

### File Locations

```
Project root: /path/to/PowerQuant

KernelBench scripts:
  /Dataset Collection/Dataset-2/Benchmark Suite/KernelBench/scripts/

Collected data:
  /Dataset Collection/Dataset-2/Dataset/*.csv

Training data symlink:
  /Model Training/data/ → /Dataset Collection/Dataset-2/Dataset

Trained models:
  /Model Training/models/exp06_quant_*.pkl

Web app:
  /www/app.py
  /www/index.html
```

### Feature Names

Used by all models:
- `total_bytes_read_mb` - Data read (MB)
- `total_bytes_written_mb` - Data written (MB)
- `total_flops_m` - Floating-point operations (millions)
- `arithmetic_intensity` - FLOP/Byte
- `num_nodes` - Layer count
- `count_unique_ops` - Unique operation types
- `architecture_*` - One-hot encoded (K80, Tesla, Ampere, etc.)

### Model Types

All support same interface:
```python
model.fit(X_train, y_train)      # Train
y_pred = model.predict(X_test)   # Predict mean
quantiles = model.quantiles()    # Get quantile bounds
```

Returns quantile predictions:
```python
{
  "mean": [...],        # Expected power (W)
  "lower_q": [...],     # 25th percentile
  "upper_q": [...]      # 75th percentile
}
```

## 🔗 Dependencies

**Core** (installed by `uv sync`):
- torch 2.0+
- pandas 1.5+
- numpy 1.24+
- scikit-learn 1.3+
- optuna 3.0+ (hyperparameter tuning)
- catboost 1.2+ (one model type)
- joblib (model serialization)
- click (CLI)
- rich (terminal UI)
- flask (web app)

**Optional**:
- fvcore (advanced FLOP counting)

**Dev**:
- pytest (testing)
- black (formatting)
- ruff (linting)

## ❓ Troubleshooting

### "Module not found"
```bash
# Ensure dependencies
python -m pip install -r requirements.txt
# Or use build system
python build.py status
```

### "Experiment not found"
```bash
# List available
python build.py list-experiments
# exp06 requires creation first time
python build.py build-model exp06
```

### "Models not found"
```bash
# Check they were trained
ls Model\ Training/models/
# Should show exp06_quant_*.pkl files
```

### "Port 5000 already in use"
```bash
# Change port in www/app.py
app.run(port=5001)
# Or kill process
lsof -ti:5000 | xargs kill -9
```

## 📞 Support

For detailed information on specific subsystems, see:
- [Build System](BUILD_SYSTEM.md) - CLI documentation
- [KernelBench README](Dataset%20Collection/Dataset-2/Benchmark%20Suite/KernelBench/README.md) - Data collection
- [Model Training README](Model%20Training/README.md) - Training pipeline
- [Web App README](www/README.md) - Web interface

## 📝 License

See parent project for license details.
