# KernelBench - PyTorch GPU Benchmark Suite

GPU benchmark suite for collecting performance metrics and power consumption data from PyTorch models.

## Overview

KernelBench profiles PyTorch operations across NVIDIA GPUs and extracts computational features including:
- **Execution time** - Kernel runtime performance
- **FLOPs** - Floating-point operations (via fvcore with 100+ custom handlers)
- **Memory** - Cumulative data movement (read + write)
- **Arithmetic Intensity** - Compute efficiency metric
- **Graph Structure** - Number of layers and unique operations
- **Power Consumption** - GPU power draw during execution

## Quick Start

### Collect Baseline Data

```bash
cd /path/to/PowerQuant
python build.py collect-dataset
```

This runs `generate_baseline_time.py` which benchmarks PyTorch models and generates CSV output.

### Output Format

Generated CSV files contain:
```
model_filename,architecture,total_flops_m,total_bytes_read_mb,
total_bytes_written_mb,arithmetic_intensity,num_nodes,
count_unique_ops,power_consumption
```

Example: `combined_static_20260127_092347.csv`

## Directory Structure

```
Benchmark Suite/KernelBench/
├── README.md                           # This file
├── scripts/
│   ├── generate_baseline_time.py      # Main data collection
│   ├── extract_model_features.py      # Feature extraction
│   └── variable_scaler.py             # Feature normalization
├── src/kernelbench/
│   ├── __init__.py
│   ├── dataset.py                     # Dataset loading
│   ├── eval.py                        # Model evaluation
│   └── utils.py                       # Helper functions
├── operations/                        # PyTorch op benchmarks
├── models/                            # Model definitions
└── configs/                           # Benchmark configurations
```

## Main Scripts

### `generate_baseline_time.py`

**Purpose**: Collect execution time and power measurements for GPU kernels

**Usage**:
```bash
python scripts/generate_baseline_time.py
# Or via build system
python ../../../build.py collect-dataset
```

**What it does**:
1. Discovers PyTorch operations in `operations/`
2. Loads model definitions from `models/`
3. Benchmarks each model on available GPUs
4. Measures execution time
5. Records power consumption (if PowerAPI available)
6. Outputs combined CSV file

**Key functions**:
```python
from src.kernelbench.eval import evaluate_model

metrics = evaluate_model(model, device, input_shape)
# Returns: {
#   "execution_time": float,
#   "power_consumption": float,
#   "memory_peak": float,
#   ...
# }
```

### `extract_model_features.py`

**Purpose**: Extract computational features from PyTorch models

**Usage**:
```bash
python scripts/extract_model_features.py <model_name> [--input-shape 1 3 224 224]
```

**What it calculates**:
- Total FLOPs using `fvcore` with custom operation handlers
- Memory read/write patterns (cumulative, not peak)
- Graph structure (nodes, unique operations)
- Arithmetic intensity

**Supported operations** (100+):
- Convolution (Conv1d/2d/3d, grouped, dilated)
- Transpose convolutions
- Batch normalization
- Pooling (max, avg, adaptive)
- Activation functions
- Linear layers
- Attention mechanisms
- Reshape/permute operations
- And more...

**Example**:
```python
from fvcore.nn import FlopCounterMode

with FlopCounterMode(model) as fcm:
    output = model(input_tensor)
    flops = fcm.flop_counts
    # flops[model] = total FLOPs
```

### `variable_scaler.py`

**Purpose**: Normalize numeric features for ML training

**Scaling methods**:
- `standardize` - Z-score: $(x - \mu) / \sigma$
- `minmax` - Min-max: $(x - x_{min}) / (x_{max} - x_{min})$
- `robust` - IQR-based: $(x - \text{median}) / \text{IQR}$

**Usage**:
```python
from scripts.variable_scaler import scale_features

X_scaled = scale_features(X, method='standardize')
X_original = inverse_scale(X_scaled, method='standardize')
```

## Data Pipeline

```
PyTorch Models
    ↓
generate_baseline_time.py
├─→ Load models from models/
├─→ Create benchmark inputs
├─→ Benchmark on GPU(s)
│
├─→ extract_model_features.py
│   ├─→ Count FLOPs (fvcore)
│   ├─→ Analyze memory
│   └─→ Extract graph structure
│
├─→ Measure execution time
├─→ Record power (PowerAPI)
└─→ Aggregate results
    ↓
combined_*.csv
    ↓
Model Training/data/ (via symlink)
```

## Feature Definitions

### Computation Features

| Feature | Type | Units | Description |
|---------|------|-------|-------------|
| `total_flops_m` | Numeric | Millions | Total floating-point operations |
| `num_nodes` | Numeric | Count | Number of layers in model |
| `count_unique_ops` | Numeric | Count | Number of unique operation types |

### Memory Features (Cumulative Model)

| Feature | Type | Units | Description |
|---------|------|-------|-------------|
| `total_bytes_read_mb` | Numeric | MB | Total data read from memory |
| `total_bytes_written_mb` | Numeric | MB | Total data written to memory |
| `total_bytes_mb` | Numeric | MB | Combined read + write |

**Note**: Uses cumulative model, not peak memory. Better predictor of power consumption.

### Efficiency Features

| Feature | Type | Units | Description |
|---------|------|-------|-------------|
| `arithmetic_intensity` | Numeric | FLOP/Byte | Compute per unit bandwidth |
| `memory_efficiency` | Numeric | % | Actual vs theoretical bandwidth |

### Target Feature

| Feature | Type | Units | Description |
|---------|------|-------|-------------|
| `power_consumption` | Numeric | Watts | GPU power draw during execution |

## GPU Support

Benchmarks NVIDIA GPUs with automatic detection:

| Architecture | Example | Compute Capability | Memory | Status |
|-------------|---------|-------------------|--------|--------|
| **Kepler** | K80 | 3.7 | 12 GB | ✓ Supported |
| **Maxwell** | GTX 980 | 5.2 | 4 GB | ✓ Supported |
| **Pascal** | P100 | 6.0 | 16 GB | ✓ Supported |
| **Volta** | V100 | 7.0 | 32 GB | ✓ Supported |
| **Turing** | RTX 2080 | 7.5 | 11 GB | ✓ Supported |
| **Ampere** | RTX 3090 | 8.6 | 24 GB | ✓ Supported |
| **Ada** | RTX 4090 | 8.9 | 24 GB | ✓ Supported |

## Memory Model

PowerQuant uses **cumulative data movement**, not peak memory:

$$\text{Memory} = \sum_{\text{layers}} (\text{bytes\_read} + \text{bytes\_written})$$

This metric captures sustained bandwidth usage and is more predictive of power consumption than peak memory allocations.

## FLOP Counting

Uses `fvcore.nn.FlopCounterMode` with 100+ custom operation handlers:

$$\text{FLOPs} = \sum_{\text{ops}} \text{flops\_per\_op}(W, H, C, K, \ldots)$$

Examples:
- **Conv2d**: $2 \times H \times W \times C \times K_h \times K_w \times K_c$
- **Linear**: $2 \times N \times M \times K$
- **BatchNorm**: $N \times H \times W \times C$

## Arithmetic Intensity

Roofline model metric measuring compute efficiency:

$$I = \frac{\text{FLOPs}}{\text{Bytes Transferred}} \quad [\text{FLOP/Byte}]$$

Higher intensity indicates better GPU utilization relative to memory bandwidth limits.

## Integration with Model Training

The collected dataset is used by Model Training pipeline:

```
Dataset CSV (Dataset/)
    ↓
Model Training/data/ (symlink)
    ↓
experiments/exp06/prepare_data()
├─→ Dynamic column selection
├─→ Filter to numeric columns
├─→ One-hot encode architecture
└─→ Split train/test
    ↓
Train 4 quantile models
    ↓
Power predictions
```

## Troubleshooting

### No CUDA GPUs Found

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"

# Install NVIDIA drivers if needed
```

### PowerAPI Module Not Found

```bash
# Install (optional, for power measurements)
pip install powerapi

# Or comment out power measurement code in generate_baseline_time.py
```

### Out of Memory During Benchmarking

- Reduce batch sizes in benchmark config
- Benchmark fewer models at once
- Use smaller input tensor dimensions
- Manually select specific architectures

### FLOP Count Mismatch

- Verify input tensor shapes match model expectations
- Check for custom operations needing handlers
- Test with known baseline models first
- Add custom FLOP handlers if needed

### CSV Not Generated

```bash
# Check output directory
ls -la Dataset/
# Should contain combined_*.csv files

# Check for errors
python scripts/generate_baseline_time.py 2>&1 | grep -i error
```

## Advanced Usage

### Custom Benchmark Configuration

Edit `configs/` files to add custom parameters:
- Model selection
- Batch sizes
- Input dimensions
- Benchmark iterations
- GPU selection

### Add New Operations

Extend `extract_model_features.py` with custom FLOP handlers:

```python
def count_custom_op(inputs, outputs):
    """Count FLOPs for custom operation."""
    flops = calculate_flops(inputs, outputs)
    return flops

# Register handler
from fvcore.nn import register_flop_counter_function
register_flop_counter_function(CustomOp, count_custom_op)
```

### Offline Benchmarking

Load cached results without re-benchmarking:

```python
from src.kernelbench.dataset import load_cached_benchmarks

benchmarks = load_cached_benchmarks("Dataset/combined_static_20260127_092347.csv")
```

### Custom Dataset Specifications

Modify `configs/` to benchmark specific model architectures or problem sizes.

## Performance Notes

### Benchmark Runtime

- **Small models** (MobileNet): ~1-5 minutes per GPU
- **Large models** (ResNet-152): ~5-10 minutes per GPU
- **Multiple GPUs**: Runs sequentially per GPU

### Memory Usage

- Benchmark process: ~2-4 GB RAM
- GPU memory: Varies by model (typically 2-8 GB)
- Output CSV: ~10-100 MB (depending on model count)

### Data Collection Tips

1. Run during low-system-load periods
2. Disable other GPU workloads
3. Use power monitoring for accurate measurements
4. Run multiple times for consistency
5. Document GPU driver and CUDA versions

## API Reference

### Main Evaluation Function

```python
from src.kernelbench.eval import evaluate_model

metrics = evaluate_model(
    model,                    # torch.nn.Module
    device,                   # torch.device
    input_shape,              # Tuple[int, ...]
    num_iterations=10,        # Benchmark iterations
    warmup_iterations=2,      # Warmup runs
)

# Returns dict with keys:
# - execution_time_ms
# - throughput_gflops
# - power_consumption
# - memory_peak
# - ...
```

### Feature Extraction

```python
from scripts.extract_model_features import extract_features

features = extract_features(
    model,                    # torch.nn.Module
    input_shape,              # Tuple[int, ...]
    use_fvcore=True,          # Use fvcore for FLOPs
)

# Returns dict with:
# - total_flops_m
# - total_bytes_read_mb
# - total_bytes_written_mb
# - arithmetic_intensity
# - num_nodes
# - count_unique_ops
```

## Related Documentation

- [Build System](../../../BUILD_SYSTEM.md) - How to run data collection
- [Model Training](../../../Model%20Training/README.md) - Uses this data
- [Main README](../../../README.md) - Project overview
- [NVIDIA fvcore docs](https://github.com/facebookresearch/fvcore) - FLOP counting library
