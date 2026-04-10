# KernelBench Dataset Collection

KernelBench is the PyTorch benchmark suite for collecting GPU kernel performance data including power consumption measurements.

## Overview

KernelBench benchmarks PyTorch operations across multiple NVIDIA GPU architectures and collects:
- **Performance metrics** - Execution time, FLOP count, memory access patterns
- **Power metrics** - GPU power consumption during execution
- **Architecture profiles** - Device properties (memory, compute capability)

This data feeds the PowerQuant model training pipeline.

## Directory Structure

```
Dataset Collection/Dataset-2(works for Python PyTorch Models)/
└── Benchmark Suite/
    └── KernelBench/
        ├── README.md                    # This file
        ├── scripts/
        │   ├── generate_baseline_time.py    # Main data collection script
        │   ├── extract_model_features.py    # Feature extraction (FLOPs, memory)
        │   └── variable_scaler.py           # Data scaling utilities
        ├── src/
        │   └── kernelbench/
        │       ├── __init__.py
        │       ├── dataset.py              # Dataset loading
        │       ├── eval.py                 # Model evaluation
        │       └── utils.py                # Helper functions
        ├── operations/                  # PyTorch op benchmarks
        ├── models/                      # Model definitions
        └── configs/                     # Benchmark configurations
```

## Quick Start

### Collect Baseline Timing Data

```bash
cd /path/to/PowerQuant
python build.py collect-dataset
```

This runs `Dataset Collection/Dataset-2/Benchmark Suite/KernelBench/scripts/generate_baseline_time.py` which:
1. Loads PyTorch models from the benchmarks
2. Measures execution time on available GPUs
3. Records power consumption (if PowerAPI available)
4. Generates CSV output to `Dataset-2/Dataset/`

### Output

Generated CSV files contain:
```
model_filename,kernel_name,input_shape,output_shape,...
                total_bytes_read_mb,total_bytes_written_mb,...
                total_flops_m,arithmetic_intensity,num_nodes,...
                power_consumption
```

Example: `combined_static_20260127_092347.csv`

## Scripts

### `generate_baseline_time.py`

**Purpose**: Collect baseline execution times and power measurements for GPU kernels.

**Usage**:
```bash
cd Dataset-2/Benchmark\ Suite/KernelBench
python scripts/generate_baseline_time.py

# Or via build system
python ../../../build.py collect-dataset
```

**What it does**:
1. Discovers available PyTorch operations
2. Creates benchmark datasets for each operation
3. Runs benchmarks on available GPUs
4. Measures execution time and power consumption
5. Aggregates results into CSV

**Output**:
- `Dataset/combined_*.csv` - Full combined results
- `Dataset/combined_*_static.csv` - Static features only (recommended)
- Logs to stdout during execution

**Key functions**:
```python
from src.kernelbench.eval import evaluate_model()

# Evaluate a single model
metrics = evaluate_model(model, device, input_shape)
# Returns: {
#   "execution_time": float,
#   "power_consumption": float,
#   "memory_peak": float,
#   ...
# }
```

### `extract_model_features.py`

**Purpose**: Extract computational features from PyTorch models (FLOPs, memory patterns, graph structure).

**Usage**:
```bash
python scripts/extract_model_features.py <model_name> [--input-shape 1 3 224 224]
```

**What it does**:
1. Uses `fvcore` to count FLOPs
2. Analyzes memory read/write patterns (cumulative, not peak)
3. Extracts graph structure (nodes, unique operations)
4. Computes arithmetic intensity ($\frac{\text{FLOPs}}{\text{Bytes Transferred}}$)

**Key functions**:
```python
from fvcore.nn import FlopCounterMode

with FlopCounterMode(model) as fcm:
    output = model(input_tensor)
    flops = fcm.flop_counts

# Memory calculation (cumulative model)
memory = sum(read_sizes) + sum(write_sizes)  # Not peak!
```

**Supported operations**: 100+ custom handlers for:
- Convolution layers (Conv1d/2d/3d, grouped, dilated)
- Transpose operations
- Batch normalization
- Pooling (avg, max, adaptive)
- Activation functions
- Linear/MLP layers
- Attention mechanisms
- And more...

### `variable_scaler.py`

**Purpose**: Normalize and scale numeric features for ML training.

**Usage**:
```python
from scripts.variable_scaler import scale_features, inverse_scale

X_scaled = scale_features(X, method='standardize')
X_original = inverse_scale(X_scaled, method='standardize')
```

**Scaling methods**:
- `standardize` - Z-score normalization: $(x - \mu) / \sigma$
- `minmax` - Min-max scaling: $(x - x_{min}) / (x_{max} - x_{min})$
- `robust` - Robust scaling using IQR: $(x - \text{median}) / \text{IQR}$

## Data Flow

```
PyTorch Models
    ↓
generate_baseline_time.py
    ├─→ extract_model_features.py
    │   ├─→ FLOPs counting (fvcore)
    │   ├─→ Memory analysis
    │   └─→ Graph extraction
    ├─→ GPU Benchmarking
    │   ├─→ Execution time
    │   └─→ Power measurement (PowerAPI)
    ↓
combined_*.csv
    ↓
Model Training/data → Exp01-06
    ↓
Trained Models
```

## Dataset Format

### CSV Columns

**Model Information**:
- `model_filename` - PyTorch model source file
- `model_name` - Model class name
- `architecture` - GPU architecture (K80, Tesla, Ampere, etc.)

**Computation**:
- `total_flops_m` - Total floating-point operations (millions)
- `num_nodes` - Graph nodes (layers)
- `count_unique_ops` - Unique operation types

**Memory** (cumulative model):
- `total_bytes_read_mb` - Total data read (MB)
- `total_bytes_written_mb` - Total data written (MB)
- `total_bytes_mb` - Combined read + write

**Efficiency**:
- `arithmetic_intensity` - FLOPs per byte transferred
- `memory_efficiency` - Actual vs. theoretical memory bandwidth

**Performance**:
- `execution_time_ms` - Time to run kernel (milliseconds)
- `throughput_gflops` - Achieved GFLOP/s

**Power** (if available):
- `power_consumption` - GPU power draw (watts)
- `energy_joules` - Total energy used (joules)

### Example Row

```csv
Conv2d_3x3_ReLU.py,Conv2d,K80,2048,64,3072,...
150000,256,45,...
1024,512,1536,...
6.25,...
45.2,...
```

## GPU Support

KernelBench benchmarks NVIDIA GPUs:

| Architecture | Model | Compute Cap | Memory | Status |
|-------------|-------|------------|--------|--------|
| **Kepler** | K80 | 3.7 | 12 GB | ✓ Supported |
| **Maxwell** | GTX 980 | 5.2 | 4 GB | ✓ Supported |
| **Pascal** | P100 | 6.0 | 16 GB | ✓ Supported |
| **Volta** | V100 | 7.0 | 32 GB | ✓ Supported |
| **Turing** | RTX 2080 | 7.5 | 11 GB | ✓ Supported |
| **Ampere** | RTX 3090 | 8.6 | 24 GB | ✓ Supported |
| **Ada** | RTX 4090 | 8.9 | 24 GB | ✓ Supported |

Run on available hardware - hardware detection is automatic.

## Performance Metrics

### Memory Model

PowerQuant uses a **cumulative data movement model**, not peak memory:

$$\text{Memory} = \sum_{\text{layers}} (\text{bytes\_read} + \text{bytes\_written})$$

This captures sustained bandwidth usage and is more predictive of power than peak memory.

### FLOP Counting

Uses `fvcore.nn.FlopCounterMode` with 100+ custom operation handlers:

$$\text{FLOPs} = \sum_{\text{ops}} \text{flops\_per\_op}(W, H, C, K, \ldots)$$

Examples:
- **Conv2d**: $2 \times H \times W \times C \times K_h \times K_w \times K_c$
- **Linear**: $2 \times N \times M \times K$
- **BatchNorm**: $N \times H \times W \times C$

### Arithmetic Intensity

Roofline model metric - compute efficiency ceiling:

$$I = \frac{\text{FLOPs}}{\text{Bytes Transferred}} \quad [\text{FLOP/Byte}]$$

Higher intensity = better compute utilization.

## Integration with Model Training

The collected dataset is used for:

1. **Feature Engineering** - FLOPs, memory, intensity become features
2. **Power Prediction** - Regression target across all models
3. **Transfer Learning** - Architecture profiles for cross-GPU generalization
4. **Benchmarking** - Baseline comparisons between models

The pipeline automatically:
```
Dataset → Model Training/data/ → prepare_data() → 
  Feature selection → Model Training → Predictions
```

## Troubleshooting

### "No CUDA GPUs found"
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"
```

### PowerAPI Module Not Found
- Install: `pip install powerapi`
- Or disable power measurements: Comment out power measurement in script

### Out of Memory During Benchmarking
- Reduce batch sizes in configs
- Benchmark fewer models at once
- Use smaller input shapes

### FLOP Count Mismatch
- Ensure model input shapes match expected dimensions
- Check operation handlers in `extract_model_features.py`
- Add custom handlers for new operations

### CSV Not Generated
```bash
# Check output directory
ls -la Dataset/
# Should contain combined_*.csv files

# Check logs for errors
python scripts/generate_baseline_time.py 2>&1 | grep -i error
```

## Advanced Usage

### Custom Benchmark Configuration

Edit `src/kernelbench/configs/` to add custom benchmark parameters.

### Add New Operations

Extend `extract_model_features.py` with custom FLOP handlers:

```python
def count_custom_op(inputs, outputs):
    """Count FLOPs for custom operation."""
    # Calculate based on inputs/outputs shapes
    flops = ...
    return flops

# Register handler
register_flop_counter_function(CustomOp, count_custom_op)
```

### Offline Benchmarking

For reproducible results without live GPUs:

```python
# Load cached results
from src.kernelbench.dataset import load_cached_benchmarks
benchmarks = load_cached_benchmarks("Dataset/combined_static_20260127_092347.csv")
```

## Related Documentation

- [Build System](../BUILD_SYSTEM.md) - How to run data collection
- [Model Training](../Model%20Training/README.md) - Uses this data
- [PowerAPI Dataset-1](../Dataset-1/README.md) - C++ benchmark alternative
- [NVIDIA fvcore docs](https://github.com/facebookresearch/fvcore) - FLOP counting
