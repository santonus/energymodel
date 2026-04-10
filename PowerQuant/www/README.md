# PowerQuant Web Application

A web-based tool for analyzing GPU kernel performance and predicting power consumption using machine learning models.

## Features

- 🎨 **Interactive Code Editor**: Write code with syntax highlighting
- 📊 **Power Prediction**: Estimate power consumption using 8 trained models:
  - **Base Models**: CatBoost, Neural Network, Random Forest, SVR
  - **Quantile Models**: Quantile-wrapped versions of above for uncertainty quantification
- 🎯 **Dual Experiment Support**:
  - **In-distribution**: Standard deployment (same architecture distribution)
  - **Out-of-distribution**: Robustness testing (leave-one-architecture-out)
- 📈 **Dataset Support**: Works with Dataset-1 (C++ CUDA) and Dataset-2 (PyTorch)
- ✅ **Model Analysis**: Real-time predictions with confidence intervals

## Installation

1. **Install dependencies:**
   ```bash
   cd www
   pip install -r requirements.txt
   ```

2. **Ensure trained models exist:**
   ```bash
   # From project root
   python build.py build-model in-distribution --dataset dataset-1
   ```

## Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser:**
   ```
   http://localhost:5000
   ```

## Usage

### 1. Load Pre-trained Models

The app automatically loads trained models from `../Model\ Training/models/`:

```
in_distribution_catboost.pkl
in_distribution_quant_neuralnet.pkl
out_of_distribution_arch0_svr.pkl
...etc
```

**Models must exist before starting the app. Train them using:**
```bash
python ../build.py build-model in-distribution --dataset dataset-1
python ../build.py build-model out-of-distribution --dataset dataset-2 --test-arch 0
```

### 2. Make Predictions

**In-Distribution (Standard):**
```json
Input: [feature1, feature2, ..., feature7/8]
Output: {
  "prediction": 157.3,  // watts
  "confidence_low": 145.2,
  "confidence_high": 169.4
}
```

**Out-of-Distribution (Generalization Test):**
```json
Input: [feature1, feature2, ..., architecture_index]
Output: {
  "prediction": 142.8,  // watts (from model trained without this architecture)
  "confidence_low": 131.5,
  "confidence_high": 154.1
}
```

### 2. Analyze the Model

Click **"Analyze & Predict"** to:
- Extract model features (FLOPs, memory usage, etc.)
- Get power consumption predictions from all 4 models
- View results with confidence intervals (lower, median, upper bounds)

### 3. Load Templates

Use the template dropdown to load pre-built examples:
- Conv2d + ReLU
- Conv2d + GELU
- Matrix Multiplication
- Scaled Dot Product Attention

## API Endpoints

## API Endpoints

### GET `/api/list-models`
List available trained models and their details.

**Response:**
```json
{
  "success": true,
  "models": {
    "in-distribution": {
      "catboost": "in_distribution_catboost.pkl",
      "quant_neuralnet": "in_distribution_quant_neuralnet.pkl",
      ...
    },
    "out-of-distribution": {
      "arch0_svr": "out_of_distribution_arch0_svr.pkl",
      ...
    }
  }
}
```

### POST `/api/predict`
Predict power consumption from features using trained models.

**Request:**
```json
{
  "experiment": "in-distribution",
  "features": {
    "total_bytes_read_mb": 123.45,
    "total_bytes_written_mb": 67.89,
    "total_flops_m": 1234.56,
    "arithmetic_intensity": 10.0,
    "num_nodes": 15,
    "count_unique_ops": 8
  },
  "models": ["catboost", "quant_neuralnet", "randomforest", "quant_svr"]
}
```

**Response:**
```json
{
  "success": true,
  "predictions": {
    "catboost": 157.3,
    "quant_neuralnet": 152.8,
    "randomforest": 161.2,
    "quant_svr": 155.9
  },
  "average": 156.8,
  "std": 3.2
}
```

### POST `/api/predict-ood`
Out-of-distribution prediction (test on unseen architecture).

**Request:**
```json
{
  "test_arch": 0,
  "features": {
    "total_bytes_read_mb": 123.45,
    ...
  }
}
```

## Project Structure

```
www/
├── app.py                 # Flask backend with model loading
├── templates/
│   └── index.html        # Frontend UI
├── models/               # Symlink to trained models (optional)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Integrating Trained Models

The app automatically loads trained models on startup. To ensure models are available:

1. **Train models from project root:**
```bash
python build.py build-model in-distribution --dataset dataset-1
python build.py build-model out-of-distribution --dataset dataset-2 --test-arch 0
```

2. **Models are loaded from:**
```
../Model\ Training/models/
  ├── in_distribution_catboost.pkl
  ├── in_distribution_quant_neuralnet.pkl
  ├── in_distribution_quant_randomforest.pkl
  ├── in_distribution_quant_svr.pkl
  ├── out_of_distribution_arch0_catboost.pkl
  ├── out_of_distribution_arch0_quant_neuralnet.pkl
  ...etc
```

3. **In `app.py`, models are loaded like:**
```python
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "Model Training" / "models"

# Load in-distribution models
in_dist_models = {
    'catboost': joblib.load(MODELS_DIR / 'in_distribution_catboost.pkl'),
    'quant_neuralnet': joblib.load(MODELS_DIR / 'in_distribution_quant_neuralnet.pkl'),
    ...
}

@app.route('/api/predict', methods=['POST'])
def predict_power():
    data = request.get_json()
    X = np.array([data['features'].values()])
    
    predictions = {}
    for model_name, model in in_dist_models.items():
        predictions[model_name] = float(model.predict(X)[0])
    
    return jsonify({
        'success': True,
        'predictions': predictions,
        'average': np.mean(list(predictions.values()))
    })
```

## Troubleshooting

### Models not found on startup
```
Error: ModuleNotFoundError: No module named 'in_distribution_catboost.pkl'
```

**Solution:**
```bash
# Train models first
python build.py build-model in-distribution --dataset dataset-1

# Verify models exist
ls ../Model\ Training/models/
```

### Port already in use
```bash
# Change port in app.py
app.run(port=5001)

# Or kill existing process
lsof -ti:5000 | xargs kill -9
```

### Feature mismatch error
- Ensure input features match training features (7 or 8 depending on dataset)
- Check Dataset-1 vs Dataset-2 feature lists in Model Training/experiments/*/utils.py

## Development

To enable debug mode and auto-reload:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## Deployment (Production)

Using Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Using Docker:
```bash
docker build -t powerquant-web .
docker run -p 5000:5000 powerquant-web
```

## License

MIT License - see parent project for details

## Credits

Built on top of PowerQuant and KernelBench projects.
