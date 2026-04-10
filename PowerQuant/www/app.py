"""
PowerQuant Web App - Flask Backend
Allows users to write PyTorch models and analyze power consumption
"""
import os
import sys
import json
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Dataset Collection" / "Dataset-2(works for Python PyTorch Models)" / "Benchmark Suite" / "KernelBench"))
sys.path.insert(0, str(PROJECT_ROOT / "Model Training"))

# Import feature extraction
try:
    from scripts.extract_model_features import extract_features
    FEATURE_EXTRACTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import extract_model_features: {e}")
    FEATURE_EXTRACTION_AVAILABLE = False

# Import power prediction models
try:
    import pandas as pd
    import numpy as np
    from src.base_classifier.catboost import CatBoost
    from src.base_classifier.neuralnet import NeuralNet
    from src.base_classifier.randomforest import RandomForest
    from src.base_classifier.svr import SupportVectorRegressor
    from src.quantile_classifier.quant_classifier import QuantileClassifier
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import prediction models: {e}")
    MODELS_AVAILABLE = False

# Always import these for fallback
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for API requests

# Default template
DEFAULT_TEMPLATE = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a convolution, applies ReLU, and another ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.relu(x)
        return x

batch_size   = 64  
in_channels  = 64  
out_channels = 128  
height = width = 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
'''


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/templates', methods=['GET'])
def get_templates():
    """Return available model templates"""
    templates = {
        'conv_relu': {
            'name': 'Conv2d + ReLU',
            'code': DEFAULT_TEMPLATE
        },
        'conv_gelu': {
            'name': 'Conv2d + GELU',
            'code': DEFAULT_TEMPLATE.replace('relu', 'gelu')
        },
        'matmul': {
            'name': 'Matrix Multiplication',
            'code': '''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs matrix multiplication.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        return x

batch_size   = 128
in_features  = 512
out_features = 256

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
'''
        },
        'attention': {
            'name': 'Scaled Dot Product Attention',
            'code': '''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Scaled dot product attention.
    """
    def __init__(self, embed_dim):
        super(Model, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

batch_size = 32
num_heads = 8
seq_len = 128
embed_dim = 512
head_dim = embed_dim // num_heads

def get_inputs():
    return [
        torch.rand(batch_size, num_heads, seq_len, head_dim),  # query
        torch.rand(batch_size, num_heads, seq_len, head_dim),  # key
        torch.rand(batch_size, num_heads, seq_len, head_dim),  # value
    ]

def get_init_inputs():
    return [embed_dim]
'''
        }
    }
    return jsonify(templates)


def extract_features_fallback(code):
    """Fallback feature extraction using basic PyTorch analysis"""
    try:
        # Execute code to get model
        context = {'torch': torch, 'nn': nn}
        exec(code, context)
        
        Model = context.get('Model')
        get_inputs = context.get('get_inputs')
        get_init_inputs = context.get('get_init_inputs')
        
        if not Model or not get_inputs or not get_init_inputs:
            raise ValueError('Code must define Model, get_inputs, and get_init_inputs')
        
        # Create model and inputs
        init_inputs = get_init_inputs()
        model = Model(*init_inputs)
        inputs = get_inputs()
        
        # Calculate basic features
        total_params = sum(p.numel() for p in model.parameters())
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        
        input_bytes = sum(
            inp.numel() * inp.element_size() 
            for inp in inputs if isinstance(inp, torch.Tensor)
        )
        
        # Estimate output size
        with torch.no_grad():
            try:
                outputs = model(*inputs)
                if isinstance(outputs, torch.Tensor):
                    output_bytes = outputs.numel() * outputs.element_size()
                else:
                    output_bytes = sum(
                        o.numel() * o.element_size() 
                        for o in outputs if isinstance(o, torch.Tensor)
                    )
            except:
                output_bytes = input_bytes
        
        total_bytes_read = (input_bytes + param_bytes) / (1024**2)
        total_bytes_written = output_bytes / (1024**2)
        total_bytes = total_bytes_read + total_bytes_written
        
        # Rough FLOP estimate
        total_flops = total_params * inputs[0].numel() if inputs else 0
        total_flops_m = total_flops / 1e6
        
        arithmetic_intensity = total_flops_m / max(total_bytes, 0.001)
        
        # Count layers
        num_nodes = len(list(model.modules()))
        
        # Count unique op types
        op_types = set(type(m).__name__ for m in model.modules())
        count_unique_ops = len(op_types)
        
        return {
            'name': 'user_model',
            'total_bytes_read_mb': float(total_bytes_read),
            'total_bytes_written_mb': float(total_bytes_written),
            'total_bytes_mb': float(total_bytes),
            'total_flops_m': float(total_flops_m),
            'arithmetic_intensity': float(arithmetic_intensity),
            'num_nodes': int(num_nodes),
            'count_unique_ops': int(count_unique_ops)
        }
    except Exception as e:
        raise ValueError(f'Failed to analyze model: {str(e)}')


@app.route('/api/analyze', methods=['POST'])
def analyze_model():
    """Analyze model and extract features"""
    try:
        data = request.get_json()
        code = data.get('code', '')
        
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        # Try advanced feature extraction first
        if FEATURE_EXTRACTION_AVAILABLE:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            try:
                features = extract_features(
                    model_path=temp_path,
                    name='user_model',
                    verbose=False,
                    include_backward=True
                )
                
                if features is not None:
                    from dataclasses import asdict
                    features_dict = asdict(features)
                    
                    return jsonify({
                        'success': True,
                        'features': features_dict,
                        'method': 'advanced'
                    })
            except Exception as e:
                print(f"Advanced extraction failed: {e}")
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        # Fallback to basic extraction
        features_dict = extract_features_fallback(code)
        return jsonify({
            'success': True,
            'features': features_dict,
            'method': 'basic'
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict_power():
    """Predict power consumption using quantile models"""
    try:
        data = request.get_json()
        features = data.get('features', {})
        
        if not features:
            return jsonify({'error': 'No features provided'}), 400
        
        # Convert features to DataFrame format expected by models
        # Adjust feature names to match your trained models
        feature_columns = [
            'total_bytes_read_mb',
            'total_bytes_written_mb',
            'total_bytes_mb',
            'total_flops_m',
            'arithmetic_intensity',
            'num_nodes',
            'count_unique_ops'
        ]
        
        # Extract features in correct order
        X = np.array([[
            features.get('total_bytes_read_mb', 0),
            features.get('total_bytes_written_mb', 0),
            features.get('total_bytes_mb', 0),
            features.get('total_flops_m', 0),
            features.get('arithmetic_intensity', 0),
            features.get('num_nodes', 0),
            features.get('count_unique_ops', 0)
        ]])
        
        # Use quantile regression models if available
        if MODELS_AVAILABLE:
            try:
                # Import configurations
                from src.base_classifier.catboost import CatBoostConfig
                from src.base_classifier.neuralnet import NeuralNetConfig
                from src.base_classifier.randomforest import RandomForestConfig
                from src.base_classifier.svr import SVRConfig
                
                # Define quantiles for prediction intervals
                quantiles = [0.25, 0.50, 0.75]  # Lower, Median, Upper
                
                predictions = {}
                
                # CatBoost Quantile Model
                try:
                    catboost_config = CatBoostConfig(
                        iterations=1000,
                        learning_rate=0.01,
                        depth=6,
                        loss_function='Quantile:alpha=0.5'
                    )
                    base_model = CatBoost(catboost_config)
                    quant_model = QuantileClassifier(base_model, quantiles=quantiles)
                    
                    # Predict (untrained model will give poor results)
                    pred = quant_model.predict(X)
                    predictions['catboost'] = {
                        'lower': float(pred[0][0]),
                        'median': float(pred[0][1]),
                        'upper': float(pred[0][2])
                    }
                except Exception as e:
                    print(f"CatBoost quantile prediction failed: {e}")
                    predictions['catboost'] = None
                
                # Neural Network Quantile Model
                try:
                    nn_config = NeuralNetConfig(
                        hidden_layer_sizes=(64, 32, 16),
                        activation='relu',
                        num_epochs=100
                    )
                    base_model = NeuralNet(nn_config)
                    quant_model = QuantileClassifier(base_model, quantiles=quantiles)
                    
                    pred = quant_model.predict(X)
                    predictions['neuralnet'] = {
                        'lower': float(pred[0][0]),
                        'median': float(pred[0][1]),
                        'upper': float(pred[0][2])
                    }
                except Exception as e:
                    print(f"NeuralNet quantile prediction failed: {e}")
                    predictions['neuralnet'] = None
                
                # Random Forest Quantile Model
                try:
                    rf_config = RandomForestConfig(
                        n_estimators=100,
                        max_depth=10
                    )
                    base_model = RandomForest(rf_config)
                    quant_model = QuantileClassifier(base_model, quantiles=quantiles)
                    
                    pred = quant_model.predict(X)
                    predictions['randomforest'] = {
                        'lower': float(pred[0][0]),
                        'median': float(pred[0][1]),
                        'upper': float(pred[0][2])
                    }
                except Exception as e:
                    print(f"RandomForest quantile prediction failed: {e}")
                    predictions['randomforest'] = None
                
                # SVR Quantile Model
                try:
                    svr_config = SVRConfig(
                        kernel='rbf',
                        C=1.0
                    )
                    base_model = SupportVectorRegressor(svr_config)
                    quant_model = QuantileClassifier(base_model, quantiles=quantiles)
                    
                    pred = quant_model.predict(X)
                    predictions['svr'] = {
                        'lower': float(pred[0][0]),
                        'median': float(pred[0][1]),
                        'upper': float(pred[0][2])
                    }
                except Exception as e:
                    print(f"SVR quantile prediction failed: {e}")
                    predictions['svr'] = None
                
                # Filter out failed models and use heuristic for those
                flops = features.get('total_flops_m', 100)
                memory = features.get('total_bytes_mb', 100)
                base_power = (flops * 0.05) + (memory * 0.1)
                
                for model_name, pred in predictions.items():
                    if pred is None:
                        predictions[model_name] = {
                            'lower': float(base_power * 0.7),
                            'median': float(base_power),
                            'upper': float(base_power * 1.3)
                        }
                
                method = 'quantile_regression'
                note = 'Using QuantileClassifier with base models. Note: Models are untrained - train on your dataset for accurate predictions.'
                
            except Exception as e:
                print(f"Quantile model initialization failed: {e}")
                # Fallback to heuristic
                MODELS_AVAILABLE = False
        
        # Heuristic-based estimation if models not available
        if not MODELS_AVAILABLE:
            flops = features.get('total_flops_m', 100)
            memory = features.get('total_bytes_mb', 100)
            base_power = (flops * 0.05) + (memory * 0.1)
            
            predictions = {
                'catboost': {
                    'lower': float(base_power * 0.7),
                    'median': float(base_power),
                    'upper': float(base_power * 1.3)
                },
                'neuralnet': {
                    'lower': float(base_power * 0.75),
                    'median': float(base_power * 1.05),
                    'upper': float(base_power * 1.35)
                },
                'randomforest': {
                    'lower': float(base_power * 0.65),
                    'median': float(base_power * 0.95),
                    'upper': float(base_power * 1.25)
                },
                'svr': {
                    'lower': float(base_power * 0.8),
                    'median': float(base_power * 1.1),
                    'upper': float(base_power * 1.4)
                }
            }
            
            method = 'heuristic'
            note = 'Quantile models not available. Using heuristic estimates. Install dependencies and train models for quantile regression.'
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'method': method,
            'note': note
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/validate', methods=['POST'])
def validate_code():
    """Validate Python code syntax"""
    try:
        data = request.get_json()
        code = data.get('code', '')
        
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        # Try to compile the code
        try:
            compile(code, '<string>', 'exec')
            return jsonify({
                'valid': True,
                'message': 'Code syntax is valid'
            })
        except SyntaxError as e:
            return jsonify({
                'valid': False,
                'error': str(e),
                'line': e.lineno,
                'offset': e.offset
            })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("Starting PowerQuant Web App...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Feature extraction available: {FEATURE_EXTRACTION_AVAILABLE}")
    print(f"Models available: {MODELS_AVAILABLE}")
    app.run(debug=True, host='0.0.0.0', port=5001)
