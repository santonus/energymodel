#!/usr/bin/env python3
"""
Extract model features from KernelBench models using fvcore for static analysis.

Features extracted:
- name: Model name from file path
- total_bytes_read_mb: Estimated memory read (MB) - from input/param tensor sizes
- total_bytes_written_mb: Estimated memory written (MB) - from output/grad tensor sizes
- total_bytes_mb: Total memory accessed (MB)
- total_flops_m: Total FLOPs in millions (forward only, multiply by ~3 for fwd+bwd)
- arithmetic_intensity: FLOPs / bytes
- num_nodes: Number of nodes in the computation graph
- count_unique_ops: Number of unique operation types

Usage:
    python scripts/extract_model_features.py --level 1 --output features.csv
    python scripts/extract_model_features.py --all --output all_features.csv
    python scripts/extract_model_features.py --file KernelBench/level1/19_ReLU.py
    
Requirements:
    pip install fvcore
"""

import argparse
import csv
import os
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Tuple, Any

import torch
import torch.nn as nn

# Add the src directory to the path
REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_TOP_PATH, "src"))

from kernelbench.utils import read_file

# =============================================================================
# Variable Scaling (inline implementation from variable_scaler.py)
# =============================================================================

def _vars_used_by_get_inputs_init_inputs(source: str) -> set:
    """
    Discover variable names to scale: module-level assignments that are
    referenced in get_inputs or get_init_inputs.
    """
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    module_level_vars = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    module_level_vars.add(t.id)

    used_in_fns = set()
    locals_in_fns = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in ("get_inputs", "get_init_inputs"):
            continue
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                if isinstance(n.ctx, ast.Store):
                    locals_in_fns.add(n.id)
                elif isinstance(n.ctx, ast.Load):
                    used_in_fns.add(n.id)

    return module_level_vars & (used_in_fns - locals_in_fns)


def scale_source_code(source_code: str, multiplier: float = 1.0) -> str:
    """
    Scale variables in source code that are used by get_inputs/get_init_inputs.
    """
    import re
    
    scalable = _vars_used_by_get_inputs_init_inputs(source_code)
    if not scalable or multiplier == 1.0:
        return source_code
    
    # Find variables and their values
    variables = {}
    pattern = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$')
    lines = source_code.split('\n')
    context = {}

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if line.startswith(' ') or line.startswith('\t'):
            continue
        match = pattern.match(stripped)
        if not match or match.group(1) not in scalable:
            continue
        var_name = match.group(1)
        expression = match.group(2).rstrip()
        safe = {'__builtins__': __builtins__, 'torch': torch}
        try:
            value = eval(expression, safe, context)
            variables[var_name] = (expression, value)
            context[var_name] = value
        except Exception:
            variables[var_name] = (expression, None)
            context[var_name] = None

    # Scale variables
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if line.startswith(' ') or line.startswith('\t'):
            new_lines.append(line)
            continue
        match = pattern.match(stripped)
        if not match or match.group(1) not in variables:
            new_lines.append(line)
            continue
        var_name = match.group(1)
        original_expr, original_value = variables[var_name]
        if original_value is not None and isinstance(original_value, (int, float)):
            new_value = int(original_value * multiplier)
            indent = len(line) - len(line.lstrip())
            new_lines.append(' ' * indent + f"{var_name} = {new_value}")
        else:
            new_lines.append(line)
    
    return '\n'.join(new_lines)


SCALER_AVAILABLE = True

# fvcore for static FLOPs analysis
try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
    from fvcore.nn.jit_handles import Handle, get_shape
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("Warning: fvcore not installed. Install with: pip install fvcore")


# =============================================================================
# Custom FLOPs handlers for operations not covered by fvcore
# =============================================================================

def _elementwise_flop_counter(flops_per_element: int = 1):
    """
    Factory for elementwise op counters.
    Most elementwise ops are O(n) where n is number of elements.
    """
    def counter(inputs, outputs):
        # Get output shape to determine number of elements
        out_shape = get_shape(outputs[0])
        return int(flops_per_element * prod(out_shape))
    return counter


def prod(shape):
    """Product of shape dimensions."""
    result = 1
    for s in shape:
        result *= s
    return result


def _matmul_flop_counter(inputs, outputs):
    """FLOPs for matrix multiplication: 2 * M * N * K"""
    # inputs[0] shape: [..., M, K], inputs[1] shape: [..., K, N]
    a_shape = get_shape(inputs[0])
    b_shape = get_shape(inputs[1])
    
    # Handle batched matmul
    if len(a_shape) >= 2 and len(b_shape) >= 2:
        m, k = a_shape[-2], a_shape[-1]
        n = b_shape[-1]
        batch = prod(a_shape[:-2]) if len(a_shape) > 2 else 1
        return int(2 * batch * m * n * k)
    return 0


def _conv_flop_counter(inputs, outputs):
    """FLOPs for convolutional layers (1D/2D/3D).

    Uses output shape and weight kernel to compute multiply-adds:
      2 * N * Cout * (spatial_out) * (Cin/groups) * (kernel_volume)
    """
    try:
        out_shape = get_shape(outputs[0])
        in_shape = get_shape(inputs[0])
        weight_shape = get_shape(inputs[1])

        # Determine input channels and kernel size from weight
        cin = in_shape[1] if len(in_shape) > 1 else 1
        # weight shape examples: [Cout, Cin/groups, k], [Cout, Cin/groups, kH, kW], [Cout, Cin/groups, kD, kH, kW]
        if len(out_shape) == 4 and len(weight_shape) >= 4:
            # 2D conv: [N, Cout, Hout, Wout]
            n, cout, h_out, w_out = out_shape
            kH, kW = weight_shape[-2], weight_shape[-1]
            cin_per_group = weight_shape[1] if len(weight_shape) > 1 else cin
            groups = max(1, cin // cin_per_group)
            return int(2 * n * cout * h_out * w_out * (cin // groups) * kH * kW)
        elif len(out_shape) == 3 and len(weight_shape) >= 3:
            # 1D conv: [N, Cout, Lout]
            n, cout, l_out = out_shape
            k = weight_shape[-1]
            cin_per_group = weight_shape[1] if len(weight_shape) > 1 else cin
            groups = max(1, cin // cin_per_group)
            return int(2 * n * cout * l_out * (cin // groups) * k)
        elif len(out_shape) == 5 and len(weight_shape) >= 5:
            # 3D conv: [N, Cout, Dout, Hout, Wout]
            n, cout, d_out, h_out, w_out = out_shape
            kD, kH, kW = weight_shape[-3], weight_shape[-2], weight_shape[-1]
            cin_per_group = weight_shape[1] if len(weight_shape) > 1 else cin
            groups = max(1, cin // cin_per_group)
            return int(2 * n * cout * d_out * h_out * w_out * (cin // groups) * kD * kH * kW)
    except Exception:
        return 0



def _softmax_flop_counter(inputs, outputs):
    """FLOPs for softmax: ~5n (exp, sum, div per element)"""
    out_shape = get_shape(outputs[0])
    n = prod(out_shape)
    return int(5 * n)


def _layer_norm_flop_counter(inputs, outputs):
    """FLOPs for layer norm: ~8n (mean, var, normalize, scale, shift)"""
    out_shape = get_shape(outputs[0])
    n = prod(out_shape)
    return int(8 * n)


def _batch_norm_flop_counter(inputs, outputs):
    """FLOPs for batch norm: ~4n"""
    out_shape = get_shape(outputs[0])
    n = prod(out_shape)
    return int(4 * n)


def _scaled_dot_product_attention_flops(inputs, outputs):
    """
    FLOPs for scaled dot product attention.
    query: [B, H, L, D], key: [B, H, S, D], value: [B, H, S, D]
    FLOPs = 2 * B * H * L * S * D (Q @ K^T) + B * H * L * S (scaling + softmax) + 2 * B * H * L * S * D (attn @ V)
    """
    query_shape = get_shape(inputs[0])  # [B, H, L, D] or [B, L, D]
    key_shape = get_shape(inputs[1])    # [B, H, S, D] or [B, S, D]
    
    if len(query_shape) == 4:
        b, h, l, d = query_shape
        s = key_shape[2]
    elif len(query_shape) == 3:
        b, l, d = query_shape
        h = 1
        s = key_shape[1]
    else:
        # Fallback
        return prod(get_shape(outputs[0])) * 4
    
    # Q @ K^T: 2 * B * H * L * S * D
    qk_flops = 2 * b * h * l * s * d
    # Softmax: ~5 * B * H * L * S  
    softmax_flops = 5 * b * h * l * s
    # Attn @ V: 2 * B * H * L * S * D (assuming D for value)
    av_flops = 2 * b * h * l * d * s
    
    return int(qk_flops + softmax_flops + av_flops)


# Register custom op handlers with fvcore
CUSTOM_OPS_HANDLERS = {
    # Activation functions (elementwise)
    "aten::gelu": _elementwise_flop_counter(14),  # GELU: erf + mul + add
    "aten::relu": _elementwise_flop_counter(1),   # ReLU: max(0, x)
    "aten::relu_": _elementwise_flop_counter(1),
    "aten::leaky_relu": _elementwise_flop_counter(2),
    "aten::leaky_relu_": _elementwise_flop_counter(2),
    "aten::elu": _elementwise_flop_counter(8),    # ELU: exp + compare + mul
    "aten::elu_": _elementwise_flop_counter(8),
    "aten::selu": _elementwise_flop_counter(10),  # SELU: elu + scale
    "aten::selu_": _elementwise_flop_counter(10),
    "aten::sigmoid": _elementwise_flop_counter(4),  # sigmoid: exp + div
    "aten::sigmoid_": _elementwise_flop_counter(4),
    "aten::tanh": _elementwise_flop_counter(6),   # tanh
    "aten::tanh_": _elementwise_flop_counter(6),
    "aten::hardtanh": _elementwise_flop_counter(2),  # clamp
    "aten::hardtanh_": _elementwise_flop_counter(2),
    "aten::hardsigmoid": _elementwise_flop_counter(3),
    "aten::hardsigmoid_": _elementwise_flop_counter(3),
    "aten::hardswish": _elementwise_flop_counter(5),
    "aten::hardswish_": _elementwise_flop_counter(5),
    "aten::softplus": _elementwise_flop_counter(6),  # log(1 + exp(x))
    "aten::softsign": _elementwise_flop_counter(3),  # x / (1 + |x|)
    "aten::mish": _elementwise_flop_counter(12),  # x * tanh(softplus(x))
    "aten::silu": _elementwise_flop_counter(5),   # x * sigmoid(x) (swish)
    "aten::silu_": _elementwise_flop_counter(5),
    
    # Basic math ops
    "aten::exp": _elementwise_flop_counter(4),
    "aten::log": _elementwise_flop_counter(4),
    "aten::sqrt": _elementwise_flop_counter(2),
    "aten::rsqrt": _elementwise_flop_counter(3),
    "aten::pow": _elementwise_flop_counter(8),
    "aten::abs": _elementwise_flop_counter(1),
    "aten::neg": _elementwise_flop_counter(1),
    "aten::clamp": _elementwise_flop_counter(2),
    "aten::clamp_": _elementwise_flop_counter(2),
    
    # Reduction ops
    "aten::sum": _elementwise_flop_counter(1),
    "aten::mean": _elementwise_flop_counter(2),
    "aten::max": _elementwise_flop_counter(1),
    "aten::min": _elementwise_flop_counter(1),
    "aten::argmax": _elementwise_flop_counter(1),
    "aten::argmin": _elementwise_flop_counter(1),
    
    # Normalization
    "aten::softmax": _softmax_flop_counter,
    "aten::log_softmax": _softmax_flop_counter,
    "aten::layer_norm": _layer_norm_flop_counter,
    "aten::group_norm": _layer_norm_flop_counter,
    "aten::batch_norm": _batch_norm_flop_counter,
    "aten::instance_norm": _batch_norm_flop_counter,
    
    # Matrix ops
    "aten::mm": _matmul_flop_counter,
    "aten::bmm": _matmul_flop_counter,
    "aten::matmul": _matmul_flop_counter,
    
    # Elementwise binary ops
    "aten::add": _elementwise_flop_counter(1),
    "aten::add_": _elementwise_flop_counter(1),
    "aten::sub": _elementwise_flop_counter(1),
    "aten::sub_": _elementwise_flop_counter(1),
    "aten::mul": _elementwise_flop_counter(1),
    "aten::mul_": _elementwise_flop_counter(1),
    "aten::div": _elementwise_flop_counter(1),
    "aten::div_": _elementwise_flop_counter(1),
    
    # Cumulative ops
    "aten::cumsum": _elementwise_flop_counter(1),
    "aten::cumprod": _elementwise_flop_counter(1),
    
    # Loss functions
    "aten::mse_loss": _elementwise_flop_counter(3),  # (a-b)^2 + mean
    "aten::l1_loss": _elementwise_flop_counter(2),
    "aten::smooth_l1_loss": _elementwise_flop_counter(4),
    "aten::cross_entropy_loss": _elementwise_flop_counter(8),
    "aten::nll_loss": _elementwise_flop_counter(2),
    "aten::nll_loss_nd": _elementwise_flop_counter(2),
    "aten::kl_div": _elementwise_flop_counter(6),
    "aten::binary_cross_entropy": _elementwise_flop_counter(8),
    "aten::hinge_embedding_loss": _elementwise_flop_counter(3),
    "aten::huber_loss": _elementwise_flop_counter(5),
    "aten::triplet_margin_loss": _elementwise_flop_counter(6),
    
    # Pooling
    "aten::max_pool1d": _elementwise_flop_counter(1),
    "aten::max_pool2d": _elementwise_flop_counter(1),
    "aten::max_pool3d": _elementwise_flop_counter(1),
    "aten::avg_pool1d": _elementwise_flop_counter(2),
    "aten::avg_pool2d": _elementwise_flop_counter(2),
    "aten::avg_pool3d": _elementwise_flop_counter(2),
    "aten::adaptive_max_pool1d": _elementwise_flop_counter(1),
    "aten::adaptive_max_pool2d": _elementwise_flop_counter(1),
    "aten::adaptive_max_pool3d": _elementwise_flop_counter(1),
    "aten::adaptive_avg_pool1d": _elementwise_flop_counter(2),
    "aten::adaptive_avg_pool2d": _elementwise_flop_counter(2),
    "aten::adaptive_avg_pool3d": _elementwise_flop_counter(2),
    
    # Attention
    "aten::scaled_dot_product_attention": _scaled_dot_product_attention_flops,
    
    # More ops
    "aten::rsub": _elementwise_flop_counter(1),
    "aten::triu": _elementwise_flop_counter(1),
    "aten::tril": _elementwise_flop_counter(1),
    "aten::flip": _elementwise_flop_counter(0),  # just reordering
    "aten::numpy_T": _elementwise_flop_counter(0),  # transpose
    "aten::transpose": _elementwise_flop_counter(0),
    "aten::t": _elementwise_flop_counter(0),
    "aten::permute": _elementwise_flop_counter(0),
    "aten::linalg_vector_norm": _elementwise_flop_counter(3),  # sqrt(sum(x^2))
    "aten::frobenius_norm": _elementwise_flop_counter(3),
    "aten::norm": _elementwise_flop_counter(3),
    "aten::broadcast_tensors": _elementwise_flop_counter(0),
    "aten::where": _elementwise_flop_counter(1),
    "aten::masked_fill": _elementwise_flop_counter(1),
    "aten::masked_fill_": _elementwise_flop_counter(1),

    # Convolution ops (1D/2D/3D)
    "aten::conv1d": _conv_flop_counter,
    "aten::conv2d": _conv_flop_counter,
    "aten::conv3d": _conv_flop_counter,

    # Reshape / view / concat / copy ops — negligible arithmetic
    "aten::flatten": _elementwise_flop_counter(0),
    "aten::view": _elementwise_flop_counter(0),
    "aten::reshape": _elementwise_flop_counter(0),
    "aten::squeeze": _elementwise_flop_counter(0),
    "aten::unsqueeze": _elementwise_flop_counter(0),
    "aten::cat": _elementwise_flop_counter(0),
    "aten::stack": _elementwise_flop_counter(0),
    "aten::clone": _elementwise_flop_counter(0),
    "aten::contiguous": _elementwise_flop_counter(0),

    # Dropout / stochastic masks (approx 1 op per element)
    "aten::dropout": _elementwise_flop_counter(1),
    "aten::dropout_": _elementwise_flop_counter(1),
}


@dataclass
class ModelFeatures:
    """Features extracted from a model."""
    name: str
    total_bytes_read_mb: float
    total_bytes_written_mb: float
    total_bytes_mb: float
    total_flops_m: float
    arithmetic_intensity: float
    num_nodes: int
    count_unique_ops: int


def load_model_from_source(source_code: str) -> tuple:
    """
    Load Model class and input functions from source code.
    
    Returns:
        tuple: (Model class, get_init_inputs fn, get_inputs fn)
    """
    context = {
        "torch": torch, 
        "nn": nn, 
        "torch.nn": nn,
        "F": torch.nn.functional,
        "torch.nn.functional": torch.nn.functional,
    }
    exec(source_code, context)
    
    Model = context.get("Model")
    get_init_inputs = context.get("get_init_inputs")
    get_inputs = context.get("get_inputs")
    
    if Model is None:
        raise ValueError("Model class not found in source code")
    if get_inputs is None:
        raise ValueError("get_inputs function not found in source code")
    if get_init_inputs is None:
        raise ValueError("get_init_inputs function not found in source code")
    
    return Model, get_init_inputs, get_inputs


def estimate_tensor_bytes(tensor: torch.Tensor) -> int:
    """Estimate bytes for a tensor."""
    return tensor.numel() * tensor.element_size()


def get_input_shapes_from_source(source_code: str) -> List[Tuple[Any, ...]]:
    """
    Extract input tensor shapes by parsing the get_inputs function.
    Returns list of (shape, dtype) tuples.
    """
    context = {
        "torch": torch, 
        "nn": nn,
        "F": torch.nn.functional,
    }
    exec(source_code, context)
    
    # Call get_inputs to get actual shapes
    get_inputs = context.get("get_inputs")
    if get_inputs:
        try:
            inputs = get_inputs()
            shapes = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    shapes.append((tuple(inp.shape), inp.dtype))
                else:
                    shapes.append((None, type(inp)))
            return shapes
        except Exception:
            pass
    return []


def create_small_inputs(inputs: list, max_elements: int = 1_000_000) -> list:
    """
    Create smaller versions of inputs for faster analysis.
    Preserves shape constraints (e.g., for matmul A[M,K] @ B[K,N], K must match).
    """
    # Check total elements
    total_numel = sum(inp.numel() for inp in inputs if isinstance(inp, torch.Tensor))
    
    if total_numel <= max_elements:
        # No scaling needed, just clone
        return [inp.clone() if isinstance(inp, torch.Tensor) else inp for inp in inputs]
    
    # Calculate uniform scale factor
    scale = (max_elements / total_numel) ** 0.5  # sqrt for 2D scaling
    scale = max(0.01, min(1.0, scale))  # clamp to reasonable range
    
    small_inputs = []
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            if inp.numel() <= 1000:  
                # Keep very small tensors as-is
                small_inputs.append(inp.clone())
            else:
                # Scale each dimension uniformly
                new_shape = tuple(max(1, int(s * scale)) for s in inp.shape)
                # Ensure at least some minimum size
                new_shape = tuple(max(s, 2) for s in new_shape)
                small_inp = torch.randn(new_shape, dtype=inp.dtype)
                small_inputs.append(small_inp)
        else:
            small_inputs.append(inp)
    
    return small_inputs


def calculate_static_memory(
    model: nn.Module, 
    inputs: list,
    include_backward: bool = True
) -> Tuple[int, int]:
    """
    Calculate memory read/write statically from tensor shapes.
    Does NOT allocate actual tensors - uses shape analysis only.
    
    Forward pass:
      - Read: input tensors + parameters
      - Write: output tensors (assumed same size as inputs for elementwise ops)
    
    Backward pass (approximate):
      - Read: activations + grad_output + parameters
      - Write: grad_input + grad_parameters
    
    Returns:
        tuple: (bytes_read, bytes_written)
    """
    bytes_read = 0
    bytes_written = 0
    
    # Input tensors (read during forward)
    input_bytes = 0
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            inp_bytes = inp.numel() * inp.element_size()
            bytes_read += inp_bytes
            input_bytes += inp_bytes
    
    # Parameters (read during forward)
    param_bytes = 0
    for param in model.parameters():
        param_bytes += param.numel() * param.element_size()
    bytes_read += param_bytes
    
    # Estimate output size - for most models, output ~ largest input
    # Try to infer from model structure
    output_bytes = _estimate_output_bytes(model, inputs)
    bytes_written += output_bytes
    
    if include_backward:
        # Backward pass memory:
        # - Read grad_output (same as output)
        # - Read activations (roughly same as forward intermediate ≈ input)
        # - Read parameters again
        # - Write grad_input
        # - Write grad_params
        
        bytes_read += output_bytes  # grad_output
        bytes_read += input_bytes   # stored activations
        bytes_read += param_bytes   # params for backward
        
        bytes_written += input_bytes  # grad_input
        bytes_written += param_bytes  # grad_params
    
    return bytes_read, bytes_written


def _estimate_output_bytes(model: nn.Module, inputs: list) -> int:
    """
    Estimate output tensor size without running the model.
    Uses heuristics based on model structure.
    """
    # For most elementwise ops, output size = input size
    input_bytes = sum(
        inp.numel() * inp.element_size()
        for inp in inputs if isinstance(inp, torch.Tensor)
    )
    
    # Check for common layer types that change output size
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Linear changes last dimension
            # Rough estimate: scale by out_features / in_features ratio
            if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                ratio = module.out_features / module.in_features
                return int(input_bytes * ratio)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            # Convs change channels and possibly spatial dims
            # Rough estimate based on output channels
            if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                ratio = module.out_channels / module.in_channels
                return int(input_bytes * ratio)
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            # Pooling reduces spatial dims by kernel_size
            kernel = module.kernel_size
            if isinstance(kernel, int):
                return input_bytes // (kernel * kernel)
            else:
                return input_bytes // (kernel[0] * kernel[1])
    
    # Default: output same size as input (elementwise ops)
    return input_bytes


def count_flops_with_fvcore(model: nn.Module, inputs: list, verbose: bool = False) -> int:
    """
    Count FLOPs using fvcore's static analysis with custom handlers
    for functional ops like F.gelu, F.relu, etc.
    
    For very large inputs, uses scaled-down versions and scales FLOPs back up.
    """
    if not FVCORE_AVAILABLE:
        return 0
    
    # Calculate total input size
    total_numel = sum(inp.numel() for inp in inputs if isinstance(inp, torch.Tensor))
    max_numel = 10_000_000  # ~40MB for float32, safe to allocate
    
    try:
        if total_numel <= max_numel:
            # Small enough - use original shapes
            trace_inputs = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    trace_inputs.append(torch.zeros(inp.shape, dtype=inp.dtype))
                else:
                    trace_inputs.append(inp)
            scale_factor = 1.0
        else:
            # Too large - scale down uniformly and track the scale
            scale = (max_numel / total_numel) ** 0.5
            scale = max(0.01, scale)
            
            trace_inputs = []
            scaled_numel = 0
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    new_shape = tuple(max(2, int(s * scale)) for s in inp.shape)
                    trace_inputs.append(torch.zeros(new_shape, dtype=inp.dtype))
                    scaled_numel += prod(new_shape)
                else:
                    trace_inputs.append(inp)
            
            # Scale factor to multiply FLOPs
            scale_factor = total_numel / scaled_numel if scaled_numel > 0 else 1.0
        
        # fvcore expects inputs as a tuple or single tensor
        if len(trace_inputs) == 1:
            flops_analyzer = FlopCountAnalysis(model, trace_inputs[0])
        else:
            flops_analyzer = FlopCountAnalysis(model, tuple(trace_inputs))
        
        # Register custom handlers for ops not covered by fvcore
        for op_name, handler in CUSTOM_OPS_HANDLERS.items():
            flops_analyzer.set_op_handle(op_name, handler)
        
        # Get total FLOPs and scale
        total_flops = int(flops_analyzer.total() * scale_factor)

        return total_flops

    except Exception as e:
        if verbose:
            warnings.warn(f"fvcore FLOPs analysis failed: {e}")
        # Attempt module-level conv FLOP estimation using forward hooks
        try:
            # Capture module output shapes with hooks
            module_output_shapes = {}
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    def make_hook(n):
                        def hook(mod, inp, out):
                            try:
                                # prefer fvcore.get_shape if available on tensors
                                if hasattr(out, 'shape'):
                                    module_output_shapes[n] = tuple(out.shape)
                                else:
                                    module_output_shapes[n] = get_shape(out)
                            except Exception:
                                try:
                                    module_output_shapes[n] = tuple(out.shape)
                                except Exception:
                                    pass
                        return hook
                    h = module.register_forward_hook(make_hook(name))
                    hooks.append(h)

            # Run a forward pass with trace inputs to populate shapes
            model.eval()
            with torch.no_grad():
                if len(trace_inputs) == 1:
                    _ = model(trace_inputs[0])
                else:
                    _ = model(*tuple(trace_inputs))

            # Remove hooks
            for h in hooks:
                h.remove()

            # Compute conv FLOPs from module params and captured output shapes
            conv_flops = 0
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    w = getattr(module, 'weight', None)
                    out_shape = module_output_shapes.get(name)
                    if w is None or out_shape is None:
                        continue
                    w_shape = tuple(w.shape)
                    groups = getattr(module, 'groups', 1)
                    # derive flops similar to _conv_flop_counter
                    try:
                        if len(out_shape) == 4 and len(w_shape) >= 4:
                            n, cout, h_out, w_out = out_shape
                            kH, kW = w_shape[-2], w_shape[-1]
                            cin = w_shape[1] * groups if len(w_shape) > 1 else None
                            cin_per_group = w_shape[1] if len(w_shape) > 1 else 1
                            conv_flops += int(2 * n * cout * h_out * w_out * (cin // groups) * kH * kW)
                        elif len(out_shape) == 3 and len(w_shape) >= 3:
                            n, cout, l_out = out_shape
                            k = w_shape[-1]
                            cin = w_shape[1] * groups if len(w_shape) > 1 else None
                            conv_flops += int(2 * n * cout * l_out * (cin // groups) * k)
                        elif len(out_shape) == 5 and len(w_shape) >= 5:
                            n, cout, d_out, h_out, w_out = out_shape
                            kD, kH, kW = w_shape[-3], w_shape[-2], w_shape[-1]
                            cin = w_shape[1] * groups if len(w_shape) > 1 else None
                            conv_flops += int(2 * n * cout * d_out * h_out * w_out * (cin // groups) * kD * kH * kW)
                    except Exception:
                        continue

            # Add conv flops to fallback estimate
            fallback = estimate_flops_fallback(model, inputs)
            total_estimate = fallback + conv_flops
            if verbose:
                warnings.warn(f"Using fallback estimate + conv_flops: {fallback} + {conv_flops} = {total_estimate}")
            return total_estimate
        except Exception:
            return estimate_flops_fallback(model, inputs)


def estimate_flops_fallback(model: nn.Module, inputs: list) -> int:
    """
    Fallback FLOPs estimation based on operation types.
    """
    total_flops = 0
    
    # Estimate from input size (element-wise ops = 1-10 FLOPs per element)
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            # Assume average 2 FLOPs per element for simple ops
            total_flops += inp.numel() * 2
    
    # Add FLOPs for parameterized layers
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # FLOPs = 2 * in_features * out_features (multiply-add)
            total_flops += 2 * module.in_features * module.out_features
        elif isinstance(module, nn.Conv2d):
            # Rough estimate for convolutions
            total_flops += 2 * module.in_channels * module.out_channels * \
                          module.kernel_size[0] * module.kernel_size[1]
    
    return total_flops


def count_graph_nodes_and_ops(model: nn.Module, inputs: list, verbose: bool = False) -> Tuple[int, int]:
    """
    Count nodes and unique operations using JIT tracing.
    Node count and unique ops don't change with input size, so use small inputs.
    """
    num_nodes = 0
    op_counts = Counter()
    
    try:
        # Use small inputs for tracing - graph structure doesn't depend on size
        max_numel = 10_000_000
        total_numel = sum(inp.numel() for inp in inputs if isinstance(inp, torch.Tensor))
        
        if total_numel <= max_numel:
            trace_inputs = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    trace_inputs.append(torch.zeros(inp.shape, dtype=inp.dtype))
                else:
                    trace_inputs.append(inp)
        else:
            # Scale down for tracing
            scale = (max_numel / total_numel) ** 0.5
            scale = max(0.01, scale)
            trace_inputs = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    new_shape = tuple(max(2, int(s * scale)) for s in inp.shape)
                    trace_inputs.append(torch.zeros(new_shape, dtype=inp.dtype))
                else:
                    trace_inputs.append(inp)
        
        # JIT trace the model
        if len(trace_inputs) == 1:
            traced = torch.jit.trace(model, trace_inputs[0], check_trace=False)
        else:
            traced = torch.jit.trace(model, tuple(trace_inputs), check_trace=False)
        
        # Get the graph
        graph = traced.graph
        
        for node in graph.nodes():
            num_nodes += 1
            kind = node.kind()
            op_name = kind.split("::")[-1] if "::" in kind else kind
            op_counts[op_name] += 1
                
    except Exception as e:
        if verbose:
            warnings.warn(f"JIT tracing failed: {e}, using module-based counting")
        num_nodes, op_counts = count_from_modules(model)
    
    return num_nodes, len(op_counts)


def count_from_modules(model: nn.Module) -> Tuple[int, Counter]:
    """Fallback: count nodes from module structure."""
    num_nodes = 0
    op_counts = Counter()
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf module
            num_nodes += 1
            op_counts[type(module).__name__] += 1
    
    # Add input/output nodes
    num_nodes += 2
    op_counts['input'] += 1
    op_counts['output'] += 1
    
    return num_nodes, op_counts


def extract_features(
    model_path: str,
    name: Optional[str] = None,
    verbose: bool = False,
    include_backward: bool = True,
    multiplier: Optional[float] = None
) -> Optional[ModelFeatures]:
    """
    Extract features from a single model file using static analysis.
    
    Args:
        model_path: Path to the model Python file
        name: Optional name override
        verbose: Print progress information
        include_backward: Include backward pass in calculations
        multiplier: Optional multiplier for input variables (uses variable_scaler)
    """
    # Build name based on file and multiplier
    base_name = Path(model_path).name  # e.g., "100_HingeLoss.py"
    if multiplier is not None and multiplier != 1.0:
        name = f"{base_name}_variable_multiplier_{multiplier}"
    elif name is None:
        name = Path(model_path).stem
    
    if verbose:
        print(f"Processing: {name}")
    
    try:
        # Load model source code
        source_code = read_file(model_path)
        
        # Apply variable scaling if multiplier specified
        if multiplier is not None and multiplier != 1.0:
            if SCALER_AVAILABLE:
                source_code = scale_source_code(source_code, multiplier=multiplier)
                if verbose:
                    print(f"  Applied variable multiplier: {multiplier}x")
            else:
                if verbose:
                    print("  Warning: variable_scaler not available, skipping multiplier")
        
        # Load model and input functions
        Model, get_init_inputs, get_inputs = load_model_from_source(source_code)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Initialize model
        init_inputs = get_init_inputs()
        model = Model(*init_inputs)
        model.eval()  # Eval mode for tracing
        
        # Get inputs (these define the shapes we analyze)
        torch.manual_seed(42)
        inputs = get_inputs()
        
        # Static FLOPs analysis with fvcore
        total_flops = count_flops_with_fvcore(model, inputs, verbose=verbose)
        
        # For forward+backward, multiply FLOPs by ~3 (backward is ~2x forward)
        if include_backward:
            total_flops = int(total_flops * 3)
        
        # Static memory analysis
        bytes_read, bytes_written = calculate_static_memory(
            model, inputs, include_backward=include_backward
        )
        
        # Count graph nodes and unique ops
        num_nodes, count_unique_ops = count_graph_nodes_and_ops(model, inputs, verbose=verbose)
        
        # Calculate derived metrics
        total_bytes = bytes_read + bytes_written
        total_bytes_read_mb = bytes_read / (1024 * 1024)
        total_bytes_written_mb = bytes_written / (1024 * 1024)
        total_bytes_mb = total_bytes / (1024 * 1024)
        total_flops_m = total_flops / 1e6
        
        # Arithmetic intensity: FLOPs per byte transferred
        arithmetic_intensity = total_flops / total_bytes if total_bytes > 0 else 0.0
        
        return ModelFeatures(
            name=name,
            total_bytes_read_mb=round(total_bytes_read_mb, 4),
            total_bytes_written_mb=round(total_bytes_written_mb, 4),
            total_bytes_mb=round(total_bytes_mb, 4),
            total_flops_m=round(total_flops_m, 4),
            arithmetic_intensity=round(arithmetic_intensity, 4),
            num_nodes=num_nodes,
            count_unique_ops=count_unique_ops,
        )
        
    except Exception as e:
        if verbose:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
        return None


def get_model_files(level: Optional[int] = None, all_levels: bool = False) -> list:
    """Get list of model files to process."""
    kernel_bench_path = os.path.join(REPO_TOP_PATH, "KernelBench")
    model_files = []
    
    if all_levels:
        levels = [1, 2, 3, 4]
    elif level is not None:
        levels = [level]
    else:
        levels = []
    
    for lvl in levels:
        level_dir = os.path.join(kernel_bench_path, f"level{lvl}")
        if os.path.exists(level_dir):
            for f in sorted(os.listdir(level_dir)):
                if f.endswith('.py'):
                    model_files.append(os.path.join(level_dir, f))
    
    return model_files


def save_features_csv(features_list: list, output_path: str):
    """Save features to CSV file."""
    if not features_list:
        print("No features to save")
        return
    
    fieldnames = list(asdict(features_list[0]).keys())
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for feat in features_list:
            writer.writerow(asdict(feat))
    
    print(f"Saved {len(features_list)} model features to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from KernelBench models (static analysis with fvcore)"
    )
    parser.add_argument(
        "--level", type=int, choices=[1, 2, 3, 4],
        help="KernelBench level to process (1-4)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all levels"
    )
    parser.add_argument(
        "--file", type=str,
        help="Process a single model file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="model_features.csv",
        help="Output CSV file path (default: model_features.csv)"
    )
    parser.add_argument(
        "--forward-only", action="store_true",
        help="Only analyze forward pass (no backward)"
    )
    parser.add_argument(
        "--multiplier", "-m", type=float, default=None,
        help="Variable multiplier for input sizes (e.g., 1.3 for 30%% larger inputs)"
    )
    parser.add_argument(
        "--multipliers", type=str, default=None,
        help="Comma-separated list of multipliers to run (e.g., '0.5,1.0,1.5,2.0')"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    if not FVCORE_AVAILABLE:
        print("ERROR: fvcore is required. Install with: pip install fvcore")
        sys.exit(1)
    
    # Suppress warnings for cleaner output
    if not args.verbose:
        warnings.filterwarnings('ignore')
    
    # Determine which files to process
    if args.file:
        model_files = [args.file]
    elif args.all:
        model_files = get_model_files(all_levels=True)
    elif args.level:
        model_files = get_model_files(level=args.level)
    else:
        print("Please specify --level, --all, or --file")
        parser.print_help()
        sys.exit(1)
    
    # Parse multipliers
    multipliers = [None]  # Default: no scaling
    if args.multipliers:
        multipliers = [float(m.strip()) for m in args.multipliers.split(',')]
    elif args.multiplier is not None:
        multipliers = [args.multiplier]
    
    total_runs = len(model_files) * len(multipliers)
    print(f"Processing {len(model_files)} model(s) x {len(multipliers)} multiplier(s) = {total_runs} runs...")
    
    # Extract features
    features_list = []
    failed = []
    
    include_backward = not args.forward_only
    run_idx = 0
    
    for multiplier in multipliers:
        mult_str = f" (multiplier={multiplier})" if multiplier else ""
        if args.verbose and len(multipliers) > 1:
            print(f"\n--- Multiplier: {multiplier or 'None (original)'} ---")
        
        for i, model_path in enumerate(model_files):
            run_idx += 1
            if args.verbose:
                print(f"[{run_idx}/{total_runs}] ", end="")
            
            features = extract_features(
                model_path, 
                verbose=args.verbose,
                include_backward=include_backward,
                multiplier=multiplier
            )
            
            if features is not None:
                features_list.append(features)
                if not args.verbose:
                    print(f".", end="", flush=True)
            else:
                failed.append((model_path, multiplier))
                if not args.verbose:
                    print(f"x", end="", flush=True)
    
    print()  # newline after progress dots
    
    # Save results
    if features_list:
        save_features_csv(features_list, args.output)
    
    # Report failures
    if failed:
        print(f"\nFailed to extract features from {len(failed)} run(s):")
        for item in failed[:10]:  # Show first 10
            if isinstance(item, tuple):
                f, m = item
                print(f"  - {Path(f).name} (multiplier={m})")
            else:
                print(f"  - {Path(item).name}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    print(f"\nSummary: {len(features_list)} succeeded, {len(failed)} failed")


if __name__ == "__main__":
    main()
