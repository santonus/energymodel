#!/usr/bin/env python3
"""
Script to scale variables in KernelBench programs and run them with modified values.

Scales only module-level variables that are used in get_inputs or get_init_inputs
(AST-based discovery, no hardcoded names). Writes scaled source to disk when
problem_path is set; restores original on __del__.
"""

import os
import sys
import re
import ast
from typing import Dict, Tuple, Optional, List, Set, Any
from pathlib import Path

# Add project root to path
REPO_TOP_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, REPO_TOP_PATH)

import torch
from src.eval import load_original_model_and_inputs, set_seed
from src.dataset import construct_problem_dataset_from_problem_dir


def _vars_used_by_get_inputs_init_inputs(source: str) -> Set[str]:
    """
    Discover variable names to scale: module-level assignments that are
    referenced in get_inputs or get_init_inputs. No hardcoded names.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    module_level_vars: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    module_level_vars.add(t.id)

    used_in_fns: Set[str] = set()
    locals_in_fns: Set[str] = set()
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


class VariableScaler:
    """
    Class to scale variables in KernelBench programs.
    
    Supports two modes:
    1. File-based: Load from KernelBench directory. Scaling writes the modified
       source to disk; when the scaler is deleted (__del__), the original
       source is restored to disk.
    2. Source-based: Work with provided source code string (no disk I/O).
    
    Usage (file-based):
        scaler = VariableScaler(level=1, problem_id=2, multiplier=2.0)
        scaler.scale_and_run()  # Writes scaled to disk, runs, restore on del
    
    Usage (source-based, for integration with generate_baseline_time.py):
        scaler = VariableScaler(source_code=code_string, multiplier=2.0)
        modified_source = scaler.scale_source()  # Get scaled source string
        # Then use modified_source with load_original_model_and_inputs()
    
    Or use the convenience function:
        from scripts.variable_scaler import scale_source_code
        modified_source = scale_source_code(code_string, multiplier=2.0)
    """
    
    def __init__(self, level: Optional[int] = None, problem_id: Optional[int] = None, 
                 multiplier: float = 1.0, source_code: Optional[str] = None,
                 problem_path: Optional[str] = None):
        """
        Initialize the VariableScaler.

        Args:
            level: KernelBench level (1, 2, or 3) - required if source_code not provided
            problem_id: Problem ID (1-indexed) - required if source_code not provided
            multiplier: Multiplier to apply to variable values (default: 1.0)
            source_code: Source code string to modify - if provided, level/problem_id not needed
            problem_path: Path to problem file. If provided with source_code, scaling writes
                to disk and __del__ restores original. Used by generate_baseline_time.
        """
        if source_code is None and (level is None or problem_id is None):
            raise ValueError("Either provide source_code or both level and problem_id")

        self.level = level
        self.problem_id = problem_id
        self.multiplier = multiplier

        # State
        self.original_source = None
        self.modified_source = None
        self.variable_values = {}  # Original values
        self.variable_expressions = {}  # Original expressions as strings
        self.problem_path = problem_path
        self._wrote_to_disk = False

        if source_code is not None:
            # Source-based mode (optionally with problem_path for disk write/restore)
            self.original_source = source_code
            self.modified_source = source_code
        else:
            # File-based mode
            self.kernel_bench_path = os.path.join(REPO_TOP_PATH, "KernelBench")
            self.problem_dir = os.path.join(self.kernel_bench_path, f"level{level}")
            self._load_problem()
    
    def _load_problem(self):
        """Load the problem file path and source code."""
        dataset = construct_problem_dataset_from_problem_dir(self.problem_dir)
        
        if self.problem_id < 1 or self.problem_id > len(dataset):
            raise ValueError(
                f"Problem ID {self.problem_id} out of range. "
                f"Level {self.level} has {len(dataset)} problems."
            )
        
        self.problem_path = dataset[self.problem_id - 1]
        
        with open(self.problem_path, 'r') as f:
            self.original_source = f.read()
        
        self.modified_source = self.original_source
    
    def _find_scalable_variables(self) -> Dict[str, Tuple[str, Any]]:
        """
        Find variable assignments to scale: only those used in get_inputs
        or get_init_inputs (AST-based, no hardcoding).
        Returns: {var_name: (original_expression, evaluated_value)}
        """
        scalable = _vars_used_by_get_inputs_init_inputs(self.original_source)
        if not scalable:
            return {}
        variables = {}
        pattern = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$')
        lines = self.original_source.split('\n')
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

        return variables
    
    def scale_variables(self) -> Dict[str, any]:
        """
        Scale variables used in get_inputs/get_init_inputs by the multiplier.
        Returns: {var_name: new_value} for scaled vars.
        """
        self.modified_source = self.original_source
        variables = self._find_scalable_variables()
        self.variable_values = {}
        self.variable_expressions = {}
        scaled_values = {}
        pattern = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$')
        lines = self.original_source.split('\n')
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
            self.variable_expressions[var_name] = original_expr
            self.variable_values[var_name] = original_value
            if original_value is not None and isinstance(original_value, (int, float)):
                new_value = int(original_value * self.multiplier)
                
                # CRITICAL: Prevent zero or negative dimensions (causes FPE crashes)
                if new_value < 1:
                    new_value = 1
                    if original_value > 0:
                        # Only warn if original was positive (not already problematic)
                        print(f"[WARNING] Variable '{var_name}' scaled to {int(original_value * self.multiplier)}, clamped to 1")
                
                indent = len(line) - len(line.lstrip())
                new_lines.append(' ' * indent + f"{var_name} = {new_value}")
                scaled_values[var_name] = new_value
            else:
                new_lines.append(line)
        
        self.modified_source = '\n'.join(new_lines)

        # Write scaled source to disk (file-based mode only); restore on __del__
        if self.problem_path is not None:
            try:
                with open(self.problem_path, 'w') as f:
                    f.write(self.modified_source)
                self._wrote_to_disk = True
            except Exception:
                pass

        return scaled_values
    
    def scale_source(self) -> str:
        """
        Scale variables and return the modified source code string.
        This is useful for integration with other scripts that work with source strings.
        
        Returns:
            Modified source code string with scaled variables
        """
        self.scale_variables()
        return self.modified_source
    
    def restore_variables(self):
        """Restore variables to their original values."""
        self.modified_source = self.original_source
    
    def get_modified_source(self) -> str:
        """Get the modified source code with scaled variables."""
        return self.modified_source
    
    def get_original_source(self) -> str:
        """Get the original source code."""
        return self.original_source
    
    def run_with_scaled_variables(
        self, 
        device: torch.device = None,
        num_trials: int = 1,
        verbose: bool = True
    ) -> Dict:
        """
        Run the program with scaled variables.
        
        Args:
            device: PyTorch device (default: cuda:0 if available, else cpu)
            num_trials: Number of trials to run
            verbose: Whether to print information
            
        Returns:
            Dictionary with execution results
        """
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Scale variables
        scaled_values = self.scale_variables()
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Running Level {self.level}, Problem {self.problem_id}")
            print(f"Problem: {os.path.basename(self.problem_path)}")
            print(f"Multiplier: {self.multiplier}x")
            print(f"\nScaled Variables:")
            for var_name, new_value in scaled_values.items():
                original_value = self.variable_values.get(var_name, "N/A")
                print(f"  {var_name}: {original_value} -> {new_value}")
            print(f"{'='*80}\n")
        
        # Execute the modified code
        context = {}
        try:
            exec(self.modified_source, context)
        except Exception as e:
            error_msg = f"Error executing modified code: {e}"
            if verbose:
                print(error_msg)
            return {"error": error_msg, "scaled_values": scaled_values}
        
        # Get the model and input functions
        Model = context.get("Model")
        get_init_inputs = context.get("get_init_inputs")
        get_inputs = context.get("get_inputs")
        
        if Model is None or get_init_inputs is None or get_inputs is None:
            error_msg = "Missing required functions (Model, get_init_inputs, get_inputs)"
            if verbose:
                print(error_msg)
            return {"error": error_msg, "scaled_values": scaled_values}
        
        # Run the model
        try:
            set_seed(42)
            init_inputs = get_init_inputs()
            inputs = get_inputs()
            
            # Move to device
            if torch.cuda.is_available() and device.type == 'cuda':
                inputs = [
                    x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                init_inputs = [
                    x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                    for x in init_inputs
                ]
            
            model = Model(*init_inputs)
            model = model.to(device)
            model.eval()
            
            # Run forward passes
            with torch.no_grad():
                for _ in range(num_trials):
                    output = model(*inputs)
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device=device)
            
            result = {
                "success": True,
                "scaled_values": scaled_values,
                "output_shape": list(output.shape) if isinstance(output, torch.Tensor) else "N/A",
                "device": str(device)
            }
            
            if verbose:
                print(f"Execution successful!")
                print(f"Output shape: {result['output_shape']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error running model: {e}"
            if verbose:
                print(error_msg)
            return {"error": error_msg, "scaled_values": scaled_values}

    def __del__(self):
        """Restore original source to disk when the scaler is destroyed."""
        if not getattr(self, "_wrote_to_disk", False):
            return
        path = getattr(self, "problem_path", None)
        orig = getattr(self, "original_source", None)
        if path is None or orig is None:
            return
        try:
            with open(path, "w") as f:
                f.write(orig)
        except Exception:
            pass
        self._wrote_to_disk = False

    def scale_and_run(
        self,
        device: torch.device = None,
        num_trials: int = 1,
        verbose: bool = True
    ) -> Dict:
        """
        Convenience method: scale variables, run, and restore.
        Same as run_with_scaled_variables (which already restores).
        """
        return self.run_with_scaled_variables(device, num_trials, verbose)


def scale_source_code(source_code: str, multiplier: float = 1.0) -> str:
    """
    Convenience function to scale variables in source code string.
    
    This is a simple wrapper for integration with other scripts like generate_baseline_time.py.
    It modifies the source code string in memory without touching files on disk.
    
    Args:
        source_code: Python source code string to modify
        multiplier: Multiplier to apply to capital letter variable values (default: 1.0)
    
    Returns:
        Modified source code string with scaled variables
    
    Example:
        modified_code = scale_source_code(code_string, multiplier=2.0)
        # Then use modified_code with load_original_model_and_inputs()
    """
    scaler = VariableScaler(source_code=source_code, multiplier=multiplier)
    return scaler.scale_source()
