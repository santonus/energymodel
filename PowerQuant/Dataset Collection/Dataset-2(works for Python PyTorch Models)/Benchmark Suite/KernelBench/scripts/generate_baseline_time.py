import os, sys
# Add project root to sys.path to prioritize local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from src.kernelbench.eval import (
    load_original_model_and_inputs,
    time_execution_with_cuda_event,
    get_timing_stats,
    set_seed,
    fetch_ref_arch_from_problem_id,
)
import gc
from src.kernelbench.dataset import construct_problem_dataset_from_problem_dir
from src.kernelbench.utils import read_file
from scripts.variable_scaler import scale_source_code, VariableScaler
import os
import json
from tqdm import tqdm
import threading
import subprocess
import time
import csv
import threading

"""
Generate baseline time for KernelBench
This profiles the wall clock time for each KernelBench reference problem

You can find a list of pre-generated baseline time in /results/timing/
But we recommend you run this script to generate the baseline time for your own hardware configurations

Using various configurations
- torch (Eager)

Torch Compile with various modes
https://pytorch.org/docs/main/generated/torch.compile.html
- torch.compile: backend="inductor", mode="default" (this is usually what happens when you do torch.compile(model))
- torch.compile: backend="inductor", mode="reduce-overhead" 
- torch.compile: backend="inductor", mode="max-autotune"
- torch.compile: backend="inductor", mode="max-autotune-no-cudagraphs"

In addition to default Torch Compile backend, you can always use other or your custom backends
https://pytorch.org/docs/stable/torch.compiler.html
- torch.compile: backend="cudagraphs" (CUDA graphs with AOT Autograd)
"""

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")

TIMING_DIR = os.path.join(REPO_TOP_PATH, "results", "timing")

# Global shared state for program metadata (thread-safe via lock)
_global_metadata_lock = threading.Lock()
_global_current_metadata = {}

# Global shared state for program metadata (thread-safe via lock)
_global_metadata_lock = threading.Lock()
_global_current_metadata = {}


def _nvidia_smi_monitor(device_id: int, interval: float, stop_event: threading.Event, data_list: list, 
                       csv_path: str, write_lock: threading.Lock, flush_interval: float = 5.0):
    """Monitor nvidia-smi in parallel thread, periodically flush to CSV with current program metadata"""
    last_flush_time = time.time()
    
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--id={device_id}", "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.current.graphics,clocks.current.memory", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=1.0
            )
            if result.returncode == 0:
                # Read current program metadata (thread-safe)
                with _global_metadata_lock:
                    current_metadata = _global_current_metadata.copy()
                # Store data with timestamp for later association with metadata
                data_list.append((result.stdout.strip(), current_metadata))
        except Exception:
            pass
        
        # Periodically flush to CSV
        current_time = time.time()
        if current_time - last_flush_time >= flush_interval and data_list:
            with write_lock:
                # Get current metadata for this flush batch
                with _global_metadata_lock:
                    flush_metadata = _global_current_metadata.copy()
                _flush_to_csv(csv_path, data_list, flush_metadata)
                data_list.clear()
            last_flush_time = current_time
        
        time.sleep(interval)


def _flush_to_csv(csv_path: str, data_list: list, default_metadata: dict):
    """Flush collected data to CSV file with metadata (uses metadata from time of collection)"""
    headers = ["ref_arch_name", "use_torch_compile", "torch_compile_backend", "torch_compile_options", "variable_multiplier",
               "timestamp", "utilization.gpu", "utilization.memory", "memory.used", "memory.total", 
               "power.draw", "temperature.gpu", "clocks.current.graphics", "clocks.current.memory"]
    
    file_exists = os.path.exists(csv_path)
    mode = 'a' if file_exists else 'w'
    with open(csv_path, mode, newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        for item in data_list:
            if isinstance(item, tuple):
                row_data, metadata = item
            else:
                # Backward compatibility
                row_data = item
                metadata = default_metadata
            parts = row_data.split(', ')
            if len(parts) >= 9:
                writer.writerow([
                    metadata.get("ref_arch_name", ""),
                    metadata.get("use_torch_compile", ""),
                    metadata.get("torch_compile_backend", ""),
                    metadata.get("torch_compile_options", ""),
                    metadata.get("variable_multiplier", ""),
                ] + parts)


def _set_power_limit(device_id: int, power_limit_watts: int):
    """
    Set GPU power limit using nvidia-smi with sudo
    
    Args:
        device_id: GPU device ID
        power_limit_watts: Power limit in watts
    """
    try:
        result = subprocess.run(
            ["sudo", "nvidia-smi", "-i", str(device_id), "-pl", str(power_limit_watts)],
            capture_output=True,
            text=True,
            timeout=10.0
        )
        if result.returncode != 0:
            print(f"WARNING: Failed to set power limit to {power_limit_watts}W: {result.stderr}", file=sys.stderr)
            return False
        print(f"Successfully set power limit to {power_limit_watts}W")
        time.sleep(1)  # Give GPU time to adjust
        return True
    except Exception as e:
        print(f"WARNING: Failed to set power limit: {e}", file=sys.stderr)
        return False


def _start_nvidia_smi_dmon(device_id: int, log_file: str):
    """
    Start nvidia-smi dmon for real-time monitoring
    
    Args:
        device_id: GPU device ID
        log_file: Path to log file for dmon output
    
    Returns:
        dmon_process: subprocess.Popen object or None if failed
    """
    try:
        dmon_file = open(log_file, 'w')
        dmon_process = subprocess.Popen(
            [
                "nvidia-smi", "dmon",
                "-i", str(device_id),
                "-s", "pucvmet",
                "--gpm-metrics=2",
                "--gpm-options=d",
                "--format=csv,noheader,nounit",
                "-d", "1",
            ],
            stdout=dmon_file,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Give dmon a moment to start
        time.sleep(0.2)
        if dmon_process.poll() is None:
            return dmon_process, dmon_file
        else:
            # dmon failed to start
            dmon_file.close()
            stderr = dmon_process.stderr.read() if dmon_process.stderr else None
            print(f"WARNING: nvidia-smi dmon failed to start: {stderr.decode() if stderr else 'unknown error'}", file=sys.stderr)
            return None, None
    except Exception as e:
        print(f"WARNING: Failed to start nvidia-smi dmon: {e}", file=sys.stderr)
        return None, None


def fetch_ref_arch_from_dataset(dataset: list[str], 
                                problem_id: int) -> tuple[str, str, str]:
    """
    Fetch the reference architecture from the problem directory
    problem_id should be logical index (1-indexed), matching the problem_id in the problem_name

    Returns:
        ref_arch_path: str, the path to the reference architecture
        ref_arch_name: str, the name of the reference architecture
        ref_arch_src: str, the source code of the reference architecture
    """
    ref_arch_path = None
    
    for file in dataset:
        if file.split("/")[-1].split("_")[0] == str(problem_id):
            ref_arch_path = file
            break
    if ref_arch_path is None:
        raise ValueError(f"No reference architecture found for problem_id {problem_id}")
    
    ref_arch_src = read_file(ref_arch_path)

    ref_arch_name = ref_arch_path.split("/")[-1]
    return (ref_arch_path, ref_arch_name, ref_arch_src)


def _stop_monitor_thread(stop_event: threading.Event, monitor_thread: threading.Thread, 
                         monitor_data: list, csv_path: str, write_lock: threading.Lock,
                         timeout: float = 2.0, verbose: bool = False):
    """
    Safely stop the monitor thread and ensure it terminates.
    Returns True if thread stopped successfully, False if it had to be abandoned.
    """
    if monitor_thread is None or not monitor_thread.is_alive():
        return True
    
    # Signal thread to stop
    stop_event.set()
    
    # Wait for thread to stop with timeout
    monitor_thread.join(timeout=timeout)
    
    # Check if thread actually stopped
    if monitor_thread.is_alive():
        if verbose:
            print(f"[WARNING] Monitor thread did not stop after {timeout}s timeout. Thread may be stuck.")
        # Thread is stuck - we can't force kill it in Python, but at least we know
        # Since it's a daemon thread, it will be terminated when main program exits
        return False
    
    # Thread stopped successfully - flush any remaining data
    if monitor_data:
        try:
            with write_lock:
                with _global_metadata_lock:
                    flush_metadata = _global_current_metadata.copy()
                _flush_to_csv(csv_path, monitor_data, flush_metadata)
                monitor_data.clear()
        except Exception as e:
            if verbose:
                print(f"[WARNING] Failed to flush remaining monitor data: {e}")
    
    return True


def measure_program_time(
        ref_arch_name: str,
        ref_arch_src: str,
        ref_arch_path: str = None,
        num_trials: int = 100,
        use_torch_compile: bool = False,
        torch_compile_backend: str="inductor",
        torch_compile_options: str="default",
        device: torch.device="cuda:0",
        verbose: bool = False,
        power_cap_watts: int = None,
        csv_path: str = None,
        variable_multiplier: float = 1.0,
) -> dict:
    """
    Measure the time of a KernelBench reference architecture.
    When variable_multiplier != 1.0 and ref_arch_path is given, scaled source is
    written to disk before load; the file is restored when the scaler is deleted.
    """
    scaler = None
    if variable_multiplier != 1.0:
        if ref_arch_path is not None:
            scaler = VariableScaler(
                source_code=ref_arch_src,
                multiplier=variable_multiplier,
                problem_path=ref_arch_path,
            )
            ref_arch_src = scaler.scale_source()
        else:
            ref_arch_src = scale_source_code(ref_arch_src, multiplier=variable_multiplier)

    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )
    dmon_log_file = None
    dmon_process = None
    dmon_file = None
    model = None
    inputs = None
    init_inputs = None
    elapsed_times = None
    
    # Monitor thread state - initialize to None to track if thread was started
    monitor_thread = None
    stop_event = None
    monitor_data = None
    write_lock = None
    csv_path_used = None
    
    # CRITICAL: Aggressive cleanup BEFORE starting to ensure clean GPU state
    # This prevents memory leaks from previous runs (especially after OOM errors)
    try:
        if device.type == 'cuda':
            torch.cuda.synchronize(device=device)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        time.sleep(1)  # Give GPU time to actually free memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            if verbose:
                free_mem, total_mem = torch.cuda.mem_get_info(device)
                free_gb = free_mem / (1024**3)
                total_gb = total_mem / (1024**3)
                print(f"[Memory] Starting {ref_arch_name}: {free_gb:.2f}/{total_gb:.2f} GB free")
    except Exception:
        pass
    
    try:
        with torch.no_grad():
            torch.cuda.synchronize(device=device)
            
            # Set default tensor type to CUDA to avoid large CPU allocations
            # This prevents memory explosion when scaling variables
            original_default_tensor_type = None
            if device.type == 'cuda':
                original_default_tensor_type = torch.get_default_dtype()
                torch.set_default_device(device)
            
            try:
                set_seed(42)
                inputs = get_inputs()
                set_seed(42)
                init_inputs = get_init_inputs()
                
                # Check estimated memory usage and warn if too large
                total_input_memory_gb = 0
                for x in inputs:
                    if isinstance(x, torch.Tensor):
                        mem_bytes = x.element_size() * x.nelement()
                        mem_gb = mem_bytes / (1024**3)
                        total_input_memory_gb += mem_gb
                        if mem_gb > 16.0:  # Warn if single tensor > 16GB
                            print(f"[WARNING] Large tensor detected: shape {list(x.shape)}, size {mem_gb:.2f} GB")
                            print(f"[WARNING] Consider reducing variable_multiplier (current: {variable_multiplier})")
                
                if total_input_memory_gb > 32.0:  # Warn if total > 32GB
                    print(f"[WARNING] Total input memory: {total_input_memory_gb:.2f} GB")
                    print(f"[WARNING] This may cause memory issues. Consider reducing variable_multiplier.")
                
                # Ensure all tensors are on the correct device
                inputs = [
                    x.to(device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                init_inputs = [
                    x.to(device) if isinstance(x, torch.Tensor) else x
                    for x in init_inputs
                ]
            except RuntimeError as oom_error:
                # Special handling for OOM during tensor creation
                if "out of memory" in str(oom_error).lower():
                    print(f"[OOM] Out of memory during tensor creation")
                    # Aggressive cleanup of partially allocated tensors
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device=device)
                        torch.cuda.empty_cache()
                        time.sleep(1)  # Extra time for OOM recovery
                        torch.cuda.empty_cache()
                    gc.collect()
                raise
            finally:
                # Restore default device
                if original_default_tensor_type is not None:
                    torch.set_default_device('cpu')

            # Initialize PyTorch model, use this for eager mode execution
            model = Model(*init_inputs)
            
            if use_torch_compile:
                print(f"Using torch.compile to compile model {ref_arch_name} with {torch_compile_backend} backend and {torch_compile_options} mode")
                model = torch.compile(model, backend=torch_compile_backend, mode=torch_compile_options)
            else:
                print(f"Using PyTorch Eager Execution on {ref_arch_name}")
            
            model = model.cuda(device=device)
            torch.cuda.synchronize(device=device)
            
            # Set current program metadata (shared across threads via global variable)
            metadata = {
                "ref_arch_name": ref_arch_name,
                "use_torch_compile": str(use_torch_compile),
                "torch_compile_backend": torch_compile_backend or "",
                "torch_compile_options": torch_compile_options or "",
                "variable_multiplier": variable_multiplier,
            }
            with _global_metadata_lock:
                _global_current_metadata.update(metadata)
            
            # Start nvidia-smi monitoring (CSV) and dmon for this problem
            device_id = int(str(device).split(":")[-1])
            monitor_data = []
            stop_event = threading.Event()
            write_lock = threading.Lock()
            
            # CSV path for nvidia-smi monitoring (use provided or create new)
            if csv_path is None:
                csv_dir = os.path.join(TIMING_DIR, "nvidia_smi_logs")
                os.makedirs(csv_dir, exist_ok=True)
                csv_path_used = os.path.join(csv_dir, f"gpu_metrics_{int(time.time())}.csv")
            else:
                csv_path_used = csv_path
            
            monitor_thread = threading.Thread(
                target=_nvidia_smi_monitor,
                args=(device_id, 0.5, stop_event, monitor_data, csv_path_used, write_lock, 35.0),
                daemon=True
            )
            monitor_thread.start()
            
            # Wrap everything after thread start in try-finally to ensure thread cleanup
            try:
                # Create unique identifier for this problem run
                problem_identifier = f"{ref_arch_name}_{int(time.time())}"
                if use_torch_compile:
                    problem_identifier += f"_compile_{torch_compile_backend}_{torch_compile_options}"
                if power_cap_watts is not None:
                    problem_identifier += f"_power{power_cap_watts}W"
                
                # Set log directory for dmon logs
                dmon_log_dir = os.path.join(TIMING_DIR, "nvidia_smi_daemon_logs")
                os.makedirs(dmon_log_dir, exist_ok=True)
                
                # Set log file for dmon (filename includes power_cap via problem_identifier)
                dmon_log_file = os.path.join(dmon_log_dir, f"dmon_{problem_identifier}_variable_multiplier_{variable_multiplier}.log")
                
                # Start dmon for this problem
                dmon_process, dmon_file = _start_nvidia_smi_dmon(device_id, dmon_log_file)
                
                execution_successful = False
                try:
                    elapsed_times = time_execution_with_cuda_event(
                        model, *inputs, num_trials=num_trials, verbose=verbose, device=device
                    )
                    execution_successful = True
                finally:
                    # Stop dmon after measurement
                    if dmon_process and dmon_process.poll() is None:
                        try:
                            dmon_process.terminate()
                            dmon_process.wait(timeout=2.0)
                        except:
                            try:
                                dmon_process.kill()
                            except:
                                pass
                    if dmon_file:
                        try:
                            dmon_file.close()
                        except:
                            pass
                    
                    # Delete dmon log file if execution failed
                    if not execution_successful and dmon_log_file and os.path.exists(dmon_log_file):
                        try:
                            os.remove(dmon_log_file)
                        except Exception:
                            pass
            finally:
                # CRITICAL: Always stop the monitor thread, even if exception occurred above
                _stop_monitor_thread(stop_event, monitor_thread, monitor_data, csv_path_used, write_lock, 
                                   timeout=2.0, verbose=verbose)
            
            if not execution_successful:
                raise RuntimeError("Model execution failed")
            
            # Don't clear metadata - let next program overwrite it
            # This ensures any data collected between programs has metadata (from last program)
            # rather than empty metadata
            
            runtime_stats = get_timing_stats(elapsed_times, device=device)
            runtime_stats["nvidia_smi_csv_path"] = csv_path_used
            # Store log directory and identifier for this problem
            runtime_stats["nvidia_smi_dmon_log_dir"] = dmon_log_dir
            runtime_stats["nvidia_smi_dmon_problem_id"] = problem_identifier
            if dmon_log_file:
                runtime_stats["nvidia_smi_dmon_log"] = dmon_log_file

            if verbose:
                print(f"{ref_arch_name} {runtime_stats}")
            
            # Explicitly delete model and inputs to free memory
            del model, inputs, init_inputs, elapsed_times
            if scaler is not None:
                del scaler
            
            # AGGRESSIVE CUDA memory cleanup after successful run
            torch.cuda.synchronize(device=device)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            
            # Give PyTorch time to release GPU memory before next benchmark
            time.sleep(1)
            torch.cuda.empty_cache()
            
            if verbose:
                free_mem = torch.cuda.mem_get_info(device)[0] / (1024**3)
                print(f"[Cleanup] Free GPU memory after run: {free_mem:.2f} GB")
            
            return runtime_stats
    except Exception as e:
        # CRITICAL: Stop monitor thread if it was started (handles case where exception occurred before inner finally)
        if monitor_thread is not None and stop_event is not None:
            _stop_monitor_thread(stop_event, monitor_thread, monitor_data or [], 
                               csv_path_used or csv_path or "", write_lock or threading.Lock(),
                               timeout=2.0, verbose=verbose)
        
        # Delete dmon log file if it was created but execution failed
        if dmon_log_file and os.path.exists(dmon_log_file):
            try:
                os.remove(dmon_log_file)
            except Exception:
                pass
        # Clean up dmon process if still running
        if dmon_process and dmon_process.poll() is None:
            try:
                dmon_process.terminate()
                dmon_process.wait(timeout=1.0)
            except:
                try:
                    dmon_process.kill()
                except:
                    pass
        if dmon_file:
            try:
                dmon_file.close()
            except:
                pass
        # Explicitly delete model/inputs if they exist to free memory
        try:
            if model is not None:
                del model
            if inputs is not None:
                del inputs
            if init_inputs is not None:
                del init_inputs
            if elapsed_times is not None:
                del elapsed_times
            if scaler is not None:
                del scaler
        except Exception:
            pass
        # AGGRESSIVE CUDA memory cleanup after failed run
        try:
            # Force synchronization and clear all caches
            if device is not None and device.type == 'cuda':
                torch.cuda.synchronize(device=device)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()
            
            # Give PyTorch time to actually release GPU memory
            # This is critical after OOM errors as the allocator needs time to clean up
            time.sleep(1)
            
            # Second aggressive cleanup pass
            if device is not None and device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            if verbose:
                if device is not None and device.type == 'cuda':
                    free_mem = torch.cuda.mem_get_info(device)[0] / (1024**3)
                    print(f"[Cleanup] Free GPU memory after cleanup: {free_mem:.2f} GB")
        except Exception as cleanup_error:
            if verbose:
                print(f"[WARNING] Cleanup error: {cleanup_error}")
        
        print(f"[Eval] Error in Measuring Performance: {e}")
        raise



def measure_program_time_subprocess_wrapper(params_json: str) -> str:
    """
    Wrapper function to run measure_program_time in subprocess.
    Takes JSON string input, returns JSON string output.
    This function is designed to be called from subprocess.
    """
    import json
    import sys
    
    try:
        # Parse parameters
        params = json.loads(params_json)
        
        # Extract parameters
        ref_arch_name = params['ref_arch_name']
        ref_arch_src = params['ref_arch_src']
        ref_arch_path = params.get('ref_arch_path')
        num_trials = params.get('num_trials', 100)
        use_torch_compile = params.get('use_torch_compile', False)
        torch_compile_backend = params.get('torch_compile_backend', 'inductor')
        torch_compile_options = params.get('torch_compile_options', 'default')
        device_id = params.get('device_id', 0)
        verbose = params.get('verbose', False)
        power_cap_watts = params.get('power_cap_watts')
        csv_path = params.get('csv_path')
        variable_multiplier = params.get('variable_multiplier', 1.0)
        
        device = torch.device(f'cuda:{device_id}')
        
        # Run the actual measurement
        runtime_stats = measure_program_time(
            ref_arch_name=ref_arch_name,
            ref_arch_src=ref_arch_src,
            ref_arch_path=ref_arch_path,
            num_trials=num_trials,
            use_torch_compile=use_torch_compile,
            torch_compile_backend=torch_compile_backend,
            torch_compile_options=torch_compile_options,
            device=device,
            verbose=verbose,
            power_cap_watts=power_cap_watts,
            csv_path=csv_path,
            variable_multiplier=variable_multiplier,
        )
        
        # Return success result as JSON
        result = {
            'status': 'success',
            'data': runtime_stats
        }
        return json.dumps(result)
        
    except Exception as e:
        # Return error result as JSON
        result = {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }
        return json.dumps(result)


def run_benchmark_isolated(
        ref_arch_name: str,
        ref_arch_src: str,
        ref_arch_path: str = None,
        num_trials: int = 100,
        use_torch_compile: bool = False,
        torch_compile_backend: str = "inductor",
        torch_compile_options: str = "default",
        device_id: int = 0,
        verbose: bool = False,
        power_cap_watts: int = None,
        csv_path: str = None,
        variable_multiplier: float = 1.0,
        timeout: int = 600,  # 10 minutes default
) -> dict:
    """
    Run a single benchmark in an isolated subprocess to protect against crashes.
    
    Returns:
        dict with runtime_stats if successful, or None if failed
    """
    import subprocess
    import json
    import sys
    import tempfile
    import os
    
    # Prepare parameters as JSON
    params = {
        'ref_arch_name': ref_arch_name,
        'ref_arch_src': ref_arch_src,
        'ref_arch_path': ref_arch_path,
        'num_trials': num_trials,
        'use_torch_compile': use_torch_compile,
        'torch_compile_backend': torch_compile_backend,
        'torch_compile_options': torch_compile_options,
        'device_id': device_id,
        'verbose': verbose,
        'power_cap_watts': power_cap_watts,
        'csv_path': csv_path,
        'variable_multiplier': variable_multiplier,
    }
    
    # Write parameters to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(params, f)
        param_file = f.name
    
    # Write output to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_file = f.name
    
    # Create Python script to run in subprocess
    script = f"""
import sys
import json
import os

# Suppress warnings and info messages
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, {repr(REPO_TOP_PATH)})

# Import after path is set
from scripts.generate_baseline_time import measure_program_time_subprocess_wrapper

# Read parameters
with open({repr(param_file)}, 'r') as f:
    params_json = f.read()

# Run benchmark
result_json = measure_program_time_subprocess_wrapper(params_json)

# Write result to file
with open({repr(output_file)}, 'w') as f:
    f.write(result_json)
"""
    
    try:
        # Run in subprocess with timeout
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, 'PYTHONWARNINGS': 'ignore'}
        )
        
        # Read result from file
        try:
            with open(output_file, 'r') as f:
                result_json = f.read()
            result_data = json.loads(result_json)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # Output file doesn't exist or has invalid JSON
            print(f"[SUBPROCESS ERROR] {ref_arch_name}: Failed to read result")
            if verbose:
                print(f"  Return code: {result.returncode}")
                print(f"  Stdout: {result.stdout[:300]}")
                print(f"  Stderr: {result.stderr[:300]}")
            return None
        finally:
            # Clean up temp files
            try:
                os.unlink(param_file)
            except:
                pass
            try:
                os.unlink(output_file)
            except:
                pass
        
        if result.returncode != 0:
            # Process crashed or returned error code
            stderr = result.stderr[:500] if result.stderr else "No error output"
            print(f"[SUBPROCESS ERROR] {ref_arch_name} crashed with code {result.returncode}")
            if verbose:
                print(f"  Error: {stderr}")
            return None
        
        # Check result status
        if result_data['status'] == 'success':
            if verbose:
                print(f"[SUBPROCESS SUCCESS] {ref_arch_name}")
            return result_data['data']
        else:
            # Python exception was caught
            print(f"[SUBPROCESS ERROR] {ref_arch_name}: {result_data.get('error_type', 'Unknown')}")
            if verbose:
                print(f"  Details: {result_data.get('error', 'No details')[:200]}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"[SUBPROCESS TIMEOUT] {ref_arch_name} exceeded {timeout}s timeout")
        # Clean up temp files
        try:
            os.unlink(param_file)
        except:
            pass
        try:
            os.unlink(output_file)
        except:
            pass
        return None
    except Exception as e:
        print(f"[SUBPROCESS ERROR] {ref_arch_name}: Unexpected error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def record_baseline_times(use_torch_compile: bool = False, 
                          torch_compile_backend: str="inductor", 
                          torch_compile_options: str="default",
                          file_name: str="baseline_time.json",
                          power_cap_watts: int = None,
                          device_id: int = 0,
                          levels: list[int] = [1, 2, 3],
                          required_problems: list[int] = None,
                          variable_multiplier: float = 1.0,
                          use_subprocess: bool = True):
    """
    Generate baseline time for KernelBench, 
    configure profiler options for PyTorch
    save to specified file
    
    Args:
        use_subprocess: If True, run each benchmark in isolated subprocess to protect
                       against crashes (FPE, segfaults, etc.). Adds ~1-2s overhead per
                       benchmark but prevents one crash from stopping the entire run.
                       Default: True (recommended)
    """
    device = torch.device(f"cuda:{device_id}")
    json_results = {}
    
    # # Set power limit if specified
    # if power_cap_watts is not None:
    #     print(f"\nSetting power limit to {power_cap_watts}W for device {device_id}...")
    #     if not _set_power_limit(device_id, power_cap_watts):
    #         print(f"WARNING: Could not set power limit, continuing anyway...")
    
    # Create shared CSV file for this batch run (same directory as JSON output)
    json_path = os.path.join(TIMING_DIR, file_name)
    csv_filename = os.path.basename(file_name).replace(".json", "_gpu_metrics.csv")
    csv_path = os.path.join(os.path.dirname(json_path), csv_filename)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    for level in levels:
        PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level" + str(level))
        dataset = construct_problem_dataset_from_problem_dir(PROBLEM_DIR)
        json_results[f"level{level}"] = {}

        if required_problems is not None:
            problem_ids = list(required_problems)
        else:
            problem_ids = range(1, len(dataset) + 1)
        problem_ids = tqdm(problem_ids, desc=f"Level {level}")
        for problem_id in problem_ids:
            try:
                ref_arch_path, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(dataset, problem_id)
            except ValueError:
                continue
            
            # Choose isolation method
            if use_subprocess:
                # Run in isolated subprocess (protects against crashes)
                runtime_stats = run_benchmark_isolated(
                    ref_arch_name=ref_arch_name,
                    ref_arch_src=ref_arch_src,
                    ref_arch_path=ref_arch_path,
                    num_trials=100,
                    use_torch_compile=use_torch_compile,
                    torch_compile_backend=torch_compile_backend,
                    torch_compile_options=torch_compile_options,
                    device_id=device_id,
                    verbose=False,
                    power_cap_watts=power_cap_watts,
                    csv_path=csv_path,
                    variable_multiplier=variable_multiplier,
                    timeout=600,  # 10 min timeout
                )
                
                if runtime_stats is not None:
                    json_results[f"level{level}"][ref_arch_name] = runtime_stats
                else:
                    # Benchmark failed, skip and continue
                    continue
            else:
                # Run directly (original behavior, no crash protection)
                try:
                    runtime_stats = measure_program_time(
                        ref_arch_name=ref_arch_name,
                        ref_arch_src=ref_arch_src,
                        ref_arch_path=ref_arch_path,
                        use_torch_compile=use_torch_compile,
                        torch_compile_backend=torch_compile_backend,
                        torch_compile_options=torch_compile_options,
                        device=device,
                        verbose=False,  # do not print
                        power_cap_watts=power_cap_watts,
                        csv_path=csv_path,
                        variable_multiplier=variable_multiplier,
                    )
                    json_results[f"level{level}"][ref_arch_name] = runtime_stats
                except Exception as e:
                    # Clear memory after error to prevent accumulation
                    try:
                        torch.cuda.synchronize(device=device)
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        gc.collect()
                    except Exception:
                        pass
                    # Continue to next problem
                    continue

    save_path = os.path.join(TIMING_DIR, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load existing JSON if it exists and merge results
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            existing_results = json.load(f)
        # Merge new results into existing results (new results overwrite existing ones for same keys)
        for level_key in json_results:
            if level_key in existing_results:
                existing_results[level_key].update(json_results[level_key])
            else:
                existing_results[level_key] = json_results[level_key]
        json_results = existing_results
    
    with open(save_path, "w") as f:
        json.dump(json_results, f)
    return json_results

def test_measure_particular_program(level_num: int, problem_id: int):
    """
    Test measure_program_time on a particular program
    """
    device = torch.device("cuda:0")

    PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level" + str(level_num))
    dataset = construct_problem_dataset_from_problem_dir(PROBLEM_DIR)

    ref_arch_path, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(dataset, problem_id)

    exec_stats = measure_program_time(
        ref_arch_name=ref_arch_name,
        ref_arch_src=ref_arch_src,
        ref_arch_path=ref_arch_path,
        use_torch_compile=True,
        torch_compile_backend="inductor",
        torch_compile_options="default",
        device=device,
        verbose=False,
    )

    print(f"Execution time for {ref_arch_name}: {exec_stats}")


if __name__ == "__main__":
    # DEBUG and simple testing
    # test_measure_particular_program(2, 28)
    
    # Replace this with whatever hardware you are running on 
    # hardware_name = "L40S_matx3"
    
    hardware_name = "NVIDIA RTX 4000 Ada Generation"

    device_id = 0  # GPU device ID
    base_hardware_name = hardware_name
    
    # Systematic recording of baseline time
    
    # Iterate through power caps from 30W to 130W in steps of 5W
    for power_cap in range(130, 131, 5):
        hardware_name = f"{base_hardware_name}_{power_cap}W"
        print(f"\n{'='*80}")
        print(f"Starting measurements with power cap: {power_cap}W")
        print(f"Hardware name: {hardware_name}")
        print(f"{'='*80}\n")
        
        # Check if directory and CSV file exist - if both exist, we'll append instead of overwrite
        hardware_dir = os.path.join(TIMING_DIR, hardware_name)
        csv_file = os.path.join(hardware_dir, "baseline_time_torch_compile_cudagraphs_gpu_metrics.csv")
        if os.path.exists(hardware_dir) and os.path.exists(csv_file):
            print(f"Directory {hardware_name} and CSV file already exist. Will append to existing files.")

        # # 1. Record Torch Eager
        # record_baseline_times(use_torch_compile=False, 
        #                       torch_compile_backend=None,
        #                       torch_compile_options=None, 
        #                       file_name=f"{hardware_name}/baseline_time_torch.json",
        #                       power_cap_watts=power_cap,
        #                       device_id=device_id)
        
        # # 2. Record Torch Compile using Inductor
        # for torch_compile_mode in ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]:
        #     record_baseline_times(use_torch_compile=True, 
        #                           torch_compile_backend="inductor",
        #                           torch_compile_options=torch_compile_mode, 
        #                           file_name=f"{hardware_name}/baseline_time_torch_compile_inductor_{torch_compile_mode}.json",
        #                           power_cap_watts=power_cap,
        #                           device_id=device_id)
     
        # 3. Record Torch Compile using cudagraphs
        def _run(levels, variable_multiplier,required_problems=None):
            try:
                record_baseline_times(
                    use_torch_compile=False,
                    torch_compile_backend=None,
                    torch_compile_options=None,
                    file_name=f"{hardware_name}/baseline_time_torch_compile_cudagraphs.json",
                    power_cap_watts=power_cap,
                    device_id=device_id,
                    levels=levels,
                    variable_multiplier=variable_multiplier,
                    required_problems=required_problems,
                )
            except Exception as e:
                print(f"[WARNING] record_baseline_times failed (levels={levels}, variable_multiplier={variable_multiplier}): {e}", file=sys.stderr)

        # _run([1,2,3], 1.3)
        # _run([1,2,3], 1.2)
        # _run([2,3], 1.1)
        _run([4], 1.0, [7])
        # _run([1,2,3], 0.9)
        # _run([1,2,3], 0.8)
        # _run([3], 1)
        # _run([1,2], 1.2)

        # _run([1,2,3], 2)
        # _run([1,2,3], 2.4)
        print(f"\nCompleted measurements for power cap: {power_cap}W\n")




    # Random debuging
    # get_torch_compile_triton(2, 12)
    # record_baseline_times()

    # run_profile(2, 43)
    # get_time(2, 43, torch_compile=False)
    # get_time(2, 43, torch_compile=True)




################################################################################
# Deprecated
################################################################################


def get_time_old(level_num, problem_id, num_trials=200, torch_compile=False):
    raise DeprecationWarning("Use New measure_program_time instead")
    ref_arch_name, ref_arch_src = fetch_ref_arch_from_level_problem_id(
        level_num, problem_id, with_name=True
    )
    ref_arch_name = ref_arch_name.split("/")[-1]
    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )
    try:
        with torch.no_grad():
            torch.cuda.synchronize(device=device)
            set_seed(42)
            inputs = get_inputs()
            set_seed(42)
            init_inputs = get_init_inputs()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            init_inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]
            model = Model(*init_inputs)
            
            if torch_compile:
                model = torch.compile(model)
                print("Compiled model Done")
            model = model.cuda(device=device)
            torch.cuda.synchronize(device=device)
            elapsed_times = time_execution_with_cuda_event(
                model, *inputs, num_trials=num_trials, verbose=False, device=device
            )
            runtime_stats = get_timing_stats(elapsed_times, device=device)
            # json_results[f"level{level_num}"][ref_arch_name] = runtime_stats
            print(f"{ref_arch_name} {runtime_stats}")
            return (ref_arch_name, runtime_stats)
    except Exception as e:
        print(f"[Eval] Error in Measuring Performance: {e}")
        raise

